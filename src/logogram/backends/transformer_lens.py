"""TransformerLens backend, built on ``TransformerBridge`` (TransformerLens 4).

Abstract sites map to TransformerLens hook points here and nowhere else.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import math
import re
import threading
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from logogram.backends.base import (
    ALL_KINDS,
    BackendError,
    Cancelled,
    EdgePatcher,
    InputGradients,
    ModelBackend,
    ModelInfo,
    Patch,
    Patches,
    ScoreFn,
    Tokenized,
    device_type,
    float64,
)

log = logging.getLogger(__name__)

HOOKS: dict[str, str] = {
    "resid_pre": "blocks.{layer}.hook_resid_pre",
    "resid_mid": "blocks.{layer}.hook_resid_mid",
    "resid_post": "blocks.{layer}.hook_resid_post",
    "attn_out": "blocks.{layer}.hook_attn_out",
    "mlp_out": "blocks.{layer}.hook_mlp_out",
    "head": "blocks.{layer}.attn.hook_z",
    # Each head's query, key and value as its projections compute them (before any rotary
    # embedding): [B, pos, heads, d_head], with key/value heads for k and v.
    "q": "blocks.{layer}.attn.hook_q",
    "k": "blocks.{layer}.attn.hook_k",
    "v": "blocks.{layer}.attn.hook_v",
    # The MLP's hidden activation after its nonlinearity (for a gated MLP, the gate's activation
    # times the up projection): the input to its output projection, [B, pos, d_mlp].
    "neuron": "blocks.{layer}.mlp.hook_post",
    # The MLP's input, normalized as the MLP reads it: what transcoders read. Not a site to patch.
    "mlp_in": "blocks.{layer}.mlp.hook_in",
}
PATTERN_HOOK = "blocks.{layer}.attn.hook_pattern"
# The final normalization's scale: held at its value in a run, it makes the normalization affine.
FINAL_SCALE = "ln_final.hook_scale"
# Each receiver's own input, for edges (logogram.edges): with the bridge's split inputs on, every
# head gets its own copy of the residual stream for its query, key and value ([B, pos, H, d_model],
# keys and values per key-value head), normalized by itself; the MLP's normalization and the final
# normalization each read their own copy ([B, pos, d_model]). Changing a copy changes only what
# its receiver reads: the residual stream itself runs on unchanged.
EDGE_HOOKS: dict[str, str] = {
    "q": "blocks.{layer}.hook_q_input",
    "k": "blocks.{layer}.hook_k_input",
    "v": "blocks.{layer}.hook_v_input",
    "mlp": "blocks.{layer}.ln2.hook_in",
    "logits": "ln_final.hook_in",
}


def edge_hook(receiver: str, layer: int) -> str:
    return EDGE_HOOKS[receiver].format(layer=layer)


DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

# Load-time checks compare numbers, so a short fixed input is enough. The tolerance is relative to
# the largest value compared, and allows for the rounding of each precision.
CHECK_TOLERANCE = {"float32": 1e-4, "float16": 2e-2, "bfloat16": 5e-2}


def hook_name(kind: str, layer: int) -> str:
    return HOOKS[kind].format(layer=layer)


def _resid_mid_hooks(patch: Patch) -> list[tuple[str, Callable[..., torch.Tensor]]]:
    """Patch the residual stream between attention and MLP.

    TransformerLens exposes ``hook_resid_mid`` as the input of the MLP's LayerNorm, so editing it
    would change only what the MLP reads while the residual stream kept its old value. Instead,
    since resid_mid = resid_pre + attn_out, set attn_out to (target - resid_pre) at the patched
    positions, using this run's own resid_pre.
    """
    seen: dict[str, torch.Tensor] = {}

    def keep_resid_pre(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
        seen["resid_pre"] = act
        return act

    def set_attn_out(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
        pre = seen["resid_pre"]
        act = act.clone()
        values = patch.values.to(dtype=act.dtype, device=act.device)
        if patch.positions is None:
            n = values.shape[1]  # the prompt; appended tokens are never patched
            act[:, :n] = values - pre[:, :n].to(act.dtype)
        else:
            rows = torch.arange(act.shape[0], device=act.device)
            pos = patch.positions.to(act.device)
            act[rows, pos] = values - pre[rows, pos].to(act.dtype)
        return act

    return [
        (hook_name("resid_pre", patch.layer), keep_resid_pre),
        (hook_name("attn_out", patch.layer), set_attn_out),
    ]


def _patch_hook(patch: Patch) -> Callable[..., torch.Tensor]:
    def fn(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
        act = act.clone()
        values = patch.values.to(dtype=act.dtype, device=act.device)
        rows = torch.arange(act.shape[0], device=act.device)
        units = patch.units()
        if patch.mask is not None:
            n = values.shape[1]  # the prompt's positions; appended tokens are never patched
            mask = patch.mask.to(act.device)
            if mask.dim() < act.dim():  # one choice per vector (a position, or a head's)
                mask = mask[..., None]
            act[:, :n] = torch.where(mask, values, act[:, :n])
        elif units is not None:
            # A head's output, query, key or value ([B, pos, d_head] or [B, d_head]), or a
            # neuron's activation ([B, pos] or [B]).
            units = units.to(act.device)
            if patch.positions is None:
                n = values.shape[1]
                act[rows, :n, units] = values
            else:
                act[rows, patch.positions.to(act.device), units] = values
        elif patch.positions is None:
            n = values.shape[1]
            act[:, :n] = values  # [B, pos, d_model]
        else:
            act[rows, patch.positions.to(act.device)] = values  # [B, d_model]
        return act

    return fn


def _patch_hooks(patch: Patches) -> list[tuple[str, Callable[..., torch.Tensor]]]:
    """Hooks that apply every patch in one forward pass. Patches at the same hook run in order."""
    patches = [] if patch is None else patch if isinstance(patch, list) else [patch]
    hooks: list[tuple[str, Callable[..., torch.Tensor]]] = []
    for p in patches:
        if p.kind == "resid_mid":
            hooks += _resid_mid_hooks(p)
        else:
            hooks.append((hook_name(p.kind, p.layer), _patch_hook(p)))
    return _combine(hooks)


# Lengthening a copy read before its normalization changes the output by about ε / variance times
# what turning it as far does. Real models' activations are far larger than their normalizations'
# epsilon; this also allows the small ones of tiny test models.
def _combine(
    hooks: list[tuple[str, Callable[..., torch.Tensor]]],
) -> list[tuple[str, Callable[..., torch.Tensor]]]:
    """One hook per hook point, applying that point's functions in order."""
    grouped: dict[str, list[Callable[..., torch.Tensor]]] = {}
    for name, fn in hooks:
        grouped.setdefault(name, []).append(fn)

    def chain(fns: list[Callable[..., torch.Tensor]]) -> Callable[..., torch.Tensor]:
        if len(fns) == 1:
            return fns[0]

        def fn(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
            for f in fns:
                act = f(act, hook)
            return act

        return fn

    return [(name, chain(fns)) for name, fns in grouped.items()]


@dataclass
class ModelChecks:
    """What Logogram measured about a model when it loaded (see :func:`check_model`)."""

    tolerance: float
    # Largest change in the log-probabilities against the original model (relative to their range).
    function: float
    # "sequential" (attention, then MLP, each added to the residual stream), "parallel" (both read
    # the same residual and are added together) or "components" (neither could be verified).
    structure: str
    # Largest error in the residual additions of that structure, if one was verified.
    residual: float | None
    # Error of the final-norm logit lens at the last layer against the model's output, if defined.
    lens: float | None
    # Error of each head's output (z through its slice of W_O) summed with b_O against the attention
    # output: small when heads add up to what attention writes, large when the model normalizes
    # after combining them (Gemma 2, OLMo 2). None if it couldn't be measured.
    heads: float | None = None
    # The heads' queries, keys and values (see check_units). Error of each layer's attention
    # output with every query, key and value taken from another input, against that input's:
    # small when they are all the attention reads.
    qkv: float | None = None
    # Largest change in the output z of a head that doesn't read a query (a key, a value) when
    # only that one is replaced: zero when each head reads its own query and each key/value head
    # is read by its group of query heads alone.
    q_alone: float | None = None
    k_alone: float | None = None
    v_alone: float | None = None
    # Error of each head's output recomputed as its attention pattern times the values it reads:
    # small when the output is linear in the values, so a value has a direct effect.
    values: float | None = None
    # Error of each layer's MLP output with every neuron's activation taken from another input:
    # small when the neurons are all the MLP's output projection reads.
    neurons: float | None = None
    # Error of each MLP output recomputed as every neuron's activation times its row of the output
    # projection, plus its bias: small when the MLP output is the sum of what its neurons write,
    # large when the model normalizes after the MLP (Gemma 2, OLMo 2).
    neuron_writes: float | None = None

    def to_dict(self) -> dict[str, Any]:
        # JSON has no infinity: a comparison that failed on values that aren't finite numbers
        # reads as not measured.
        return {
            k: None if isinstance(v, float) and not math.isfinite(v) else v
            for k, v in asdict(self).items()
        }


def _probe_tokens(d_vocab: int, device: str) -> torch.Tensor:
    rows = [[(7 * i + 3) % d_vocab for i in range(8)], [(5 * i + 11) % d_vocab for i in range(8)]]
    return torch.tensor(rows, dtype=torch.long, device=device)


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    error = float((a - b).abs().max() / b.abs().max().clamp_min(1e-6))
    # An overflow (inf or NaN) agrees with nothing. A NaN would pass every "error <= tolerance"
    # test, and max() over several errors can drop it.
    return error if math.isfinite(error) else math.inf


# What to do when a dtype can't hold a model's values.
RANGE_FIX = {
    "float16": "Load it in bfloat16 or float32, which have a wider range than float16.",
    "bfloat16": "Load it in float32.",
    "float32": "Its weights may be damaged: delete it from the Hugging Face cache and load it again.",
}


def _log_prob_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return _relative(torch.log_softmax(a.float(), dim=-1), torch.log_softmax(b.float(), dim=-1))


def _soft_cap(cfg: Any) -> float | None:
    cap = float(getattr(cfg, "output_logits_soft_cap", 0) or 0)
    return cap if cap > 0 else None


def final_projection(bridge: Any, residual: torch.Tensor) -> torch.Tensor:
    """What the model does after its last layer: final normalization, unembedding, and the logit
    soft-capping some models (Gemma 2) apply outside both."""
    logits = bridge.unembed(bridge.ln_final(residual))
    cap = _soft_cap(bridge.cfg)
    if cap is not None:
        logits = cap * torch.tanh(logits / cap)
    return logits


def check_model(
    bridge: Any,
    probe: torch.Tensor,
    reference: torch.Tensor,
    n_layers: int,
    kinds: tuple[str, ...],
    dtype: str,
) -> ModelChecks:
    """Check, on a short input, what Logogram's measurements assume about a loaded model.

    TransformerLens supports many architectures, and processes some weights; rather than trusting
    a list, this verifies the model in front of it: that TransformerLens's version predicts what the
    original model predicts, how each layer adds attention and MLP into the residual stream, and
    whether the final normalization and unembedding reproduce the output (the logit lens).
    """
    tolerance = CHECK_TOLERANCE[dtype]
    stream = [
        k for k in ("resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out") if k in kinds
    ]
    names = [hook_name(k, layer) for layer in range(n_layers) for k in stream]
    if "head" in kinds:
        names += [hook_name("head", layer) for layer in range(n_layers)]
    with torch.no_grad():
        logits, cache = bridge.run_with_cache(probe, names_filter=names)

    def act(kind: str, layer: int) -> torch.Tensor:
        return cache[hook_name(kind, layer)].float()

    structure, residual = "components", None
    if {"resid_pre", "resid_post", "attn_out", "mlp_out"} <= set(stream):
        chain = max(
            (_relative(act("resid_pre", i + 1), act("resid_post", i)) for i in range(n_layers - 1)),
            default=0.0,
        )
        parallel = max(
            _relative(
                act("resid_pre", i) + act("attn_out", i) + act("mlp_out", i), act("resid_post", i)
            )
            for i in range(n_layers)
        )
        sequential = None
        if "resid_mid" in stream:
            sequential = max(
                max(
                    _relative(act("resid_pre", i) + act("attn_out", i), act("resid_mid", i)),
                    _relative(act("resid_mid", i) + act("mlp_out", i), act("resid_post", i)),
                )
                for i in range(n_layers)
            )
        if sequential is not None and max(sequential, chain) <= tolerance:
            structure, residual = "sequential", max(sequential, chain)
        elif max(parallel, chain) <= tolerance:
            structure, residual = "parallel", max(parallel, chain)

    heads = None
    if "head" in kinds and "attn_out" in stream:
        try:
            errors = []
            for i in range(n_layers):
                attention = bridge.blocks[i].attn
                z = cache[hook_name("head", i)].float()
                with torch.no_grad():
                    out = torch.einsum("bphd,hdm->bpm", z, attention.W_O.float())
                    if getattr(attention, "b_O", None) is not None:
                        out = out + attention.b_O.float()
                errors.append(_relative(out, act("attn_out", i)))
            heads = max(errors)
        except Exception:  # noqa: BLE001 - no per-head output weights to check with
            heads = None

    lens = None
    if "resid_post" in stream:
        try:
            with torch.no_grad():
                final = cache[hook_name("resid_post", n_layers - 1)]
                lens = _log_prob_error(final_projection(bridge, final), logits)
        except Exception:  # noqa: BLE001 - no usable final norm or unembedding
            lens = None
    return ModelChecks(
        tolerance=tolerance,
        function=_log_prob_error(logits, reference),
        structure=structure,
        residual=residual,
        lens=lens,
        heads=heads,
    )


def kv_group(n_heads: int, n_kv: int) -> int:
    """How many query heads read each key/value head: 1 without grouped-query attention. Query
    head h reads key/value head h // kv_group (the order Hugging Face's models repeat them in,
    which check_units measures)."""
    return n_heads // n_kv if n_kv > 0 and n_heads % n_kv == 0 else 0


UNIT_CHECKS = ("qkv", "q_alone", "k_alone", "v_alone", "values", "neurons", "neuron_writes")


def check_units(
    bridge: Any, probe: torch.Tensor, n_layers: int, n_heads: int, n_kv: int, d_mlp: int | None
) -> dict[str, float | None]:
    """Measure, on a short input, what patching a single query, key, value or neuron assumes.

    The probe's two rows are two different inputs. Each measurement is an error relative to the
    largest value compared (see :class:`ModelChecks`), None where a hook or weight is missing or
    an activation doesn't have the shape the model's configuration gives it:

    * ``qkv`` and ``neurons``: with every query, key and value (every neuron's activation) of
      every layer taken from the second row, the first row's attention (MLP) outputs must be the
      second row's. Then they are all the attention (the MLP's output projection) reads, and the
      hooks that hold them are live.
    * ``q_alone``, ``k_alone`` and ``v_alone``: with the queries, keys and values of every layer
      held at the first row's, replacing a single one may change the output z only of the heads
      that read it: a query's own head, or the query heads of a key/value head's group. Holding
      every layer's makes each layer's z depend on its own alone, so one pass measures them all.
    * ``values``: each head's output z is its attention pattern times the values it reads.
    * ``neuron_writes``: each MLP output is the neurons' activations times the rows of its output
      projection, plus its bias.
    """
    out: dict[str, float | None] = dict.fromkeys(UNIT_CHECKS)
    hooks = bridge.hook_dict
    layers = range(n_layers)

    def present(kind: str) -> bool:
        return all(hook_name(kind, layer) in hooks for layer in layers)

    group = kv_group(n_heads, n_kv)
    has_qkv = group > 0 and all(present(k) for k in ("q", "k", "v", "head", "attn_out"))
    has_neurons = d_mlp is not None and present("neuron") and present("mlp_out")
    if not (has_qkv or has_neurons):
        return out
    kinds = (["q", "k", "v", "head", "attn_out"] if has_qkv else []) + (
        ["neuron", "mlp_out"] if has_neurons else []
    )
    names = [hook_name(k, layer) for layer in layers for k in kinds]
    if has_qkv:
        names += [PATTERN_HOOK.format(layer=layer) for layer in layers]
    with torch.no_grad():
        _, cache = bridge.run_with_cache(probe, names_filter=names)

    def act(kind: str, layer: int) -> torch.Tensor:
        return cache[hook_name(kind, layer)]

    def fired(kinds: list[str]) -> bool:  # a hook that exists but never runs holds nothing
        return all(hook_name(k, layer) in cache for k in kinds for layer in layers)

    has_qkv = has_qkv and fired(["q", "k", "v", "head", "attn_out"])
    has_neurons = has_neurons and fired(["neuron", "mlp_out"])
    B, P = probe.shape
    d_head = int(act("head", 0).shape[-1]) if has_qkv else 0
    expected = {"q": n_heads, "k": n_kv, "v": n_kv}
    has_qkv = has_qkv and all(
        tuple(act(k, layer).shape) == (B, P, expected[k], d_head)
        for k in ("q", "k", "v")
        for layer in layers
    )
    has_neurons = has_neurons and all(
        tuple(act("neuron", layer).shape) == (B, P, d_mlp) for layer in layers
    )
    seen: dict[str, torch.Tensor] = {}

    def keep(name: str) -> Callable[..., torch.Tensor]:
        def fn(a: torch.Tensor, hook: Any = None) -> torch.Tensor:
            seen[name] = a.detach()
            return a

        return fn

    def from_second(kind: str, layer: int) -> Callable[..., torch.Tensor]:
        own = act(kind, layer)

        def fn(a: torch.Tensor, hook: Any = None) -> torch.Tensor:
            a = a.clone()
            a[0] = own[1].to(a.dtype)
            return a

        return fn

    # Every query, key and value (every neuron) from the second row, in one pass.
    swapped = (["q", "k", "v"] if has_qkv else []) + (["neuron"] if has_neurons else [])
    outputs = (["attn_out"] if has_qkv else []) + (["mlp_out"] if has_neurons else [])
    if swapped:
        fwd = [(hook_name(k, layer), from_second(k, layer)) for layer in layers for k in swapped]
        fwd += [
            (hook_name(k, layer), keep(hook_name(k, layer))) for layer in layers for k in outputs
        ]
        with torch.no_grad():
            bridge.run_with_hooks(probe, fwd_hooks=fwd, return_type=None)
        for kind, name in (("attn_out", "qkv"), ("mlp_out", "neurons")):
            if kind in outputs:
                out[name] = max(
                    _relative(seen[hook_name(kind, layer)][0], seen[hook_name(kind, layer)][1])
                    for layer in layers
                )

    if has_qkv:
        for part in ("q", "k", "v"):
            out[f"{part}_alone"] = _check_alone(
                bridge, probe, act, n_layers, part, expected[part], n_heads, group
            )
        patterns = [PATTERN_HOOK.format(layer=layer) for layer in layers]
        if all(name in cache for name in patterns):
            with torch.no_grad():
                errors = []
                for layer in layers:
                    pattern = cache[patterns[layer]].float()  # [B, H, query, key]
                    values = act("v", layer).float().repeat_interleave(group, dim=2)
                    z = torch.einsum("bhqk,bkhd->bqhd", pattern, values)  # [B, query, H, d]
                    errors.append(_relative(z, act("head", layer)))
                out["values"] = max(errors)

    if has_neurons:
        try:
            errors = []
            with torch.no_grad():
                for layer in layers:
                    mlp = bridge.blocks[layer].mlp
                    written = act("neuron", layer).float() @ mlp.W_out.float()
                    bias = getattr(mlp, "b_out", None)
                    if bias is not None:
                        written = written + bias.float()
                    errors.append(_relative(written, act("mlp_out", layer)))
            out["neuron_writes"] = max(errors)
        except Exception:  # noqa: BLE001 - no output projection to check with
            out["neuron_writes"] = None
    return out


def _check_alone(
    bridge: Any,
    probe: torch.Tensor,
    act: Callable[[str, int], torch.Tensor],
    n_layers: int,
    part: str,
    count: int,
    n_heads: int,
    group: int,
) -> float:
    """The largest change, relative to the layer's largest output, in the output z of a head that
    doesn't read the one query (key, value) replaced, over every one of them and every layer.

    Row 0 of the batch is the probe's first row with every query, key and value held at its own;
    row u + 1 also takes unit u of ``part`` from the probe's second row."""
    rows = count + 1
    tokens = probe[:1].expand(rows, -1)
    units = torch.arange(count, device=probe.device)
    seen: dict[int, torch.Tensor] = {}

    def hold(kind: str, layer: int) -> Callable[..., torch.Tensor]:
        own = act(kind, layer)

        def fn(a: torch.Tensor, hook: Any = None) -> torch.Tensor:
            held = own[0].to(device=a.device, dtype=a.dtype)
            held = held.unsqueeze(0).repeat(rows, *([1] * held.dim()))
            if kind == part:
                other = own[1].to(device=a.device, dtype=a.dtype)  # [pos, units, d_head]
                held[units + 1, :, units] = other[:, units].transpose(0, 1)
            return held

        return fn

    def keep(layer: int) -> Callable[..., torch.Tensor]:
        def fn(a: torch.Tensor, hook: Any = None) -> torch.Tensor:
            seen[layer] = a.detach().float()
            return a

        return fn

    fwd = [(hook_name(k, layer), hold(k, layer)) for layer in range(n_layers) for k in "qkv"]
    fwd += [(hook_name("head", layer), keep(layer)) for layer in range(n_layers)]
    with torch.no_grad():
        bridge.run_with_hooks(tokens, fwd_hooks=fwd, return_type=None)
    # Which heads read each unit: a query its own head; a key or value its group's query heads.
    heads = torch.arange(n_heads, device=probe.device)
    reads = (heads if part == "q" else heads // group)[None, :] == units[:, None]
    worst = 0.0
    for layer in range(n_layers):
        z = seen[layer]  # [rows, pos, H, d_head]
        scale = float(z[0].abs().max().clamp_min(1e-6))
        change = (z[1:] - z[:1]).abs().amax(dim=(1, 3))  # [units, H]
        error = float(change.masked_fill(reads.to(change.device), 0.0).max()) / scale
        if not math.isfinite(error):
            return math.inf
        worst = max(worst, error)
    return worst


def _share(error: float | None) -> str:
    return (
        "values that aren't finite numbers"
        if error is None or not math.isfinite(error)
        else (f"{error:.1%} of its range")
    )


def unit_sites(checks: ModelChecks) -> tuple[tuple[str, ...], dict[str, str]]:
    """Which of q, k, v and neuron are sites of this model, by what check_units measured, and why
    each one that isn't is missing, in words."""
    tolerance = checks.tolerance

    def holds(error: float | None) -> bool:
        return error is not None and error <= tolerance

    kinds: list[str] = []
    notes: dict[str, str] = {}
    if checks.qkv is None:
        text = (
            "TransformerLens doesn't expose this model's queries, keys and values head by head, "
            "so they can't be patched."
        )
        notes.update(dict.fromkeys(("q", "k", "v"), text))
    elif not holds(checks.qkv):
        text = (
            "When this model loaded, taking every head's query, key and value from another input "
            "didn't reproduce that input's attention output (it was off by up to "
            f"{_share(checks.qkv)}): they aren't all its attention reads, so patching them would "
            "miss the rest."
        )
        notes.update(dict.fromkeys(("q", "k", "v"), text))
    else:
        for part, error, what, these in (
            ("q", checks.q_alone, "query", "queries"),
            ("k", checks.k_alone, "key", "keys"),
            ("v", checks.v_alone, "value", "values"),
        ):
            if holds(error):
                kinds.append(part)
                continue
            reader = "its own head" if part == "q" else "the query heads of its group"
            notes[part] = (
                f"When this model loaded, replacing one head's {what} also changed the output of "
                f"heads other than {reader} (by up to {_share(error)}): its {these} aren't "
                f"separate per head (OLMo 2, for one, normalizes every head's {these} together), "
                f"so a single head's {what} can't be patched alone."
            )
    if checks.neurons is None:
        notes["neuron"] = (
            "TransformerLens doesn't expose this model's MLP neurons (a mixture of experts, for "
            "one, has no single set of them), so they can't be patched."
        )
    elif not holds(checks.neurons):
        notes["neuron"] = (
            "When this model loaded, taking every neuron's activation from another input didn't "
            f"reproduce that input's MLP output (it was off by up to {_share(checks.neurons)}): "
            "they aren't all its MLP's output projection reads, so patching them would miss the "
            "rest."
        )
    else:
        kinds.append("neuron")
    return tuple(kinds), notes


class TransformerLensBackend(ModelBackend):
    def __init__(self, bridge: Any, info: ModelInfo) -> None:
        super().__init__()
        self.bridge = bridge
        self.info = info
        self.tokenizer = bridge.tokenizer
        self.bos_token_id = own_bos_token_id(bridge)
        self._chat_template, self._chat_tokens = own_chat_template(bridge)
        self._logits_to_keep = self._supports_logits_to_keep()
        # What edge attribution relies on, measured on first use: the errors, or why it can't.
        self._edges: dict[str, float] | str | None = None

    # -- construction ------------------------------------------------------------------------

    @classmethod
    def from_bridge(
        cls,
        bridge: Any,
        *,
        model_id: str,
        revision: str | None,
        dtype: str,
        process_weights: bool,
        n_params: int | None = None,
        source: str = "hub",
        sha256: str | None = None,
    ) -> TransformerLensBackend:
        bridge.eval()
        # Gradients are only ever taken with respect to activations (attribution patching, direct
        # effects); weight gradients would make autograd keep every layer's inputs for nothing.
        for parameter in bridge.parameters():
            parameter.requires_grad_(False)
        cfg = bridge.cfg
        probe = _probe_tokens(int(cfg.d_vocab), str(cfg.device))
        with torch.no_grad():
            # The model's own predictions, before TransformerLens touches its weights.
            reference = bridge.original_model(probe).logits.float()
        if not torch.isfinite(reference).all():
            raise BackendError(
                f"In {dtype}, {model_id}'s own forward pass gives values that aren't finite "
                f"numbers (it overflows), so its predictions can't be checked. {RANGE_FIX[dtype]}"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bridge.enable_compatibility_mode(
                disable_warnings=True, no_processing=not process_weights
            )
        device = full_device(str(cfg.device))
        kinds = tuple(k for k in ALL_KINDS if hook_name(k, 0) in bridge.hook_dict)
        if "head" not in kinds or PATTERN_HOOK.format(layer=0) not in bridge.hook_dict:
            raise BackendError(
                f"{model_id} doesn't expose per-head attention hooks in TransformerLens, so "
                "Logogram can't run head experiments on it."
            )
        checks = check_model(bridge, probe, reference, int(cfg.n_layers), kinds, dtype)
        if math.isinf(checks.function):
            what = (
                "Processing the weights of" if process_weights else "TransformerLens's version of"
            )
            raise BackendError(
                f"{what} {model_id} gives values that aren't finite numbers in {dtype}, while the "
                f"model itself doesn't. {RANGE_FIX[dtype]}"
            )
        if checks.function > checks.tolerance:
            if process_weights:
                raise BackendError(
                    f"Processing the weights of {model_id} changed its predictions "
                    f"(log-probabilities moved by up to {checks.function:.1%} of their range), so "
                    "results wouldn't describe the original model. Load it with weight "
                    "processing off."
                )
            raise BackendError(
                f"TransformerLens's version of {model_id} doesn't reproduce the model's own "
                f"predictions (log-probabilities differ by up to {checks.function:.1%} of their "
                "range), so Logogram can't measure it reliably."
            )
        if not checks.structure.startswith("sequential"):
            # resid_mid is patched as resid_pre + attn_out (see _resid_mid_hooks), which is only
            # the residual stream between attention and MLP when the layer adds them in turn.
            kinds = tuple(k for k in kinds if k != "resid_mid")
        # Queries, keys, values and neurons are sites only where they were measured to be what
        # the attention and the MLP read (see check_units); the notes say why one isn't.
        n_kv = int(getattr(cfg, "n_key_value_heads", None) or cfg.n_heads)
        d_mlp = int(cfg.d_mlp) if getattr(cfg, "d_mlp", None) else None
        measured = check_units(bridge, probe, int(cfg.n_layers), int(cfg.n_heads), n_kv, d_mlp)
        checks = replace(checks, **measured)
        unit_kinds, site_notes = unit_sites(checks)
        kinds = (*kinds, *unit_kinds)
        if n_params is None:
            try:
                n_params = int(bridge.n_params_total)
            except Exception:  # noqa: BLE001
                n_params = None
        model_type = getattr(bridge.original_model.config, "model_type", None)
        structure = checks.structure
        if structure == "sequential" and model_type == "gpt2":
            structure = "sequential_pre_norm"  # GPT-2's layout, normalization included, is known
        # The model's own chat template and end tokens, as boot_local read them before the
        # bridge's tokenizer could change them.
        recorded: tuple[str | None, list[str]] | None = bridge.__dict__.get(_CHAT)
        template, ends = recorded or chat_facts(bridge.tokenizer, bridge.original_model)
        default_template, _ = own_chat_template(bridge)
        from logogram.chat import template_sha256

        info = ModelInfo(
            id=model_id,
            revision=revision,
            architecture=str(getattr(cfg, "architecture", None) or "unknown"),
            n_layers=int(cfg.n_layers),
            n_heads=int(cfg.n_heads),
            d_model=int(cfg.d_model),
            d_head=int(cfg.d_head),
            d_mlp=int(cfg.d_mlp) if getattr(cfg, "d_mlp", None) else None,
            d_vocab=int(cfg.d_vocab),
            n_ctx=int(cfg.n_ctx),
            n_params=n_params,
            dtype=dtype,
            device=device,
            device_name=_device_name(device),
            process_weights=process_weights,
            site_kinds=kinds,
            backend="transformer_lens",
            backend_version=version("transformer-lens"),
            extra={
                # Sites an SAE or transcoder can read here, beyond the patchable ones.
                "sae_sites": [k for k in ("mlp_in",) if hook_name(k, 0) in bridge.hook_dict],
                "block_structure": structure,
                "normalization": str(getattr(cfg, "normalization_type", "unknown")),
                "activation": str(getattr(cfg, "act_fn", "unknown")),
                "prediction_method": "final_norm_logit_lens"
                if checks.lens is not None and checks.lens <= checks.tolerance
                else None,
                "model_type": str(model_type or "unknown"),
                "n_key_value_heads": n_kv,
                # Why a site this model doesn't offer is missing (q, k, v or neuron), in words.
                "site_notes": site_notes,
                "bos": own_bos_token_id(bridge) is not None,
                # Whether it is made for chat: prompts run as plain text, without its template.
                "chat": chat_format(model_id, template, ends),
                # The template a spec's tokenization.chat_template wraps prompts in (the one its
                # tokenizer uses by default), by its SHA-256; None when it has none.
                "chat_template_sha256": None
                if default_template is None
                else template_sha256(default_template),
                # TransformerLens marks "no soft-cap" with a value of zero or below.
                "logit_soft_cap": _soft_cap(cfg),
                "checks": checks.to_dict(),
            },
            source=source,
            sha256=sha256,
        )
        return cls(bridge, info)

    def _supports_logits_to_keep(self) -> bool:
        probe = torch.tensor([[self._bos_or_zero(), self._bos_or_zero()]], device=self.device)
        try:
            with torch.no_grad():
                out = self._bridge()(probe, return_type="logits", logits_to_keep=1)
            return tuple(out.shape[:2]) == (1, 1)
        except Exception:  # noqa: BLE001 - architecture without logits_to_keep
            return False

    def _bos_or_zero(self) -> int:
        return int(self.bos_token_id) if self.bos_token_id is not None else 0

    # -- tokens ------------------------------------------------------------------------------

    def tokenize(self, text: str, prepend_bos: bool) -> Tokenized:
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(enc["input_ids"])
        offsets = [tuple(o) for o in enc["offset_mapping"]]
        if prepend_bos:
            bos = self.bos_token_id
            if bos is None:
                raise BackendError(
                    "This model's tokenizer has no beginning-of-sequence token. Set "
                    "tokenization.prepend_bos to false in the spec."
                )
            ids = [int(bos), *ids]
            offsets = [(0, 0), *offsets]
        tokens = [self.token_str(i) for i in ids]
        return Tokenized(ids=ids, tokens=tokens, offsets=offsets)  # type: ignore[arg-type]

    def tokenize_many(self, texts: Sequence[str], prepend_bos: bool) -> list[list[int]]:
        if not texts:
            return []
        bos = self.bos_token_id
        if prepend_bos and bos is None:
            raise BackendError(
                "This model's tokenizer has no beginning-of-sequence token. Set prepend_bos to "
                "false."
            )
        # One call for every text: the same ids as tokenize gives, one text at a time.
        encoded = self.tokenizer(list(texts), add_special_tokens=False)["input_ids"]
        first = [int(bos)] if prepend_bos and bos is not None else []
        return [first + [int(i) for i in ids] for ids in encoded]

    @property
    def chat_template(self) -> str | None:
        return self._chat_template

    def render_chat(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        continue_final_message: bool,
    ) -> str:
        if self._chat_template is None:
            return super().render_chat(
                messages,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
            )
        import jinja2

        # The special tokens the template reads are the model's own, as its tokenizer had them
        # before the bridge gave it substitutes: one the model lacks is undefined, as it is to
        # the model's own tokenizer.
        context: dict[str, Any] = {
            key: jinja2.Undefined(name=key) for key in self.tokenizer.special_tokens_map
        }
        context.update(self._chat_tokens)
        # No clock: a template that writes today's date uses its own fallback, or can't run, so
        # the same spec reads the same text on any day (see logogram.chat).
        context["strftime_now"] = jinja2.Undefined(name="strftime_now")
        text = self.tokenizer.apply_chat_template(
            messages,
            chat_template=self._chat_template,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            **context,
        )
        return str(text)

    def single_token_id(self, text: str) -> int | None:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            return None
        # A word-level tokenizer reads every word it doesn't know as its unknown token, one token
        # that stands for all of them: not a token of the word's own.
        unknown = getattr(self.tokenizer, "unk_token_id", None)
        if unknown is not None and int(ids[0]) == int(unknown) and text != self.tokenizer.unk_token:
            return None
        return int(ids[0])

    def token_str(self, token_id: int) -> str:
        return self.tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)

    # -- forward passes ----------------------------------------------------------------------

    def logits(
        self, tokens: torch.Tensor, patch: Patches = None, keep: int = 1, grad: bool = False
    ) -> torch.Tensor:
        kwargs: dict[str, Any] = {"return_type": "logits"}
        if self._logits_to_keep:
            kwargs["logits_to_keep"] = keep
        with self.lock, torch.enable_grad() if grad else torch.no_grad():
            bridge = self._bridge()
            tokens = tokens.to(self.device)
            hooks = _patch_hooks(patch)
            if not hooks:
                logits = bridge(tokens, **kwargs)
            else:
                logits = bridge.run_with_hooks(tokens, fwd_hooks=hooks, **kwargs)
            return logits[:, -keep:, :].float()

    def logits_and_reads(
        self,
        tokens: torch.Tensor,
        patch: Patches,
        keep: int,
        reads: list[tuple[str, int]],
    ) -> tuple[torch.Tensor, dict[tuple[str, int], torch.Tensor]]:
        seen: dict[tuple[str, int], torch.Tensor] = {}

        def read(site: tuple[str, int]) -> Callable[..., torch.Tensor]:
            def fn(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
                seen[site] = act.detach()
                return act

            return fn

        # A read at a patched hook comes after the patch: _combine keeps the order given.
        hooks = _combine(
            _patch_hooks(patch)
            + [(hook_name(kind, layer), read((kind, layer))) for kind, layer in reads]
        )
        kwargs: dict[str, Any] = {"return_type": "logits"}
        if self._logits_to_keep:
            kwargs["logits_to_keep"] = keep
        with self.lock, torch.no_grad():
            logits = self._bridge().run_with_hooks(
                tokens.to(self.device), fwd_hooks=hooks, **kwargs
            )
            missing = [site for site in reads if site not in seen]
            if missing:
                raise BackendError(
                    f"{self.info.id} never computed {missing[0][0]} in layer {missing[0][1]}, so it "
                    "can't be read there."
                )
            return logits[:, -keep:, :].float(), seen

    def capture(
        self, tokens: torch.Tensor, sites: list[tuple[str, int]]
    ) -> dict[tuple[str, int], torch.Tensor]:
        names = {hook_name(kind, layer): (kind, layer) for kind, layer in sites}
        last = max(layer for _, layer in sites)
        kwargs: dict[str, Any] = {}
        if last + 1 < self.info.n_layers:
            kwargs["stop_at_layer"] = last + 1
        with self.lock, torch.no_grad():
            _, cache = self._bridge().run_with_cache(
                tokens.to(self.device), names_filter=list(names), **kwargs
            )
            return {site: cache[name].detach() for name, site in names.items()}

    def attention_pattern(self, tokens: torch.Tensor, layer: int) -> torch.Tensor:
        name = PATTERN_HOOK.format(layer=layer)
        kwargs: dict[str, Any] = {}
        if layer + 1 < self.info.n_layers:
            kwargs["stop_at_layer"] = layer + 1
        with self.lock, torch.no_grad():
            _, cache = self._bridge().run_with_cache(
                tokens.to(self.device), names_filter=[name], **kwargs
            )
            return cache[name].detach().float()

    def attention_patterns(
        self, tokens: torch.Tensor, layers: list[int]
    ) -> dict[int, torch.Tensor]:
        names = {PATTERN_HOOK.format(layer=layer): layer for layer in layers}
        kwargs: dict[str, Any] = {}
        if max(layers) + 1 < self.info.n_layers:
            kwargs["stop_at_layer"] = max(layers) + 1
        with self.lock, torch.no_grad():
            _, cache = self._bridge().run_with_cache(
                tokens.to(self.device), names_filter=list(names), **kwargs
            )
            return {layer: cache[name].detach().float() for name, layer in names.items()}

    def token_space(self) -> tuple[int, frozenset[int]]:
        # The tokenizer can have fewer ids than the embedding has rows (padded vocabularies).
        size = min(len(self.tokenizer), self.info.d_vocab)
        special = frozenset(int(i) for i in self.tokenizer.all_special_ids if i is not None)
        return size, special

    def _bridge(self) -> Any:
        if self.bridge is None:
            raise BackendError("The model was unloaded. Load it again to continue.")
        return self.bridge

    def edit_logits(
        self,
        tokens: torch.Tensor,
        kind: str,
        layer: int,
        edit: Any,
        keep: int = 1,
        read: str | None = None,
    ) -> torch.Tensor:
        if kind not in ("resid_pre", "resid_post", "attn_out", "mlp_out", "head"):
            raise BackendError(f"Editing {kind} activations isn't supported.")
        seen: dict[str, torch.Tensor] = {}

        def keep_read(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
            seen["read"] = act
            return act

        def fn(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
            out = edit(act) if read is None else edit(act, seen["read"])
            return out.to(dtype=act.dtype)

        hooks: list[tuple[str, Callable[..., torch.Tensor]]] = []
        if read is not None:
            hooks.append((hook_name(read, layer), keep_read))
        hooks.append((hook_name(kind, layer), fn))
        kwargs: dict[str, Any] = {"return_type": "logits"}
        if self._logits_to_keep:
            kwargs["logits_to_keep"] = keep
        with self.lock, torch.no_grad():
            logits = self._bridge().run_with_hooks(
                tokens.to(self.device), fwd_hooks=_combine(hooks), **kwargs
            )
            return logits[:, -keep:, :].float()

    def path_patch(
        self,
        tokens: torch.Tensor,
        sender: Patch,
        frozen_heads: dict[int, torch.Tensor],
        frozen_mlps: dict[int, torch.Tensor] | None,
        receivers: list[tuple[str, int, int, str]],
    ) -> torch.Tensor:
        n = self.info.n_layers
        tokens = tokens.to(self.device)

        def hold_heads(layer: int) -> Callable[..., torch.Tensor]:
            def fn(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
                out = frozen_heads[layer].to(device=act.device, dtype=act.dtype).clone()
                if sender.kind == "head" and sender.layer == layer:
                    rows = torch.arange(act.shape[0], device=act.device)
                    heads = sender.heads.to(act.device)  # type: ignore[union-attr]
                    values = sender.values.to(device=act.device, dtype=act.dtype)
                    if sender.positions is None:
                        out[rows, :, heads] = values
                    else:
                        out[rows, sender.positions.to(act.device), heads] = values
                return out

            return fn

        def hold(value: torch.Tensor) -> Callable[..., torch.Tensor]:
            def fn(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
                return value.to(device=act.device, dtype=act.dtype)

            return fn

        first: list[tuple[str, Callable[..., torch.Tensor]]] = [
            (hook_name("head", layer), hold_heads(layer)) for layer in range(n)
        ]
        if sender.kind == "resid_mid":
            first += _resid_mid_hooks(sender)
        elif sender.kind != "head":
            first.append((hook_name(sender.kind, sender.layer), _patch_hook(sender)))
        for layer, value in (frozen_mlps or {}).items():
            if not (sender.kind == "mlp_out" and sender.layer == layer):
                first.append((hook_name("mlp_out", layer), hold(value)))

        # What each receiver reads, recorded in the first pass and patched in the second: a head's
        # query, key or value (only that head's), or an MLP's own input before its normalization
        # (all of it: only that MLP reads it).
        final = hook_name("resid_post", n - 1)
        reads: dict[str, list[int] | None] = {}
        for kind, layer, head, part in receivers:
            if kind == "head":
                heads = reads.setdefault(f"blocks.{layer}.attn.hook_{part}", [])
                assert heads is not None
                heads.append(head)
            elif kind == "mlp":
                reads[edge_hook("mlp", layer)] = None
        logits_receiver = any(kind == "logits" for kind, *_ in receivers)
        recorded: dict[str, torch.Tensor] = {}

        def record(name: str) -> Callable[..., torch.Tensor]:
            def fn(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
                recorded[name] = act.detach().clone()
                return act

            return fn

        def replay(name: str, heads: list[int] | None) -> Callable[..., torch.Tensor]:
            def fn(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
                if heads is None:
                    return recorded[name].to(dtype=act.dtype)
                act = act.clone()
                act[:, :, heads] = recorded[name].to(dtype=act.dtype)[:, :, heads]
                return act

            return fn

        def add_direct(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
            if not reads:  # the logits are the only receiver: the first pass's stream, exactly
                return recorded[final].to(dtype=act.dtype)
            # The final residual stream also carries what the head receivers changed in this
            # pass: add only the change the sender made there directly (first pass minus the
            # receiver's own run), so neither path overwrites the other.
            change = recorded[final] - recorded["own"]
            return act + change.to(dtype=act.dtype)

        def keep_own(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
            recorded["own"] = act.detach().clone()
            return act

        kwargs: dict[str, Any] = {"return_type": None}
        if not logits_receiver:
            last = max(layer for kind, layer, *_ in receivers if kind in ("head", "mlp"))
            if last + 1 < n:
                kwargs["stop_at_layer"] = last + 1
        recording = [(name, record(name)) for name in reads]
        if logits_receiver:
            recording.append((final, record(final)))
        with self.lock, torch.no_grad():
            bridge = self._bridge()
            bridge.run_with_hooks(tokens, fwd_hooks=_combine(first + recording), **kwargs)
            if logits_receiver and reads:
                # The receiver run's own final residual stream, with every head held as in the
                # first pass, so the difference is the sender's direct effect alone.
                held = [(hook_name("head", layer), hold(frozen_heads[layer])) for layer in range(n)]
                for layer, value in (frozen_mlps or {}).items():
                    held.append((hook_name("mlp_out", layer), hold(value)))
                bridge.run_with_hooks(
                    tokens, fwd_hooks=_combine([*held, (final, keep_own)]), return_type=None
                )
            second: dict[str, Any] = {"return_type": "logits"}
            if self._logits_to_keep:
                second["logits_to_keep"] = 1
            hooks = [(name, replay(name, heads)) for name, heads in reads.items()]
            if logits_receiver:
                hooks.append((final, add_direct))
            logits = bridge.run_with_hooks(tokens, fwd_hooks=_combine(hooks), **second)
            return logits[:, -1, :].float()

    # -- edges (logogram.edges) -------------------------------------------------------------------

    @contextlib.contextmanager
    def _split_inputs(self) -> Iterator[Any]:
        """The bridge with each head reading its own copies of the residual stream for its query,
        key and value (TransformerLens's split inputs), as it was afterwards. Hold the lock."""
        bridge = self._bridge()
        before = bool(getattr(bridge.cfg, "use_split_qkv_input", False))
        if not before:
            bridge.set_use_split_qkv_input(True)
        try:
            yield bridge
        finally:
            if not before:
                bridge.set_use_split_qkv_input(False)

    def edge_checks(self) -> dict[str, float]:
        with self.lock:
            if self._edges is None:
                try:
                    self._edges = self._measure_edges()
                except BackendError as exc:
                    self._edges = str(exc)
            if isinstance(self._edges, str):
                raise BackendError(self._edges)
            return dict(self._edges)

    def _measure_edges(self) -> dict[str, float]:
        """See :meth:`ModelBackend.edge_checks`. Every check runs on a short fixed input."""
        info = self.info
        checks = info.extra.get("checks") or {}
        tolerance = float(checks.get("tolerance") or CHECK_TOLERANCE[info.dtype])
        why = "so edges can't be measured exactly on it"
        if str(info.extra.get("block_structure")) not in (
            "sequential",
            "sequential_pre_norm",
            "parallel",
        ):
            raise BackendError(
                f"When {info.id} loaded, Logogram couldn't verify how its layers add attention and "
                f"MLP into the residual stream, {why}."
            )
        heads = checks.get("heads")
        if heads is None or not heads <= tolerance:
            raise BackendError(
                f"{info.id}'s heads don't add up to what its attention writes (it normalizes after "
                f"combining them, or this couldn't be measured when it loaded), {why}. Edge "
                "attribution and ACDC need a model whose heads each write into the residual "
                "stream, such as GPT-2, Pythia, Llama or Qwen 2."
            )
        kv = int(info.extra.get("n_key_value_heads") or info.n_heads)
        if kv < 1 or info.n_heads % kv:
            raise BackendError(
                f"{info.id} has {info.n_heads} heads and {kv} key/value heads a layer, which don't "
                f"divide into groups of query heads, {why}."
            )
        bridge = self._bridge()
        probe = _probe_tokens(info.d_vocab, str(bridge.cfg.device))
        try:
            with torch.no_grad():
                plain = bridge(probe, return_type="logits").float()
                with self._split_inputs():
                    split = bridge(probe, return_type="logits").float()
                    stream = self._copies_are_the_stream(bridge, probe)
                    separate = self._separate_copies(bridge, probe)
        except BackendError:
            raise
        except Exception as exc:  # attention TransformerLens can't split into copies
            raise BackendError(
                f"TransformerLens can't give {info.id}'s heads their own copies of their inputs "
                f"({type(exc).__name__}: {exc}), {why}."
            ) from exc
        out = {"split": _relative(split, plain), "stream": stream, "separate": separate}
        if not out["split"] <= tolerance:
            raise BackendError(
                f"With each head reading its own copy of its inputs, {info.id} predicts something "
                f"else (log-probabilities differ by up to {out['split']:.1%} of their range), {why}."
            )
        if not out["stream"] <= tolerance:
            raise BackendError(
                f"{info.id}'s heads, MLPs or logits don't each read the residual stream itself and "
                f"normalize their own copy, {why}."
            )
        if not out["separate"] <= tolerance:
            raise BackendError(
                f"Changing one head's input copy changes other heads of {info.id}, so its heads "
                f"don't read their own copies, {why}."
            )
        return out

    def _copies_are_the_stream(self, bridge: Any, probe: torch.Tensor) -> float:
        """How far each receiver's own copy is from the residual stream it copies, relative to the
        stream: 0 when every head's query, key and value, every MLP and the logits read the stream
        itself and normalize their copy on their own, whatever their normalization's epsilon. A
        copy taken after a normalization would differ."""
        info = self.info
        n = info.n_layers
        parallel = str(info.extra.get("block_structure")) == "parallel"
        mlp_reads = "resid_pre" if parallel else "resid_mid"
        pairs = [
            (edge_hook(part, layer), hook_name("resid_pre", layer))
            for layer in range(n)
            for part in ("q", "k", "v")
        ]
        pairs += [(edge_hook("mlp", layer), hook_name(mlp_reads, layer)) for layer in range(n)]
        pairs.append((edge_hook("logits", n), hook_name("resid_post", n - 1)))
        names = sorted({name for pair in pairs for name in pair})
        _, cache = bridge.run_with_cache(probe, names_filter=names)
        worst = 0.0
        for copy_name, stream_name in pairs:
            copy, stream = cache[copy_name], cache[stream_name]
            if copy.dim() == stream.dim() + 1:  # one copy for each head: [batch, pos, head, d]
                stream = stream.unsqueeze(2).expand_as(copy)
            worst = max(worst, _relative(copy, stream))
        return worst

    def _separate_copies(self, bridge: Any, probe: torch.Tensor) -> float:
        """Move one head's query, key or value copy in the last layer: that head's query, key or
        value (its projection of the copy) must change, and no other head's. With grouped-query
        attention a key or value copy is a key/value head's, and the heads compared are the
        key/value heads. The largest change of another head, relative to the moved head's,
        measured three times (q, k, v). The head moved is the first whose projection responds: a
        head with no query weights (one that attends evenly) reads its copy without changing."""
        layer = self.info.n_layers - 1
        kv = int(self.info.extra.get("n_key_value_heads") or self.info.n_heads)
        generator = torch.Generator(device="cpu").manual_seed(0)
        worst = 0.0
        for part in ("q", "k", "v"):
            heads = self.info.n_heads if part == "q" else kv
            name = HOOKS[part].format(layer=layer)  # [B, pos, H (or key/value heads), d_head]
            _, base = bridge.run_with_cache(probe, names_filter=[name])
            before = base[name].float()
            shape = (probe.shape[0], probe.shape[1], self.info.d_model)
            delta = torch.randn(shape, generator=generator).to(probe.device)
            for head in range(heads):
                seen: dict[str, torch.Tensor] = {}

                def keep(act: torch.Tensor, hook: Any = None, seen: Any = seen) -> torch.Tensor:
                    seen["x"] = act.detach().float()
                    return act

                def move(
                    act: torch.Tensor,
                    hook: Any = None,
                    delta: torch.Tensor = delta,
                    head: int = head,
                ) -> torch.Tensor:
                    act = act.clone()
                    copy = act[:, :, head]
                    act[:, :, head] = copy + delta.to(act.dtype) * copy.std()
                    return act

                bridge.run_with_hooks(
                    probe,
                    fwd_hooks=[(edge_hook(part, layer), move), (name, keep)],
                    return_type=None,
                )
                change = (seen["x"] - before).abs().movedim(2, 0)
                moved = float(change[head].max())
                if moved > 0:
                    others = [h for h in range(heads) if h != head]
                    if others:
                        worst = max(worst, float(change[others].max()) / moved)
                    break
            else:
                return math.inf  # no head read the copy that was moved
        return worst

    def head_output_weights(self, layer: int) -> torch.Tensor:
        with self.lock, torch.no_grad():
            return float64(self._bridge().blocks[layer].attn.W_O.detach())

    def input_gradients(
        self,
        tokens: torch.Tensor,
        score: ScoreFn,
        layers: list[int],
        logits: bool,
        embeddings: torch.Tensor | None = None,
    ) -> InputGradients:
        n = self.info.n_layers
        probes: dict[tuple[str, int], torch.Tensor] = {}

        def probe(key: tuple[str, int]) -> Callable[..., torch.Tensor]:
            # A zero added to the copy: the gradient with respect to it is the gradient with
            # respect to the copy, through the receiver's normalization as the model computes it.
            def fn(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
                zero = torch.zeros_like(act, requires_grad=True)
                probes[key] = zero
                return act + zero

            return fn

        hooks: list[tuple[str, Callable[..., torch.Tensor]]] = []
        if embeddings is not None:

            def interpolate(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
                act = act.clone()
                act[:, : embeddings.shape[1]] = embeddings.to(dtype=act.dtype, device=act.device)
                return act

            hooks.append((hook_name("resid_pre", 0), interpolate))
        for layer in layers:
            for part in ("q", "k", "v", "mlp"):
                hooks.append((edge_hook(part, layer), probe((part, layer))))
        if logits:
            hooks.append((edge_hook("logits", n), probe(("logits", n))))
        kwargs: dict[str, Any] = {"return_type": "logits"}
        if self._logits_to_keep:
            kwargs["logits_to_keep"] = 1
        with self.lock, torch.enable_grad(), self._split_inputs() as bridge:
            out = bridge.run_with_hooks(tokens.to(self.device), fwd_hooks=_combine(hooks), **kwargs)
            values = score(out[:, -1:, :])
            order = list(probes)
            grads = torch.autograd.grad(values.sum(), [probes[k] for k in order])
        got = {key: g.detach() for key, g in zip(order, grads, strict=True)}

        def per_layer(part: str) -> list[torch.Tensor | None]:
            return [got.get((part, layer)) for layer in range(n)]

        return InputGradients(
            q=per_layer("q"),
            k=per_layer("k"),
            v=per_layer("v"),
            mlp=per_layer("mlp"),
            logits=got.get(("logits", n)),
        )

    def edge_patch(self, tokens: torch.Tensor, patcher: EdgePatcher) -> torch.Tensor:
        n = self.info.n_layers

        def sent(node: str, layer: int) -> Callable[..., torch.Tensor]:
            def fn(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
                patcher.sent(node, layer, act.detach())
                return act

            return fn

        def add(receiver: str, layer: int) -> Callable[..., torch.Tensor]:
            def fn(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
                update = patcher.update(receiver, layer)
                if update is None:
                    return act
                return act + update.to(dtype=act.dtype, device=act.device)

            return fn

        # In the order the pass reaches them: a layer's copies, its heads' outputs, its MLP's
        # input and output; the final normalization's input last.
        hooks: list[tuple[str, Callable[..., torch.Tensor]]] = [
            (hook_name("resid_pre", 0), sent("embed", -1))
        ]
        for layer in range(n):
            hooks += [(edge_hook(part, layer), add(part, layer)) for part in ("q", "k", "v")]
            hooks.append((hook_name("head", layer), sent("head", layer)))
            hooks.append((edge_hook("mlp", layer), add("mlp", layer)))
            hooks.append((hook_name("mlp_out", layer), sent("mlp", layer)))
        hooks.append((edge_hook("logits", n), add("logits", n)))
        kwargs: dict[str, Any] = {"return_type": "logits"}
        if self._logits_to_keep:
            kwargs["logits_to_keep"] = 1
        with self.lock, torch.no_grad(), self._split_inputs() as bridge:
            logits = bridge.run_with_hooks(
                tokens.to(self.device), fwd_hooks=_combine(hooks), **kwargs
            )
            return logits[:, -1, :].float()

    def gradients(
        self,
        tokens: torch.Tensor,
        sites: list[tuple[str, int]],
        score: ScoreFn,
        keep: int = 1,
        embeddings: torch.Tensor | None = None,
    ) -> tuple[dict[tuple[str, int], torch.Tensor], dict[tuple[str, int], torch.Tensor]]:
        # A zero tensor added at each hook point: the gradient with respect to it is the gradient
        # with respect to the activation there, through every later use, without cutting the graph.
        # hook_resid_mid only feeds the MLP's normalization (see _resid_mid_hooks); the residual
        # stream between attention and MLP is resid_pre + attn_out, so its gradient is taken at
        # attn_out, which is added to it.
        kept: dict[tuple[str, int], torch.Tensor] = {}
        zeros: dict[tuple[str, int], torch.Tensor] = {}

        def value(site: tuple[str, int]) -> Callable[..., torch.Tensor]:
            def fn(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
                kept[site] = act.detach()
                return act

            return fn

        def probe(site: tuple[str, int]) -> Callable[..., torch.Tensor]:
            def fn(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
                zero = torch.zeros_like(act, requires_grad=True)
                zeros[site] = zero
                if site[0] != "resid_mid":
                    kept[site] = act.detach()
                return act + zero

            return fn

        hooks: list[tuple[str, Callable[..., torch.Tensor]]] = []
        if embeddings is not None:
            # Replace the stream entering the first layer before anything reads or probes it.
            def interpolate(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
                act = act.clone()
                n = embeddings.shape[1]
                act[:, :n] = embeddings.to(dtype=act.dtype, device=act.device)
                return act

            hooks.append((hook_name("resid_pre", 0), interpolate))
        for kind, layer in dict.fromkeys(sites):
            if kind == "resid_mid":
                hooks.append((hook_name("resid_mid", layer), value((kind, layer))))
                hooks.append((hook_name("attn_out", layer), probe((kind, layer))))
            else:
                hooks.append((hook_name(kind, layer), probe((kind, layer))))
        kwargs: dict[str, Any] = {"return_type": "logits"}
        if self._logits_to_keep:
            kwargs["logits_to_keep"] = keep
        with self.lock, torch.enable_grad():
            bridge = self._bridge()
            logits = bridge.run_with_hooks(
                tokens.to(self.device), fwd_hooks=_combine(hooks), **kwargs
            )
            values = score(logits[:, -keep:, :])
            order = list(zeros)
            grads = torch.autograd.grad(values.sum(), [zeros[s] for s in order])
        return kept, {site: g.detach() for site, g in zip(order, grads, strict=True)}

    def direct_effects(
        self,
        tokens: torch.Tensor,
        answers: torch.Tensor,
        distractors: torch.Tensor,
        heads: bool,
        *,
        neurons: Sequence[int] = (),
        values: Sequence[int] = (),
    ) -> dict[str, torch.Tensor]:
        n = self.info.n_layers
        kept: dict[str, torch.Tensor] = {}

        def keep(name: str, last: bool = True) -> Callable[..., torch.Tensor]:
            def fn(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
                # Only the last position is read out; a value is read at every position.
                kept[name] = (act[:, -1] if last else act).detach()
                return act

            return fn

        def keep_last_query(name: str) -> Callable[..., torch.Tensor]:
            def fn(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
                kept[name] = act[:, :, -1].detach()  # [B, H, key]: the last token's attention
                return act

            return fn

        names = [hook_name("resid_pre", 0), hook_name("resid_post", n - 1)]
        names += [hook_name(k, layer) for layer in range(n) for k in ("attn_out", "mlp_out")]
        if heads:
            names += [hook_name("head", layer) for layer in range(n)]
        names += [hook_name("neuron", layer) for layer in neurons]
        fwd = [(name, keep(name)) for name in dict.fromkeys(names)]
        fwd += [
            (hook_name("v", layer), keep(hook_name("v", layer), last=False)) for layer in values
        ]
        fwd += [
            (PATTERN_HOOK.format(layer=layer), keep_last_query(PATTERN_HOOK.format(layer=layer)))
            for layer in values
        ]
        with self.lock:
            bridge = self._bridge()
            with torch.no_grad():
                bridge.run_with_hooks(tokens.to(self.device), fwd_hooks=fwd, return_type=None)

            # The direction in the residual stream that the logit difference reads, with the final
            # normalization's scale held at its value for each prompt: the logit difference is then
            # an affine function of the residual stream, and splits over its components.
            def hold(scale: torch.Tensor, hook: Any = None) -> torch.Tensor:
                return scale.detach()

            with torch.enable_grad(), warnings.catch_warnings():
                # TransformerLens warns that an edited scale is recomputed from the hooked values;
                # that is the point, and the result is checked against the measured logit difference.
                warnings.simplefilter("ignore")
                final = kept[hook_name("resid_post", n - 1)].clone().requires_grad_(True)
                with bridge.hooks(fwd_hooks=[(FINAL_SCALE, hold)]):
                    logits = final_projection(bridge, final[:, None, :])[:, 0]
                rows = torch.arange(final.shape[0], device=final.device)
                ld = (
                    logits[rows, answers.to(final.device)]
                    - logits[rows, distractors.to(final.device)]
                )
                (direction,) = torch.autograd.grad(ld.sum(), final)
            g = float64(direction)

            def term(vector: torch.Tensor) -> torch.Tensor:
                return (float64(vector) * g).sum(-1)

            out: dict[str, torch.Tensor] = {
                "embed": term(kept[hook_name("resid_pre", 0)]),
                "attn_out": torch.stack(
                    [term(kept[hook_name("attn_out", layer)]) for layer in range(n)], dim=1
                ),
                "mlp_out": torch.stack(
                    [term(kept[hook_name("mlp_out", layer)]) for layer in range(n)], dim=1
                ),
                "logit_diff": float64(ld.detach()),
                # How large the two logits are, which sets how finely the dtype resolves them.
                "logit_scale": float64(
                    torch.maximum(
                        logits[rows, answers.to(final.device)].abs(),
                        logits[rows, distractors.to(final.device)].abs(),
                    ).detach()
                ),
            }
            if heads:
                per_layer = []
                for layer in range(n):
                    z = float64(kept[hook_name("head", layer)])  # [B, H, d_head]
                    w_o = float64(bridge.blocks[layer].attn.W_O.detach())  # [H, d_head, d_model]
                    written = torch.einsum("bhd,hdm->bhm", z, w_o)
                    per_layer.append((written * g[:, None, :]).sum(-1))
                out["head"] = torch.stack(per_layer, dim=1)
            if neurons:
                # A neuron writes its activation times its row of the output projection.
                per_layer = []
                for layer in neurons:
                    post = float64(kept[hook_name("neuron", layer)])  # [B, d_mlp]
                    w_out = float64(bridge.blocks[layer].mlp.W_out.detach())  # [d_mlp, d_model]
                    per_layer.append(post * (g @ w_out.T))
                out["neuron"] = torch.stack(per_layer, dim=1)
            if values:
                # A value at each position reaches the last token through every query head that
                # reads it: that head's attention to the position, times the value through the
                # head's share of the output projection (which g reads back as u).
                group = kv_group(self.info.n_heads, int(self.info.extra["n_key_value_heads"]))
                per_layer = []
                for layer in values:
                    value = float64(kept[hook_name("v", layer)])  # [B, pos, H_kv, d_head]
                    pattern = float64(kept[PATTERN_HOOK.format(layer=layer)])  # [B, H, pos]
                    w_o = float64(bridge.blocks[layer].attn.W_O.detach())  # [H, d_head, d_model]
                    u = torch.einsum("hdm,bm->bhd", w_o, g)
                    read = value.repeat_interleave(group, dim=2)  # [B, pos, H, d_head]
                    each = pattern * torch.einsum("bphd,bhd->bhp", read, u)  # [B, H, pos]
                    B, H, P = each.shape
                    per_layer.append(each.reshape(B, H // group, group, P).sum(2))
                out["value"] = torch.stack(per_layer, dim=1)
        total = out["embed"] + out["attn_out"].sum(1) + out["mlp_out"].sum(1)
        out["remainder"] = out["logit_diff"] - total
        return {k: v.cpu() for k, v in out.items()}

    def unembed_directions(self, residual: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        if self.info.extra.get("prediction_method") != "final_norm_logit_lens":
            raise BackendError(
                "Reading a direction into the logits needs the model's final normalization and "
                "unembedding to reproduce its output, and for this model they didn't when it "
                "loaded."
            )
        with self.lock, torch.no_grad():
            bridge = self._bridge()
            if FINAL_SCALE not in bridge.hook_dict:
                raise BackendError(
                    "This model's final normalization has no scale to hold, so a direction can't "
                    "be read through it linearly."
                )
            dtype = DTYPES[self.info.dtype]
            held: dict[str, torch.Tensor] = {}

            def keep(scale: torch.Tensor, hook: Any = None) -> torch.Tensor:
                held["scale"] = scale.detach().clone()
                return scale

            def hold(scale: torch.Tensor, hook: Any = None) -> torch.Tensor:
                return held["scale"].to(scale.dtype).expand_as(scale).clone()

            # The scale at this token, then each vector and zero read with it held: with the scale
            # held, the normalization and unembedding are affine, so their difference is exactly
            # what the vector adds (biases cancel).
            with bridge.hooks(fwd_hooks=[(FINAL_SCALE, keep)]):
                bridge.ln_final(residual.to(device=self.device, dtype=dtype).reshape(1, 1, -1))
            batch = vectors.to(device=self.device, dtype=dtype)[:, None, :]
            batch = torch.cat([batch, torch.zeros_like(batch[:1])])
            with warnings.catch_warnings():
                # The bridge warns that the held scale is used instead of the computed one: that
                # is the point.
                warnings.simplefilter("ignore")
                with bridge.hooks(fwd_hooks=[(FINAL_SCALE, hold)]):
                    logits = bridge.unembed(bridge.ln_final(batch))[:, 0].float()
            return (logits[:-1] - logits[-1:]).cpu()

    def head_writes(self, layer: int, z: torch.Tensor) -> torch.Tensor:
        with self.lock, torch.no_grad():
            w_o = float64(self._bridge().blocks[layer].attn.W_O.detach())  # [H, d_head, d_model]
            return torch.einsum("khd,hdm->km", float64(z).to(w_o.device), w_o).cpu()

    def read_out(self, residual: torch.Tensor) -> torch.Tensor:
        if self.info.extra.get("prediction_method") != "final_norm_logit_lens":
            raise BackendError(
                "Reading the residual stream into logits needs the model's final normalization "
                "and unembedding to reproduce its output, and for this model they didn't when it "
                "loaded."
            )
        with self.lock:
            bridge = self._bridge()
            shape = residual.shape
            stream = residual.to(device=self.device, dtype=DTYPES[self.info.dtype])
            # As the logit lens reads a token: one position of a batch, [rows, 1, d_model].
            logits = final_projection(bridge, stream.reshape(-1, 1, shape[-1]))
            return logits.float().reshape(*shape[:-1], logits.shape[-1])

    def layer_logits(
        self,
        tokens: torch.Tensor,
        position: int,
        row: int,
        translators: Mapping[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        if self.info.extra.get("prediction_method") != "final_norm_logit_lens":
            raise BackendError(
                "Per-layer predictions need the model's final normalization and unembedding to "
                "reproduce its output, and for this model they didn't when it loaded. The other "
                "analyses remain available."
            )
        if not 0 <= position < tokens.shape[1] or not 0 <= row < tokens.shape[0]:
            raise ValueError("The prediction token position or batch row is out of range.")
        lens = dict(translators or {})
        with self.lock, torch.no_grad():
            bridge = self._bridge()
            captured: dict[int, torch.Tensor] = {}

            def project(layer: int) -> Callable[..., torch.Tensor]:
                def keep(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
                    # Recompute final normalization at each layer. Cached final-layer scales
                    # would implement attribution, not the logit lens. Keep the original
                    # batch shape and dtype through both modules, including learned biases.
                    residual = act[:, position : position + 1, :]
                    if layer in lens:
                        # A tuned lens's affine translator, in float32, before the readout.
                        weight, bias = lens[layer]
                        moved = residual.float() @ weight.to(act.device) + bias.to(act.device)
                        residual = moved.to(act.dtype)
                    logits = final_projection(bridge, residual)
                    captured[layer] = logits[row, 0].detach().float().cpu()
                    return act

                return keep

            hooks = [
                (hook_name("resid_post", layer), project(layer))
                for layer in range(self.info.n_layers)
            ]
            bridge.run_with_hooks(tokens.to(self.device), fwd_hooks=hooks, return_type=None)
            if len(captured) != self.info.n_layers:
                raise BackendError("This model didn't expose every layer's residual output.")
            return torch.stack([captured[layer] for layer in range(self.info.n_layers)])

    def close(self) -> None:
        # Waits for a forward pass in progress; callers hold the lock across multi-step work.
        with self.lock:
            self.bridge = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def architecture_support(architecture: str) -> str | None:
    """None if TransformerLens can load this Hugging Face architecture; otherwise, why not."""
    try:
        import transformer_lens.model_bridge  # noqa: F401 - loads before the factory (a cycle)
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )
    except Exception:  # noqa: BLE001 - can't tell; loading the model will say
        return None
    if architecture == "unknown" or architecture in SUPPORTED_ARCHITECTURES:
        return None
    return (
        f"TransformerLens {version('transformer-lens')} can't load {architecture} models, so "
        "Logogram can't either. Choose a model of a supported family, such as GPT-2, Llama, "
        "Qwen, Gemma, Pythia or OLMo."
    )


def _device_name(device: str) -> str:
    """The name of the device a model runs on: the GPU it is on, by its index."""
    if device_type(device) == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.device(device))
    if device == "mps":
        return "Apple GPU (Metal)"
    from logogram.system import cpu_name

    return cpu_name()


def full_device(device: str) -> str:
    """A device as a model on it runs: a GPU always by its index ("cuda" is the default GPU,
    the one torch's current device names)."""
    if device == "cuda":
        return f"cuda:{torch.cuda.current_device()}"
    return device


def resolve_device(device: str) -> str:
    """What a device (a spec's model.device, or the one chosen in the app or passed to
    ``load_model``) runs on here: "cpu", "mps", or one GPU by its index. "cuda" is the default GPU
    (as torch's current device names it, normally GPU 0), so "cuda" and "cuda:0" resolve alike
    there and a model loaded with either matches a spec that asks for the other. A device this
    machine doesn't have is refused, saying what to choose instead."""
    if device == "auto":
        # Apple's MPS only when asked for: TransformerLens reports that it can give silently wrong
        # results, and Logogram hasn't been checked on it yet (see system.MPS_NOTE).
        return full_device("cuda") if torch.cuda.is_available() else "cpu"
    if device_type(device) == "cuda":
        if not torch.cuda.is_available():
            raise BackendError(
                "CUDA was chosen, but PyTorch can't see a CUDA GPU here. Run `logogram doctor` "
                "for the fix, or choose the device auto or cpu (model.device in a spec)."
            )
        if device == "cuda":
            return full_device(device)
        index = int(device.split(":", 1)[1])
        count = torch.cuda.device_count()
        if index >= count:
            have = "one GPU, cuda:0" if count == 1 else f"{count} GPUs, cuda:0 to cuda:{count - 1}"
            raise BackendError(
                f"GPU {index} ({device}) was chosen, but this machine has {have}. Choose one of "
                "them, or cuda for the default GPU (model.device in a spec); `logogram doctor` "
                "lists them."
            )
        return f"cuda:{index}"
    if device == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise BackendError(
            "Apple's MPS was chosen, but it isn't available here. Choose the device auto or cpu "
            "(model.device in a spec)."
        )
    return device


def check_request(
    model_id: str,
    *,
    revision: str | None,
    dtype: str,
    device: str,
    source: str,
    sha256: str | None,
) -> str:
    """What can be checked about loading a model before anything is downloaded or loaded: the
    dtype, that the device is there, and for a local model that this machine has it and its folder
    is still there. Returns the device as it resolves here. The app checks this before it closes
    the model that is loaded, so a request that can't load (a GPU this machine lacks, say) leaves
    that model loaded."""
    if dtype not in DTYPES:
        raise BackendError(f"Unknown dtype {dtype!r}; use float32, float16 or bfloat16.")
    resolved = resolve_device(device)
    if source == "local":
        from logogram import local_models

        if revision is not None:
            raise BackendError(
                "A local model has no revisions: its files are pinned by their hash."
            )
        local_models.registered_folder(model_id, sha256)
    return resolved


def load_model(
    model_id: str,
    *,
    revision: str | None = None,
    dtype: str = "float32",
    device: str = "auto",
    process_weights: bool = True,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    cancel: threading.Event | None = None,
    source: str = "hub",
    sha256: str | None = None,
) -> TransformerLensBackend:
    """Resolve, download (with progress) and load a model from the Hugging Face Hub, or, with
    ``source="local"``, load the local model whose files hash to ``sha256`` (added under the name
    ``model_id``; see :mod:`logogram.local_models`), once its files are checked.

    ``cancel`` stops a download promptly and a load at the next step; it raises ``Cancelled``.
    """
    from logogram.backends import hub

    def emit(stage: str, **data: Any) -> None:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        if on_progress:
            on_progress({"stage": stage, **data})

    resolved_device = check_request(
        model_id, revision=revision, dtype=dtype, device=device, source=source, sha256=sha256
    )
    if source == "local":
        from logogram import local_models

        emit("checking", model_id=model_id, done=0, total=0, file="")

        def on_check(done: int, total: int, file: str) -> None:
            emit("checking", model_id=model_id, done=done, total=total, file=file)

        folder = local_models.checked_folder(model_id, sha256, on_progress=on_check, cancel=cancel)
        pinned: str | None = None
    else:
        emit("resolving", model_id=model_id)
        repo = hub.resolve(model_id, revision)
        emit("downloading", done=0, total=sum(s for _, s in repo.files), file="")

        def on_download(done: int, total: int, file: str) -> None:
            emit("downloading", done=done, total=total, file=file)

        folder = hub.download(model_id, repo, on_download, cancel=cancel)
        pinned = repo.revision
        sha256 = None
    emit("loading", revision=pinned)

    def boot_and_check() -> TransformerLensBackend:
        bridge = boot_local(folder, device=resolved_device, dtype=dtype)
        emit("processing")
        return TransformerLensBackend.from_bridge(
            bridge,
            model_id=model_id,
            revision=pinned,
            dtype=dtype,
            process_weights=process_weights,
            source=source,
            sha256=sha256,
        )

    release_memory()  # what an earlier attempt that failed may still hold
    try:
        backend: TransformerLensBackend | None = boot_and_check()
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        if not is_out_of_memory(exc):
            raise
        backend = None
    if backend is None:
        # The partly loaded model is freed once the exception and its frames are gone; give its
        # memory back, so the next attempt has it.
        release_memory()
        raise BackendError(out_of_memory_message(resolved_device, process_weights))
    if cancel is not None and cancel.is_set():
        backend.close()
        raise Cancelled()
    if source == "local":
        # A toy model (logogram.toys), known by the hash of its files, is checked as built: it
        # solves its prompts and its planted heads attend where they should.
        from logogram.toys import check_loaded_toy

        try:
            check_loaded_toy(backend)
        except BackendError:
            backend.close()
            raise
    backend.requested_device = device
    emit("ready")
    return backend


def is_out_of_memory(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    # Some devices (Apple's MPS, the CPU allocator) report it as a plain RuntimeError.
    text = str(exc)
    return (
        isinstance(exc, RuntimeError)
        and not isinstance(exc, BackendError)
        and ("out of memory" in text or "can't allocate memory" in text)
    )


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        mps.empty_cache()


def memory_name(device: str) -> str:
    """The memory a device computes in, in words: "GPU memory", "unified memory" or "memory"."""
    return {"cuda": "GPU memory", "mps": "unified memory"}.get(device_type(device), "memory")


def analysis_out_of_memory_message(device: str | None) -> str:
    """What an analysis that ran out of memory says (the app's Explore views, Your text)."""
    where = memory_name(device) if device is not None else "memory"
    return (
        f"The analysis ran out of {where}. Run it on fewer prompts or with a smaller batch size "
        "(or a shorter text), or load a smaller model or a 16-bit dtype, then try again."
    )


def out_of_memory_message(device: str, process_weights: bool) -> str:
    where = memory_name(device)
    if process_weights:
        return (
            f"The model doesn't fit in {where} while its weights are processed: processing works "
            "on float32 copies of the weights and needs several times their size for a moment. "
            "Load it with weight processing off, or choose a smaller model or another device."
        )
    return (
        f"The model doesn't fit in {where}. Choose a smaller model, a 16-bit dtype or another "
        "device."
    )


# Where boot_local records the model's own beginning-of-sequence token.
_OWN_BOS = "_logogram_bos_token_id"


def own_bos_token_id(bridge: Any) -> int | None:
    """The beginning-of-sequence token of the model's own tokenizer, or None if it has none.

    TransformerLens gives a tokenizer without one a substitute (for Qwen, its end-of-text token,
    which the model never saw at the start of a sequence), so :func:`boot_local` records the
    model's own before the bridge's tokenizer replaces it. A bridge booted elsewhere falls back to
    its tokenizer's.
    """
    if _OWN_BOS in bridge.__dict__:
        return bridge.__dict__[_OWN_BOS]
    return getattr(bridge.tokenizer, "bos_token_id", None)


# Where boot_local records the model's own chat template and end-of-sequence tokens.
_CHAT = "_logogram_chat"
# Words in a model's name that mark it as made for chat: instruction-tuned (gemma-3-270m-it,
# Qwen2.5-0.5B-Instruct), chat models, and the post-training stages that make them.
_CHAT_NAME = re.compile(
    r"(?:^|[-_.])(it|instruct|instruction|chat|sft|dpo|rlhf|orpo)(?=$|[-_.\d])", re.IGNORECASE
)


def chat_facts(tokenizer: Any, model: Any) -> tuple[str | None, list[str]]:
    """A tokenizer's chat template (None without one), and the tokens the model ends a sequence
    with: its tokenizer's end-of-sequence token and those of its configuration and generation
    configuration."""
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, dict):  # several named templates
        template = "\n".join(str(t) for t in template.values())
    elif isinstance(template, list):
        template = "\n".join(str(t.get("template", "")) for t in template if isinstance(t, dict))
    ids: set[int] = set()
    for source in (getattr(model, "config", None), getattr(model, "generation_config", None)):
        value = getattr(source, "eos_token_id", None)
        ids.update([value] if isinstance(value, int) else value or [])
    ends = [str(t) for t in tokenizer.convert_ids_to_tokens(sorted(i for i in ids if i >= 0))]
    if tokenizer.eos_token:
        ends.append(str(tokenizer.eos_token))
    return (template or None) if isinstance(template, str) else None, sorted(set(ends))


def chat_format(model_id: str, template: str | None, ends: list[str]) -> dict[str, Any]:
    """Whether a model is made for chat, which matters because Logogram feeds prompts as plain
    text, without the chat template such models are trained on. ``template`` is its tokenizer's
    chat template; ``ends`` are the tokens it ends a sequence with.

    A model is made for chat when its name says so, or when it stops at a token that ends a turn
    in its chat template (Qwen3-0.6B, whose name doesn't say). A chat template alone isn't
    enough: some base models ship one (Qwen2.5-0.5B), while they stop at their end of text."""
    reason = None
    match = _CHAT_NAME.search(model_id.rsplit("/", 1)[-1])
    if match:
        reason = f"its name marks it as made for chat ({match.group(1)})"
    elif template:
        turn = next((t for t in ends if len(t) > 1 and t in template), None)
        if turn is not None:
            reason = f"it stops at {turn}, which ends each turn in its chat template"
    return {"instruct": reason is not None, "template": template is not None, "reason": reason}


# Where boot_local records the template a tokenizer uses by default, and its special tokens.
_CHAT_TEMPLATE = "_logogram_chat_template"


def default_chat_template(tokenizer: Any) -> tuple[str | None, dict[str, Any]]:
    """The chat template a tokenizer uses when none is named (of several named ones, "default"),
    or None; and its special tokens (bos_token, eos_token, ...), which templates read."""
    template = None
    if getattr(tokenizer, "chat_template", None):
        try:
            template = tokenizer.get_chat_template()
        except Exception:  # noqa: BLE001 - named templates without a default: none to use
            template = None
    tokens = dict(getattr(tokenizer, "special_tokens_map", None) or {})
    return (template if isinstance(template, str) and template else None), tokens


def own_chat_template(bridge: Any) -> tuple[str | None, dict[str, Any]]:
    """The model's own default chat template and special tokens, as :func:`boot_local` read them
    before the bridge's tokenizer could change them (a bridge booted elsewhere: its tokenizer's)."""
    recorded = bridge.__dict__.get(_CHAT_TEMPLATE)
    return recorded if recorded is not None else default_chat_template(bridge.tokenizer)


def boot_local(folder: Path, *, device: str, dtype: str) -> Any:
    from transformer_lens.model_bridge import TransformerBridge
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from logogram.backends.hub import validate_local_weights

    _quiet_transformers()
    validate_local_weights(folder)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # The bridge otherwise lets Transformers fall back to pickle weights. Load both
            # objects explicitly and locally so neither a cache nor an index can change that.
            model = AutoModelForCausalLM.from_pretrained(
                str(folder),
                use_safetensors=True,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=DTYPES[dtype],
                attn_implementation="eager",
            )
            model = model.to(device)  # type: ignore[arg-type]  # transformers types .to oddly
            tokenizer = AutoTokenizer.from_pretrained(
                str(folder),
                local_files_only=True,
                trust_remote_code=False,
            )
            own_bos = tokenizer.bos_token_id  # read first: booting gives the tokenizer a substitute
            chat = chat_facts(tokenizer, model)
            template = default_chat_template(tokenizer)
            bridge = TransformerBridge.boot_transformers(
                str(folder),
                device=device,
                dtype=DTYPES[dtype],
                hf_model=model,
                tokenizer=tokenizer,
                trust_remote_code=False,
            )
            bridge.__dict__[_OWN_BOS] = own_bos
            bridge.__dict__[_CHAT] = chat
            bridge.__dict__[_CHAT_TEMPLATE] = template
            return bridge
    except BackendError:
        raise
    except Exception as exc:
        if is_out_of_memory(exc):
            raise  # load_model frees the memory and says what to do
        raise BackendError(
            f"TransformerLens couldn't load this model ({type(exc).__name__}: {exc}). Models "
            "TransformerLens supports are listed in its documentation."
        ) from exc


def _quiet_transformers() -> None:
    try:
        import transformers

        transformers.utils.logging.set_verbosity_error()
        transformers.utils.logging.disable_progress_bar()
    except Exception:  # noqa: BLE001, S110 - quieter logs are a nicety
        pass
