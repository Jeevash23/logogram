"""TransformerLens backend, built on ``TransformerBridge`` (TransformerLens 4).

Abstract sites map to TransformerLens hook points here and nowhere else.
"""

from __future__ import annotations

import gc
import logging
import math
import threading
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from logogram.backends.base import (
    ALL_KINDS,
    BackendError,
    Cancelled,
    ModelBackend,
    ModelInfo,
    Patch,
    Patches,
    ScoreFn,
    Tokenized,
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
    # The MLP's input, normalized as the MLP reads it: what transcoders read. Not a site to patch.
    "mlp_in": "blocks.{layer}.mlp.hook_in",
}
PATTERN_HOOK = "blocks.{layer}.attn.hook_pattern"

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
        if patch.mask is not None:
            n = values.shape[1]  # the prompt's positions; appended tokens are never patched
            mask = patch.mask.to(act.device)[..., None]
            act[:, :n] = torch.where(mask, values, act[:, :n])
        elif patch.kind == "head":
            heads = patch.heads.to(act.device)  # type: ignore[union-attr]
            if patch.positions is None:
                n = values.shape[1]
                act[rows, :n, heads] = values  # [B, pos, d_head]
            else:
                act[rows, patch.positions.to(act.device), heads] = values  # [B, d_head]
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


class TransformerLensBackend(ModelBackend):
    def __init__(self, bridge: Any, info: ModelInfo) -> None:
        super().__init__()
        self.bridge = bridge
        self.info = info
        self.tokenizer = bridge.tokenizer
        self.bos_token_id = own_bos_token_id(bridge)
        self._logits_to_keep = self._supports_logits_to_keep()

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
        device = torch.device(str(cfg.device)).type
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
        if n_params is None:
            try:
                n_params = int(bridge.n_params_total)
            except Exception:  # noqa: BLE001
                n_params = None
        model_type = getattr(bridge.original_model.config, "model_type", None)
        structure = checks.structure
        if structure == "sequential" and model_type == "gpt2":
            structure = "sequential_pre_norm"  # GPT-2's layout, normalization included, is known
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
                "n_key_value_heads": int(getattr(cfg, "n_key_value_heads", None) or cfg.n_heads),
                "bos": own_bos_token_id(bridge) is not None,
                # TransformerLens marks "no soft-cap" with a value of zero or below.
                "logit_soft_cap": _soft_cap(cfg),
                "checks": checks.to_dict(),
            },
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

    def single_token_id(self, text: str) -> int | None:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return int(ids[0]) if len(ids) == 1 else None

    def token_str(self, token_id: int) -> str:
        return self.tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)

    # -- forward passes ----------------------------------------------------------------------

    def logits(self, tokens: torch.Tensor, patch: Patches = None, keep: int = 1) -> torch.Tensor:
        kwargs: dict[str, Any] = {"return_type": "logits"}
        if self._logits_to_keep:
            kwargs["logits_to_keep"] = keep
        with self.lock, torch.no_grad():
            bridge = self._bridge()
            tokens = tokens.to(self.device)
            hooks = _patch_hooks(patch)
            if not hooks:
                logits = bridge(tokens, **kwargs)
            else:
                logits = bridge.run_with_hooks(tokens, fwd_hooks=hooks, **kwargs)
            return logits[:, -keep:, :].float()

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

        # What each receiver reads, recorded in the first pass and patched in the second.
        final = hook_name("resid_post", n - 1)
        reads: dict[str, list[int]] = {}
        for kind, layer, head, part in receivers:
            if kind != "logits":
                reads.setdefault(f"blocks.{layer}.attn.hook_{part}", []).append(head)
        logits_receiver = any(kind == "logits" for kind, *_ in receivers)
        recorded: dict[str, torch.Tensor] = {}

        def record(name: str) -> Callable[..., torch.Tensor]:
            def fn(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
                recorded[name] = act.detach().clone()
                return act

            return fn

        def replay(name: str, heads: list[int]) -> Callable[..., torch.Tensor]:
            def fn(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
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
            last = max(layer for kind, layer, *_ in receivers if kind == "head")
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
    ) -> dict[str, torch.Tensor]:
        n = self.info.n_layers
        kept: dict[str, torch.Tensor] = {}

        def keep(name: str) -> Callable[..., torch.Tensor]:
            def fn(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
                kept[name] = act[:, -1].detach()  # only the last position is read out
                return act

            return fn

        names = [hook_name("resid_pre", 0), hook_name("resid_post", n - 1)]
        names += [hook_name(k, layer) for layer in range(n) for k in ("attn_out", "mlp_out")]
        if heads:
            names += [hook_name("head", layer) for layer in range(n)]
        with self.lock:
            bridge = self._bridge()
            with torch.no_grad():
                bridge.run_with_hooks(
                    tokens.to(self.device),
                    fwd_hooks=[(name, keep(name)) for name in names],
                    return_type=None,
                )

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
                with bridge.hooks(fwd_hooks=[("ln_final.hook_scale", hold)]):
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
        total = out["embed"] + out["attn_out"].sum(1) + out["mlp_out"].sum(1)
        out["remainder"] = out["logit_diff"] - total
        return {k: v.cpu() for k, v in out.items()}

    def layer_logits(self, tokens: torch.Tensor, position: int, row: int) -> torch.Tensor:
        if self.info.extra.get("prediction_method") != "final_norm_logit_lens":
            raise BackendError(
                "Per-layer predictions need the model's final normalization and unembedding to "
                "reproduce its output, and for this model they didn't when it loaded. The other "
                "analyses remain available."
            )
        if not 0 <= position < tokens.shape[1] or not 0 <= row < tokens.shape[0]:
            raise ValueError("The prediction token position or batch row is out of range.")
        with self.lock, torch.no_grad():
            bridge = self._bridge()
            captured: dict[int, torch.Tensor] = {}

            def project(layer: int) -> Callable[..., torch.Tensor]:
                def keep(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
                    # Recompute final normalization at each layer. Cached final-layer scales
                    # would implement attribution, not the logit lens. Keep the original
                    # batch shape and dtype through both modules, including learned biases.
                    residual = act[:, position : position + 1, :]
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
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.cuda.current_device())
    if device == "mps":
        return "Apple GPU (Metal)"
    from logogram.system import cpu_name

    return cpu_name()


def resolve_device(device: str) -> str:
    if device == "auto":
        # Apple's MPS only when asked for: TransformerLens reports that it can give silently wrong
        # results, and Logogram hasn't been checked on it yet (see system.MPS_NOTE).
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise BackendError(
            "The spec asks for CUDA, but PyTorch can't see a CUDA GPU here. Run `logogram doctor` "
            "for the fix, or set model.device to auto or cpu."
        )
    if device == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise BackendError(
            "The spec asks for Apple's MPS, which isn't available here. Set model.device to auto "
            "or cpu."
        )
    return device


def load_model(
    model_id: str,
    *,
    revision: str | None = None,
    dtype: str = "float32",
    device: str = "auto",
    process_weights: bool = True,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    cancel: threading.Event | None = None,
) -> TransformerLensBackend:
    """Resolve, download (with progress) and load a model from the Hugging Face Hub.

    ``cancel`` stops a download promptly and a load at the next step; it raises ``Cancelled``.
    """
    from logogram.backends import hub

    def emit(stage: str, **data: Any) -> None:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        if on_progress:
            on_progress({"stage": stage, **data})

    if dtype not in DTYPES:
        raise BackendError(f"Unknown dtype {dtype!r}; use float32, float16 or bfloat16.")
    resolved_device = resolve_device(device)
    emit("resolving", model_id=model_id)
    repo = hub.resolve(model_id, revision)
    emit("downloading", done=0, total=sum(s for _, s in repo.files), file="")

    def on_download(done: int, total: int, file: str) -> None:
        emit("downloading", done=done, total=total, file=file)

    folder = hub.download(model_id, repo, on_download, cancel=cancel)
    emit("loading", revision=repo.revision)

    def boot_and_check() -> TransformerLensBackend:
        bridge = boot_local(folder, device=resolved_device, dtype=dtype)
        emit("processing")
        return TransformerLensBackend.from_bridge(
            bridge,
            model_id=model_id,
            revision=repo.revision,
            dtype=dtype,
            process_weights=process_weights,
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


def out_of_memory_message(device: str, process_weights: bool) -> str:
    where = {"cuda": "GPU memory", "mps": "unified memory"}.get(device, "memory")
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
            bridge = TransformerBridge.boot_transformers(
                str(folder),
                device=device,
                dtype=DTYPES[dtype],
                hf_model=model,
                tokenizer=tokenizer,
                trust_remote_code=False,
            )
            bridge.__dict__[_OWN_BOS] = own_bos
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
