"""Direct logit attribution: what each component writes straight into the logit difference.

At the last position the residual stream is the embeddings plus every attention and MLP output.
With the final normalization's scale held at its value in the run, the logit difference is an
affine function of that sum, so it splits into one term per component plus a constant from biases.
A component's term is its *direct* effect: it leaves out everything the component does through
later components, which patching measures. The backend computes the terms; this module checks that
the split is valid for the loaded model and the chosen sites, and turns the terms into a result.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from logogram.backends.base import Cancelled, ModelBackend, ModelInfo
from logogram.engine import (
    Baselines,
    EngineError,
    EngineResult,
    LayerFn,
    ProgressFn,
    PromptsFn,
    _chunks,
    answer_tensors,
    behavior_warnings,
    check_finite,
    compute_baselines,
    make_scorer,
)
from logogram.prompts import PreparedPrompt, group_by_length
from logogram.sites import ResolvedSite, ScopeError, expand_scope, resolve_position
from logogram.spec import AllPositions, DirectLogitAttribution, Spec

DIRECT_KINDS = ("head", "attn_out", "mlp_out", "neuron", "v")
# Read from the residual stream, not written into it: they act on the logits through the softmax
# of their head's attention, so they have no direct effect.
NO_DIRECT_EFFECT = ("q", "k")

# The spacing of representable numbers near 1 in each dtype: a logit L is rounded to about L times
# this, so two computations of a logit difference agree to a few of these steps.
PRECISION = {"float32": 2.0**-23, "float16": 2.0**-10, "bfloat16": 2.0**-7}


def check_direct_sites(
    sites: list[ResolvedSite], info: ModelInfo, prompts: list[PreparedPrompt]
) -> None:
    """Refuse sites, or models, for which a direct effect isn't defined."""
    for prompt in prompts:
        if not (prompt.answer.single and prompt.distractor.single):
            raise ScopeError(
                f"Prompt {prompt.index} has an answer or distractor that isn't a single token. "
                "Direct logit attribution splits the logit difference between two tokens, which "
                "is linear in the residual stream; sets and continuations aren't. Use single-token "
                "answers, or activation patching."
            )
    extra = info.extra
    if not str(extra.get("block_structure", "")).startswith(("sequential", "parallel")):
        raise ScopeError(
            "When this model loaded, its residual stream couldn't be verified to be the sum of its "
            "attention and MLP outputs, so the logit difference can't be split among them."
        )
    if extra.get("logit_soft_cap"):
        raise ScopeError(
            "This model soft-caps its logits (tanh), so the logit difference isn't a sum of "
            "direct effects. Use activation patching for this model."
        )
    for rs in sites:
        if rs.kind in NO_DIRECT_EFFECT:
            what = "query" if rs.kind == "q" else "key"
            raise ScopeError(
                f"{rs.label} is a head's {what}: it is read from the residual stream, not written "
                "into it, and acts on the logits only through the softmax of the head's "
                "attention, so it has no direct effect. Use the head's output or its values, or "
                f"patch the {what}."
            )
        if rs.kind not in DIRECT_KINDS:
            raise ScopeError(
                f"{rs.label} is a state of the residual stream, not something written into it. "
                "Direct logit attribution splits the logit difference among heads, attention "
                "outputs and MLP outputs; choose those."
            )
    checks = extra.get("checks") or {}

    def measured(name: str) -> bool:
        value = checks.get(name)
        return value is not None and value <= checks.get("tolerance", 0)

    if any(rs.kind in ("head", "v") for rs in sites) and not measured("heads"):
        raise ScopeError(
            "In this model the attention output isn't the sum of its heads' outputs (it is "
            "normalized after they are combined), so a single head has no direct effect of its "
            "own. Use attention and MLP outputs per layer."
        )
    if any(rs.kind == "v" for rs in sites) and not measured("values"):
        raise ScopeError(
            "When this model loaded, its heads' outputs weren't their attention patterns times "
            "the values they read, so a value has no direct effect of its own here. Use the "
            "heads' outputs, or patch the values."
        )
    if any(rs.kind == "neuron" for rs in sites) and not measured("neuron_writes"):
        raise ScopeError(
            "In this model the MLP output isn't the sum of what its neurons write (it is "
            "normalized after the MLP, as in Gemma 2 and OLMo 2), so a single neuron has no "
            "direct effect of its own. Use MLP outputs per layer, or patch the neurons."
        )
    checked: set[str] = set()
    for rs in sites:
        if rs.kind == "v":
            continue  # a value at any position reaches the last token through the attention
        key = rs.site.position.model_dump_json()
        if key in checked:
            continue
        checked.add(key)
        for prompt in prompts:
            if resolve_position(rs.site.position, prompt) != prompt.length - 1:
                raise ScopeError(
                    "Direct effects are read at the last token, where the logit difference is "
                    f"measured, but {rs.label} is at another position in prompt {prompt.index}. "
                    "Set the position to the last token."
                )


def run_direct_effects(
    spec: Spec,
    backend: ModelBackend,
    prompts: list[PreparedPrompt],
    *,
    on_progress: ProgressFn | None = None,
    on_layer: LayerFn | None = None,
    cancel: threading.Event | None = None,
    on_start: Callable[[list[ResolvedSite], dict[str, Any]], None] | None = None,
    on_prompts: PromptsFn | None = None,
) -> EngineResult:
    """Split every prompt's logit difference, batch by batch of prompts. Each finished batch is
    reported through ``on_prompts``, so a run that stops early keeps every site's term over the
    prompts that finished."""
    exp = spec.experiment
    assert isinstance(exp, DirectLogitAttribution)
    if spec.metric.kind != "logit_diff":
        raise ScopeError(
            "Direct logit attribution splits the logit difference, the one metric that is a sum "
            "of what each component writes. Choose the logit difference as the metric."
        )
    info = backend.info
    batch_size = spec.execution.batch_size
    sites, layout = expand_scope(spec, info, prompts)
    check_direct_sites(sites, info, prompts)
    if on_start is not None:
        on_start(sites, layout)
    groups = group_by_length(prompts)
    baselines = compute_baselines(
        backend, prompts, groups, batch_size, cancel, make_scorer(spec, prompts)
    )
    check_finite(baselines, info.dtype)
    which = exp.prompts
    n, n_layers = len(prompts), info.n_layers
    heads = any(rs.kind == "head" for rs in sites)
    # Neurons and values are read only in the layers that have them.
    neuron_layers = sorted({rs.layer for rs in sites if rs.kind == "neuron"})
    value_layers = sorted({rs.layer for rs in sites if rs.kind == "v"})
    unit_sites = [rs for rs in sites if rs.kind in ("neuron", "v")]

    terms: dict[str, np.ndarray] = {
        "embed": np.zeros(n),
        "attn_out": np.zeros((n, n_layers)),
        "mlp_out": np.zeros((n, n_layers)),
        "logit_diff": np.zeros(n),
        "remainder": np.zeros(n),
        "logit_scale": np.zeros(n),
    }
    if heads:
        terms["head"] = np.zeros((n, n_layers, info.n_heads))
    if unit_sites:
        # Each neuron's or value's term, per prompt: [n, sites].
        terms["units"] = np.zeros((n, len(sites)))
    done = 0
    finished: list[int] = []

    def over(at: list[int]) -> EngineResult:
        """Every site's term over the prompts at positions ``at``, as a run of only those prompts
        gives it."""
        chosen = {key: values[at] for key, values in terms.items()}
        return _direct_result(
            spec, info, sites, layout, [prompts[i] for i in at], baselines.subset(at), chosen
        )

    for group in groups:
        tokens = group.clean if which == "clean" else group.corrupt
        for sl in _chunks(len(group.members), batch_size):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            idx = group.members[sl]
            out = backend.direct_effects(
                tokens[sl],
                *answer_tensors(prompts, idx),
                heads,
                neurons=neuron_layers,
                values=value_layers,
            )
            for key, values in out.items():
                if key not in ("neuron", "value"):
                    terms[key][idx] = values.numpy()
            for rs in unit_sites:
                terms["units"][idx, rs.index] = _unit_terms(
                    rs, out, neuron_layers, value_layers, [prompts[i] for i in idx]
                )
            done += len(idx)
            finished.extend(idx)
            if on_progress is not None:
                on_progress(done, n, n_layers - 1)
            if on_prompts is not None:
                at = sorted(finished)
                on_prompts(at, lambda at=at: over(at))

    result = _direct_result(spec, info, sites, layout, prompts, baselines, terms)
    if on_layer is not None:
        for layer in sorted({rs.layer for rs in sites}):
            on_layer(layer, [rs.index for rs in sites if rs.layer == layer], result)
    return result


def _unit_terms(
    rs: ResolvedSite,
    out: dict[str, Any],
    neuron_layers: list[int],
    value_layers: list[int],
    prompts: list[PreparedPrompt],
) -> np.ndarray:
    """A neuron's or a value's term for a batch of prompts, from the backend's terms: a neuron at
    the last token; a key/value head's value at its position (summed over every position for all
    positions), through every query head that reads it."""
    if rs.kind == "neuron":
        return out["neuron"][:, neuron_layers.index(rs.layer), rs.site.neuron].numpy()
    each = out["value"][:, value_layers.index(rs.layer), rs.head]  # [B, pos]
    if isinstance(rs.site.position, AllPositions):
        return each.sum(-1).numpy()
    positions = [resolve_position(rs.site.position, p) for p in prompts]
    return each[torch.arange(len(prompts)), torch.tensor(positions)].numpy()


def _direct_result(
    spec: Spec,
    info: ModelInfo,
    sites: list[ResolvedSite],
    layout: dict[str, Any],
    prompts: list[PreparedPrompt],
    baselines: Baselines,
    terms: dict[str, np.ndarray],
) -> EngineResult:
    """The result of splitting ``prompts``' logit differences into ``terms`` (per prompt)."""
    exp = spec.experiment
    assert isinstance(exp, DirectLogitAttribution)
    which = exp.prompts
    n = len(prompts)
    gap = baselines.metric(which)
    warnings: list[str] = behavior_warnings(baselines)
    mean_gap = float(gap.mean())
    if spec.metric.normalization == "dataset_gap" and abs(mean_gap) < 1e-3:
        raise EngineError(
            f"The {which} prompts' mean logit difference is almost zero ({mean_gap:.4f}), so a "
            "share of it is undefined. Normalize by each prompt's own logit difference, or check "
            "the baseline."
        )
    if spec.metric.normalization == "prompt_gap":
        zero = [p.index for p, g in zip(prompts, gap, strict=True) if abs(g) < 1e-6]
        if zero:
            raise EngineError(
                f"Prompt(s) {', '.join(map(str, zero[:5]))} have a logit difference of zero, so a "
                "share of it is undefined. Normalize by the dataset's mean, or fix those prompts."
            )
        small = int((np.abs(gap) < 0.1).sum())
        if small:
            warnings.append(
                f"{small} prompt(s) have a logit difference below 0.1, so their per-prompt shares "
                "are unstable."
            )
    # The split is of the model's own logit difference; the two ways of reading it out must agree,
    # up to how finely the model's dtype resolves the two logits.
    drift = np.abs(terms["logit_diff"] - gap)
    allowed = (
        1e-3 * max(1.0, float(np.abs(gap).max())) + 4 * PRECISION[info.dtype] * terms["logit_scale"]
    )
    if n and bool((drift > allowed).any()):
        warnings.append(
            f"The decomposed logit difference differs from the measured one by up to "
            f"{float(drift.max()):.2g}, more than rounding explains."
        )

    delta = np.zeros((len(sites), n))
    for rs in sites:
        if rs.kind == "head":
            delta[rs.index] = terms["head"][:, rs.layer, rs.head]
        elif rs.kind in ("neuron", "v"):
            delta[rs.index] = terms["units"][:, rs.index]
        else:
            delta[rs.index] = terms[rs.kind][:, rs.layer]
    nan = np.full((len(sites), n), np.nan)
    return EngineResult(
        sites=sites,
        layout=layout,
        prompts=prompts,
        baselines=baselines,
        receiver=which,
        reference=which,
        patched=nan,
        patched_prob=nan.copy(),
        patched_pref=nan.copy(),
        warnings=warnings,
        measure="attribution",
        delta=delta,
        gap=gap,
        extra={
            "direct": {
                "prompts": which,
                "logit_diff": float(terms["logit_diff"].mean()),
                "embeddings": float(terms["embed"].mean()),
                "attention": float(terms["attn_out"].sum(1).mean()),
                "mlp": float(terms["mlp_out"].sum(1).mean()),
                "biases": float(terms["remainder"].mean()),
            }
        },
    )
