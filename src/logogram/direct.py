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

from logogram.backends.base import Cancelled, ModelBackend, ModelInfo
from logogram.engine import (
    EngineError,
    EngineResult,
    LayerFn,
    ProgressFn,
    _answer_tensors,
    _chunks,
    behavior_warnings,
    compute_baselines,
)
from logogram.prompts import PreparedPrompt, group_by_length
from logogram.sites import ResolvedSite, ScopeError, expand_scope, resolve_position
from logogram.spec import DirectLogitAttribution, Spec

DIRECT_KINDS = ("head", "attn_out", "mlp_out")


def check_direct_sites(
    sites: list[ResolvedSite], info: ModelInfo, prompts: list[PreparedPrompt]
) -> None:
    """Refuse sites, or models, for which a direct effect isn't defined."""
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
        if rs.kind not in DIRECT_KINDS:
            raise ScopeError(
                f"{rs.label} is a state of the residual stream, not something written into it. "
                "Direct logit attribution splits the logit difference among heads, attention "
                "outputs and MLP outputs; choose those."
            )
    if any(rs.kind == "head" for rs in sites):
        checks = extra.get("checks") or {}
        heads = checks.get("heads")
        if heads is None or heads > checks.get("tolerance", 0):
            raise ScopeError(
                "In this model the attention output isn't the sum of its heads' outputs (it is "
                "normalized after they are combined), so a single head has no direct effect of its "
                "own. Use attention and MLP outputs per layer."
            )
    checked: set[str] = set()
    for rs in sites:
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
) -> EngineResult:
    exp = spec.experiment
    assert isinstance(exp, DirectLogitAttribution)
    info = backend.info
    batch_size = spec.execution.batch_size
    sites, layout = expand_scope(spec, info, prompts)
    check_direct_sites(sites, info, prompts)
    if on_start is not None:
        on_start(sites, layout)
    groups = group_by_length(prompts)
    baselines = compute_baselines(backend, prompts, groups, batch_size, cancel)
    which = exp.prompts
    n, n_layers = len(prompts), info.n_layers
    heads = any(rs.kind == "head" for rs in sites)

    terms: dict[str, np.ndarray] = {
        "embed": np.zeros(n),
        "attn_out": np.zeros((n, n_layers)),
        "mlp_out": np.zeros((n, n_layers)),
        "logit_diff": np.zeros(n),
        "remainder": np.zeros(n),
    }
    if heads:
        terms["head"] = np.zeros((n, n_layers, info.n_heads))
    done = 0
    for group in groups:
        tokens = group.clean if which == "clean" else group.corrupt
        for sl in _chunks(len(group.members), batch_size):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            idx = group.members[sl]
            out = backend.direct_effects(tokens[sl], *_answer_tensors(prompts, idx), heads)
            for key, values in out.items():
                terms[key][idx] = values.numpy()
            done += len(idx)
            if on_progress is not None:
                on_progress(done, n, n_layers - 1)

    gap = baselines.ld(which)
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
    # The split is of the model's own logit difference; the two ways of reading it out must agree.
    drift = float(np.abs(terms["logit_diff"] - gap).max()) if n else 0.0
    if drift > 1e-3 * max(1.0, float(np.abs(gap).max())):
        warnings.append(
            f"The decomposed logit difference differs from the measured one by up to {drift:.2g}, "
            "more than rounding explains."
        )

    delta = np.zeros((len(sites), n))
    for rs in sites:
        if rs.kind == "head":
            delta[rs.index] = terms["head"][:, rs.layer, rs.head]
        else:
            delta[rs.index] = terms[rs.kind][:, rs.layer]
    nan = np.full((len(sites), n), np.nan)
    result = EngineResult(
        sites=sites,
        layout=layout,
        prompts=prompts,
        baselines=baselines,
        receiver=which,
        reference=which,
        patched_ld=nan,
        patched_prob=nan.copy(),
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
    if on_layer is not None:
        for layer in sorted({rs.layer for rs in sites}):
            on_layer(layer, [rs.index for rs in sites if rs.layer == layer], result)
    return result
