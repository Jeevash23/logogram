"""Attribution patching: estimate activation patching at every site from gradients.

Patching a site changes the metric by f(receiver with the source activation) - f(receiver). To
first order that is (source activation - receiver activation) · the gradient of the metric with
respect to the activation, at the receiver run. One forward pass on the source prompts and one
forward and backward pass on the receiver prompts give the estimate for every site at once, so a
sweep costs a few passes per batch instead of one patched run per site and prompt.

The estimate is only first order. It misses saturation (in attention patterns, normalization and
the final softmax) and can miss or even invert an effect, which is why the results say "estimated"
and offer to verify the strongest sites with real patching.

Integrated gradients (as in EAP-IG) average the gradient over runs whose input embeddings lie
evenly between the receiver's and the source's, at the midpoints of ``steps`` equal intervals. The
average follows the metric along the way from one prompt to the other instead of only at its
start, which corrects much of what a single gradient misses, at ``steps`` times the cost.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from logogram.backends.base import Cancelled, ModelBackend, float64
from logogram.engine import (
    EngineResult,
    LayerFn,
    ProgressFn,
    _chunks,
    check_finite,
    check_gap,
    check_values,
    compute_baselines,
    make_scorer,
)
from logogram.metrics import Scorer, extend_tokens
from logogram.prompts import PreparedPrompt, group_by_length
from logogram.sites import ResolvedSite, expand_scope, resolve_position
from logogram.spec import AllPositions, AttributionPatching, Spec

RESIDUAL = ("resid_pre", "resid_mid", "resid_post")

Grads = dict[tuple[str, int], torch.Tensor]


def metric_gradients(
    backend: ModelBackend,
    scorer: Scorer,
    tokens: torch.Tensor,
    rows: list[int],
    sites: list[tuple[str, int]],
    embeddings: torch.Tensor | None = None,
) -> tuple[Grads, Grads]:
    """Activations and the metric's gradients at ``sites``, over the prompts' positions only.

    Answers or distractors of several tokens need a forward pass with each continuation
    appended; the metric is a sum of what each pass contributes, so their gradients add up. The
    prompt's activations are the same in every pass (a causal model never reads ahead)."""
    length = tokens.shape[1]
    acts: Grads | None = None
    total: Grads | None = None
    for part, conts, keep in scorer.passes(rows):

        def score(logits: torch.Tensor, part: str = part) -> torch.Tensor:
            return scorer.differentiable(logits, rows, part)

        kept, grads = backend.gradients(
            extend_tokens(tokens, conts, keep), sites, score, keep, embeddings
        )
        if acts is None:
            acts = {k: v[:, :length] for k, v in kept.items()}
        sliced = {k: v[:, :length] for k, v in grads.items()}
        total = sliced if total is None else {k: total[k] + sliced[k] for k in total}
    assert acts is not None and total is not None
    return acts, total


def pass_products(
    backend: ModelBackend,
    scorer: Scorer,
    receiver_tokens: torch.Tensor,
    source_tokens: torch.Tensor,
    rows: list[int],
    sites: list[tuple[str, int]],
    steps: int | None,
) -> tuple[dict[tuple[str, int], torch.Tensor], dict[tuple[str, int], torch.Tensor] | None]:
    """(source - receiver activation) · gradient at every position of ``sites``: at the prompt's
    positions ``[B, L]`` (``[B, L, heads]`` for heads), and summed over the positions of tokens
    appended to read continuations (None when nothing was appended).

    Each forward pass the metric needs (one per continuation) contributes its own products: the
    source prompt runs with the same continuation appended, so the appended positions are compared
    like for like. With ``steps``, the gradient is averaged over runs whose input embeddings lie
    between the receiver's and the source's (integrated gradients)."""
    length = receiver_tokens.shape[1]
    prompt: dict[tuple[str, int], torch.Tensor] = {}
    tail: dict[tuple[str, int], torch.Tensor] | None = None
    for part, conts, keep in scorer.passes(rows):

        def score(logits: torch.Tensor, part: str = part) -> torch.Tensor:
            return scorer.differentiable(logits, rows, part)

        receiver_ext = extend_tokens(receiver_tokens, conts, keep)
        source_ext = extend_tokens(source_tokens, conts, keep)
        source_acts = backend.capture(source_ext, sites)
        if steps is None:
            receiver_acts, grads = backend.gradients(receiver_ext, sites, score, keep)
        else:
            receiver_acts = backend.capture(receiver_ext, sites)
            grads = _integrated(backend, receiver_ext, source_ext, sites, score, keep, steps)
        for key in sites:
            product = (
                (float64(source_acts[key]) - float64(receiver_acts[key])) * float64(grads[key])
            ).sum(-1)  # [B, pos] or [B, pos, heads]
            here = product[:, :length]
            prompt[key] = here if key not in prompt else prompt[key] + here
            if product.shape[1] > length:
                rest = product[:, length:].sum(1)
                tail = {} if tail is None else tail
                tail[key] = rest if key not in tail else tail[key] + rest
    return prompt, tail


def _integrated(
    backend: ModelBackend,
    receiver_tokens: torch.Tensor,
    source_tokens: torch.Tensor,
    sites: list[tuple[str, int]],
    score: Callable[[torch.Tensor], torch.Tensor],
    keep: int,
    steps: int,
) -> dict[tuple[str, int], torch.Tensor]:
    """The gradients at ``sites``, averaged over ``steps`` runs whose input embeddings lie between
    the receiver's and the source's, at the midpoints of equal intervals."""
    key = ("resid_pre", 0)
    start = backend.capture(receiver_tokens, [key])[key]
    end = backend.capture(source_tokens, [key])[key]
    total: dict[tuple[str, int], torch.Tensor] | None = None
    for k in range(steps):
        alpha = (k + 0.5) / steps
        _, grads = backend.gradients(
            receiver_tokens, sites, score, keep, start + alpha * (end - start)
        )
        grads = {s: float64(g) for s, g in grads.items()}
        total = grads if total is None else {s: total[s] + grads[s] for s in total}
    assert total is not None
    return {s: g / steps for s, g in total.items()}


def run_attribution_patching(
    spec: Spec,
    backend: ModelBackend,
    prompts: list[PreparedPrompt],
    *,
    on_progress: ProgressFn | None = None,
    on_layer: LayerFn | None = None,
    cancel: threading.Event | None = None,
    on_start: Callable[[list[ResolvedSite], dict[str, Any]], None] | None = None,
    receiver_override: str | None = None,
    source_override: str | None = None,
) -> EngineResult:
    exp = spec.experiment
    assert isinstance(exp, AttributionPatching)
    info = backend.info
    batch_size = spec.execution.batch_size
    sites, layout = expand_scope(spec, info, prompts)
    if on_start is not None:
        on_start(sites, layout)
    groups = group_by_length(prompts)
    scorer = make_scorer(spec, prompts)
    baselines = compute_baselines(backend, prompts, groups, batch_size, cancel, scorer)
    check_finite(baselines, info.dtype)
    receiver, source = (
        ("corrupt", "clean") if exp.direction == "clean_to_corrupt" else ("clean", "corrupt")
    )
    reference = source
    receiver = receiver_override or receiver
    source = source_override or source
    warnings = check_gap(spec, baselines, prompts, receiver, reference)

    n = len(prompts)
    needed = list(dict.fromkeys((rs.kind, rs.layer) for rs in sites))
    estimate = np.zeros((len(sites), n))
    done = 0
    for group in groups:
        receiver_tokens = group.clean if receiver == "clean" else group.corrupt
        source_tokens = group.clean if source == "clean" else group.corrupt
        for sl in _chunks(len(group.members), batch_size):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            idx = group.members[sl]
            # Each layer's products once, then every site reads its own part of them.
            products, tail = pass_products(
                backend,
                scorer,
                receiver_tokens[sl],
                source_tokens[sl],
                idx,
                needed,
                exp.steps if exp.method == "integrated_gradients" else None,
            )
            rows = torch.arange(len(idx))
            for rs in sites:
                key = (rs.kind, rs.layer)
                per_position = products[key]
                if rs.kind == "head":
                    per_position = per_position[:, :, rs.head]
                if isinstance(rs.site.position, AllPositions):
                    values = per_position.sum(1)
                    if tail is not None:  # every position includes the appended ones
                        values = values + (
                            tail[key][:, rs.head] if rs.kind == "head" else tail[key]
                        )
                else:
                    positions = torch.tensor(
                        [resolve_position(rs.site.position, prompts[p]) for p in idx]
                    )
                    values = per_position[rows, positions]
                estimate[rs.index, idx] = values.cpu().numpy()
            done += len(idx)
            if on_progress is not None:
                on_progress(done, n, info.n_layers - 1)

    if any(rs.kind in RESIDUAL for rs in sites):
        warnings.append(
            "Estimates at residual stream sites can miss effects where patching replaces a whole "
            "token's representation: in GPT-2 small's IOI example, patching the residual stream at "
            "the changed name in the first layer restores the whole answer, while its estimate is "
            "slightly negative. Verifying the top estimates can't catch such a miss; Check "
            "robustness patches every site."
        )
    warnings.extend(check_values(estimate, "estimates", info.dtype))
    nan = np.full((len(sites), n), np.nan)
    result = EngineResult(
        sites=sites,
        layout=layout,
        prompts=prompts,
        baselines=baselines,
        receiver=receiver,
        reference=reference,
        # The metric patching would give, to first order. Sign flips and probabilities aren't
        # estimated: they are left undefined rather than read off an estimate.
        patched=baselines.metric(receiver)[None, :] + estimate,
        patched_prob=nan,
        patched_pref=nan.copy(),
        warnings=warnings,
        measure="estimate",
        delta=estimate,
        extra={"attribution": {"method": exp.method, "steps": exp.steps}},
    )
    if on_layer is not None:
        for layer in sorted({rs.layer for rs in sites}):
            on_layer(layer, [rs.index for rs in sites if rs.layer == layer], result)
    return result
