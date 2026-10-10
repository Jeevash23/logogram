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


def integrated_gradients(
    backend: ModelBackend,
    scorer: Scorer,
    receiver_tokens: torch.Tensor,
    source_tokens: torch.Tensor,
    rows: list[int],
    sites: list[tuple[str, int]],
    steps: int,
) -> Grads:
    """The metric's gradients at ``sites``, averaged over ``steps`` runs whose input embeddings
    lie between the receiver's and the source's (at the midpoints of equal intervals)."""
    key = ("resid_pre", 0)
    start = backend.capture(receiver_tokens, [key])[key]
    end = backend.capture(source_tokens, [key])[key]
    total: Grads | None = None
    for k in range(steps):
        alpha = (k + 0.5) / steps
        embeddings = start + alpha * (end - start)
        _, grads = metric_gradients(backend, scorer, receiver_tokens, rows, sites, embeddings)
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
            source_acts = backend.capture(source_tokens[sl], needed)
            if exp.method == "integrated_gradients":
                assert exp.steps is not None
                receiver_acts = backend.capture(receiver_tokens[sl], needed)
                grads = integrated_gradients(
                    backend, scorer, receiver_tokens[sl], source_tokens[sl], idx, needed, exp.steps
                )
            else:
                receiver_acts, grads = metric_gradients(
                    backend, scorer, receiver_tokens[sl], idx, needed
                )
            # Each layer's products once, then every site reads its own part of them.
            products = {
                key: (
                    (float64(source_acts[key]) - float64(receiver_acts[key])) * float64(grads[key])
                ).sum(-1)  # [B, pos] or [B, pos, heads]
                for key in needed
            }
            rows = torch.arange(len(idx))
            for rs in sites:
                per_position = products[(rs.kind, rs.layer)]
                if rs.kind == "head":
                    per_position = per_position[:, :, rs.head]
                if isinstance(rs.site.position, AllPositions):
                    values = per_position.sum(1)
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
