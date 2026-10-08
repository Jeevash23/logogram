"""Attribution patching: estimate activation patching at every site from one gradient.

Patching a site changes the logit difference by f(receiver with the source activation) - f(receiver).
To first order that is (source activation - receiver activation) · the gradient of the logit
difference with respect to the activation, at the receiver run. One forward pass on the source
prompts and one forward and backward pass on the receiver prompts give the estimate for every site
at once, so a sweep costs a few passes per batch instead of one patched run per site and prompt.

The estimate is only first order. It misses saturation (in attention patterns, normalization and
the final softmax) and can miss or even invert an effect, which is why the results say "estimated"
and offer to verify the strongest sites with real patching.
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
    _answer_tensors,
    _chunks,
    check_gap,
    compute_baselines,
)
from logogram.prompts import PreparedPrompt, group_by_length
from logogram.sites import ResolvedSite, expand_scope, resolve_position
from logogram.spec import AllPositions, AttributionPatching, Spec


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
    baselines = compute_baselines(backend, prompts, groups, batch_size, cancel)
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
            receiver_acts, grads = backend.gradients(
                receiver_tokens[sl], *_answer_tensors(prompts, idx), needed
            )
            rows = torch.arange(len(idx))
            for rs in sites:
                key = (rs.kind, rs.layer)
                diff = float64(source_acts[key]) - float64(receiver_acts[key])
                grad = float64(grads[key])
                if rs.kind == "head":
                    diff, grad = diff[:, :, rs.head], grad[:, :, rs.head]
                per_position = (diff * grad).sum(-1)  # [B, pos]
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

    receiver_ld = baselines.ld(receiver)
    result = EngineResult(
        sites=sites,
        layout=layout,
        prompts=prompts,
        baselines=baselines,
        receiver=receiver,
        reference=reference,
        # The logit difference patching would give, to first order. Probabilities aren't estimated.
        patched_ld=receiver_ld[None, :] + estimate,
        patched_prob=np.full((len(sites), n), np.nan),
        warnings=warnings,
        measure="estimate",
        delta=estimate,
    )
    if on_layer is not None:
        for layer in sorted({rs.layer for rs in sites}):
            on_layer(layer, [rs.index for rs in sites if rs.layer == layer], result)
    return result
