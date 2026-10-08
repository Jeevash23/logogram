"""Path patching: a component's effect through chosen receivers only.

For each sender (a head, attention output or MLP output of the scope) and receiver prompt:

1. run the receiver prompt with the sender's activation from the source prompt, and every other
   attention head held at its own value in the receiver run (and the MLPs too, if frozen), so the
   change reaches later layers only through the residual stream (and the MLPs); record what the
   receivers read: a later head's query, key or value, or the final residual stream;
2. run the receiver prompt again with only those receiver inputs replaced by the recorded ones,
   and read the metric.

The normalized effect is computed as for patching. A sender with no receiver after it has no path,
so a sweep keeps only the layers before the last receiver.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from logogram.backends.base import Cancelled, ModelBackend, ModelInfo
from logogram.engine import (
    EngineResult,
    LayerFn,
    ProgressFn,
    _answer_tensors,
    _build_patch,
    _chunks,
    _LayerSources,
    _metric,
    _Row,
    check_gap,
    compute_baselines,
)
from logogram.prompts import PreparedPrompt, group_by_length
from logogram.sites import ResolvedSite, ScopeError, expand_scope
from logogram.spec import AllPositions, HeadReceiver, PathPatching, Spec

SENDER_KINDS = ("head", "attn_out", "mlp_out")


def check_receivers(exp: PathPatching, info: ModelInfo) -> None:
    for receiver in exp.receivers:
        if not isinstance(receiver, HeadReceiver):
            continue
        if receiver.layer >= info.n_layers or receiver.head >= info.n_heads:
            raise ScopeError(
                f"L{receiver.layer} H{receiver.head} doesn't exist; this model has "
                f"{info.n_layers} layers of {info.n_heads} heads."
            )
        kv = int(info.extra.get("n_key_value_heads") or info.n_heads)
        if receiver.input in ("k", "v") and kv != info.n_heads:
            raise ScopeError(
                f"This model shares each key and value among {info.n_heads // kv} heads, so a "
                "single head's key or value can't receive a path on its own. Use query receivers."
            )


def path_sites(
    spec: Spec, info: ModelInfo, prompts: list[PreparedPrompt]
) -> tuple[list[ResolvedSite], dict[str, Any]]:
    """The scope's senders that come before a receiver, re-indexed, and their layout."""
    exp = spec.experiment
    assert isinstance(exp, PathPatching)
    sites, layout = expand_scope(spec, info, prompts)
    for rs in sites:
        if rs.kind not in SENDER_KINDS:
            raise ScopeError(
                f"{rs.label} is a residual stream state. Path patching sends from heads, "
                "attention outputs or MLP outputs; choose those."
            )
    heads = [r.layer for r in exp.receivers if isinstance(r, HeadReceiver)]
    limit = info.n_layers if len(heads) < len(exp.receivers) else max(heads)
    kept = [rs for rs in sites if rs.layer < limit]
    if not kept:
        raise ScopeError(
            "No sender comes before a receiver, so there is no path to patch. Choose receivers in "
            "later layers, or add the logits."
        )
    if layout["kind"] in ("heads", "layer_components", "layer_position"):
        layout = {**layout, "rows": layout["rows"][:limit]}
    elif layout["kind"] == "sites":
        layout = {**layout, "rows": [layout["rows"][rs.row] for rs in kept]}
    out = []
    for i, rs in enumerate(kept):
        row = i if layout["kind"] == "sites" else rs.row
        out.append(ResolvedSite(i, rs.site, row, rs.col, rs.label))
    return out, layout


def run_path_patching(
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
    assert isinstance(exp, PathPatching)
    info = backend.info
    batch_size = spec.execution.batch_size
    check_receivers(exp, info)
    sites, layout = path_sites(spec, info, prompts)
    if on_start is not None:
        on_start(sites, layout)
    groups = group_by_length(prompts)
    baselines = compute_baselines(backend, prompts, groups, batch_size, cancel)
    receiver, source = (
        ("corrupt", "clean") if exp.direction == "clean_to_corrupt" else ("clean", "corrupt")
    )
    reference = source
    warnings = check_gap(spec, baselines, prompts, receiver, reference)
    receivers = [
        ("head", r.layer, r.head, r.input)
        if isinstance(r, HeadReceiver)
        else ("logits", -1, -1, "")
        for r in exp.receivers
    ]
    group_of = {i: gi for gi, g in enumerate(groups) for i in g.members}
    local_of = {i: li for g in groups for li, i in enumerate(g.members)}
    n = len(prompts)
    n_layers = info.n_layers
    model_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = model_dtype[info.dtype]
    patched_ld = np.zeros((len(sites), n))
    patched_prob = np.zeros((len(sites), n))
    result = EngineResult(
        sites=sites,
        layout=layout,
        prompts=prompts,
        baselines=baselines,
        receiver=receiver,
        reference=reference,
        patched_ld=patched_ld,
        patched_prob=patched_prob,
        warnings=warnings,
    )
    held = [("head", layer) for layer in range(n_layers)]
    if exp.freeze_mlps:
        held += [("mlp_out", layer) for layer in range(n_layers)]
    total = len(sites) * n
    done = 0
    for layer in sorted({rs.layer for rs in sites}):
        layer_sites = [rs for rs in sites if rs.layer == layer]
        kinds = list(dict.fromkeys(rs.kind for rs in layer_sites))
        sources = _LayerSources(backend, prompts, groups, layer, kinds, source, batch_size, cancel)
        for gi, group in enumerate(groups):
            tokens = group.clean if receiver == "clean" else group.corrupt
            buckets: dict[tuple[str, bool], list[_Row]] = {}
            for rs in layer_sites:
                all_pos = isinstance(rs.site.position, AllPositions)
                for p in group.members:
                    buckets.setdefault((rs.kind, all_pos), []).append(_Row(rs, p, local_of[p]))
            for (kind, all_pos), rows in buckets.items():
                for sl in _chunks(len(rows), batch_size):
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
                    chunk = rows[sl]
                    sender = _build_patch(
                        chunk,
                        kind,
                        layer,
                        all_pos,
                        "patch",
                        sources,
                        gi,
                        group.length,
                        group_of,
                        local_of,
                        prompts,
                        info.d_model,
                        info.d_head,
                        dtype,
                        backend.device,
                    )
                    local = torch.tensor([r.local for r in chunk], dtype=torch.long)
                    # The receiver run's own head (and MLP) outputs, which the first pass holds.
                    own = backend.capture(tokens[local], held)
                    frozen_heads = {i: own[("head", i)] for i in range(n_layers)}
                    frozen_mlps = (
                        {i: own[("mlp_out", i)] for i in range(n_layers)}
                        if exp.freeze_mlps
                        else None
                    )
                    logits = backend.path_patch(
                        tokens[local], sender, frozen_heads, frozen_mlps, receivers
                    )
                    ld, prob = _metric(logits, *_answer_tensors(prompts, [r.prompt for r in chunk]))
                    for r, value_ld, value_prob in zip(chunk, ld, prob, strict=True):
                        patched_ld[r.site.index, r.prompt] = value_ld
                        patched_prob[r.site.index, r.prompt] = value_prob
                    done += len(chunk)
                    if on_progress is not None:
                        on_progress(done, total, layer)
        if on_layer is not None:
            on_layer(layer, [rs.index for rs in layer_sites], result)
    return result
