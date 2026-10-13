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

Between prompts of different token lengths, the sender is patched at a label or the last token:
from the source prompt's token there into the receiver's. The rest of both passes runs on the
receiver prompt alone.

With grouped-query attention, a key or value receiver is a key/value head ("L5 KV1 k"): what it
reads is recorded and patched for that key/value head, so the path reaches every query head of its
group through their shared key or value.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import torch

from logogram.backends.base import Cancelled, ModelBackend, ModelInfo, Patch
from logogram.engine import (
    EngineResult,
    LayerFn,
    ProgressFn,
    _build_patch,
    _chunks,
    _LayerSources,
    _Row,
    check_finite,
    check_gap,
    check_values,
    compute_baselines,
    empty_values,
    make_scorer,
)
from logogram.prompts import PreparedPrompt, group_by_length
from logogram.sites import ResolvedSite, ScopeError, check_lined_up, expand_scope, kv_heads
from logogram.spec import (
    ALL_POSITIONS,
    AllPositions,
    Edge,
    EdgesScope,
    HeadReceiver,
    LogitsReceiver,
    MlpReceiver,
    PathPatching,
    Site,
    Spec,
    edge_text,
    receiver_text,
    sender_text,
)

SENDER_KINDS = ("head", "attn_out", "mlp_out")


def check_receivers(exp: PathPatching, info: ModelInfo) -> None:
    for receiver in exp.receivers:
        if isinstance(receiver, MlpReceiver):
            if receiver.layer >= info.n_layers:
                raise ScopeError(
                    f"L{receiver.layer} MLP doesn't exist; this model has {info.n_layers} layers."
                )
            if "mlp_out" not in info.site_kinds:
                raise ScopeError("This model has no MLPs to receive a path.")
            continue
        if not isinstance(receiver, HeadReceiver):
            continue
        if receiver.layer >= info.n_layers or receiver.head >= info.n_heads:
            raise ScopeError(
                f"L{receiver.layer} H{receiver.head} doesn't exist; this model has "
                f"{info.n_layers} layers of {info.n_heads} heads."
            )
        kv = kv_heads(info)
        if receiver.input in ("k", "v") and receiver.head >= kv:
            group = info.n_heads // kv
            raise ScopeError(
                f"L{receiver.layer} KV{receiver.head} doesn't exist: this model shares each key and "
                f"value among {group} heads, so its layers have {kv} key/value heads (0 to "
                f"{kv - 1}), and a key or value receiver names one of them (key/value head 0 is "
                f"read by query heads 0 to {group - 1})."
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
            what = (
                "a residual stream state"
                if rs.kind.startswith("resid")
                else "a part of a head or an MLP, not a component that writes into the residual "
                "stream"
            )
            raise ScopeError(
                f"{rs.label} is {what}. Path patching sends from heads' outputs, attention "
                "outputs or MLP outputs; choose those."
            )
    # A sender reaches a head in a later layer, and an MLP in its own layer or a later one (a
    # layer's MLP reads its heads where the layer adds attention first: checked when it runs).
    reach = [
        r.layer if isinstance(r, HeadReceiver) else r.layer + 1
        for r in exp.receivers
        if not isinstance(r, LogitsReceiver)
    ]
    limit = info.n_layers if len(reach) < len(exp.receivers) else max(reach)
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
    if isinstance(spec.scope, EdgesScope):
        return run_edge_paths(
            spec,
            backend,
            prompts,
            on_progress=on_progress,
            on_layer=on_layer,
            cancel=cancel,
            on_start=on_start,
        )
    info = backend.info
    batch_size = spec.execution.batch_size
    check_receivers(exp, info)
    sites, layout = path_sites(spec, info, prompts)
    # Between prompts of different lengths, the sender is patched at a label or the last token.
    check_lined_up([rs.site.position for rs in sites], prompts, "path")
    if on_start is not None:
        on_start(sites, layout)
    receiver, source = (
        ("corrupt", "clean") if exp.direction == "clean_to_corrupt" else ("clean", "corrupt")
    )
    # The receivers run in batches of their own length.
    groups = group_by_length(prompts, receiver)
    scorer = make_scorer(spec, prompts)
    if not scorer.single_position():
        raise ScopeError(
            "Path patching reads the metric at the last prompt position, so answers and "
            "distractors must be single tokens or sets of single tokens. Use activation patching "
            "for continuations of several tokens."
        )
    baselines = compute_baselines(backend, prompts, groups, batch_size, cancel, scorer)
    check_finite(baselines, info.dtype)
    reference = source
    warnings = check_gap(spec, baselines, prompts, receiver, reference)
    receivers = [
        ("head", r.layer, r.head, r.input)
        if isinstance(r, HeadReceiver)
        else ("mlp", r.layer, -1, "")
        if isinstance(r, MlpReceiver)
        else ("logits", -1, -1, "")
        for r in exp.receivers
    ]
    group_of = {i: gi for gi, g in enumerate(groups) for i in g.members}
    local_of = {i: li for g in groups for li, i in enumerate(g.members)}
    n = len(prompts)
    n_layers = info.n_layers
    model_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = model_dtype[info.dtype]
    patched, patched_prob, patched_pref = empty_values(len(sites), n)
    result = EngineResult(
        sites=sites,
        layout=layout,
        prompts=prompts,
        baselines=baselines,
        receiver=receiver,
        reference=reference,
        patched=patched,
        patched_prob=patched_prob,
        patched_pref=patched_pref,
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
            tokens = group.tokens(receiver)
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
                        receiver=receiver,
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

                    def forward(
                        toks: torch.Tensor,
                        keep: int,
                        sender: Any = sender,
                        frozen_heads: Any = frozen_heads,
                        frozen_mlps: Any = frozen_mlps,
                    ) -> torch.Tensor:
                        assert keep == 1  # checked above: everything is read at one position
                        logits = backend.path_patch(
                            toks, sender, frozen_heads, frozen_mlps, receivers
                        )
                        return logits[:, None, :]

                    scores = scorer.score(forward, tokens[local], [r.prompt for r in chunk])
                    for r, value, prob, pref in zip(
                        chunk, scores.metric, scores.prob, scores.pref, strict=True
                    ):
                        patched[r.site.index, r.prompt] = value
                        patched_prob[r.site.index, r.prompt] = prob
                        patched_pref[r.site.index, r.prompt] = pref
                    done += len(chunk)
                    if on_progress is not None:
                        on_progress(done, total, layer)
        if on_layer is not None:
            on_layer(layer, [rs.index for rs in layer_sites], result)
    warnings.extend(check_values(patched, "patched values", info.dtype))
    return result


# -- edges, one at a time ------------------------------------------------------------------------


def edge_rows(
    scope: EdgesScope, n_layers: int, shared: bool = False
) -> tuple[list[ResolvedSite], dict[str, Any]]:
    """One row per edge, in a senders × receivers layout of the senders and receivers it names.
    A row stands at its receiver's layer (the logits at the last), as an edge attribution run's
    rows do. ``shared``: the model's keys and values are key/value heads' (grouped-query
    attention), named "L5 KV1 k"."""
    senders = {e.sender.model_dump_json(): e.sender for e in scope.edges}
    receivers = {e.receiver.model_dump_json(): e.receiver for e in scope.edges}
    sender_keys, receiver_keys = list(senders), list(receivers)
    sites = []
    for i, edge in enumerate(scope.edges):
        r = edge.receiver
        layer = n_layers - 1 if isinstance(r, LogitsReceiver) else r.layer
        sites.append(
            ResolvedSite(
                index=i,
                site=Site(kind="resid_pre", layer=layer, position=ALL_POSITIONS),
                row=sender_keys.index(edge.sender.model_dump_json()),
                col=receiver_keys.index(r.model_dump_json()),
                label=edge_text(edge, shared),
                edge=edge,
            )
        )
    layout = {
        "kind": "edges",
        "row_title": "Sender",
        "col_title": "Receiver",
        "rows": [
            {"key": str(i), "label": sender_text(senders[k]), "node": senders[k].kind}
            for i, k in enumerate(sender_keys)
        ],
        "cols": [
            {"key": str(i), "label": receiver_text(receivers[k], shared), "node": receivers[k].kind}
            for i, k in enumerate(receiver_keys)
        ],
    }
    return sites, layout


def _check_edge(edge: Edge, info: ModelInfo) -> None:
    s, r = edge.sender, edge.receiver
    layers = [s.layer, None if isinstance(r, LogitsReceiver) else r.layer]
    if any(layer is not None and layer >= info.n_layers for layer in layers):
        raise ScopeError(
            f"{edge_text(edge)} names a layer this model doesn't have; it has {info.n_layers}."
        )
    shared = kv_heads(info) != info.n_heads
    if s.head is not None and s.head >= info.n_heads:
        raise ScopeError(f"{edge_text(edge, shared)} names a head this model doesn't have.")
    parallel = str(info.extra.get("block_structure")) == "parallel"
    if parallel and isinstance(r, MlpReceiver) and s.kind == "head" and s.layer == r.layer:
        raise ScopeError(
            "In this model a layer's MLP reads the residual stream its heads read, not what they "
            f"write, so {edge_text(edge, shared)} isn't an edge."
        )


def run_edge_paths(
    spec: Spec,
    backend: ModelBackend,
    prompts: list[PreparedPrompt],
    *,
    on_progress: ProgressFn | None = None,
    on_layer: LayerFn | None = None,
    cancel: threading.Event | None = None,
    on_start: Callable[[list[ResolvedSite], dict[str, Any]], None] | None = None,
) -> EngineResult:
    """Path patching of each edge alone: the receiver prompt runs with the edge's sender taking
    its source value and every head and MLP held at the receiver run's own, so the change reaches
    the edge's receiver only directly; what the receiver reads there is recorded and patched into
    an otherwise unchanged receiver run, where the metric is read. Each edge is a row, reported
    once every prompt has run at it, so a run that stops early keeps the edges that finished."""
    exp = spec.experiment
    scope = spec.scope
    assert isinstance(exp, PathPatching) and isinstance(scope, EdgesScope)
    info = backend.info
    batch_size = spec.execution.batch_size
    check_receivers(exp, info)
    for edge in scope.edges:
        _check_edge(edge, info)
    sites, layout = edge_rows(scope, info.n_layers, kv_heads(info) != info.n_heads)
    if on_start is not None:
        on_start(sites, layout)
    groups = group_by_length(prompts)
    scorer = make_scorer(spec, prompts)
    if not scorer.single_position():
        raise ScopeError(
            "Path patching reads the metric at the last prompt position, so answers and "
            "distractors must be single tokens or sets of single tokens. Use activation patching "
            "for continuations of several tokens."
        )
    baselines = compute_baselines(backend, prompts, groups, batch_size, cancel, scorer)
    check_finite(baselines, info.dtype)
    receiver, source = (
        ("corrupt", "clean") if exp.direction == "clean_to_corrupt" else ("clean", "corrupt")
    )
    warnings = check_gap(spec, baselines, prompts, receiver, source)
    n = len(prompts)
    patched, patched_prob, patched_pref = empty_values(len(sites), n)
    result = EngineResult(
        sites=sites,
        layout=layout,
        prompts=prompts,
        baselines=baselines,
        receiver=receiver,
        reference=source,
        patched=patched,
        patched_prob=patched_prob,
        patched_pref=patched_pref,
        warnings=warnings,
    )
    held = [(kind, layer) for layer in range(info.n_layers) for kind in ("head", "mlp_out")]
    total = len(sites) * n
    done = 0
    for rs in sites:
        edge = rs.edge
        assert edge is not None
        s, r = edge.sender, edge.receiver
        # The embeddings are the residual stream entering the first layer.
        kind = {"embed": "resid_pre", "head": "head", "mlp": "mlp_out"}[s.kind]
        key = (kind, s.layer or 0)
        target = (
            ("logits", -1, -1, "")
            if isinstance(r, LogitsReceiver)
            else ("mlp", r.layer, -1, "")
            if isinstance(r, MlpReceiver)
            else ("head", r.layer, r.head, r.input)
        )
        for group in groups:
            tokens = group.clean if receiver == "clean" else group.corrupt
            source_tokens = group.clean if source == "clean" else group.corrupt
            for sl in _chunks(len(group.members), batch_size):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                idx = group.members[sl]
                value = backend.capture(source_tokens[sl], [key])[key]
                if s.head is not None:
                    heads = torch.full((len(idx),), s.head, dtype=torch.long)
                    sender = Patch(
                        kind="head", layer=key[1], values=value[:, :, s.head], heads=heads
                    )
                else:
                    sender = Patch(kind=kind, layer=key[1], values=value)
                own = backend.capture(tokens[sl], held)
                frozen_heads = {i: own[("head", i)] for i in range(info.n_layers)}
                frozen_mlps = {i: own[("mlp_out", i)] for i in range(info.n_layers)}

                def forward(
                    toks: torch.Tensor,
                    keep: int,
                    sender: Patch = sender,
                    frozen_heads: Any = frozen_heads,
                    frozen_mlps: Any = frozen_mlps,
                    target: tuple[str, int, int, str] = target,
                ) -> torch.Tensor:
                    assert keep == 1  # checked above: everything is read at one position
                    logits = backend.path_patch(toks, sender, frozen_heads, frozen_mlps, [target])
                    return logits[:, None, :]

                scores = scorer.score(forward, tokens[sl], idx)
                patched[rs.index, idx] = scores.metric
                patched_prob[rs.index, idx] = scores.prob
                patched_pref[rs.index, idx] = scores.pref
                done += len(idx)
                if on_progress is not None:
                    on_progress(done, total, rs.layer)
        if on_layer is not None:
            on_layer(rs.layer, [rs.index], result)
    warnings.extend(check_values(patched, "patched values", info.dtype))
    return result
