"""Automated circuit discovery (ACDC, as in Conmy et al. 2023): a circuit found by removing the
edges of the model's graph one at a time.

The graph
---------
The graph is edge attribution's (logogram.edges): its nodes are the embeddings, every attention
head, every MLP and the logits, and an edge runs from a node's output to a later node's input (a
head's query, key or value, an MLP's input, or the logits), each receiver reading its own copy of
the residual stream. The ``graph`` scope chooses the candidate edges: those into the heads and MLPs
of its layers, and into the logits. Every other edge of the model stays in the circuit.

The order
---------
ACDC tests the candidate edges in the reverse of the graph's order: the receivers from the logits
back toward the embeddings (the logits; then each layer from the last: its MLP's input, then its
heads from the last, each head's value, key and query), and each receiver's edges from its latest
sender back to the embeddings. Where heads share keys and values (grouped-query attention), a
layer's heads come as its key/value heads from the last, each one's value and key, then the
queries of its group's heads from the last. A receiver's output reaches only receivers that come
after it in the graph's order, so each receiver is tested once every receiver its output reaches
has been: whether it matters to the circuit is settled by then (a reverse topological order).

The rule
--------
At first every edge is in the circuit, and the receiver prompt runs as it is. For each edge in
turn, the receiver prompt runs once more, edge-patched (logogram.edges): every removed edge, and
this one, carries its sender's output on the source prompt into its receiver, and every other edge
carries what its sender outputs in that pass. The change is the mean of the metric over the
prompts minus the current circuit's. With ``rule="toward_source"``, the change counted is how far
it moves toward the source run, s · change, where s is the sign of the mean of (source − receiver)
in the metric; a removal that moves the metric away from the source counts as negative, so that
edge always stays removed (the paper's rule: running the clean prompt, with the KL divergence from
its own prediction, KL(new) − KL(current) < threshold). With ``rule="either_way"``, the change
counted is its size, |change|. When the change counted is below ``threshold``, the edge stays
removed and the circuit's metric is the new one; otherwise the edge is put back.

Each tested edge costs one forward pass per batch of prompts, in batches of fixed composition (the
prompts grouped by length, in order), so the same spec gives the same circuit, bit for bit. The
senders' outputs on the source prompts are captured once and kept for every pass.

What a run records
------------------
Every candidate edge is a site. Its per-prompt values are the change its removal made when it was
tested: the metric with the edge removed minus the circuit's at that moment, normalized like
patching (1 would be a removal that moves the metric as far as the whole gap from the receiver run
to the source run). The summary records which edges were kept (the circuit), and the circuit's
per-prompt metric beside the receiver run's (every edge kept) and the source run's: its
faithfulness, normalized as the faithfulness curve normalizes with every other edge patched (0 is
the source run, 1 the receiver run), with the run's bootstrap. The circuit's metric is exactly
what edge patching gives with the removed edges patched: the masks are laid out as
``PatchPlan.of`` lays them out for those edges in the graph's order.

A run that stops early keeps nothing: until every edge has been tested, the edges left aren't
ACDC's circuit.
"""

from __future__ import annotations

import math
import threading
import time
import weakref
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from logogram.backends.base import Cancelled, ModelBackend, ModelInfo
from logogram.edges import (
    Block,
    Graph,
    PatchPlan,
    _faithfulness,
    _finite,
    _places,
    build_graph,
    edge_sites,
    free_memory,
    patched_forward,
    sender_keys,
)
from logogram.engine import (
    EngineError,
    EngineResult,
    LayerFn,
    ProgressFn,
    _chunks,
    _keep_on_cpu,
    check_finite,
    check_gap,
    check_values,
    compute_baselines,
    make_scorer,
)
from logogram.prompts import LengthGroup, PreparedPrompt, UnevenPairs, group_by_length
from logogram.sites import ResolvedSite, ScopeError
from logogram.spec import ACDC, METRIC_LABELS, GraphScope, Spec


def order_text(grouped: bool = False) -> str:
    """The order ACDC tests the edges in, in words. ``grouped``: the model's heads share keys and
    values (grouped-query attention), so a layer's receivers are each key/value head's value and
    key, then the queries of its group's heads."""
    heads = (
        "its key/value heads from the last, each one's value and key, then the queries of its "
        "group's heads from the last"
        if grouped
        else "its heads from the last, each head's value, key and query"
    )
    return (
        "the reverse of the graph's order: the receivers from the logits back toward the "
        f"embeddings (the logits; then each layer from the last: its MLP's input, then {heads}), "
        "and each receiver's edges from its latest sender back to the embeddings"
    )


ORDER_TEXT = order_text()

Pair = tuple[Block, Block]


# -- what a run costs ----------------------------------------------------------------------------


def acdc_cost(
    graph: Graph, info: ModelInfo, groups: list[LengthGroup], batch_size: int, kl: bool
) -> dict[str, Any]:
    """Passes and memory: the passes per batch of prompts and for the whole run, the memory the
    source prompts' senders take (kept for every pass), and the largest batch's."""
    batches = sum(math.ceil(len(g.members) / batch_size) for g in groups)
    length = max(g.length for g in groups)
    rows = min(batch_size, max(len(g.members) for g in groups))
    n = sum(len(g.members) for g in groups)
    E = len(graph)
    size = {"float32": 4, "float16": 2, "bfloat16": 2}.get(info.dtype, 4)
    # The embeddings, and each sending layer's heads' outputs z and MLP output.
    per_token = info.d_model + len(graph.sender_layers) * (
        info.n_heads * info.d_head + info.d_model
    )
    # The unpatched baselines (each side, and the KL divergence's target first), the senders
    # captured on the source prompts, then the circuit with every edge kept and one pass per edge.
    unpatched = 3 + (1 if kl else 0)
    return {
        "edges": E,
        "batches": batches,
        "forward_per_batch": unpatched + 1 + E,
        "backward_per_batch": 0,
        "patched_per_batch": 1 + E,
        "forward": (unpatched + 1 + E) * batches,
        "backward": 0,
        # A batch's senders on the source prompt, and its differences in float32, as a pass holds
        # them beside the model's own activations.
        "memory_per_batch": int(2 * per_token * size * length * rows),
        "largest_batch": rows,
        "cache": int(per_token * size * length * n),
        "values": E * n,
        "seconds": None,
    }


def check_acdc_cost(cost: dict[str, Any], device: torch.device) -> None:
    """Refuse a run that won't fit: too many edges or per-prompt values to keep (edge
    attribution's limits), or the source prompts' senders too large for the memory free here."""
    from logogram import edges

    if cost["edges"] > edges.MAX_EDGES:
        raise EngineError(
            f"This graph has {cost['edges']:,} edges. Logogram keeps every edge's per-prompt values "
            f"and statistics, and more than {edges.MAX_EDGES:,} don't fit. Choose fewer layers."
        )
    if cost["values"] > edges.MAX_VALUES:
        raise EngineError(
            f"{cost['edges']:,} edges over these prompts are {cost['values']:,} per-prompt values, "
            f"more than the {edges.MAX_VALUES:,} a run keeps. Use fewer prompts (dataset.limit) or "
            "fewer layers."
        )
    # On a GPU they move to CPU memory when large (see run_acdc), so the system's memory is what
    # holds them at most.
    need = int(cost["cache"])
    free = edges.free_memory(torch.device("cpu") if device.type == "cuda" else device)
    if free is not None and need > 0.8 * free:
        raise EngineError(
            f"The source prompts' senders, kept for every pass, need about {need / 1e9:.1f} GB "
            f"here, and {free / 1e9:.1f} GB is free. Use fewer prompts (dataset.limit)."
        )


def duration_text(seconds: float) -> str:
    """A duration, roughly, in words: "about 40 seconds", "about 3 minutes", "about 2.5 hours"."""
    if seconds < 90:
        return f"about {max(1, round(seconds))} second{'s' if round(seconds) > 1 else ''}"
    if seconds < 90 * 60:
        return f"about {round(seconds / 60)} minutes"
    hours = seconds / 3600
    return f"about {hours:.1f} hours" if hours < 10 else f"about {round(hours)} hours"


def acdc_cost_text(cost: dict[str, Any]) -> str:
    """What a run costs, in words, for the summary and the command line."""
    batches = cost["batches"]
    parts = [
        f"{cost['edges']:,} edges tested one at a time",
        f"{batches} batch{'es' if batches != 1 else ''}",
        f"{cost['patched_per_batch']:,} edge-patched passes per batch",
        f"{cost['forward']:,} forward passes in all",
    ]
    if cost.get("seconds") is not None:
        parts.append(f"{duration_text(cost['seconds'])} here")
    return ", ".join(parts)


# Seconds an edge-patched pass takes, measured once per model, graph and batch shape.
_TIMINGS: weakref.WeakKeyDictionary[ModelBackend, dict[tuple[Any, ...], float]] = (
    weakref.WeakKeyDictionary()
)


def time_pass(
    backend: ModelBackend,
    graph: Graph,
    groups: list[LengthGroup],
    batch_size: int,
    receiver: str,
    source: str,
) -> float:
    """Seconds one edge-patched pass of the largest batch takes here, with every edge of the
    scope removed, as most of a run's passes nearly are: the second of two, the first warming up.
    Measured once for each graph and batch shape."""
    group = max(groups, key=lambda g: (min(len(g.members), batch_size) * g.length, g.length))
    rows = min(len(group.members), batch_size)
    key = (len(graph), tuple(graph.receiver_layers), graph.logits, rows, group.length)
    known = _TIMINGS.setdefault(backend, {})
    if key not in known:
        keys = sender_keys(graph)
        weights = {layer: backend.head_output_weights(layer) for layer in graph.sender_layers}
        src = backend.capture(group.tokens(source)[:rows], keys)
        plan = PatchPlan.of(graph, np.arange(len(graph)), backend.device)
        forward = patched_forward(backend, plan, src, weights)
        tokens = group.tokens(receiver)[:rows]
        forward(tokens, 1).cpu()
        start = time.perf_counter()
        forward(tokens, 1).cpu()  # on the CPU: the pass has finished, wherever it ran
        known[key] = time.perf_counter() - start
    return known[key]


def acdc_plan(spec: Spec, backend: ModelBackend, prompts: list[PreparedPrompt]) -> dict[str, Any]:
    """What an ACDC spec would do on the loaded model and these prompts, before it runs: the
    graph's size, the passes the run takes and how long they may take here (one pass timed), and
    why it would be refused (None when it wouldn't), from the same checks the run makes."""
    from logogram.backends.base import BackendError

    exp, scope = spec.experiment, spec.scope
    assert isinstance(exp, ACDC) and isinstance(scope, GraphScope)
    refusal: str | None = None
    try:
        backend.edge_checks()
    except BackendError as exc:
        refusal = str(exc)
    try:
        graph = build_graph(backend.info, scope)
    except ScopeError as exc:
        return {"refusal": str(exc), "n_edges": None, "n_senders": None, "n_receivers": None}
    out: dict[str, Any] = {
        "n_edges": len(graph),
        "n_senders": len(graph.senders),
        "n_receivers": len(graph.receivers),
        "structure": "parallel" if graph.parallel else "sequential",
    }
    if not prompts:
        return {**out, "refusal": refusal or "Choose prompts first."}
    try:
        groups = group_by_length(prompts)
    except UnevenPairs as exc:
        return {**out, "refusal": str(exc)}
    batch_size = spec.execution.batch_size
    cost = acdc_cost(graph, backend.info, groups, batch_size, spec.metric.kind == "kl")
    if refusal is None:
        try:
            check_acdc_cost(cost, backend.device)
        except EngineError as exc:
            refusal = str(exc)
    if refusal is None:
        receiver, source = _sides(exp)
        with backend.lock:
            seconds = time_pass(backend, graph, groups, batch_size, receiver, source)
        cost["seconds_per_pass"] = seconds
        cost["seconds"] = seconds * cost["forward"]
    return {
        **out,
        "cost": cost,
        "text": acdc_cost_text(cost),
        "free": free_memory(backend.device),
        "refusal": refusal,
    }


def _sides(exp: ACDC) -> tuple[str, str]:
    """The prompt that runs (the receiver) and the prompt removed edges carry (the source)."""
    return ("corrupt", "clean") if exp.direction == "clean_to_corrupt" else ("clean", "corrupt")


# -- the edges removed ---------------------------------------------------------------------------


class _Removed:
    """The edges removed so far, as edge patching's masks: a 0/1 matrix per (sending block,
    receiving block) over the blocks' places.

    ``plan`` lays the masks out as ``PatchPlan.of`` does for the removed edges in the graph's
    order: each pair in the order of its first removed edge, so a pass sums each receiver's terms
    in the same order, and the circuit's metric is what edge patching that set gives, bit for
    bit. Edges are removed in the reverse of the graph's order, so each removal is its pair's
    first edge in that order."""

    def __init__(self, graph: Graph, device: torch.device) -> None:
        self.graph = graph
        self.device = device
        self.masks: dict[Pair, torch.Tensor] = {}
        self.count: dict[Pair, int] = {}
        self.first: dict[Pair, int] = {}

    def _place(self, e: int) -> tuple[Pair, int, int]:
        graph = self.graph
        sender = graph.senders[int(graph.edge_sender[e])]
        receiver = graph.receivers[int(graph.edge_receiver[e])]
        (sblock, s), (rblock, r) = sender.block(), receiver.block()
        return (sblock, rblock), s, r

    def plan(self, trying: int | None = None) -> PatchPlan:
        """The removed edges, and ``trying`` with them, as a patch plan."""
        first = {p: i for p, i in self.first.items() if self.count.get(p)}
        if trying is not None:
            pair, s, r = self._place(trying)
            if pair not in self.masks:
                g = self.graph
                self.masks[pair] = torch.zeros(
                    (_places(g, pair[0]), _places(g, pair[1])), device=self.device
                )
            self.masks[pair][s, r] = 1.0
            first[pair] = trying
        order = sorted(first, key=lambda p: first[p])
        return PatchPlan({p: self.masks[p] for p in order})

    def keep_out(self, e: int) -> None:
        """The edge tried stays removed."""
        pair, _, _ = self._place(e)
        self.count[pair] = self.count.get(pair, 0) + 1
        self.first[pair] = e

    def put_back(self, e: int) -> None:
        """The edge tried is put back in the circuit."""
        pair, s, r = self._place(e)
        self.masks[pair][s, r] = 0.0


# -- the run -------------------------------------------------------------------------------------


def run_acdc(
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
    """Test every candidate edge of the scope in turn, removing those whose removal changes the
    metric by less than the threshold (see the module's docstring). Nothing is reported before
    the end: a run that stops early keeps nothing."""
    exp = spec.experiment
    scope = spec.scope
    assert isinstance(exp, ACDC) and isinstance(scope, GraphScope)
    info = backend.info
    batch_size = spec.execution.batch_size
    checks = backend.edge_checks()
    graph = build_graph(info, scope)
    scorer = make_scorer(spec, prompts)
    if not scorer.single_position():
        raise ScopeError(
            "ACDC reads the metric at the last prompt position, so answers and distractors must "
            "be single tokens or sets of single tokens."
        )
    groups = group_by_length(prompts)
    cost = acdc_cost(graph, info, groups, batch_size, spec.metric.kind == "kl")
    check_acdc_cost(cost, backend.device)
    sites, layout = edge_sites(graph)
    if on_start is not None:
        on_start(sites, layout)
    baselines = compute_baselines(backend, prompts, groups, batch_size, cancel, scorer)
    check_finite(baselines, info.dtype)
    receiver, source = _sides(exp)
    reference = source
    receiver = receiver_override or receiver
    source = source_override or source
    warnings = check_gap(spec, baselines, prompts, receiver, reference)
    gap = float(np.mean(baselines.metric(source) - baselines.metric(receiver)))
    if exp.rule == "toward_source" and not abs(gap) >= 1e-3:
        what = METRIC_LABELS[spec.metric.kind]
        raise EngineError(
            f"The {receiver} and {source} prompts give almost the same mean {what} "
            f"({gap:.4f}), so a change toward the {source} prompts has no direction. Count changes "
            "either way (rule either_way), or run the baseline check: the model may not show the "
            "behavior on these prompts."
        )
    sign = 1 if gap > 0 else -1

    n = len(prompts)
    E = len(graph)
    keys = sender_keys(graph)
    weights = {layer: backend.head_output_weights(layer) for layer in graph.sender_layers}
    # The senders' outputs on the source prompts, captured once for every pass. On a GPU they
    # are kept in CPU memory when they would take more than a quarter of its free memory, and
    # each batch's are moved back for its passes (exact copies).
    offload = _keep_on_cpu(backend.device, int(cost["cache"]))
    batches: list[tuple[list[int], torch.Tensor, dict[tuple[str, int], torch.Tensor]]] = []
    for group in groups:
        receiver_tokens = group.tokens(receiver)
        source_tokens = group.tokens(source)
        for sl in _chunks(len(group.members), batch_size):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            captured = backend.capture(source_tokens[sl], keys)
            if offload:
                captured = {k: v.to("cpu") for k, v in captured.items()}
            batches.append((group.members[sl], receiver_tokens[sl], captured))
    total = (1 + E) * len(batches)
    done = 0

    def evaluate(plan: PatchPlan, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The metric, P(answer) and the preference of every prompt, with ``plan``'s edges
        removed."""
        nonlocal done
        metric, prob, pref = np.zeros(n), np.zeros(n), np.zeros(n)
        for idx, tokens, captured in batches:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            src = {k: v.to(backend.device) for k, v in captured.items()} if offload else captured
            scores = scorer.score(patched_forward(backend, plan, src, weights), tokens, idx)
            metric[idx], prob[idx], pref[idx] = scores.metric, scores.prob, scores.pref
            done += 1
            if on_progress is not None:
                on_progress(done, total, layer)
        return metric, prob, pref

    last = info.n_layers - 1
    removed = _Removed(graph, backend.device)
    current, _, _ = evaluate(removed.plan(), last)
    full = current.copy()
    current_mean = float(current.mean())
    patched = np.zeros((E, n))
    patched_prob = np.zeros((E, n))
    patched_pref = np.zeros((E, n))
    delta = np.zeros((E, n))
    kept = np.ones(E, dtype=bool)
    for e in range(E - 1, -1, -1):  # the reverse of the graph's order
        layer = min(graph.receivers[int(graph.edge_receiver[e])].layer, last)
        metric, prob, pref = evaluate(removed.plan(e), layer)
        mean = float(metric.mean())
        change = mean - current_mean
        counted = sign * change if exp.rule == "toward_source" else abs(change)
        patched[e], patched_prob[e], patched_pref[e] = metric, prob, pref
        delta[e] = metric - current
        if counted < exp.threshold:  # a change that isn't a finite number keeps the edge
            removed.keep_out(e)
            kept[e] = False
            current, current_mean = metric, mean
        else:
            removed.put_back(e)

    circuit = [int(e) for e in np.flatnonzero(kept)]
    extra: dict[str, Any] = {
        "threshold": exp.threshold,
        "rule": exp.rule,
        "source_sign": sign,
        "order": order_text(graph.n_kv_heads != graph.n_heads),
        "n_edges": E,
        "n_kept": len(circuit),
        "n_senders": len(graph.senders),
        "n_receivers": len(graph.receivers),
        "reported": E,
        "structure": "parallel" if graph.parallel else "sequential",
        "checks": checks,
        "cost": {**cost, "text": acdc_cost_text(cost)},
        "kept": circuit,
        "full": _finite(full.mean()),
        "circuit": _finite(current.mean()),
        "per_prompt": {
            "circuit": [_finite(v) for v in current],
            "full": [_finite(v) for v in full],
        },
    }
    warnings.extend(check_values(patched, "metrics with an edge removed", info.dtype))
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
        delta=delta,
        extra={"acdc": extra},
    )
    extra["faithfulness"] = circuit_faithfulness(spec, result, current, receiver, source)
    return result


def circuit_faithfulness(
    spec: Spec, result: EngineResult, circuit: np.ndarray, receiver: str, source: str
) -> dict[str, float | None]:
    """The circuit's faithfulness, from its per-prompt metric: 0 is the source run, 1 the
    receiver run (every edge kept), normalized as the faithfulness curve normalizes with every
    edge but the chosen patched, with a percentile bootstrap interval from the run's resamples."""
    from logogram.results import bootstrap_counts

    low, high = result.baselines.metric(source), result.baselines.metric(receiver)
    counts = bootstrap_counts(spec, result)
    return _faithfulness(spec, circuit[None, :], low, high, counts)[0]
