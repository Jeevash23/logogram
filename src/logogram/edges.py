"""Edges: which component's output, read by which later component's input, carries the behavior.

The graph
---------
Nodes are the embeddings, every attention head (its output: z through its slice of the output
projection, without the projection's bias, which is the same on every prompt), every MLP and the
logits. Heads and MLPs write into the residual stream, which every later component reads; an edge
runs from a node's output to one later node's input:

* a head's query, key or value input (three edges from each earlier node, one per input);
* an MLP's input, from the embeddings, every earlier head and MLP, and, where a layer adds its
  attention before its MLP (GPT-2, Llama), the heads of its own layer; where both read the same
  residual stream (Pythia), not those;
* the logits, from every node.

With grouped-query attention (Llama, Qwen, Mistral, Gemma) a layer has fewer key/value heads than
query heads, and every query head of a group reads its key/value head's key and value. A key or
value input then belongs to the key/value head: an edge into "L5 KV1 k" reaches every query head of
group 1 through their shared key. Queries stay one per query head.

GPT-2 small (12 layers of 12 heads) has 32,491 edges. The graph relies on what the model was
measured to do (``ModelBackend.edge_checks``): its heads and MLPs add into the residual stream, and
each receiver can be given its own copy of the stream to read.

A receiver's own input
----------------------
In the model, a layer's heads all read one residual stream, normalized once. An edge's receiver v
reads its own copy: the residual stream x at v's layer, before the normalization, which v then
normalizes by itself, as TransformerLens's split inputs do for every head's query, key and value
(an MLP normalizes its input on its own, and the final normalization reads the logits' copy). The
residual stream itself runs on unchanged; only what v reads can differ.

The edge's estimate (EAP, as in Syed et al. 2023)
------------------------------------------------
Patching the edge u -> v means v reads x + (u's output on the source prompt − u's output on the
receiver prompt) in place of x. To first order the metric changes by

    estimate(u -> v) = Σ_positions Δu · ∂M/∂x_v,

the gradient of the metric M with respect to v's own copy x_v, at the receiver run, summed over
the positions (u's output at a position only reaches v's input at that position).

The gradient through the normalization. v reads n(x_v) = γ ⊙ (x_v − μ(x_v)) / σ(x_v) + β with
σ(x) = √(mean((x − μ)²) + ε) (RMSNorm leaves out the centering). The gradient is taken through n as
the model computes it, its scale σ recomputed from x_v, not held:

    ∂M/∂x_v = (1/σ) (I − 11ᵀ/d − x̂x̂ᵀ/d) (γ ⊙ ∂M/∂n),   x̂ = (x − μ)/σ.

Holding the scale, as direct logit attribution and steering's reading of the logits do, would
drop the x̂x̂ᵀ term: it linearizes a different function. Here the estimate is of a patch, and when
an edge is patched v's normalization recomputes σ from the patched copy, so the derivative of what
patching changes keeps the term. With it, the gradient is orthogonal to x_v (scaling the input
leaves the normalization's output as it was, up to ε): a sender that writes along the receiver's
own input is estimated to do little. The tests check this gradient against finite differences of
the metric with respect to each kind of copy, and that holding the scale would differ.

With ``method="integrated_gradients"`` (EAP-IG, as in Hanna et al. 2024) the gradient is averaged
over ``steps`` runs whose input embeddings lie between the receiver's and the source's, at the
midpoints of equal intervals, as attribution patching does; the sender differences stay those of
the two prompts.

Edge patching (for the faithfulness curve)
------------------------------------------
To patch a set of edges, the receiver prompt runs once, and each receiver's own input gets, for
each patched edge into it, the sender's difference: its output on the source prompt minus its
output in this pass, as this pass computes it. Every other edge carries what its sender outputs
in this pass. So:

    v reads x_v + Σ_{patched u -> v} (u_source − u_this pass).

Patching every edge of the graph makes every receiver read the source prompt's residual stream,
so the pass is the source run (to rounding); patching none is the receiver run. One edge patched
alone is path patching from u into v with every other head and MLP held, which the derived
verification spec measures independently (logogram.paths). The curve patches the k strongest
edges (``patch="chosen"``) or every edge but them (``patch="rest"``), beside edges chosen at
random, and reports faithfulness: the share of the gap from no edge chosen to every edge chosen.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from logogram.backends.base import (
    Cancelled,
    EdgePatcher,
    InputGradients,
    ModelBackend,
    ModelInfo,
    float64,
)
from logogram.engine import (
    Baselines,
    EngineError,
    EngineResult,
    LayerFn,
    ProgressFn,
    PromptsFn,
    _chunks,
    check_finite,
    check_gap,
    check_values,
    compute_baselines,
    make_scorer,
)
from logogram.metrics import Scorer
from logogram.prompts import LengthGroup, PreparedPrompt, group_by_length
from logogram.schema import EDGES_FILE
from logogram.sites import ResolvedSite, ScopeError, kv_heads
from logogram.spec import (
    ALL_POSITIONS,
    NAME_MAX,
    Edge,
    EdgeAttribution,
    EdgeSender,
    EdgesScope,
    GraphScope,
    HeadReceiver,
    LogitsReceiver,
    MlpReceiver,
    PathPatching,
    Site,
    Spec,
)

# Every edge keeps its per-prompt values and statistics: these bound what a run holds.
MAX_EDGES = 100_000
MAX_VALUES = 10_000_000  # edges × prompts
# The summary lists this many of the strongest edges; edges.parquet (EDGES_FILE) holds every edge.
SUMMARY_EDGES = 2_000
__all__ = ["EDGES_FILE"]
# How many of an estimate's strongest edges "Verify by path patching" measures, at most.
VERIFY_MAX = 50

GRADIENT_TEXT = (
    "taken with respect to each receiver's own copy of the residual stream, before its "
    "normalization, through that normalization with its scale recomputed from the copy (not "
    "held), as patching the edge would recompute it"
)

Block = tuple[str, int]  # a group of senders or receivers that share a tensor: (kind, layer)


# -- the graph -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    """A node whose output edges carry: the embeddings (layer -1), a head or an MLP."""

    kind: str
    layer: int
    head: int | None = None

    @property
    def label(self) -> str:
        if self.kind == "embed":
            return "embeddings"
        if self.kind == "head":
            return f"L{self.layer} H{self.head}"
        return f"L{self.layer} MLP"

    @property
    def key(self) -> str:
        return {"embed": "embed", "head": f"L{self.layer}H{self.head}"}.get(
            self.kind, f"L{self.layer}M"
        )

    def block(self) -> tuple[Block, int]:
        """The tensor the node's output is in, and its place there."""
        if self.kind == "head":
            assert self.head is not None
            return ("head", self.layer), self.head
        return (self.kind, self.layer), 0

    def spec(self) -> EdgeSender:
        if self.kind == "embed":
            return EdgeSender(kind="embed")
        if self.kind == "head":
            return EdgeSender(kind="head", layer=self.layer, head=self.head)
        return EdgeSender(kind="mlp", layer=self.layer)


@dataclass(frozen=True)
class Input:
    """A receiver: a head's query, key or value input, an MLP's input, or the logits (whose layer
    is the model's number of layers: they come after the last). A key or value input is a key/value
    head's: with grouped-query attention (``shared``) every query head of its group reads it, and
    it is named "L5 KV1 k"."""

    kind: str
    layer: int
    head: int | None = None
    shared: bool = False

    @property
    def unit(self) -> str:
        return "KV" if self.shared else "H"

    @property
    def label(self) -> str:
        if self.kind == "logits":
            return "logits"
        if self.kind == "mlp":
            return f"L{self.layer} MLP"
        return f"L{self.layer} {self.unit}{self.head} {self.kind}"

    @property
    def key(self) -> str:
        if self.kind == "logits":
            return "logits"
        if self.kind == "mlp":
            return f"L{self.layer}M"
        return f"L{self.layer}{self.unit}{self.head}{self.kind}"

    def block(self) -> tuple[Block, int]:
        if self.kind in ("q", "k", "v"):
            assert self.head is not None
            return (self.kind, self.layer), self.head
        return (self.kind, self.layer), 0

    def spec(self) -> HeadReceiver | MlpReceiver | LogitsReceiver:
        if self.kind == "logits":
            return LogitsReceiver(kind="logits")
        if self.kind == "mlp":
            return MlpReceiver(kind="mlp", layer=self.layer)
        assert self.head is not None
        return HeadReceiver(kind="head", layer=self.layer, head=self.head, input=self.kind)  # type: ignore[arg-type]


@dataclass
class Graph:
    """The edges a scope covers: senders, receivers, and each edge as a pair of their indices."""

    n_layers: int
    n_heads: int
    # Key/value heads a layer: fewer than n_heads with grouped-query attention.
    n_kv_heads: int
    parallel: bool
    senders: list[Node]
    receivers: list[Input]
    edge_sender: np.ndarray  # [E] index into senders
    edge_receiver: np.ndarray  # [E] index into receivers
    whole: bool  # every edge of the model's graph
    # Tensor layout, derived: each sender's and receiver's (block, place), the blocks with
    # edges between them, and each edge's pair and places.
    sender_blocks: dict[Block, list[int]] = field(default_factory=dict)
    receiver_blocks: dict[Block, list[int]] = field(default_factory=dict)
    pairs: list[tuple[Block, Block]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.edge_sender)

    def edge(self, e: int) -> Edge:
        return Edge(
            sender=self.senders[int(self.edge_sender[e])].spec(),
            receiver=self.receivers[int(self.edge_receiver[e])].spec(),
        )

    def label(self, e: int) -> str:
        return (
            f"{self.senders[int(self.edge_sender[e])].label} → "
            f"{self.receivers[int(self.edge_receiver[e])].label}"
        )

    @property
    def receiver_layers(self) -> list[int]:
        return sorted({r.layer for r in self.receivers if r.kind != "logits"})

    @property
    def logits(self) -> bool:
        return any(r.kind == "logits" for r in self.receivers)

    @property
    def sender_layers(self) -> list[int]:
        """The layers whose heads or MLPs send at least one edge."""
        used = {int(s) for s in np.unique(self.edge_sender)}
        return sorted({self.senders[s].layer for s in used if self.senders[s].kind != "embed"})


def senders_of(receiver: Input, n_heads: int, parallel: bool) -> list[Node]:
    """The nodes whose outputs ``receiver`` reads, in the order the residual stream adds them."""
    out = [Node("embed", -1)]
    for layer in range(receiver.layer + 1):
        heads = [Node("head", layer, h) for h in range(n_heads)]
        if layer < receiver.layer:
            out += heads
            out.append(Node("mlp", layer))
        elif receiver.kind == "mlp" and not parallel:
            out += heads  # a sequential layer adds its attention before its MLP reads
    return out


def build_graph(info: ModelInfo, scope: GraphScope) -> Graph:
    """The edges of ``scope`` for the loaded model, in a fixed order: receivers layer by layer
    (each head's query, key and value, then the MLP; the logits last), and each receiver's senders
    in the order the residual stream adds them. With grouped-query attention, each group's query
    heads' queries come first, then its key/value head's key and value."""
    n, H = info.n_layers, info.n_heads
    kv = kv_heads(info)
    if kv < 1 or H % kv:
        raise ScopeError(
            f"This model has {H} heads and {kv} key/value heads a layer, which don't divide into "
            "groups, so its keys and values can't be laid out as edges."
        )
    group = H // kv
    parallel = str(info.extra.get("block_structure")) == "parallel"
    layers = list(range(n)) if scope.layers is None else list(scope.layers)
    for layer in layers:
        if layer >= n:
            raise ScopeError(f"Layer {layer} doesn't exist; this model has {n} layers.")
    senders = [Node("embed", -1)]
    for layer in range(n):
        senders += [Node("head", layer, h) for h in range(H)]
        senders.append(Node("mlp", layer))
    index = {s: i for i, s in enumerate(senders)}
    receivers: list[Input] = []
    for layer in layers:
        for g in range(kv):
            receivers += [Input("q", layer, h) for h in range(g * group, (g + 1) * group)]
            receivers += [Input(part, layer, g, shared=group > 1) for part in ("k", "v")]
        receivers.append(Input("mlp", layer))
    if scope.logits:
        receivers.append(Input("logits", n))
    if not receivers:
        raise ScopeError("The scope has no receivers: choose some layers, or the logits.")
    edge_sender: list[int] = []
    edge_receiver: list[int] = []
    for r, receiver in enumerate(receivers):
        for sender in senders if receiver.kind == "logits" else senders_of(receiver, H, parallel):
            edge_sender.append(index[sender])
            edge_receiver.append(r)
    graph = Graph(
        n_layers=n,
        n_heads=H,
        n_kv_heads=kv,
        parallel=parallel,
        senders=senders,
        receivers=receivers,
        edge_sender=np.asarray(edge_sender, dtype=np.int64),
        edge_receiver=np.asarray(edge_receiver, dtype=np.int64),
        whole=scope.layers is None and scope.logits,
    )
    for i, s in enumerate(senders):
        graph.sender_blocks.setdefault(s.block()[0], []).append(i)
    for i, r in enumerate(receivers):
        graph.receiver_blocks.setdefault(r.block()[0], []).append(i)
    pairs = {
        (senders[int(s)].block()[0], receivers[int(r)].block()[0])
        for s, r in zip(edge_sender, edge_receiver, strict=True)
    }
    graph.pairs = sorted(pairs, key=lambda p: (p[1][1], p[1][0], p[0][1], p[0][0]))
    return graph


def count_edges(
    n_layers: int,
    n_heads: int,
    parallel: bool,
    layers: list[int] | None,
    logits: bool,
    n_kv_heads: int | None = None,
) -> int:
    """How many edges a graph scope has, without building it (the app shows this before a run).
    With grouped-query attention, keys and values are ``n_kv_heads`` a layer."""
    kv = n_heads if n_kv_heads is None else n_kv_heads
    chosen = range(n_layers) if layers is None else layers
    total = 0
    for layer in chosen:
        before = 1 + layer * (n_heads + 1)  # the embeddings, and every head and MLP before
        total += (n_heads + 2 * kv) * before  # each query head's query, each key/value head's two
        total += before + (0 if parallel else n_heads)
    if logits:
        total += 1 + n_layers * (n_heads + 1)
    return total


def edge_sites(graph: Graph) -> tuple[list[ResolvedSite], dict[str, Any]]:
    """One result row per edge, at (sender, receiver) in a senders × receivers layout. With
    grouped-query attention, a key or value receiver's column names the query heads that read it
    (``query_heads``), as a map of keys or values does."""
    last = graph.n_layers - 1
    group = graph.n_heads // graph.n_kv_heads

    def column(r: Input) -> dict[str, Any]:
        out = {"key": r.key, "label": r.label, "node": r.kind, "layer": r.layer, "head": r.head}
        if r.shared and r.head is not None:
            out["query_heads"] = list(range(r.head * group, (r.head + 1) * group))
        return out

    sites = []
    for e in range(len(graph)):
        s, r = int(graph.edge_sender[e]), int(graph.edge_receiver[e])
        layer = min(graph.receivers[r].layer, last)
        sites.append(
            ResolvedSite(
                index=e,
                site=Site(kind="resid_pre", layer=layer, position=ALL_POSITIONS),
                row=s,
                col=r,
                label=graph.label(e),
                edge=graph.edge(e),
            )
        )
    layout = {
        "kind": "edges",
        "row_title": "Sender",
        "col_title": "Receiver",
        "rows": [
            {"key": s.key, "label": s.label, "node": s.kind, "layer": s.layer, "head": s.head}
            for s in graph.senders
        ],
        "cols": [column(r) for r in graph.receivers],
    }
    return sites, layout


# -- what a run costs ----------------------------------------------------------------------------


def edge_cost(
    graph: Graph,
    info: ModelInfo,
    groups: list[LengthGroup],
    batch_size: int,
    exp: EdgeAttribution,
) -> dict[str, Any]:
    """Passes and memory: per batch of prompts and for the whole run, and the largest batch's
    memory in bytes (an estimate, from the tensors the passes hold)."""
    batches = sum(math.ceil(len(g.members) / batch_size) for g in groups)
    length = max(g.length for g in groups)
    rows = min(batch_size, max(len(g.members) for g in groups))
    steps = (exp.steps or 1) if exp.method == "integrated_gradients" else 1
    curve = exp.curve
    # The curve's circuits: each number of edges the graph has, strongest and in each random
    # draw, and every edge. Its batches first capture the senders on the source prompts.
    kept = 0 if curve is None else sum(1 for k in curve.counts if k <= len(graph))
    patched = 0 if curve is None else kept * (1 + curve.control.draws) + 1
    D, H = info.d_model, info.n_heads
    n_receivers = len(graph.receivers)
    # Gradients at every receiver's copy (float32), the split copies the backward pass keeps
    # (about a dozen [B, pos, H, d_model] tensors per layer), the senders' differences (float64)
    # and the scores (float64).
    per_row = (
        length
        * D
        * (
            4 * n_receivers
            + 4 * 12 * H * info.n_layers
            + 8 * (1 + len(graph.sender_layers) * (H + 1))
        )
        + 8 * len(graph.senders) * n_receivers
    )
    return {
        "edges": len(graph),
        "batches": batches,
        # The senders captured on both prompts, then a forward and backward pass per gradient.
        "forward_per_batch": 2 + steps,
        "backward_per_batch": steps,
        "patched_per_batch": patched,
        "forward": (2 + steps + (patched + 1 if patched else 0)) * batches,
        "backward": steps * batches,
        "memory_per_batch": int(per_row * rows),
        # The prompts in the largest batch: fewer than the batch size when no group fills it.
        "largest_batch": rows,
        "values": len(graph) * sum(len(g.members) for g in groups),
    }


def free_memory(device: torch.device) -> int | None:
    """Bytes free where a batch runs: on the GPU for CUDA, otherwise the system's."""
    if device.type == "cuda":
        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001 - unknown: don't refuse on a guess
        return None


def check_cost(cost: dict[str, Any], device: torch.device) -> None:
    """Refuse a run that won't fit: too many edges or values to keep, or a batch too large for
    the memory free here."""
    if cost["edges"] > MAX_EDGES:
        raise EngineError(
            f"This graph has {cost['edges']:,} edges. Logogram keeps every edge's per-prompt values "
            f"and statistics, and more than {MAX_EDGES:,} don't fit. Choose fewer layers."
        )
    if cost["values"] > MAX_VALUES:
        raise EngineError(
            f"{cost['edges']:,} edges over these prompts are {cost['values']:,} per-prompt values, "
            f"more than the {MAX_VALUES:,} a run keeps. Use fewer prompts (dataset.limit) or "
            "fewer layers."
        )
    free = free_memory(device)
    need = int(cost["memory_per_batch"])
    if free is not None and need > 0.8 * free:
        # Memory grows with the prompts a batch actually holds, which is fewer than batch_size
        # when no group of prompts of one length fills it.
        rows = int(cost["largest_batch"])
        fits = int(rows * 0.8 * free / need)
        if fits < 1:
            raise EngineError(
                f"Even one prompt needs about {need / rows / 1e9:.1f} GB here to keep the "
                f"gradient at every receiver, and {free / 1e9:.1f} GB is free. Choose fewer "
                "layers."
            )
        raise EngineError(
            f"The largest batch, {rows} prompt{'s' if rows != 1 else ''}, needs about "
            f"{need / 1e9:.1f} GB here to keep the gradient at every receiver, and "
            f"{free / 1e9:.1f} GB is free. Lower execution.batch_size to {fits} or fewer, or "
            "choose fewer layers."
        )


def edge_plan(spec: Spec, backend: ModelBackend, prompts: list[PreparedPrompt]) -> dict[str, Any]:
    """What an edge attribution spec would do on the loaded model and these prompts, before it
    runs: the graph's size, what the run costs, and why it would be refused (None when it
    wouldn't), from the same checks the run makes."""
    from logogram.backends.base import BackendError

    exp, scope = spec.experiment, spec.scope
    assert isinstance(exp, EdgeAttribution) and isinstance(scope, GraphScope)
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
    cost = edge_cost(graph, backend.info, group_by_length(prompts), spec.execution.batch_size, exp)
    if refusal is None:
        try:
            check_cost(cost, backend.device)
        except EngineError as exc:
            refusal = str(exc)
    return {
        **out,
        "cost": cost,
        "text": cost_text(cost),
        "free": free_memory(backend.device),
        "refusal": refusal,
    }


def cost_text(cost: dict[str, Any]) -> str:
    """What a run costs, in words, for the summary and the command line."""
    backward = cost["backward_per_batch"]
    patched = cost["patched_per_batch"]
    parts = [
        f"{cost['edges']:,} edges",
        f"{cost['batches']} batch{'es' if cost['batches'] != 1 else ''}",
        f"{cost['forward_per_batch']} forward and {backward} backward pass"
        f"{'es' if backward != 1 else ''} per batch",
    ]
    if patched:
        parts.append(f"{patched} edge-patched passes per batch for the curve")
    parts.append(f"about {cost['memory_per_batch'] / 1e9:.2f} GB per batch")
    return ", ".join(parts)


# -- what each node sends ------------------------------------------------------------------------


def sender_keys(graph: Graph) -> list[tuple[str, int]]:
    """The activations a batch's senders are read from: the embeddings (the residual stream
    entering the first layer), and the heads' outputs z and MLP outputs of the sending layers."""
    keys: list[tuple[str, int]] = [("resid_pre", 0)]
    for layer in graph.sender_layers:
        keys += [("head", layer), ("mlp_out", layer)]
    return keys


def _differences(
    backend: ModelBackend,
    graph: Graph,
    source: dict[tuple[str, int], torch.Tensor],
    receiver: dict[tuple[str, int], torch.Tensor],
    weights: dict[int, torch.Tensor],
) -> dict[Block, torch.Tensor]:
    """Each sending block's output on the source prompt minus on the receiver prompt, in
    float64, ``[B, pos, senders, d_model]``: a head's is its z difference through its slice of
    the output projection."""
    out: dict[Block, torch.Tensor] = {}
    key = ("resid_pre", 0)
    out[("embed", -1)] = (float64(source[key]) - float64(receiver[key]))[:, :, None, :]
    for layer in graph.sender_layers:
        dz = float64(source[("head", layer)]) - float64(receiver[("head", layer)])
        w = weights[layer].to(dz.device)
        out[("head", layer)] = torch.einsum("bphe,hed->bphd", dz, w)
        key = ("mlp_out", layer)
        out[("mlp", layer)] = (float64(source[key]) - float64(receiver[key]))[:, :, None, :]
    return out


def _gradient_block(grads: InputGradients, block: Block) -> torch.Tensor:
    """A receiving block's gradients, ``[B, pos, receivers, d_model]``."""
    kind, layer = block
    if kind == "logits":
        assert grads.logits is not None
        return grads.logits[:, :, None, :]
    if kind == "mlp":
        g = grads.mlp[layer]
        assert g is not None
        return g[:, :, None, :]
    g = getattr(grads, kind)[layer]
    assert g is not None
    return g


def _scores(graph: Graph, deltas: dict[Block, torch.Tensor], grads: InputGradients) -> torch.Tensor:
    """Every edge's estimate for each row, ``[B, E]`` in float64: the sender's difference dotted
    with the gradient at the receiver's own input, summed over positions."""
    like = next(iter(deltas.values()))  # float64, where float64() put it
    device = like.device
    dense = like.new_zeros((like.shape[0], len(graph.senders), len(graph.receivers)))
    # The pairs come receiving block by receiving block: each block's gradients convert once.
    converted: dict[Block, torch.Tensor] = {}
    for sblock, rblock in graph.pairs:
        if rblock not in converted:
            converted = {rblock: float64(_gradient_block(grads, rblock)).to(device)}
        g = converted[rblock]
        d = deltas[sblock]
        part = torch.einsum("bpsd,bprd->bsr", d, g)
        rows = torch.tensor(graph.sender_blocks[sblock], device=device)
        cols = torch.tensor(graph.receiver_blocks[rblock], device=device)
        dense[:, rows[:, None], cols[None, :]] = part
    e_s = torch.as_tensor(graph.edge_sender, device=device)
    e_r = torch.as_tensor(graph.edge_receiver, device=device)
    return dense[:, e_s, e_r]


# -- edge patching -------------------------------------------------------------------------------


@dataclass
class PatchPlan:
    """Which edges carry the source prompt's values: per (sending block, receiving block), a
    0/1 matrix ``[senders, receivers]`` over the blocks' places."""

    masks: dict[tuple[Block, Block], torch.Tensor]

    @classmethod
    def of(cls, graph: Graph, edges: np.ndarray, device: torch.device) -> PatchPlan:
        masks: dict[tuple[Block, Block], np.ndarray] = {}
        for e in edges:
            sender = graph.senders[int(graph.edge_sender[e])]
            receiver = graph.receivers[int(graph.edge_receiver[e])]
            (sblock, s), (rblock, r) = sender.block(), receiver.block()
            pair = (sblock, rblock)
            if pair not in masks:
                masks[pair] = np.zeros(
                    (_places(graph, sblock), _places(graph, rblock)), dtype=np.float32
                )
            masks[pair][s, r] = 1.0
        return cls({k: torch.from_numpy(v).to(device) for k, v in masks.items()})


def _places(graph: Graph, block: Block) -> int:
    if block[0] in ("k", "v"):
        return graph.n_kv_heads
    return graph.n_heads if block[0] in ("head", "q") else 1


class _Patcher(EdgePatcher):
    """One edge-patched pass: each sender's difference (source − this pass) as the pass sends
    it, added into the receivers' own inputs the plan patches."""

    def __init__(
        self,
        plan: PatchPlan,
        source: dict[tuple[str, int], torch.Tensor],
        weights: dict[int, torch.Tensor],
    ) -> None:
        self.plan = plan
        self.source = source
        self.weights = weights
        self.into: dict[Block, list[tuple[Block, torch.Tensor]]] = {}
        for (sblock, rblock), mask in plan.masks.items():
            self.into.setdefault(rblock, []).append((sblock, mask))
        self.needed = {sblock for sblock, _ in plan.masks}
        self.deltas: dict[Block, torch.Tensor] = {}

    def sent(self, node: str, layer: int, output: torch.Tensor) -> None:
        block = (node, layer)
        if block not in self.needed:
            return
        key = {"embed": ("resid_pre", 0), "head": ("head", layer), "mlp": ("mlp_out", layer)}[node]
        difference = float64(self.source[key]) - float64(output)
        if node == "head":
            w = self.weights[layer].to(difference.device)
            self.deltas[block] = torch.einsum("bphe,hed->bphd", difference, w).float()
        else:
            self.deltas[block] = difference.float()[:, :, None, :]

    def update(self, receiver: str, layer: int) -> torch.Tensor | None:
        terms = self.into.get((receiver, layer))
        if not terms:
            return None
        total: torch.Tensor | None = None
        for sblock, mask in terms:
            d = self.deltas[sblock]
            part = torch.einsum("bpsd,sr->bprd", d, mask.to(d.device))
            total = part if total is None else total + part
        assert total is not None
        return total if receiver in ("q", "k", "v") else total[:, :, 0]


def patched_forward(
    backend: ModelBackend,
    plan: PatchPlan,
    source: dict[tuple[str, int], torch.Tensor],
    weights: dict[int, torch.Tensor],
) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """A forward pass, as the scorer calls it, with the plan's edges patched."""

    def forward(tokens: torch.Tensor, keep: int) -> torch.Tensor:
        assert keep == 1  # checked: everything is read at the last position
        return backend.edge_patch(tokens, _Patcher(plan, source, weights))[:, None, :]

    return forward


# -- the run -------------------------------------------------------------------------------------


def _ranking(values: np.ndarray) -> np.ndarray:
    """Edges by the magnitude of their mean, strongest first; ties (and undefined means, last) in
    the order of the graph."""
    strength = np.where(np.isfinite(values), np.abs(values), -np.inf)
    return np.lexsort((np.arange(len(values)), -strength))


def random_orders(n_edges: int, seed: int, draws: int) -> list[np.ndarray]:
    """``draws`` orderings of the edges, each a seeded shuffle: the raw PCG64 stream (stable across
    NumPy versions), sorted as keys."""
    bitgen = np.random.PCG64(seed)
    return [np.argsort(bitgen.random_raw(n_edges), kind="stable") for _ in range(draws)]


def run_edge_attribution(
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
    on_prompts: PromptsFn | None = None,
) -> EngineResult:
    """Estimate every edge of the scope, batch by batch of prompts (each finished batch is
    reported through ``on_prompts``, so a run that stops early keeps every edge over the prompts
    that finished), then measure the faithfulness curve."""
    exp = spec.experiment
    scope = spec.scope
    assert isinstance(exp, EdgeAttribution) and isinstance(scope, GraphScope)
    info = backend.info
    batch_size = spec.execution.batch_size
    checks = backend.edge_checks()
    graph = build_graph(info, scope)
    scorer = make_scorer(spec, prompts)
    if not scorer.single_position():
        raise ScopeError(
            "Edge attribution reads the metric at the last prompt position, so answers and "
            "distractors must be single tokens or sets of single tokens."
        )
    groups = group_by_length(prompts)
    cost = edge_cost(graph, info, groups, batch_size, exp)
    check_cost(cost, backend.device)
    sites, layout = edge_sites(graph)
    if on_start is not None:
        on_start(sites, layout)
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
    keys = sender_keys(graph)
    weights = {layer: backend.head_output_weights(layer) for layer in range(info.n_layers)}
    steps = exp.steps if exp.method == "integrated_gradients" else None
    estimate = np.zeros((len(graph), n))
    finished: list[int] = []
    total_rows = cost["batches"] * (1 + (0 if exp.curve is None else cost["patched_per_batch"]))
    done_rows = 0
    extra = {
        "method": exp.method,
        "steps": exp.steps,
        "n_edges": len(graph),
        "n_senders": len(graph.senders),
        "n_receivers": len(graph.receivers),
        "reported": min(len(graph), SUMMARY_EDGES),
        "structure": "parallel" if graph.parallel else "sequential",
        "gradient": GRADIENT_TEXT,
        "checks": checks,
        "cost": {**cost, "text": cost_text(cost)},
    }

    def over(at: list[int]) -> EngineResult:
        """Every edge's estimates over the prompts at positions ``at``, as a run of only those
        prompts gives them (without the curve, which runs after every prompt)."""
        chosen = [prompts[i] for i in at]
        base = baselines.subset(at)
        notes = check_gap(spec, base, chosen, receiver, reference)
        return _result(
            sites,
            layout,
            chosen,
            base,
            receiver,
            reference,
            estimate[:, at],
            notes,
            info.dtype,
            {**extra},
        )

    for group in groups:
        receiver_tokens = group.clean if receiver == "clean" else group.corrupt
        source_tokens = group.clean if source == "clean" else group.corrupt
        for sl in _chunks(len(group.members), batch_size):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            idx = group.members[sl]
            rcv = receiver_tokens[sl]
            src_acts = backend.capture(source_tokens[sl], keys)
            rcv_acts = backend.capture(rcv, keys)
            deltas = _differences(backend, graph, src_acts, rcv_acts, weights)

            def score(logits: torch.Tensor, idx: list[int] = idx) -> torch.Tensor:
                return scorer.differentiable(logits, idx, "answer")

            layers = graph.receiver_layers
            if steps is None:
                grads = backend.input_gradients(rcv, score, layers, graph.logits)
                scores = _scores(graph, deltas, grads)
            else:
                start, end = rcv_acts[("resid_pre", 0)], src_acts[("resid_pre", 0)]
                scores = None
                for k in range(steps):
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
                    alpha = (k + 0.5) / steps
                    grads = backend.input_gradients(
                        rcv, score, layers, graph.logits, start + alpha * (end - start)
                    )
                    part = _scores(graph, deltas, grads)
                    scores = part if scores is None else scores + part
                assert scores is not None
                scores = scores / steps
            estimate[:, idx] = scores.cpu().numpy().T
            del deltas
            finished.extend(idx)
            done_rows += 1
            if on_progress is not None:
                on_progress(done_rows, total_rows, info.n_layers - 1)
            if on_prompts is not None:
                at = sorted(finished)
                on_prompts(at, lambda at=at: over(at))

    if exp.curve is not None:

        def advance() -> None:
            nonlocal done_rows
            done_rows += 1
            if on_progress is not None:
                on_progress(done_rows, total_rows, info.n_layers - 1)

        curve, notes = faithfulness_curve(
            spec,
            backend,
            prompts,
            groups,
            scorer,
            baselines,
            graph,
            estimate,
            receiver,
            source,
            weights,
            keys,
            cancel=cancel,
            step=advance,
        )
        extra["curve"] = curve
        warnings.extend(notes)
    return _result(
        sites,
        layout,
        prompts,
        baselines,
        receiver,
        reference,
        estimate,
        warnings,
        info.dtype,
        extra,
    )


def _result(
    sites: list[ResolvedSite],
    layout: dict[str, Any],
    prompts: list[PreparedPrompt],
    baselines: Baselines,
    receiver: str,
    reference: str,
    estimate: np.ndarray,
    warnings: list[str],
    dtype: str,
    extra: dict[str, Any],
) -> EngineResult:
    warnings = [*warnings, *check_values(estimate, "edge estimates", dtype)]
    nan = np.full(estimate.shape, np.nan)
    return EngineResult(
        sites=sites,
        layout=layout,
        prompts=prompts,
        baselines=baselines,
        receiver=receiver,
        reference=reference,
        patched=baselines.metric(receiver)[None, :] + estimate,
        patched_prob=nan,
        patched_pref=nan.copy(),
        warnings=warnings,
        measure="estimate",
        delta=estimate,
        extra={"edges": extra},
    )


def faithfulness_curve(
    spec: Spec,
    backend: ModelBackend,
    prompts: list[PreparedPrompt],
    groups: list[LengthGroup],
    scorer: Scorer,
    baselines: Baselines,
    graph: Graph,
    estimate: np.ndarray,
    receiver: str,
    source: str,
    weights: dict[int, torch.Tensor],
    keys: list[tuple[str, int]],
    *,
    cancel: threading.Event | None = None,
    step: Callable[[], None] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Patch the k strongest edges (or every edge but them), for each k of the spec's curve,
    beside edges chosen at random; and every edge of the scope, which with the whole graph is the
    source run. Returns the curve for the summary, and notes for the run's warnings."""
    from logogram.results import bootstrap_counts
    from logogram.stats import normalize

    exp = spec.experiment
    assert isinstance(exp, EdgeAttribution) and exp.curve is not None
    curve = exp.curve
    E = len(graph)
    n = len(prompts)
    notes: list[str] = []
    counts = [k for k in curve.counts if k <= E]
    dropped = [k for k in curve.counts if k > E]
    if dropped:
        notes.append(
            f"The curve's counts above this graph's {E:,} edges ({', '.join(map(str, dropped))}) "
            "weren't measured: every edge is the curve's last point."
        )
    gap = baselines.metric(source) - baselines.metric(receiver)
    effect, _ = normalize(estimate, gap, spec.metric.normalization)
    with np.errstate(invalid="ignore"):
        strongest = _ranking(effect.mean(axis=1))
    orders = random_orders(E, curve.control.seed, curve.control.draws)
    everything = np.arange(E)

    def patched(chosen: np.ndarray) -> np.ndarray:
        """The edges a point patches: the chosen ones, or every other edge of the scope."""
        if curve.patch == "chosen":
            return np.sort(chosen)
        keep = np.ones(E, dtype=bool)
        keep[chosen] = False
        return everything[keep]

    device = backend.device
    plans: list[PatchPlan] = []
    for k in counts:
        plans.append(PatchPlan.of(graph, patched(strongest[:k]), device))
    for k in counts:
        for order in orders:
            plans.append(PatchPlan.of(graph, patched(order[:k]), device))
    plans.append(PatchPlan.of(graph, everything, device))  # every edge patched
    metric = np.zeros((len(plans), n))
    for group in groups:
        receiver_tokens = group.clean if receiver == "clean" else group.corrupt
        source_tokens = group.clean if source == "clean" else group.corrupt
        for sl in _chunks(len(group.members), spec.execution.batch_size):
            idx = group.members[sl]
            src = backend.capture(source_tokens[sl], keys)
            for i, plan in enumerate(plans):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                forward = patched_forward(backend, plan, src, weights)
                metric[i, idx] = scorer.score(forward, receiver_tokens[sl], idx).metric
                if step is not None:
                    step()
    # Faithfulness: 0 for the run with no edge chosen, 1 for every edge of the graph chosen. With
    # the chosen edges patched, no edge chosen is the receiver run and every edge the source run;
    # with every other edge patched, the other way round.
    if curve.patch == "chosen":
        low, high = baselines.metric(receiver), baselines.metric(source)
    else:
        low, high = baselines.metric(source), baselines.metric(receiver)
    K, D = len(counts), curve.control.draws
    strong_rows = metric[:K]
    draw_rows = metric[K : K + K * D].reshape(K, D, n)
    random_rows = draw_rows.mean(axis=1)
    every_row = metric[-1:]
    probe_result = EngineResult(
        sites=[],
        layout={},
        prompts=prompts,
        baselines=baselines,
        receiver=receiver,
        reference=source,
        patched=np.zeros((0, n)),
        patched_prob=np.zeros((0, n)),
        patched_pref=np.zeros((0, n)),
    )
    resamples = bootstrap_counts(spec, probe_result)
    rows = np.concatenate([strong_rows, random_rows, every_row])
    stats = _faithfulness(spec, rows, low, high, resamples)
    per_draw = [
        [_mean_faithfulness(spec, draw_rows[i, d], low, high) for d in range(D)] for i in range(K)
    ]

    def point(i: int, k: int, draws: list[float | None] | None = None) -> dict[str, Any]:
        return {
            "k": k,
            "faithfulness": stats[i],
            "metric": _finite(rows[i].mean()),
            "draws": draws,
        }

    out: dict[str, Any] = {
        "patch": curve.patch,
        "counts": counts,
        "dropped": dropped,
        "seed": curve.control.seed,
        "draws": D,
        "strongest": [point(i, k) for i, k in enumerate(counts)],
        "random": [point(K + i, k, per_draw[i]) for i, k in enumerate(counts)],
        "every": point(2 * K, E if curve.patch == "chosen" else 0),
        "source_difference": (
            _finite(np.abs(metric[-1] - baselines.metric(source)).max()) if graph.whole else None
        ),
        "per_prompt": {
            "strongest": [[_finite(v) for v in r] for r in strong_rows],
            "random": [[_finite(v) for v in r] for r in draw_rows.reshape(K * D, n)],
            "every": [[_finite(v) for v in every_row[0]]],
        },
    }
    if graph.whole and out["source_difference"] is not None:
        scale = float(np.abs(baselines.metric(source)).max()) or 1.0
        if out["source_difference"] > 1e-3 * scale:
            notes.append(
                f"Patching every edge should reproduce the {source} run, and differs from it by up "
                f"to {out['source_difference']:.3g} in the metric: the edges don't add up to the "
                "model exactly in this precision. Read the curve with that margin."
            )
    return out, notes


def _faithfulness(
    spec: Spec, rows: np.ndarray, low: np.ndarray, high: np.ndarray, counts: np.ndarray
) -> list[dict[str, float | None]]:
    """Each row's faithfulness with the run's statistics: its normalized position between ``low``
    (0) and ``high`` (1), with a percentile bootstrap interval from the run's resamples."""
    from logogram.stats import compute_site_stats

    nan = np.full(rows.shape, np.nan)
    stats = compute_site_stats(
        patched=rows,
        patched_prob=nan,
        patched_pref=nan.copy(),
        receiver=low,
        reference=high,
        receiver_prob=np.full(len(low), np.nan),
        receiver_pref=np.full(len(low), np.nan),
        normalization=spec.metric.normalization,
        counts=counts,
        ci=spec.statistics.ci,
    )
    return [
        {
            "mean": _finite(stats.effect_mean[i]),
            "sd": _finite(stats.effect_sd[i]),
            "lo": _finite(stats.effect_lo[i]),
            "hi": _finite(stats.effect_hi[i]),
        }
        for i in range(len(rows))
    ]


def _mean_faithfulness(
    spec: Spec, row: np.ndarray, low: np.ndarray, high: np.ndarray
) -> float | None:
    from logogram.stats import normalize

    effect, _ = normalize((row - low)[None, :], high - low, spec.metric.normalization)
    return _finite(effect.mean())


def _finite(x: Any) -> float | None:
    value = float(x)
    return value if math.isfinite(value) else None


# -- verification by path patching ---------------------------------------------------------------


def edge_verification_spec(spec: Spec, summary: dict[str, Any], top: int) -> Spec:
    """The spec that patches, with path patching, the ``top`` edges an edge attribution run
    estimated to matter most (by magnitude), one at a time: each edge's sender into its receiver
    alone, every head and MLP held, so the change reaches the receiver directly as the edge
    carries it."""
    exp = spec.experiment
    if not isinstance(exp, EdgeAttribution):
        raise ValueError("Only edge attribution runs are verified edge by edge.")
    top = min(top, VERIFY_MAX)
    ranked = sorted(
        (s for s in summary["sites"] if s.get("edge") and s["effect"]["mean"] is not None),
        key=lambda s: (-abs(s["effect"]["mean"]), s["index"]),
    )[:top]
    if not ranked:
        raise ValueError("This run has no estimated edges to verify.")
    edges = [Edge.model_validate(s["edge"]) for s in ranked]
    receivers = list({e.receiver.model_dump_json(): e.receiver for e in edges}.values())
    suffix = " · verified"
    data = spec.model_dump(mode="json")
    data.update(
        {
            "name": spec.name[: NAME_MAX - len(suffix)].rstrip() + suffix,
            "experiment": PathPatching(
                direction=exp.direction, receivers=receivers, freeze_mlps=True
            ).model_dump(mode="json"),
            "scope": EdgesScope(edges=edges).model_dump(mode="json"),
        }
    )
    return Spec.model_validate(data)


def edge_key(edge: dict[str, Any] | Edge) -> str:
    """An edge's identity, the same in any run that names it."""
    value = edge if isinstance(edge, Edge) else Edge.model_validate(edge)
    return value.model_dump_json()


def verification_table(estimated: dict[str, Any], measured: dict[str, Any]) -> dict[str, Any]:
    """Each verified edge's estimate beside its measurement by path patching (normalized effects,
    with their intervals), ranked both ways; and how well the two agree: Spearman's rank
    correlation of the means and how many have the same sign."""
    from logogram.stats import spearman

    estimates = {edge_key(s["edge"]): s for s in estimated["sites"] if s.get("edge")}
    rows = []
    for s in measured["sites"]:
        if not s.get("edge"):
            continue
        e = estimates.get(edge_key(s["edge"]))
        if e is None:
            continue
        rows.append(
            {
                "label": s["label"],
                "edge": s["edge"],
                "estimate": e["effect"],
                "measured": s["effect"],
                "estimate_index": e["index"],
                "measured_index": s["index"],
            }
        )
    pairs = [
        r for r in rows if r["estimate"]["mean"] is not None and r["measured"]["mean"] is not None
    ]
    a = np.array([r["estimate"]["mean"] for r in pairs], dtype=np.float64)
    b = np.array([r["measured"]["mean"] for r in pairs], dtype=np.float64)
    for which, values in (("estimate_rank", a), ("measured_rank", b)):
        order = np.lexsort((np.arange(len(values)), -np.abs(values)))
        for rank, i in enumerate(order, start=1):
            pairs[int(i)][which] = rank
    rho = spearman(a, b) if len(pairs) >= 2 else float("nan")
    rows.sort(key=lambda r: (r.get("estimate_rank") or len(rows) + 1, r["estimate_index"]))
    return {
        "run": measured["run_id"],
        "n": len(pairs),
        "spearman": rho if math.isfinite(rho) else None,
        "same_sign": int(np.sum(np.sign(a) == np.sign(b))),
        "rows": rows,
    }


def latest_verification(
    project: Any, run_id: str, summary: dict[str, Any]
) -> dict[str, Any] | None:
    """An edge attribution run's estimates against its latest finished verification by path
    patching (:func:`verification_table`), or None when none has finished."""
    done = [
        r
        for r in project.list_runs()
        if r.derived_from
        and r.derived_from.run == run_id
        and r.derived_from.kind == "verification"
        and r.status == "finished"
    ]
    if not done or not summary.get("edges"):
        return None
    latest = max(done, key=lambda r: (r.created or "", r.id))
    measured = project.read_json(project.run_dir(latest.id) / "summary.json", optional=True)
    return verification_table(summary, measured) if measured else None


# -- what a run writes ---------------------------------------------------------------------------


def correction_counts(stats: Any, alpha: float) -> dict[str, int] | None:
    """Over every edge (the summary lists only the strongest): how many intervals exclude zero
    one edge at a time, how many simultaneous bands do, and how many edges have a q-value below
    ``alpha``; None when the run has no corrections (fewer than two edges)."""
    if stats.band_lo is None or stats.band_hi is None or stats.q is None:
        return None
    lo, hi = np.asarray(stats.effect_lo), np.asarray(stats.effect_hi)
    band_lo, band_hi = np.asarray(stats.band_lo), np.asarray(stats.band_hi)
    with np.errstate(invalid="ignore"):
        return {
            "edges": len(lo),
            "single": int(np.sum((lo > 0) | (hi < 0))),
            "banded": int(np.sum((band_lo > 0) | (band_hi < 0))),
            "discoveries": int(np.sum(np.asarray(stats.q) < alpha)),
        }


def reported_edges(stats: Any, first: list[int] | None = None) -> list[int]:
    """The edges a summary lists: the strongest :data:`SUMMARY_EDGES` by the magnitude of their
    mean effect (ties in the graph's order), in the graph's order. ``first`` (an ACDC circuit's
    edges) come before every other edge: all of them, or their strongest when they are more."""
    strongest = _ranking(np.asarray(stats.effect_mean, dtype=np.float64))
    if first:
        listed = np.zeros(len(strongest), dtype=bool)
        listed[np.asarray(first, dtype=np.int64)] = True
        ahead = listed[strongest]
        strongest = np.concatenate([strongest[ahead], strongest[~ahead]])
    return sorted(int(e) for e in strongest[:SUMMARY_EDGES])


def edge_table(result: EngineResult, stats: Any) -> Any:
    """Every edge's statistics, one row per edge (edges.parquet): what the summary reports for
    the strongest, for all of them. ``stats`` rows are the result's edges. An ACDC run's table
    also says whether each edge was kept in the circuit (``kept``) and when it was tested
    (``step``, from 0: the reverse of the graph's order)."""
    import pyarrow as pa

    def column(values: Any) -> Any:
        return np.asarray(values, dtype=np.float64)

    sites = result.sites
    edges: list[Edge] = []
    for s in sites:
        assert s.edge is not None
        edges.append(s.edge)
    nan = np.full(len(sites), np.nan)
    band_lo = nan if stats.band_lo is None else stats.band_lo
    band_hi = nan if stats.band_hi is None else stats.band_hi
    q = nan if stats.q is None else stats.q
    index = np.array([s.index for s in sites], dtype=np.int32)
    columns = {
        "edge": index,
        "label": [s.label for s in sites],
        "sender": [e.sender.kind for e in edges],
        "sender_layer": np.array(
            [-1 if e.sender.layer is None else e.sender.layer for e in edges],
            dtype=np.int16,
        ),
        "sender_head": np.array(
            [-1 if e.sender.head is None else e.sender.head for e in edges],
            dtype=np.int16,
        ),
        "receiver": [
            "logits"
            if isinstance(e.receiver, LogitsReceiver)
            else "mlp"
            if isinstance(e.receiver, MlpReceiver)
            else e.receiver.input
            for e in edges
        ],
        "receiver_layer": np.array(
            [getattr(e.receiver, "layer", -1) for e in edges], dtype=np.int16
        ),
        "receiver_head": np.array([getattr(e.receiver, "head", -1) for e in edges], dtype=np.int16),
        "n": np.full(len(sites), stats.n, dtype=np.int32),
        "effect_mean": column(stats.effect_mean),
        "effect_sd": column(stats.effect_sd),
        "effect_lo": column(stats.effect_lo),
        "effect_hi": column(stats.effect_hi),
        "delta_mean": column(stats.delta_mean),
        "delta_sd": column(stats.delta_sd),
        "delta_lo": column(stats.delta_lo),
        "delta_hi": column(stats.delta_hi),
        "band_lo": column(band_lo),
        "band_hi": column(band_hi),
        "q": column(q),
        "opposite_sign": np.asarray(stats.opposite_sign, dtype=np.int32),
    }
    acdc = result.extra.get("acdc")
    if acdc is not None:
        kept = np.zeros(int(acdc["n_edges"]), dtype=bool)
        kept[np.asarray(acdc["kept"], dtype=np.int64)] = True
        columns["kept"] = kept[index]
        columns["step"] = (int(acdc["n_edges"]) - 1 - index).astype(np.int32)
    return pa.table(columns)


def write_edge_table(path: Any, result: EngineResult, stats: Any) -> None:
    """edges.parquet for an edge attribution or ACDC run's result (``stats`` over every edge)."""
    from logogram.results import write_results

    write_results(path, edge_table(result, stats))
