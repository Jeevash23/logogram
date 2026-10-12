"""Edge attribution: the graph, the gradient through each receiver's normalization, exact edge
patching, estimates against path patching, the faithfulness curve, and runs end to end."""

from __future__ import annotations

import json
import threading
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from logogram.backends.base import BackendError, EdgePatcher, ModelInfo
from logogram.datasets import load_dataset
from logogram.edges import (
    EDGES_FILE,
    PatchPlan,
    _Patcher,
    build_graph,
    count_edges,
    edge_verification_spec,
    random_orders,
    sender_keys,
    verification_table,
)
from logogram.engine import EngineError, run_experiment
from logogram.prompts import group_by_length, prepare_prompts
from logogram.runner import run_spec
from logogram.spec import GraphScope, Spec
from logogram.stats import compute_site_stats, resample_counts

CURVE = {"counts": [1, 3, 10, 30], "patch": "chosen", "control": {"seed": 0, "draws": 2}}


def edge_spec(spec_factory: Any, **experiment: Any) -> Spec:
    return spec_factory(
        experiment={
            "kind": "edge_attribution",
            "direction": "clean_to_corrupt",
            "method": "gradient",
            "steps": None,
            "curve": None,
            **experiment,
        },
        scope={"kind": "graph", "layers": None, "logits": True},
    )


def _prompts(backend: Any, project: Any) -> Any:
    return prepare_prompts(backend, load_dataset(project.datasets_dir / "ioi.jsonl"), True)


@pytest.fixture(scope="module")
def smooth_backend(tmp_path_factory: Any) -> Any:
    """Tiny models with small weights, smooth enough for finite differences to measure slopes."""
    from conftest import build_tiny_architecture
    from logogram.backends.transformer_lens import TransformerLensBackend, boot_local

    made: dict[str, Any] = {}

    def get(arch: str) -> Any:
        if arch not in made:
            folder = tmp_path_factory.mktemp(f"smooth-edges-{arch}")
            build_tiny_architecture(folder, arch, initializer_range=0.05)
            made[arch] = TransformerLensBackend.from_bridge(
                boot_local(folder, device="cpu", dtype="float32"),
                model_id=f"smooth-{arch}",
                revision="test",
                dtype="float32",
                process_weights=True,
            )
        return made[arch]

    return get


def _info(n_layers: int, n_heads: int, structure: str) -> ModelInfo:
    return ModelInfo(
        id="shape",
        revision=None,
        architecture="test",
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=8,
        d_head=2,
        d_mlp=32,
        d_vocab=10,
        n_ctx=16,
        n_params=None,
        dtype="float32",
        device="cpu",
        device_name="cpu",
        process_weights=True,
        site_kinds=("resid_pre", "resid_post", "attn_out", "mlp_out", "head"),
        backend="test",
        backend_version="0",
        extra={"block_structure": structure},
    )


# -- the graph -----------------------------------------------------------------------------------


def test_the_graph_has_the_edges_of_the_literature() -> None:
    # GPT-2 small: 12 layers of 12 heads, attention before the MLP in each layer.
    assert count_edges(12, 12, False, None, True) == 32_491
    graph = build_graph(_info(12, 12, "sequential_pre_norm"), GraphScope(layers=None, logits=True))
    assert len(graph) == 32_491
    assert (len(graph.senders), len(graph.receivers)) == (157, 445)
    # Where attention and MLP read the same stream (Pythia), an MLP doesn't read its own heads.
    parallel = build_graph(_info(12, 12, "parallel"), GraphScope(layers=None, logits=True))
    assert len(parallel) == count_edges(12, 12, True, None, True) == 32_491 - 12 * 12
    for layers, logits in (([0], False), ([3, 4], True), (None, False), ([11], True)):
        scope = GraphScope(layers=layers, logits=logits)
        assert len(build_graph(_info(12, 12, "sequential"), scope)) == count_edges(
            12, 12, False, layers, logits
        )


def test_every_edge_runs_forward_and_is_named() -> None:
    graph = build_graph(_info(3, 2, "sequential"), GraphScope(layers=None, logits=True))
    labels = {graph.label(e) for e in range(len(graph))}
    assert "embeddings → L0 H0 q" in labels
    assert "L0 H1 → L1 H0 v" in labels
    assert "L1 H1 → L1 MLP" in labels  # a layer's MLP reads its own heads here
    assert "L2 MLP → logits" in labels
    assert "L1 MLP → L1 H0 q" not in labels  # nothing reads its own layer's MLP but later ones
    assert len(labels) == len(graph)
    for e in range(len(graph)):
        edge = graph.edge(e)  # validates that the sender comes first
        assert graph.label(e) == f"{graph.senders[graph.edge_sender[e]].label} → " + (
            graph.receivers[graph.edge_receiver[e]].label
        )
        assert edge.sender.kind in ("embed", "head", "mlp")


# -- the spec ------------------------------------------------------------------------------------


def test_the_spec_states_every_choice(spec_factory: Any) -> None:
    spec = edge_spec(spec_factory, curve=CURVE)
    assert spec.experiment.kind == "edge_attribution"
    again = Spec.model_validate(spec.model_dump(mode="json"))
    assert again == spec
    data = spec.model_dump(mode="json")
    for broken, message in (
        ({"scope": {"kind": "heads", "position": {"kind": "all"}}}, "graph scope"),
        ({"experiment": {**data["experiment"], "method": "integrated_gradients"}}, "steps"),
        ({"experiment": {**data["experiment"], "steps": 4}}, "takes no steps"),
        (
            {"experiment": {**data["experiment"], "curve": {**CURVE, "counts": [3, 1]}}},
            "increase",
        ),
        ({"scope": {"kind": "graph", "layers": [], "logits": True}}, "at least one layer"),
        ({"scope": {"kind": "graph", "layers": [2, 1], "logits": True}}, "in order"),
    ):
        with pytest.raises(ValueError, match=message):
            Spec.model_validate({**data, **broken})
    without = {k: v for k, v in data["experiment"].items() if k != "curve"}
    with pytest.raises(ValueError, match="curve"):
        Spec.model_validate({**data, "experiment": without})
    with pytest.raises(ValueError, match="graph scope"):
        spec_factory(scope={"kind": "graph", "layers": None, "logits": True})


def test_an_edges_scope_is_path_patched_with_everything_held(spec_factory: Any) -> None:
    edge = {
        "sender": {"kind": "head", "layer": 0, "head": 1},
        "receiver": {"kind": "head", "layer": 1, "head": 0, "input": "k"},
    }
    path = {
        "kind": "path_patching",
        "direction": "clean_to_corrupt",
        "receivers": [edge["receiver"]],
        "freeze_mlps": True,
    }
    spec = spec_factory(experiment=path, scope={"kind": "edges", "edges": [edge]})
    assert spec.scope.kind == "edges"
    with pytest.raises(ValueError, match="freeze_mlps"):
        spec_factory(experiment={**path, "freeze_mlps": False}, scope=spec.scope.model_dump())
    with pytest.raises(ValueError, match="each edge's receiver"):
        spec_factory(
            experiment={**path, "receivers": [{"kind": "logits"}]}, scope=spec.scope.model_dump()
        )
    with pytest.raises(ValueError, match="earlier layer"):
        spec_factory(
            experiment=path,
            scope={
                "kind": "edges",
                "edges": [{**edge, "sender": {"kind": "head", "layer": 1, "head": 0}}],
            },
        )
    with pytest.raises(ValueError, match="starts before it"):
        spec_factory(
            experiment={**path, "receivers": [{"kind": "mlp", "layer": 0}]},
            scope={
                "kind": "edges",
                "edges": [
                    {"sender": {"kind": "mlp", "layer": 0}, "receiver": {"kind": "mlp", "layer": 0}}
                ],
            },
        )


# -- what the model must do ----------------------------------------------------------------------


def test_the_model_is_measured_before_its_edges_are(tiny_backend: Any, arch_backend: Any) -> None:
    checks = tiny_backend.edge_checks()
    assert set(checks) == {"split", "stream", "separate"}
    assert 0 <= checks["split"] <= 1e-4 and 0 <= checks["separate"] <= 1e-4
    # Every receiver's copy is the residual stream itself, before its normalization.
    assert checks["stream"] == 0
    neox = arch_backend("gpt_neox").edge_checks()  # its MLPs read the stream its heads read
    assert neox["separate"] <= 1e-4 and neox["stream"] == 0
    with pytest.raises(BackendError, match="shares each key and value"):
        arch_backend("llama").edge_checks()
    for arch in ("gemma2", "olmo2"):  # they normalize after combining their heads
        with pytest.raises(BackendError, match="don't add up"):
            arch_backend(arch).edge_checks()


def test_a_copy_taken_after_its_normalization_is_refused(
    tiny_backend: Any, monkeypatch: Any
) -> None:
    """A receiver must read the residual stream itself and normalize its own copy, whatever its
    normalization's epsilon: a copy taken after the normalization isn't the stream, and the
    model's edges are refused."""
    from logogram.backends import transformer_lens as tl

    monkeypatch.setitem(tl.EDGE_HOOKS, "mlp", "blocks.{layer}.ln2.hook_normalized")
    monkeypatch.setattr(tiny_backend, "_edges", None)
    with pytest.raises(BackendError, match="read the residual stream itself"):
        tiny_backend.edge_checks()


def test_a_model_that_cant_be_measured_is_refused(
    arch_backend: Any, project: Any, spec_factory: Any
) -> None:
    backend = arch_backend("olmo2")
    with pytest.raises(BackendError, match="don't add up"):
        run_experiment(edge_spec(spec_factory), backend, _prompts(backend, project))


# -- the gradient through the normalization -------------------------------------------------------


def test_the_normalization_s_scale_is_recomputed_not_held() -> None:
    """The derivation in logogram.edges, on a bare LayerNorm: the gradient with the scale
    recomputed matches finite differences and is orthogonal to the input; holding the scale gives
    another vector, which finite differences don't follow."""
    torch.manual_seed(0)
    d = 16
    norm = torch.nn.LayerNorm(d, dtype=torch.float64)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(d, dtype=torch.float64))
        norm.bias.copy_(torch.randn(d, dtype=torch.float64))
    norm.requires_grad_(False)
    w = torch.randn(d, dtype=torch.float64)
    x = torch.randn(d, dtype=torch.float64) * 3 + 1

    def f(v: torch.Tensor) -> torch.Tensor:
        return (w * norm(v)).sum()

    x1 = x.clone().requires_grad_(True)
    (full,) = torch.autograd.grad(f(x1), x1)
    x2 = x.clone().requires_grad_(True)
    centered = x2 - x2.mean()
    sigma = torch.sqrt(centered.pow(2).mean() + norm.eps).detach()  # the scale, held
    (held,) = torch.autograd.grad((w * (centered / sigma * norm.weight + norm.bias)).sum(), x2)
    direction = torch.randn(d, dtype=torch.float64)
    eps = 1e-6
    numeric = (f(x + eps * direction) - f(x - eps * direction)) / (2 * eps)
    assert float((full * direction).sum()) == pytest.approx(float(numeric), rel=1e-6)
    assert abs(float((held * direction).sum()) - float(numeric)) > 1e-3 * abs(float(numeric))
    assert abs(float((full * x).sum())) < 1e-6 * float(full.norm() * x.norm())
    assert abs(float((held * x).sum())) > 1e-2 * float(held.norm() * x.norm())


class _Nudge(EdgePatcher):
    """Adds a fixed tensor to one receiver's own input copy, and nothing else anywhere."""

    def __init__(self, receiver: str, layer: int, value: torch.Tensor) -> None:
        self.target, self.value = (receiver, layer), value

    def sent(self, node: str, layer: int, output: torch.Tensor) -> None:
        pass

    def update(self, receiver: str, layer: int) -> torch.Tensor | None:
        return self.value if (receiver, layer) == self.target else None


@pytest.mark.parametrize("arch", ["gpt2", "gpt_neox"])
def test_the_gradient_at_each_receiver_s_own_input_matches_finite_differences(
    smooth_backend: Any, project: Any, arch: str
) -> None:
    """Nudge one receiver's own copy of the residual stream (a head's query, key or value input,
    an MLP's input, the logits'): the logit difference changes by the gradient's projection on the
    nudge, with the normalization recomputing its scale from the nudged copy. Along the copy itself
    nothing changes, and the gradient says so."""
    backend = smooth_backend(arch)
    prompts = _prompts(backend, project)
    group = group_by_length(prompts)[0]
    tokens = group.corrupt[:6]
    idx = group.members[:6]
    answers = torch.tensor([prompts[i].answer_id for i in idx])
    distractors = torch.tensor([prompts[i].distractor_id for i in idx])
    rows = torch.arange(len(idx))

    def score(logits: torch.Tensor) -> torch.Tensor:
        last = logits[:, -1].float()
        return last[rows, answers] - last[rows, distractors]

    n = backend.info.n_layers
    grads = backend.input_gradients(tokens, score, list(range(n)), True)
    parallel = backend.info.extra["block_structure"] == "parallel"
    stream = backend.capture(
        tokens,
        [
            ("resid_pre", 1),
            ("resid_mid" if not parallel else "resid_pre", 0),
            ("resid_post", n - 1),
        ],
    )
    generator = torch.Generator().manual_seed(1)
    # Each step moves the copy by a fixed share of its own length at every position. On this
    # smooth model the metric is close to linear over 3% of it, and a five-point stencil cancels
    # the curvature up to fourth order; a much smaller step would measure float32's rounding in the
    # attention pattern instead of the slope.
    eps = 3e-2
    cases = [
        ("q", 1, grads.q[1], stream[("resid_pre", 1)]),
        ("k", 1, grads.k[1], stream[("resid_pre", 1)]),
        ("v", 1, grads.v[1], stream[("resid_pre", 1)]),
        ("mlp", 0, grads.mlp[0], stream[("resid_mid" if not parallel else "resid_pre", 0)]),
        ("logits", n, grads.logits, stream[("resid_post", n - 1)]),
    ]
    for receiver, layer, g, x in cases:
        assert g is not None
        direction = torch.randn(g.shape, generator=generator)
        length = x.norm(dim=-1, keepdim=True)
        if receiver in ("q", "k", "v"):
            direction[:, :, 1:] = 0  # one head's copy only
            length = length[:, :, None, :]
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12) * length

        def metric(
            sign: float,
            direction: torch.Tensor = direction,
            receiver: str = receiver,
            layer: int = layer,
        ) -> np.ndarray:
            nudge = _Nudge(receiver, layer, sign * eps * direction)
            logits = backend.edge_patch(tokens, nudge).double()
            return (logits[rows, answers] - logits[rows, distractors]).numpy()

        f = {k: metric(k) for k in (-2, -1, 1, 2)}
        numeric = (-f[2] + 8 * f[1] - 8 * f[-1] + f[-2]) / (12 * eps)
        analytic = (g.double() * direction.double()).flatten(1).sum(1).numpy()
        scale = max(1e-3, float(np.abs(numeric).max()))
        np.testing.assert_allclose(
            analytic, numeric, rtol=2e-3, atol=2e-3 * scale, err_msg=f"{receiver} {layer}"
        )
        # The copy's own direction: the normalization ignores its length (up to its epsilon, a
        # fraction of a percent of the variance of these small activations), so the gradient is
        # orthogonal to the copy, far more than to a random direction.
        copy = x[:, :, None, :].expand_as(g) if g.dim() == 4 else x
        read = g.double().norm(dim=-1) > 1e-12  # positions the last position's metric reads
        size = g.double().norm(dim=-1) * copy.double().norm(dim=-1)
        along = ((g.double() * copy.double()).sum(-1).abs() / size.clamp_min(1e-12))[read]
        noise = torch.randn(copy.shape, generator=generator, dtype=torch.float64)
        random = (g.double() * noise).sum(-1).abs()[read] / (
            g.double().norm(dim=-1) * noise.norm(dim=-1)
        )[read]
        assert float(along.max()) < 1e-2, receiver
        assert float(along.mean()) < 0.05 * float(random.mean()), receiver


# -- edge patching ---------------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ["gpt2", "gpt_neox"])
def test_patching_every_edge_is_the_source_run_and_none_the_receiver_run(
    tiny_backend: Any, arch_backend: Any, project: Any, arch: str
) -> None:
    backend = tiny_backend if arch == "gpt2" else arch_backend(arch)
    prompts = _prompts(backend, project)
    group = group_by_length(prompts)[0]
    clean, corrupt = group.clean, group.corrupt
    graph = build_graph(backend.info, GraphScope(layers=None, logits=True))
    keys = sender_keys(graph)
    source = backend.capture(clean, keys)
    weights = {layer: backend.head_output_weights(layer) for layer in range(backend.info.n_layers)}
    clean_logits = backend.logits(clean)[:, -1]
    corrupt_logits = backend.logits(corrupt)[:, -1]
    scale = float(clean_logits.abs().max())

    def patched(edges: np.ndarray) -> torch.Tensor:
        plan = PatchPlan.of(graph, edges, backend.device)
        return backend.edge_patch(corrupt, _Patcher(plan, source, weights))

    every = patched(np.arange(len(graph)))
    assert float((every - clean_logits).abs().max()) < 1e-5 * scale
    assert float((every - corrupt_logits).abs().max()) > 1e-2 * scale  # it did change something
    none = patched(np.array([], dtype=np.int64))
    assert float((none - corrupt_logits).abs().max()) < 1e-5 * scale
    # Every edge into the logits: the final normalization reads the source's residual stream.
    into_logits = np.flatnonzero(
        [graph.receivers[int(r)].kind == "logits" for r in graph.edge_receiver]
    )
    assert float((patched(into_logits) - clean_logits).abs().max()) < 1e-5 * scale


@pytest.mark.parametrize("arch", ["gpt2", "gpt_neox"])
def test_one_edge_patched_is_path_patching_of_that_edge(
    tiny_backend: Any, arch_backend: Any, project: Any, spec_factory: Any, arch: str
) -> None:
    """Edge patching and path patching (two passes, everything held) are separate code paths; for
    one edge they measure the same thing."""
    backend = tiny_backend if arch == "gpt2" else arch_backend(arch)
    prompts = _prompts(backend, project)
    graph = build_graph(backend.info, GraphScope(layers=None, logits=True))
    wanted = {"embeddings → L1 H2 q", "L0 H1 → L1 H0 k", "L0 H3 → L1 H3 v", "L0 MLP → L1 MLP"}
    wanted |= {"L1 H2 → logits", "embeddings → L0 MLP", "L1 MLP → logits"}
    if arch == "gpt2":
        wanted.add("L1 H0 → L1 MLP")  # a sequential layer's MLP reads its own heads
    chosen = [e for e in range(len(graph)) if graph.label(e) in wanted]
    assert len(chosen) == len(wanted)
    edges = [graph.edge(e).model_dump(mode="json") for e in chosen]
    receivers = list({json.dumps(e["receiver"]): e["receiver"] for e in edges}.values())
    path = spec_factory(
        experiment={
            "kind": "path_patching",
            "direction": "clean_to_corrupt",
            "receivers": receivers,
            "freeze_mlps": True,
        },
        scope={"kind": "edges", "edges": edges},
    )
    measured = run_experiment(path, backend, prompts)
    assert [s.label for s in measured.sites] == [graph.label(e) for e in chosen]

    from logogram.edges import patched_forward
    from logogram.engine import make_scorer

    scorer = make_scorer(path, prompts)
    weights = {layer: backend.head_output_weights(layer) for layer in range(backend.info.n_layers)}
    for row, e in enumerate(chosen):
        values = np.zeros(len(prompts))
        for group in group_by_length(prompts):
            source = backend.capture(group.clean, sender_keys(graph))
            plan = PatchPlan.of(graph, np.array([e]), backend.device)
            forward = patched_forward(backend, plan, source, weights)
            values[group.members] = scorer.score(forward, group.corrupt, group.members).metric
        np.testing.assert_allclose(
            values, measured.patched[row], rtol=1e-4, atol=1e-4, err_msg=graph.label(e)
        )


# -- estimates -------------------------------------------------------------------------------------


def test_no_difference_estimates_no_effect(
    tiny_backend: Any, project: Any, spec_factory: Any
) -> None:
    result = run_experiment(
        edge_spec(spec_factory),
        tiny_backend,
        _prompts(tiny_backend, project),
        receiver_override="corrupt",
        source_override="corrupt",
    )
    assert result.measure == "estimate"
    np.testing.assert_allclose(result.delta, 0.0, atol=1e-12)


@pytest.mark.parametrize("arch", ["gpt2", "gpt_neox"])
def test_each_estimate_is_the_slope_of_patching_its_edge(
    smooth_backend: Any, project: Any, spec_factory: Any, arch: str
) -> None:
    """Patch a growing share α of one edge's sender difference into its receiver: the metric's
    slope at α = 0, by finite differences of real edge-patched runs, is the edge's estimate, for
    every edge of the graph and every prompt."""
    from logogram.edges import patched_forward
    from logogram.engine import make_scorer

    backend = smooth_backend(arch)
    prompts = _prompts(backend, project)[:6]
    spec = edge_spec(spec_factory)
    estimated = run_experiment(spec, backend, prompts)
    graph = build_graph(backend.info, GraphScope(layers=None, logits=True))
    scorer = make_scorer(spec, prompts)
    weights = {layer: backend.head_output_weights(layer) for layer in range(backend.info.n_layers)}
    keys = sender_keys(graph)
    slope = np.zeros(estimated.delta.shape)
    step = 0.05
    for group in group_by_length(prompts):
        clean, corrupt = backend.capture(group.clean, keys), backend.capture(group.corrupt, keys)

        def metric(
            e: int, alpha: float, clean: Any = clean, corrupt: Any = corrupt, group: Any = group
        ) -> Any:
            # The sender's output moved a share alpha of the way to the source prompt's: the
            # patch adds alpha times its difference into the receiver's own input only.
            source = {k: corrupt[k] + alpha * (clean[k] - corrupt[k]) for k in keys}
            plan = PatchPlan.of(graph, np.array([e]), backend.device)
            forward = patched_forward(backend, plan, source, weights)
            return scorer.score(forward, group.corrupt, group.members).metric

        for e in range(len(graph)):
            f = {k: metric(e, k * step) for k in (-2, -1, 1, 2)}
            slope[e, group.members] = (-f[2] + 8 * f[1] - 8 * f[-1] + f[-2]) / (12 * step)
    scale = float(np.abs(slope).max())
    np.testing.assert_allclose(estimated.delta, slope, rtol=1e-2, atol=1e-3 * scale)


def test_integrated_gradients_average_the_gradient_over_steps(
    tiny_backend: Any, project: Any, spec_factory: Any
) -> None:
    prompts = _prompts(tiny_backend, project)
    one = run_experiment(edge_spec(spec_factory), tiny_backend, prompts)
    ig = run_experiment(
        edge_spec(spec_factory, method="integrated_gradients", steps=4), tiny_backend, prompts
    )
    assert ig.extra["edges"]["method"] == "integrated_gradients"
    assert np.isfinite(ig.delta).all()
    assert not np.allclose(ig.delta, one.delta)


def test_continuations_of_several_tokens_are_refused(
    tiny_backend: Any, project: Any, spec_factory: Any
) -> None:
    from logogram.datasets import write_dataset
    from logogram.sites import ScopeError

    records = load_dataset(project.datasets_dir / "ioi.jsonl")
    for r in records:
        r.answer = f"{r.answer}{r.distractor}"
    write_dataset(project.datasets_dir / "long.jsonl", records)
    data = edge_spec(spec_factory).model_dump(mode="json")
    data["dataset"]["path"] = "datasets/long.jsonl"
    data["metric"] = {"kind": "logprob", "normalization": "dataset_gap"}
    spec = Spec.model_validate(data)
    prompts = prepare_prompts(
        tiny_backend, load_dataset(project.datasets_dir / "long.jsonl"), True, continuations=True
    )
    with pytest.raises(ScopeError, match="single tokens"):
        run_experiment(spec, tiny_backend, prompts)


def test_a_graph_too_large_to_keep_is_refused(
    tiny_backend: Any, project: Any, spec_factory: Any, monkeypatch: Any
) -> None:
    prompts = _prompts(tiny_backend, project)
    monkeypatch.setattr("logogram.edges.MAX_EDGES", 50)
    with pytest.raises(EngineError, match="Choose fewer layers"):
        run_experiment(edge_spec(spec_factory), tiny_backend, prompts)
    monkeypatch.setattr("logogram.edges.MAX_EDGES", 100_000)
    monkeypatch.setattr("logogram.edges.MAX_VALUES", 100)
    with pytest.raises(EngineError, match=r"dataset\.limit"):
        run_experiment(edge_spec(spec_factory), tiny_backend, prompts)
    monkeypatch.setattr("logogram.edges.MAX_VALUES", 10_000_000)
    monkeypatch.setattr("logogram.edges.free_memory", lambda device: 1000)
    with pytest.raises(EngineError, match=r"Lower execution\.batch_size"):
        run_experiment(edge_spec(spec_factory), tiny_backend, prompts)


# -- the curve -------------------------------------------------------------------------------------


def test_random_orders_are_seeded_shuffles() -> None:
    a, b = random_orders(500, 3, 2)
    assert sorted(a.tolist()) == list(range(500))
    assert not np.array_equal(a, b)
    again = random_orders(500, 3, 2)
    assert np.array_equal(a, again[0]) and np.array_equal(b, again[1])


@pytest.mark.parametrize("patch", ["chosen", "rest"])
def test_the_curve_runs_from_no_edge_to_every_edge(
    tiny_backend: Any, project: Any, spec_factory: Any, patch: str
) -> None:
    spec = edge_spec(spec_factory, curve={**CURVE, "patch": patch, "counts": [1, 3, 10, 30, 500]})
    result = run_experiment(spec, tiny_backend, _prompts(tiny_backend, project))
    curve = result.extra["edges"]["curve"]
    assert curve["counts"] == [1, 3, 10, 30]
    assert curve["dropped"] == [500]
    assert any("500" in w for w in result.warnings)
    every = curve["every"]["faithfulness"]["mean"]
    # Every edge patched is the source run: faithfulness 1 when the chosen edges are patched,
    # 0 when every edge but the chosen ones is.
    assert every == pytest.approx(1.0 if patch == "chosen" else 0.0, abs=1e-4)
    assert curve["source_difference"] < 1e-4
    assert len(curve["per_prompt"]["strongest"]) == 4
    assert len(curve["per_prompt"]["random"]) == 4 * 2
    assert all(p["draws"] and len(p["draws"]) == 2 for p in curve["random"])


# -- statistics for many sites ---------------------------------------------------------------------


@pytest.mark.parametrize("normalization", ["dataset_gap", "prompt_gap"])
def test_statistics_in_chunks_are_those_of_one_pass(monkeypatch: Any, normalization: str) -> None:
    rng = np.random.default_rng(0)
    S, n = 300, 9
    delta = rng.normal(size=(S, n))
    delta[5] = 0.0  # a site with no effect at all: tied resamples
    delta[7, 3] = np.nan  # a site with an undefined value: no band
    receiver = rng.normal(size=n)
    reference = receiver + rng.uniform(1, 2, size=n)
    nan = np.full((S, n), np.nan)
    counts = resample_counts(n, 200, 4)
    args = dict(
        patched=receiver[None, :] + delta,
        patched_prob=nan,
        patched_pref=nan,
        receiver=receiver,
        reference=reference,
        receiver_prob=np.full(n, np.nan),
        receiver_pref=np.full(n, np.nan),
        normalization=normalization,
        counts=counts,
        ci=0.95,
        delta=delta,
        family=True,
    )
    whole = compute_site_stats(**args)
    monkeypatch.setattr("logogram.stats.CHUNK_VALUES", 200 * 7)  # 7 sites a chunk
    chunked = compute_site_stats(**args, keep_boot=False)
    assert chunked.effect_boot is None
    for name in (
        "effect_mean", "effect_sd", "effect_lo", "effect_hi", "delta_mean", "delta_lo",
        "delta_hi", "opposite_sign", "band_lo", "band_hi", "q",
    ):  # fmt: skip
        np.testing.assert_array_equal(getattr(chunked, name), getattr(whole, name), err_msg=name)
    assert chunked.band_level == whole.band_level


# -- runs ------------------------------------------------------------------------------------------


def test_a_run_keeps_every_edge_and_lists_the_strongest(
    tiny_backend: Any, project: Any, spec_factory: Any, monkeypatch: Any
) -> None:
    monkeypatch.setattr("logogram.edges.SUMMARY_EDGES", 20)
    spec = edge_spec(spec_factory, curve=CURVE)
    outcome = run_spec(spec, project, backend=tiny_backend)
    assert outcome.status == "finished", outcome.manifest.get("error")
    summary = json.loads((outcome.folder / "summary.json").read_text(encoding="utf-8"))
    edges = summary["edges"]
    assert edges["n_edges"] == 110 and edges["n_senders"] == 11 and edges["n_receivers"] == 27
    assert summary["layout"]["kind"] == "edges"
    assert summary["measure"] == "estimate"
    assert "scale recomputed" in summary["metric"]["description"]
    assert len(summary["sites"]) == summary["n_sites"] == 20
    table = pq.read_table(outcome.folder / EDGES_FILE).to_pylist()
    assert len(table) == 110
    strongest = sorted(table, key=lambda r: -abs(r["effect_mean"]))[:20]
    assert {r["edge"] for r in strongest} == {s["index"] for s in summary["sites"]}
    for site in summary["sites"]:
        row = table[site["index"]]
        assert row["label"] == site["label"]
        assert row["effect_mean"] == site["effect"]["mean"]
        assert row["q"] == site["q"]
    results = pq.read_table(outcome.folder / "results.parquet")
    assert results.num_rows == 110 * summary["n_prompts"]
    assert set(results.column("kind").to_pylist()) == {"edge"}
    assert edges["curve"]["every"]["faithfulness"]["mean"] == pytest.approx(1.0, abs=1e-4)
    cost = edges["cost"]
    assert cost["edges"] == 110
    # Per batch: the senders captured on both prompts and one gradient, then the curve's circuits
    # (each number of edges, strongest and in each random draw, and every edge).
    assert (cost["forward_per_batch"], cost["backward_per_batch"]) == (3, 1)
    kept = len(edges["curve"]["counts"])
    assert cost["patched_per_batch"] == kept * (1 + CURVE["control"]["draws"]) + 1
    # The corrections count every edge, not only those listed.
    alpha = 1 - spec.statistics.ci
    assert edges["corrections"] == {
        "edges": 110,
        "single": sum(1 for r in table if r["effect_lo"] > 0 or r["effect_hi"] < 0),
        "banded": sum(1 for r in table if r["band_lo"] > 0 or r["band_hi"] < 0),
        "discoveries": sum(1 for r in table if r["q"] < alpha),
    }


def test_the_same_spec_twice_gives_identical_results(
    tiny_backend: Any, project: Any, spec_factory: Any
) -> None:
    spec = edge_spec(spec_factory, method="integrated_gradients", steps=3, curve=CURVE)
    first = run_spec(spec, project, backend=tiny_backend)
    second = run_spec(spec, project, backend=tiny_backend)
    from logogram.results import largest_change

    a = pq.read_table(first.folder / "results.parquet")
    b = pq.read_table(second.folder / "results.parquet")
    assert largest_change(a, b) == 0.0  # NaN where nothing is measured matches itself
    a, b = pq.read_table(first.folder / EDGES_FILE), pq.read_table(second.folder / EDGES_FILE)
    assert a.to_pylist() == b.to_pylist()
    a = json.loads((first.folder / "summary.json").read_text(encoding="utf-8"))
    b = json.loads((second.folder / "summary.json").read_text(encoding="utf-8"))
    assert a["edges"]["curve"] == b["edges"]["curve"]
    assert a["sites"] == b["sites"]


def test_a_run_stopped_during_the_curve_keeps_every_edge(
    tiny_backend: Any, project: Any, spec_factory: Any
) -> None:
    spec = edge_spec(spec_factory, curve=CURVE)
    cancel = threading.Event()

    def on_event(kind: str, data: dict[str, Any]) -> None:
        # The estimates take one step per batch; the curve's passes come after.
        if kind == "progress" and data["done"] == 2:
            cancel.set()

    outcome = run_spec(spec, project, backend=tiny_backend, on_event=on_event, cancel=cancel)
    assert outcome.status == "cancelled"
    summary = json.loads((outcome.folder / "summary.json").read_text(encoding="utf-8"))
    partial = summary["partial"]
    assert partial["unit"] == "prompts"
    assert any("faithfulness curve" in item for item in partial["left_out"])
    assert summary["edges"].get("curve") is None
    assert pq.read_table(outcome.folder / EDGES_FILE).num_rows == 110


def test_the_strongest_edges_are_verified_by_path_patching(
    tiny_backend: Any, project: Any, spec_factory: Any
) -> None:
    from logogram.compare import compare_runs
    from logogram.report import run_report

    estimated = run_spec(edge_spec(spec_factory, curve=CURVE), project, backend=tiny_backend)
    summary = json.loads((estimated.folder / "summary.json").read_text(encoding="utf-8"))
    check = edge_verification_spec(Spec.from_path(estimated.folder / "spec.json"), summary, 6)
    assert check.experiment.kind == "path_patching" and check.experiment.freeze_mlps
    assert check.scope.kind == "edges" and len(check.scope.edges) == 6
    strongest = sorted(summary["sites"], key=lambda s: (-abs(s["effect"]["mean"]), s["index"]))
    assert [e.model_dump(mode="json") for e in check.scope.edges] == [
        s["edge"] for s in strongest[:6]
    ]
    verified = run_spec(check, project, backend=tiny_backend)
    assert verified.status == "finished", verified.manifest.get("error")
    measured = json.loads((verified.folder / "summary.json").read_text(encoding="utf-8"))
    assert measured["layout"]["kind"] == "edges"
    table = verification_table(summary, measured)
    assert table["n"] == 6
    assert [r["label"] for r in table["rows"]] == [s["label"] for s in strongest[:6]]
    assert table["spearman"] is not None
    comparison = compare_runs(project, estimated.run_id, verified.run_id)
    assert comparison["n_common"] == 6
    page = run_report(
        Spec.from_path(estimated.folder / "spec.json"),
        summary,
        estimated.manifest,
        estimated.run_id,
        verification=table,
    )
    assert "Estimates against path patching" in page
    assert "How much of the behavior the strongest edges carry" in page
    assert "<svg" in page and "edges.parquet" in page


def test_the_command_line_runs_and_verifies_edges(
    tiny_backend: Any, project: Any, spec_factory: Any, monkeypatch: Any, tmp_path: Any
) -> None:
    from typer.testing import CliRunner

    from logogram.cli import app as cli

    monkeypatch.setattr(
        "logogram.runner.default_provider", lambda on_event=None: lambda ref: tiny_backend
    )
    monkeypatch.setattr("logogram.edges.SUMMARY_EDGES", 20)
    path = project.root / "edge-spec.json"
    path.write_text(edge_spec(spec_factory, curve=CURVE).to_json(), encoding="utf-8")
    result = CliRunner().invoke(cli, ["run", str(path), "--project", str(project.root)])
    assert result.exit_code == 0, result.output
    assert "110 edges (11 senders, 27 receivers)" in result.output
    assert "Faithfulness, the chosen edges patched" in result.output
    run_id = next(r.id for r in project.list_runs() if r.status == "finished")
    # One row per edge of the graph, though the summary lists only the strongest 20.
    out = tmp_path / "edges.csv"
    result = CliRunner().invoke(
        cli, ["export", run_id, "--sites", "-o", str(out), "--project", str(project.root)]
    )
    assert result.exit_code == 0, result.output
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1 + 110
    result = CliRunner().invoke(
        cli, ["verify", run_id, "--top", "4", "--project", str(project.root)]
    )
    assert result.exit_code == 0, result.output
    assert "Estimate against path patching, 4 edges" in result.output
    monkeypatch.setattr("logogram.edges.VERIFY_MAX", 3)
    result = CliRunner().invoke(
        cli, ["verify", run_id, "--top", "4", "--project", str(project.root)]
    )
    assert result.exit_code == 0, result.output
    assert "at most the strongest 3 are verified" in result.output
    assert "Estimate against path patching, 3 edges" in result.output


def test_the_app_runs_edges_and_shows_their_verification(
    tiny_backend: Any, tmp_path: Any, monkeypatch: Any, spec_factory: Any
) -> None:
    import time

    from fastapi.testclient import TestClient

    from logogram.server.app import create_app
    from logogram.server.security import SecurityConfig

    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    application = create_app(SecurityConfig(token="t", port=8765), serve_web=False)
    application.state.logogram.backend = tiny_backend
    client = TestClient(application, base_url="http://127.0.0.1:8765")
    auth = {"authorization": "Bearer t", "origin": "http://127.0.0.1:8765"}
    made = client.post(
        "/api/projects", json={"name": "Edges", "parent": str(tmp_path)}, headers=auth
    )
    assert made.status_code == 200, made.text
    client.post("/api/datasets/ioi", json={"name": "ioi", "n": 8, "seed": 3}, headers=auth)

    def finish(run_id: str) -> None:
        for _ in range(1200):
            listing = client.get("/api/runs", headers=auth).json()
            status = next(r["status"] for r in listing if r["id"] == run_id)
            if status in ("finished", "failed", "cancelled"):
                assert status == "finished", listing
                return
            time.sleep(0.05)
        raise AssertionError("the run did not finish")

    spec = edge_spec(spec_factory, curve=CURVE).model_dump(mode="json")
    run_id = client.post("/api/runs", json={"spec": spec}, headers=auth).json()["run_id"]
    finish(run_id)
    assert client.get(f"/api/runs/{run_id}/edge-verification", headers=auth).json() is None
    figure = client.get(f"/api/runs/{run_id}/figure.svg", headers=auth)
    assert figure.status_code == 409 and "report" in figure.text
    sites = client.get(f"/api/runs/{run_id}/sites.csv", headers=auth)
    assert sites.status_code == 200 and len(sites.text.strip().splitlines()) == 111
    started = client.post(f"/api/runs/{run_id}/verify", json={"top": 5}, headers=auth).json()
    finish(started["run_id"])
    table = client.get(f"/api/runs/{run_id}/edge-verification", headers=auth).json()
    assert table["run"] == started["run_id"] and table["n"] == 5
    report = client.get(f"/api/runs/{run_id}/report.html", headers=auth).text
    assert "Estimates against path patching" in report
