"""ACDC (logogram.acdc): the order edges are tested in, the rule that keeps them, the circuit's
faithfulness against the faithfulness curve's for the same edges, reruns, refusals, and what a
run leaves for the app, the command line and the Python API. What ACDC finds on the toy models,
whose circuits are known, is checked in test_toys.py."""

from __future__ import annotations

import json
import threading
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest

from logogram.backends.base import BackendError, ModelInfo
from logogram.datasets import load_dataset
from logogram.edges import EDGES_FILE, Graph, build_graph
from logogram.engine import EngineError, run_experiment
from logogram.prompts import prepare_prompts
from logogram.runner import run_spec
from logogram.spec import GraphScope, Spec

GRAPH = {"kind": "graph", "layers": None, "logits": True}


def acdc_spec(spec_factory: Any, scope: dict[str, Any] | None = None, **experiment: Any) -> Spec:
    return spec_factory(
        experiment={
            "kind": "acdc",
            "direction": "corrupt_to_clean",
            "threshold": 0.05,
            "rule": "toward_source",
            **experiment,
        },
        scope=scope or GRAPH,
    )


def _summary(outcome: Any) -> dict[str, Any]:
    assert outcome.status == "finished", outcome.manifest.get("error")
    return json.loads((outcome.folder / "summary.json").read_text(encoding="utf-8"))


def _info(n_layers: int, n_heads: int, structure: str, n_kv: int | None = None) -> ModelInfo:
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
        extra={"block_structure": structure, "n_key_value_heads": n_kv or n_heads},
    )


# -- the spec ------------------------------------------------------------------------------------


def test_the_spec_states_every_choice(spec_factory: Any) -> None:
    spec = acdc_spec(spec_factory)
    assert spec.experiment.kind == "acdc"
    assert Spec.model_validate(spec.model_dump(mode="json")) == spec
    data = spec.model_dump(mode="json")
    for broken, message in (
        ({"scope": {"kind": "heads", "position": {"kind": "all"}}}, "graph scope"),
        ({"experiment": {**data["experiment"], "threshold": 0}}, "greater than 0"),
        ({"experiment": {**data["experiment"], "threshold": float("inf")}}, "finite"),
        ({"experiment": {**data["experiment"], "rule": "absolute"}}, "toward_source"),
    ):
        with pytest.raises(ValueError, match=message):
            Spec.model_validate({**data, **broken})
    for field in ("direction", "threshold", "rule"):  # no default for any of them
        without = {k: v for k, v in data["experiment"].items() if k != field}
        with pytest.raises(ValueError, match=field):
            Spec.model_validate({**data, "experiment": without})


# -- the order -----------------------------------------------------------------------------------


def _reads(graph: Graph, r: int) -> set[tuple[str, int, int | None]]:
    """The nodes whose output receiver ``r``'s input feeds: a head's query its head; a key or
    value every head of its group; an MLP's input that MLP; the logits none."""
    receiver = graph.receivers[r]
    if receiver.kind == "logits":
        return set()
    if receiver.kind == "mlp":
        return {("mlp", receiver.layer, None)}
    assert receiver.head is not None
    if receiver.kind == "q":
        return {("head", receiver.layer, receiver.head)}
    group = graph.n_heads // graph.n_kv_heads
    heads = range(receiver.head * group, (receiver.head + 1) * group)
    return {("head", receiver.layer, h) for h in heads}


@pytest.mark.parametrize(
    "info, layers, logits",
    [
        (_info(3, 2, "sequential"), None, True),
        (_info(3, 2, "parallel"), None, True),
        (_info(4, 2, "sequential"), [1, 3], False),
        (_info(3, 4, "sequential", n_kv=2), None, True),
    ],
)
def test_every_receiver_is_tested_after_every_receiver_its_output_reaches(
    info: ModelInfo, layers: list[int] | None, logits: bool
) -> None:
    """ACDC tests the edges in the reverse of the graph's order. For every receiver, the edges
    out of the node it feeds are all tested before any edge into it: by then, whether the node
    matters to the circuit is settled (a reverse topological order)."""
    graph = build_graph(info, GraphScope(layers=layers, logits=logits))
    step = {e: len(graph) - 1 - e for e in range(len(graph))}  # ACDC's test order
    into: dict[int, list[int]] = {}
    out_of: dict[tuple[str, int, int | None], list[int]] = {}
    for e in range(len(graph)):
        into.setdefault(int(graph.edge_receiver[e]), []).append(e)
        sender = graph.senders[int(graph.edge_sender[e])]
        out_of.setdefault((sender.kind, sender.layer, sender.head), []).append(e)
    checked = 0
    for r, edges in into.items():
        first_in = min(step[e] for e in edges)
        for node in _reads(graph, r):
            for e in out_of.get(node, []):
                assert step[e] < first_in, (graph.label(e), graph.receivers[r].label)
                checked += 1
    assert checked > 0
    # Within a receiver, from the latest sender back to the embeddings.
    for edges in into.values():
        senders = [graph.senders[int(graph.edge_sender[e])] for e in sorted(edges, key=step.get)]
        assert senders[-1].kind == "embed"
        assert [s.layer for s in senders] == sorted((s.layer for s in senders), reverse=True)


# -- the rule ------------------------------------------------------------------------------------


@pytest.mark.parametrize("rule", ["toward_source", "either_way"])
def test_the_threshold_keeps_the_edges_the_rule_says(
    tiny_backend: Any, project: Any, spec_factory: Any, rule: str
) -> None:
    """Each edge's recorded change (its removal's, from the circuit as it stood) decides it:
    kept when the change counted reaches the threshold, removed when it falls short; counted
    toward the source (the sign of source − receiver), or either way. Every edge is in
    edges.parquet, with when it was tested."""
    threshold = 0.05
    outcome = run_spec(acdc_spec(spec_factory, rule=rule), project, backend=tiny_backend)
    summary = _summary(outcome)
    acdc = summary["acdc"]
    table = pq.read_table(outcome.folder / EDGES_FILE).to_pylist()
    assert len(table) == acdc["n_edges"] == 110
    # corrupt_to_clean: the clean prompts run, removed edges carry corrupt values.
    assert summary["receiver"] == "clean" and summary["reference"] == "corrupt"
    assert acdc["source_sign"] == (1 if summary["baseline"]["gap"]["mean"] > 0 else -1)
    kept = set(acdc["kept"])
    assert 0 < len(kept) < len(table)  # the threshold splits this model's edges
    for row in table:
        assert row["step"] == acdc["n_edges"] - 1 - row["edge"]
        assert row["kept"] == (row["edge"] in kept)
        change = row["delta_mean"]
        counted = acdc["source_sign"] * change if rule == "toward_source" else abs(change)
        if abs(counted - threshold) > 1e-9:
            assert row["kept"] == (counted >= threshold), row["label"]
    assert acdc["n_kept"] == len(kept)
    # The summary lists the circuit's edges first.
    assert kept <= {s["index"] for s in summary["sites"]}


def test_a_threshold_above_every_change_removes_every_edge(
    tiny_backend: Any, project: Any, spec_factory: Any
) -> None:
    """With every edge removed, every receiver reads the source prompt's residual stream: the
    circuit is the source run, faithfulness 0."""
    outcome = run_spec(acdc_spec(spec_factory, threshold=1e6), project, backend=tiny_backend)
    acdc = _summary(outcome)["acdc"]
    assert acdc["n_kept"] == 0 and acdc["kept"] == []
    assert acdc["faithfulness"]["mean"] == pytest.approx(0.0, abs=1e-4)
    corrupt = _summary(outcome)["per_prompt"]["corrupt_metric"]
    np.testing.assert_allclose(acdc["per_prompt"]["circuit"], corrupt, atol=1e-5)
    np.testing.assert_allclose(
        acdc["per_prompt"]["full"], _summary(outcome)["per_prompt"]["clean_metric"], atol=1e-5
    )


@pytest.mark.parametrize("arch", ["llama", "qwen2"])
def test_keys_and_values_shared_by_query_heads_are_tested_once(
    arch_backend: Any, project: Any, spec_factory: Any, arch: str
) -> None:
    """With grouped-query attention, an edge into a key or value reaches a key/value head ("L1
    KV0 k"), read by every query head of its group. Removing every edge still makes the source
    run, and a circuit of some of them sits between the two runs."""
    from logogram.edges import count_edges

    backend = arch_backend(arch)  # four query heads, two key/value heads

    def on(spec: Spec) -> Spec:
        data = spec.model_dump(mode="json")
        data["model"]["id"] = f"tiny-{arch}"
        return Spec.model_validate(data)

    every = run_spec(on(acdc_spec(spec_factory, threshold=1e6)), project, backend=backend)
    summary = _summary(every)
    acdc = summary["acdc"]
    assert acdc["n_edges"] == count_edges(2, 4, False, None, True, n_kv_heads=2) == 82
    assert acdc["n_kept"] == 0
    np.testing.assert_allclose(
        acdc["per_prompt"]["circuit"], summary["per_prompt"]["corrupt_metric"], atol=1e-4
    )
    table = pq.read_table(every.folder / EDGES_FILE).to_pylist()
    shared = [r for r in table if r["receiver"] in ("k", "v")]
    assert shared and all(" KV" in r["label"] and r["receiver_head"] < 2 for r in shared)
    some = _summary(run_spec(on(acdc_spec(spec_factory)), project, backend=backend))["acdc"]
    assert 0 < some["n_kept"] < some["n_edges"]


# -- the circuit ---------------------------------------------------------------------------------


def test_the_circuit_s_faithfulness_is_the_curve_s_for_its_edges(
    tiny_backend: Any, project: Any, spec_factory: Any
) -> None:
    """The circuit's faithfulness is what the faithfulness curve gives for the same edges, every
    other edge carrying the source prompt's values: the same edge-patched passes, bit for bit,
    and the same bootstrap."""
    from logogram.edges import faithfulness_curve, sender_keys
    from logogram.engine import compute_baselines, make_scorer
    from logogram.prompts import group_by_length

    spec = acdc_spec(spec_factory)
    outcome = run_spec(spec, project, backend=tiny_backend)
    acdc = _summary(outcome)["acdc"]
    kept = acdc["kept"]
    assert kept
    curve_spec = Spec.model_validate(
        {
            **spec.model_dump(mode="json"),
            "experiment": {
                "kind": "edge_attribution",
                "direction": "corrupt_to_clean",
                "method": "gradient",
                "steps": None,
                "curve": {
                    "counts": [len(kept)],
                    "patch": "rest",
                    "control": {"seed": 0, "draws": 1},
                },
            },
        }
    )
    prompts = prepare_prompts(tiny_backend, load_dataset(project.datasets_dir / "ioi.jsonl"), True)
    groups = group_by_length(prompts)
    scorer = make_scorer(curve_spec, prompts)
    baselines = compute_baselines(tiny_backend, prompts, groups, 16, None, scorer)
    graph = build_graph(tiny_backend.info, GraphScope(layers=None, logits=True))
    estimate = np.zeros((len(graph), len(prompts)))
    estimate[kept] = 1.0  # the circuit's edges are the strongest
    weights = {layer: tiny_backend.head_output_weights(layer) for layer in range(2)}
    curve, _ = faithfulness_curve(
        curve_spec,
        tiny_backend,
        prompts,
        groups,
        scorer,
        baselines,
        graph,
        estimate,
        "clean",
        "corrupt",
        weights,
        sender_keys(graph),
    )
    assert curve["per_prompt"]["strongest"][0] == acdc["per_prompt"]["circuit"]
    assert curve["strongest"][0]["faithfulness"] == acdc["faithfulness"]


def test_the_same_spec_twice_gives_identical_results(
    tiny_backend: Any, project: Any, spec_factory: Any
) -> None:
    from logogram.results import largest_change

    spec = acdc_spec(spec_factory, rule="either_way")
    first = run_spec(spec, project, backend=tiny_backend)
    second = run_spec(spec, project, backend=tiny_backend)
    a = pq.read_table(first.folder / "results.parquet")
    b = pq.read_table(second.folder / "results.parquet")
    assert largest_change(a, b) == 0.0
    a, b = pq.read_table(first.folder / EDGES_FILE), pq.read_table(second.folder / EDGES_FILE)
    assert a.to_pylist() == b.to_pylist()
    assert _summary(first)["acdc"] == _summary(second)["acdc"]
    assert _summary(first)["sites"] == _summary(second)["sites"]


# -- refusals ------------------------------------------------------------------------------------


def test_models_without_measured_edges_are_refused(
    arch_backend: Any, project: Any, spec_factory: Any
) -> None:
    backend = arch_backend("olmo2")  # it normalizes after combining its heads
    prompts = prepare_prompts(backend, load_dataset(project.datasets_dir / "ioi.jsonl"), True)
    with pytest.raises(BackendError, match="don't add up"):
        run_experiment(acdc_spec(spec_factory), backend, prompts)


def test_a_run_that_cant_fit_or_has_no_direction_is_refused(
    tiny_backend: Any, project: Any, spec_factory: Any, monkeypatch: Any
) -> None:
    prompts = prepare_prompts(tiny_backend, load_dataset(project.datasets_dir / "ioi.jsonl"), True)
    monkeypatch.setattr("logogram.edges.MAX_EDGES", 50)
    with pytest.raises(EngineError, match="Choose fewer layers"):
        run_experiment(acdc_spec(spec_factory), tiny_backend, prompts)
    monkeypatch.setattr("logogram.edges.MAX_EDGES", 100_000)
    monkeypatch.setattr("logogram.edges.free_memory", lambda device: 1000)
    with pytest.raises(EngineError, match=r"dataset\.limit"):
        run_experiment(acdc_spec(spec_factory), tiny_backend, prompts)
    monkeypatch.undo()
    # A prompt patched into itself has no gap: "toward the source" has no direction.
    with pytest.raises(EngineError, match="either_way"):
        run_experiment(
            acdc_spec(spec_factory, scope={"kind": "graph", "layers": [1], "logits": False}),
            tiny_backend,
            prompts,
            receiver_override="clean",
            source_override="clean",
        )


def test_continuations_of_several_tokens_are_refused(
    tiny_backend: Any, project: Any, spec_factory: Any
) -> None:
    from logogram.datasets import write_dataset
    from logogram.sites import ScopeError

    records = load_dataset(project.datasets_dir / "ioi.jsonl")
    for r in records:
        r.answer = f"{r.answer}{r.distractor}"
    write_dataset(project.datasets_dir / "long.jsonl", records)
    data = acdc_spec(spec_factory).model_dump(mode="json")
    data["dataset"]["path"] = "datasets/long.jsonl"
    data["metric"] = {"kind": "logprob", "normalization": "dataset_gap"}
    prompts = prepare_prompts(
        tiny_backend, load_dataset(project.datasets_dir / "long.jsonl"), True, continuations=True
    )
    with pytest.raises(ScopeError, match="single tokens"):
        run_experiment(Spec.model_validate(data), tiny_backend, prompts)


def test_a_cancelled_run_keeps_nothing(tiny_backend: Any, project: Any, spec_factory: Any) -> None:
    """Until every edge has been tested, the edges left aren't ACDC's circuit."""
    cancel = threading.Event()

    def on_event(kind: str, data: dict[str, Any]) -> None:
        if kind == "progress" and data["done"] == 20:
            cancel.set()

    outcome = run_spec(
        acdc_spec(spec_factory), project, backend=tiny_backend, on_event=on_event, cancel=cancel
    )
    assert outcome.status == "cancelled" and outcome.summary is None
    for name in ("summary.json", "results.parquet", EDGES_FILE):
        assert not (outcome.folder / name).exists()


# -- the app, the command line and the Python API ------------------------------------------------


def test_the_plan_counts_the_passes_and_times_one(
    tiny_backend: Any, tmp_path: Any, monkeypatch: Any, spec_factory: Any
) -> None:
    """Before a run, the app is told the graph's size, how many passes ACDC takes (one per edge
    and batch, beside the unpatched baselines) and how long they may take here, from one pass
    timed; an edge attribution spec is planned as before."""
    from fastapi.testclient import TestClient

    from logogram.server.app import create_app
    from logogram.server.security import SecurityConfig

    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    application = create_app(SecurityConfig(token="t", port=8765), serve_web=False)
    application.state.logogram.backend = tiny_backend
    client = TestClient(application, base_url="http://127.0.0.1:8765")
    auth = {"authorization": "Bearer t", "origin": "http://127.0.0.1:8765"}
    made = client.post(
        "/api/projects", json={"name": "ACDC", "parent": str(tmp_path)}, headers=auth
    )
    assert made.status_code == 200, made.text
    client.post("/api/datasets/ioi", json={"name": "ioi", "n": 8, "seed": 3}, headers=auth)
    spec = acdc_spec(spec_factory).model_dump(mode="json")
    plan = client.post("/api/edges/plan", json={"spec": spec}, headers=auth)
    assert plan.status_code == 200, plan.text
    plan = plan.json()
    assert plan["refusal"] is None and plan["n_edges"] == 110
    cost = plan["cost"]
    batches = cost["batches"]
    # Each side's baseline, the source prompts' senders, every edge kept, then one per edge.
    assert cost["forward"] == (3 + 1 + 110) * batches
    assert cost["patched_per_batch"] == 111 and cost["backward"] == 0
    assert cost["seconds_per_pass"] > 0 and cost["seconds"] == pytest.approx(
        cost["seconds_per_pass"] * cost["forward"]
    )
    assert "forward passes in all" in plan["text"] and "here" in plan["text"]
    monkeypatch.setattr("logogram.edges.MAX_EDGES", 50)
    refused = client.post("/api/edges/plan", json={"spec": spec}, headers=auth).json()
    assert "Choose fewer layers" in refused["refusal"]
    monkeypatch.setattr("logogram.edges.MAX_EDGES", 100_000)
    edges = {
        "kind": "edge_attribution",
        "direction": "clean_to_corrupt",
        "method": "gradient",
        "steps": None,
        "curve": None,
    }
    estimated = client.post(
        "/api/edges/plan", json={"spec": {**spec, "experiment": edges}}, headers=auth
    ).json()
    assert estimated["refusal"] is None and estimated["cost"]["backward"] > 0


def test_the_command_line_the_api_and_the_report_read_the_circuit(
    tiny_backend: Any, project: Any, spec_factory: Any, monkeypatch: Any
) -> None:
    from typer.testing import CliRunner

    import logogram as lg
    from logogram.cli import app as cli
    from logogram.report import methods_text, run_report

    monkeypatch.setattr(
        "logogram.runner.default_provider", lambda on_event=None: lambda ref: tiny_backend
    )
    path = project.root / "acdc-spec.json"
    path.write_text(acdc_spec(spec_factory).to_json(), encoding="utf-8")
    result = CliRunner().invoke(cli, ["run", str(path), "--project", str(project.root)])
    assert result.exit_code == 0, result.output
    assert "ACDC kept" in result.output and "of 110 edges" in result.output
    assert "The circuit's faithfulness" in result.output

    run_id = next(r.id for r in project.list_runs() if r.status == "finished")
    run = lg.load_run(project, run_id)
    circuit = run.circuit()
    summary = run.summary
    assert circuit["n_kept"] == len(circuit["edges"]) == summary["acdc"]["n_kept"]
    assert circuit["faithfulness"] == summary["acdc"]["faithfulness"]
    listed = {s["index"]: s for s in summary["sites"]}
    for edge in circuit["edges"]:
        assert edge["edge"] == listed[edge["index"]]["edge"]
        assert edge["label"] == listed[edge["index"]]["label"]
        assert edge["change"] == listed[edge["index"]]["effect"]["mean"]
    table = run.edges()
    assert int(table["kept"].sum()) == circuit["n_kept"]

    spec = Spec.from_path(run.folder / "spec.json")
    text = methods_text(spec, summary, run.manifest, run.id)
    assert "ACDC (automated circuit discovery)" in text and "The circuit kept" in text
    page = run_report(spec, summary, run.manifest, run.id)
    assert "The circuit ACDC found" in page and "Edges kept" in page


def test_check_robustness_reruns_with_another_threshold(spec_factory: Any) -> None:
    from logogram.robustness import robustness_spec

    spec = acdc_spec(spec_factory)
    variant, change = robustness_spec(
        spec, experiment={**spec.experiment.model_dump(mode="json"), "threshold": 0.005}
    )
    assert variant.experiment.threshold == 0.005
    assert change == "ACDC, threshold 0.005 on changes toward the source (corrupt → clean)"
