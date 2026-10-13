"""Path patching: whole effects where nothing lies between, held heads, and only upstream senders."""

from __future__ import annotations

import json

import numpy as np
import pytest

from logogram.datasets import load_dataset
from logogram.engine import run_experiment
from logogram.prompts import prepare_prompts
from logogram.runner import run_spec
from logogram.sites import ScopeError

TOL = 1e-4


def _prompts(backend, project):
    return prepare_prompts(backend, load_dataset(project.datasets_dir / "ioi.jsonl"), True)


def path(receivers, direction="corrupt_to_clean", freeze_mlps=False):
    return {
        "kind": "path_patching",
        "direction": direction,
        "receivers": receivers,
        "freeze_mlps": freeze_mlps,
    }


LOGITS = [{"kind": "logits"}]


@pytest.mark.parametrize("arch", ["gpt2", "llama", "gpt_neox"])
def test_from_the_last_layer_to_the_logits_a_path_is_the_whole_effect(
    tiny_backend, arch_backend, project, spec_factory, arch
):
    """Nothing but the last layer's MLP lies between a last-layer head and the logits, and nothing
    at all after the last MLP: with MLPs recomputed, the path is the component's whole effect."""
    backend = tiny_backend if arch == "gpt2" else arch_backend(arch)
    last = backend.info.n_layers - 1
    sites = [{"kind": "mlp_out", "layer": last, "position": {"kind": "all"}}] + [
        {"kind": "head", "layer": last, "head": h, "position": {"kind": "last"}} for h in range(2)
    ]
    scope = {"kind": "sites", "sites": sites}
    prompts = _prompts(backend, project)
    paths = run_experiment(spec_factory(experiment=path(LOGITS), scope=scope), backend, prompts)
    patched = run_experiment(
        spec_factory(
            experiment={"kind": "activation_patching", "direction": "corrupt_to_clean"}, scope=scope
        ),
        backend,
        prompts,
    )
    np.testing.assert_allclose(paths.patched, patched.patched, atol=TOL)
    assert np.abs(paths.patched - paths.receiver_metric[None, :]).max() > 1e-3


def test_holding_the_mlps_leaves_only_the_direct_path(tiny_backend, project, spec_factory):
    last = tiny_backend.info.n_layers - 1
    scope = {
        "kind": "sites",
        "sites": [{"kind": "head", "layer": last, "head": 0, "position": {"kind": "all"}}],
    }
    prompts = _prompts(tiny_backend, project)
    through = run_experiment(
        spec_factory(experiment=path(LOGITS), scope=scope), tiny_backend, prompts
    )
    direct = run_experiment(
        spec_factory(experiment=path(LOGITS, freeze_mlps=True), scope=scope), tiny_backend, prompts
    )
    assert np.abs(through.patched - direct.patched).max() > 1e-4


def test_only_senders_before_a_receiver_are_swept(tiny_backend, project, spec_factory):
    receivers = [{"kind": "head", "layer": 1, "head": 2, "input": "q"}]
    spec = spec_factory(
        experiment=path(receivers), scope={"kind": "heads", "position": {"kind": "all"}}
    )
    result = run_experiment(spec, tiny_backend, _prompts(tiny_backend, project))
    assert {rs.layer for rs in result.sites} == {0}
    assert len(result.layout["rows"]) == 1 and len(result.sites) == tiny_backend.info.n_heads
    assert [rs.index for rs in result.sites] == list(range(len(result.sites)))
    assert np.isfinite(result.patched).all()
    with pytest.raises(ScopeError, match="No sender comes before"):
        late = {
            "kind": "sites",
            "sites": [{"kind": "head", "layer": 1, "head": 0, "position": {"kind": "all"}}],
        }
        run_experiment(
            spec_factory(experiment=path(receivers), scope=late),
            tiny_backend,
            _prompts(tiny_backend, project),
        )


def test_a_head_receiver_changes_only_through_its_input(tiny_backend, project, spec_factory):
    """Patching the path into one head's value moves the output less than patching the sender."""
    receivers = [
        {"kind": "head", "layer": 1, "head": h, "input": part} for h in range(4) for part in "qkv"
    ]
    scope = {
        "kind": "sites",
        "sites": [{"kind": "head", "layer": 0, "head": 1, "position": {"kind": "all"}}],
    }
    prompts = _prompts(tiny_backend, project)
    every = run_experiment(
        spec_factory(experiment=path(receivers), scope=scope), tiny_backend, prompts
    )
    one = run_experiment(
        spec_factory(
            experiment=path([{"kind": "head", "layer": 1, "head": 0, "input": "v"}]), scope=scope
        ),
        tiny_backend,
        prompts,
    )
    assert np.abs(every.patched - every.receiver_metric).max() > 1e-4
    assert not np.allclose(every.patched, one.patched)


@pytest.mark.parametrize("arch", ["llama", "qwen2"])
def test_a_shared_key_or_value_receives_a_path_through_its_key_value_head(
    arch_backend, project, spec_factory, arch
):
    """With grouped-query attention a key or value receiver names a key/value head, read by every
    query head of its group. Path patching a layer-0 head into it, with everything else held,
    is that head's edge into the key/value head, which edge patching measures by another code
    path. A key/value head the model doesn't have is refused, saying how many it has."""
    from logogram.edges import PatchPlan, build_graph, patched_forward, sender_keys
    from logogram.engine import make_scorer
    from logogram.prompts import group_by_length
    from logogram.spec import GraphScope

    backend = arch_backend(arch)  # two key/value heads for four query heads
    scope = {"kind": "heads", "position": {"kind": "all"}}
    prompts = _prompts(backend, project)
    for part in ("k", "v"):
        receiver = {"kind": "head", "layer": 1, "head": 1, "input": part}
        spec = spec_factory(
            experiment=path([receiver], direction="clean_to_corrupt", freeze_mlps=True),
            scope=scope,
        )
        result = run_experiment(spec, backend, prompts)
        assert [rs.label for rs in result.sites] == [f"L0 H{h}" for h in range(4)]
        assert np.abs(result.patched - result.receiver_metric).max() > 1e-4
        graph = build_graph(backend.info, GraphScope(layers=None, logits=True))
        scorer = make_scorer(spec, prompts)
        weights = {layer: backend.head_output_weights(layer) for layer in range(2)}
        for rs in result.sites:
            label = f"L0 H{rs.head} → L1 KV1 {part}"
            e = next(e for e in range(len(graph)) if graph.label(e) == label)
            values = np.zeros(len(prompts))
            for group in group_by_length(prompts):
                source = backend.capture(group.clean, sender_keys(graph))
                plan = PatchPlan.of(graph, np.array([e]), backend.device)
                forward = patched_forward(backend, plan, source, weights)
                values[group.members] = scorer.score(forward, group.corrupt, group.members).metric
            np.testing.assert_allclose(result.patched[rs.index], values, atol=TOL, err_msg=label)
    with pytest.raises(ScopeError, match=r"L1 KV2 doesn't exist.*2 key/value heads"):
        run_experiment(
            spec_factory(
                experiment=path([{"kind": "head", "layer": 1, "head": 2, "input": "k"}]),
                scope=scope,
            ),
            backend,
            prompts,
        )


def test_receivers_must_be_distinct_and_senders_components(tiny_backend, project, spec_factory):
    with pytest.raises(ValueError, match="must not repeat"):
        spec_factory(experiment=path(LOGITS + LOGITS))
    resid = {"kind": "layer_components", "components": ["resid_pre"], "position": {"kind": "all"}}
    with pytest.raises(ScopeError, match="residual stream state"):
        run_experiment(
            spec_factory(experiment=path(LOGITS), scope=resid),
            tiny_backend,
            _prompts(tiny_backend, project),
        )


def test_a_run_is_stored_like_patching(tiny_backend, project, spec_factory):
    receivers = [{"kind": "head", "layer": 1, "head": 0, "input": "q"}, {"kind": "logits"}]
    spec = spec_factory(
        experiment=path(receivers), scope={"kind": "heads", "position": {"kind": "all"}}
    )
    outcome = run_spec(spec, project, backend=tiny_backend)
    assert outcome.status == "finished", outcome.manifest.get("error")
    summary = json.loads((outcome.folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["measure"] == "intervention"
    assert "Path patching" in summary["description"] and "L1 H0 q" in summary["description"]
    assert (
        len(summary["layout"]["rows"]) == tiny_backend.info.n_layers
    )  # the logits come after every layer


def test_heads_and_logits_together_keep_both_paths(tiny_backend, project, spec_factory):
    """With heads and the logits as receivers, each sender's effect goes through both: the
    logits path no longer overwrites what the head receivers changed. Senders after the last head
    receiver reach only the logits, exactly as with the logits alone."""
    heads = [{"kind": "head", "layer": 1, "head": h, "input": x} for h in range(4) for x in "qkv"]
    scope = {"kind": "heads", "position": {"kind": "all"}}
    prompts = _prompts(tiny_backend, project)

    def change(receivers):
        result = run_experiment(
            spec_factory(experiment=path(receivers), scope=scope), tiny_backend, prompts
        )
        return (result.patched - result.receiver_metric).mean(1)

    through_heads = change(heads)
    through_logits = change(LOGITS)
    both = change(heads + LOGITS)
    n_heads = tiny_backend.info.n_heads
    np.testing.assert_allclose(both[n_heads:], through_logits[n_heads:], atol=1e-6)
    first = slice(0, n_heads)
    assert np.abs(both[first] - through_logits[first]).max() > 1e-2
    # Close to the sum of the two paths (not exactly: the logits read them through a nonlinearity).
    np.testing.assert_allclose(both[first], (through_heads + through_logits[first]), atol=0.05)


def test_a_path_into_an_mlp_is_measured_at_that_mlp_s_input(tiny_backend, project, spec_factory):
    """Path patching into an MLP patches what the sender's change does to that MLP's own input:
    with every other component held, each head's path into the last layer's MLP is the edge from
    it to that MLP, as the edges scope measures it, and not its path into the logits."""
    last = tiny_backend.info.n_layers - 1
    heads = [(layer, h) for layer in range(last + 1) for h in range(2)]
    scope = {
        "kind": "sites",
        "sites": [
            {"kind": "head", "layer": layer, "head": h, "position": {"kind": "all"}}
            for layer, h in heads
        ],
    }
    mlp = [{"kind": "mlp", "layer": last}]
    prompts = _prompts(tiny_backend, project)
    into_mlp = run_experiment(
        spec_factory(experiment=path(mlp, freeze_mlps=True), scope=scope), tiny_backend, prompts
    )
    into_logits = run_experiment(
        spec_factory(experiment=path(LOGITS, freeze_mlps=True), scope=scope), tiny_backend, prompts
    )
    edges = [
        {"sender": {"kind": "head", "layer": layer, "head": h}, "receiver": mlp[0]}
        for layer, h in heads
    ]
    measured = run_experiment(
        spec_factory(
            experiment=path(mlp, freeze_mlps=True), scope={"kind": "edges", "edges": edges}
        ),
        tiny_backend,
        prompts,
    )
    np.testing.assert_allclose(into_mlp.patched, measured.patched, rtol=1e-4, atol=TOL)
    assert np.abs(into_mlp.patched - into_logits.patched).max() > 1e-3
