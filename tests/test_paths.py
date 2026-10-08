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
    np.testing.assert_allclose(paths.patched_ld, patched.patched_ld, atol=TOL)
    assert np.abs(paths.patched_ld - paths.receiver_ld[None, :]).max() > 1e-3


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
    assert np.abs(through.patched_ld - direct.patched_ld).max() > 1e-4


def test_only_senders_before_a_receiver_are_swept(tiny_backend, project, spec_factory):
    receivers = [{"kind": "head", "layer": 1, "head": 2, "input": "q"}]
    spec = spec_factory(
        experiment=path(receivers), scope={"kind": "heads", "position": {"kind": "all"}}
    )
    result = run_experiment(spec, tiny_backend, _prompts(tiny_backend, project))
    assert {rs.layer for rs in result.sites} == {0}
    assert len(result.layout["rows"]) == 1 and len(result.sites) == tiny_backend.info.n_heads
    assert [rs.index for rs in result.sites] == list(range(len(result.sites)))
    assert np.isfinite(result.patched_ld).all()
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
    assert np.abs(every.patched_ld - every.receiver_ld).max() > 1e-4
    assert not np.allclose(every.patched_ld, one.patched_ld)


def test_shared_keys_and_values_are_refused(arch_backend, project, spec_factory):
    backend = arch_backend("llama")  # two key/value heads for four query heads
    scope = {"kind": "heads", "position": {"kind": "all"}}
    prompts = _prompts(backend, project)
    with pytest.raises(ScopeError, match="shares each key and value"):
        run_experiment(
            spec_factory(
                experiment=path([{"kind": "head", "layer": 1, "head": 0, "input": "k"}]),
                scope=scope,
            ),
            backend,
            prompts,
        )
    result = run_experiment(
        spec_factory(
            experiment=path([{"kind": "head", "layer": 1, "head": 0, "input": "q"}]), scope=scope
        ),
        backend,
        prompts,
    )
    assert np.isfinite(result.patched_ld).all()


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
