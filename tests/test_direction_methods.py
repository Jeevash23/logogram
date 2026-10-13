"""Methods that find directions (probes, distributed alignment search) and methods that use them
(directional ablation, subspace patching, readouts, steering along a saved direction), and the
identities each must satisfy on the tiny model."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
import torch

from conftest import make_spec
from logogram.datasets import PromptRecord, write_dataset
from logogram.directions import read_direction, read_vectors, save_direction, save_run_direction
from logogram.ioi import generate_ioi
from logogram.runner import run_spec

STREAM = {"kind": "layer_components", "components": ["resid_post"], "position": {"kind": "last"}}


def _run(project: Any, backend: Any, **overrides: Any) -> Any:
    outcome = run_spec(make_spec(**overrides), project, backend=backend)
    assert outcome.status == "finished", outcome.manifest.get("error")
    return outcome


def _values(outcome: Any) -> np.ndarray:
    import pyarrow.parquet as pq

    table = pq.read_table(outcome.folder / "results.parquet").to_pandas()
    n_sites = int(table["site"].max()) + 1
    return table.sort_values(["site", "prompt"])["patched_metric"].to_numpy().reshape(n_sites, -1)


def _save_basis(project: Any, name: str, vectors: torch.Tensor) -> None:
    save_direction(
        project,
        name,
        vectors,
        model={"id": "tiny-gpt2", "source": "hub", "revision": "test", "sha256": None},
        site={"kind": "resid_post", "layer": 0},
        method="import",
        run=None,
        detail="a test",
    )


# -- scale and directional ablation ------------------------------------------------------------


def test_scaling_by_one_changes_nothing_and_by_zero_is_zero_ablation(project, tiny_backend):
    one = _run(
        project,
        tiny_backend,
        experiment={"kind": "ablation", "baseline": {"kind": "scale", "factor": 1.0}},
    )
    # Runs in batches of other shapes can differ in the last bits (as tests/test_sanity.py allows).
    assert all(s["delta"]["mean"] == pytest.approx(0.0, abs=1e-6) for s in one.summary["sites"])
    zero = _run(
        project,
        tiny_backend,
        experiment={"kind": "ablation", "baseline": {"kind": "scale", "factor": 0.0}},
    )
    ablated = _run(
        project, tiny_backend, experiment={"kind": "ablation", "baseline": {"kind": "zero"}}
    )
    assert np.allclose(_values(zero), _values(ablated), atol=1e-5)


def test_removing_the_whole_space_is_zero_ablation_and_its_mean_is_mean_ablation(
    project, tiny_backend
):
    _save_basis(project, "everything", torch.eye(tiny_backend.info.d_model))
    removed = _run(
        project,
        tiny_backend,
        experiment={
            "kind": "ablation",
            "baseline": {
                "kind": "projection",
                "direction": {"name": "everything", "sha256": None},
                "reference": None,
            },
        },
        scope=STREAM,
    )
    zero = _run(
        project,
        tiny_backend,
        experiment={"kind": "ablation", "baseline": {"kind": "zero"}},
        scope=STREAM,
    )
    assert np.allclose(_values(removed), _values(zero), atol=1e-4)
    to_mean = _run(
        project,
        tiny_backend,
        experiment={
            "kind": "ablation",
            "baseline": {
                "kind": "projection",
                "direction": {"name": "everything", "sha256": None},
                "reference": "corrupt",
            },
        },
        scope=STREAM,
    )
    mean = _run(
        project,
        tiny_backend,
        experiment={"kind": "ablation", "baseline": {"kind": "mean", "reference": "corrupt"}},
        scope=STREAM,
    )
    assert np.allclose(_values(to_mean), _values(mean), atol=1e-4)
    # The run pinned the direction and says so.
    manifest = json.loads((removed.folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["directions"][0]["name"] == "everything"
    saved = json.loads((removed.folder / "spec.json").read_text(encoding="utf-8"))
    assert (
        saved["experiment"]["baseline"]["direction"]["sha256"]
        == manifest["directions"][0]["sha256"]
    )


def test_subspace_patching_of_the_whole_space_is_activation_patching(project, tiny_backend):
    _save_basis(project, "everything", torch.eye(tiny_backend.info.d_model))
    for scope in (
        STREAM,
        {
            "kind": "layer_components",
            "components": ["resid_pre", "mlp_out"],
            "position": {"kind": "all"},
        },
    ):
        within = _run(
            project,
            tiny_backend,
            experiment={
                "kind": "subspace_patching",
                "direction": "clean_to_corrupt",
                "subspace": {"name": "everything", "sha256": None},
            },
            scope=scope,
        )
        patched = _run(
            project,
            tiny_backend,
            experiment={"kind": "activation_patching", "direction": "clean_to_corrupt"},
            scope=scope,
        )
        assert np.allclose(_values(within), _values(patched), atol=1e-4)


def test_heads_have_no_component_along_a_stream_direction(project, tiny_backend):
    _save_basis(project, "u", torch.eye(tiny_backend.info.d_model)[:1])
    outcome = run_spec(
        make_spec(
            experiment={
                "kind": "subspace_patching",
                "direction": "clean_to_corrupt",
                "subspace": {"name": "u", "sha256": None},
            }
        ),
        project,
        backend=tiny_backend,
    )
    assert (
        outcome.status == "failed" and "as wide as the residual stream" in outcome.manifest["error"]
    )


# -- readouts ------------------------------------------------------------------------------------


def test_a_readout_patched_at_its_own_site_moves_all_the_way(project, tiny_backend):
    _save_basis(
        project,
        "u",
        torch.randn(tiny_backend.info.d_model, generator=torch.Generator().manual_seed(1)),
    )
    outcome = _run(
        project,
        tiny_backend,
        metric={
            "kind": "readout",
            "unit": {
                "kind": "direction",
                "direction": {"name": "u", "sha256": None},
                "site": "resid_post",
                "layer": 0,
            },
            "position": {"kind": "last"},
            "normalization": "prompt_gap",
        },
        scope={
            "kind": "layer_components",
            "components": ["resid_post"],
            "position": {"kind": "last"},
        },
    )
    by_layer = {s["layer"]: s["effect"]["mean"] for s in outcome.summary["sites"]}
    assert by_layer[0] == pytest.approx(1.0, abs=1e-6)  # the value read is the value patched
    assert by_layer[1] == pytest.approx(0.0, abs=1e-6)  # after the readout: it can't change it
    assert any("can't change it" in w for w in outcome.summary["warnings"])


def test_a_neuron_readout_reads_the_neuron(project, tiny_backend):
    outcome = _run(
        project,
        tiny_backend,
        metric={
            "kind": "readout",
            "unit": {"kind": "neuron", "layer": 1, "neuron": 3},
            "position": {"kind": "last"},
            "normalization": "dataset_gap",
        },
        scope={"kind": "heads", "position": {"kind": "all"}},
    )
    from logogram.prompts import prepare_prompts

    records = [PromptRecord.model_validate(r.model_dump()) for r in generate_ioi(12, seed=0)]
    prompts = prepare_prompts(tiny_backend, records, True, continuations=False)
    tokens = torch.tensor([p.clean.ids for p in prompts[:1]])
    acts = tiny_backend.capture(tokens, [("neuron", 1)])[("neuron", 1)]
    assert outcome.summary["per_prompt"]["clean_metric"][0] == pytest.approx(
        float(acts[0, -1, 3]), abs=1e-6
    )


def test_readouts_are_refused_where_they_can_t_be_read():
    with pytest.raises(ValueError, match="readout metric is read by"):
        make_spec(
            experiment={
                "kind": "attribution_patching",
                "direction": "clean_to_corrupt",
                "method": "gradient",
                "steps": None,
            },
            metric={
                "kind": "readout",
                "unit": {"kind": "neuron", "layer": 0, "neuron": 0},
                "position": {"kind": "last"},
                "normalization": "dataset_gap",
            },
        )


# -- probes --------------------------------------------------------------------------------------


def _fixed_ending(project: Any) -> None:
    """Pairs whose clean prompt ends with " to" and corrupt prompt with " at": the side is one
    token's identity at the last position."""
    records = []
    for r in generate_ioi(16, seed=3):
        corrupt = r.clean[: -len(" to")] + " at"
        records.append(
            PromptRecord(clean=r.clean, corrupt=corrupt, answer=r.answer, distractor=r.distractor)
        )
    write_dataset(project.datasets_dir / "endings.jsonl", records)


PROBE = {
    "kind": "probe",
    "labels": {"kind": "side"},
    "train_fraction": 0.5,
    "seed": 0,
    "regularization": 0.01,
    "iterations": 100,
    "control": True,
}


def test_a_probe_reads_a_label_that_is_there_and_its_control_does_not(project, tiny_backend):
    _fixed_ending(project)
    outcome = _run(
        project,
        tiny_backend,
        dataset={"path": "datasets/endings.jsonl"},
        experiment=PROBE,
        metric=None,
        scope={
            "kind": "layer_components",
            "components": ["resid_pre"],
            "position": {"kind": "last"},
        },
    )
    summary = outcome.summary
    assert summary["measure"] == "probe" and summary["layout"]["kind"] == "probe"
    probe = summary["probe"]
    assert probe["classes"] == ["clean", "corrupt"] and probe["chance"] == pytest.approx(0.5)
    sites = {(s["layer"], s["variant_key"]): s for s in summary["sites"]}
    assert sites[(0, "probe")]["effect"]["mean"] == 1.0  # one token decides it
    assert sites[(0, "control")]["effect"]["mean"] < 1.0
    assert probe["selectivity"][0]["selective"]
    # The probe's direction can be saved and used elsewhere.
    saved = save_run_direction(project, outcome.run_id, 0, "toward-at")
    assert saved.meta.made_by.method == "probe" and saved.meta.made_by.run == outcome.run_id
    assert "toward corrupt from clean" in saved.meta.made_by.detail
    assert read_direction(project, "toward-at").dimensions == 1
    assert set(read_vectors(project, outcome.run_id)) == {
        "site0",
        "site0.bias",
        "site2",
        "site2.bias",
    }


def test_a_probe_is_bit_identical_twice(project, tiny_backend):
    _fixed_ending(project)
    kwargs = dict(
        dataset={"path": "datasets/mixed.jsonl"},
        experiment={**PROBE, "labels": {"kind": "meta", "field": "template", "prompts": "both"}},
        metric=None,
        scope={
            "kind": "layer_components",
            "components": ["resid_mid", "resid_post"],
            "position": {"kind": "last"},
        },
    )
    first, second = _run(project, tiny_backend, **kwargs), _run(project, tiny_backend, **kwargs)
    assert first.summary["sites"] == second.summary["sites"]
    assert first.summary["probe"]["classes"] == sorted(first.summary["probe"]["classes"])


def test_a_probe_has_no_metric_and_needs_labels_every_prompt_has(project, tiny_backend):
    with pytest.raises(ValueError, match="set metric to null"):
        make_spec(experiment=PROBE)
    outcome = run_spec(
        make_spec(
            experiment={
                **PROBE,
                "labels": {"kind": "meta", "field": "nowhere", "prompts": "clean"},
            },
            metric=None,
            scope=STREAM,
        ),
        project,
        backend=tiny_backend,
    )
    assert outcome.status == "failed" and "no 'nowhere' in their meta" in outcome.manifest["error"]


# -- distributed alignment search -----------------------------------------------------------------


DAS = {
    "kind": "das",
    "direction": "clean_to_corrupt",
    "dimensions": 2,
    "train_fraction": 0.5,
    "seed": 0,
    "steps": 30,
    "learning_rate": 0.05,
    "control": True,
}


def test_das_learns_a_subspace_measured_against_the_whole_site(project, tiny_backend):
    outcome = _run(
        project,
        tiny_backend,
        experiment=DAS,
        scope={
            "kind": "layer_components",
            "components": ["resid_pre"],
            "position": {"kind": "label", "label": "S2"},
        },
    )
    summary = outcome.summary
    assert summary["layout"]["kind"] == "das"
    assert [c["key"] for c in summary["layout"]["cols"]] == ["learned", "full", "random"]
    for site in summary["das"]["sites"]:
        curve = site["curve"]
        assert len(curve) == DAS["steps"] and curve[-1] > curve[0]  # training helps
    # The whole site, patched, is activation patching on the held-out prompts.
    held_out = summary["das"]["test"]
    full = [s for s in summary["sites"] if s["variant_key"] == "full"]
    patched = _run(
        project,
        tiny_backend,
        experiment={"kind": "activation_patching", "direction": "clean_to_corrupt"},
        scope={
            "kind": "layer_components",
            "components": ["resid_pre"],
            "position": {"kind": "label", "label": "S2"},
        },
    )
    every = _values(patched)
    das_values = _values(outcome)
    assert full
    for s in full:
        # The same patch, in batches of the held-out pairs only: equal up to float32 rounding.
        np.testing.assert_allclose(das_values[s["index"]], every[s["row"]][held_out], atol=1e-5)
    # Saved as a direction of two dimensions.
    learned = next(s for s in summary["sites"] if s["variant_key"] == "learned")
    saved = save_run_direction(project, outcome.run_id, learned["index"], "s2-subspace")
    assert saved.dimensions == 2 and saved.meta.made_by.method == "das"


def test_das_is_bit_identical_twice(project, tiny_backend):
    kwargs = dict(
        experiment={**DAS, "steps": 5},
        scope={
            "kind": "layer_components",
            "components": ["resid_post"],
            "position": {"kind": "last"},
        },
    )
    first, second = _run(project, tiny_backend, **kwargs), _run(project, tiny_backend, **kwargs)
    assert np.array_equal(_values(first), _values(second))
    a = read_vectors(project, first.run_id)
    b = read_vectors(project, second.run_id)
    assert all(torch.equal(a[k], b[k]) for k in a)


# -- steering along a saved direction ---------------------------------------------------------


def test_steering_along_a_saved_direction(project, tiny_backend):
    _save_basis(
        project,
        "u",
        torch.randn(tiny_backend.info.d_model, generator=torch.Generator().manual_seed(2)),
    )
    outcome = _run(
        project,
        tiny_backend,
        experiment={
            "kind": "steering",
            "apply_to": "corrupt",
            "direction": {"kind": "direction", "direction": {"name": "u", "sha256": None}},
            "coefficients": [0.0, 1.0],
            "train_fraction": 0.5,
            "seed": 0,
            "control": False,
            "continuations": None,
        },
        scope={
            "kind": "sites",
            "sites": [{"kind": "resid_pre", "layer": 1, "position": {"kind": "last"}}],
        },
    )
    zero = next(s for s in outcome.summary["sites"] if s["variant"]["coefficient"] == 0.0)
    assert zero["delta"]["mean"] == pytest.approx(0.0, abs=1e-6)  # strength 0 adds nothing


# -- reading these runs ---------------------------------------------------------------------------


def test_probe_and_das_runs_read_from_python_the_command_line_and_the_report(project, tiny_backend):
    from typer.testing import CliRunner

    import logogram as lg
    from logogram.cli import app
    from logogram.report import methods_text, run_report

    _fixed_ending(project)
    probe = lg.run(
        make_spec(
            dataset={"path": "datasets/endings.jsonl"},
            experiment=PROBE,
            metric=None,
            scope={
                "kind": "layer_components",
                "components": ["resid_pre"],
                "position": {"kind": "last"},
            },
        ),
        project,
        model=tiny_backend,
    )
    table = probe.probes()
    assert list(table["layer"]) == list(range(tiny_backend.info.n_layers))  # controls aside
    first = table.iloc[0]
    assert first["accuracy"] == 1.0 and first["chance"] == pytest.approx(0.5)
    assert first["selectivity"] == pytest.approx(first["accuracy"] - first["control"])
    saved = probe.save_direction(int(first["site"]), "toward-at-api")
    assert saved["name"] == "toward-at-api" and saved["dimensions"] == 1
    with pytest.raises(lg.api.LogogramError, match="isn't a distributed alignment search run"):
        probe.subspaces()
    # A head's output is narrower than the residual stream, where directions are used.
    heads = lg.run(
        make_spec(
            dataset={"path": "datasets/endings.jsonl"},
            experiment={**PROBE, "control": False},
            metric=None,
            scope={"kind": "heads", "position": {"kind": "last"}},
        ),
        project,
        model=tiny_backend,
    )
    with pytest.raises(lg.api.LogogramError, match="isn't as wide as the residual stream"):
        heads.save_direction(0, "a-head")

    shown = CliRunner().invoke(app, ["show", probe.id, "--project", str(project.root)])
    assert shown.exit_code == 0, shown.output
    assert "The probes read which prompt it is" in shown.output and "selective" in shown.output

    spec = lg.load_spec(probe.folder / "spec.json")
    text = methods_text(spec, probe.summary, probe.manifest, probe.id)
    assert "probe" in text.lower()
    page = run_report(spec, probe.summary, probe.manifest, probe.id)
    assert "<svg" in page and "accuracy" in page.lower()

    # Ablating along the probe's direction names the direction in the methods paragraph.
    removed = lg.run(
        make_spec(
            experiment={
                "kind": "ablation",
                "baseline": {
                    "kind": "projection",
                    "direction": {"name": "toward-at-api", "sha256": None},
                    "reference": None,
                },
            },
            scope=STREAM,
        ),
        project,
        model=tiny_backend,
    )
    pinned = removed.spec.experiment.baseline.direction  # type: ignore[union-attr]
    assert pinned.sha256 == read_direction(project, "toward-at-api").meta.sha256
    assert [d["name"] for d in removed.manifest["directions"]] == ["toward-at-api"]
    words = methods_text(removed.spec, removed.summary, removed.manifest, removed.id)
    assert "toward-at-api" in words

    das = lg.run(
        make_spec(experiment={**DAS, "steps": 3}, scope=STREAM), project, model=tiny_backend
    )
    rows = das.subspaces()
    assert set(rows.columns) >= {"learned", "whole", "random", "share", "train_effect"}
    assert len(rows) == tiny_backend.info.n_layers
    page = run_report(das.spec, das.summary, das.manifest, das.id)
    assert "learned" in page.lower()


def test_methods_of_one_site_at_a_time_refuse_sets_of_sites():
    sets = {
        "kind": "site_sets",
        "universe": None,
        "sets": [
            {
                "label": "early",
                "sites": [{"kind": "resid_post", "layer": 0, "position": {"kind": "last"}}],
                "complement": False,
            }
        ],
    }
    toward = {"name": "u", "sha256": None}
    scaled = {"kind": "ablation", "baseline": {"kind": "scale", "factor": 0.5}}
    with pytest.raises(ValueError, match="scaling and directional ablation act on one site"):
        make_spec(experiment=scaled, scope=sets)
    removed = {
        "kind": "ablation",
        "baseline": {"kind": "projection", "direction": toward, "reference": None},
    }
    with pytest.raises(ValueError, match="scaling and directional ablation act on one site"):
        make_spec(experiment=removed, scope=sets)
    along = {"kind": "subspace_patching", "direction": "clean_to_corrupt", "subspace": toward}
    with pytest.raises(ValueError, match="subspace patching act on one site at a time"):
        make_spec(experiment=along, scope=sets)
    with pytest.raises(ValueError, match="subspace patching act on one site at a time"):
        make_spec(experiment=DAS, scope=sets)
    with pytest.raises(ValueError, match="subspace patching act on one site at a time"):
        make_spec(experiment=PROBE, metric=None, scope=sets)
    # Zero, mean and resample ablation of sets are as before.
    make_spec(experiment={"kind": "ablation", "baseline": {"kind": "zero"}}, scope=sets)
