"""Version 2 specs state every choice; version 1 specs are read and say what they left out. And
runs that go wrong say so plainly, without leaking this machine's paths."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from logogram.runner import run_spec, scrub_paths
from logogram.spec import Spec, upgrade_v1

V1_MINIMAL = {
    "logogram_spec": 1,
    "name": "minimal",
    "model": {"id": "tiny-gpt2", "revision": "test", "device": "cpu"},
    "dataset": {"path": "datasets/ioi.jsonl"},
    "experiment": {"kind": "ablation", "baseline": {"kind": "resample", "donors": 2}},
    "scope": {"kind": "heads"},
    "statistics": {"bootstrap": 200},
}


def test_version_1_specs_list_what_they_left_out():
    spec = Spec.model_validate(V1_MINIMAL)
    assert spec.logogram_spec == 2
    filled = "\n".join(spec.upgraded_fields)
    for expected in (
        'model.dtype = "float32"',
        "model.process_weights = true",
        "tokenization",
        'metric = {"kind": "logit_diff", "normalization": "dataset_gap"}',
        "statistics.ci = 0.95",
        "statistics.seed = 0",
        "execution",
        'experiment.baseline.pool = "corrupt"',
        "experiment.baseline.seed = 0",
        'scope.position = {"kind": "all"}',
    ):
        assert expected in filled
    assert "donors" not in filled  # it was given
    # What a version 1 spec as written by Logogram 0.1 holds is complete: nothing is filled.
    dumped = spec.model_dump(mode="json")
    dumped["logogram_spec"] = 1
    assert Spec.model_validate(dumped).upgraded_fields == []


def test_version_2_specs_must_state_every_choice():
    full = Spec.model_validate(V1_MINIMAL).model_dump(mode="json")
    assert Spec.model_validate(full).upgraded_fields == []
    for path in (
        ("metric",),
        ("statistics", "seed"),
        ("statistics", "cluster"),
        ("execution",),
        ("tokenization",),
        ("model", "dtype"),
        ("model", "revision"),
        ("dataset", "limit"),
        ("experiment", "baseline", "donors"),
        ("scope", "position"),
    ):
        data = json.loads(json.dumps(full))
        target = data
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]
        with pytest.raises(ValidationError, match="Field required"):
            Spec.model_validate(data)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {"dataset": {"path": "/srv/data/ioi.jsonl", "sha256": None, "limit": None}},
            "inside the project",
        ),
        (
            {"dataset": {"path": "../ioi.jsonl", "sha256": None, "limit": None}},
            "inside the project",
        ),
        (
            {"dataset": {"path": "C:/data/ioi.jsonl", "sha256": None, "limit": None}},
            "inside the project",
        ),
        ({"sae": {"repo": "a/b", "path": "../../x", "revision": None}}, "inside the project"),
        ({"sae": {"repo": "a/b", "path": "", "revision": "../../etc"}}, "commit, tag or branch"),
        ({"metric": {"kind": "kl", "normalization": "dataset_gap"}}, "target"),
        (
            {"statistics": {"bootstrap": 200, "ci": 0.95, "seed": -1, "cluster": None}},
            "greater than or equal",
        ),
        (
            {"statistics": {"bootstrap": 200, "ci": 0.95, "seed": 2**40, "cluster": None}},
            "less than",
        ),
        ({"statistics": {"bootstrap": 200, "ci": 0.95, "seed": 0, "cluster": "a b"}}, "meta"),
    ],
)
def test_specs_are_checked(change, message):
    data = {**Spec.model_validate(V1_MINIMAL).model_dump(mode="json"), **change}
    with pytest.raises(ValidationError, match=message):
        Spec.model_validate(data)


def test_attribution_patching_states_its_method():
    base = Spec.model_validate(V1_MINIMAL).model_dump(mode="json")
    ok = {"kind": "attribution_patching", "direction": "clean_to_corrupt"}
    for experiment, error in [
        ({**ok, "method": "integrated_gradients", "steps": None}, "steps"),
        ({**ok, "method": "gradient", "steps": 8}, "no steps"),
        ({**ok, "method": "integrated_gradients", "steps": 1}, "greater than or equal"),
    ]:
        with pytest.raises(ValidationError, match=error):
            Spec.model_validate({**base, "experiment": experiment})
    # Version 1 attribution patching was the single gradient.
    old = {**V1_MINIMAL, "experiment": ok}
    assert Spec.model_validate(old).experiment.method == "gradient"


def test_a_version_1_run_says_what_it_assumed(tiny_backend, project):
    spec = Spec.model_validate(V1_MINIMAL)
    outcome = run_spec(spec, project, backend=tiny_backend)
    assert outcome.status == "finished", outcome.manifest.get("error")
    assert "This spec is version 1" in outcome.summary["warnings"][0]
    assert "statistics.seed = 0" in outcome.summary["warnings"][0]
    saved = json.loads((outcome.folder / "spec.json").read_text(encoding="utf-8"))
    assert saved["logogram_spec"] == 2 and saved["statistics"]["cluster"] is None


def test_upgrading_never_changes_the_input():
    data = json.loads(json.dumps(V1_MINIMAL))
    upgrade_v1(data)
    assert data == V1_MINIMAL


# -- runs that go wrong --------------------------------------------------------------------------


def test_values_that_are_not_finite_are_refused(tiny_backend, project, spec_factory, monkeypatch):
    original = tiny_backend.logits

    def overflowing(tokens, patch=None, keep=1):
        return original(tokens, patch, keep) * float("nan")

    monkeypatch.setattr(tiny_backend, "logits", overflowing)
    outcome = run_spec(spec_factory(), project, backend=tiny_backend)
    assert outcome.status == "failed"
    assert "aren't finite numbers" in outcome.manifest["error"]


def test_patched_values_that_are_not_finite_are_reported(
    tiny_backend, project, spec_factory, monkeypatch
):
    original = tiny_backend.logits

    def overflowing(tokens, patch=None, keep=1):
        logits = original(tokens, patch, keep)
        return logits * float("nan") if patch is not None else logits

    monkeypatch.setattr(tiny_backend, "logits", overflowing)
    outcome = run_spec(spec_factory(), project, backend=tiny_backend)
    assert outcome.status == "finished"
    assert any("aren't finite numbers" in w for w in outcome.summary["warnings"])
    assert all(site["effect"]["mean"] is None for site in outcome.summary["sites"])


def test_running_out_of_memory_in_a_run_says_what_to_change(
    tiny_backend, project, spec_factory, monkeypatch
):
    import logogram.runner as runner

    def exhausted(*args, **kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(runner, "run_experiment", exhausted)
    outcome = run_spec(spec_factory(), project, backend=tiny_backend)
    assert outcome.status == "failed"
    assert "execution.batch_size" in outcome.manifest["error"]
    assert "Tried to allocate" not in outcome.manifest["error"]


def test_failed_runs_never_record_this_machines_paths(tiny_backend, project, spec_factory):
    home = str(Path.home())

    def broken(ref):
        raise RuntimeError(
            f"Unrecognized model in {home}/.cache/huggingface/hub/models--x/snapshots/abc "
            f"(project at {project.root})"
        )

    other = spec_factory(model={"id": "other/model", "revision": "test", "device": "cpu"})
    outcome = run_spec(other, project, provider=broken)
    assert outcome.status == "failed"
    written = "".join(
        p.read_text(encoding="utf-8") for p in outcome.folder.iterdir() if p.suffix == ".json"
    )
    assert home not in written and str(project.root) not in written
    assert "<project>" in outcome.manifest["error"]


def test_scrubbing_paths():
    home = str(Path.home())
    text = scrub_paths(f"open {home}/a and /tmp/proj/b", "/tmp/proj")
    assert text == "open ~/a and <project>/b"
    assert scrub_paths("nothing here", None) == "nothing here"


def test_reproduction_checks_skip_what_a_version_does_not_measure():
    import pyarrow as pa

    from logogram.results import largest_change

    a = pa.table(
        {
            "site": [0, 0],
            "prompt": [0, 1],
            "delta": [0.5, 0.25],
            "effect": [1.0, 0.5],
            "patched_logit_diff": [1.0, 2.0],
        }
    )
    b = pa.table(
        {
            "site": [0, 0],
            "prompt": [0, 1],
            "delta": [0.5, 0.25],
            "effect": [1.0, 0.5],
            "patched_logit_diff": [np.nan, np.nan],
            "patched_metric": [3.0, 4.0],
        }
    )
    assert largest_change(a, b) == 0.0
    c = b.set_column(2, "delta", pa.array([0.5, 0.75]))
    assert largest_change(a, c) == pytest.approx(0.5)
