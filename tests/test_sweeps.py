"""Sweeps: one spec run across several values of one field. The specs a sweep makes differ from the
original in that field alone, each run records the sweep, the original run stands in for its own
value, and the sweep's table reports the runs' own numbers, with missing values missing."""

from __future__ import annotations

import csv
import dataclasses
import io
import json
import re
import statistics
import time
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import logogram as lg
from logogram.backends import hub
from logogram.backends.base import BackendError
from logogram.backends.transformer_lens import TransformerLensBackend
from logogram.cli import app as cli
from logogram.compare import site_key, spec_differences
from logogram.datasets import load_dataset, write_dataset
from logogram.history import delete_run
from logogram.ioi import generate_ioi
from logogram.runner import run_spec
from logogram.schema import Manifest, SweepMember
from logogram.server.app import _project_scoped, create_app
from logogram.server.security import SecurityConfig
from logogram.sweeps import (
    FIELDS,
    SweepError,
    _site_row,
    plan_sweep,
    run_sweep,
    sweep_table,
    table_rows,
)

TOKEN = "test-token"
BASE = "http://127.0.0.1:8765"
HEADERS = {"authorization": f"Bearer {TOKEN}", "origin": BASE}

# Quick runs: attention and MLP of each layer at the last position, on eight prompts.
QUICK = {
    "scope": {"kind": "layer_components", "position": {"kind": "last"}},
    "dataset": {"path": "datasets/ioi.jsonl", "limit": 8},
}
RESAMPLE = {
    "kind": "ablation",
    "baseline": {"kind": "resample", "pool": "corrupt", "donors": 2, "seed": 0},
}
STEERING = {
    "kind": "steering",
    "apply_to": "corrupt",
    "coefficients": [1.0],
    "train_fraction": 0.5,
    "seed": 0,
    "control": False,
}
SAE = {"repo": "local/exact", "path": "", "revision": "test"}
FEATURES = {"kind": "features", "position": {"kind": "all"}, "top": 4, "choose_on": 0.5, "seed": 0}
ATTRIBUTION = {"kind": "attribution_patching", "direction": "clean_to_corrupt"}


def _changed(a: Any, b: Any) -> set[str]:
    return {
        d["path"] for d in spec_differences(a.model_dump(mode="json"), b.model_dump(mode="json"))
    }


def _manifest(project: Any, run_id: str) -> dict[str, Any]:
    return json.loads((project.run_dir(run_id) / "manifest.json").read_text())


def _summary(project: Any, run_id: str) -> dict[str, Any]:
    return json.loads((project.run_dir(run_id) / "summary.json").read_text())


@pytest.fixture
def quiet(monkeypatch, project):
    """Runs never load a model: the tiny one is the model every spec names."""
    monkeypatch.setattr("logogram.project.config_dir", lambda: project.root / "config")


@pytest.mark.parametrize(
    ("path", "overrides"),
    [
        ("statistics.seed", {}),
        ("experiment.baseline.seed", {"experiment": RESAMPLE}),
        ("experiment.seed", {"experiment": STEERING}),
        ("scope.seed", {"experiment": ATTRIBUTION, "scope": FEATURES, "sae": SAE}),
    ],
)
def test_each_run_of_a_seed_sweep_differs_from_the_spec_in_that_seed_alone(
    project, spec_factory, path, overrides
):
    spec = spec_factory(name="Heads", **overrides)
    plan = plan_sweep(project, spec, path, [0, 7, 3])
    field = FIELDS[path]
    assert [r.value for r in plan.runs] == [0, 7, 3]  # in the order given
    assert plan.original is None and plan.stands_in is None
    for item in plan.runs:
        assert item.spec is not None
        # Seed 0 is the spec's own, which only a run it came from could stand in for.
        assert _changed(spec, item.spec) <= {path}
        assert _changed(spec, item.spec) == (set() if item.value == 0 else {path})
        assert item.spec.name == f"Heads · {field.label} {item.value}"
        record = SweepMember.model_validate(item.derived_from)
        assert (record.sweep, record.field, record.value) == (plan.id, path, item.value)
        assert (record.index, record.of, record.values) == (item.index, 3, [0, 7, 3])
        assert record.run is None and record.original is None
        assert record.change == f"{field.label} {item.value}"


def test_a_dataset_sweep_pins_each_dataset_by_its_hash(project, spec_factory):
    spec = spec_factory(**QUICK)
    # A bare name is looked for in datasets/.
    plan = plan_sweep(project, spec, "dataset.path", ["mixed.jsonl", "datasets/ioi.jsonl"])
    assert plan.values == ["datasets/mixed.jsonl", "datasets/ioi.jsonl"]
    for item in plan.runs:
        assert item.spec is not None
        sha = hashlib_of(project, item.value)
        assert item.spec.dataset.path == item.value and item.spec.dataset.sha256 == sha
        assert item.spec.dataset.limit == 8  # the rest of the dataset's choices stay
        assert _changed(spec, item.spec) <= {"dataset.path", "dataset.sha256"}


def hashlib_of(project: Any, path: str) -> str:
    import hashlib

    return hashlib.sha256((project.root / path).read_bytes()).hexdigest()


def test_a_revision_sweep_keeps_the_names_and_each_run_pins_its_commit(
    tiny_backend, project, spec_factory, quiet
):
    spec = spec_factory(name="Checkpoints", model={"id": "tiny-gpt2", "revision": None}, **QUICK)
    plan = plan_sweep(project, spec, "model.revision", ["step1000", "step2000", "main"])
    for item in plan.runs:
        assert item.spec is not None and item.spec.model.revision == item.value
        assert _changed(spec, item.spec) == {"model.revision"}
    commits = {name: f"{i + 1}" * 40 for i, name in enumerate(["step1000", "step2000", "main"])}
    loaded: list[Any] = []

    def provider(ref: Any) -> Any:
        # The tiny model, as if Hugging Face had resolved the revision to this commit: a backend of
        # its own over the same weights, so closing it leaves the shared one loaded. (A copy
        # wouldn't do: once a test has patched the shared backend's close, its instance holds the
        # bound method, and a copy's close would close the shared one.)
        info = dataclasses.replace(tiny_backend.info, revision=commits[ref.revision])
        model = TransformerLensBackend(tiny_backend.bridge, info)
        loaded.append(model)
        return model

    outcomes, last = run_sweep(plan, project, provider=provider)
    assert [o.status for o in outcomes if o is not None] == ["finished"] * 3
    for item, outcome in zip(plan.runs, outcomes, strict=True):
        assert outcome is not None
        ran = json.loads((outcome.folder / "spec.json").read_text())
        assert ran["model"]["revision"] == commits[item.value]
        assert _manifest(project, outcome.run_id)["derived_from"]["value"] == item.value
    # Each revision's model was closed once the next was loaded; the last is still loaded.
    assert [m.bridge is None for m in loaded] == [True, True, False] and last is loaded[-1]
    table = sweep_table(project, plan.id)
    assert [v["revision"] for v in table["values"]] == list(commits.values())
    assert [v["label"] for v in table["values"]] == ["step1000", "step2000", "main"]
    assert "the line joins each run's mean" in table["spread"]


def test_values_are_checked_and_errors_say_how_to_fix_them(project, spec_factory):
    spec = spec_factory()
    cases = [
        ("model.dtype", ["float32", "float16"], "can't sweep 'model.dtype'"),
        ("experiment.baseline.seed", [0, 1], "only resample ablation draws donors"),
        ("experiment.seed", [0, 1], "only steering splits its pairs"),
        ("scope.seed", [0, 1], "only a scope over every SAE feature"),
        ("statistics.seed", [0], "at least two bootstrap seeds"),
        ("statistics.seed", [0, 2**32], "whole numbers from 0 to 4,294,967,295"),
        ("statistics.seed", [0, True], "isn't one"),
        ("statistics.seed", [1, "2"], "isn't one"),
        ("statistics.seed", [1, 2, 1], "Bootstrap seed 1 is listed twice"),
        ("statistics.seed", list(range(257)), "at most 256 values"),
        ("dataset.path", ["ioi.jsonl", "nope.jsonl"], "There is no dataset datasets/nope.jsonl"),
        ("dataset.path", ["ioi.jsonl", "../ioi.jsonl"], "without '..'"),
        ("model.revision", ["main", "../x"], "A revision must be a commit, tag or branch name"),
        ("model.revision", ["main", "main"], "listed twice"),
    ]
    for path, values, message in cases:
        with pytest.raises(SweepError, match=re.escape(message)):
            plan_sweep(project, spec, path, values)
    unfeatured = spec_factory(
        experiment=ATTRIBUTION, scope={**FEATURES, "choose_on": None, "seed": None}, sae=SAE
    )
    with pytest.raises(SweepError, match="choose_on is null"):
        plan_sweep(project, unfeatured, "scope.seed", [0, 1])
    # Two datasets that hold the same prompts would make the same run twice.
    write_dataset(project.dataset_file("copy.jsonl"), generate_ioi(12, seed=0))
    with pytest.raises(SweepError, match="hold the same prompts"):
        plan_sweep(project, spec, "dataset.path", ["ioi.jsonl", "copy.jsonl"])
    local = spec_factory(
        model={
            "id": "mine",
            "source": "local",
            "revision": None,
            "sha256": "a" * 64,
            "device": "cpu",
        }
    )
    with pytest.raises(SweepError, match="local model has no revisions"):
        plan_sweep(project, local, "model.revision", ["a", "b"])


def test_the_run_a_sweep_is_made_from_stands_in_for_its_own_value(
    tiny_backend, project, spec_factory, quiet
):
    original = run_spec(
        spec_factory(name="Donors", experiment=RESAMPLE, **QUICK), project, backend=tiny_backend
    )
    listing = project.run_listing(original.run_id)
    spec = lg.load_run(project, original.run_id).spec
    for values, at in (([0, 1, 2], 0), ([3, 0], 1), ([4, 5], None)):
        plan = plan_sweep(project, spec, "experiment.baseline.seed", values, original=listing)
        assert plan.stands_in == at and plan.original == original.run_id
        for item in plan.runs:
            assert (item.spec is None) == (item.index == at)
            if item.derived_from is not None:
                assert item.derived_from["run"] == original.run_id
                assert item.derived_from["original"] == at
    # Datasets match by their prompts: the run read a snapshot of ioi.jsonl.
    plan = plan_sweep(project, spec, "dataset.path", ["mixed.jsonl", "ioi.jsonl"], original=listing)
    assert plan.stands_in == 1
    # Only a finished run stands in.
    failed = listing.model_copy(update={"status": "failed"})
    assert (
        plan_sweep(project, spec, "experiment.baseline.seed", [0, 1], original=failed).stands_in
        is None
    )


def test_verdicts_say_which_sites_keep_their_sign_and_size():
    field = FIELDS["statistics.seed"]
    labels = ["0", "1", "2", "3"]

    def row(points: list[Any], reasons: list[str | None] | None = None) -> dict[str, Any]:
        pts = [
            None if p is None else {"mean": p[0], "lo": p[1], "hi": p[2], "sd": 0.1, "n": 8}
            for p in points
        ]
        meta = {"label": "L0 attn", "kind": "attn_out", "layer": 0, "position_key": "last"}
        return _site_row(
            "k", meta, pts, reasons or [None] * len(points), field, labels[: len(points)], 1
        )

    holds = row([(0.5, 0.4, 0.6), (0.55, 0.45, 0.65), (0.52, 0.41, 0.63)])
    assert holds["verdict"] == "holds" and holds["above"] == [0, 1, 2] and holds["below"] == []
    assert holds["text"].startswith("Keeps its sign and size at all 3 bootstrap seeds")
    assert "excludes zero at every one" in holds["text"] and holds["excludes_zero"] == "every"
    assert holds["spread"]["sd"] == pytest.approx(statistics.stdev([0.5, 0.55, 0.52]))
    assert holds["half_width"] == pytest.approx(0.1)
    size = row([(0.5, 0.45, 0.55), (0.2, 0.15, 0.25)])
    assert size["verdict"] == "size" and size["apart"] == [0, 1] and size["sign"] is None
    assert "intervals at bootstrap seeds 0 and 1 exclude each other" in size["text"]
    sign = row([(0.3, 0.1, 0.5), (0.0, -0.2, 0.2), (-0.3, -0.5, -0.1)])
    assert sign["verdict"] == "sign" and sign["sign"] == [0, 2]
    assert "above zero at bootstrap seed 0 and below zero at bootstrap seed 2" in sign["text"]
    # Opposite signs that neither interval is sure of aren't a flip.
    unsure = row([(0.05, -0.1, 0.2), (-0.03, -0.2, 0.1)])
    assert unsure["verdict"] == "holds" and unsure["excludes_zero"] == "none"
    assert "includes zero at every one, so no value shows an effect" in unsure["text"]
    # Missing values are left out, never read as zero, and say why.
    gap = row(
        [(0.5, 0.4, 0.6), None, (0.52, 0.41, 0.63), None], [None, "the run failed", None, "no run"]
    )
    assert gap["verdict"] == "holds" and gap["measured"] == 2 and gap["missing"] == [1, 3]
    assert gap["spread"]["min"] == 0.5 and gap["points"][1] is None
    assert "Not measured at bootstrap seeds 1 and 3 (no run; the run failed)." in gap["text"]
    alone = row([(0.5, 0.4, 0.6), None], [None, "the run failed"])
    assert alone["verdict"] == "unknown" and alone["spread"]["sd"] is None


def _reversed(project: Any) -> None:
    """datasets/reversed.jsonl: ioi.jsonl's prompts in reverse order. Another file, with the gap
    ioi.jsonl has on the tiny model whatever its weights (which change with library versions)."""
    records = load_dataset(project.datasets_dir / "ioi.jsonl")
    write_dataset(project.dataset_file("reversed.jsonl"), records[::-1])


def test_the_command_line_sweeps_seeds_and_datasets(
    tiny_backend, project, spec_factory, quiet, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "logogram.runner.default_provider", lambda on_event=None: lambda ref: tiny_backend
    )
    original = run_spec(
        spec_factory(name="Donors", experiment=RESAMPLE, **QUICK), project, backend=tiny_backend
    )
    table = tmp_path / "sweep.csv"
    result = CliRunner().invoke(
        cli,
        [
            "sweep",
            original.run_id,
            "--field",
            "experiment.baseline.seed",
            "--values",
            "0,1,2",
            "--csv",
            str(table),
            "--project",
            str(project.root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert f"{original.run_id} stands in for donor seed 0: it isn't run again." in result.output
    assert "Running donor seed 2 (2 of 2)" in result.output
    assert "sites keep their sign and size across 3 donor seeds" in result.output
    assert "Chosen on the first value, donor seed 0" in result.output
    swept = [r for r in project.list_runs() if isinstance(r.derived_from, SweepMember)]
    assert sorted(r.derived_from.index for r in swept) == [1, 2]
    for r in swept:
        record = r.derived_from
        assert record.run == original.run_id and record.original == 0 and record.values == [0, 1, 2]
        assert r.status == "finished" and r.name == f"Donors · donor seed {record.value}"
        # The manifest says so, and validates.
        Manifest.model_validate(_manifest(project, r.id))
    rows = list(csv.DictReader(io.StringIO(table.read_text())))
    assert len(rows) == 4 * 3 and {r["value"] for r in rows} == {"0", "1", "2"}
    assert {r["run"] for r in rows if r["value"] == "0"} == {original.run_id}

    # A spec file, swept over datasets: every value runs, and none stands in.
    _reversed(project)
    spec_file = project.root / "datasets.json"
    spec_file.write_text(spec_factory(name="Prompts", **QUICK).to_json())
    result = CliRunner().invoke(
        cli,
        ["sweep", str(spec_file), "--datasets", "ioi.jsonl,datasets/reversed.jsonl", "--top", "2"],
    )
    assert result.exit_code == 0, result.output
    assert "over datasets datasets/ioi.jsonl, datasets/reversed.jsonl" in result.output
    made = [
        r
        for r in project.list_runs()
        if isinstance(r.derived_from, SweepMember) and r.derived_from.field == "dataset.path"
    ]
    assert sorted(r.derived_from.value for r in made) == [
        "datasets/ioi.jsonl",
        "datasets/reversed.jsonl",
    ]
    assert all(r.derived_from.run is None and r.derived_from.original is None for r in made)


def test_the_command_line_says_what_it_needs(project, spec_factory, quiet):
    run = ["--project", str(project.root)]
    spec_file = project.root / "spec.json"
    spec_file.write_text(spec_factory().to_json())
    for args, message in (
        ([str(spec_file)], "choose what to sweep"),
        (
            [str(spec_file), "--field", "statistics.seed"],
            "give the field with --field and its values",
        ),
        ([str(spec_file), "--field", "statistics.seed", "--values", "a,b"], "whole numbers"),
        ([str(spec_file), "--datasets", "a", "--revisions", "b"], "choose what to sweep"),
        ([str(spec_file), "--field", "statistics.seed", "--values", "1,1"], "listed twice"),
        ([str(spec_file), "--datasets", "ioi.jsonl,nope.jsonl"], "There is no dataset"),
        (
            [str(spec_file), "--field", "statistics.seed", "--values", "0,1", "--choose", "best"],
            "--choose",
        ),
    ):
        result = CliRunner().invoke(cli, ["sweep", *args, *run])
        assert result.exit_code == 1, result.output
        assert message in result.output


def test_the_python_api_sweeps_and_its_table_reports_the_runs_own_numbers(
    tiny_backend, project, spec_factory, quiet
):
    original = lg.run(
        spec_factory(name="Donors", experiment=RESAMPLE, **QUICK), project, model=tiny_backend
    )
    sweep = lg.sweep(original, field="experiment.baseline.seed", values=[0, 1, 2])
    assert isinstance(sweep, list) and len(sweep) == 3 and sweep[0] is original
    assert [r.status for r in sweep] == ["finished"] * 3
    assert all(r.model is tiny_backend for r in sweep[1:])
    assert sweep.field == "experiment.baseline.seed" and sweep.values == [0, 1, 2]
    again = lg.load_sweep(project, sweep.id)
    assert [r.id for r in again] == [r.id for r in sweep]

    table = sweep.summary(choose="all", top=0)
    summaries = [_summary(project, r.id) for r in sweep]
    assert table["finished"] == 3 and table["n_sites"] == 4 and len(table["sites"]) == 4
    for site in table["sites"]:
        for i, summary in enumerate(summaries):
            own = next(
                s
                for s in summary["sites"]
                if "|".join("" if x is None else str(x) for x in site_key(s)) == site["key"]
            )
            point = site["points"][i]
            assert point is not None
            assert (point["mean"], point["lo"], point["hi"], point["sd"], point["n"]) == (
                own["effect"]["mean"],
                own["effect"]["lo"],
                own["effect"]["hi"],
                own["effect"]["sd"],
                own["n"],
            )
        means = [p["mean"] for p in site["points"]]
        assert site["spread"]["mean"] == pytest.approx(statistics.fmean(means))
        assert site["spread"]["sd"] == pytest.approx(statistics.stdev(means))
    # Chosen on all values: by the mean |effect| over them, strongest first.
    strength = [statistics.fmean(abs(p["mean"]) for p in s["points"]) for s in table["sites"]]
    assert strength == sorted(strength, reverse=True)
    # Chosen on the first value: by |effect| there, and the table says so.
    first = sweep.summary(choose="first", top=2)
    assert first["chosen"].startswith("Chosen on the first value, donor seed 0")
    ranked = sorted(summaries[0]["sites"], key=lambda s: -abs(s["effect"]["mean"]))[:2]
    assert [s["label"] for s in first["sites"]] == [s["label"] for s in ranked]
    rows = table_rows(first)
    assert len(rows) == 2 * 3 and [r["value"] for r in rows] == [0, 1, 2, 0, 1, 2]
    assert len(sweep.table(top=2)) == 2 * 3
    assert sweep._repr_html_().count("<tr>") == 1 + 4


def test_failed_and_missing_runs_are_missing_never_zero(tiny_backend, project, spec_factory, quiet):
    original = lg.run(spec_factory(name="Prompts", **QUICK), project, model=tiny_backend)
    _reversed(project)
    write_dataset(project.dataset_file("third.jsonl"), generate_ioi(12, seed=5))
    write_dataset(project.dataset_file("fourth.jsonl"), generate_ioi(12, seed=6))
    plan = plan_sweep(
        project,
        original.spec,
        "dataset.path",
        ["ioi.jsonl", "reversed.jsonl", "third.jsonl", "fourth.jsonl"],
        original=project.run_listing(original.id),
    )
    # The third dataset changes after the sweep pinned it: its run fails, as a pinned spec should.
    write_dataset(project.dataset_file("third.jsonl"), generate_ioi(12, seed=7))
    outcomes, _ = run_sweep(plan, project, backend=tiny_backend)
    assert [o.status if o else None for o in outcomes] == [None, "finished", "failed", "finished"]
    # The fourth value's run is deleted: it has no run any more.
    delete_run(project, outcomes[3].run_id)
    table = sweep_table(project, plan.id, choose="first", top=0)
    assert [v["status"] for v in table["values"]] == ["finished", "finished", "failed", "missing"]
    assert table["values"][0]["original"] and table["values"][0]["run"] == original.id
    assert table["finished"] == 2
    assert any("2 of 4 values have no results" in note for note in table["notes"])
    for site in table["sites"]:
        assert site["points"][2] is None and site["points"][3] is None
        assert site["missing"] == [2, 3]
        assert "(no run; the run failed)" in site["text"]
    cells = [r for r in table_rows(table) if r["value_index"] in (2, 3)]
    assert cells and all(r["effect"] is None and r["effect_lo"] is None for r in cells)
    from logogram.exports import sweep_csv

    lines = list(csv.DictReader(io.StringIO(sweep_csv(table_rows(table)))))
    assert {r["effect"] for r in lines if r["value_index"] in ("2", "3")} == {""}
    # When the first value has no results, sites can't be chosen on it, and the table says so.
    delete_run(project, original.id)
    gone = sweep_table(project, plan.id, choose="first")
    assert gone["values"][0]["status"] == "deleted" and gone["sites"] == []
    assert gone["problem"].startswith("Sites can't be chosen on the first value, dataset ioi.jsonl")
    assert sweep_table(project, plan.id, choose="all")["sites"]


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    return create_app(SecurityConfig(token=TOKEN, port=8765), serve_web=False)


def _settled(client: TestClient, timeout: float = 120) -> dict[str, dict[str, Any]]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = client.get("/api/state", headers=HEADERS).json()
        job = state["job"]
        if not state["queue"] and (job is None or job["status"] != "running"):
            runs = client.get("/api/runs", headers=HEADERS).json()
            if all(r["status"] != "running" for r in runs):
                return {r["id"]: r for r in runs}
        time.sleep(0.05)
    raise AssertionError("the queue did not settle")


def test_the_app_queues_a_sweep_and_reads_it_back(app, project, tiny_backend, spec_factory):
    state = app.state.logogram
    state.backend = tiny_backend
    state.model_status = {"state": "ready"}
    client = TestClient(app, base_url=BASE)
    assert (
        client.post(
            "/api/projects/open", json={"path": str(project.root)}, headers=HEADERS
        ).status_code
        == 200
    )
    spec = spec_factory(name="Donors", experiment=RESAMPLE, **QUICK).model_dump(mode="json")
    run_id = client.post("/api/runs", json={"spec": spec}, headers=HEADERS).json()["run_id"]
    _settled(client)

    fields = client.get(f"/api/runs/{run_id}/sweep-fields", headers=HEADERS).json()
    by_field = {f["field"]: f for f in fields}
    assert set(by_field) == {
        "statistics.seed",
        "experiment.baseline.seed",
        "dataset.path",
        "model.revision",
    }
    assert by_field["experiment.baseline.seed"]["suggested"] == [0, 1, 2, 3, 4]
    # The run read a snapshot of ioi.jsonl: the app offers, and names, the file it came from.
    assert by_field["dataset.path"]["suggested"] == ["datasets/ioi.jsonl"]
    assert by_field["dataset.path"]["current"].startswith("datasets/snapshots/")
    assert by_field["dataset.path"]["current_label"] == "ioi.jsonl"
    assert by_field["model.revision"]["suggested"] == ["test"]

    body = {"run": run_id, "field": "experiment.baseline.seed", "values": [0, 1, 2]}
    plan = client.post("/api/sweeps/plan", json=body, headers=HEADERS).json()
    assert plan["stands_in"] == 0 and [r["original"] for r in plan["runs"]] == [True, False, False]
    assert len(_settled(client)) == 1  # a plan runs nothing
    bad = client.post("/api/sweeps/plan", json={**body, "values": [0, 0]}, headers=HEADERS)
    assert bad.status_code == 400 and "listed twice" in bad.json()["error"]
    both = client.post("/api/sweeps", json={**body, "spec": spec}, headers=HEADERS)
    assert both.status_code == 422 and "either a run or a spec" in both.text

    started = client.post("/api/sweeps", json=body, headers=HEADERS).json()
    assert started["run_id"] and started["job"]["status"] == "running"
    runs = _settled(client)
    members = [r for r in runs.values() if (r["derived_from"] or {}).get("kind") == "sweep"]
    assert sorted(r["derived_from"]["index"] for r in members) == [1, 2]
    assert all(
        r["derived_from"]["sweep"] == started["sweep"] and r["status"] == "finished"
        for r in members
    )

    table = client.get(f"/api/sweeps/{started['sweep']}?choose=all&top=3", headers=HEADERS)
    assert table.status_code == 200, table.text
    data = table.json()
    assert data["finished"] == 3 and len(data["sites"]) == 3 and data["choose"] == "all"
    assert data["values"][0]["run"] == run_id and data["values"][0]["original"]
    assert data["experiment"]["kind"] == "ablation"
    exported = client.get(
        f"/api/sweeps/{started['sweep']}/table.csv?choose=all&top=3", headers=HEADERS
    )
    assert exported.headers["content-type"].startswith("text/csv")
    assert f'filename="sweep-{started["sweep"]}.csv"' in exported.headers["content-disposition"]
    assert len(list(csv.DictReader(io.StringIO(exported.text)))) == 3 * 3
    missing = client.get("/api/sweeps/nope", headers=HEADERS)
    assert missing.status_code == 409 and "There is no sweep nope" in missing.json()["error"]
    # A tab must say which project it shows, as for every route of the project.
    assert _project_scoped(f"/api/sweeps/{started['sweep']}") and _project_scoped("/api/sweeps")


def test_revisions_are_listed_with_checkpoints_in_training_order(monkeypatch, app):
    class FakeApi:
        def list_repo_refs(self, model_id: str) -> Any:
            branch = lambda name: SimpleNamespace(name=name, target_commit=name[-1] * 40)  # noqa: E731
            return SimpleNamespace(
                branches=[branch(n) for n in ("step10", "main", "step2", "step1000", "step143000")],
                tags=[branch(n) for n in ("v1", "v0.2")],
            )

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    names = [r.name for r in hub.list_revisions("EleutherAI/pythia-70m")]
    assert names == ["main", "step2", "step10", "step1000", "step143000", "v0.2", "v1"]
    client = TestClient(app, base_url=BASE)
    out = client.get("/api/models/revisions?id=EleutherAI/pythia-70m", headers=HEADERS).json()
    assert [r["name"] for r in out["revisions"]][:3] == ["main", "step2", "step10"]
    assert out["revisions"][1] == {"name": "step2", "commit": "2" * 40, "kind": "branch"}

    class Offline:
        def list_repo_refs(self, model_id: str) -> Any:
            raise ConnectionError("Failed to connect")

    monkeypatch.setattr("huggingface_hub.HfApi", Offline)
    with pytest.raises(BackendError, match="Type the revisions instead"):
        hub.list_revisions("EleutherAI/pythia-70m")
