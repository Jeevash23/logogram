"""Run folders as earlier releases wrote them stay readable.

tests/fixtures/old-runs is a project holding what earlier releases wrote, made by running those
releases themselves on the tiny test model (conftest._build_tiny_model, unchanged since 0.1.0)
and the same six IOI prompts:

* made-by-0.1.0: a heads sweep that Logogram 0.1.0 ran from a version 1 spec, patching clean
  into corrupt. Its results have no metric columns: 0.1.0 measured the logit difference only.
* made-by-0.2.0: the same sweep patching the other way, run by 0.2.0 from a version 2 spec.
* written-for-0.1.0: a version 1 spec as a person wrote it for 0.1.0, leaving choices out. It
  hasn't run.

Each manifest's processor name was then replaced with a placeholder, so the files hold nothing
about the machine they were made on. The current code lists, opens, summarizes, compares and
exports them through the server, the Python API and the command line.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import logogram as lg
from logogram.cli import app as cli
from logogram.project import Project
from logogram.runner import run_spec
from logogram.server.app import create_app
from logogram.server.security import SecurityConfig
from logogram.spec import SPEC_VERSION, Spec

FIXTURE = Path(__file__).parent / "fixtures" / "old-runs"
# The tiny test model the fixture's runs were made on, by the sum of its weights and of their
# sizes. The model is built from a seed, and other versions of PyTorch or Transformers can build
# other weights from the same seed (the lowest declared versions do): no number can match then.
# Other platforms build the same weights to the last digits, which the comparison allows for.
FIXTURE_WEIGHTS = (174.96260210211085, 7916.290018921845)


def weights_totals(folder: Path) -> tuple[float, float]:
    from safetensors.torch import load_file

    tensors = load_file(folder / "model.safetensors")
    total = sum(float(t.double().sum()) for _, t in sorted(tensors.items()))
    size = sum(float(t.double().abs().sum()) for _, t in sorted(tensors.items()))
    return total, size


V1, V2, DRAFT = "made-by-0.1.0", "made-by-0.2.0", "written-for-0.1.0"
BASE = "http://127.0.0.1:8765"
AUTH = {"authorization": "Bearer test", "origin": BASE}
N_SITES, N_PROMPTS = 8, 6  # two layers of four heads, on six prompts
runner = CliRunner()


@pytest.fixture
def old(tmp_path):
    """A copy of the fixture project, so opening and running in it changes nothing committed."""
    root = tmp_path / "old-runs"
    shutil.copytree(FIXTURE, root)
    return Project.open(root)


@pytest.fixture
def client(old, tmp_path, monkeypatch):
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    app = create_app(SecurityConfig(token="test", port=8765), serve_web=False)
    with TestClient(app, base_url=BASE, headers=AUTH) as test_client:
        opened = test_client.post("/api/projects/open", json={"path": str(old.root)})
        assert opened.status_code == 200, opened.text
        yield test_client


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_fixtures_are_what_those_releases_wrote():
    runs = FIXTURE / "experiments"
    assert _json(runs / V1 / "spec.json")["logogram_spec"] == 1
    assert _json(runs / V1 / "manifest.json")["versions"]["logogram"] == "0.1.0"
    assert _json(runs / V2 / "spec.json")["logogram_spec"] == 2
    assert _json(runs / V2 / "manifest.json")["versions"]["logogram"] == "0.2.0"
    assert "patched_metric" not in pq.read_schema(runs / V1 / "results.parquet").names
    assert "patched_metric" in pq.read_schema(runs / V2 / "results.parquet").names
    assert "logogram_spec" not in _json(runs / DRAFT / "spec.json")
    for run_id in (V1, V2):
        assert _json(runs / run_id / "manifest.json")["device"]["name"] == "Example CPU"


def test_old_runs_list_open_and_summarize_through_the_server(client, old):
    listed = {r["id"]: r for r in client.get("/api/runs").json()}
    assert set(listed) == {V1, V2, DRAFT}
    assert listed[DRAFT]["status"] == "draft"
    for run_id in (V1, V2):
        listing = listed[run_id]
        assert listing["status"] == "finished" and listing["error"] is None
        assert listing["n_prompts"] == N_PROMPTS and listing["layout_kind"] == "heads"
        assert listing["profile"] is not None and len(listing["profile"]) == 2
        assert listing["elsewhere"] is False

        detail = client.get(f"/api/runs/{run_id}").json()
        assert detail["spec"]["logogram_spec"] == SPEC_VERSION  # read as this version's spec
        assert detail["summary"]["n_sites"] == N_SITES
        assert detail["manifest"]["status"] == "finished" and detail["traceback"] is None
        site = client.get(f"/api/runs/{run_id}/sites/0").json()
        assert len(site["prompts"]) == N_PROMPTS
        assert site["dataset_available"] and not site["dataset_changed"]
    # 0.1.0 measured the logit difference only: it reads as the run's metric.
    first = client.get(f"/api/runs/{V1}/sites/0").json()["prompts"][0]
    assert first["patched_metric"] is not None
    assert first["patched_metric"] == first["patched_logit_diff"]

    draft = client.get(f"/api/runs/{DRAFT}").json()
    assert draft["spec"]["model"]["dtype"] == "float32" and draft["summary"] is None


def test_old_runs_compare_and_export_through_the_server(client):
    comparison = client.get("/api/compare", params={"a": V1, "b": V2})
    assert comparison.status_code == 200, comparison.text
    result = comparison.json()
    assert result["n_common"] == N_SITES and result["paired"] is True
    assert [d["path"] for d in result["spec_differences"]] == ["experiment.direction"]
    assert result["environment_differences"] == [
        {"key": "logogram", "label": "Logogram", "a": "0.1.0", "b": "0.2.0"}
    ]
    for run_id in (V1, V2):
        response = client.get(f"/api/runs/{run_id}/export.csv")
        assert response.status_code == 200
        rows = list(csv.DictReader(io.StringIO(response.text)))
        assert len(rows) == N_SITES * N_PROMPTS
        assert {"site", "prompt", "effect", "delta"} <= set(rows[0])


def test_old_runs_through_the_python_api(old):
    listed = {r["id"]: r["status"] for r in lg.list_runs(old)}
    assert listed == {V1: "finished", V2: "finished", DRAFT: "draft"}
    runs = {run_id: lg.load_run(old, run_id) for run_id in (V1, V2)}
    for run in runs.values():
        assert run.status == "finished" and run.spec.logogram_spec == SPEC_VERSION
        assert len(run.sites()) == N_SITES
        assert len(run.per_prompt()) == N_SITES * N_PROMPTS
        assert len(run.site(0)["prompts"]) == N_PROMPTS
        assert len(run.strongest(3)) == 3
        assert run.spec.name in run._repr_html_()
        assert run.traceback is None
    comparison = lg.compare_runs(runs[V1], runs[V2])
    assert comparison["paired"] and comparison["n_common"] == N_SITES
    assert [d["key"] for d in comparison["environment_differences"]] == ["logogram"]


def test_old_runs_through_the_command_line(old, tmp_path):
    where = ["--project", str(old.root)]
    listed = runner.invoke(cli, ["list", *where])
    assert listed.exit_code == 0 and V1 in listed.output and DRAFT in listed.output
    for run_id in (V1, V2):
        shown = runner.invoke(cli, ["show", run_id, *where])
        assert shown.exit_code == 0, shown.output
        assert "Normalized effect" in shown.output and "L1 H0" in shown.output
        # Corrections for the number of sites came with 0.2.0; 0.1.0's summary has none to show.
        assert ("Corrected for 8 sites" in shown.output) == (run_id == V2)
        out = tmp_path / f"{run_id}.csv"
        exported = runner.invoke(cli, ["export", run_id, "-o", str(out), *where])
        assert exported.exit_code == 0, exported.output
        assert len(out.read_text(encoding="utf-8").splitlines()) == N_SITES * N_PROMPTS + 1
    compared = runner.invoke(cli, ["compare", V1, V2, *where])
    assert compared.exit_code == 0, compared.output
    assert (
        "Paired prompt by prompt" in compared.output
        and "Logogram: 0.1.0 → 0.2.0" in compared.output
    )
    validated = runner.invoke(cli, ["validate", str(old.run_dir(DRAFT) / "spec.json")])
    assert validated.exit_code == 0
    assert "This is a version 1 spec" in validated.output and "model.dtype" in validated.output


def test_old_specs_still_run(old, tiny_backend, tiny_model_dir):
    """The 0.1.0 run's spec, and the version 1 spec written for 0.1.0, run as they are. The run
    of the hand-written one says which choices it left out."""
    rerun = run_spec(Spec.from_path(old.run_dir(V1) / "spec.json"), old, backend=tiny_backend)
    assert rerun.status == "finished", rerun.manifest.get("error")
    comparison = lg.compare_runs(lg.load_run(old, V1), lg.load_run(old, rerun.run_id))
    assert comparison["paired"] and comparison["n_common"] == N_SITES
    old_effects = pq.read_table(old.run_dir(V1) / "results.parquet").column("effect").to_pylist()
    new_effects = pq.read_table(rerun.folder / "results.parquet").column("effect").to_pylist()

    draft = Spec.from_path(old.run_dir(DRAFT) / "spec.json")
    assert 'model.dtype = "float32"' in draft.upgraded_fields
    written = run_spec(draft, old, run_id=DRAFT, backend=tiny_backend)
    assert written.status == "finished", written.manifest.get("error")
    assert written.summary is not None and "version 1" in written.summary["warnings"][0]

    if weights_totals(tiny_model_dir) != pytest.approx(FIXTURE_WEIGHTS, rel=1e-6):
        pytest.skip(
            "These versions of PyTorch and Transformers build another tiny model from the same "
            "seed, so the numbers can't be compared with runs made on the fixture's model."
        )
    # On the same model, other library versions may change the last digits, never more.
    assert new_effects == pytest.approx(old_effects, abs=1e-4)
