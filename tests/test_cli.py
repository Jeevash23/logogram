"""The command line, and that it reproduces the app's results exactly."""

from __future__ import annotations

import time

import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from logogram import __version__
from logogram.cli import app as cli
from logogram.server.app import create_app
from logogram.server.security import SecurityConfig

AUTH = {"authorization": "Bearer t", "origin": "http://127.0.0.1:8765"}


def test_version():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"logogram {__version__}"


def test_doctor_reports_the_backend():
    result = CliRunner().invoke(cli, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "Compute backend" in result.output
    assert "Recommended precision" in result.output


def test_run_refuses_a_missing_spec(tmp_path):
    result = CliRunner().invoke(cli, ["run", str(tmp_path / "nope.json")])
    assert result.exit_code == 1
    assert "doesn't exist" in result.output


def test_cli_reproduces_the_app_exactly(tiny_backend, tmp_path, monkeypatch, spec_factory):
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    # Start the run from the app's API, as the GUI does.
    application = create_app(SecurityConfig(token="t", port=8765), serve_web=False)
    application.state.logogram.backend = tiny_backend
    client = TestClient(application, base_url="http://127.0.0.1:8765")
    project = client.post(
        "/api/projects", json={"name": "Repro", "parent": str(tmp_path)}, headers=AUTH
    )
    assert project.status_code == 200, project.text
    client.post("/api/datasets/ioi", json={"name": "ioi", "n": 8, "seed": 3}, headers=AUTH)
    spec = spec_factory(
        experiment={
            "kind": "ablation",
            "baseline": {"kind": "resample", "pool": "corrupt", "donors": 2, "seed": 1},
        }
    ).model_dump(mode="json")
    run_id = client.post("/api/runs", json={"spec": spec}, headers=AUTH).json()["run_id"]
    for _ in range(600):
        listing = client.get("/api/runs", headers=AUTH).json()
        if next(r["status"] for r in listing if r["id"] == run_id) == "finished":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("the app's run did not finish")

    # Rerun the spec it saved, headlessly.
    root = tmp_path / "repro"
    monkeypatch.setattr(
        "logogram.runner.default_provider", lambda on_event=None: lambda ref: tiny_backend
    )
    result = CliRunner().invoke(cli, ["run", str(root / "experiments" / run_id / "spec.json")])
    assert result.exit_code == 0, result.output
    assert f"Identical to {run_id}" in result.output
    runs = sorted(p for p in (root / "experiments").iterdir() if p.name != run_id)
    assert len(runs) == 1
    a = pq.read_table(root / "experiments" / run_id / "results.parquet")
    b = pq.read_table(runs[0] / "results.parquet")
    assert a.equals(b)


@pytest.mark.parametrize(
    ("experiment", "scope"),
    [
        (
            {"kind": "direct_logit_attribution", "prompts": "clean"},
            {"kind": "heads", "position": {"kind": "last"}},
        ),
        (
            {"kind": "attribution_patching", "direction": "clean_to_corrupt"},
            {"kind": "heads", "position": {"kind": "all"}},
        ),
    ],
)
def test_a_rerun_of_a_method_that_leaves_values_unmeasured_is_identical(
    tiny_backend, project, monkeypatch, spec_factory, experiment, scope
):
    # Direct effects and estimates have no patched probability (NaN), which must match itself.
    from logogram.runner import run_spec

    monkeypatch.setattr("logogram.project.config_dir", lambda: project.root / "config")
    first = run_spec(
        spec_factory(experiment=experiment, scope=scope), project, backend=tiny_backend
    )
    assert first.status == "finished"
    monkeypatch.setattr(
        "logogram.runner.default_provider", lambda on_event=None: lambda ref: tiny_backend
    )
    result = CliRunner().invoke(cli, ["run", str(first.folder / "spec.json")])
    assert result.exit_code == 0, result.output
    assert f"Identical to {first.run_id}" in result.output


def test_open_refuses_a_folder_without_a_project(tmp_path):
    result = CliRunner().invoke(cli, ["open", str(tmp_path)])
    assert result.exit_code == 1
    assert "isn't a Logogram project" in result.output


def test_server_can_start_with_a_project_open(project, tmp_path, monkeypatch):
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    application = create_app(
        SecurityConfig(token="t", port=8765), initial_project=project.root, serve_web=False
    )
    client = TestClient(application, base_url="http://127.0.0.1:8765")
    state = client.get("/api/state", headers=AUTH).json()
    assert state["project"]["name"] == "Test project"


def test_ports_are_checked_and_never_overflow():
    from logogram.cli import _bind

    for preferred in (0, 65535):
        sock = _bind(preferred)
        try:
            assert 0 < sock.getsockname()[1] <= 65535
        finally:
            sock.close()
    result = CliRunner().invoke(cli, ["--port", "70000", "--no-browser"])
    assert result.exit_code == 2  # refused before anything starts


def test_a_taken_port_moves_to_the_next_one():
    from logogram.cli import _bind

    first = _bind(0)
    try:
        first.listen()
        port = first.getsockname()[1]
        second = _bind(port)
        try:
            assert second.getsockname()[1] != port
        finally:
            second.close()
    finally:
        first.close()


def test_the_summary_names_each_method_s_values(
    tiny_backend, project, spec_factory, capsys, tmp_path
):
    from logogram.cli import _print_summary
    from logogram.runner import run_spec
    from test_sae import exact_sae

    sae = exact_sae(tmp_path / "sae", tiny_backend.info.d_model)
    cases = [
        (
            {
                "experiment": {"kind": "direct_logit_attribution", "prompts": "clean"},
                "scope": {"kind": "heads", "position": {"kind": "last"}},
            },
            ["share", "splits into attention"],
        ),
        (
            {"experiment": {"kind": "attribution_patching", "direction": "clean_to_corrupt"}},
            ["estimated"],
        ),
        (
            {
                "experiment": {
                    "kind": "steering",
                    "apply_to": "corrupt",
                    "coefficients": [1.0],
                    "train_fraction": 0.5,
                    "seed": 0,
                    "control": True,
                },
                "scope": {
                    "kind": "layer_components",
                    "components": ["resid_pre"],
                    "position": {"kind": "last"},
                },
            },
            ["flipped", "held-out pairs"],
        ),
        (
            {
                "sae": {"repo": "local/exact", "path": "", "revision": "test"},
                "experiment": {"kind": "attribution_patching", "direction": "clean_to_corrupt"},
                "scope": {"kind": "features", "position": {"kind": "last"}, "top": 3},
            },
            ["explains 100.0%", "the features carry"],
        ),
    ]
    for overrides, expected in cases:
        outcome = run_spec(spec_factory(**overrides), project, backend=tiny_backend, sae=sae)
        assert outcome.status == "finished", outcome.manifest.get("error")
        _print_summary(outcome.summary, 3)
        out = capsys.readouterr().out
        for text in expected:
            assert text in out, (text, out)
        if overrides["experiment"]["kind"] != "steering":
            assert "flipped" not in out
