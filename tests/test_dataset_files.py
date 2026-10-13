"""A project's dataset files: replacing, renaming and deleting them, from the app, the command line
and Python. A saved draft or a queued run that names a file refuses all three, since it would fail
or run on other prompts; finished runs keep their own copy of the prompts they ran on."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import logogram as lg
from logogram import runqueue
from logogram.api import LogogramError
from logogram.cli import app as cli
from logogram.datasets import dataset_users, file_sha256
from logogram.ioi import generate_ioi
from logogram.runner import write_json
from logogram.server.app import create_app
from logogram.server.security import SecurityConfig

TOKEN = "test-token"
BASE = "http://127.0.0.1:8765"
HEADERS = {"authorization": f"Bearer {TOKEN}", "origin": BASE}
runner = CliRunner()


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    return create_app(SecurityConfig(token=TOKEN, port=8765), serve_web=False)


@pytest.fixture
def client(app, project):
    client = TestClient(app, base_url=BASE)
    r = client.post("/api/projects/open", json={"path": str(project.root)}, headers=HEADERS)
    assert r.status_code == 200, r.text
    return client


def _quick(spec_factory: Any, name: str, dataset: str = "datasets/ioi.jsonl") -> Any:
    return spec_factory(
        name=name,
        dataset={"path": dataset, "limit": 4},
        scope={"kind": "layer_components", "position": {"kind": "last"}},
    )


def _draft(project: Any, spec: Any) -> str:
    """Save ``spec`` as a draft, as the app's Save does."""
    run_id, folder = project.new_run_dir(spec.name)
    write_json(folder / "spec.json", spec.model_dump(mode="json"))
    return run_id


def test_what_waits_on_a_dataset_file(project, spec_factory):
    assert dataset_users(project, "datasets/ioi.jsonl") == []
    draft = _draft(project, _quick(spec_factory, "Heads"))
    _draft(project, _quick(spec_factory, "Mixed", "datasets/mixed.jsonl"))
    runqueue.add_to_queue(project, _quick(spec_factory, "Seeds"))
    assert dataset_users(project, "datasets/ioi.jsonl") == [
        {"kind": "draft", "id": draft, "title": "Heads", "queued": False},
        {"kind": "queued", "id": runqueue.read_queue(project)[0].id, "title": "Seeds",
         "queued": True},
    ]  # fmt: skip
    # A queued draft is the draft, queued.
    runqueue.add_to_queue(project, _quick(spec_factory, "Heads"), draft_id=draft)
    users = dataset_users(project, "datasets/ioi.jsonl")
    assert [(u["kind"], u["queued"]) for u in users] == [("draft", True), ("queued", True)]


def test_a_dataset_a_draft_or_queued_run_names_is_never_replaced(client, project, spec_factory):
    path = project.datasets_dir / "ioi.jsonl"
    before = file_sha256(path)
    draft = _draft(project, _quick(spec_factory, "Heads"))
    runqueue.add_to_queue(project, _quick(spec_factory, "Seeds"))
    expected = (
        f"datasets/ioi.jsonl can't be replaced: the draft “Heads” (experiments/{draft}) and the "
        "queued run “Seeds” name it, and they would fail, or run on other prompts. Write the new "
        "prompts under another name, or first take it out of the queue and open the draft in the "
        "form to choose other prompts, or delete the draft."
    )
    # The command line,
    where = ["--project", str(project.root)]
    generated = runner.invoke(
        cli, ["generate", "ioi", "-n", "4", "--seed", "3", "--overwrite", *where]
    )
    assert generated.exit_code == 1 and expected in " ".join(generated.output.split())
    # Python,
    with pytest.raises(LogogramError) as refused:
        lg.write_dataset(project, "ioi", generate_ioi(4, seed=3), overwrite=True)
    assert str(refused.value) == expected
    # and the app.
    r = client.post(
        "/api/datasets/ioi", json={"name": "ioi", "n": 4, "seed": 3, "overwrite": True},
        headers=HEADERS,
    )  # fmt: skip
    assert r.status_code == 409 and r.json()["error"] == expected
    assert file_sha256(path) == before
    # Without them waiting on it, it is replaced, as asked.
    runqueue.clear_queue(project)
    assert lg.delete_run(project, draft) is None
    assert lg.write_dataset(project, "ioi", generate_ioi(4, seed=3), overwrite=True) == (
        "datasets/ioi.jsonl"
    )
    assert file_sha256(path) != before
