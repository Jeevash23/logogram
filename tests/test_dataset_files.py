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


def _ran(project: Any, tiny_backend: Any, spec_factory: Any) -> Any:
    from logogram.runner import run_spec

    outcome = run_spec(_quick(spec_factory, "Ran"), project, backend=tiny_backend)
    assert outcome.status == "finished", outcome.manifest.get("error")
    return outcome


def test_a_dataset_is_renamed_and_deleted_in_the_app(
    app, client, project, tiny_backend, spec_factory
):
    ran = _ran(project, tiny_backend, spec_factory)
    sha = file_sha256(project.datasets_dir / "ioi.jsonl")
    tab = {**HEADERS, "x-logogram-project": app.state.logogram.project.session_id}
    seen: list[tuple[str, dict[str, Any]]] = []
    publish = app.state.logogram.hub.publish
    app.state.logogram.hub.publish = lambda kind, data: (
        seen.append((kind, data)),
        publish(kind, data),
    )

    renamed = client.post("/api/datasets/ioi.jsonl/rename", json={"name": "names"}, headers=tab)
    assert renamed.status_code == 200, renamed.text
    assert renamed.json() == {"name": "names.jsonl", "path": "datasets/names.jsonl"}
    assert not (project.datasets_dir / "ioi.jsonl").exists()
    assert file_sha256(project.datasets_dir / "names.jsonl") == sha  # the same prompts
    session = app.state.logogram.project.session_id
    assert ("project.changed", {"project_session": session}) in seen
    names = [d["name"] for d in client.get("/api/project", headers=tab).json()["datasets"]]
    assert names == ["mixed.jsonl", "names.jsonl"]
    taken = client.post("/api/datasets/names/rename", json={"name": "mixed"}, headers=tab)
    assert taken.status_code == 400
    assert taken.json()["error"] == "mixed.jsonl already exists. Choose another name."
    bad = client.post("/api/datasets/names/rename", json={"name": "../x"}, headers=tab)
    assert bad.status_code == 400 and "Dataset names can use letters" in bad.json()["error"]

    deleted = client.delete("/api/datasets/names.jsonl", headers=tab)
    assert deleted.status_code == 200 and deleted.json()["path"] == "datasets/names.jsonl"
    assert not (project.datasets_dir / "names.jsonl").exists()
    missing = client.delete("/api/datasets/names.jsonl", headers=tab)
    assert missing.status_code == 400
    assert missing.json()["error"] == "This project has no dataset called names.jsonl."
    # The run that ran on it is untouched: it reads its own copy.
    detail = client.get(f"/api/runs/{ran.run_id}/sites/0", headers=tab).json()
    assert detail["prompts"][0]["clean"]
    assert lg.load_run(project, ran.run_id).status == "finished"

    # A draft that names a dataset keeps it from being renamed or deleted.
    draft = _draft(project, _quick(spec_factory, "Mixed", "datasets/mixed.jsonl"))
    for refused in (
        client.delete("/api/datasets/mixed.jsonl", headers=tab),
        client.post("/api/datasets/mixed.jsonl/rename", json={"name": "other"}, headers=tab),
    ):
        assert refused.status_code == 409
        assert f"the draft “Mixed” (experiments/{draft}) names it" in refused.json()["error"]
    assert (project.datasets_dir / "mixed.jsonl").is_file()


def test_datasets_are_listed_renamed_and_deleted_from_the_command_line_and_python(
    project, spec_factory
):
    where = ["--project", str(project.root)]
    _draft(project, _quick(spec_factory, "Heads"))
    listed = runner.invoke(cli, ["datasets", *where])
    assert listed.exit_code == 0, listed.output
    assert listed.output.splitlines() == [
        "ioi.jsonl    12 prompt pairs",
        "             named by the draft “Heads” (experiments/" + project.list_runs()[0].id + ")",
        "mixed.jsonl  12 prompt pairs",
    ]
    refused = runner.invoke(cli, ["datasets", "delete", "ioi.jsonl", *where])
    assert refused.exit_code == 1 and "can't be deleted" in refused.output
    assert "Delete it?" not in refused.output  # refused before asking
    kept = runner.invoke(cli, ["datasets", "delete", "mixed", *where], input="n\n")
    assert kept.exit_code == 1 and "Nothing was deleted." in kept.output
    assert "Runs that ran on it keep their own copy of its prompts" in " ".join(kept.output.split())
    renamed = runner.invoke(cli, ["datasets", "rename", "mixed", "templates", *where])
    assert renamed.exit_code == 0, renamed.output
    assert "Renamed datasets/mixed.jsonl to datasets/templates.jsonl." in renamed.output
    deleted = runner.invoke(cli, ["datasets", "delete", "templates.jsonl", "--yes", *where])
    assert deleted.exit_code == 0 and "Deleted datasets/templates.jsonl." in deleted.output
    assert sorted(p.name for p in project.datasets_dir.glob("*.jsonl")) == ["ioi.jsonl"]

    # Python
    with pytest.raises(LogogramError, match="can't be renamed: the draft “Heads”"):
        lg.rename_dataset(project, "ioi", "names")
    lg.write_dataset(project, "spare", generate_ioi(3, seed=1))
    assert lg.rename_dataset(project, "spare", "extra") == "datasets/extra.jsonl"
    assert lg.delete_dataset(project, "extra.jsonl") == "datasets/extra.jsonl"
    with pytest.raises(LogogramError, match=r"no dataset called extra\.jsonl"):
        lg.delete_dataset(project, "extra")


def test_a_dataset_that_is_a_link_is_deleted_as_a_link(project, tmp_path):
    import sys

    if sys.platform == "win32":
        pytest.skip("POSIX symlinks")
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes((project.datasets_dir / "ioi.jsonl").read_bytes())
    (project.datasets_dir / "linked.jsonl").symlink_to(outside)
    assert lg.delete_dataset(project, "linked") == "datasets/linked.jsonl"
    assert outside.is_file()  # what it pointed to stays
