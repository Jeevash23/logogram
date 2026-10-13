"""Managing runs: titles, stars and the archive (annotations.json beside a run, never its spec),
deleting runs, and runs whose spec can't be read, which are listed, saying why, and can be
deleted."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import logogram as lg
from logogram import runqueue
from logogram.api import LogogramError
from logogram.cli import app as cli
from logogram.history import ANNOTATIONS, annotate_run, delete_run
from logogram.project import Project, ProjectError
from logogram.runner import run_spec
from logogram.server.app import create_app
from logogram.server.security import SecurityConfig
from logogram.server.state import Job

TOKEN = "test-token"
BASE = "http://127.0.0.1:8765"
HEADERS = {"authorization": f"Bearer {TOKEN}", "origin": BASE}

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks and pipes")


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    return create_app(SecurityConfig(token=TOKEN, port=8765), serve_web=False)


@pytest.fixture
def client(app):
    return TestClient(app, base_url=BASE)


@pytest.fixture
def ready(app, client, project, tiny_backend):
    """The test project open and the tiny model loaded."""
    state = app.state.logogram
    state.backend = tiny_backend
    state.model_status = {"state": "ready"}
    r = client.post("/api/projects/open", json={"path": str(project.root)}, headers=HEADERS)
    assert r.status_code == 200, r.text
    return state


@pytest.fixture
def events(ready, monkeypatch):
    """Every event the server publishes from here on, as (kind, data)."""
    seen: list[tuple[str, dict[str, Any]]] = []
    publish = ready.hub.publish

    def record(kind: str, data: dict[str, Any]) -> None:
        seen.append((kind, data))
        publish(kind, data)

    monkeypatch.setattr(ready.hub, "publish", record)
    return seen


def _quick(spec_factory, name: str = "Quick", **overrides: Any) -> Any:
    """A small, fast run of the tiny model."""
    settings = {
        "name": name,
        "dataset": {"path": "datasets/ioi.jsonl", "limit": 4},
        "scope": {"kind": "layer_components", "position": {"kind": "last"}},
        "statistics": {"bootstrap": 100, "seed": 0},
    }
    return spec_factory(**{**settings, **overrides})


def _finished(project, tiny_backend, spec_factory, name: str = "Quick", **kwargs: Any) -> str:
    outcome = run_spec(_quick(spec_factory, name), project, backend=tiny_backend, **kwargs)
    assert outcome.status == "finished", outcome.manifest.get("error")
    return outcome.run_id


def _listed(client) -> dict[str, dict[str, Any]]:
    response = client.get("/api/runs", headers=HEADERS)
    assert response.status_code == 200, response.text
    return {r["id"]: r for r in response.json()}


def _patch(client, run_id: str, **body: Any):
    return client.patch(f"/api/runs/{run_id}", json=body, headers=HEADERS)


def _annotations(project, run_id: str) -> dict[str, Any]:
    return json.loads((project.run_dir(run_id) / ANNOTATIONS).read_text(encoding="utf-8"))


# -- titles, stars and the archive -------------------------------------------------------------


def test_a_run_is_renamed_starred_and_archived_beside_its_spec(
    client, ready, project, tiny_backend, spec_factory, events
):
    run_id = _finished(project, tiny_backend, spec_factory, "Heads sweep")
    spec = project.run_dir(run_id) / "spec.json"
    ran = spec.read_bytes()

    r = _patch(client, run_id, title="  Name   movers,\n clean run ", starred=True)
    assert r.status_code == 200, r.text
    assert r.json()["title"] == "Name movers, clean run" and r.json()["starred"] is True
    assert r.json()["name"] == "Heads sweep" and r.json()["status"] == "finished"
    assert _annotations(project, run_id) == {
        "logogram_annotations": 1,
        "title": "Name movers, clean run",
        "starred": True,
        "archived": False,
    }
    assert spec.read_bytes() == ran  # what ran never changes
    listed = _listed(client)[run_id]
    assert (listed["title"], listed["starred"], listed["archived"]) == (
        "Name movers, clean run",
        True,
        False,
    )
    assert listed["annotations_error"] is None and listed["damaged"] is False
    detail = client.get(f"/api/runs/{run_id}", headers=HEADERS).json()
    assert detail["listing"]["title"] == "Name movers, clean run"
    assert detail["spec"]["name"] == "Heads sweep"
    # Every open tab hears of it, and reads the history again.
    session = ready.project.session_id
    assert ("runs.changed", {"project_session": session, "run_id": run_id, "deleted": False}) in (
        events
    )

    # What is left out stays as it is.
    archived = _patch(client, run_id, archived=True).json()
    assert (archived["title"], archived["starred"], archived["archived"]) == (
        "Name movers, clean run",
        True,
        True,
    )
    assert _patch(client, run_id, starred=None).json()["starred"] is True
    # The spec's own name, or nothing, shows the spec's name: there is no title to keep.
    assert _patch(client, run_id, title="Heads sweep").json()["title"] is None
    assert _patch(client, run_id, title="Again").json()["title"] == "Again"
    assert _patch(client, run_id, title="   ").json()["title"] is None
    assert _patch(client, run_id, title="Again").json()["title"] == "Again"
    assert _patch(client, run_id, title=None).json()["title"] is None
    # Once nothing is said about the run, its file goes.
    cleared = _patch(client, run_id, starred=False, archived=False).json()
    assert (cleared["title"], cleared["starred"], cleared["archived"]) == (None, False, False)
    assert not os.path.lexists(project.run_dir(run_id) / ANNOTATIONS)
    assert spec.read_bytes() == ran

    # A title is one line of a sensible length; only these three things can be said.
    long = _patch(client, run_id, title="x" * 201)
    assert long.status_code == 400 and "at most 200 characters" in long.json()["error"]
    assert _patch(client, run_id, title="y" * 200).status_code == 200
    control = _patch(client, run_id, title="bell\x07")
    assert control.status_code == 400 and "one line of text" in control.json()["error"]
    extra = _patch(client, run_id, name="Renamed spec")
    assert extra.status_code == 422 and "extra" in extra.text
    assert _patch(client, run_id, starred="maybe").status_code == 422
    assert spec.read_bytes() == ran

    missing = _patch(client, "20990101-000000-gone", starred=True)
    assert missing.status_code == 409 and "no run" in missing.json()["error"]
    assert _patch(client, ".hidden", starred=True).status_code == 400


def test_drafts_and_running_runs_can_be_annotated(client, ready, project, spec_factory):
    draft = ready.save_draft(_quick(spec_factory, "Draft"))
    r = _patch(client, draft, title="Next week", starred=True)
    assert r.status_code == 200 and r.json()["status"] == "draft"
    assert r.json()["title"] == "Next week"
    # A draft saved again keeps what is said about it: that isn't part of its spec.
    ready.save_draft(_quick(spec_factory, "Draft, edited"), draft)
    listed = _listed(client)[draft]
    assert listed["name"] == "Draft, edited" and listed["title"] == "Next week"
    ready.job = Job(id="busy", kind="run", title="Draft", run_id=draft)
    assert _patch(client, draft, archived=True).status_code == 200


@pytest.mark.parametrize(
    ("content", "why"),
    [
        ("{", "Invalid JSON"),
        ('{"starred": "yes"}', "starred"),
        ('{"stars": 5}', "stars"),
        ('{"title": "two\\nlines"}', "one line of text"),
        ('{"title": ""}', "title"),
        ("[]", ""),
        ('{"logogram_annotations": 2}', "logogram_annotations"),
        ("[" * 100_000, ""),
    ],
    # Named, so the deeply nested one's id isn't 100,000 characters (too long for Windows).
    ids=["not JSON", "a string", "a field", "two lines", "empty", "a list", "a version", "nested"],
)
def test_annotations_that_cant_be_read_are_ignored_with_a_note(
    client, ready, project, spec_factory, content, why
):
    draft = ready.save_draft(_quick(spec_factory, "Draft"))
    path = project.run_dir(draft) / ANNOTATIONS
    path.write_text(content, encoding="utf-8")
    listed = _listed(client)[draft]
    assert listed["status"] == "draft" and listed["name"] == "Draft"
    assert (listed["title"], listed["starred"], listed["archived"]) == (None, False, False)
    note = listed["annotations_error"]
    assert note.startswith(f"Its {ANNOTATIONS} can't be read") and why in note
    assert note.endswith("Correct the file, or delete it.")
    # It is never replaced: it may hold what someone wrote.
    refused = _patch(client, draft, starred=True)
    assert refused.status_code == 400 and "won't replace it" in refused.json()["error"]
    assert path.read_text(encoding="utf-8") == content
    path.unlink()
    assert _patch(client, draft, starred=True).json()["starred"] is True


def test_annotations_too_large_or_not_a_file_are_ignored(
    client, ready, project, spec_factory, monkeypatch
):
    draft = ready.save_draft(_quick(spec_factory, "Draft"))
    path = project.run_dir(draft) / ANNOTATIONS
    path.write_text(json.dumps({"title": "x" * 150, "starred": True}), encoding="utf-8")
    assert _listed(client)[draft]["title"] == "x" * 150
    monkeypatch.setattr("logogram.history.ANNOTATIONS_LIMIT", 64)
    listed = _listed(client)[draft]
    assert listed["title"] is None and "larger than the 64 bytes" in listed["annotations_error"]
    monkeypatch.undo()
    path.unlink()
    path.mkdir()
    listed = _listed(client)[draft]
    assert "isn't a regular file" in listed["annotations_error"]
    assert _patch(client, draft, starred=True).status_code == 400
    assert path.is_dir()


@posix_only
def test_annotations_are_never_read_or_written_through_links(
    client, ready, project, spec_factory, tmp_path
):
    outside = tmp_path / "outside.json"
    outside.write_text('{"title": "From outside", "starred": true}', encoding="utf-8")
    draft = ready.save_draft(_quick(spec_factory, "Draft"))
    path = project.run_dir(draft) / ANNOTATIONS
    path.symlink_to(outside)
    listed = _listed(client)[draft]
    assert listed["title"] is None and "outside the project" in listed["annotations_error"]
    assert "From outside" not in json.dumps(listed)
    assert _patch(client, draft, title="Mine").status_code == 400
    assert outside.read_text(encoding="utf-8") == '{"title": "From outside", "starred": true}'

    # A link to another run's annotations inside the project is read, and replaced (as a link)
    # when they change: the file it points to is never written through it.
    other = ready.save_draft(_quick(spec_factory, "Other"))
    assert _patch(client, other, title="Other's title").status_code == 200
    shared = project.run_dir(other) / ANNOTATIONS
    before = shared.read_bytes()
    path.unlink()
    path.symlink_to(shared)
    assert _listed(client)[draft]["title"] == "Other's title"
    assert _patch(client, draft, title="Mine").json()["title"] == "Mine"
    assert not path.is_symlink() and shared.read_bytes() == before
    assert _listed(client)[other]["title"] == "Other's title"
    # And removing them removes the link, never what it points to.
    path.unlink()
    path.symlink_to(shared)
    assert _patch(client, draft, title=None).json()["title"] is None
    assert not os.path.lexists(path) and shared.read_bytes() == before

    # A pipe planted as the file can't hang the history.
    path_pipe = project.run_dir(other) / ANNOTATIONS
    path_pipe.unlink()
    os.mkfifo(path_pipe)
    listed = _listed(client)[other]
    assert "isn't a regular file" in listed["annotations_error"]

    # A run folder that links out of the project is neither read nor written.
    folder = tmp_path / "elsewhere"
    folder.mkdir()
    (folder / "spec.json").write_text(_quick(spec_factory).model_dump_json(), encoding="utf-8")
    (project.experiments_dir / "linked").symlink_to(folder, target_is_directory=True)
    assert "linked" not in _listed(client)
    r = _patch(client, "linked", title="Mine")
    assert r.status_code == 400 and "outside the project" in r.json()["error"]
    assert sorted(p.name for p in folder.iterdir()) == ["spec.json"]


# -- deleting runs -----------------------------------------------------------------------------


def test_a_run_is_deleted_and_what_referred_to_it_stays(
    client, ready, project, tiny_backend, spec_factory, events
):
    original = _finished(project, tiny_backend, spec_factory, "Original")
    rerun = _finished(
        project,
        tiny_backend,
        spec_factory,
        "Original",
        derived_from={"run": original, "kind": "rerun"},
    )
    keep = _finished(project, tiny_backend, spec_factory, "Kept")
    assert _patch(client, original, title="To go", starred=True).status_code == 200
    note = {
        "title": "Seen in the original",
        "body": "",
        "model": _quick(spec_factory).model.model_dump(mode="json"),
        "sites": [{"kind": "attn_out", "layer": 0, "head": None, "position": {"kind": "last"}}],
        "run_id": original,
    }
    saved = client.post("/api/research", json=note, headers=HEADERS).json()
    snapshots = sorted(p.name for p in (project.datasets_dir / "snapshots").iterdir())

    r = client.delete(f"/api/runs/{original}", headers=HEADERS)
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert not os.path.lexists(project.run_dir(original))
    session = ready.project.session_id
    assert ("runs.changed", {"project_session": session, "run_id": original, "deleted": True}) in (
        events
    )
    listed = _listed(client)
    assert original not in listed and {rerun, keep} <= set(listed)
    # A run derived from it stays, and still names it, though it can't be opened.
    assert listed[rerun]["derived_from"]["run"] == original
    assert client.get(f"/api/runs/{rerun}", headers=HEADERS).status_code == 200
    gone = client.get(f"/api/runs/{original}", headers=HEADERS)
    assert gone.status_code == 409 and "no run" in gone.json()["error"]
    compared = client.get("/api/compare", params={"a": original, "b": rerun}, headers=HEADERS)
    assert compared.status_code == 400 and "may have been deleted" in compared.json()["error"]
    assert client.get(f"/api/runs/{original}/derived", headers=HEADERS).json()[0]["id"] == rerun
    # Dataset snapshots belong to every run that read them: they stay.
    assert sorted(p.name for p in (project.datasets_dir / "snapshots").iterdir()) == snapshots
    # A research note keeps the run it came from, and can still be edited.
    edit = {**note, "body": "Still worth checking.", "revision": saved["revision"]}
    edited = client.put(f"/api/research/{saved['id']}", json=edit, headers=HEADERS)
    assert edited.status_code == 200, edited.text
    assert edited.json()["run_id"] == original
    new = client.post("/api/research", json=note, headers=HEADERS)
    assert new.status_code == 400  # a new note names a run that exists
    again = client.delete(f"/api/runs/{original}", headers=HEADERS)
    assert again.status_code == 409 and "may have been deleted" in again.json()["error"]


def test_drafts_failed_and_cancelled_runs_are_deleted(
    client, ready, project, tiny_backend, spec_factory
):
    draft = ready.save_draft(_quick(spec_factory, "Draft"))
    failed = run_spec(
        _quick(spec_factory, "Failed", dataset={"path": "datasets/missing.jsonl"}),
        project,
        backend=tiny_backend,
    )
    assert failed.status == "failed"
    cancelled = _finished(project, tiny_backend, spec_factory, "Cancelled")
    manifest = project.run_dir(cancelled) / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(json.dumps({**data, "status": "cancelled"}), encoding="utf-8")
    listed = _listed(client)
    assert [listed[i]["status"] for i in (draft, failed.run_id, cancelled)] == [
        "draft",
        "failed",
        "cancelled",
    ]
    for run_id in (draft, failed.run_id, cancelled):
        assert client.delete(f"/api/runs/{run_id}", headers=HEADERS).status_code == 200
        assert not os.path.lexists(project.experiments_dir / run_id)
    assert _listed(client) == {}


def test_a_run_in_progress_or_in_the_queue_is_never_deleted(
    client, ready, project, tiny_backend, spec_factory
):
    draft = ready.save_draft(_quick(spec_factory, "Waiting"))
    assert _patch(client, draft, title="Next").status_code == 200
    ready.job = Job(id="busy", kind="run", title="Waiting", run_id=draft)
    r = client.delete(f"/api/runs/{draft}", headers=HEADERS)
    assert r.status_code == 409 and r.json()["error"].startswith("“Next” is running.")
    ready.job.update(status="finished")
    runqueue.add_to_queue(project, _quick(spec_factory), draft_id=draft)
    r = client.delete(f"/api/runs/{draft}", headers=HEADERS)
    assert r.status_code == 409 and "waiting in the queue" in r.json()["error"]
    runqueue.clear_queue(project)
    assert (project.run_dir(draft) / "spec.json").is_file()

    # A run another process is running (its manifest says so) is refused too.
    elsewhere = _finished(project, tiny_backend, spec_factory, "Elsewhere")
    manifest = project.run_dir(elsewhere) / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(json.dumps({**data, "status": "running"}), encoding="utf-8")
    r = client.delete(f"/api/runs/{elsewhere}", headers=HEADERS)
    assert r.status_code == 409 and "is running, so its folder is in use" in r.json()["error"]
    assert (project.run_dir(elsewhere) / "results.parquet").is_file()
    with pytest.raises(ProjectError, match="is running"):
        delete_run(project, elsewhere)
    # Once the server reopens the project, an orphaned running run is interrupted: it can go.
    reopened = client.post("/api/projects/open", json={"path": str(project.root)}, headers=HEADERS)
    assert reopened.status_code == 200
    assert _listed(client)[elsewhere]["status"] == "failed"
    assert client.delete(f"/api/runs/{elsewhere}", headers=HEADERS).status_code == 200
    assert client.delete(f"/api/runs/{draft}", headers=HEADERS).status_code == 200


def test_a_queued_draft_is_never_deleted_from_the_command_line_or_python(project, spec_factory):
    from logogram.runner import write_json

    run_id, folder = project.new_run_dir("Waiting")
    write_json(folder / "spec.json", _quick(spec_factory, "Waiting").model_dump(mode="json"))
    runqueue.add_to_queue(project, _quick(spec_factory, "Waiting"), draft_id=run_id)
    where = ["--project", str(project.root)]
    refused = CliRunner().invoke(cli, ["delete", run_id, *where], input="y\n")
    assert refused.exit_code == 1
    assert "“Waiting” is waiting in the queue to run. Take it out of the queue first" in (
        refused.output
    )
    assert "Delete it?" not in refused.output  # refused before asking
    with pytest.raises(LogogramError, match="waiting in the queue"):
        lg.delete_run(project, run_id)
    with pytest.raises(ProjectError, match="waiting in the queue"):
        delete_run(project, run_id)
    assert (folder / "spec.json").is_file()
    # Out of the queue, it can go.
    runqueue.clear_queue(project)
    deleted = CliRunner().invoke(cli, ["delete", run_id, "--yes", *where])
    assert deleted.exit_code == 0, deleted.output
    assert not folder.exists()


@posix_only
def test_deleting_never_follows_a_link_or_leaves_the_experiments_folder(
    client, ready, project, tiny_backend, spec_factory, tmp_path
):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep me", encoding="utf-8")
    (tmp_path / "keep-file.txt").write_text("keep me too", encoding="utf-8")

    # Links inside a run's folder are removed, never followed.
    run_id = _finished(project, tiny_backend, spec_factory, "With links")
    folder = project.run_dir(run_id)
    (folder / "dir-link").symlink_to(outside, target_is_directory=True)
    (folder / "file-link").symlink_to(tmp_path / "keep-file.txt")
    (folder / "datasets").symlink_to(project.datasets_dir, target_is_directory=True)
    assert client.delete(f"/api/runs/{run_id}", headers=HEADERS).status_code == 200
    assert not os.path.lexists(folder)
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep me"
    assert (tmp_path / "keep-file.txt").read_text(encoding="utf-8") == "keep me too"
    assert (project.datasets_dir / "ioi.jsonl").is_file()

    # A run folder that is a link to another run: the link goes, the run it points to stays.
    target = _finished(project, tiny_backend, spec_factory, "Target")
    (project.experiments_dir / "alias").symlink_to(
        project.run_dir(target), target_is_directory=True
    )
    assert {"alias", target} <= set(_listed(client))
    assert client.delete("/api/runs/alias", headers=HEADERS).status_code == 200
    assert not os.path.lexists(project.experiments_dir / "alias")
    assert (project.run_dir(target) / "results.parquet").is_file()
    assert target in _listed(client)

    # A run folder that links out of the project is refused, and left alone.
    (project.experiments_dir / "outward").symlink_to(outside, target_is_directory=True)
    r = client.delete("/api/runs/outward", headers=HEADERS)
    assert r.status_code == 400 and "outside the project" in r.json()["error"]
    assert (outside / "keep.txt").is_file() and os.path.islink(project.experiments_dir / "outward")

    # Names that aren't run ids never reach the file system.
    for bad in (".hidden", "..", "a/b", ""):
        with pytest.raises(ProjectError):
            delete_run(project, bad)
    assert client.delete("/api/runs/.hidden", headers=HEADERS).status_code == 400

    # No run is deleted through an experiments folder that is itself a link.
    real = project.root / "runs-elsewhere"
    project.experiments_dir.rename(real)
    project.experiments_dir.symlink_to(real, target_is_directory=True)
    assert target in _listed(client)
    r = client.delete(f"/api/runs/{target}", headers=HEADERS)
    assert r.status_code == 400 and "links to another folder" in r.json()["error"]
    assert (real / target / "results.parquet").is_file()


def test_a_folder_that_cant_be_fully_deleted_says_so(project, spec_factory, monkeypatch):
    draft_id, folder = project.new_run_dir("Stuck")
    (folder / "spec.json").write_text(_quick(spec_factory).model_dump_json(), encoding="utf-8")

    def refuse(path, *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(folder / "spec.json"))

    monkeypatch.setattr("logogram.history.shutil.rmtree", refuse)
    with pytest.raises(ProjectError, match=r"couldn't delete all of .*Permission denied \(spec"):
        delete_run(project, draft_id)
    # Nor can a project that can't be written to keep a title: it says so plainly.
    monkeypatch.setattr("logogram.history.write_text_atomic", refuse)
    with pytest.raises(ProjectError, match=r"can't write annotations\.json .*Permission denied"):
        annotate_run(project, draft_id, title="Stuck")


# -- runs whose spec can't be read -------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "why"),
    [
        (None, "Its spec.json is missing"),
        ("{", "Invalid JSON"),
        ('{"name": 3}', "can't be read"),
        ("", "Invalid JSON"),
        ("dir", "isn't a regular file"),
    ],
)
def test_a_run_whose_spec_cant_be_read_is_listed_and_can_be_deleted(
    client, ready, project, tiny_backend, spec_factory, spec, why
):
    healthy = _finished(project, tiny_backend, spec_factory, "Healthy")
    broken = _finished(
        project,
        tiny_backend,
        spec_factory,
        "Broken",
        derived_from={"run": healthy, "kind": "rerun"},
    )
    started = json.loads((project.run_dir(broken) / "manifest.json").read_text())["started_at"]
    path = project.run_dir(broken) / "spec.json"
    path.unlink()
    if spec == "dir":
        path.mkdir()
    elif spec is not None:
        path.write_text(spec, encoding="utf-8")

    listed = _listed(client)
    assert listed[healthy]["status"] == "finished"
    row = listed[broken]
    assert row["status"] == "failed" and row["damaged"] is True
    assert row["error"].startswith(why) or why in row["error"]
    assert "Restore" in row["error"] and "or delete the run" in row["error"]
    assert row["name"] == broken and row["created"] == started
    # The manifest still places it under the run it came from.
    assert row["derived_from"]["run"] == healthy
    # It can be renamed, and deleted.
    assert _patch(client, broken, title="Broken, keep for now").json()["title"] == (
        "Broken, keep for now"
    )
    assert _listed(client)[broken]["title"] == "Broken, keep for now"
    assert client.delete(f"/api/runs/{broken}", headers=HEADERS).status_code == 200
    assert broken not in _listed(client) and healthy in _listed(client)


def test_any_folder_in_experiments_is_listed_and_hidden_ones_are_not(
    client, ready, project, monkeypatch
):
    (project.experiments_dir / "empty-folder").mkdir()
    (project.experiments_dir / ".partly-deleted").mkdir()
    (project.experiments_dir / "a-file.txt").write_text("not a run", encoding="utf-8")
    listed = _listed(client)
    assert set(listed) == {"empty-folder"}
    assert listed["empty-folder"]["damaged"] is True
    assert listed["empty-folder"]["created"] is not None
    monkeypatch.setattr("logogram.spec.JSON_LIMIT", 8)
    folder = project.experiments_dir / "too-large"
    folder.mkdir()
    (folder / "spec.json").write_text("{" + " " * 64 + "}", encoding="utf-8")
    assert "larger than the 8 bytes" in _listed(client)["too-large"]["error"]
    assert client.delete("/api/runs/empty-folder", headers=HEADERS).status_code == 200
    assert client.delete("/api/runs/too-large", headers=HEADERS).status_code == 200
    assert _listed(client) == {}


@posix_only
def test_a_spec_that_links_outside_is_listed_without_reading_it(
    client, ready, project, spec_factory, tmp_path
):
    outside = tmp_path / "spec.json"
    spec = _quick(spec_factory, "external contents must stay unread")
    outside.write_text(spec.model_dump_json(), encoding="utf-8")
    folder = project.experiments_dir / "linked-spec"
    folder.mkdir()
    (folder / "spec.json").symlink_to(outside)
    response = client.get("/api/runs", headers=HEADERS)
    assert "external contents" not in response.text
    row = {r["id"]: r for r in response.json()}["linked-spec"]
    assert row["damaged"] and "outside the project" in row["error"]
    assert client.delete("/api/runs/linked-spec", headers=HEADERS).status_code == 200
    assert outside.read_text(encoding="utf-8") == spec.model_dump_json()


# -- the server's security ---------------------------------------------------------------------


def test_managing_runs_needs_the_token_its_origin_and_the_project(
    app, client, ready, project, spec_factory
):
    draft = ready.save_draft(_quick(spec_factory, "Draft"))
    routes = [
        ("PATCH", f"/api/runs/{draft}", {"starred": True}),
        ("DELETE", f"/api/runs/{draft}", None),
    ]
    for method, path, body in routes:
        for headers, status in (
            ({"origin": BASE}, 401),
            ({**HEADERS, "authorization": "Bearer wrong"}, 401),
            ({**HEADERS, "host": "evil.example"}, 421),
            ({**HEADERS, "origin": "http://evil.example"}, 403),
            ({**HEADERS, "origin": "http://127.0.0.1:9999"}, 403),
            ({**HEADERS, "sec-fetch-site": "cross-site"}, 403),
        ):
            r = client.request(method, path, json=body, headers=headers)
            assert r.status_code == status, (method, headers)
    assert _listed(client)[draft]["starred"] is False

    # A browser tab, signed in by its cookie, must say where a change comes from, and which
    # project it is for.
    tab = TestClient(app, base_url=BASE)
    tab.get(f"/?token={TOKEN}", follow_redirects=False)
    session = ready.project.session_id
    for method, path, body in routes:
        r = tab.request(method, path, json=body, headers={"origin": BASE})
        assert r.status_code == 409 and "which project" in r.json()["error"]
        stale = {"origin": BASE, "x-logogram-project": "an-older-project"}
        r = tab.request(method, path, json=body, headers=stale)
        assert r.status_code == 409 and "changed in another tab" in r.json()["error"]
        r = tab.request(method, path, json=body, headers={"x-logogram-project": session})
        assert r.status_code == 403
    same = {"origin": BASE, "x-logogram-project": session}
    assert tab.patch(f"/api/runs/{draft}", json={"starred": True}, headers=same).status_code == 200
    assert tab.delete(f"/api/runs/{draft}", headers=same).status_code == 200
    assert not os.path.lexists(project.experiments_dir / draft)


# -- Python and the command line ---------------------------------------------------------------


def test_python_renames_stars_archives_and_deletes(project, tiny_backend, spec_factory, tmp_path):
    run = lg.load_run(project, _finished(project, tiny_backend, spec_factory, "From Python"))
    assert (run.title, run.starred, run.archived) == ("From Python", False, False)
    assert run.rename("  Patched,\nfrom a notebook ").star().archive() is run
    assert (run.title, run.starred, run.archived) == ("Patched, from a notebook", True, True)
    assert "Patched, from a notebook" in run._repr_html_()
    listed = lg.list_runs(project)[0]
    assert (listed["title"], listed["starred"], listed["archived"]) == (
        "Patched, from a notebook",
        True,
        True,
    )
    run.archive(False).star(False).rename(None)
    assert (run.title, run.starred, run.archived) == ("From Python", False, False)
    with pytest.raises(LogogramError, match="at most 200"):
        run.rename("x" * 300)
    other = Project.create(tmp_path, "Other")
    with pytest.raises(LogogramError, match="another project"):
        lg.delete_run(other, run)
    lg.delete_run(project, run)
    assert not os.path.lexists(run.folder)
    with pytest.raises(LogogramError, match="no run"):
        lg.load_run(project, run.id)
    with pytest.raises(LogogramError, match="may have been deleted"):
        lg.delete_run(project, run.id)
    draft_id, folder = project.new_run_dir("A draft")
    (folder / "spec.json").write_text(_quick(spec_factory).model_dump_json(), encoding="utf-8")
    assert annotate_run(project, draft_id, starred=True).starred
    lg.delete_run(project, draft_id)
    assert lg.list_runs(project) == []


def test_the_command_line_manages_runs(project, tiny_backend, spec_factory, tmp_path):
    runner = CliRunner()
    where = ["--project", str(project.root)]
    first = _finished(project, tiny_backend, spec_factory, "First")
    second = _finished(
        project, tiny_backend, spec_factory, "Second", derived_from={"run": first, "kind": "rerun"}
    )

    result = runner.invoke(cli, ["rename", first, "The first one", *where])
    assert result.exit_code == 0, result.output
    assert f"{first} is now called “The first one”." in result.output
    assert runner.invoke(cli, ["star", first, *where]).output == "Starred “The first one”.\n"
    result = runner.invoke(cli, ["archive", second, *where])
    assert result.output == "Archived “Second”.\n"
    listing = runner.invoke(cli, ["list", *where]).output
    assert "★ The first one" in listing and "Second" not in listing
    assert "1 archived run not listed: list it with --archived." in listing
    listing = runner.invoke(cli, ["list", "--archived", *where]).output
    assert "Second (archived)" in listing
    assert runner.invoke(cli, ["archive", second, "--restore", *where]).output == (
        "“Second” is back from the archive.\n"
    )
    assert runner.invoke(cli, ["star", first, "--remove", *where]).output == (
        "Removed the star from “The first one”.\n"
    )
    result = runner.invoke(cli, ["rename", first, "", *where])
    assert result.output == f"{first} shows its spec's name again: “First”.\n"

    # Deleting asks first, saying what goes and what stays.
    result = runner.invoke(cli, ["delete", first, *where], input="n\n")
    assert result.exit_code == 1 and "Nothing was deleted." in result.output
    assert "It can't be undone." in result.output
    assert "1 run derived from it stays, but no longer links back to it." in result.output
    assert (project.run_dir(first) / "spec.json").is_file()
    result = runner.invoke(cli, ["delete", str(project.run_dir(first)), *where], input="y\n")
    assert result.exit_code == 0, result.output
    assert f"Deleted “First” (experiments/{first})." in result.output
    assert not os.path.lexists(project.run_dir(first))

    # A damaged run is listed with what is wrong, and can be deleted without asking.
    (project.run_dir(second) / "spec.json").write_text("{", encoding="utf-8")
    listing = runner.invoke(cli, ["list", *where]).output
    assert "Its spec.json can't be read" in listing
    result = runner.invoke(cli, ["delete", second, "--yes", *where])
    assert result.exit_code == 0 and not os.path.lexists(project.run_dir(second))

    # A folder elsewhere is never taken for a run of the same name, and runs in use are refused.
    elsewhere = tmp_path / "copies" / second
    elsewhere.mkdir(parents=True)
    result = runner.invoke(cli, ["delete", str(elsewhere), "--yes", *where])
    assert result.exit_code == 1 and "isn't a run folder" in result.output
    assert elsewhere.is_dir()
    assert runner.invoke(cli, ["delete", "no-such-run", "--yes", *where]).exit_code == 1
    third = _finished(project, tiny_backend, spec_factory, "Third")
    manifest = project.run_dir(third) / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(json.dumps({**data, "status": "running"}), encoding="utf-8")
    result = runner.invoke(cli, ["delete", third, "--yes", *where])
    assert result.exit_code == 1 and "is running" in result.output
    (project.run_dir(third) / ANNOTATIONS).write_text("{", encoding="utf-8")
    result = runner.invoke(cli, ["star", third, *where])
    assert result.exit_code == 1 and "won't replace it" in result.output


@posix_only
def test_the_command_line_deletes_a_linked_run_folder_as_a_link(
    project, tiny_backend, spec_factory
):
    target = _finished(project, tiny_backend, spec_factory, "Target")
    alias = project.experiments_dir / "alias"
    alias.symlink_to(project.run_dir(target), target_is_directory=True)
    where = ["--project", str(project.root)]
    result = CliRunner().invoke(cli, ["delete", str(alias), "--yes", *where])
    assert result.exit_code == 0, result.output
    assert not os.path.lexists(alias) and (project.run_dir(target) / "spec.json").is_file()
