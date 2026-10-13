"""Projects, from the app, the command line and Python: the recent list (which keeps a project
whose folder is missing until it is removed from it, never writes into the projects it lists, and
never stops a project opening), new projects, renaming, and deleting a project's folder, which
needs its name typed and never happens to a folder holding what Logogram didn't make."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import logogram as lg
from logogram.api import LogogramError
from logogram.cli import app as cli
from logogram.project import Project, forget_recent, load_recent, remember_recent
from logogram.server.app import create_app
from logogram.server.security import SecurityConfig

TOKEN = "test-token"
BASE = "http://127.0.0.1:8765"
HEADERS = {"authorization": f"Bearer {TOKEN}", "origin": BASE}
runner = CliRunner()


@pytest.fixture
def config(tmp_path, monkeypatch):
    """Logogram's own folder (settings, the recent projects), in the test's folder."""
    folder = tmp_path / "config"
    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: folder)
    monkeypatch.setattr("logogram.project.config_dir", lambda: folder)
    return folder


@pytest.fixture
def app(config):
    return create_app(SecurityConfig(token=TOKEN, port=8765), serve_web=False)


@pytest.fixture
def client(app):
    return TestClient(app, base_url=BASE)


@pytest.fixture
def events(app, monkeypatch):
    """Every event the server publishes from here on, as (kind, data)."""
    seen: list[tuple[str, dict[str, Any]]] = []
    hub = app.state.logogram.hub
    publish = hub.publish

    def record(kind: str, data: dict[str, Any]) -> None:
        seen.append((kind, data))
        publish(kind, data)

    monkeypatch.setattr(hub, "publish", record)
    return seen


def _make(parent, name):
    return Project.create(parent, name)


# -- the recent list ---------------------------------------------------------------------------


def test_a_project_whose_folder_is_missing_stays_listed_until_it_is_removed(config, tmp_path):
    a, b = _make(tmp_path, "Alpha"), _make(tmp_path, "Beta")
    remember_recent(a)
    remember_recent(b)
    moved = tmp_path / "elsewhere"
    a.root.rename(moved)  # an unplugged drive, say
    listed = load_recent()
    assert [(p["name"], p["missing"]) for p in listed] == [("Beta", False), ("Alpha", True)]
    gone = listed[1]
    assert gone["path"] == str(a.root) and gone["problem"] == "Folder not found"
    assert gone["modified"] is None and gone["opened"]
    # Opening another project keeps it in the list, and it comes back with its folder.
    remember_recent(Project.open(moved))
    assert [p["name"] for p in load_recent()] == ["Alpha", "Beta", "Alpha"]
    moved.rename(a.root)
    assert [p["missing"] for p in load_recent()] == [True, False, False]
    # Only removing it takes it out of the list; its folder is left as it is.
    assert forget_recent(moved) is True
    assert forget_recent(moved) is False
    assert [p["path"] for p in load_recent()] == [str(b.root), str(a.root)]
    assert (a.root / "project.json").is_file()


def test_listing_recent_projects_writes_nothing_into_them(config, tmp_path):
    project = _make(tmp_path, "Read only")
    remember_recent(project)
    (project.root / ".gitignore").unlink()
    project.datasets_dir.rmdir()
    before = sorted(p.name for p in project.root.iterdir())
    assert load_recent()[0]["name"] == "Read only"
    assert sorted(p.name for p in project.root.iterdir()) == before  # no folder, no .gitignore


def test_a_recent_list_with_entries_that_arent_projects_lists_the_rest(config, tmp_path):
    project = _make(tmp_path, "Kept")
    config.mkdir(parents=True, exist_ok=True)
    entries = ["a", 3, None, {"path": 5}, {"opened": "x"}, {"path": str(project.root)}]
    (config / "recent.json").write_text(json.dumps({"projects": entries}), encoding="utf-8")
    assert [p["name"] for p in load_recent()] == ["Kept"]
    for content in ("null", "[1, 2]", '{"projects": 3}', "{not json"):
        (config / "recent.json").write_text(content, encoding="utf-8")
        assert load_recent() == []
    remember_recent(project)  # rewrites the file with what it can keep
    saved = json.loads((config / "recent.json").read_text(encoding="utf-8"))
    assert [p["path"] for p in saved["projects"]] == [str(project.root)]
    assert saved["projects"][0]["name"] == "Kept"


def test_opening_a_project_never_fails_on_the_recent_list(client, config, tmp_path, events):
    project = _make(tmp_path, "Opens")
    # Logogram's own folder can't be written (here a file stands where it would be).
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("", encoding="utf-8")
    r = client.post("/api/projects/open", json={"path": str(project.root)}, headers=HEADERS)
    assert r.status_code == 200, r.text
    assert client.get("/api/state", headers=HEADERS).json()["project"]["name"] == "Opens"
    assert ("project", {"session_id": r.json()["session_id"]}) in events
    assert client.get("/api/projects/recent", headers=HEADERS).json() == []


def test_opening_a_project_with_a_broken_recent_list_mends_it(client, config, tmp_path):
    project = _make(tmp_path, "Mends")
    config.mkdir(parents=True, exist_ok=True)
    (config / "recent.json").write_text('{"projects": ["not", "objects"]}', encoding="utf-8")
    r = client.post("/api/projects/open", json={"path": str(project.root)}, headers=HEADERS)
    assert r.status_code == 200, r.text
    listed = client.get("/api/projects/recent", headers=HEADERS).json()
    assert [(p["name"], p["missing"]) for p in listed] == [("Mends", False)]


def test_opening_a_project_whose_folder_is_gone_says_so(client, config, tmp_path):
    project = _make(tmp_path, "Gone")
    remember_recent(project)
    project.root.rename(tmp_path / "moved")
    r = client.post("/api/projects/open", json={"path": str(project.root)}, headers=HEADERS)
    assert r.status_code == 400
    assert r.json()["error"].startswith(f"The folder {project.root} isn't there any more")
    listed = client.get("/api/projects/recent", headers=HEADERS).json()
    assert listed[0]["missing"] is True and listed[0]["problem"] == "Folder not found"
    # Removed from the list, it is gone from it; nothing else changes.
    left = client.delete(
        "/api/projects/recent", params={"path": str(project.root)}, headers=HEADERS
    )
    assert left.status_code == 200 and left.json() == []
    assert (tmp_path / "moved" / "project.json").is_file()


# -- renaming ----------------------------------------------------------------------------------


def test_a_project_is_renamed_in_its_project_json_from_every_front_end(
    client, config, tmp_path, events
):
    project = _make(tmp_path, "First name")
    meta = project.root / "project.json"
    data = json.loads(meta.read_text(encoding="utf-8"))
    meta.write_text(json.dumps({**data, "note": "kept"}), encoding="utf-8")  # what else it says
    r = client.post("/api/projects/open", json={"path": str(project.root)}, headers=HEADERS)
    session = r.json()["session_id"]
    tab = {**HEADERS, "x-logogram-project": session}
    renamed = client.patch("/api/project", json={"name": "  Name  movers\n"}, headers=tab)
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["name"] == "Name movers"
    saved = json.loads(meta.read_text(encoding="utf-8"))
    assert saved["name"] == "Name movers" and saved["note"] == "kept"
    assert project.root.name == "first-name"  # the folder keeps its name
    assert ("project.changed", {"project_session": session}) in events
    assert client.get("/api/projects/recent", headers=HEADERS).json()[0]["name"] == "Name movers"
    for bad, message in (("   ", "Give the project a name."), ("x" * 201, "at most 200")):
        r = client.patch("/api/project", json={"name": bad}, headers=tab)
        assert r.status_code == 400 and message in r.json()["error"]
    # A tab showing another project can't rename this one.
    stale = {**HEADERS, "x-logogram-project": "an-earlier-project"}
    assert client.patch("/api/project", json={"name": "No"}, headers=stale).status_code == 409

    # The command line, and Python.
    where = ["--project", str(project.root)]
    out = runner.invoke(cli, ["project", "rename", "From the terminal", *where])
    assert out.exit_code == 0, out.output
    assert "“Name movers” is now called “From the terminal”" in out.output
    assert Project.read(project.root).meta.name == "From the terminal"
    assert Project.open(project.root).rename("From Python").meta.name == "From Python"
    assert json.loads(meta.read_text(encoding="utf-8"))["note"] == "kept"


# -- the recent list from the command line, and new projects -----------------------------------


def test_the_command_line_lists_forgets_and_makes_projects(config, tmp_path):
    made = runner.invoke(cli, ["new", "Name movers", "--parent", str(tmp_path)])
    assert made.exit_code == 0, made.output
    root = (tmp_path / "name-movers").resolve()
    assert f"Created the project “Name movers” in\n\n  {root}\n" in made.output
    assert Project.read(root).meta.name == "Name movers"
    again = runner.invoke(cli, ["new", "Name movers", "--parent", str(tmp_path)])
    assert again.exit_code == 1 and "already exists" in again.output
    other = _make(tmp_path, "Other")
    remember_recent(other)
    other.root.rename(tmp_path / "unplugged")
    listed = runner.invoke(cli, ["projects"])
    assert listed.exit_code == 0, listed.output
    assert listed.output == (f"Other  (Folder not found)\n  {other.root}\nName movers\n  {root}\n")
    forgot = runner.invoke(cli, ["projects", "forget", str(other.root)])
    assert forgot.exit_code == 0 and "Its folder is left as it is." in forgot.output
    assert [p["name"] for p in load_recent()] == ["Name movers"]
    missing = runner.invoke(cli, ["projects", "forget", str(other.root)])
    assert missing.exit_code == 1 and "isn't in the recent list" in missing.output


# -- deleting ----------------------------------------------------------------------------------


def _fill(project, spec_factory) -> None:
    """A run, a draft and a queued run, as a project gathers them."""
    from logogram import runqueue
    from logogram.runner import write_json

    _, folder = project.new_run_dir("Finished")
    write_json(folder / "spec.json", spec_factory(name="Finished").model_dump(mode="json"))
    write_json(folder / "manifest.json", {"status": "failed", "error": "stopped"})
    _, folder = project.new_run_dir("Draft")
    write_json(folder / "spec.json", spec_factory(name="Draft").model_dump(mode="json"))
    runqueue.add_to_queue(project, spec_factory(name="Queued"))
    (project.root / "research.json").write_text("{}", encoding="utf-8")


def test_deleting_a_project_says_what_goes_and_needs_its_name(
    client, config, project, spec_factory, events
):
    _fill(project, spec_factory)
    root = project.root
    r = client.post("/api/projects/open", json={"path": str(root)}, headers=HEADERS)
    session = r.json()["session_id"]
    tab = {**HEADERS, "x-logogram-project": session}
    plan = client.get("/api/projects/deletion", params={"path": str(root)}, headers=HEADERS).json()
    assert (plan["runs"], plan["drafts"], plan["datasets"], plan["queued"]) == (1, 1, 2, 1)
    assert plan["others"] == [] and plan["problem"] is None
    assert plan["text"] == (
        f"This deletes the folder {root} and everything in it, for good: 1 run, 1 draft, 2 "
        "datasets and 1 run waiting in its queue. It can't be undone. Nothing outside the folder "
        "is touched: links in it are removed, not followed."
    )
    wrong = client.delete("/api/project", params={"name": "test project"}, headers=tab)
    assert wrong.status_code == 400
    assert (
        wrong.json()["error"] == "Type the project's name, “Test project”, exactly, to delete it."
    )
    assert root.is_dir()
    deleted = client.delete("/api/project", params={"name": "Test project"}, headers=tab)
    assert deleted.status_code == 200, deleted.text
    assert not root.exists()
    # The project was closed first, and every tab told; it left the recent list.
    assert client.get("/api/state", headers=HEADERS).json()["project"] is None
    assert ("project", {"session_id": None}) in events
    assert client.get("/api/projects/recent", headers=HEADERS).json() == []


def test_a_folder_holding_what_logogram_didnt_make_is_never_deleted(
    client, config, project, tmp_path
):
    (project.root / ".git").mkdir()
    (project.root / "notes.md").write_text("mine", encoding="utf-8")
    (project.root / ".DS_Store").write_text("", encoding="utf-8")  # a file browser's, not yours
    where = {"path": str(project.root)}
    plan = client.get("/api/projects/deletion", params=where, headers=HEADERS).json()
    assert plan["others"] == [".git/", "notes.md"]
    assert plan["problem"] == (
        f"The folder {project.root} also holds what Logogram didn't make (.git/, notes.md), so "
        "Logogram won't delete it. Move those out of the folder, or delete the folder yourself."
    )
    r = client.delete("/api/projects", params={**where, "name": "Test project"}, headers=HEADERS)
    assert r.status_code == 400 and r.json()["error"] == plan["problem"]
    out = runner.invoke(cli, ["project", "delete", "--yes", "--project", str(project.root)])
    assert out.exit_code == 1 and "Logogram won't delete it" in " ".join(out.output.split())
    with pytest.raises(LogogramError, match="Logogram won't delete it"):
        lg.delete_project(project)
    assert (project.root / "notes.md").read_text(encoding="utf-8") == "mine"
    # Without them, it goes.
    (project.root / ".git").rmdir()
    (project.root / "notes.md").unlink()
    lg.delete_project(project.root)
    assert not project.root.exists() and tmp_path.is_dir()


def test_a_project_with_a_run_in_progress_or_its_queue_running_is_not_deleted(
    app, client, config, project, spec_factory
):
    from logogram import runqueue
    from logogram.server.state import Job

    state = app.state.logogram
    r = client.post("/api/projects/open", json={"path": str(project.root)}, headers=HEADERS)
    tab = {**HEADERS, "x-logogram-project": r.json()["session_id"]}
    state.job = Job(id="busy", kind="run", title="Heads", run_id="x")
    refused = client.delete("/api/project", params={"name": "Test project"}, headers=tab)
    assert refused.status_code == 409
    assert refused.json()["error"].startswith("“Heads” is running. Wait for it to end")
    state.job.update(status="finished")
    # Run by another process (`logogram queue run`): refused, saying who.
    hold = runqueue.take_queue(project, "cli")
    plan = client.get(
        "/api/projects/deletion", params={"path": str(project.root)}, headers=HEADERS
    ).json()
    assert plan["busy"].startswith("The project's queue is being run by")
    refused = client.delete("/api/project", params={"name": "Test project"}, headers=tab)
    assert refused.status_code == 409 and refused.json()["error"] == plan["busy"]
    out = runner.invoke(cli, ["project", "delete", "--yes", "--project", str(project.root)])
    assert out.exit_code == 1 and "queue is being run by" in out.output
    runqueue.release_queue(project, hold)
    assert project.root.is_dir()


def test_deleting_a_project_from_the_command_line_asks_for_its_name(config, project):
    where = ["--project", str(project.root)]
    kept = runner.invoke(cli, ["project", "delete", *where], input="test\n")
    assert kept.exit_code == 1 and "That isn't its name. Nothing was deleted." in kept.output
    assert f"This deletes the folder {project.root}" in " ".join(kept.output.split())
    assert project.root.is_dir()
    gone = runner.invoke(cli, ["project", "delete", *where], input="Test project\n")
    assert gone.exit_code == 0, gone.output
    assert f"Deleted “Test project” ({project.root})." in gone.output
    assert not project.root.exists()


def test_deleting_a_toy_project_deletes_its_model_and_forgets_it(config, tmp_path):
    from logogram import local_models, toys

    made = toys.write_toy("induction", tmp_path / "toy")  # its model is added from model/
    assert (made.project.root / "model").is_dir()
    assert local_models.find_model(sha256=toys.get_toy("induction").sha256) is not None
    lg.delete_project(made.project)
    assert not made.project.root.exists()
    assert local_models.find_model(sha256=toys.get_toy("induction").sha256) is None
