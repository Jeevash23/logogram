"""Projects: the recent list (which keeps a project whose folder is missing until it is removed
from it, and never writes into the projects it lists), and opening a project when the list can't
be read or written."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from logogram.project import Project, forget_recent, load_recent, remember_recent
from logogram.server.app import create_app
from logogram.server.security import SecurityConfig

TOKEN = "test-token"
BASE = "http://127.0.0.1:8765"
HEADERS = {"authorization": f"Bearer {TOKEN}", "origin": BASE}


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
