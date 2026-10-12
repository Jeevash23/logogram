"""The run queue: runs line up behind the job in progress and start, in order, as each one ends."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from logogram.project import Project
from logogram.server.app import create_app
from logogram.server.security import SecurityConfig
from logogram.server.state import Job

TOKEN = "test-token"
BASE = "http://127.0.0.1:8765"
HEADERS = {"authorization": f"Bearer {TOKEN}", "origin": BASE}


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
def held(monkeypatch):
    """Every run waits before it starts until ``held.release()``, even when cancelled, so a test
    can queue behind it; ``held.started`` names the runs in the order they started."""
    from logogram import runner

    real = runner.run_spec
    release = threading.Event()
    started: list[str] = []

    def run_spec(spec: Any, project: Any, **kwargs: Any) -> Any:
        started.append(spec.name)
        release.wait(60)
        return real(spec, project, **kwargs)

    monkeypatch.setattr(runner, "run_spec", run_spec)
    yield SimpleNamespace(release=release.set, started=started)
    release.set()  # never leave a job waiting


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


def _spec(spec_factory, name: str, **overrides: Any) -> dict[str, Any]:
    """A quick run: attention and MLP of each layer, at the last position."""
    scope = {"kind": "layer_components", "position": {"kind": "last"}}
    return spec_factory(name=name, scope=scope, **overrides).model_dump(mode="json")


def _run(client, spec: dict[str, Any]) -> str:
    r = client.post("/api/runs", json={"spec": spec}, headers=HEADERS)
    assert r.status_code == 200, r.text
    return r.json()["run_id"]


def _queue(client, spec: dict[str, Any], **extra: Any) -> dict[str, Any]:
    r = client.post("/api/queue", json={"spec": spec, **extra}, headers=HEADERS)
    assert r.status_code == 200, r.text
    return r.json()


def _titles(queue: list[dict[str, Any]]) -> list[str]:
    return [q["title"] for q in queue]


def _settled(client, timeout: float = 120) -> dict[str, dict[str, Any]]:
    """The runs, by name, once nothing is running or waiting: the queue is empty and the last job
    has let go of the server and of the queue (a run is listed as ended a moment before that)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = client.get("/api/state", headers=HEADERS).json()
        job = state["job"]
        idle = state["queue_status"]["state"] in ("idle", "paused")
        if not state["queue"] and idle and (job is None or job["status"] != "running"):
            runs = client.get("/api/runs", headers=HEADERS).json()
            if all(r["status"] != "running" for r in runs):
                return {r["name"]: r for r in runs}
        time.sleep(0.05)
    raise AssertionError("the queue did not settle")


def _told(events: list[tuple[str, dict[str, Any]]], timeout: float = 30) -> list[dict[str, Any]]:
    """The queue's events, once the last of them says it is idle: the tabs hear it a moment after
    the server lets go of the queue."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        told = [data for kind, data in events if kind == "queue"]
        if told and told[-1]["state"] == "idle":
            return told
        time.sleep(0.02)
    raise AssertionError("the tabs were never told the queue is idle")


def test_queueing_with_nothing_running_starts_the_run(client, ready, events, spec_factory):
    out = _queue(client, _spec(spec_factory, "At once"))
    assert out["run_id"] and out["job"]["status"] == "running" and out["job"]["kind"] == "run"
    assert out["queued"] is None and out["queue"] == []
    runs = _settled(client)
    assert runs["At once"]["id"] == out["run_id"] and runs["At once"]["status"] == "finished"
    assert not [kind for kind, _ in events if kind.startswith("queue")]


def test_runs_queued_while_one_runs_start_in_order(client, ready, held, events, spec_factory):
    first = _run(client, _spec(spec_factory, "First"))
    # Starting another still says the server is busy; the queue takes it instead.
    busy = client.post("/api/runs", json={"spec": _spec(spec_factory, "Not now")}, headers=HEADERS)
    assert busy.status_code == 409 and "still running" in busy.json()["error"]
    second = _queue(client, _spec(spec_factory, "Second"))
    assert second["run_id"] is None and second["job"] is None
    assert second["queued"]["title"] == "Second" and second["queued"]["draft_id"] is None
    assert second["queued"]["project_session"] == ready.project.session_id
    assert second["queued"]["spec"]["name"] == "Second" and second["queued"]["queued_at"] > 0
    third = _queue(client, _spec(spec_factory, "Third"))
    assert _titles(third["queue"]) == ["Second", "Third"]
    assert _titles(client.get("/api/queue", headers=HEADERS).json()) == ["Second", "Third"]
    assert _titles(client.get("/api/state", headers=HEADERS).json()["queue"]) == ["Second", "Third"]

    held.release()
    runs = _settled(client)
    assert held.started == ["First", "Second", "Third"]
    assert runs["First"]["id"] == first and "Not now" not in runs
    assert [runs[name]["status"] for name in ("First", "Second", "Third")] == ["finished"] * 3
    # Each change was told with the whole queue and how it stood, for this project's tabs: its
    # runs waited for the first one, ran one at a time, and the queue went idle once empty.
    told = _told(events)
    expected = [
        (["Second"], "waiting"),
        (["Second", "Third"], "waiting"),
        (["Third"], "running"),
        ([], "running"),
        ([], "idle"),
    ]
    assert [(_titles(data["runs"]), data["state"]) for data in told] == expected
    assert {data["project_session"] for data in told} == {ready.project.session_id}
    assert all(data["runner"] is None and data["error"] is None for data in told)


def test_a_queued_run_starts_after_the_run_before_it_fails(
    client, ready, project, held, spec_factory
):
    (project.datasets_dir / "broken.jsonl").write_text("not a prompt\n", encoding="utf-8")
    _run(client, _spec(spec_factory, "Broken", dataset={"path": "datasets/broken.jsonl"}))
    _queue(client, _spec(spec_factory, "After the failure"))
    held.release()
    runs = _settled(client)
    assert runs["Broken"]["status"] == "failed" and runs["Broken"]["error"]
    assert runs["After the failure"]["status"] == "finished"
    assert held.started == ["Broken", "After the failure"]


def test_cancelling_the_running_job_leaves_the_queue(client, ready, held, spec_factory):
    _run(client, _spec(spec_factory, "Cancelled"))
    _queue(client, _spec(spec_factory, "Still queued"))
    cancel = client.post("/api/jobs/cancel", headers=HEADERS)
    assert cancel.status_code == 200 and cancel.json()["job"]["title"] == "Cancelled"
    assert _titles(client.get("/api/queue", headers=HEADERS).json()) == ["Still queued"]
    held.release()
    runs = _settled(client)
    assert runs["Cancelled"]["status"] == "cancelled"
    assert runs["Still queued"]["status"] == "finished"


def test_a_queued_run_can_be_removed_before_it_starts(client, ready, held, spec_factory):
    _run(client, _spec(spec_factory, "Running"))
    removed = _queue(client, _spec(spec_factory, "Removed"))["queued"]
    _queue(client, _spec(spec_factory, "Kept"))
    left = client.delete(f"/api/queue/{removed['id']}", headers=HEADERS)
    assert left.status_code == 200 and _titles(left.json()) == ["Kept"]
    again = client.delete(f"/api/queue/{removed['id']}", headers=HEADERS)
    assert again.status_code == 409 and "isn't queued any more" in again.json()["error"]
    held.release()
    runs = _settled(client)
    assert held.started == ["Running", "Kept"] and "Removed" not in runs


def test_a_queued_run_that_cant_start_is_skipped_with_the_reason(
    client, ready, project, held, events, spec_factory
):
    gone = project.datasets_dir / "gone.jsonl"
    gone.write_bytes((project.datasets_dir / "ioi.jsonl").read_bytes())
    _run(client, _spec(spec_factory, "Running"))
    lost = _spec(spec_factory, "Lost its prompts", dataset={"path": "datasets/gone.jsonl"})
    skipped = _queue(client, lost)["queued"]
    _queue(client, _spec(spec_factory, "Next"))
    gone.unlink()  # after it was queued: it can't start when its turn comes
    held.release()
    runs = _settled(client)
    assert held.started == ["Running", "Next"]
    assert "Lost its prompts" not in runs and runs["Next"]["status"] == "finished"
    assert [data for kind, data in events if kind == "queue.skipped"] == [
        {
            "project_session": ready.project.session_id,
            "id": skipped["id"],
            "title": "Lost its prompts",
            "error": "gone.jsonl isn't a regular file inside the project.",
        }
    ]
    told = [_titles(data["runs"]) for data in _told(events)]
    assert told[-3:] == [["Next"], [], []]  # Next leaves the queue as it starts, then it idles


def test_what_cant_start_isnt_queued(client, ready, project, held, spec_factory):
    running = _run(client, _spec(spec_factory, "Running"))
    nowhere = _spec(spec_factory, "Nowhere", dataset={"path": "datasets/nowhere.jsonl"})
    r = client.post("/api/queue", json={"spec": nowhere}, headers=HEADERS)
    assert r.status_code == 400 and "nowhere.jsonl" in r.json()["error"]
    again = {"spec": _spec(spec_factory, "Again"), "draft_id": running}
    r = client.post("/api/queue", json=again, headers=HEADERS)
    assert r.status_code == 409 and "running now" in r.json()["error"]
    assert client.get("/api/queue", headers=HEADERS).json() == []
    held.release()
    _settled(client)


def test_a_queued_draft_runs_in_its_folder_as_it_was_queued(
    client, ready, project, held, spec_factory
):
    spec = _spec(spec_factory, "Draft")
    draft = client.post("/api/drafts", json={"spec": spec}, headers=HEADERS).json()["run_id"]
    _run(client, _spec(spec_factory, "Running"))
    assert _queue(client, spec, draft_id=draft)["queued"]["draft_id"] == draft
    twice = client.post("/api/queue", json={"spec": spec, "draft_id": draft}, headers=HEADERS)
    assert twice.status_code == 409 and "already queued" in twice.json()["error"]
    # Its spec stays as it was queued.
    body = {"spec": _spec(spec_factory, "Edited"), "draft_id": draft}
    saved = client.post("/api/drafts", json=body, headers=HEADERS)
    assert saved.status_code == 409 and "queued" in saved.json()["error"]
    held.release()
    runs = _settled(client)
    assert runs["Draft"]["id"] == draft and runs["Draft"]["status"] == "finished"
    assert "Edited" not in runs


def test_a_queued_run_keeps_where_it_came_from(client, ready, held, spec_factory):
    from logogram.spec import Spec

    original = _run(client, _spec(spec_factory, "Original"))
    rerun = Spec.model_validate(_spec(spec_factory, "Rerun"))
    ready.queue_run(rerun, derived_from={"run": original, "kind": "rerun"})
    queued = client.get("/api/queue", headers=HEADERS).json()
    assert queued[0]["derived_from"] == {"run": original, "kind": "rerun", "change": None}
    held.release()
    runs = _settled(client)
    detail = client.get(f"/api/runs/{runs['Rerun']['id']}", headers=HEADERS).json()
    assert detail["manifest"]["derived_from"] == {"run": original, "kind": "rerun", "change": None}


def test_the_queue_belongs_to_its_project(client, ready, project, events, spec_factory, tmp_path):
    from logogram import runqueue

    # A job still running: runs queue behind it, and the project can't change.
    ready.job = Job(id="busy", kind="run", title="Another run")
    _queue(client, _spec(spec_factory, "Left behind"))
    other = Project.create(tmp_path / "elsewhere", "Other project")
    opened = client.post("/api/projects/open", json={"path": str(other.root)}, headers=HEADERS)
    assert opened.status_code == 409
    assert _titles(client.get("/api/queue", headers=HEADERS).json()) == ["Left behind"]

    # The moment after the job ends, before its thread starts the next run: a switch can win
    # that race. The queue stays with the project it belongs to, which this server lets go of.
    ready.job.update(status="finished")
    opened = client.post("/api/projects/open", json={"path": str(other.root)}, headers=HEADERS)
    assert opened.status_code == 200, opened.text
    assert client.get("/api/queue", headers=HEADERS).json() == []
    state = client.get("/api/state", headers=HEADERS).json()
    assert state["queue"] == [] and state["queue_status"]["state"] == "idle"
    assert [q.title for q in runqueue.read_queue(project)] == ["Left behind"]
    assert runqueue.queue_runner(project) is None
    ready._start_next()  # the job's thread, late: nothing is left to start
    assert ready.job.id == "busy" and project.list_runs() == [] and other.list_runs() == []

    # Closing a project keeps its queue too.
    ready.job = Job(id="busy again", kind="run", title="Another run")
    prompts = (project.datasets_dir / "ioi.jsonl").read_bytes()
    (other.datasets_dir / "ioi.jsonl").write_bytes(prompts)
    _queue(client, _spec(spec_factory, "Closed with it"))
    assert client.post("/api/projects/close", headers=HEADERS).status_code == 409
    ready.job.update(status="cancelled")
    assert client.post("/api/projects/close", headers=HEADERS).status_code == 200
    assert client.get("/api/state", headers=HEADERS).json()["queue"] == []
    ready._start_next()
    assert other.list_runs() == []
    assert [q.title for q in runqueue.read_queue(other)] == ["Closed with it"]

    # Opened again, its queue is back, paused: nothing starts on its own.
    reopened = client.post("/api/projects/open", json={"path": str(project.root)}, headers=HEADERS)
    assert reopened.status_code == 200, reopened.text
    status = client.get("/api/queue/state", headers=HEADERS).json()
    assert status["state"] == "paused" and _titles(status["runs"]) == ["Left behind"]
    assert status["project_session"] == reopened.json()["session_id"]
    assert ready.job.id == "busy again" and project.list_runs() == []


def test_the_queue_needs_the_token_the_origin_and_the_project(app, client, ready, spec_factory):
    ready.job = Job(id="busy", kind="run", title="Another run")  # so a queued run waits
    spec = _spec(spec_factory, "Checked")
    routes = [
        ("GET", "/api/queue", None),
        ("POST", "/api/queue", {"spec": spec}),
        ("DELETE", "/api/queue/abc123", None),
        ("GET", "/api/queue/state", None),
        ("POST", "/api/queue/start", None),
        ("POST", "/api/queue/pause", None),
        ("DELETE", "/api/queue", None),
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
            assert r.status_code == status, (method, path, headers)
    assert client.get("/api/queue", headers=HEADERS).json() == []

    # A browser tab, signed in by its cookie, must say where a change comes from, and which
    # project each request is for.
    tab = TestClient(app, base_url=BASE)
    tab.get(f"/?token={TOKEN}", follow_redirects=False)
    session = ready.project.session_id
    for method, path, body in routes:
        r = tab.request(method, path, json=body, headers={"origin": BASE})
        assert r.status_code == 409 and "which project" in r.json()["error"]
        stale = {"origin": BASE, "x-logogram-project": "an-older-project"}
        r = tab.request(method, path, json=body, headers=stale)
        assert r.status_code == 409 and "changed in another tab" in r.json()["error"]
    unsaid = tab.post("/api/queue", json={"spec": spec}, headers={"x-logogram-project": session})
    assert unsaid.status_code == 403
    same = {"origin": BASE, "x-logogram-project": session}
    queued = tab.post("/api/queue", json={"spec": spec}, headers=same)
    assert queued.status_code == 200 and queued.json()["queued"]["title"] == "Checked"
    assert _titles(tab.get("/api/queue", headers=same).json()) == ["Checked"]
    removed = tab.delete(f"/api/queue/{queued.json()['queued']['id']}", headers=same)
    assert removed.status_code == 200 and removed.json() == []
