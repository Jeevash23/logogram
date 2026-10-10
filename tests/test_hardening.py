"""Odd and hostile situations: shared projects with symlinks, unreadable files, a busy server,
browser tabs that fall behind, and how the session token is handled."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time

import pytest
import torch
from fastapi.testclient import TestClient

from logogram.backends.base import BackendError, Cancelled
from logogram.project import Project
from logogram.server.app import create_app
from logogram.server.security import SecurityConfig
from logogram.server.state import QUEUE_SIZE, EventHub, Job
from logogram.spec import ModelRef

TOKEN = "test-token"
BASE = "http://127.0.0.1:8765"
HEADERS = {"authorization": f"Bearer {TOKEN}", "origin": BASE}

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks and modes")


@pytest.fixture
def security():
    return SecurityConfig(token=TOKEN, port=8765)


@pytest.fixture
def app(tmp_path, monkeypatch, security):
    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    return create_app(security, serve_web=False)


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


def _wait(client, run_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        runs = client.get("/api/runs", headers=HEADERS).json()
        status = next((r["status"] for r in runs if r["id"] == run_id), None)
        if status not in ("running", "draft"):
            return status
        time.sleep(0.05)
    raise AssertionError("run did not finish")


# -- the session token -------------------------------------------------------------------------


def test_the_browser_launch_code_works_once(client, security):
    code = security.new_launch_code()
    assert client.get(f"/?token={code}", follow_redirects=False).status_code == 303
    assert client.get("/api/state").status_code == 200  # logged in by cookie
    client.cookies.clear()
    assert client.get(f"/?token={code}", follow_redirects=False).status_code == 401
    expired = security.new_launch_code(ttl=-1)
    assert client.get(f"/?token={expired}", follow_redirects=False).status_code == 401


def test_a_planted_cookie_doesnt_lock_you_out(client):
    # Cookies aren't separated by port: a page on another local port can set one with our name.
    both = f"logogram_8765=planted; logogram_8765={TOKEN}"
    assert client.get("/api/state", headers={"cookie": both}).status_code == 200
    assert client.get("/api/state", headers={"cookie": "logogram_8765=planted"}).status_code == 401


def test_cookie_writes_must_say_where_they_come_from(client):
    client.get(f"/?token={TOKEN}", follow_redirects=False)
    assert client.get("/api/state").status_code == 200
    assert client.post("/api/projects/close").status_code == 403
    same_origin = {"sec-fetch-site": "same-origin"}
    assert client.post("/api/projects/close", headers=same_origin).status_code == 200
    assert client.post("/api/projects/close", headers={"origin": BASE}).status_code == 200


# -- project folders ---------------------------------------------------------------------------


@posix_only
def test_runs_are_never_written_through_symlinks(client, ready, project, tmp_path, spec_factory):
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim"
    victim.write_text("keep me", encoding="utf-8")
    spec = spec_factory().model_dump(mode="json")

    # A draft carrying a link where a temporary file might be written.
    draft = project.experiments_dir / "draft-a"
    draft.mkdir()
    (draft / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (draft / ".spec.json.tmp").symlink_to(victim)
    r = client.post("/api/runs", json={"spec": spec, "draft_id": "draft-a"}, headers=HEADERS)
    assert r.status_code == 200, r.text
    assert _wait(client, "draft-a") == "finished"
    assert victim.read_text(encoding="utf-8") == "keep me"

    # A run folder that links out of the project is refused, and left alone.
    (outside / "spec.json").write_text("{}", encoding="utf-8")
    (project.experiments_dir / "draft-b").symlink_to(outside, target_is_directory=True)
    r = client.post("/api/runs", json={"spec": spec, "draft_id": "draft-b"}, headers=HEADERS)
    assert r.status_code == 400 and "outside the project" in r.json()["error"]
    assert (outside / "spec.json").read_text(encoding="utf-8") == "{}"
    assert sorted(p.name for p in outside.iterdir()) == ["spec.json", "victim"]


@posix_only
def test_datasets_are_never_read_or_written_through_symlinks(client, ready, project, tmp_path):
    outside = tmp_path / "outside.jsonl"
    outside.write_text((project.datasets_dir / "ioi.jsonl").read_text(encoding="utf-8"))
    original = outside.read_text(encoding="utf-8")
    (project.datasets_dir / "linked.jsonl").symlink_to(outside)
    (project.datasets_dir / "zero.jsonl").symlink_to("/dev/zero")

    state = client.get("/api/state", headers=HEADERS)  # returns: /dev/zero is never read
    assert state.status_code == 200
    datasets = {d["name"]: d for d in state.json()["project"]["datasets"]}
    assert "outside the project" in datasets["linked.jsonl"]["error"]
    assert "error" in datasets["zero.jsonl"]
    assert client.get("/api/datasets/linked.jsonl", headers=HEADERS).status_code == 400

    # Replacing a linked dataset replaces the link, not the file it points to.
    text = (project.datasets_dir / "ioi.jsonl").read_text(encoding="utf-8").splitlines()[0]
    body = {"name": "linked", "text": text, "overwrite": True}
    assert client.post("/api/datasets/import", json=body, headers=HEADERS).status_code == 200
    assert outside.read_text(encoding="utf-8") == original
    assert not (project.datasets_dir / "linked.jsonl").is_symlink()


@posix_only
def test_a_project_with_a_linked_datasets_folder_is_not_written(client, ready, project, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    for child in project.datasets_dir.iterdir():
        child.unlink()
    project.datasets_dir.rmdir()
    project.datasets_dir.symlink_to(elsewhere, target_is_directory=True)
    r = client.post("/api/datasets/ioi", json={"name": "ioi", "n": 4}, headers=HEADERS)
    assert r.status_code == 400 and "outside the project" in r.json()["error"]
    assert list(elsewhere.iterdir()) == []


@posix_only
def test_a_planted_gitignore_link_is_not_followed(project, tmp_path):
    (project.root / ".gitignore").unlink()
    (project.root / ".gitignore").symlink_to(tmp_path / "planted")
    Project.open(project.root)
    assert not (tmp_path / "planted").exists()


def test_odd_dataset_files_dont_break_a_project(client, ready, project):
    record = {"clean": "a b", "corrupt": "a c", "answer": " x", "distractor": " y"}
    (project.datasets_dir / "utf16.jsonl").write_text(json.dumps(record) + "\n", "utf-16")
    bad_span = json.dumps({**record, "positions": {"IO": 5}})
    (project.datasets_dir / "spans.jsonl").write_text(bad_span + "\n", encoding="utf-8")
    (project.datasets_dir / "folder.jsonl").mkdir()

    r = client.post("/api/projects/open", json={"path": str(project.root)}, headers=HEADERS)
    assert r.status_code == 200, r.text
    datasets = {d["name"]: d for d in r.json()["datasets"]}
    assert "UTF-8" in datasets["utf16.jsonl"]["error"]
    assert "line 1" in datasets["spans.jsonl"]["error"]
    assert "regular file" in datasets["folder.jsonl"]["error"]
    assert datasets["ioi.jsonl"]["n"] == 12
    assert client.get("/api/state", headers=HEADERS).status_code == 200
    body = {"name": "bad", "text": bad_span}
    assert client.post("/api/datasets/import", json=body, headers=HEADERS).status_code == 400


def test_files_too_large_or_too_deep_dont_break_a_project(
    client, ready, project, spec_factory, monkeypatch
):
    spec = spec_factory().model_dump(mode="json")
    runs = []
    for _ in range(2):
        run_id = client.post("/api/runs", json={"spec": spec}, headers=HEADERS).json()["run_id"]
        assert _wait(client, run_id) == "finished"
        runs.append(run_id)
    # A summary nested past Python's recursion limit, and one far larger than Logogram writes.
    (project.run_dir(runs[0]) / "summary.json").write_text("[" * 100_000, encoding="utf-8")
    listed = {r["id"]: r for r in client.get("/api/runs", headers=HEADERS).json()}
    assert listed[runs[0]]["status"] == "failed" and listed[runs[1]]["status"] == "finished"
    monkeypatch.setattr("logogram.project.JSON_LIMIT", 1024)
    listed = {r["id"]: r for r in client.get("/api/runs", headers=HEADERS).json()}
    assert listed[runs[1]]["status"] == "failed" and "unreadable" in listed[runs[1]]["error"]
    detail = client.get(f"/api/runs/{runs[1]}", headers=HEADERS).json()
    assert detail["summary"] is None and detail["listing"]["status"] == "failed"

    monkeypatch.undo()
    monkeypatch.setattr("logogram.datasets.DATASET_LIMIT", 64)
    datasets = client.get("/api/state", headers=HEADERS).json()["project"]["datasets"]
    error = next(d for d in datasets if d["name"] == "ioi.jsonl")["error"]
    assert "larger than" in error and "smaller files" in error
    run_id = client.post("/api/runs", json={"spec": spec}, headers=HEADERS).json()["run_id"]
    assert _wait(client, run_id) == "failed"
    listed = {r["id"]: r for r in client.get("/api/runs", headers=HEADERS).json()}
    assert "ioi.jsonl is larger than the 64 bytes Logogram reads" in listed[run_id]["error"]


@posix_only
def test_a_dangling_link_doesnt_break_recent_projects(client, ready, project):
    (project.experiments_dir / "gone").symlink_to(project.root / "nowhere")
    recent = client.get("/api/projects/recent", headers=HEADERS)
    assert recent.status_code == 200 and recent.json()[0]["name"] == "Test project"


@posix_only
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores modes")
def test_folder_listing_skips_what_it_cant_enter(client, tmp_path):
    locked = tmp_path / "locked"
    (locked / "inner").mkdir(parents=True)
    locked.chmod(0o600)  # names can be listed, but not looked at
    try:
        r = client.get("/api/fs", params={"path": str(locked)}, headers=HEADERS)
        assert r.status_code == 200 and r.json()["entries"] == []
    finally:
        locked.chmod(0o755)


def test_creating_a_project_where_you_cant_write_says_so(client, tmp_path):
    blocker = tmp_path / "a-file"
    blocker.write_text("", encoding="utf-8")
    r = client.post("/api/projects", json={"name": "x", "parent": str(blocker)}, headers=HEADERS)
    assert r.status_code == 400 and r.json()["error"]


# -- jobs --------------------------------------------------------------------------------------


def test_a_busy_server_writes_nothing(client, ready, project, spec_factory):
    spec = spec_factory().model_dump(mode="json")
    draft = client.post("/api/drafts", json={"spec": spec}, headers=HEADERS).json()["run_id"]
    saved = (project.experiments_dir / draft / "spec.json").read_text(encoding="utf-8")
    before = sorted(p.name for p in project.experiments_dir.iterdir())

    ready.job = Job(id="busy", kind="run", title="Another run")
    assert client.post("/api/runs", json={"spec": spec}, headers=HEADERS).status_code == 409
    other = spec_factory(name="changed").model_dump(mode="json")
    body = {"spec": other, "draft_id": draft}
    assert client.post("/api/runs", json=body, headers=HEADERS).status_code == 409
    assert (project.experiments_dir / draft / "spec.json").read_text(encoding="utf-8") == saved
    assert sorted(p.name for p in project.experiments_dir.iterdir()) == before


def test_job_snapshots_are_versioned(client, ready, spec_factory):
    spec = spec_factory().model_dump(mode="json")
    started = client.post("/api/runs", json={"spec": spec}, headers=HEADERS).json()
    assert _wait(client, started["run_id"]) == "finished"
    final = client.get("/api/state", headers=HEADERS).json()["job"]
    assert final["status"] == "finished" and final["version"] >= 2
    assert final["version"] >= started["job"]["version"]


def test_long_names_still_get_robustness_runs(client, ready, spec_factory):
    spec = spec_factory(name="n" * 195).model_dump(mode="json")
    run_id = client.post("/api/runs", json={"spec": spec}, headers=HEADERS).json()["run_id"]
    assert _wait(client, run_id) == "finished"
    experiment = {"kind": "ablation", "baseline": {"kind": "zero"}}
    r = client.post(
        f"/api/runs/{run_id}/robustness", json={"experiment": experiment}, headers=HEADERS
    )
    assert r.status_code == 200, r.text
    assert _wait(client, r.json()["run_id"]) == "finished"
    names = [x["name"] for x in client.get("/api/runs", headers=HEADERS).json()]
    assert any(n.endswith("· robustness") and len(n) <= 200 for n in names)


def test_cancelling_a_model_load_leaves_no_model(app, monkeypatch):
    state = app.state.logogram
    started = threading.Event()

    def slow_load(model_id, *, cancel=None, **kwargs):
        started.set()
        while not cancel.is_set():
            time.sleep(0.01)
        raise Cancelled()

    monkeypatch.setattr("logogram.backends.transformer_lens.load_model", slow_load)
    job = state.load_model_job(
        ModelRef(
            id="some/model", revision=None, dtype="float32", device="auto", process_weights=True
        )
    )
    assert started.wait(5)
    state.cancel_job()
    deadline = time.time() + 5
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.01)
    assert job.status == "cancelled"
    assert state.backend is None and state.model_status["state"] == "none"


def test_a_cancelled_load_stops_before_the_network():
    from logogram.backends.transformer_lens import load_model

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        load_model("openai-community/gpt2", device="cpu", cancel=cancel)


def test_closing_a_model_waits_for_work_in_progress(tiny_model_dir):
    from logogram.backends.transformer_lens import TransformerLensBackend, boot_local

    backend = TransformerLensBackend.from_bridge(
        boot_local(tiny_model_dir, device="cpu", dtype="float32"),
        model_id="tiny",
        revision="test",
        dtype="float32",
        process_weights=True,
    )
    tokens = torch.tensor([backend.tokenize("Hello world", True).ids])
    closed = threading.Event()
    with backend.lock:  # an analysis in progress
        closer = threading.Thread(target=lambda: (backend.close(), closed.set()))
        closer.start()
        time.sleep(0.2)
        assert not closed.is_set()
        backend.final_logits(tokens)  # still usable until the analysis lets go
    closer.join(5)
    assert closed.is_set()
    with pytest.raises(BackendError, match="unloaded"):
        backend.final_logits(tokens)


# -- the event stream --------------------------------------------------------------------------


def _drain(queue):
    items = []
    while not queue.empty():
        item = queue.get_nowait()
        items.append(None if item is None else json.loads(item)["type"])
    return items


def test_a_tab_that_connects_mid_run_gets_the_run_so_far():
    async def scenario() -> None:
        hub = EventHub()
        hub.bind(asyncio.get_running_loop())
        hub.publish("run.started", {"run_id": "r"})
        hub.publish("run.layer", {"run_id": "r", "layer": 0})
        for i in range(50):  # throttled: only the latest progress is kept
            hub.publish("run.progress", {"run_id": "r", "done": i, "total": 100})
        hub.publish("job", {"id": "j"})
        await asyncio.sleep(0.05)
        queue = hub.subscribe()
        assert _drain(queue) == ["run.started", "run.layer", "run.progress"]
        hub.publish("run.finished", {"run_id": "r"})
        await asyncio.sleep(0.05)
        assert _drain(queue) == ["run.finished"]
        late = hub.subscribe()  # a later tab learns how it ended, marked as a replay
        assert json.loads(late.get_nowait()) == {
            "type": "run.finished",
            "run_id": "r",
            "replay": True,
        }

    asyncio.run(scenario())


def test_a_tab_that_falls_behind_is_told_to_reconnect():
    async def scenario() -> None:
        hub = EventHub()
        hub.bind(asyncio.get_running_loop())
        queue = hub.subscribe()
        for i in range(QUEUE_SIZE + 5):
            hub.publish("job", {"i": i})
        await asyncio.sleep(0.1)
        assert _drain(queue)[0] is None

    asyncio.run(scenario())


def test_a_draft_can_be_saved_again_until_it_runs(client, ready, project, spec_factory):
    spec = spec_factory().model_dump(mode="json")
    draft = client.post("/api/drafts", json={"spec": spec}, headers=HEADERS).json()["run_id"]
    edited = spec_factory(name="edited").model_dump(mode="json")
    again = client.post("/api/drafts", json={"spec": edited, "draft_id": draft}, headers=HEADERS)
    assert again.json()["run_id"] == draft
    saved = json.loads((project.experiments_dir / draft / "spec.json").read_text(encoding="utf-8"))
    assert saved["name"] == "edited"
    body = {"spec": edited, "draft_id": draft}
    assert client.post("/api/runs", json=body, headers=HEADERS).status_code == 200
    assert _wait(client, draft) == "finished"
    late = client.post("/api/drafts", json={"spec": spec, "draft_id": draft}, headers=HEADERS)
    assert late.status_code == 409


def test_finished_runs_list_the_profile_that_writes_their_logogram(client, ready, spec_factory):
    spec = spec_factory().model_dump(mode="json")
    run_id = client.post("/api/runs", json={"spec": spec}, headers=HEADERS).json()["run_id"]
    assert _wait(client, run_id) == "finished"
    listing = next(r for r in client.get("/api/runs", headers=HEADERS).json() if r["id"] == run_id)
    summary = client.get(f"/api/runs/{run_id}", headers=HEADERS).json()["summary"]
    expected = []
    for layer in range(summary["model"]["n_layers"]):
        values = [s["effect"]["mean"] for s in summary["sites"] if s["layer"] == layer]
        expected.append(round(max(values, key=abs), 4))
    assert listing["profile"] == expected


def test_a_run_that_has_ended_is_never_listed_as_running(client, ready, project, spec_factory):
    spec = spec_factory().model_dump(mode="json")
    run_id = client.post("/api/runs", json={"spec": spec}, headers=HEADERS).json()["run_id"]
    assert _wait(client, run_id) == "finished"
    draft = client.post("/api/drafts", json={"spec": spec}, headers=HEADERS).json()["run_id"]
    # The moment between writing the manifest and the job reporting that it finished.
    ready.job = Job(id="late", kind="run", title="test", run_id=run_id)
    status = {r["id"]: r["status"] for r in client.get("/api/runs", headers=HEADERS).json()}
    assert status[run_id] == "finished"
    ready.job = Job(id="draft", kind="run", title="test", run_id=draft)
    status = {r["id"]: r["status"] for r in client.get("/api/runs", headers=HEADERS).json()}
    assert status[draft] == "running"


def test_a_run_can_be_checked_in_float32(
    app, client, ready, spec_factory, monkeypatch, tiny_model_dir
):
    from logogram.backends.transformer_lens import TransformerLensBackend, boot_local

    def tiny(dtype):
        return TransformerLensBackend.from_bridge(
            boot_local(tiny_model_dir, device="cpu", dtype=dtype),
            model_id="tiny-gpt2",
            revision="test",
            dtype=dtype,
            process_weights=True,
        )

    loaded = []

    def load(model_id, *, dtype="float32", **kwargs):
        loaded.append(dtype)
        return tiny(dtype)

    # Its own model, since switching precision closes the loaded one.
    ready.backend = tiny("bfloat16")
    monkeypatch.setattr("logogram.backends.transformer_lens.load_model", load)
    model = {"id": "tiny-gpt2", "revision": "test", "device": "cpu", "dtype": "bfloat16"}
    spec = spec_factory(model=model).model_dump(mode="json")
    run_id = client.post("/api/runs", json={"spec": spec}, headers=HEADERS).json()["run_id"]
    assert _wait(client, run_id) == "finished"
    url = f"/api/runs/{run_id}/robustness"
    assert client.post(url, json={"dtype": "bfloat16"}, headers=HEADERS).status_code == 400
    assert client.post(url, json={}, headers=HEADERS).status_code in (400, 422)
    r = client.post(url, json={"dtype": "float32"}, headers=HEADERS)
    assert r.status_code == 200, r.text
    assert _wait(client, r.json()["run_id"]) == "finished"
    assert loaded == ["float32"]
    run = client.get(f"/api/runs/{r.json()['run_id']}", headers=HEADERS).json()
    assert run["spec"]["model"]["dtype"] == "float32"
    assert run["spec"]["experiment"] == spec["experiment"]
