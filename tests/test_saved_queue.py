"""The run queue is kept in the project: it survives a restart of the server and closing the
project, comes back paused, runs from the command line, and only one process runs it at a time."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from typing import Any

import psutil
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from logogram import runqueue
from logogram.cli import app as cli
from logogram.project import Project, ProjectError
from logogram.schema import QueueFile, QueueRunner
from logogram.server.app import create_app
from logogram.server.security import SecurityConfig
from logogram.server.state import Job

TOKEN = "test-token"
BASE = "http://127.0.0.1:8765"
HEADERS = {"authorization": f"Bearer {TOKEN}", "origin": BASE}
QUICK = {"kind": "layer_components", "position": {"kind": "last"}}
runner = CliRunner()


def _spec(spec_factory: Any, name: str, **overrides: Any) -> Any:
    return spec_factory(name=name, scope=QUICK, **overrides)


def _titles(items: list[Any]) -> list[str]:
    return [q.title if hasattr(q, "title") else q["title"] for q in items]


@pytest.fixture
def server(tmp_path, monkeypatch, tiny_backend):
    """A server with the tiny model loaded; ``server()`` makes another, as a restart would."""
    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")

    def make() -> tuple[Any, TestClient]:
        app = create_app(SecurityConfig(token=TOKEN, port=8765), serve_web=False)
        state = app.state.logogram
        state.backend = tiny_backend
        state.model_status = {"state": "ready"}
        return state, TestClient(app, base_url=BASE)

    return make


def _settled(client: TestClient, timeout: float = 120) -> dict[str, dict[str, Any]]:
    """The runs, by name, once nothing is running or waiting here."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = client.get("/api/state", headers=HEADERS).json()
        job = state["job"]
        idle = state["queue_status"]["state"] in ("idle", "paused")
        if idle and (job is None or job["status"] != "running"):
            runs = client.get("/api/runs", headers=HEADERS).json()
            if all(r["status"] != "running" for r in runs):
                return {r["name"]: r for r in runs}
        time.sleep(0.05)
    raise AssertionError("the queue did not settle")


# -- the queue file ------------------------------------------------------------------------------


def test_the_queue_file_round_trips(project, spec_factory):
    first = _spec(spec_factory, "First")
    second = _spec(spec_factory, "Second", notes="A note")
    assert runqueue.read_queue(project) == []
    a = runqueue.add_to_queue(project, first)
    b = runqueue.add_to_queue(
        project,
        second,
        draft_id="20260101-000000-second",
        derived_from={"run": "20250101-000000-x", "kind": "rerun"},
    )
    read = runqueue.read_queue(project)
    assert [q.id for q in read] == [a.id, b.id]
    assert read[0].spec == first and read[1].spec == second
    assert read[0].queued_at == a.queued_at and a.queued_at <= time.time()
    assert read[1].draft_id == "20260101-000000-second"
    assert read[1].derived_from == {"run": "20250101-000000-x", "kind": "rerun", "change": None}

    # A small JSON file in the project: the specs as queued and when, nothing about the machine.
    path = project.root / ".logogram" / "queue.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert QueueFile.model_validate(data).runs[0].id == a.id
    assert data["logogram_queue"] == 1 and data["runs"][0]["spec"]["name"] == "First"
    assert data["runs"][0]["queued_at"].endswith("Z")
    text = path.read_text(encoding="utf-8")
    assert str(project.root) not in text and os.sep + "home" + os.sep not in text
    assert ".logogram/" in (project.root / ".gitignore").read_text(encoding="utf-8")

    assert runqueue.remove_from_queue(project, a.id) == a
    assert runqueue.remove_from_queue(project, a.id) is None
    assert _titles(runqueue.read_queue(project)) == ["Second"]
    assert runqueue.clear_queue(project) == 1
    assert runqueue.read_queue(project) == [] and not path.exists()


def test_a_queue_file_that_cant_be_read_is_kept_until_cleared(project, spec_factory):
    folder = project.root / ".logogram"
    folder.mkdir()
    (folder / "queue.json").write_text('{"logogram_queue": 7, "runs": []}', encoding="utf-8")
    with pytest.raises(runqueue.QueueError, match=r"can't be read.*Clear the queue"):
        runqueue.read_queue(project)
    # Never overwritten by an addition: it may hold someone's runs.
    with pytest.raises(runqueue.QueueError, match="can't be read"):
        runqueue.add_to_queue(project, _spec(spec_factory, "New"))
    assert "7" in (folder / "queue.json").read_text(encoding="utf-8")
    assert runqueue.clear_queue(project) == 0
    assert runqueue.read_queue(project) == []
    runqueue.add_to_queue(project, _spec(spec_factory, "New"))
    assert _titles(runqueue.read_queue(project)) == ["New"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_the_queue_is_never_kept_outside_the_project(project, spec_factory, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (project.root / ".logogram").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ProjectError, match="outside the project"):
        runqueue.add_to_queue(project, _spec(spec_factory, "Elsewhere"))
    assert list(outside.iterdir()) == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_a_linked_queue_file_is_never_read(project, tmp_path):
    folder = project.root / ".logogram"
    folder.mkdir()
    target = tmp_path / "queue.json"
    target.write_text('{"logogram_queue": 1, "runs": []}', encoding="utf-8")
    (folder / "queue.json").symlink_to(target)
    with pytest.raises(runqueue.QueueError, match="can't be read"):
        runqueue.read_queue(project)
    (folder / "queue.lock").symlink_to(tmp_path / "lock")
    with pytest.raises(runqueue.QueueError, match="is a link"):
        runqueue.clear_queue(project)
    assert not (tmp_path / "lock").exists()


def test_changes_from_several_threads_are_all_kept(project, spec_factory):
    spec = _spec(spec_factory, "Many")
    holding = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with runqueue.locked(project):
            holding.set()
            release.wait(10)

    holder = threading.Thread(target=hold)
    holder.start()
    holding.wait(10)
    added: list[Any] = []
    adders = [
        threading.Thread(target=lambda: added.append(runqueue.add_to_queue(project, spec)))
        for _ in range(8)
    ]
    for t in adders:
        t.start()
    time.sleep(0.2)
    assert added == []  # they wait for the lock
    release.set()
    holder.join()
    for t in adders:
        t.join()
    assert sorted(q.id for q in runqueue.read_queue(project)) == sorted(q.id for q in added)
    assert len(added) == 8


# -- one runner at a time ----------------------------------------------------------------------


@pytest.fixture
def other_process():
    """Another process, standing in for one that runs the queue."""
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    yield process
    process.kill()
    process.wait()


def _runner_record(project: Project, process: Any, program: str) -> None:
    record = QueueRunner(
        program=program,  # type: ignore[arg-type]
        pid=process.pid,
        started=psutil.Process(process.pid).create_time(),
        token="1" * 32,
    )
    folder = project.root / ".logogram"
    folder.mkdir(exist_ok=True)
    (folder / "queue-runner.json").write_text(record.model_dump_json(), encoding="utf-8")


def test_two_runners_cant_take_the_same_queue(project, other_process):
    mine = runqueue.take_queue(project, "cli")
    assert runqueue.queue_runner(project) == mine
    with pytest.raises(
        runqueue.QueueBusy, match=r"already being run by `logogram queue run` \(process"
    ):
        runqueue.take_queue(project, "app")
    # Only its own record is removed.
    runqueue.release_queue(project, mine.model_copy(update={"token": "2" * 32}))
    assert runqueue.queue_runner(project) == mine
    runqueue.release_queue(project, mine)
    assert runqueue.queue_runner(project) is None

    # Another process runs it: refused, saying who; once that process has ended, it is free.
    _runner_record(project, other_process, "app")
    held = runqueue.queue_runner(project)
    assert held is not None and runqueue.runner_text(held) == (
        f"the Logogram app (process {other_process.pid})"
    )
    with pytest.raises(runqueue.QueueBusy, match="Pause it in the app"):
        runqueue.take_queue(project, "cli")
    other_process.kill()
    other_process.wait()
    assert runqueue.queue_runner(project) is None
    taken = runqueue.take_queue(project, "cli")
    assert runqueue.holds_queue(project, taken)


def test_the_app_wont_run_a_queue_another_process_runs(
    server, project, spec_factory, other_process
):
    state, client = server()
    state.open_project(project)
    runqueue.add_to_queue(project, _spec(spec_factory, "Theirs"))
    _runner_record(project, other_process, "cli")
    status = client.get("/api/queue/state", headers=HEADERS).json()
    assert status["state"] == "running"
    assert status["runner"] == f"`logogram queue run` (process {other_process.pid})"
    started = client.post("/api/queue/start", headers=HEADERS)
    assert started.status_code == 409
    assert "already being run by `logogram queue run`" in started.json()["error"]
    paused = client.post("/api/queue/pause", headers=HEADERS)
    assert paused.status_code == 409 and "Stop it there" in paused.json()["error"]
    # A run started here while it runs elsewhere joins its end, rather than running beside it.
    queued = client.post(
        "/api/queue",
        json={"spec": _spec(spec_factory, "Mine").model_dump(mode="json")},
        headers=HEADERS,
    ).json()
    assert queued["job"] is None and _titles(queued["queue"]) == ["Theirs", "Mine"]
    assert project.list_runs() == []


# -- a restart -----------------------------------------------------------------------------------


def test_a_queue_comes_back_paused_after_a_restart(server, project, spec_factory):
    first_state, first = server()
    first_state.open_project(project)
    first_state.job = Job(id="busy", kind="run", title="A run in progress")
    for name in ("One", "Two"):
        r = first.post(
            "/api/queue",
            json={"spec": _spec(spec_factory, name).model_dump(mode="json")},
            headers=HEADERS,
        )
        assert r.status_code == 200 and r.json()["queued"]["title"] == name
    status = first.get("/api/queue/state", headers=HEADERS).json()
    assert status["state"] == "waiting" and _titles(status["runs"]) == ["One", "Two"]
    first_state.stop()  # the server stops (its job never ends)

    state, client = server()
    state.open_project(project)
    status = client.get("/api/state", headers=HEADERS).json()["queue_status"]
    assert status["state"] == "paused" and _titles(status["runs"]) == ["One", "Two"]
    # Paused: Run starts a run at once, and the queue stays as it was.
    now = client.post(
        "/api/queue",
        json={"spec": _spec(spec_factory, "Now").model_dump(mode="json")},
        headers=HEADERS,
    ).json()
    assert now["job"] is not None and now["queued"] is None
    runs = _settled(client)
    assert set(runs) == {"Now"}
    assert _titles(runqueue.read_queue(project)) == ["One", "Two"]

    # Started, it runs to the end, in order, and lets go of the queue.
    started = client.post("/api/queue/start", headers=HEADERS)
    assert started.status_code == 200, started.text
    runs = _settled(client)
    assert [runs[name]["status"] for name in ("One", "Two")] == ["finished", "finished"]
    assert runs["One"]["created"] <= runs["Two"]["created"]
    status = client.get("/api/queue/state", headers=HEADERS).json()
    assert status == {**status, "state": "idle", "runs": [], "runner": None}
    assert runqueue.queue_runner(project) is None
    empty = client.post("/api/queue/start", headers=HEADERS)
    assert empty.status_code == 409 and "nothing to start" in empty.json()["error"]


def test_pausing_and_clearing_the_queue(server, project, spec_factory):
    state, client = server()
    state.open_project(project)
    state.job = Job(id="busy", kind="run", title="A run in progress")
    for name in ("One", "Two"):
        client.post(
            "/api/queue",
            json={"spec": _spec(spec_factory, name).model_dump(mode="json")},
            headers=HEADERS,
        )
    assert client.get("/api/queue/state", headers=HEADERS).json()["state"] == "waiting"
    paused = client.post("/api/queue/pause", headers=HEADERS).json()
    assert paused["state"] == "paused" and _titles(paused["runs"]) == ["One", "Two"]
    state.job.update(status="finished")
    state._start_next()  # the job's thread: the queue is paused, so nothing starts
    assert project.list_runs() == []
    cleared = client.delete("/api/queue", headers=HEADERS)
    assert cleared.status_code == 200 and cleared.json()["runs"] == []
    assert cleared.json()["state"] == "idle"
    assert runqueue.read_queue(project) == []


def test_a_saved_queue_that_cant_be_read_is_reported(server, project, spec_factory):
    state, client = server()
    state.open_project(project)
    folder = project.root / ".logogram"
    folder.mkdir()
    (folder / "queue.json").write_text("not json", encoding="utf-8")
    status = client.get("/api/queue/state", headers=HEADERS).json()
    assert status["runs"] == [] and "can't be read" in status["error"]
    # Nothing runs from it, but Run still runs.
    started = client.post("/api/queue/start", headers=HEADERS)
    assert started.status_code == 400 and "Clear the queue" in started.json()["error"]
    now = client.post(
        "/api/queue",
        json={"spec": _spec(spec_factory, "Now").model_dump(mode="json")},
        headers=HEADERS,
    ).json()
    assert now["job"] is not None
    _settled(client)
    assert client.delete("/api/queue", headers=HEADERS).json()["error"] is None


# -- the command line ------------------------------------------------------------------------------


def _spec_file(project: Project, spec_factory: Any, name: str, **overrides: Any) -> str:
    path = project.root / f"{name.lower().replace(' ', '-')}.json"
    path.write_text(
        json.dumps(_spec(spec_factory, name, **overrides).model_dump(mode="json")), encoding="utf-8"
    )
    return str(path)


@pytest.fixture
def loads_tiny(tiny_backend, monkeypatch):
    """Runs from the command line load the tiny model instead of downloading one."""
    monkeypatch.setattr(
        "logogram.runner.default_provider", lambda on_event=None: lambda ref: tiny_backend
    )


def test_the_command_line_adds_lists_and_clears_the_queue(project, spec_factory):
    where = ["--project", str(project.root)]
    empty = runner.invoke(cli, ["queue", "list", *where])
    assert empty.exit_code == 0 and "The queue is empty." in empty.output
    added = runner.invoke(cli, ["queue", "add", _spec_file(project, spec_factory, "Heads first")])
    assert added.exit_code == 0, added.output
    assert "Queued “Heads first”. 1 run waits in the queue." in added.output
    assert "It is paused: `logogram queue run` runs it to the end" in added.output
    runner.invoke(cli, ["queue", "add", _spec_file(project, spec_factory, "Then layers"), *where])
    listed = runner.invoke(cli, ["queue", "list", *where])
    assert listed.exit_code == 0, listed.output
    assert "The queue holds 2 runs. It is paused" in listed.output
    assert listed.output.index("1. Heads first") < listed.output.index("2. Then layers")
    assert "tiny-gpt2 · queued " in listed.output

    missing = runner.invoke(
        cli,
        [
            "queue",
            "add",
            _spec_file(project, spec_factory, "Lost", dataset={"path": "datasets/gone.jsonl"}),
        ],
    )
    assert missing.exit_code == 1 and "gone.jsonl" in missing.output
    kept = runner.invoke(cli, ["queue", "clear", *where], input="n\n")
    assert kept.exit_code == 1 and "This takes out the 2 runs waiting in the queue." in kept.output
    assert len(runqueue.read_queue(project)) == 2
    cleared = runner.invoke(cli, ["queue", "clear", "--yes", *where])
    assert cleared.exit_code == 0 and "Cleared the queue." in cleared.output
    assert runqueue.read_queue(project) == []
    again = runner.invoke(cli, ["queue", "clear", *where])
    assert again.exit_code == 0 and "already empty" in again.output


def test_logogram_queue_run_runs_the_queue_to_the_end(project, spec_factory, loads_tiny):
    where = ["--project", str(project.root)]
    for name in ("First", "Second"):
        runqueue.add_to_queue(project, _spec(spec_factory, name))
    gone = project.datasets_dir / "gone.jsonl"
    gone.write_bytes((project.datasets_dir / "ioi.jsonl").read_bytes())
    runqueue.add_to_queue(
        project, _spec(spec_factory, "Lost", dataset={"path": "datasets/gone.jsonl"})
    )
    runqueue.add_to_queue(project, _spec(spec_factory, "Last"))
    gone.unlink()  # it can't start when its turn comes

    result = runner.invoke(cli, ["queue", "run", *where])
    assert result.exit_code == 1, result.output  # one run couldn't start
    out = result.output
    assert out.index("[1 of 4] First") < out.index("[2 of 4] Second") < out.index("[3 of 4] Lost")
    assert "[4 of 4] Last" in out
    assert "It couldn't start and left the queue: gone.jsonl" in out
    assert "The queue is empty. Of its 4 runs: 3 finished, 1 couldn't start." in out
    runs = {r.name: r for r in project.list_runs()}
    assert set(runs) == {"First", "Second", "Last"}
    assert all(r.status == "finished" for r in runs.values())
    assert runqueue.read_queue(project) == [] and runqueue.queue_runner(project) is None
    nothing = runner.invoke(cli, ["queue", "run", *where])
    assert nothing.exit_code == 0 and "there was nothing to run" in nothing.output


def test_logogram_queue_run_runs_a_queued_draft_in_its_folder(project, spec_factory, loads_tiny):
    from logogram.runner import write_json

    spec = _spec(spec_factory, "Draft")
    draft, folder = project.new_run_dir(spec.name)
    write_json(folder / "spec.json", spec.model_dump(mode="json"))
    runqueue.add_to_queue(
        project, spec, draft_id=draft, derived_from={"run": draft, "kind": "rerun"}
    )
    result = runner.invoke(cli, ["queue", "run", "--project", str(project.root)])
    assert result.exit_code == 0, result.output
    listing = project.run_listing(draft)
    assert listing is not None and listing.status == "finished"
    assert listing.derived_from is not None and listing.derived_from.kind == "rerun"


def test_logogram_queue_run_wont_run_a_queue_the_app_runs(server, project, spec_factory):
    state, client = server()
    state.open_project(project)
    state.job = Job(id="busy", kind="run", title="A run in progress")
    client.post(
        "/api/queue",
        json={"spec": _spec(spec_factory, "Theirs").model_dump(mode="json")},
        headers=HEADERS,
    )
    assert client.get("/api/queue/state", headers=HEADERS).json()["state"] == "waiting"
    result = runner.invoke(cli, ["queue", "run", "--project", str(project.root)])
    assert result.exit_code == 1
    assert f"already being run by the Logogram app (process {os.getpid()})" in result.output
    assert "Pause it in the app" in result.output
    listed = runner.invoke(cli, ["queue", "list", "--project", str(project.root)])
    assert f"It is being run by the Logogram app (process {os.getpid()})." in listed.output
    assert _titles(runqueue.read_queue(project)) == ["Theirs"]


def test_ctrl_c_stops_the_queue_and_keeps_the_rest(
    project, spec_factory, tiny_backend, monkeypatch
):
    from logogram import runner as runs

    real = runs.run_spec

    def interrupted_mid_sweep(spec: Any, project: Any, **kwargs: Any) -> Any:
        given = kwargs.pop("on_event")

        def on_event(kind: str, data: dict[str, Any]) -> None:
            given(kind, data)
            if spec.name == "Second" and kind == "progress" and data["layer"] == 1:
                raise KeyboardInterrupt

        kwargs["backend"] = tiny_backend
        return real(spec, project, on_event=on_event, **kwargs)

    monkeypatch.setattr(runs, "run_spec", interrupted_mid_sweep)
    heads = {"kind": "heads", "position": {"kind": "all"}}
    for name in ("First", "Second", "Third", "Fourth"):
        runqueue.add_to_queue(project, spec_factory(name=name, scope=heads))
    result = runner.invoke(cli, ["queue", "run", "--project", str(project.root)])
    assert result.exit_code == 130, result.output
    assert "Interrupted. The run is recorded as cancelled." in result.output
    assert "Stopped after 1 of 2 layers: these are the layers that finished." in result.output
    assert "Stopped. 2 runs are still queued: `logogram queue run` runs them." in result.output
    assert _titles(runqueue.read_queue(project)) == ["Third", "Fourth"]
    assert runqueue.queue_runner(project) is None
    statuses = {r.name: r for r in project.list_runs()}
    assert statuses["First"].status == "finished"
    assert statuses["Second"].status == "cancelled" and statuses["Second"].partial is not None


def test_ctrl_c_before_a_queued_run_starts_puts_it_back(project, spec_factory, monkeypatch):
    from logogram import cli as commands

    made = commands._queued_folder

    def interrupted_as_it_starts(project: Any, item: Any) -> str:
        made(project, item)  # its folder is made, and then Ctrl+C comes before the run starts
        raise KeyboardInterrupt

    monkeypatch.setattr(commands, "_queued_folder", interrupted_as_it_starts)
    for name in ("First", "Second"):
        runqueue.add_to_queue(project, _spec(spec_factory, name))
    before = runqueue.read_queue(project)
    result = runner.invoke(cli, ["queue", "run", "--project", str(project.root)])
    assert result.exit_code == 130, result.output
    assert "Stopped. 2 runs are still queued: `logogram queue run` runs them." in result.output
    # Back at the front of the queue, as it was, and no folder is left behind for it.
    assert runqueue.read_queue(project) == before
    assert project.list_runs() == []
    assert runqueue.queue_runner(project) is None


def test_a_version_1_spec_says_what_it_left_out_when_queued_or_swept(project, tiny_backend):
    from logogram.runner import run_spec
    from logogram.spec import Spec
    from logogram.sweeps import get_field, variant_spec

    old = Spec.model_validate(
        {
            "logogram_spec": 1,
            "name": "Old",
            "model": {"id": "tiny-gpt2", "revision": "test", "device": "cpu"},
            "dataset": {"path": "datasets/ioi.jsonl", "limit": 4},
            "experiment": {"kind": "activation_patching", "direction": "clean_to_corrupt"},
            "scope": {"kind": "layer_components", "position": {"kind": "last"}},
            "execution": {"batch_size": 16},
            "statistics": {"bootstrap": 200, "seed": 0},
        }
    )
    filled = old.upgraded_fields
    assert filled
    runqueue.add_to_queue(project, old)
    (queued,) = runqueue.read_queue(project)
    assert queued.spec.upgraded_fields == filled
    swept = variant_spec(old, get_field("statistics.seed"), 1)
    assert swept.upgraded_fields == filled
    for spec in (queued.spec, swept):
        outcome = run_spec(spec, project, backend=tiny_backend)
        assert outcome.summary is not None
        assert outcome.summary["warnings"][0].startswith("This spec is version 1")
