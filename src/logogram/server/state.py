"""Server state: the open project, the loaded model, the running job and the event stream."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from logogram.backends.base import Cancelled, ModelBackend
from logogram.project import Project, config_dir, remember_recent
from logogram.schema import validate_event
from logogram.spec import ModelRef, Spec

log = logging.getLogger(__name__)
request_project: ContextVar[tuple[Project | None] | None] = ContextVar(
    "request_project", default=None
)


class Conflict(RuntimeError):
    """Another job is running."""


class Missing(Exception):
    """Something the request needs isn't there (no project open, no model loaded, no such run)."""


QUEUE_SIZE = 2000
THROTTLED = ("progress", "run.progress")
RUN_ENDED = ("run.finished", "run.failed", "run.cancelled")


class EventHub:
    """Broadcast events from worker threads to every connected WebSocket.

    A tab that connects mid-run first receives the run so far, so its map fills in; a tab that
    reconnects after a run ended learns how it ended. A tab that falls far behind is disconnected
    rather than silently missing events; it reconnects and catches up the same way.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queues: set[asyncio.Queue[str | None]] = set()
        self._last_progress: dict[str, float] = {}
        self._lock = threading.Lock()
        # The current run's events, marked as replayed (only touched on the event loop).
        self._replay: list[tuple[str, str]] = []

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue[str | None]:
        """Call on the event loop. The queue starts with the current run's events."""
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=QUEUE_SIZE + len(self._replay))
        for _, message in self._replay:
            queue.put_nowait(message)
        self._queues.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[str | None]) -> None:
        self._queues.discard(queue)

    def publish(self, kind: str, data: dict[str, Any]) -> None:
        if kind in THROTTLED:
            # Progress can fire many times a second; the status line needs ~10 updates a second.
            now = time.monotonic()
            with self._lock:
                final = data.get("done") == data.get("total")
                if not final and now - self._last_progress.get(kind, 0.0) < 0.1:
                    return
                self._last_progress[kind] = now
        message = json.dumps({"type": kind, **data})
        # Replayed copies say so, so the app doesn't repeat one-time notices.
        replay = (
            json.dumps({"type": kind, **data, "replay": True}) if kind.startswith("run.") else ""
        )
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        with contextlib.suppress(RuntimeError):  # the loop closed in between (shutdown)
            loop.call_soon_threadsafe(self._deliver, kind, message, replay)

    def _deliver(self, kind: str, message: str, replay: str) -> None:
        if kind == "run.started" or kind in RUN_ENDED:
            # A new run starts the record over; a run that ended leaves only how it ended.
            self._replay = [(kind, replay)]
        elif kind.startswith("run.") and self._replay:
            if kind == "run.progress" and self._replay[-1][0] == kind:
                self._replay[-1] = (kind, replay)  # only the latest progress matters
            else:
                self._replay.append((kind, replay))
        for queue in list(self._queues):
            _offer(queue, message)


def _offer(queue: asyncio.Queue[str | None], message: str) -> None:
    try:
        queue.put_nowait(message)
    except asyncio.QueueFull:
        # The tab has fallen far behind: drop its backlog and close it (None), so it reconnects
        # and catches up instead of showing a run that never ends.
        while not queue.empty():
            queue.get_nowait()
        queue.put_nowait(None)


@dataclass
class Job:
    id: str
    kind: str  # "load_model" | "run"
    title: str
    status: str = "running"  # running | finished | failed | cancelled
    run_id: str | None = None
    project_session: str | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    started: float = field(default_factory=time.time)
    cancel: threading.Event = field(default_factory=threading.Event)
    # Increases with every change of status, so the UI can tell an old snapshot from a new one.
    version: int = 1
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(self, key, value)
            self.version += 1

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "title": self.title,
                "status": self.status,
                "run_id": self.run_id,
                "project_session": self.project_session,
                "progress": self.progress,
                "error": self.error,
                "started": self.started,
                "version": self.version,
            }


class AppState:
    def __init__(self, hub: EventHub) -> None:
        self.hub = hub
        self.project: Project | None = None
        self.backend: ModelBackend | None = None
        self.model_status: dict[str, Any] = {"state": "none"}
        self.model_ref: ModelRef | None = None  # what the user asked for (device may be "auto")
        self.job: Job | None = None
        self._job_lock = threading.Lock()

    # -- settings ----------------------------------------------------------------------------

    @staticmethod
    def settings_path() -> Path:
        return config_dir() / "settings.json"

    def settings(self) -> dict[str, Any]:
        try:
            return json.loads(self.settings_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def update_settings(self, **values: Any) -> None:
        data = self.settings()
        data.update(values)
        self.settings_path().parent.mkdir(parents=True, exist_ok=True)
        self.settings_path().write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    # -- project -----------------------------------------------------------------------------

    def open_project(self, project: Project) -> None:
        from logogram.runner import write_json

        with self._job_lock:
            self._check_switch()
            # A running manifest from an earlier process is an interrupted run, never a draft.
            for listing in project.list_runs():
                if listing.status == "running":
                    project.interrupted_runs.add(listing.id)
                    path = project.run_dir(listing.id) / "manifest.json"
                    data = project.read_json(path)
                    assert data is not None
                    data.update(
                        status="failed",
                        error="The server stopped before this run finished. Rerun its saved spec.",
                    )
                    with contextlib.suppress(OSError):  # read-only projects remain inspectable
                        write_json(project.writable(path), data)
            self.project = project
        remember_recent(project)
        self.hub.publish("project", {"session_id": project.session_id})

    def _check_switch(self) -> None:
        if self.job is not None and self.job.status == "running":
            raise Conflict(
                "Wait for the running job to finish, or cancel it, before changing projects."
            )

    def close_project(self) -> None:
        with self._job_lock:
            captured = request_project.get()
            if captured is not None and captured[0] is not self.project:
                raise Conflict(
                    "The project changed in another tab. Wait for this tab to refresh, then try again."
                )
            self._check_switch()
            self.project = None
        self.hub.publish("project", {"session_id": None})

    def require_project(self) -> Project:
        captured = request_project.get()
        if captured is not None and captured[0] is not self.project:
            raise Conflict(
                "The project changed in another tab. Wait for this tab to refresh, then try again."
            )
        if self.project is None:
            raise Missing("Open a project first.")
        return self.project

    # -- model -------------------------------------------------------------------------------

    def require_backend(self) -> ModelBackend:
        if self.backend is None:
            raise Missing("Load a model first.")
        return self.backend

    def model_payload(self) -> dict[str, Any]:
        payload = dict(self.model_status)
        if self.backend is not None:
            payload["info"] = self.backend.info.to_dict()
            payload["memory"] = self.backend.memory_in_use()
            if self.model_ref is not None:
                payload["ref"] = self.model_ref.model_dump()
        return payload

    def _set_model_status(self, **status: Any) -> None:
        self.model_status = status
        self.hub.publish("model", self.model_payload())

    def load_model(self, ref: ModelRef, cancel: threading.Event | None = None) -> ModelBackend:
        """Load (or reuse) a model. Runs on a worker thread.

        The current model is closed first to make room, so if the new one fails to load, no model
        is loaded. Closing waits for an analysis that is using the old model.
        """
        from logogram.backends.transformer_lens import load_model
        from logogram.runner import model_matches

        if model_matches(self.backend, ref):
            return self.backend  # type: ignore[return-value]
        if self.backend is not None:
            old, self.backend = self.backend, None
            old.close()
        self._set_model_status(state="loading", id=ref.id, stage="resolving")

        last = {"stage": None, "time": 0.0}

        def on_progress(p: dict[str, Any]) -> None:
            self.model_status = {"state": "loading", "id": ref.id, **p}
            if self.job is not None:
                self.job.progress = {"model": p}
            # Downloads report often; ~10 updates a second are enough for a progress bar.
            now = time.monotonic()
            final = p.get("done") == p.get("total")
            if p.get("stage") != last["stage"] or final or now - last["time"] >= 0.1:
                last.update(stage=p.get("stage"), time=now)
                self.hub.publish("model", self.model_payload())

        try:
            backend = load_model(
                ref.id,
                revision=ref.revision,
                dtype=ref.dtype,
                device=ref.device,
                process_weights=ref.process_weights,
                on_progress=on_progress,
                cancel=cancel,
            )
        except Cancelled:
            self._set_model_status(state="none")
            raise
        except Exception as exc:
            self._set_model_status(state="error", id=ref.id, error=str(exc))
            raise
        self.backend = backend
        self.model_ref = ref.model_copy(update={"revision": backend.info.revision})
        self._set_model_status(state="ready")
        return backend

    def unload_model(self) -> None:
        if self.job is not None and self.job.status == "running":
            raise Conflict("Wait for the running job to finish, or cancel it, before unloading.")
        backend, self.backend = self.backend, None
        self.model_ref = None
        if backend is not None:
            backend.close()  # waits for an analysis that is using it
        self._set_model_status(state="none")

    # -- jobs --------------------------------------------------------------------------------

    def start_job(
        self,
        kind: str,
        title: str,
        work: Callable[[Job], None],
        prepare: Callable[[], str | None] | None = None,
    ) -> Job:
        """Run ``work`` on a worker thread, unless another job is running.

        ``prepare`` runs first, under the same lock, so nothing is written for a job that can't
        start (a double-click on Run, say). It returns the job's run id, if it has one.
        """
        with self._job_lock:
            if self.job is not None and self.job.status == "running":
                raise Conflict(
                    f"“{self.job.title}” is still running. Wait for it to finish or cancel it."
                )
            run_id = prepare() if prepare is not None else None
            job = Job(
                id=uuid.uuid4().hex[:12],
                kind=kind,
                title=title,
                run_id=run_id,
                project_session=self.project.session_id if run_id and self.project else None,
            )
            self.job = job
        self.hub.publish("job", job.to_dict())

        def target() -> None:
            try:
                work(job)
                if job.status == "running":
                    job.update(status="finished")
            except Cancelled:
                job.update(status="cancelled")
            except Exception as exc:  # noqa: BLE001 - reported to the UI
                log.exception("job failed")
                job.update(status="failed", error=str(exc) or type(exc).__name__)
            self.hub.publish("job", job.to_dict())

        threading.Thread(target=target, name=f"logogram-{kind}", daemon=True).start()
        return job

    def run_spec_job(
        self,
        spec: Spec,
        derived_from: dict[str, Any] | None = None,
        draft_id: str | None = None,
    ) -> Job:
        """Start a run. A saved draft (a spec that hasn't run) runs in its own folder."""
        from logogram.runner import run_spec, write_json

        project = self.require_project()

        def prepare() -> str:
            self.require_project()  # check again under the switch/job lock
            project.resolve_dataset(spec.dataset.path)
            if draft_id is not None:
                folder = project.run_dir(draft_id)
                project.readable(folder / "spec.json")
                if (folder / "manifest.json").exists():
                    raise Conflict(f"{draft_id} has already run. Start a new run instead.")
                run_id, folder = draft_id, project.prepare_run_dir(draft_id)
            else:
                run_id, folder = project.new_run_dir(spec.name)
            # Write the spec now, so the history lists the run as soon as it is started.
            write_json(folder / "spec.json", spec.model_dump(mode="json"))
            return run_id

        def work(job: Job) -> None:
            def on_event(kind: str, data: dict[str, Any]) -> None:
                if kind == "progress":
                    job.progress = data
                if kind == "model":
                    self.model_status = {"state": "loading", "id": spec.model.id, **data}
                    self.hub.publish("model", self.model_payload())
                    return
                self.hub.publish(
                    f"run.{kind}",
                    {**validate_event(f"run.{kind}", data), "project_session": project.session_id},
                )

            outcome = run_spec(
                spec,
                project,
                run_id=job.run_id,
                backend=self.backend,
                provider=lambda ref: self.load_model(ref, cancel=job.cancel),
                on_event=on_event,
                cancel=job.cancel,
                derived_from=derived_from,
            )
            if outcome.backend is not None and outcome.backend is not self.backend:
                self.backend = outcome.backend
                self._set_model_status(state="ready")
            job.update(
                status=outcome.status,
                error=outcome.manifest.get("error") if outcome.status == "failed" else None,
            )

        return self.start_job("run", spec.name, work, prepare=prepare)

    def save_draft(self, spec: Spec, draft_id: str | None = None) -> str:
        """Save ``spec`` as a new draft, or over the draft ``draft_id`` if it hasn't run."""
        from logogram.runner import write_json

        project = self.require_project()
        with self._job_lock:  # a run can't start in the folder while it is being written
            self.require_project()
            project.resolve_dataset(spec.dataset.path)
            if draft_id is None:
                run_id, folder = project.new_run_dir(spec.name)
            else:
                folder = project.run_dir(draft_id)
                project.readable(folder / "spec.json")
                running = self.job is not None and self.job.status == "running"
                if (folder / "manifest.json").exists() or (running and self.job.run_id == draft_id):
                    raise Conflict(
                        f"{draft_id} has run, so its spec stays as it ran. Save a new draft instead."
                    )
                run_id, folder = draft_id, project.prepare_run_dir(draft_id)
            write_json(folder / "spec.json", spec.model_dump(mode="json"))
        return run_id

    def load_model_job(self, ref: ModelRef) -> Job:
        def work(job: Job) -> None:
            self.load_model(ref, cancel=job.cancel)

        return self.start_job("load_model", f"Load {ref.id}", work)

    def cancel_job(self) -> Job | None:
        job = self.job
        if job is not None and job.status == "running":
            job.cancel.set()
            job.update()
            self.hub.publish("job", {**job.to_dict(), "cancelling": True})
        return job
