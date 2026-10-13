"""Server state: the open project, the loaded model, the running job, the runs queued behind it
and the event stream."""

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

from logogram import runqueue
from logogram.backends.base import Cancelled, ModelBackend
from logogram.fileio import write_text_atomic
from logogram.project import (
    DeletionPlan,
    Project,
    ProjectError,
    config_dir,
    remember_recent,
)
from logogram.project import deletion_plan as plan_deletion
from logogram.runqueue import Queued, QueueError
from logogram.schema import QueueRunner, RunListing, validate_event
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


@dataclass(frozen=True, eq=False)
class QueuedRun:
    """A run waiting for the job in progress to end: what ``run_spec_job`` starts it with."""

    spec: Spec
    project_session: str
    draft_id: str | None = None
    derived_from: dict[str, Any] | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    queued_at: float = field(default_factory=time.time)

    @classmethod
    def of(cls, item: Queued, session: str) -> QueuedRun:
        """A run of the project's saved queue, in the open project (``session``)."""
        return cls(item.spec, session, item.draft_id, item.derived_from, item.id, item.queued_at)

    @property
    def title(self) -> str:
        return self.spec.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "spec": self.spec.model_dump(mode="json"),
            "draft_id": self.draft_id,
            "derived_from": self.derived_from,
            "project_session": self.project_session,
            "queued_at": self.queued_at,
        }


class AppState:
    def __init__(self, hub: EventHub) -> None:
        self.hub = hub
        self.project: Project | None = None
        self.backend: ModelBackend | None = None
        self.model_status: dict[str, Any] = {"state": "none"}
        self.model_ref: ModelRef | None = None  # what the user asked for (device may be "auto")
        # The loaded sparse autoencoder, if any. It belongs to the loaded model: changing the
        # model drops it.
        self.sae: Any = None
        self.sae_status: dict[str, Any] = {"state": "none"}
        self.job: Job | None = None
        # Re-entrant, so the queue can decide that a run may start and start it in one hold.
        self._job_lock = threading.RLock()
        # The open project's queue is kept in the project (logogram.runqueue), so it survives a
        # restart of the server and closing the project. This server runs it while it holds it
        # (see queue_run and start_queue): its runner record, and the job it started last. Only
        # changed under _job_lock.
        self._queue_hold: QueueRunner | None = None
        self._queue_job: str | None = None
        self._stop = threading.Event()
        self._update_lock = threading.Lock()

    # -- updates -----------------------------------------------------------------------------

    @staticmethod
    def updates_path() -> Path:
        return config_dir() / "updates.json"

    # Replaced in tests; the only network call Logogram makes on its own, and only with consent.
    fetch_updates = None

    def update_status(self) -> dict[str, Any]:
        from logogram import updates

        choice = self.settings().get("update_check")
        cache = updates.read_cache(self.updates_path())
        return updates.status(choice if isinstance(choice, bool) else None, cache).to_dict()

    def check_updates(self) -> dict[str, Any]:
        """Ask PyPI for the newest version now, then tell every open tab."""
        from logogram import updates

        with self._update_lock:  # one check at a time
            updates.check(self.updates_path(), fetch=self.fetch_updates or updates.fetch_pypi)
        payload = self.update_status()
        self.hub.publish("update", payload)
        return payload

    def maybe_check_updates(self) -> bool:
        """The daily check: only with the user's consent, and only once the last one is a day old."""
        from logogram import updates

        if self.settings().get("update_check") is not True:
            return False
        if not updates.due(updates.read_cache(self.updates_path())):
            return False
        self.check_updates()
        return True

    def start_update_checks(self) -> None:
        """While the server runs, look once an hour whether a daily check is allowed and due."""

        def loop() -> None:
            wait = 5.0  # let the server start first
            while not self._stop.wait(wait):
                wait = 3600.0
                self.maybe_check_updates()

        threading.Thread(target=loop, name="logogram-updates", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        # The server is stopping: its queue stays in the project, paused.
        with self._job_lock:
            self._release_queue(publish=False)

    # -- settings ----------------------------------------------------------------------------

    @staticmethod
    def settings_path() -> Path:
        return config_dir() / "settings.json"

    def settings(self) -> dict[str, Any]:
        """The settings file's object; {} when there is none, or it holds anything else (null, a
        list, text that isn't JSON), as if nothing had been set."""
        try:
            data = json.loads(self.settings_path().read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):
            return {}
        return data if isinstance(data, dict) else {}

    def update_settings(self, **values: Any) -> None:
        """Change some settings, keeping the others. The file is replaced whole, never left half
        written."""
        data = self.settings()
        data.update(values)
        self.settings_path().parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(self.settings_path(), json.dumps(data, indent=2) + "\n")

    # -- project -----------------------------------------------------------------------------

    def open_project(self, project: Project) -> None:
        """Open ``project``. Runs it had queued come back paused: nothing starts on its own."""
        with self._job_lock:
            self._check_switch()
            mark_interrupted_runs(project)
            # The queue of the project open until now stays in it, paused.
            self._release_queue(publish=False)
            self.project = project
        try:
            remember_recent(project)
        except Exception as exc:  # noqa: BLE001 - the recent list never stops a project opening
            log.warning("Couldn't add the project to the recent projects: %s", _reason(exc))
        log.info("Opened the project %s", project.root)
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
            self._release_queue(publish=False)  # its queue stays in it, paused
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

    def project_changed(
        self, project: Project, dataset: str | None = None, to: str | None = None
    ) -> None:
        """Tell the tabs showing ``project`` to read it again: its name, or its datasets after
        the file ``dataset`` was renamed ``to`` another path or deleted (``to`` None)."""
        payload: dict[str, Any] = {"project_session": project.session_id}
        if dataset is not None:
            payload["dataset"] = {"path": dataset, "to": to}
        self.hub.publish("project.changed", validate_event("project.changed", payload))

    def rename_project(self, name: str) -> Project:
        """Give the open project another name, in its project.json."""
        project = self.require_project()
        with self._job_lock:
            self.require_project()
            project.rename(name)
        self.project_changed(project)
        return project

    def delete_dataset(self, name: str) -> str:
        """Delete a dataset file of the open project (see logogram.datasets.delete_dataset):
        under the job lock, so no run starts on it meanwhile. Returns the path it had."""
        from logogram.datasets import delete_dataset

        project = self.require_project()
        with self._job_lock:
            self.require_project()
            rel = delete_dataset(project, name)
        self.project_changed(project, rel)
        return rel

    def rename_dataset(self, name: str, new: str) -> str:
        """Give a dataset file of the open project another name (see
        logogram.datasets.rename_dataset). Returns its new path."""
        from logogram.datasets import existing_dataset, rename_dataset

        project = self.require_project()
        with self._job_lock:
            self.require_project()
            old = f"datasets/{existing_dataset(project, name).name}"
            rel = rename_dataset(project, name, new)
        self.project_changed(project, old, rel)
        return rel

    def project_at(self, path: str | None) -> Project:
        """The open project (``path`` None, or its folder), or the project in the folder
        ``path``, read without writing into it."""
        if path is None:
            return self.require_project()
        target = Project.read(path)
        current = self.project
        return current if current is not None and current.root == target.root else target

    def deletion_plan(self, path: str | None) -> DeletionPlan:
        """What deleting the project at ``path`` (the open one when None) would delete, and what
        stops it, with what this server is doing in it."""
        with self._job_lock:
            target = self.project_at(path)
            plan = plan_deletion(target)
            busy = self._busy_in(target)
        if busy and not plan.busy:
            plan.busy = busy
        return plan

    def _busy_in(self, project: Project) -> str | None:
        """Why this server keeps ``project`` from being deleted now (under _job_lock)."""
        if project is not self.project:
            return None
        job = self.job
        if job is not None and job.status == "running":
            return (
                f"“{job.title}” is running. Wait for it to end, or cancel it, before deleting "
                "the project."
            )
        if self._holds_queue(project):
            return (
                "This project's queue is being run here. Pause it, or wait for it to finish, "
                "before deleting the project."
            )
        return None

    def delete_project(self, path: str | None, name: str) -> None:
        """Delete the project at ``path`` (the open one when None) for good, once ``name`` is
        typed as the project is called. The open project is closed first. Refused while a job
        runs in it or this server runs its queue, and whenever logogram.project.delete_project
        refuses: a folder that holds what Logogram didn't make, a run in progress elsewhere."""
        from logogram.project import ProjectInUse, delete_project

        closed = False
        try:
            with self._job_lock:
                target = self.project_at(path)
                if name.strip() != target.meta.name:
                    raise ProjectError(
                        f"Type the project's name, “{target.meta.name}”, exactly, to delete it."
                    )
                busy = self._busy_in(target)
                if busy:
                    raise ProjectInUse(busy)
                plan = plan_deletion(target)
                if plan.problem:
                    raise (ProjectError if plan.others else ProjectInUse)(plan.problem)
                if target is self.project:
                    self._release_queue(publish=False)  # its queue goes with it
                    self.project = None
                    closed = True
                delete_project(target)
            log.info("Deleted the project %s", target.root)
        finally:
            if closed:
                self.hub.publish("project", {"session_id": None})

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

        What can be checked without loading anything (the device, the dtype, a local model's
        folder) is checked first, so a request that can't load, such as one for a GPU this machine
        lacks, leaves the loaded model as it is. Then the current model is closed to make room,
        so if the new one fails to load, no model is loaded. Closing waits for an analysis that is
        using the old model.
        """
        from logogram.backends.transformer_lens import check_request, load_model
        from logogram.runner import model_matches

        if model_matches(self.backend, ref):
            return self.backend  # type: ignore[return-value]
        try:
            check_request(
                ref.id,
                revision=ref.revision,
                dtype=ref.dtype,
                device=ref.device,
                source=ref.source,
                sha256=ref.sha256,
            )
        except Exception as exc:
            if self.backend is None:  # otherwise the loaded model stays, and its status with it
                self._set_model_status(state="error", id=ref.id, error=str(exc))
            raise
        if self.backend is not None:
            old, self.backend = self.backend, None
            old.close()
        self._drop_sae()
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
                source=ref.source,
                sha256=ref.sha256,
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
        with self._job_lock:  # a job can't start with the model while it is being unloaded
            if self.job is not None and self.job.status == "running":
                raise Conflict(
                    "Wait for the running job to finish, or cancel it, before unloading."
                )
            backend, self.backend = self.backend, None
            self.model_ref = None
        if backend is not None:
            backend.close()  # waits for an analysis that is using it
        self._set_model_status(state="none")
        self._drop_sae()

    # -- sparse autoencoder ------------------------------------------------------------------

    def sae_payload(self) -> dict[str, Any]:
        payload = dict(self.sae_status)
        if self.sae is not None:
            payload["info"] = self.sae.describe()
        return payload

    def _set_sae_status(self, **status: Any) -> None:
        self.sae_status = status
        self.hub.publish("sae", self.sae_payload())

    def _drop_sae(self) -> None:
        if self.sae is not None or self.sae_status.get("state") != "none":
            self.sae = None
            self._set_sae_status(state="none")

    def load_sae(self, ref: Any, cancel: threading.Event | None = None) -> Any:
        """Load (or reuse) an SAE for the loaded model. Runs on a worker thread."""
        from logogram.sae import load_sae, sae_matches

        backend = self.require_backend()
        if sae_matches(self.sae, ref):
            return self.sae
        self.sae = None
        self._set_sae_status(state="loading", repo=ref.repo, path=ref.path, stage="downloading")
        last = {"time": 0.0}

        def progress(done: int, total: int, file: str) -> None:
            now = time.monotonic()
            if done == total or now - last["time"] >= 0.1:
                last["time"] = now
                self._set_sae_status(
                    state="loading",
                    repo=ref.repo,
                    path=ref.path,
                    stage="downloading",
                    done=done,
                    total=total,
                )

        try:
            sae = load_sae(ref, backend.info.device, progress, cancel)
            from logogram.features import check_sae_fits

            check_sae_fits(sae, backend.info)
        except Cancelled:
            self._set_sae_status(state="none")
            raise
        except Exception as exc:
            self._set_sae_status(state="error", repo=ref.repo, path=ref.path, error=str(exc))
            raise
        self.sae = sae
        self._set_sae_status(state="ready")
        return sae

    def load_sae_job(self, ref: Any) -> Job:
        def work(job: Job) -> None:
            self.load_sae(ref, cancel=job.cancel)

        return self.start_job("load_sae", f"Load the SAE {ref.repo}", work)

    def unload_sae(self) -> None:
        with self._job_lock:
            if self.job is not None and self.job.status == "running":
                raise Conflict(
                    "Wait for the running job to finish, or cancel it, before unloading."
                )
            self._drop_sae()

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
            log.info("Started the job %s (%s): %s", job.id, kind, title)
            try:
                work(job)
                if job.status == "running":
                    job.update(status="finished")
            except Cancelled:
                job.update(status="cancelled")
            except Exception as exc:
                log.exception("job failed")
                job.update(status="failed", error=str(exc) or type(exc).__name__)
            reason = f" ({job.error})" if job.error else ""
            log.info("The job %s ended: %s%s", job.id, job.status, reason)
            self.hub.publish("job", job.to_dict())
            # This job has let go of the server, however it ended: the queue's next run starts.
            self._start_next()

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
            self._check_run(project, spec, draft_id)
            if draft_id is not None:
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
                sae=self.sae,
                sae_provider=lambda ref, backend: self.load_sae(ref, cancel=job.cancel),
            )
            if outcome.backend is not None and outcome.backend is not self.backend:
                self.backend = outcome.backend
                self._set_model_status(state="ready")
            job.update(
                status=outcome.status,
                error=outcome.manifest.get("error") if outcome.status == "failed" else None,
            )

        return self.start_job("run", spec.name, work, prepare=prepare)

    def _check_run(self, project: Project, spec: Spec, draft_id: str | None) -> None:
        """What a run needs to start: its dataset, and a draft, whose folder it fills, that
        hasn't run."""
        project.resolve_dataset(spec.dataset.path)
        if draft_id is not None:
            folder = project.run_dir(draft_id)
            project.readable(folder / "spec.json")
            if (folder / "manifest.json").exists():
                raise Conflict(f"{draft_id} has already run. Start a new run instead.")

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
                job = self.job
                running = job is not None and job.status == "running" and job.run_id == draft_id
                if (folder / "manifest.json").exists() or running:
                    raise Conflict(
                        f"{draft_id} has run, so its spec stays as it ran. Save a new draft instead."
                    )
                if self._queued_draft(project, draft_id):
                    raise Conflict(
                        f"{draft_id} is queued to run as it was saved. Remove it from the queue "
                        "to change it, or save a new draft."
                    )
                run_id, folder = draft_id, project.prepare_run_dir(draft_id)
            write_json(folder / "spec.json", spec.model_dump(mode="json"))
        return run_id

    def load_model_job(self, ref: ModelRef) -> Job:
        def work(job: Job) -> None:
            self.load_model(ref, cancel=job.cancel)

        return self.start_job("load_model", f"Load {ref.id}", work)

    def add_model_job(self, folder: Path, name: str) -> Job:
        """Hash a model folder and add it to the local models, reporting progress in bytes."""
        from logogram.local_models import add_model

        def work(job: Job) -> None:
            last = {"time": 0.0}

            def progress(done: int, total: int, file: str) -> None:
                now = time.monotonic()
                if done == total or now - last["time"] >= 0.1:
                    last["time"] = now
                    job.update(progress={"done": done, "total": total, "file": file})
                    self.hub.publish("job", job.to_dict())

            model = add_model(folder, name, progress, job.cancel)
            job.update(progress={"done": model.size, "total": model.size, "file": ""})
            self.hub.publish("local_models", {"added": model.name})

        return self.start_job("add_model", f"Add the model folder {name}", work)

    def cancel_job(self) -> Job | None:
        """Cancel the job in progress. The queue stays: its next run starts when this one ends."""
        job = self.job
        if job is not None and job.status == "running":
            job.cancel.set()
            job.update()
            self.hub.publish("job", {**job.to_dict(), "cancelling": True})
        return job

    # -- the run queue -----------------------------------------------------------------------
    #
    # The queue is the open project's, kept in the project (logogram.runqueue): it survives a
    # restart of the server and closing the project, and `logogram queue add` adds to it. This
    # server runs it while it holds it: from when runs are queued behind a job here on an empty
    # queue, or someone starts it, until it is empty or paused. A queue that comes back with its
    # project is paused, and only one process runs a queue at a time.

    def queued_runs(self) -> list[dict[str, Any]]:
        """The runs waiting in the open project's queue, first in line first."""
        project = self.project
        if project is None:
            return []
        try:
            items = runqueue.read_queue(project)
        except (QueueError, OSError):
            return []
        return [QueuedRun.of(item, project.session_id).to_dict() for item in items]

    def queue_status(self) -> dict[str, Any]:
        """The open project's queue and how it stands (see schema.QueueChanged): ``waiting``
        when this server starts its next run as the job in progress ends, ``running`` when one
        of its runs is in progress here (or has just ended, its next yet to start) or another
        process runs it, ``paused`` when nothing starts until it is started, ``idle`` when it is
        empty."""
        project = self.project
        if project is None:
            return {"project_session": None, "runs": [], "state": "idle"}
        error = None
        try:
            items = runqueue.read_queue(project)
        except (QueueError, OSError) as exc:
            items, error = [], _reason(exc)
        runner = runqueue.queue_runner(project)
        mine = self._queue_hold is not None and runner is not None
        mine = mine and runner.token == self._queue_hold.token  # type: ignore[union-attr]
        job = self.job
        if mine:
            # The queue's own job runs it, even once that job has ended and the next run is yet
            # to start (a quick run can end before the tabs hear that it started). The queue
            # waits only behind a job that isn't one of its own.
            from_queue = job is not None and job.id == self._queue_job
            state = "running" if from_queue else "waiting"
        elif runner is not None:
            state = "running"
        else:
            state = "paused" if items else "idle"
        return {
            "project_session": project.session_id,
            "runs": [QueuedRun.of(item, project.session_id).to_dict() for item in items],
            "state": state,
            "runner": runqueue.runner_text(runner) if runner is not None and not mine else None,
            "error": error,
        }

    def queue_run(
        self,
        spec: Spec,
        derived_from: dict[str, Any] | None = None,
        draft_id: str | None = None,
    ) -> Job | QueuedRun:
        """Run ``spec`` after the jobs ahead of it: at once when nothing is running here and the
        queue isn't about to start runs (the job is returned), otherwise last in the queue (the
        queued run is returned).

        Queued behind a job here on an empty queue, it makes the queue this server's to run: its
        runs start one at a time as each job ends. A paused queue (one that came back with its
        project) stays paused: what is added waits for Start the queue. A queue that another
        process runs (`logogram queue run`) gets the run at its end.

        When its turn comes it starts like any run, through ``run_spec_job``, which checks it
        again. What can be checked now is checked now, so a run that can't start isn't queued.
        """
        project = self.require_project()
        with self._job_lock:
            self.require_project()  # check again under the switch/job lock
            job = self.job
            running = job is not None and job.status == "running"
            broken: QueueError | None = None
            try:
                items = runqueue.read_queue(project)
            except QueueError as exc:
                items, broken = [], exc
            runner = runqueue.queue_runner(project)
            mine = self._holds_queue(project)
            if not running and (runner is None or mine) and not (mine and items):
                return self.run_spec_job(spec, derived_from, draft_id)
            if broken is not None:
                raise broken
            self._check_run(project, spec, draft_id)
            if draft_id is not None:
                if running and job is not None and job.run_id == draft_id:
                    raise Conflict(f"{draft_id} is running now. Queue a new run instead.")
                if any(item.draft_id == draft_id for item in items):
                    raise Conflict(f"{draft_id} is already queued. It runs when its turn comes.")
            item = runqueue.add_to_queue(
                project, spec, draft_id=draft_id, derived_from=derived_from
            )
            if running and runner is None and not items:
                # Behind the job in progress, on an empty queue: this server runs the queue. (A
                # process that took it in the meantime runs it instead.)
                with contextlib.suppress(Conflict):
                    self._take_queue(project)
            self._publish_queue(project.session_id)
        return QueuedRun.of(item, project.session_id)

    def remove_queued_run(self, queued_id: str) -> list[dict[str, Any]]:
        """Take a run out of the queue before it starts. Returns the runs still waiting."""
        project = self.require_project()
        with self._job_lock:
            self.require_project()
            if runqueue.remove_from_queue(project, queued_id) is None:
                raise Missing("That run isn't queued any more: it has started or was removed.")
            self._publish_queue(project.session_id)
            return self.queued_runs()

    def start_queue(self) -> dict[str, Any]:
        """Run the queue here (a paused one, say, that came back with its project): its next run
        starts now if nothing is running, and the rest follow, one at a time. Refused, saying
        who, while another process runs it."""
        project = self.require_project()
        with self._job_lock:
            self.require_project()
            if not runqueue.read_queue(project):
                raise Conflict("The queue is empty: there is nothing to start.")
            if not self._holds_queue(project):
                self._take_queue(project)
            self._publish_queue(project.session_id)
        self._start_next()
        return self.queue_status()

    def pause_queue(self) -> dict[str, Any]:
        """Stop starting the queue's runs. The run in progress goes on; the rest wait, paused,
        until the queue is started again."""
        project = self.require_project()
        with self._job_lock:
            self.require_project()
            if not self._holds_queue(project):
                runner = runqueue.queue_runner(project)
                if runner is not None:
                    raise Conflict(
                        f"The queue is being run by {runqueue.runner_text(runner)}, not here. Stop "
                        "it there to pause it."
                    )
            self._release_queue()
        return self.queue_status()

    def clear_queue(self) -> dict[str, Any]:
        """Take every run out of the queue, even a saved queue that can't be read. A run in
        progress goes on; nothing starts after it."""
        project = self.require_project()
        with self._job_lock:
            self.require_project()
            runqueue.clear_queue(project)
            self._release_queue()
        return self.queue_status()

    @staticmethod
    def _queued_draft(project: Project, draft_id: str) -> bool:
        """Whether the saved draft ``draft_id`` waits in the queue to run as it was saved."""
        try:
            return any(item.draft_id == draft_id for item in runqueue.read_queue(project))
        except (QueueError, OSError):
            return False

    def _holds_queue(self, project: Project) -> bool:
        """Whether this server runs the project's queue (and no one has taken it over)."""
        hold = self._queue_hold
        return hold is not None and runqueue.holds_queue(project, hold)

    def _take_queue(self, project: Project) -> None:
        try:
            self._queue_hold = runqueue.take_queue(project, "app")
        except runqueue.QueueBusy as exc:
            raise Conflict(str(exc)) from exc

    def _start_next(self) -> None:
        """Start the queue's next run, now that no job is running, if this server runs the queue.
        A run that can't start (its dataset was deleted, say) leaves the queue, the tabs are told
        why, and the next one is tried. Once the queue is empty, this server lets go of it.
        Called by every job's thread once the job has ended, so never two jobs at once."""
        with self._job_lock:
            while self._queue_hold is not None and (
                self.job is None or self.job.status != "running"
            ):
                project = self.project
                if project is None or not self._holds_queue(project):
                    self._queue_hold = self._queue_job = None  # its project closed, or it was taken
                    return
                try:
                    # Under the queue's lock: no other process changes the queue between reading
                    # its first run, starting it and taking it out.
                    with runqueue.locked(project):
                        items = runqueue.read_queue(project)
                        if not items:
                            self._release_queue()
                            return
                        item = items[0]
                        failure: Exception | None = None
                        try:
                            self._queue_job = self.run_spec_job(
                                item.spec, item.derived_from, item.draft_id
                            ).id
                        except Exception as exc:  # noqa: BLE001 - reported, and the next is tried
                            failure = exc
                        # Only now, once the job has taken its place: the queue and the job never
                        # both look idle while a run starts.
                        runqueue.remove_from_queue(project, item.id)
                except (QueueError, OSError) as exc:
                    # Unreadable (the tabs hear why from the queue's error), or a run that would
                    # start again and again: stop running the queue instead.
                    log.warning("The queue stops: %s", _reason(exc))
                    self._release_queue()
                    return
                if failure is not None:
                    reason = _reason(failure)
                    log.warning("A queued run couldn't start: %s", reason)
                    skipped = {
                        "project_session": project.session_id,
                        "id": item.id,
                        "title": item.title,
                        "error": reason,
                    }
                    self.hub.publish("queue.skipped", validate_event("queue.skipped", skipped))
                self._publish_queue(project.session_id)

    def _publish_queue(self, session: str | None = None) -> None:
        """Tell the tabs the open project's whole queue and how it stands. Called under
        _job_lock, so they hear the changes in the order they were made."""
        self.hub.publish("queue", validate_event("queue", self.queue_status()))

    def _release_queue(self, publish: bool = True) -> None:
        """Stop running the open project's queue (under _job_lock): it stays in the project,
        paused, until it is started again. When it is empty or paused, and when its project
        closes or the server stops."""
        hold, self._queue_hold, self._queue_job = self._queue_hold, None, None
        project = self.project
        if hold is not None and project is not None:
            try:
                runqueue.release_queue(project, hold)
            except (QueueError, OSError, ValueError) as exc:  # its record goes when we do
                log.warning("Couldn't let go of the queue: %s", _reason(exc))
        if publish and project is not None:
            self._publish_queue(project.session_id)

    # -- managing runs -----------------------------------------------------------------------

    def annotate_run(self, run_id: str, changes: dict[str, Any]) -> RunListing:
        """Rename, star or archive a run: its annotations.json, never its spec."""
        from logogram.history import annotate_run

        project = self.require_project()
        with self._job_lock:  # never while the run's folder is being deleted
            self.require_project()
            listing = annotate_run(project, run_id, **changes)
        self._runs_changed(project, run_id)
        return listing

    def delete_run(self, run_id: str) -> None:
        """Delete a finished, failed or cancelled run, or a draft. Not one that is running or
        queued (logogram.history refuses a queued draft): under the job lock, so none can start
        in its folder meanwhile."""
        from logogram.history import RunInUse, delete_run, display_name

        project = self.require_project()
        with self._job_lock:
            self.require_project()
            listing = project.run_listing(run_id)
            name = display_name(listing) if listing else run_id
            job = self.job
            if job is not None and job.status == "running" and job.run_id == run_id:
                raise RunInUse(
                    f"{name} is running. Wait for it to end, or cancel it, then delete it."
                )
            delete_run(project, run_id)
        self._runs_changed(project, run_id, deleted=True)

    def _runs_changed(self, project: Project, run_id: str, deleted: bool = False) -> None:
        payload = {"project_session": project.session_id, "run_id": run_id, "deleted": deleted}
        self.hub.publish("runs.changed", validate_event("runs.changed", payload))


def mark_interrupted_runs(project: Project) -> None:
    """Record as failed the runs whose manifest says they are running but whose process is gone,
    so they read as interrupted, never as drafts or as still going. A run still going in another
    process (``logogram run``, a notebook, another server) is left alone. A run with no owner
    record, from before Logogram kept one, is taken to be interrupted, as it always was."""
    from logogram.owner import OWNER_STOPPED, SERVER_STOPPED, owner_state, read_owner, remove_owner
    from logogram.runner import write_json

    for listing in project.list_runs():
        if listing.status not in ("running", "failed") or listing.elsewhere:
            continue
        folder = project.run_dir(listing.id)
        path = folder / "manifest.json"
        try:
            data = project.read_json(path, optional=True)
        except (ValueError, OSError):  # unreadable: its listing already says so
            continue
        if not data or data.get("status") != "running":
            continue
        state = owner_state(read_owner(project, folder))
        if state in ("here", "elsewhere"):
            continue  # still going
        if state == "unknown":
            project.interrupted_runs.add(listing.id)
        data.update(status="failed", error=SERVER_STOPPED if state == "unknown" else OWNER_STOPPED)
        log.info("The run %s was interrupted: %s", listing.id, data["error"])
        with contextlib.suppress(OSError):  # read-only projects remain inspectable
            write_json(project.writable(path), data)
            remove_owner(project, folder)


def _reason(exc: Exception) -> str:
    """Why a queued run couldn't start, in the words the API's error responses use."""
    if isinstance(exc, OSError):
        name = Path(exc.filename).name if isinstance(exc.filename, str) else None
        what = exc.strerror or type(exc).__name__
        return f"{what}: {name}" if name else what
    return str(exc) or type(exc).__name__
