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

from logogram.backends.base import Cancelled, ModelBackend
from logogram.project import Project, config_dir, remember_recent
from logogram.schema import RunListing, validate_event
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
        # Runs waiting for the job in progress to end, first in line first (see queue_run). Only
        # changed under _job_lock. The queue lives in memory, so it doesn't survive a restart of
        # the server, and it belongs to the open project: switching or closing it empties it.
        self._queue: list[QueuedRun] = []
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
        with self._job_lock:
            self._check_switch()
            mark_interrupted_runs(project)
            self._clear_queue()
            self.project = project
        remember_recent(project)
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
            self._clear_queue()
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
                if any(item.draft_id == draft_id for item in self._queue):
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

    def queued_runs(self) -> list[dict[str, Any]]:
        """The runs waiting in the queue, first in line first."""
        return [item.to_dict() for item in list(self._queue)]

    def queue_run(
        self,
        spec: Spec,
        derived_from: dict[str, Any] | None = None,
        draft_id: str | None = None,
    ) -> Job | QueuedRun:
        """Run ``spec`` after the jobs ahead of it: at once when nothing is running or waiting
        (the job is returned), otherwise last in the queue (the queued run is returned).

        When its turn comes it starts like any run, through ``run_spec_job``, which checks it
        again. What can be checked now is checked now, so a run that can't start isn't queued.
        """
        project = self.require_project()
        with self._job_lock:
            self.require_project()  # check again under the switch/job lock
            job = self.job
            running = job is not None and job.status == "running"
            if not running and not self._queue:
                return self.run_spec_job(spec, derived_from, draft_id)
            self._check_run(project, spec, draft_id)
            if draft_id is not None:
                if running and job is not None and job.run_id == draft_id:
                    raise Conflict(f"{draft_id} is running now. Queue a new run instead.")
                if any(item.draft_id == draft_id for item in self._queue):
                    raise Conflict(f"{draft_id} is already queued. It runs when its turn comes.")
            item = QueuedRun(spec, project.session_id, draft_id, derived_from)
            self._queue.append(item)
            self._publish_queue(project.session_id)
        return item

    def remove_queued_run(self, queued_id: str) -> list[dict[str, Any]]:
        """Take a run out of the queue before it starts. Returns the runs still waiting."""
        project = self.require_project()
        with self._job_lock:
            self.require_project()
            item = next((q for q in self._queue if q.id == queued_id), None)
            if item is None:
                raise Missing("That run isn't queued any more: it has started or was removed.")
            self._queue.remove(item)
            self._publish_queue(project.session_id)
            return self.queued_runs()

    def _start_next(self) -> None:
        """Start the queue's next run, now that no job is running. A run that can't start (its
        dataset was deleted, say) leaves the queue, the tabs are told why, and the next one is
        tried. Called by every job's thread once the job has ended, so never two jobs at once."""
        with self._job_lock:
            while self._queue and (self.job is None or self.job.status != "running"):
                item = self._queue[0]
                try:
                    if self.project is None or self.project.session_id != item.project_session:
                        raise Missing("Its project isn't open any more.")  # cleared on a switch
                    self.run_spec_job(item.spec, item.derived_from, item.draft_id)
                except Exception as exc:  # noqa: BLE001 - reported, and the next run is tried
                    reason = _reason(exc)
                    log.warning("A queued run couldn't start: %s", reason)
                    skipped = {
                        "project_session": item.project_session,
                        "id": item.id,
                        "title": item.title,
                        "error": reason,
                    }
                    self.hub.publish("queue.skipped", validate_event("queue.skipped", skipped))
                finally:
                    # Only now, once the job has taken its place: the queue and the job never
                    # both look idle while a run starts.
                    self._queue.remove(item)
                    self._publish_queue(item.project_session)

    def _publish_queue(self, session: str | None) -> None:
        """Tell the tabs the whole queue. Called under _job_lock, so they hear the changes in
        the order they were made."""
        payload = {"project_session": session, "runs": self.queued_runs()}
        self.hub.publish("queue", validate_event("queue", payload))

    def _clear_queue(self) -> None:
        """The queue belongs to the open project: empty it when another project opens or this
        one closes (under _job_lock, once the switch is allowed)."""
        if self._queue:
            session = self._queue[0].project_session
            self._queue.clear()
            self._publish_queue(session)

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
        queued: under the job lock, so none can start in its folder meanwhile."""
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
            if any(item.draft_id == run_id for item in self._queue):
                raise RunInUse(
                    f"{name} is waiting in the queue to run. Take it out of the queue first, then "
                    "delete it."
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
