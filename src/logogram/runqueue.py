"""A project's run queue, kept in the project so it outlives the server and closing the project.

The queue is ``.logogram/queue.json``: the queued specs, first in line first, each with when it
was added, and for a saved draft the folder it runs in, for a rerun or a check the run it derives
from (see :class:`logogram.schema.QueueFile`). The app and ``logogram queue add`` add to it; the
app, ``logogram queue clear`` and whatever runs the queue take from it. Every change reads the
file, changes it and writes it back (atomically, through logogram.fileio) while holding
``.logogram/queue.lock``, an operating-system lock that is released if its process dies, so
processes never lose each other's changes. A queued run leaves the queue when it starts.

One process at a time runs the queue: the app (from when runs are queued behind its job, or it is
started, until it is empty or paused) or ``logogram queue run``. The one running it says so in
``.logogram/queue-runner.json``, an owner record like a run's (see logogram.owner): which program
it is, its process id and start time, and a random token. A record whose process has ended holds
nothing, so a crash never leaves the queue taken. Nothing in these files names the machine or
the person.

``.logogram/`` is in the project's .gitignore: a queue is this machine's work in progress.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import secrets
import stat
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ValidationError

from logogram.fileio import read_limited, write_text_atomic
from logogram.owner import owner_state, this_process
from logogram.schema import QueueEntry, QueueFile, QueueRunner, RunOwner
from logogram.spec import Spec

if TYPE_CHECKING:
    from logogram.project import Project

QUEUE_FOLDER = ".logogram"
QUEUE_FILE = "queue.json"
LOCK_FILE = "queue.lock"
RUNNER_FILE = "queue-runner.json"
# Far larger than a queue Logogram writes (a spec is a few kilobytes).
QUEUE_LIMIT = 64 * 1024**2
RUNNER_LIMIT = 4096
# How long a change waits for another process's change to the queue (they take milliseconds).
LOCK_WAIT_S = 10.0

Program = Literal["app", "cli"]


class QueueError(ValueError):
    """The queue can't be read or changed; the message says why and what to do."""


class QueueBusy(QueueError):
    """Another process runs the queue."""


@dataclasses.dataclass(frozen=True)
class Queued:
    """A run waiting in the queue."""

    id: str
    spec: Spec
    # When it was added, in seconds since the epoch.
    queued_at: float
    draft_id: str | None = None
    derived_from: dict[str, Any] | None = None

    @property
    def title(self) -> str:
        return self.spec.name


# -- the lock ----------------------------------------------------------------------------------

_held = threading.local()


def _folder(project: Project) -> Path:
    """``.logogram/``, created if missing, and checked to be the project's own folder."""
    folder = project.root / QUEUE_FOLDER
    try:
        folder.mkdir(exist_ok=True)
    except OSError as exc:
        raise QueueError(
            f"Logogram can't create {QUEUE_FOLDER}/ in the project to keep its queue: "
            f"{exc.strerror or exc}. Make the project folder writable."
        ) from exc
    project.writable(folder / LOCK_FILE)  # a link to a folder outside the project is refused
    return folder


def _open_lock(path: Path) -> int:
    """The lock file, opened (and created) without following a link at it."""
    if os.path.islink(path):
        raise QueueError(
            f"{QUEUE_FOLDER}/{LOCK_FILE} is a link. Delete it; Logogram makes the file it needs."
        )
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags, 0o666)
    except OSError as exc:
        raise QueueError(
            f"Logogram can't open {QUEUE_FOLDER}/{LOCK_FILE}: {exc.strerror or exc}. Make the "
            "project folder writable."
        ) from exc
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise QueueError(f"{QUEUE_FOLDER}/{LOCK_FILE} isn't a regular file. Delete it.")
    return fd


def _try_lock(fd: int) -> bool:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:  # held by another open file: another process, or another thread here
        return False
    return True


def _unlock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


@contextlib.contextmanager
def locked(project: Project) -> Iterator[Path]:
    """Hold the project's queue lock (``.logogram/queue.lock``) while the queue's files are read
    and changed: in any process, and in any thread of this one. It may be taken again by the
    thread that holds it. Yields ``.logogram/``."""
    folder = _folder(project)
    key = str((folder / LOCK_FILE).resolve())
    held: dict[str, list[int]] = getattr(_held, "locks", None) or {}
    _held.locks = held
    if key in held:  # this thread holds it already
        held[key][1] += 1
        try:
            yield folder
        finally:
            held[key][1] -= 1
        return
    fd = _open_lock(folder / LOCK_FILE)
    try:
        deadline = time.monotonic() + LOCK_WAIT_S
        while not _try_lock(fd):
            if time.monotonic() > deadline:
                raise QueueError(
                    "Another process has been changing the queue for a while. Try again in a "
                    "moment."
                )
            time.sleep(0.01)
        held[key] = [fd, 1]
        try:
            yield folder
        finally:
            del held[key]
            _unlock(fd)
    finally:
        os.close(fd)


# -- the queued runs ---------------------------------------------------------------------------


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _reason(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "the file"
        return f"{where}: {first['msg']}"
    if isinstance(exc, OSError):
        return exc.strerror or type(exc).__name__
    return str(exc) or type(exc).__name__


def _spec(data: dict[str, Any], upgraded: list[str]) -> Spec:
    """A queued spec as it was queued, saying what a version 1 spec left out (the queue keeps
    it in version 3's form, which states every value)."""
    spec = Spec.model_validate(data)
    spec._upgraded = list(upgraded)
    return spec


def _read(project: Project) -> list[Queued]:
    path = project.root / QUEUE_FOLDER / QUEUE_FILE
    if not os.path.lexists(path):
        return []
    try:
        raw = read_limited(
            project.readable(path), QUEUE_LIMIT, "Logogram never writes one that large."
        )
        data = QueueFile.model_validate_json(raw)
        return [
            Queued(
                id=entry.id,
                spec=_spec(entry.spec, entry.upgraded),
                queued_at=datetime.fromisoformat(entry.queued_at).timestamp(),
                draft_id=entry.draft_id,
                derived_from=(
                    entry.derived_from.model_dump(mode="json") if entry.derived_from else None
                ),
            )
            for entry in data.runs
        ]
    except (ValueError, OSError) as exc:
        raise QueueError(
            f"The project's saved queue ({QUEUE_FOLDER}/{QUEUE_FILE}) can't be read "
            f"({_reason(exc)}). Clear the queue to start a new one."
        ) from exc


def _write(project: Project, folder: Path, runs: list[Queued]) -> None:
    path = project.writable(folder / QUEUE_FILE)
    if not runs:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()  # a link is removed, never followed
        return
    data = QueueFile(
        runs=[
            QueueEntry(
                id=q.id,
                queued_at=_iso(q.queued_at),
                spec=q.spec.model_dump(mode="json"),
                draft_id=q.draft_id,
                derived_from=q.derived_from,  # type: ignore[arg-type]
                upgraded=q.spec.upgraded_fields,
            )
            for q in runs
        ]
    )
    write_text_atomic(path, json.dumps(data.model_dump(mode="json"), indent=2) + "\n")


def read_queue(project: Project) -> list[Queued]:
    """The project's queued runs, first in line first. Raises QueueError when the saved queue
    can't be read."""
    return _read(project)


def add_to_queue(
    project: Project,
    spec: Spec,
    *,
    draft_id: str | None = None,
    derived_from: dict[str, Any] | None = None,
) -> Queued:
    """Add a run at the end of the queue. A saved queue that can't be read is left as it is: it
    is refused, rather than overwritten."""
    with locked(project) as folder:
        runs = _read(project)
        if draft_id is not None and any(q.draft_id == draft_id for q in runs):
            raise QueueError(f"{draft_id} is already queued. It runs when its turn comes.")
        item = Queued(
            id=uuid.uuid4().hex[:12],
            spec=spec,
            queued_at=float(int(time.time())),
            draft_id=draft_id,
            derived_from=derived_from,
        )
        _write(project, folder, [*runs, item])
        return item


def remove_from_queue(project: Project, queued_id: str) -> Queued | None:
    """Take a run out of the queue (it starts, or someone removed it). None when it isn't
    queued any more."""
    with locked(project) as folder:
        runs = _read(project)
        item = next((q for q in runs if q.id == queued_id), None)
        if item is not None:
            _write(project, folder, [q for q in runs if q.id != queued_id])
        return item


def put_back(project: Project, item: Queued) -> None:
    """Put a run taken out of the queue to start back at its front, as it was: it didn't start
    (Ctrl+C came first)."""
    with locked(project) as folder:
        runs = _read(project)
        if all(q.id != item.id for q in runs):
            _write(project, folder, [item, *runs])


def clear_queue(project: Project) -> int:
    """Empty the queue, even one whose saved file can't be read. Returns how many runs it held
    (0 for an unreadable one)."""
    with locked(project) as folder:
        try:
            count = len(_read(project))
        except QueueError:
            count = 0
        _write(project, folder, [])
        return count


# -- who runs it -------------------------------------------------------------------------------


def _read_runner(project: Project) -> QueueRunner | None:
    path = project.root / QUEUE_FOLDER / RUNNER_FILE
    if not os.path.lexists(path):
        return None
    try:
        raw = read_limited(project.readable(path), RUNNER_LIMIT, "Logogram never writes one.")
        return QueueRunner.model_validate_json(raw)
    except (ValueError, OSError):  # unreadable or not a runner record: it holds nothing
        return None


def _alive(record: QueueRunner) -> bool:
    owner = RunOwner(pid=record.pid, started=record.started, token=record.token)
    return owner_state(owner) in ("here", "elsewhere")


def queue_runner(project: Project) -> QueueRunner | None:
    """Who runs the project's queue now; None when nobody does (no record, or its process has
    ended)."""
    record = _read_runner(project)
    return record if record is not None and _alive(record) else None


def runner_text(record: QueueRunner) -> str:
    """Who runs the queue, in words: "the Logogram app (process 1234)"."""
    who = "the Logogram app" if record.program == "app" else "`logogram queue run`"
    return f"{who} (process {record.pid})"


def take_queue(project: Project, program: Program) -> QueueRunner:
    """Record that this process runs the queue. Raises QueueBusy, saying who, when another one
    does (in another process, or elsewhere in this one)."""
    with locked(project) as folder:
        current = queue_runner(project)
        if current is not None:
            stop = (
                "Pause it in the app, or wait for it to finish."
                if current.program == "app"
                else "Wait for it to finish, or stop it there with Ctrl+C."
            )
            raise QueueBusy(f"The queue is already being run by {runner_text(current)}. {stop}")
        pid, started = this_process()
        record = QueueRunner(program=program, pid=pid, started=started, token=secrets.token_hex(16))
        text = json.dumps(record.model_dump(mode="json"), indent=2) + "\n"
        write_text_atomic(project.writable(folder / RUNNER_FILE), text)
        return record


def release_queue(project: Project, record: QueueRunner) -> None:
    """Stop running the queue: remove ``record``, unless another runner has taken its place."""
    with locked(project) as folder:
        current = _read_runner(project)
        if current is None or current.token != record.token:
            return
        with contextlib.suppress(FileNotFoundError):
            project.writable(folder / RUNNER_FILE).unlink()


def holds_queue(project: Project, record: QueueRunner) -> bool:
    """Whether ``record`` still runs the queue (nobody has taken it over)."""
    current = _read_runner(project)
    return current is not None and current.token == record.token
