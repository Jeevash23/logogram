"""Which process runs a run, so that a run still going elsewhere isn't taken for an interrupted one.

While a run runs, its folder holds ``owner.json``: the process id, the process's start time as
psutil reports it, and a random token (see :class:`logogram.schema.RunOwner`). Nothing in it names
the machine or the person. The runner writes it before the manifest says "running" and removes it
once the manifest says how the run ended, so a manifest that still says "running" belongs to a run
that is either still going or was interrupted: the process that ran it was killed, crashed, or the
computer stopped. Its owner record tells the two apart.

A running manifest with no owner record was written before Logogram kept one (0.3.0 and earlier,
which also ran runs only in the server's own process): opening the project marks it interrupted,
as those versions did.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import secrets
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import psutil

from logogram.fileio import read_limited, write_text_atomic
from logogram.schema import RunOwner

if TYPE_CHECKING:
    from logogram.project import Project

OWNER_FILE = "owner.json"
# Far larger than any record Logogram writes (about 120 bytes).
OWNER_LIMIT = 4096
# How far two readings of one process's start time may differ. On Linux psutil adds the boot time
# to it, which moves when the clock is set. A process that reuses the pid can't start within this
# of the run's own process: that process was still alive when it wrote the record, seconds after
# it started (PyTorch alone takes longer to import).
START_TOLERANCE_S = 1.0

# Why a run that says it is running isn't any more.
SERVER_STOPPED = "The server stopped before this run finished. Rerun its saved spec."
OWNER_STOPPED = "The process running this run stopped before it finished. Rerun its saved spec."

OwnerState = Literal["here", "elsewhere", "gone", "unknown"]


@functools.cache
def _started(pid: int) -> float:
    return psutil.Process(pid).create_time()


def this_process() -> tuple[int, float]:
    """This process's id and start time (looked up again after a fork, which changes the id)."""
    pid = os.getpid()
    return pid, _started(pid)


def claim_run(project: Project, folder: Path) -> RunOwner:
    """Record that this process runs the run in ``folder``. Call it before the run's manifest
    says "running"."""
    pid, started = this_process()
    owner = RunOwner(pid=pid, started=started, token=secrets.token_hex(16))
    text = json.dumps(owner.model_dump(mode="json"), indent=2) + "\n"
    write_text_atomic(project.writable(folder / OWNER_FILE), text)
    return owner


def release_run(project: Project, folder: Path, owner: RunOwner) -> None:
    """Remove ``owner``'s record from ``folder``, once the manifest says how the run ended. A
    record another process has written since stays."""
    current = read_owner(project, folder)
    if current is not None and current.token != owner.token:
        return
    remove_owner(folder)


def remove_owner(folder: Path) -> None:
    """Remove the owner record in ``folder``, if there is one (a link is removed, never followed)."""
    with contextlib.suppress(OSError):  # gone already, or a read-only project
        (folder / OWNER_FILE).unlink()


def read_owner(project: Project, folder: Path) -> RunOwner | None:
    """The owner record in ``folder``, read like any project file: a regular file inside the
    project, small, and valid. None when there is none, or it can't be trusted."""
    path = folder / OWNER_FILE
    if not os.path.lexists(path):
        return None
    try:
        raw = read_limited(
            project.readable(path), OWNER_LIMIT, "Logogram never writes one that large."
        )
        return RunOwner.model_validate_json(raw)
    except (ValueError, OSError):  # unreadable, too large, not JSON or not an owner record
        return None


def owner_state(owner: RunOwner | None) -> OwnerState:
    """Where the run that ``owner`` recorded runs: in this process ("here"), in another process
    that is still going ("elsewhere"), or nowhere any more ("gone"); "unknown" with no record.

    A process is the run's when its id exists and its start time matches the record's, so a
    later process that reuses the id doesn't count. One that can't be inspected (another user's,
    on some systems) might be the run's, so it counts as going.
    """
    if owner is None:
        return "unknown"
    pid, started = this_process()
    if owner.pid == pid and abs(owner.started - started) <= START_TOLERANCE_S:
        return "here"
    try:
        process = psutil.Process(owner.pid)
        if abs(process.create_time() - owner.started) > START_TOLERANCE_S:
            return "gone"  # a later process has the id now
        # A zombie has exited; only its parent hasn't collected it yet.
        return "gone" if process.status() == psutil.STATUS_ZOMBIE else "elsewhere"
    except psutil.NoSuchProcess:
        return "gone"
    except psutil.AccessDenied:
        return "elsewhere"
    except (ValueError, OverflowError, OSError):  # an id this system can't have
        return "gone"
