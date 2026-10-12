"""Small file helpers shared by everything that reads or writes project folders."""

from __future__ import annotations

import contextlib
import errno
import os
import secrets
import stat
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

# The largest files Logogram reads from a project. Project folders come from other people, and a
# file far larger than anything Logogram writes would only exhaust memory.
JSON_LIMIT = 256 * 1024**2  # summaries, manifests, research notes, specs
DATASET_LIMIT = 256 * 1024**2  # a JSONL file of prompts (about a million pairs)


class FileTooLarge(ValueError):
    """A file is larger than Logogram reads."""


def read_limited(path: Path, limit: int, fix: str) -> bytes:
    """The bytes of the regular file at ``path``, refusing one larger than ``limit`` bytes.

    The file is opened without blocking and checked to be a regular file, so a pipe or device
    planted in a project can't hang the reader. ``fix`` says what to do about a file that is too
    large.
    """
    fd = _open_for_reading(path)
    with os.fdopen(fd, "rb") as fh:
        info = os.fstat(fh.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{path.name} isn't a regular file.")
        data = fh.read(limit + 1) if info.st_size <= limit else b""
    if info.st_size > limit or len(data) > limit:
        raise FileTooLarge(
            f"{path.name} is larger than the {size_text(limit)} Logogram reads. {fix}"
        )
    return data


def _open_for_reading(path: Path) -> int:
    """Open ``path`` to read it, without blocking. On Windows a file can't be opened for a moment
    while another thread replaces it (a run rewrites its spec and manifest as it goes, while the
    history reads them), so try again briefly, as ``replace_file`` does from the writer's side."""
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    for attempt in range(25):
        try:
            return os.open(path, flags)
        except PermissionError:
            if sys.platform != "win32" or attempt == 24:
                raise
            time.sleep(0.02)
    raise AssertionError("unreachable")


def size_text(n: int) -> str:
    """A size limit in words, such as "256 MB"."""
    if n >= 1024**2:
        return f"{n // 1024**2} MB"
    return f"{n // 1024} KB" if n >= 1024 else f"{n} bytes"


def replace_file(tmp: Path, path: Path) -> None:
    """Atomically move ``tmp`` over ``path``.

    On Windows the move fails while another thread has ``path`` open for reading (the history
    panel reads specs while runs write them), so retry briefly.
    """
    for attempt in range(50):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(0.02)


def remove_file(path: Path) -> None:
    """Remove the file at ``path``, if there is one (a link is removed itself, never followed).

    On Windows a file can't be removed while another thread or process has it open (a tab reads
    the queue while the runner empties it), so retry briefly, as ``replace_file`` does.
    """
    for attempt in range(50):
        try:
            os.unlink(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if sys.platform != "win32" or attempt == 49:
                raise
            time.sleep(0.02)


def _create_exclusive(path: Path) -> tuple[int, Path]:
    # O_EXCL never opens an existing file or follows a symlink, so a link planted next to the
    # target can't redirect the write. The mode is filtered by the umask, as for any new file.
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for _ in range(100):
        tmp = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
        try:
            return os.open(tmp, flags, 0o666), tmp
        except FileExistsError:
            continue
    raise FileExistsError(f"Can't create a temporary file next to {path.name}.")


def place_new(tmp: Path, path: Path) -> None:
    """Move the finished file ``tmp`` to ``path``, where nothing may be yet, not even a link.

    A hard link fails rather than replace what is at ``path``, so a file created there in the
    meantime is never overwritten. A file system without hard links is checked first instead.
    """
    try:
        os.link(tmp, path)
    except FileExistsError:
        raise
    except OSError:
        if os.path.lexists(path):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(path)) from None
        replace_file(tmp, path)
        return
    with contextlib.suppress(OSError):  # a stray temporary file is hidden and harmless
        tmp.unlink()


@contextlib.contextmanager
def atomic_output(path: Path, *, replace: bool = True) -> Iterator[BinaryIO]:
    """Write ``path`` through a fresh temporary file, then move it into place.

    Readers never see a half-written file, and the final move replaces ``path`` itself (a symlink
    at ``path`` is replaced, not followed). With ``replace`` false, anything already at ``path``
    stays as it is and the write fails with ``FileExistsError``.
    """
    fd, tmp = _create_exclusive(path)
    try:
        with os.fdopen(fd, "wb") as fh:
            yield fh
        if replace:
            replace_file(tmp, path)
        else:
            place_new(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def write_text_atomic(path: Path, text: str, *, replace: bool = True) -> None:
    with atomic_output(path, replace=replace) as fh:
        fh.write(text.encode("utf-8"))
