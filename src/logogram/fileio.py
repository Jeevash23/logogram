"""Small file helpers shared by everything that reads or writes project folders."""

from __future__ import annotations

import contextlib
import os
import secrets
import stat
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
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
    with os.fdopen(fd, "rb") as fh:
        info = os.fstat(fh.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{path.name} isn't a regular file.")
        data = fh.read(limit + 1) if info.st_size <= limit else b""
    if info.st_size > limit or len(data) > limit:
        raise FileTooLarge(f"{path.name} is larger than the {_size(limit)} Logogram reads. {fix}")
    return data


def _size(n: int) -> str:
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


@contextlib.contextmanager
def atomic_output(path: Path) -> Iterator[BinaryIO]:
    """Write ``path`` through a fresh temporary file, then move it into place.

    Readers never see a half-written file, and the final move replaces ``path`` itself (a symlink
    at ``path`` is replaced, not followed).
    """
    fd, tmp = _create_exclusive(path)
    try:
        with os.fdopen(fd, "wb") as fh:
            yield fh
        replace_file(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def write_text_atomic(path: Path, text: str) -> None:
    with atomic_output(path) as fh:
        fh.write(text.encode("utf-8"))
