"""Small file helpers shared by everything that writes into project folders."""

from __future__ import annotations

import contextlib
import os
import secrets
import time
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO


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
