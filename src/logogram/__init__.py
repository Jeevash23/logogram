"""Logogram: a local-first workbench for causal experiments inside language models.

The Python API (``logogram.run``, ``logogram.load_run``, ...) lives in :mod:`logogram.api` and is
imported on first use, so the command line starts without loading PyTorch. (Its names don't
clash with Logogram's modules: ``load_spec``, not ``spec``.)
"""

from typing import Any

__version__ = "0.1.2"
# The release date of this version: after a few months the app suggests looking for a newer one
# (no network needed). Set it with every release.
__released__ = "2026-10-08"

_API = (
    "Run",
    "check_robustness",
    "compare_runs",
    "create_project",
    "generate",
    "list_runs",
    "load_model",
    "load_run",
    "load_spec",
    "open_project",
    "run",
    "verify_run",
    "write_dataset",
)


def __getattr__(name: str) -> Any:
    if name in _API:
        from logogram import api

        return getattr(api, name)
    raise AttributeError(f"module 'logogram' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *_API])
