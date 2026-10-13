"""Logogram: a local-first workbench for causal experiments inside language models.

The Python API (``logogram.run``, ``logogram.load_run``, ...) lives in :mod:`logogram.api` and is
imported on first use, so the command line starts without loading PyTorch. (Its names don't
clash with Logogram's modules: ``load_spec``, not ``spec``.)
"""

from typing import Any

__version__ = "0.6.0"
# The release date of this version: after a few months the app suggests looking for a newer one
# (no network needed). Set it with every release.
__released__ = "2026-10-12"

_API = (
    "Run",
    "Sweep",
    "activations",
    "add_local_model",
    "check_robustness",
    "compare_runs",
    "continue_text",
    "create_project",
    "delete_run",
    "edge_verification",
    "export_activations",
    "feature_writes",
    "generate",
    "head_kinds",
    "import_corpus",
    "list_corpora",
    "list_directions",
    "list_local_models",
    "list_runs",
    "list_saes",
    "load_model",
    "load_run",
    "load_sae",
    "load_spec",
    "load_sweep",
    "look",
    "mean_difference",
    "open_example",
    "open_project",
    "principal_components",
    "read_corpus",
    "read_direction",
    "remove_local_model",
    "run",
    "save_direction",
    "sweep",
    "top_examples",
    "toy_model",
    "verify_run",
    "write_corpus",
    "write_dataset",
)


def __getattr__(name: str) -> Any:
    if name in _API:
        from logogram import api

        return getattr(api, name)
    raise AttributeError(f"module 'logogram' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *_API])
