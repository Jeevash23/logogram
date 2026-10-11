"""Read finished runs: the per-prompt evidence behind one site's number."""

from __future__ import annotations

from typing import Any

import numpy as np

from logogram.analysis import answer_text
from logogram.datasets import DatasetError, file_sha256, load_dataset
from logogram.project import Project
from logogram.results import read_site_rows
from logogram.spec import Spec


class RunNotReady(ValueError):
    pass


def load_summary(project: Project, run_id: str) -> dict[str, Any]:
    path = project.run_dir(run_id) / "summary.json"
    if not path.is_file():
        raise RunNotReady("This run has no results yet.")
    return project.read_json(path) or {}


def site_detail(
    project: Project, run_id: str, site_index: int, extremes: int = 3
) -> dict[str, Any]:
    folder = project.run_dir(run_id)
    summary = load_summary(project, run_id)
    sites = summary["sites"]
    if not 0 <= site_index < len(sites):
        raise RunNotReady(f"Site {site_index} isn't part of this run.")
    site = sites[site_index]
    spec = Spec.from_path(project.readable(folder / "spec.json"))
    rows = read_site_rows(project.readable(folder / "results.parquet"), site_index)

    records = None
    dataset_changed = False
    try:
        path = project.resolve_dataset(spec.dataset.path)
        records = load_dataset(path)
        if spec.dataset.limit:
            records = records[: spec.dataset.limit]
        if spec.dataset.sha256 and file_sha256(path) != spec.dataset.sha256:
            dataset_changed = True
    except (DatasetError, ValueError):
        records = None

    effects = np.asarray(rows["effect"], dtype=np.float64)
    prompts = []
    for i, prompt_index in enumerate(rows["prompt"]):
        record = records[prompt_index] if records and prompt_index < len(records) else None
        receiver = rows["receiver_logit_diff"][i]
        patched = rows["patched_logit_diff"][i]
        prompts.append(
            {
                "index": prompt_index,
                "clean": record.clean if record else None,
                "corrupt": record.corrupt if record else None,
                "answer": answer_text(record.answer) if record else None,
                "distractor": answer_text(record.distractor) if record else None,
                "effect": _f(rows["effect"][i]),
                "delta": _f(rows["delta"][i]),
                "patched_metric": _f(rows["patched_metric"][i]),
                "receiver_metric": _f(rows["receiver_metric"][i]),
                "reference_metric": _f(rows["reference_metric"][i]),
                "patched_logit_diff": _f(patched),
                "receiver_logit_diff": _f(receiver),
                "reference_logit_diff": _f(rows["reference_logit_diff"][i]),
                "patched_answer_prob": _f(rows["patched_answer_prob"][i]),
                "receiver_answer_prob": _f(rows["receiver_answer_prob"][i]),
                "flipped": _flipped(receiver, patched),
            }
        )
    mean = site["effect"]["mean"] or 0.0
    direction = 1.0 if mean >= 0 else -1.0
    finite = np.where(np.isfinite(effects), effects, 0.0)
    order = np.argsort(-direction * finite, kind="mergesort")
    k = min(extremes, len(order))
    return {
        "run_id": run_id,
        "site": site,
        "receiver": summary["receiver"],
        "reference": summary["reference"],
        "metric": summary["metric"],
        "statistics": summary["statistics"],
        "prompts": prompts,
        "strongest": [int(rows["prompt"][j]) for j in order[:k]],
        "weakest": [int(rows["prompt"][j]) for j in order[::-1][:k]],
        "dataset_changed": dataset_changed,
        "dataset_available": records is not None,
    }


def _flipped(receiver: Any, patched: Any) -> bool:
    """Whether the preference changed sign (never, where it wasn't measured)."""
    r, p = _f(receiver), _f(patched)
    if r is None or p is None:
        return False
    return (r > 0 > p) or (r < 0 < p)


def _f(x: Any) -> float | None:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None
