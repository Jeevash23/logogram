"""Turn engine output into the files a run leaves behind: summary.json and results.parquet."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from logogram.engine import EngineResult
from logogram.fileio import atomic_output
from logogram.schema import Summary
from logogram.spec import Spec, describe_experiment
from logogram.stats import SiteStats, compute_site_stats, resample_counts

SUMMARY_VERSION = 1


def _f(x: Any) -> float | None:
    """A JSON-safe float (None for NaN or infinity)."""
    value = float(x)
    return value if math.isfinite(value) else None


def compute_stats(
    spec: Spec,
    result: EngineResult,
    counts: np.ndarray | None = None,
    indices: list[int] | None = None,
) -> SiteStats:
    """Statistics for all sites, or for ``indices`` only (rows then follow ``indices``)."""
    n = result.patched_ld.shape[1]
    if counts is None:
        counts = resample_counts(n, spec.statistics.bootstrap, spec.statistics.seed)
    rows = slice(None) if indices is None else np.asarray(indices, dtype=np.int64)
    return compute_site_stats(
        patched_ld=result.patched_ld[rows],
        patched_prob=result.patched_prob[rows],
        receiver_ld=result.receiver_ld,
        source_ld=result.reference_ld,
        receiver_prob=result.receiver_prob,
        normalization=spec.metric.normalization,
        counts=counts,
        ci=spec.statistics.ci,
        delta=None if result.delta is None else result.delta[rows],
        gap=result.gap,
    )


def site_payload(
    result: EngineResult, stats: SiteStats, indices: list[int]
) -> list[dict[str, Any]]:
    """Payload for sites ``indices``; row ``i`` of ``stats`` belongs to ``indices[i]``."""
    out = []
    for i, site_index in enumerate(indices):
        site = result.sites[site_index]
        out.append(
            {
                **site.to_dict(),
                "n": stats.n,
                "effect": {
                    "mean": _f(stats.effect_mean[i]),
                    "sd": _f(stats.effect_sd[i]),
                    "lo": _f(stats.effect_lo[i]),
                    "hi": _f(stats.effect_hi[i]),
                },
                "delta": {
                    "mean": _f(stats.delta_mean[i]),
                    "sd": _f(stats.delta_sd[i]),
                    "lo": _f(stats.delta_lo[i]),
                    "hi": _f(stats.delta_hi[i]),
                },
                "patched_logit_diff": _f(stats.patched_mean[i]),
                "answer_prob": _f(stats.prob_mean[i]),
                "answer_prob_delta": _f(stats.prob_delta_mean[i]),
                "sign_flips": int(stats.sign_flips[i]),
                "opposite_sign": int(stats.opposite_sign[i]),
            }
        )
    return out


def _group_stats(values: np.ndarray) -> dict[str, float | None]:
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {"mean": _f(values.mean()), "sd": _f(sd)}


def baseline_payload(result: EngineResult) -> dict[str, Any]:
    b = result.baselines
    gap = result.reference_ld - result.receiver_ld
    return {
        "clean": {
            "logit_diff": _group_stats(b.clean_ld),
            "answer_prob": _group_stats(b.clean_prob),
            "prefers_answer": int((b.clean_ld > 0).sum()),
        },
        "corrupt": {
            "logit_diff": _group_stats(b.corrupt_ld),
            "answer_prob": _group_stats(b.corrupt_prob),
            "prefers_answer": int((b.corrupt_ld > 0).sum()),
        },
        "gap": _group_stats(gap),
    }


def build_summary(
    spec: Spec,
    result: EngineResult,
    stats: SiteStats,
    run_id: str,
    model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    b = result.baselines
    normalization = spec.metric.normalization
    description = "logit(answer) − logit(distractor) at the last position"
    if result.measure == "attribution":
        description = (
            "direct contribution to logit(answer) − logit(distractor) at the last position, with "
            "the final normalization's scale held at its value in the run"
        )
        if normalization == "dataset_gap":
            norm_text = (
                f"contribution ÷ mean {result.receiver} logit difference; "
                f"mean = {stats.denominator:.4f}"
            )
        else:
            norm_text = f"contribution ÷ each prompt's own {result.receiver} logit difference"
    elif normalization == "dataset_gap":
        norm_text = (
            f"(patched − {result.receiver}) ÷ mean({result.reference} − {result.receiver}) "
            f"logit difference; mean gap = {stats.denominator:.4f}"
        )
    else:
        norm_text = (
            f"(patched − {result.receiver}) ÷ ({result.reference} − {result.receiver}) "
            "logit difference, per prompt"
        )
    if result.measure == "estimate":
        description = (
            "first-order estimate of the change patching would cause in logit(answer) − "
            "logit(distractor) at the last position: (source − receiver activation) · its "
            "gradient at the receiver run"
        )
        norm_text = f"estimated {norm_text}"
    summary = {
        "logogram_summary": SUMMARY_VERSION,
        "run_id": run_id,
        "name": spec.name,
        "description": describe_experiment(spec),
        "model": model,
        "n_prompts": len(result.prompts),
        "n_sites": len(result.sites),
        "layout": result.layout,
        "receiver": result.receiver,
        "reference": result.reference,
        "measure": result.measure,
        "metric": {
            "kind": spec.metric.kind,
            "normalization": normalization,
            "denominator": _f(stats.denominator) if normalization == "dataset_gap" else None,
            "description": description,
            "normalized_effect": norm_text,
        },
        "statistics": {
            "bootstrap": spec.statistics.bootstrap,
            "ci": spec.statistics.ci,
            "seed": spec.statistics.seed,
            "method": "percentile bootstrap over prompts",
        },
        "baseline": baseline_payload(result),
        "sites": site_payload(result, stats, list(range(len(result.sites)))),
        "per_prompt": {
            "clean_logit_diff": [_f(x) for x in b.clean_ld],
            "corrupt_logit_diff": [_f(x) for x in b.corrupt_ld],
            "clean_answer_prob": [_f(x) for x in b.clean_prob],
            "corrupt_answer_prob": [_f(x) for x in b.corrupt_prob],
        },
        "donors": result.donors,
        "warnings": result.warnings,
        **({"direct": result.extra["direct"]} if "direct" in result.extra else {}),
    }
    return Summary.model_validate(summary).model_dump(mode="json")


def results_table(result: EngineResult, stats: SiteStats) -> pa.Table:
    """Long format: one row per (site, prompt), with everything needed to recompute effects."""
    n_sites, n = result.patched_ld.shape
    site_idx = np.repeat(np.arange(n_sites, dtype=np.int32), n)
    prompt_idx = np.tile(np.arange(n, dtype=np.int32), n_sites)
    sites = result.sites
    columns = {
        "site": site_idx,
        "kind": pa.array(
            [sites[i].kind for i in site_idx], type=pa.dictionary(pa.int8(), pa.string())
        ),
        "layer": np.array([sites[i].layer for i in site_idx], dtype=np.int16),
        "head": np.array(
            [-1 if sites[i].head is None else sites[i].head for i in site_idx], dtype=np.int16
        ),
        "position": pa.array(
            [sites[i].position_key() for i in site_idx], type=pa.dictionary(pa.int16(), pa.string())
        ),
        "prompt": prompt_idx,
        "patched_logit_diff": result.patched_ld.ravel(),
        "patched_answer_prob": result.patched_prob.ravel(),
        "receiver_logit_diff": np.tile(result.receiver_ld, n_sites),
        "reference_logit_diff": np.tile(result.reference_ld, n_sites),
        "receiver_answer_prob": np.tile(result.receiver_prob, n_sites),
        "delta": stats.delta.ravel(),
        "effect": stats.effect.ravel(),
    }
    return pa.table(columns)


def write_results(path: Path, table: pa.Table) -> None:
    with atomic_output(path) as fh:
        pq.write_table(table, fh, compression="zstd", write_statistics=False)


def read_site_rows(path: Path, site: int) -> dict[str, list[Any]]:
    table = pq.read_table(path, filters=[("site", "=", site)])
    table = table.sort_by("prompt")
    return {name: table.column(name).to_pylist() for name in table.column_names}
