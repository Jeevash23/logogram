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
from logogram.spec import METRIC_LABELS, Spec, describe_experiment, describe_metric
from logogram.stats import SiteStats, cluster_counts, compute_site_stats, resample_counts

SUMMARY_VERSION = 1


def _f(x: Any) -> float | None:
    """A JSON-safe float (None for NaN or infinity)."""
    value = float(x)
    return value if math.isfinite(value) else None


class StatsError(ValueError):
    pass


def bootstrap_counts(spec: Spec, result: EngineResult) -> np.ndarray:
    """The resamples a run's statistics use: over prompts, or over clusters of prompts that share
    the value of ``statistics.cluster`` in their ``meta``."""
    n = len(result.prompts)
    stats = spec.statistics
    if stats.cluster is None:
        return resample_counts(n, stats.bootstrap, stats.seed)
    clusters = []
    for p in result.prompts:
        meta = p.record.meta or {}
        if stats.cluster not in meta:
            raise StatsError(
                f"Prompt {p.index} has no {stats.cluster!r} in its meta, so it can't be put in a "
                "cluster. Choose a field every prompt has, or resample single prompts."
            )
        clusters.append(str(meta[stats.cluster]))
    if len(set(clusters)) < 2:
        raise StatsError(
            f"Every prompt has the same {stats.cluster!r}, so there is one cluster and nothing to "
            "resample. Choose another field, or resample single prompts."
        )
    return cluster_counts(clusters, stats.bootstrap, stats.seed)


def compute_stats(
    spec: Spec,
    result: EngineResult,
    counts: np.ndarray | None = None,
    indices: list[int] | None = None,
) -> SiteStats:
    """Statistics for all sites (with the corrections for their number), or for ``indices`` only
    (rows then follow ``indices``, and nothing is corrected: the corrections need every site).
    A site's own statistics depend only on its rows and the resamples, so they are the same
    either way."""
    if counts is None:
        counts = bootstrap_counts(spec, result)
    rows = slice(None) if indices is None else np.asarray(indices, dtype=np.int64)
    return compute_site_stats(
        patched=result.patched[rows],
        patched_prob=result.patched_prob[rows],
        patched_pref=result.patched_pref[rows],
        receiver=result.receiver_metric,
        reference=result.reference_metric,
        receiver_prob=result.receiver_prob,
        receiver_pref=result.receiver_pref,
        normalization=spec.metric.normalization,
        counts=counts,
        ci=spec.statistics.ci,
        delta=None if result.delta is None else result.delta[rows],
        gap=result.gap,
        family=indices is None,
        # An edge attribution run's many edges don't keep their resampled means (nothing
        # compares edges by them), so they go through in chunks.
        keep_boot="edges" not in result.extra,
    )


def take_rows(stats: SiteStats, rows: list[int]) -> SiteStats:
    """``stats`` for the sites at ``rows`` only (row i is site rows[i]), with the corrections
    computed over every site kept as they are."""
    at = np.asarray(rows, dtype=np.int64)
    names = (
        "effect", "effect_mean", "effect_sd", "effect_lo", "effect_hi", "delta", "delta_mean",
        "delta_sd", "delta_lo", "delta_hi", "patched_mean", "pref_mean", "prob_mean",
        "prob_delta_mean", "sign_flips", "opposite_sign", "effect_boot", "delta_boot", "band_lo",
        "band_hi", "q",
    )  # fmt: skip
    arrays = {name: getattr(stats, name) for name in names}
    return SiteStats(
        n=stats.n,
        denominator=stats.denominator,
        band_level=stats.band_level,
        **{name: None if value is None else value[at] for name, value in arrays.items()},
    )


def by_site(stats: SiteStats, indices: list[int], n_sites: int) -> SiteStats:
    """``stats`` for the sites ``indices`` (row i is site indices[i]) laid out by site index, with
    NaN for the sites they leave out: for what compares sites by their index (a circuit's sets, a
    steered site and its control), so whatever needs a site that isn't there is undefined."""
    at = np.asarray(indices, dtype=np.int64)

    def spread(values: np.ndarray | None) -> np.ndarray | None:
        if values is None:
            return None
        out = np.full((n_sites, *values.shape[1:]), np.nan)
        out[at] = values
        return out

    arrays = {
        name: spread(getattr(stats, name))
        for name in (
            "effect", "effect_mean", "effect_sd", "effect_lo", "effect_hi", "delta",
            "delta_mean", "delta_sd", "delta_lo", "delta_hi", "patched_mean", "pref_mean",
            "prob_mean", "prob_delta_mean", "sign_flips", "opposite_sign", "effect_boot",
            "delta_boot",
        )
    }  # fmt: skip
    return SiteStats(n=stats.n, denominator=stats.denominator, **arrays)  # type: ignore[arg-type]


def _stat(stats: SiteStats, name: str, i: int) -> dict[str, float | None]:
    return {
        "mean": _f(getattr(stats, f"{name}_mean")[i]),
        "sd": _f(getattr(stats, f"{name}_sd")[i]),
        "lo": _f(getattr(stats, f"{name}_lo")[i]),
        "hi": _f(getattr(stats, f"{name}_hi")[i]),
    }


def site_payload(
    result: EngineResult, stats: SiteStats, indices: list[int]
) -> list[dict[str, Any]]:
    """Payload for sites ``indices``; row ``i`` of ``stats`` belongs to ``indices[i]``."""
    out = []
    for i, site_index in enumerate(indices):
        site = result.sites[site_index]
        band = None
        if stats.band_lo is not None and stats.band_hi is not None:
            lo, hi = _f(stats.band_lo[i]), _f(stats.band_hi[i])
            band = {"lo": lo, "hi": hi} if lo is not None and hi is not None else None
        out.append(
            {
                **site.to_dict(),
                "n": stats.n,
                "effect": _stat(stats, "effect", i),
                "delta": _stat(stats, "delta", i),
                "patched_metric": _f(stats.patched_mean[i]),
                "patched_logit_diff": _f(stats.pref_mean[i]),
                "answer_prob": _f(stats.prob_mean[i]),
                "answer_prob_delta": _f(stats.prob_delta_mean[i]),
                "sign_flips": int(stats.sign_flips[i]),
                "opposite_sign": int(stats.opposite_sign[i]),
                "band": band,
                "q": None if stats.q is None else _f(stats.q[i]),
            }
        )
    return out


def _group_stats(values: np.ndarray) -> dict[str, float | None]:
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {"mean": _f(values.mean()), "sd": _f(sd)}


def baseline_payload(result: EngineResult) -> dict[str, Any]:
    b = result.baselines
    gap = result.reference_metric - result.receiver_metric
    return {
        "clean": {
            "logit_diff": _group_stats(b.clean_pref),
            "answer_prob": _group_stats(b.clean_prob),
            "prefers_answer": int((b.clean_pref > 0).sum()),
            "metric": _group_stats(b.clean),
        },
        "corrupt": {
            "logit_diff": _group_stats(b.corrupt_pref),
            "answer_prob": _group_stats(b.corrupt_prob),
            "prefers_answer": int((b.corrupt_pref > 0).sum()),
            "metric": _group_stats(b.corrupt),
        },
        "gap": _group_stats(gap),
    }


def build_summary(
    spec: Spec,
    result: EngineResult,
    stats: SiteStats,
    run_id: str,
    model: dict[str, Any] | None = None,
    chat_template: dict[str, Any] | None = None,
    *,
    kept: list[int] | None = None,
    partial: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The summary of a run. ``chat_template``: how the prompts were wrapped (logogram.chat), if
    they were, for the methods text to say. With ``kept`` (a run that stopped early), only those
    sites are reported, and ``stats`` rows follow ``kept``; ``partial`` says what the run kept.
    An edge attribution run reports its strongest edges (edges.parquet beside the summary holds
    every edge's statistics)."""
    b = result.baselines
    if kept is None and "edges" in result.extra:
        from logogram.edges import correction_counts, reported_edges

        # Counted over every edge: the summary lists only the strongest.
        result.extra["edges"]["corrections"] = correction_counts(stats, 1 - spec.statistics.ci)
        top = reported_edges(stats)
        if len(top) < len(result.sites):
            kept, stats = top, take_rows(stats, top)
    indices = list(range(len(result.sites))) if kept is None else list(kept)
    # What compares sites by their index reads the statistics laid out by site.
    aligned = stats if kept is None else by_site(stats, indices, len(result.sites))
    reported = set(indices)
    normalization = spec.metric.normalization
    label = METRIC_LABELS[spec.metric.kind]
    description = describe_metric(spec.metric)
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
            f"{label}; mean gap = {stats.denominator:.4f}"
        )
    else:
        norm_text = (
            f"(patched − {result.receiver}) ÷ ({result.reference} − {result.receiver}) "
            f"{label}, per prompt"
        )
    if "steering" in result.extra:
        norm_text = norm_text.replace("patched", "steered")
    warnings = list(result.warnings)
    if "steering" in result.extra:
        from logogram.steering import control_comparison, control_warnings

        steering = result.extra["steering"]
        control = [
            c
            for c in control_comparison(result, aligned, spec.statistics.ci)
            if c["index"] in reported and c["control_index"] in reported
        ]
        steering["control"] = control
        warnings += control_warnings(
            control, spec.statistics.ci, feature=steering.get("feature") is not None
        )
    if "circuit" in result.extra:
        from logogram.circuits import circuit_summary

        circuit = circuit_summary(result, aligned, spec.statistics.ci)
        circuit["rows"] = [r for r in circuit["rows"] if r["index"] in reported]
        result.extra["circuit"].update(circuit)
    if result.measure == "estimate":
        estimated = result.extra.get("attribution") or result.extra.get("edges") or {}
        how = (
            "integrated gradients along the way from receiver to source"
            if estimated.get("method") == "integrated_gradients"
            else "its gradient at the receiver run"
        )
        description = (
            f"first-order estimate of the change patching would cause in {description}: "
            f"(source − receiver activation) · {how}"
        )
        if "edges" in result.extra:
            edges = result.extra["edges"]
            gradient = (
                f"the metric's gradient averaged over {edges['steps']} runs between the two "
                "prompts' input embeddings (integrated gradients)"
                if edges["method"] == "integrated_gradients"
                else "the metric's gradient at the receiver run"
            )
            description = (
                f"first-order estimate of the change patching each edge alone would cause in "
                f"{describe_metric(spec.metric)}: (the sender's output on the source prompt − on "
                f"the receiver prompt) · {gradient}, summed over positions; the gradient is "
                f"{edges['gradient']}"
            )
        norm_text = f"estimated {norm_text}"
    method = "percentile bootstrap over prompts"
    if spec.statistics.cluster is not None:
        method = (
            f"percentile bootstrap over clusters of prompts with the same {spec.statistics.cluster}"
        )
        clusters = _n_clusters(spec, result) or 0
        if clusters < FEW_CLUSTERS:
            warnings.append(
                f"The bootstrap resamples only {clusters} clusters (prompts with the same "
                f"{spec.statistics.cluster}), too few for reliable intervals. Use prompts from "
                f"more {spec.statistics.cluster} values, or read the intervals as rough."
            )
    summary = {
        "logogram_summary": SUMMARY_VERSION,
        "run_id": run_id,
        "name": spec.name,
        "description": describe_experiment(spec),
        "model": model,
        "n_prompts": len(result.prompts),
        "n_sites": len(indices),
        "layout": result.layout,
        "receiver": result.receiver,
        "reference": result.reference,
        "measure": result.measure,
        "metric": {
            "kind": spec.metric.kind,
            "target": getattr(spec.metric, "target", None),
            "label": label,
            "normalization": normalization,
            "denominator": _f(stats.denominator) if normalization == "dataset_gap" else None,
            "description": description,
            "normalized_effect": norm_text,
        },
        "statistics": {
            "bootstrap": spec.statistics.bootstrap,
            "ci": spec.statistics.ci,
            "seed": spec.statistics.seed,
            "method": method,
            "cluster": spec.statistics.cluster,
            "clusters": _n_clusters(spec, result),
            "band_level": stats.band_level,
            "multiple_comparisons": (
                f"Simultaneous bands hold for all {len(result.sites)} sites together with "
                f"{spec.statistics.ci:.0%} confidence (max-rank percentile method); q-values are "
                "Benjamini–Hochberg adjusted bootstrap p-values over the same sites."
                if stats.band_level is not None
                else None
            ),
        },
        "baseline": baseline_payload(result),
        "sites": site_payload(result, stats, indices),
        "per_prompt": {
            "clean_logit_diff": [_f(x) for x in b.clean_pref],
            "corrupt_logit_diff": [_f(x) for x in b.corrupt_pref],
            "clean_answer_prob": [_f(x) for x in b.clean_prob],
            "corrupt_answer_prob": [_f(x) for x in b.corrupt_prob],
            "clean_metric": [_f(x) for x in b.clean],
            "corrupt_metric": [_f(x) for x in b.corrupt],
        },
        "donors": result.donors,
        "warnings": warnings,
        **({"direct": result.extra["direct"]} if "direct" in result.extra else {}),
        **({"steering": result.extra["steering"]} if "steering" in result.extra else {}),
        **({"features": result.extra["features"]} if "features" in result.extra else {}),
        **({"circuit": result.extra["circuit"]} if "circuit" in result.extra else {}),
        **({"attribution": result.extra["attribution"]} if "attribution" in result.extra else {}),
        **({"edges": result.extra["edges"]} if "edges" in result.extra else {}),
        **({"chat_template": chat_template} if chat_template is not None else {}),
        **({"partial": partial} if partial is not None else {}),
    }
    return Summary.model_validate(summary).model_dump(mode="json")


# Below this many clusters, a cluster bootstrap's intervals are unreliable.
FEW_CLUSTERS = 10


def _n_clusters(spec: Spec, result: EngineResult) -> int | None:
    key = spec.statistics.cluster
    if key is None:
        return None
    return len({str((p.record.meta or {}).get(key)) for p in result.prompts})


def results_table(
    result: EngineResult, stats: SiteStats, kept: list[int] | None = None
) -> pa.Table:
    """Long format: one row per (site, prompt), with everything needed to recompute effects.
    With ``kept`` (a run that stopped early), only those sites' rows, and ``stats`` rows follow
    ``kept``; a site's ``site`` value is its index in the whole sweep either way.

    ``*_logit_diff`` columns hold the preference, log P(answer) − log P(distractor) (the logit
    difference for single tokens); ``*_metric`` columns hold the spec's metric."""
    at = list(range(len(result.sites))) if kept is None else list(kept)
    rows = np.asarray(at, dtype=np.int64)
    n_sites, n = len(at), len(result.prompts)
    position = np.repeat(np.arange(n_sites), n)
    sites = [result.sites[i] for i in at]
    # The prompt's index in the dataset (a method may measure only some prompts, as steering
    # measures the held-out ones).
    prompt_idx = np.tile(np.array([p.index for p in result.prompts], dtype=np.int32), n_sites)
    columns = {
        "site": np.array([sites[i].index for i in position], dtype=np.int32),
        "kind": pa.array(
            [sites[i].kind for i in position], type=pa.dictionary(pa.int8(), pa.string())
        ),
        "layer": np.array([sites[i].layer for i in position], dtype=np.int16),
        "head": np.array(
            [-1 if sites[i].head is None else sites[i].head for i in position], dtype=np.int16
        ),
        "feature": np.array(
            [-1 if sites[i].site.feature is None else sites[i].site.feature for i in position],
            dtype=np.int32,
        ),
        "position": pa.array(
            [sites[i].position_key() for i in position], type=pa.dictionary(pa.int16(), pa.string())
        ),
        "variant": pa.array(
            [sites[i].variant_key or "" for i in position],
            type=pa.dictionary(pa.int16(), pa.string()),
        ),
        "prompt": prompt_idx,
        "patched_logit_diff": result.patched_pref[rows].ravel(),
        "patched_answer_prob": result.patched_prob[rows].ravel(),
        "receiver_logit_diff": np.tile(result.receiver_pref, n_sites),
        "reference_logit_diff": np.tile(result.reference_pref, n_sites),
        "receiver_answer_prob": np.tile(result.receiver_prob, n_sites),
        "delta": stats.delta.ravel(),
        "effect": stats.effect.ravel(),
        "patched_metric": result.patched[rows].ravel(),
        "receiver_metric": np.tile(result.receiver_metric, n_sites),
        "reference_metric": np.tile(result.reference_metric, n_sites),
    }
    return pa.table(columns)


def write_results(path: Path, table: pa.Table) -> None:
    with atomic_output(path) as fh:
        pq.write_table(table, fh, compression="zstd", write_statistics=False)


def read_site_rows(path: Path, site: int) -> dict[str, list[Any]]:
    table = pq.read_table(path, filters=[("site", "=", site)])
    table = table.sort_by("prompt")
    rows = {name: table.column(name).to_pylist() for name in table.column_names}
    # Runs from Logogram 0.1 measured the logit difference only: it is their metric.
    for name in ("patched", "receiver", "reference"):
        rows.setdefault(f"{name}_metric", rows.get(f"{name}_logit_diff", []))
    return rows


def largest_change(a: pa.Table, b: pa.Table) -> float | None:
    """The largest difference between two results tables' per-prompt values, over the columns
    both have: 0.0 when they match exactly, None when they don't hold the same rows. Values a
    method doesn't measure (a direct effect's patched probability) are NaN in both and match;
    Arrow's own equality never matches NaN. Columns only one table has (a newer version records
    more) are left out."""
    if a.num_rows != b.num_rows:
        return None
    shared = [name for name in a.column_names if name in b.column_names]
    if not {"site", "prompt", "delta", "effect"} <= set(shared):
        return None
    largest = 0.0
    for name in shared:
        x, y = a.column(name), b.column(name)
        if not pa.types.is_floating(x.type) or not pa.types.is_floating(y.type):
            if not x.equals(y):
                return math.inf
            continue
        u, v = x.to_numpy(), y.to_numpy()
        if len(u) and (np.isnan(u).all() or np.isnan(v).all()):
            continue  # one version doesn't measure this (an estimate's patched preference)
        change = np.where(np.isnan(u) & np.isnan(v), 0.0, np.abs(u - v))
        largest = max(largest, float(np.nan_to_num(change, nan=np.inf).max(initial=0.0)))
    return largest
