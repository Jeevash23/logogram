"""Per-site statistics: means, standard deviations, bootstrap confidence intervals, sign counts.

Everything is vectorized over sites. Bootstrap resamples are drawn once per run from the raw
PCG64 bit stream (stable across NumPy versions) and shared by every site, so differences between
sites are not blurred by different resamples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Normalization = str  # "dataset_gap" | "prompt_gap"


def resample_counts(n: int, n_boot: int, seed: int) -> np.ndarray:
    """A [n_boot, n] matrix: how often each prompt appears in each bootstrap resample."""
    raw = np.random.PCG64(seed).random_raw(n_boot * n)
    idx = (raw % np.uint64(n)).astype(np.int64).reshape(n_boot, n)
    counts = np.zeros((n_boot, n), dtype=np.float64)
    rows = np.repeat(np.arange(n_boot), n)
    np.add.at(counts, (rows, idx.ravel()), 1.0)
    return counts


def _quantiles(boot: np.ndarray, ci: float) -> tuple[np.ndarray, np.ndarray]:
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha], axis=1)
    return lo, hi


def _sd(values: np.ndarray) -> np.ndarray:
    n = values.shape[1]
    if n < 2:
        return np.zeros(values.shape[0])
    return values.std(axis=1, ddof=1)


@dataclass
class SiteStats:
    """Statistics for S sites over n prompts. Arrays have length S unless noted."""

    n: int
    effect: np.ndarray  # [S, n] per-prompt normalized effect
    effect_mean: np.ndarray
    effect_sd: np.ndarray
    effect_lo: np.ndarray
    effect_hi: np.ndarray
    delta: np.ndarray  # [S, n] patched - receiver logit difference
    delta_mean: np.ndarray
    delta_sd: np.ndarray
    delta_lo: np.ndarray
    delta_hi: np.ndarray
    patched_mean: np.ndarray
    prob_mean: np.ndarray
    prob_delta_mean: np.ndarray
    sign_flips: np.ndarray  # prompts whose logit difference changed sign
    opposite_sign: np.ndarray  # prompts whose effect has the opposite sign to the mean
    denominator: float  # mean gap (dataset_gap) or nan (prompt_gap)


def normalize(
    delta: np.ndarray, gap: np.ndarray, normalization: Normalization
) -> tuple[np.ndarray, float]:
    if normalization == "dataset_gap":
        denom = float(gap.mean())
        return delta / denom, denom
    if normalization == "prompt_gap":
        return delta / gap[None, :], float("nan")
    raise ValueError(f"unknown normalization {normalization!r}")


def compute_site_stats(
    patched_ld: np.ndarray,
    patched_prob: np.ndarray,
    receiver_ld: np.ndarray,
    source_ld: np.ndarray,
    receiver_prob: np.ndarray,
    normalization: Normalization,
    counts: np.ndarray,
    ci: float,
) -> SiteStats:
    """``patched_*`` are [S, n]; ``receiver_*``/``source_ld`` are [n]; ``counts`` is [B, n]."""
    patched_ld = np.asarray(patched_ld, dtype=np.float64)
    patched_prob = np.asarray(patched_prob, dtype=np.float64)
    receiver_ld = np.asarray(receiver_ld, dtype=np.float64)
    source_ld = np.asarray(source_ld, dtype=np.float64)
    n = patched_ld.shape[1]

    delta = patched_ld - receiver_ld[None, :]
    gap = source_ld - receiver_ld
    effect, denom = normalize(delta, gap, normalization)

    weights = counts.T / n  # [n, B]
    delta_boot = delta @ weights  # [S, B]
    if normalization == "dataset_gap":
        gap_boot = gap @ weights  # [B]
        effect_boot = delta_boot / gap_boot[None, :]
    else:
        effect_boot = effect @ weights
    effect_lo, effect_hi = _quantiles(effect_boot, ci)
    delta_lo, delta_hi = _quantiles(delta_boot, ci)
    if n < 2:
        # One prompt has no sampling distribution: report no interval rather than a false one.
        nan = np.full(len(effect_lo), np.nan)
        effect_lo, effect_hi, delta_lo, delta_hi = nan, nan.copy(), nan.copy(), nan.copy()

    effect_mean = effect.mean(axis=1)
    flips = ((receiver_ld[None, :] > 0) & (patched_ld < 0)) | (
        (receiver_ld[None, :] < 0) & (patched_ld > 0)
    )
    mean_sign = np.sign(effect_mean)[:, None]
    opposite = (np.sign(effect) == -mean_sign) & (mean_sign != 0)

    return SiteStats(
        n=n,
        effect=effect,
        effect_mean=effect_mean,
        effect_sd=_sd(effect),
        effect_lo=effect_lo,
        effect_hi=effect_hi,
        delta=delta,
        delta_mean=delta.mean(axis=1),
        delta_sd=_sd(delta),
        delta_lo=delta_lo,
        delta_hi=delta_hi,
        patched_mean=patched_ld.mean(axis=1),
        prob_mean=patched_prob.mean(axis=1),
        prob_delta_mean=(patched_prob - np.asarray(receiver_prob)[None, :]).mean(axis=1),
        sign_flips=flips.sum(axis=1),
        opposite_sign=opposite.sum(axis=1),
        denominator=denom,
    )


def rankdata(values: np.ndarray) -> np.ndarray:
    """Ranks starting at 1, ties get the average rank (as in scipy.stats.rankdata)."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom == 0:
        return float("nan")
    return float((ra * rb).sum() / denom)
