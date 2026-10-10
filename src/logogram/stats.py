"""Per-site statistics: means, standard deviations, bootstrap confidence intervals, sign counts.

Everything is vectorized over sites. Bootstrap resamples are drawn once per run from the raw
PCG64 bit stream (stable across NumPy versions) and shared by every site, so differences between
sites are not blurred by different resamples.

Resamples are drawn over prompts, or over clusters of prompts (all prompts from one template,
say), since prompts that share a template aren't independent. A run with many sites also gets
statistics that account for the number of sites: a simultaneous band per site (all sites' bands
hold together with the run's confidence level) and Benjamini–Hochberg q-values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Normalization = str  # "dataset_gap" | "prompt_gap"

# Effects smaller than this are numerically zero: their sign says nothing.
SIGN_TOLERANCE = 1e-9


def resample_counts(n: int, n_boot: int, seed: int) -> np.ndarray:
    """A [n_boot, n] matrix: how often each prompt appears in each bootstrap resample."""
    raw = np.random.PCG64(seed).random_raw(n_boot * n)
    idx = (raw % np.uint64(n)).astype(np.int64).reshape(n_boot, n)
    counts = np.zeros((n_boot, n), dtype=np.float64)
    rows = np.repeat(np.arange(n_boot), n)
    np.add.at(counts, (rows, idx.ravel()), 1.0)
    return counts


def cluster_counts(clusters: list[str], n_boot: int, seed: int) -> np.ndarray:
    """A [n_boot, n] matrix for a cluster bootstrap: each resample draws as many clusters as there
    are, with replacement, and every prompt counts as often as its cluster was drawn."""
    names = sorted(set(clusters))
    k = len(names)
    member = np.array([names.index(c) for c in clusters], dtype=np.int64)
    draws = resample_counts(k, n_boot, seed)  # [n_boot, k]: how often each cluster was drawn
    return draws[:, member]


def _quantiles(boot: np.ndarray, ci: float) -> tuple[np.ndarray, np.ndarray]:
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha], axis=1)
    return lo, hi


def _sd(values: np.ndarray) -> np.ndarray:
    n = values.shape[1]
    if n < 2:
        return np.zeros(values.shape[0])
    return values.std(axis=1, ddof=1)


def _weights(counts: np.ndarray) -> np.ndarray:
    """[n, B]: the weight of each prompt in each resample's mean."""
    n = counts.shape[1]
    totals = counts.sum(axis=1)
    if np.all(totals == n):
        return counts.T / n  # every resample has n prompts (Logogram 0.1's arithmetic)
    return counts.T / totals[None, :]


@dataclass
class SiteStats:
    """Statistics for S sites over n prompts. Arrays have length S unless noted."""

    n: int
    effect: np.ndarray  # [S, n] per-prompt normalized effect
    effect_mean: np.ndarray
    effect_sd: np.ndarray
    effect_lo: np.ndarray
    effect_hi: np.ndarray
    delta: np.ndarray  # [S, n] patched - receiver metric
    delta_mean: np.ndarray
    delta_sd: np.ndarray
    delta_lo: np.ndarray
    delta_hi: np.ndarray
    patched_mean: np.ndarray  # the metric, patched
    pref_mean: np.ndarray  # log P(answer) - log P(distractor), patched
    prob_mean: np.ndarray
    prob_delta_mean: np.ndarray
    sign_flips: np.ndarray  # prompts whose preference changed sign
    opposite_sign: np.ndarray  # prompts whose effect has the opposite sign to the mean
    denominator: float  # mean gap (dataset_gap) or nan (prompt_gap)
    effect_boot: np.ndarray | None = None  # [S, B] each resample's mean effect
    delta_boot: np.ndarray | None = None  # [S, B] each resample's mean change in the metric
    # Corrected for the number of sites (only for a run's full set of sites).
    band_lo: np.ndarray | None = None
    band_hi: np.ndarray | None = None
    band_level: float | None = None  # the two-sided tail kept per site
    q: np.ndarray | None = None


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
    patched: np.ndarray,
    patched_prob: np.ndarray,
    patched_pref: np.ndarray,
    receiver: np.ndarray,
    reference: np.ndarray,
    receiver_prob: np.ndarray,
    receiver_pref: np.ndarray,
    normalization: Normalization,
    counts: np.ndarray,
    ci: float,
    delta: np.ndarray | None = None,
    gap: np.ndarray | None = None,
    family: bool = False,
) -> SiteStats:
    """``patched*`` are [S, n]; ``receiver*``/``reference`` are [n]; ``counts`` is [B, n].

    ``delta`` [S, n] and ``gap`` [n] replace patched - receiver and reference - receiver for
    methods whose per-prompt values aren't patched runs (estimates, and terms of a decomposition).
    Where nothing ran patched, ``patched*`` are NaN and count no sign flips. With ``family``,
    simultaneous bands and q-values over these sites are computed too.
    """
    patched = np.asarray(patched, dtype=np.float64)
    patched_prob = np.asarray(patched_prob, dtype=np.float64)
    patched_pref = np.asarray(patched_pref, dtype=np.float64)
    receiver = np.asarray(receiver, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    receiver_pref = np.asarray(receiver_pref, dtype=np.float64)
    n = patched.shape[1]

    delta = patched - receiver[None, :] if delta is None else np.asarray(delta, dtype=np.float64)
    gap = reference - receiver if gap is None else np.asarray(gap, dtype=np.float64)
    effect, denom = normalize(delta, gap, normalization)

    weights = _weights(counts)  # [n, B]
    with np.errstate(invalid="ignore", divide="ignore"):
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
    flips = ((receiver_pref[None, :] > 0) & (patched_pref < 0)) | (
        (receiver_pref[None, :] < 0) & (patched_pref > 0)
    )
    mean_sign = np.sign(effect_mean)[:, None]
    opposite = (
        (np.sign(effect) == -mean_sign) & (mean_sign != 0) & (np.abs(effect) > SIGN_TOLERANCE)
    )

    stats = SiteStats(
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
        patched_mean=patched.mean(axis=1),
        pref_mean=patched_pref.mean(axis=1),
        prob_mean=patched_prob.mean(axis=1),
        prob_delta_mean=(patched_prob - np.asarray(receiver_prob)[None, :]).mean(axis=1),
        sign_flips=flips.sum(axis=1),
        opposite_sign=opposite.sum(axis=1),
        denominator=denom,
        effect_boot=effect_boot,
        delta_boot=delta_boot,
    )
    if family and n >= 2:
        stats.band_lo, stats.band_hi, stats.band_level = simultaneous_band(effect_boot, ci)
        stats.q = bh_qvalues(bootstrap_pvalues(effect_boot))
    return stats


def simultaneous_band(boot: np.ndarray, ci: float) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Percentile bands that hold for every site at once with probability ``ci``.

    Each resample's most extreme site sets how far into the tails the bands reach: the level is
    chosen so that in a fraction ``ci`` of the resamples, every site lies within its band (the
    max-rank method of Davison and Hinkley). A site's band contains its ordinary interval.
    Sites whose resamples aren't all finite get no band.
    """
    S, B = boot.shape
    finite = np.isfinite(boot).all(axis=1)
    lo = np.full(S, np.nan)
    hi = np.full(S, np.nan)
    if not finite.any() or B < 2:
        return lo, hi, None
    values = boot[finite]
    order = np.argsort(values, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    rows = np.arange(values.shape[0])[:, None]
    ranks[rows, order] = np.arange(B, dtype=np.float64)[None, :]
    # Ties share their average rank, so constant sites are never the extreme one.
    sorted_values = np.take_along_axis(values, order, axis=1)
    for r in range(values.shape[0]):
        same = np.diff(sorted_values[r]) == 0
        if same.any():
            ranks[r] = _average_ties(sorted_values[r], order[r], B)
    # How far into the nearer tail each value is (0 at the extremes, 0.5 in the middle).
    tail = np.minimum(ranks + 1.0, B - ranks) / B
    extreme = tail.min(axis=0)  # per resample: its most extreme site
    level = float(np.quantile(extreme, 1.0 - ci, method="lower"))
    level = max(level, 1.0 / B)
    q_lo, q_hi = np.quantile(values, [level, 1.0 - level], axis=1)
    lo[finite], hi[finite] = q_lo, q_hi
    return lo, hi, level


def _average_ties(sorted_values: np.ndarray, order: np.ndarray, B: int) -> np.ndarray:
    ranks = np.empty(B, dtype=np.float64)
    i = 0
    while i < B:
        j = i
        while j + 1 < B and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def bootstrap_pvalues(boot: np.ndarray) -> np.ndarray:
    """Two-sided bootstrap p-values for "the mean effect is zero": twice the smaller share of
    resamples on either side of zero (with the usual +1), so p < α exactly when the percentile
    interval at level 1 - α excludes zero. NaN for sites without finite resamples."""
    B = boot.shape[1]
    with np.errstate(invalid="ignore"):
        below = (boot <= 0).sum(axis=1)
        above = (boot >= 0).sum(axis=1)
    p = np.minimum(1.0, 2.0 * (np.minimum(below, above) + 1.0) / (B + 1.0))
    p[~np.isfinite(boot).all(axis=1)] = np.nan
    return p


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg q-values: the smallest false discovery rate at which each site counts
    as a discovery. NaN p-values stay NaN and don't count as tests."""
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if not m:
        return q
    values = p[ok]
    order = np.argsort(values, kind="mergesort")
    ranked = values[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def paired_difference(
    a: np.ndarray, b: np.ndarray, counts: np.ndarray, ci: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean of b - a per row, with a percentile interval from the same resamples: a and b are
    [S, n] per-prompt values measured on the same prompts."""
    diff = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    boot = diff @ _weights(counts)
    lo, hi = _quantiles(boot, ci)
    return diff.mean(axis=1), lo, hi


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
