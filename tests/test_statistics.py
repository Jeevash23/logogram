"""Statistics over many sites and clustered prompts."""

from __future__ import annotations

import numpy as np
import pytest

from logogram.results import StatsError, bootstrap_counts, compute_stats
from logogram.runner import run_spec
from logogram.stats import (
    bh_qvalues,
    bootstrap_pvalues,
    cluster_counts,
    compute_site_stats,
    paired_difference,
    resample_counts,
    simultaneous_band,
)


def _stats(effects, counts, ci=0.95):
    """Site statistics of per-prompt effects (a receiver at 0 and a reference at 1, so effects
    are the deltas)."""
    effects = np.asarray(effects, dtype=float)
    n = effects.shape[1]
    zeros, ones = np.zeros(n), np.ones(n)
    return compute_site_stats(
        patched=effects,
        patched_prob=effects * 0,
        patched_pref=effects,
        receiver=zeros,
        reference=ones,
        receiver_prob=zeros,
        receiver_pref=zeros,
        normalization="dataset_gap",
        counts=counts,
        ci=ci,
        family=True,
    )


def test_a_cluster_bootstrap_resamples_whole_clusters():
    clusters = ["a", "a", "b", "b", "b", "c"]
    counts = cluster_counts(clusters, 300, seed=0)
    assert counts.shape == (300, 6)
    # Prompts of a cluster are always drawn together.
    np.testing.assert_array_equal(counts[:, 0], counts[:, 1])
    np.testing.assert_array_equal(counts[:, 2], counts[:, 4])
    # Each resample draws as many clusters as there are.
    per_cluster = counts[:, [0, 2, 5]]
    np.testing.assert_array_equal(per_cluster.sum(1), 3)
    # Same seed, same resamples.
    np.testing.assert_array_equal(counts, cluster_counts(clusters, 300, seed=0))


def test_clustered_intervals_are_wider_when_clusters_differ():
    """Prompts within a template agree and templates differ: resampling single prompts treats 30
    prompts as 30 observations, resampling templates as 3, and the interval says so."""
    rng = np.random.default_rng(0)
    template = np.repeat([0.0, 1.0, 2.0], 10)
    effects = (template + rng.normal(0, 0.05, 30))[None, :]
    single = _stats(effects, resample_counts(30, 2000, 0))
    clustered = _stats(effects, cluster_counts([str(t) for t in template], 2000, 0))
    width = lambda s: float(s.effect_hi[0] - s.effect_lo[0])  # noqa: E731
    assert width(clustered) > 2 * width(single)


def test_simultaneous_bands_contain_each_interval_and_hold_together():
    rng = np.random.default_rng(1)
    S, n = 40, 25
    effects = rng.normal(0, 1, (S, n))  # no site has an effect
    counts = resample_counts(n, 1000, 0)
    stats = _stats(effects, counts)
    assert stats.band_lo is not None and stats.band_hi is not None
    assert (stats.band_lo <= stats.effect_lo + 1e-12).all()
    assert (stats.band_hi >= stats.effect_hi - 1e-12).all()
    # In about 95% of the resamples every site lies within its band (the definition).
    boot = stats.effect_boot
    inside = ((boot >= stats.band_lo[:, None]) & (boot <= stats.band_hi[:, None])).all(0)
    assert inside.mean() == pytest.approx(0.95, abs=0.02)
    # One site alone: the band is its ordinary interval, near enough.
    lo, _hi, level = simultaneous_band(boot[:1], 0.95)
    assert level == pytest.approx(0.025, abs=0.003)
    assert lo[0] == pytest.approx(stats.effect_lo[0], abs=0.02)


def test_bootstrap_pvalues_match_the_intervals():
    rng = np.random.default_rng(2)
    effects = rng.normal(0, 1, (30, 20)) + np.linspace(-1.5, 1.5, 30)[:, None]
    stats = _stats(effects, resample_counts(20, 2000, 3))
    p = bootstrap_pvalues(stats.effect_boot)
    excludes = (stats.effect_lo > 0) | (stats.effect_hi < 0)
    # p < 0.05 exactly when the 95% percentile interval excludes zero (up to the +1).
    agree = (p < 0.05) == excludes
    assert agree.mean() > 0.9
    assert ((p > 0) & (p <= 1)).all()


def test_benjamini_hochberg_q_values():
    p = np.array([0.01, 0.04, 0.03, 0.20, np.nan])
    q = bh_qvalues(p)
    # Sorted p: 0.01, 0.03, 0.04, 0.20 (m = 4); q = min over larger ranks of p * m / rank.
    np.testing.assert_allclose(q[:4], [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.20])
    assert np.isnan(q[4])
    assert (bh_qvalues(np.array([0.5, 0.9])) <= 1).all()


def test_paired_differences_use_shared_resamples():
    rng = np.random.default_rng(4)
    base = rng.normal(0, 3, (1, 40))  # large spread between prompts
    shifted = base + 0.3 + rng.normal(0, 0.01, (1, 40))  # a small, consistent shift
    counts = resample_counts(40, 1000, 0)
    mean, lo, _hi = paired_difference(base, shifted, counts, 0.95)
    assert mean[0] == pytest.approx(0.3, abs=0.01)
    assert lo[0] > 0  # paired, the shift is clear; unpaired intervals would overlap
    a, b = _stats(base, counts), _stats(shifted, counts)
    assert a.effect_hi[0] > b.effect_lo[0]


def test_opposite_signs_ignore_rounding_noise():
    effects = np.array([[1e-12, -1e-12, 2e-12, 1e-13], [0.5, -0.1, 0.4, 0.3]])
    stats = _stats(effects, resample_counts(4, 200, 0))
    assert stats.opposite_sign.tolist() == [0, 1]


def test_a_run_can_resample_templates(tiny_backend, project, spec_factory):
    spec = spec_factory(
        dataset={"path": "datasets/mixed.jsonl"},
        statistics={"bootstrap": 200, "seed": 0, "cluster": "template"},
    )
    outcome = run_spec(spec, project, backend=tiny_backend)
    assert outcome.status == "finished", outcome.manifest.get("error")
    stats = outcome.summary["statistics"]
    assert stats["cluster"] == "template" and stats["clusters"] == 3
    assert "clusters of prompts" in stats["method"]
    site = outcome.summary["sites"][0]
    assert site["band"] is not None and site["q"] is not None
    assert stats["multiple_comparisons"].startswith("Simultaneous bands hold for all")
    # Three templates are too few clusters for reliable intervals, and the run says so.
    assert any("only 3 clusters" in w for w in outcome.summary["warnings"])

    missing = spec_factory(statistics={"bootstrap": 200, "seed": 0, "cluster": "nothing"})
    failed = run_spec(missing, project, backend=tiny_backend)
    assert failed.status == "failed" and "has no 'nothing' in its meta" in failed.manifest["error"]


def test_one_cluster_is_refused(tiny_backend, project, spec_factory):
    from logogram.datasets import load_dataset
    from logogram.engine import run_engine
    from logogram.prompts import prepare_prompts

    spec = spec_factory(statistics={"bootstrap": 200, "seed": 0, "cluster": "corruption"})
    prompts = prepare_prompts(tiny_backend, load_dataset(project.datasets_dir / "ioi.jsonl"), True)
    result = run_engine(spec, tiny_backend, prompts)
    with pytest.raises(StatsError, match="one cluster"):
        bootstrap_counts(spec, result)
    with pytest.raises(StatsError):
        compute_stats(spec, result)
