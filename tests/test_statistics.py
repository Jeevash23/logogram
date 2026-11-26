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
    crossed_counts,
    empty_resamples,
    holm_pvalues,
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


@pytest.mark.parametrize("normalization", ["dataset_gap", "prompt_gap"])
def test_a_sites_numbers_dont_change_with_the_sites_after_it(normalization):
    """A site's interval is the same whether the run measured the sites after it or not, so a run
    that stopped early keeps exactly what the whole run gives each site it kept. One matrix
    product over every site rounds a row differently as the number of rows changes (one row
    alone goes through another BLAS routine on common machines), which this catches."""
    rng = np.random.default_rng(3)
    n = 5
    patched = rng.normal(0.4, 1.0, (17, n))
    receiver, reference = rng.normal(0.0, 0.1, n), rng.normal(1.0, 0.1, n)
    counts = resample_counts(n, 200, seed=4)

    def stats(rows):
        return compute_site_stats(
            patched=patched[rows],
            patched_prob=patched[rows] * 0,
            patched_pref=patched[rows],
            receiver=receiver,
            reference=reference,
            receiver_prob=np.zeros(n),
            receiver_pref=receiver,
            normalization=normalization,
            counts=counts,
            ci=0.95,
        )

    every = stats(slice(None))
    for rows in ([0], [0, 1], list(range(3)), list(range(16))):
        some = stats(rows)
        for name in ("effect_boot", "delta_boot", "effect_lo", "effect_hi", "delta_lo", "delta_hi"):
            np.testing.assert_array_equal(getattr(some, name), getattr(every, name)[rows])
    a, b = rng.normal(0, 1, (9, n)), rng.normal(0, 1, (9, n))
    whole = paired_difference(a, b, counts, 0.95)
    alone = paired_difference(a[:1], b[:1], counts, 0.95)
    for x, y in zip(alone, whole, strict=True):
        np.testing.assert_array_equal(x, y[:1])


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


def test_holm_adjusted_p_values():
    p = np.array([0.01, 0.04, 0.03, 0.20, np.nan])
    adjusted = holm_pvalues(p)
    # Sorted p: 0.01, 0.03, 0.04, 0.20 (m = 4), times 4, 3, 2, 1, and never falling.
    np.testing.assert_allclose(adjusted[:4], [0.04, 0.09, 0.09, 0.20])
    assert np.isnan(adjusted[4])
    # Never below the p-value or the q-value, and never above 1.
    assert (adjusted[:4] >= p[:4]).all() and (adjusted[:4] >= bh_qvalues(p)[:4] - 1e-12).all()
    assert (holm_pvalues(np.array([0.5, 0.9])) <= 1).all()
    assert len(holm_pvalues(np.array([]))) == 0
    # A run's sites carry them beside their q-values.
    rng = np.random.default_rng(0)
    effects = rng.normal(size=(6, 40)) + np.array([[3.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    stats = _stats(effects, resample_counts(40, 500, seed=0))
    assert stats.p_holm is not None and stats.q is not None
    assert (stats.p_holm >= stats.q - 1e-12).all()
    assert stats.p_holm[0] < 0.05 < stats.p_holm[1:].min()


def test_a_crossed_bootstrap_resamples_two_fields_at_once():
    names = ["a", "a", "a", "b", "b", "b", "c", "c", "c"]
    places = ["x", "y", "z", "x", "y", "z", "x", "y", "z"]
    counts = crossed_counts(names, places, 400, seed=0)
    assert counts.shape == (400, 9)
    # The first field is drawn as a bootstrap over it alone draws it.
    by_name = cluster_counts(names, 400, seed=0)
    by_place = counts / np.where(by_name == 0, 1, by_name)
    # A prompt counts as often as its two values' draws multiplied: what is left once the name's
    # draws are divided out is one draw per place, the same for every name.
    for b in range(400):
        drawn = by_name[b] > 0
        table = by_place[b].reshape(3, 3)[drawn.reshape(3, 3)[:, 0]]
        assert (table == table[0]).all() and table[0].sum() == 3
    # Each prompt is drawn once on average, and the same seed draws the same resamples.
    assert counts.mean() == pytest.approx(1.0, abs=0.08)
    np.testing.assert_array_equal(counts, crossed_counts(names, places, 400, seed=0))
    assert empty_resamples(counts) == 0
    # Prompts that differ by name and by place get wider intervals than a bootstrap over either.
    rng = np.random.default_rng(1)
    name_effect = {"a": -1.0, "b": 0.0, "c": 1.0}
    place_effect = {"x": -1.0, "y": 0.0, "z": 1.0}
    values = np.array(
        [[name_effect[n] + place_effect[p] for n, p in zip(names, places, strict=True)]]
    )
    values = values + rng.normal(scale=0.01, size=values.shape) + 5.0

    def width(c):
        stats = _stats(values, c)
        return float(stats.delta_hi[0] - stats.delta_lo[0])

    crossed = width(crossed_counts(names, places, 2000, seed=0))
    assert crossed > width(cluster_counts(names, 2000, seed=0)) * 1.2
    assert crossed > width(cluster_counts(places, 2000, seed=0)) * 1.2
    # A design where each name has its own place can draw no prompt at all.
    sparse = crossed_counts(["a", "b", "c"], ["x", "y", "z"], 400, seed=0)
    assert empty_resamples(sparse) > 0


def test_a_run_can_resample_two_crossed_fields(tiny_backend, project, spec_factory):
    spec = spec_factory(
        statistics={"bootstrap": 200, "seed": 0, "cluster": "template", "cross": "pattern"}
    )
    outcome = run_spec(spec, project, backend=tiny_backend)
    assert outcome.status == "finished", outcome.manifest.get("error")
    stats = outcome.summary["statistics"]
    assert (stats["cluster"], stats["clusters"]) == ("template", 3)
    assert (stats["cross"], stats["cross_clusters"]) == ("pattern", 2)
    assert "crossed over the prompts' template and pattern" in stats["method"]
    assert outcome.summary["sites"][0]["p_holm"] is not None
    assert "Holm-adjusted" in stats["multiple_comparisons"]
    # The effects are the same as with any other bootstrap; only the intervals differ.
    plain = run_spec(
        spec_factory(statistics={"bootstrap": 200, "seed": 0, "cluster": "template"}),
        project,
        backend=tiny_backend,
    )
    assert [s["effect"]["mean"] for s in plain.summary["sites"]] == [
        s["effect"]["mean"] for s in outcome.summary["sites"]
    ]
    assert plain.summary["statistics"]["cross"] is None

    for statistics, message in (
        ({"cluster": None, "cross": "pattern"}, "second field beside statistics.cluster"),
        ({"cluster": "pattern", "cross": "pattern"}, "another field than"),
        ({"cluster": "pattern", "cross": "a b"}, "must name a field"),
    ):
        with pytest.raises(ValueError, match=message):
            spec_factory(statistics={"bootstrap": 200, "seed": 0, **statistics})
    missing = spec_factory(
        statistics={"bootstrap": 200, "seed": 0, "cluster": "template", "cross": "nothing"}
    )
    failed = run_spec(missing, project, backend=tiny_backend)
    assert failed.status == "failed" and "has no 'nothing' in its meta" in failed.manifest["error"]


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
