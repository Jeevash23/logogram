"""Steering: a seeded split, exact where the difference is exact, a fair control, stored per strength."""

from __future__ import annotations

import json

import numpy as np
import pytest

from logogram.datasets import PromptRecord, load_dataset, write_dataset
from logogram.engine import EngineError, run_experiment
from logogram.ioi import NAMES
from logogram.prompts import prepare_prompts
from logogram.results import compute_stats
from logogram.runner import run_spec
from logogram.sites import ScopeError
from logogram.steering import split_pairs


def steering(**changes):
    exp = {
        "kind": "steering",
        "apply_to": "corrupt",
        "coefficients": [0.0, 1.0],
        "train_fraction": 0.5,
        "seed": 0,
        "control": True,
    }
    exp.update(changes)
    return exp


def test_the_split_is_seeded_disjoint_and_complete():
    train, test = split_pairs(12, 0.5, seed=3)
    assert len(train) == 6 and len(test) == 6
    assert sorted(train + test) == list(range(12))
    assert split_pairs(12, 0.5, seed=3) == (train, test)
    assert split_pairs(12, 0.5, seed=4) != (train, test)
    with pytest.raises(EngineError, match="at least one prompt pair"):
        split_pairs(3, 0.5, seed=0)


def _one_token_swap(project):
    """Pairs that differ by the same token at the same position: the clean prompt goes to the
    store, the corrupt one to the park. Their difference at the embeddings is the same for all."""
    records = []
    for i in range(12):
        a, b = NAMES[2 * i], NAMES[2 * i + 1]
        clean = f"When {a} and {b} went to the store, {b} gave a drink to"
        records.append(
            PromptRecord(
                clean=clean,
                corrupt=clean.replace("store", "park"),
                answer=f" {a}",
                distractor=f" {b}",
            )
        )
    write_dataset(project.datasets_dir / "swap.jsonl", records)
    return "datasets/swap.jsonl"


def test_steering_by_the_exact_difference_reproduces_the_other_prompt(
    tiny_backend, project, spec_factory
):
    path = _one_token_swap(project)
    prompts = prepare_prompts(tiny_backend, load_dataset(project.root / path), True)
    place = next(i for i, tok in enumerate(prompts[0].clean.tokens) if tok == " store")
    spec = spec_factory(
        dataset={"path": path},
        experiment=steering(coefficients=[0.0, 1.0, 2.0]),
        scope={
            "kind": "sites",
            "sites": [
                {"kind": "resid_pre", "layer": 0, "position": {"kind": "index", "index": place}}
            ],
        },
    )
    result = run_experiment(spec, tiny_backend, prompts)
    stats = compute_stats(spec, result)
    by_key = {s.variant_key: s.index for s in result.sites}
    # At the embeddings every pair differs by the same vector, so its mean is exact: adding it to a
    # corrupt prompt turns it into its clean prompt, prompt by prompt.
    np.testing.assert_allclose(result.patched[by_key["×1"]], result.baselines.clean, atol=1e-4)
    assert stats.effect_mean[by_key["×1"]] == pytest.approx(1.0, abs=1e-4)
    # Strength 0 changes nothing (up to the rounding of a hooked forward pass, as in the sanity tests).
    np.testing.assert_allclose(stats.effect[by_key["×0"]], 0.0, atol=1e-4)
    np.testing.assert_allclose(stats.effect[by_key["random ×0"]], 0.0, atol=1e-4)
    assert abs(stats.effect_mean[by_key["×2"]] - 1.0) > 1e-3  # twice the step overshoots somewhere
    assert len(result.prompts) == 6
    norms = result.extra["steering"]["norms"]
    assert len(norms) == 1 and norms[0] > 0


def test_the_control_has_the_direction_s_length(tiny_backend, project, spec_factory):
    prompts = prepare_prompts(tiny_backend, load_dataset(project.datasets_dir / "ioi.jsonl"), True)
    spec = spec_factory(
        experiment=steering(coefficients=[1.0]),
        scope={
            "kind": "layer_components",
            "components": ["resid_mid"],
            "position": {"kind": "last"},
        },
    )
    result = run_experiment(spec, tiny_backend, prompts)
    assert [s.variant_key for s in result.sites[:2]] == ["×1", "random ×1"]
    assert {(s.row, s.col) for s in result.sites} == {(r, c) for r in range(2) for c in range(2)}
    assert result.layout["kind"] == "steering" and result.layout["row_title"] == "Layer"
    again = run_experiment(spec, tiny_backend, prompts)
    np.testing.assert_array_equal(result.patched, again.patched)


def test_steering_needs_one_residual_site_at_one_token(tiny_backend, project, spec_factory):
    prompts = prepare_prompts(tiny_backend, load_dataset(project.datasets_dir / "ioi.jsonl"), True)
    for scope, message in (
        ({"kind": "heads", "position": {"kind": "last"}}, "residual site in every layer"),
        (
            {"kind": "layer_components", "components": ["attn_out"], "position": {"kind": "last"}},
            "one residual",
        ),
        (
            {"kind": "layer_components", "components": ["resid_pre"], "position": {"kind": "all"}},
            "one token",
        ),
    ):
        with pytest.raises(ScopeError, match=message):
            run_experiment(spec_factory(experiment=steering(), scope=scope), tiny_backend, prompts)
    with pytest.raises(ValueError, match="must not repeat"):
        spec_factory(experiment=steering(coefficients=[1.0, 1.0]))


def test_a_run_stores_each_strength_and_the_held_out_prompts(tiny_backend, project, spec_factory):
    spec = spec_factory(
        experiment=steering(coefficients=[-1.0, 1.0]),
        scope={
            "kind": "layer_components",
            "components": ["resid_pre"],
            "position": {"kind": "last"},
        },
    )
    outcome = run_spec(spec, project, backend=tiny_backend)
    assert outcome.status == "finished"
    summary = json.loads((outcome.folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["layout"]["kind"] == "steering"
    assert [c["key"] for c in summary["layout"]["cols"]] == ["×−1", "×1", "random ×−1", "random ×1"]
    steer = summary["steering"]
    assert sorted(steer["train"] + steer["test"]) == list(range(12))
    assert summary["n_prompts"] == len(steer["test"])
    assert "steered" in summary["metric"]["normalized_effect"]
    assert summary["sites"][1]["variant"] == {"coefficient": 1.0, "control": False}

    from logogram.runs import site_detail

    detail = site_detail(project, outcome.run_id, 1)
    assert [p["index"] for p in detail["prompts"]] == steer["test"]
    record = load_dataset(project.datasets_dir / "ioi.jsonl")[steer["test"][0]]
    assert detail["prompts"][0]["clean"] == record.clean


def test_a_direction_that_does_no_more_than_its_control_is_reported():
    from types import SimpleNamespace

    from logogram.sites import ResolvedSite
    from logogram.spec import Site
    from logogram.steering import control_comparison, control_warnings

    site = Site(kind="resid_pre", layer=0, position={"kind": "last"})
    variants = [(1.0, False), (1.0, True), (2.0, False), (2.0, True)]
    sites = [
        ResolvedSite(i, site, 0, i, "", variant={"coefficient": c, "control": control})
        for i, (c, control) in enumerate(variants)
    ]
    result = SimpleNamespace(sites=sites)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.05, size=(4, 400))

    def stats(means):
        boot = np.asarray(means)[:, None] + noise
        return SimpleNamespace(effect_boot=boot, effect_mean=boot.mean(1), n=20)

    # The direction moves the prompts about as far as its random control: no effect.
    same = control_comparison(result, stats([0.2, 0.2, 0.3, 0.3]), 0.95)
    assert [c["beats_control"] for c in same] == [False, False]
    (warning,) = control_warnings(same, 0.95)
    assert "random control" in warning and "|direction| − |control|" in warning
    # One strength that clearly moves them further is an effect of the direction.
    apart = control_comparison(result, stats([0.2, 0.2, 0.9, 0.3]), 0.95)
    assert [c["beats_control"] for c in apart] == [False, True]
    assert control_warnings(apart, 0.95) == []
    # A direction that does significantly less than its control doesn't beat it.
    less = control_comparison(result, stats([0.0, 0.6, 0.0, 0.6]), 0.95)
    assert not any(c["beats_control"] for c in less)
    assert len(control_warnings(less, 0.95)) == 1
