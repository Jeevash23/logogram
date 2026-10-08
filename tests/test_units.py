"""Pure-Python pieces: IOI generator, datasets, statistics, specs, comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from logogram.compare import compare_summaries, spec_differences
from logogram.datasets import DatasetError, parse_jsonl, to_jsonl
from logogram.ioi import IOIError, generate_ioi
from logogram.spec import Spec, describe_experiment
from logogram.stats import compute_site_stats, rankdata, resample_counts, spearman

EXAMPLE = Path(__file__).parents[1] / "src" / "logogram" / "examples" / "ioi-gpt2"


# -- IOI ---------------------------------------------------------------------------------------


def test_ioi_is_deterministic_and_balanced():
    a = generate_ioi(40, seed=3)
    b = generate_ioi(40, seed=3)
    assert [r.model_dump() for r in a] == [r.model_dump() for r in b]
    assert [r.model_dump() for r in a] != [r.model_dump() for r in generate_ioi(40, seed=4)]
    patterns = [r.meta["pattern"] for r in a]
    assert patterns.count("ABBA") == patterns.count("BABA") == 20
    assert len({r.clean for r in a}) == 40


def test_ioi_flip_and_abc_corruptions():
    for r in generate_ioi(10, seed=0, corruption="flip"):
        io, s = r.answer.strip(), r.distractor.strip()
        start, end = r.positions["S2"]
        assert r.clean[start:end] == s
        # Only the second mention of the subject changes, and it becomes the IO name.
        assert r.corrupt[:start] == r.clean[:start]
        assert r.corrupt[start : start + len(io)] == io
        assert r.corrupt[start + len(io) :] == r.clean[end:]
    for r in generate_ioi(10, seed=0, corruption="abc"):
        assert r.answer.strip() not in r.corrupt and r.distractor.strip() not in r.corrupt


def test_ioi_spans_name_the_right_words():
    for r in generate_ioi(20, seed=2):
        pos = r.positions
        assert r.clean[slice(*pos["IO"])] == r.answer.strip()
        assert r.clean[slice(*pos["S1"])] == r.distractor.strip()
        assert r.clean[slice(*pos["end"])] == "to"


def test_ioi_rejects_bad_settings():
    with pytest.raises(IOIError):
        generate_ioi(0)
    with pytest.raises(IOIError, match="Unknown template"):
        generate_ioi(5, templates=["nope"])
    with pytest.raises(IOIError, match="splits too many"):
        generate_ioi(5, single_token=lambda w: False)


# -- datasets ----------------------------------------------------------------------------------


def test_jsonl_round_trip():
    records = generate_ioi(5, seed=0)
    again = parse_jsonl(to_jsonl(records))
    assert [r.model_dump() for r in again] == [r.model_dump() for r in records]


@pytest.mark.parametrize(
    "text, message",
    [
        ("{not json}", "line 1: not valid JSON"),
        ('{"clean": "a", "corrupt": "b", "answer": "c"}', "missing 'distractor'"),
        ("[1, 2]", "expected a JSON object"),
        ("", "has no prompts"),
        (
            '{"clean": "ab", "corrupt": "b", "answer": "c", "distractor": "d", "positions": {"X": [0, 9]}}',
            "ends at character 9",
        ),
    ],
)
def test_jsonl_errors_say_where_and_why(text, message):
    with pytest.raises(DatasetError, match=message):
        parse_jsonl(text, source="data.jsonl")


# -- statistics --------------------------------------------------------------------------------


def test_resample_counts_are_stable_and_complete():
    a = resample_counts(7, 50, seed=1)
    assert a.shape == (50, 7) and (a.sum(axis=1) == 7).all()
    assert np.array_equal(a, resample_counts(7, 50, seed=1))
    assert not np.array_equal(a, resample_counts(7, 50, seed=2))


def test_site_stats_ratio_of_means_and_ci():
    rng = np.random.default_rng(0)
    receiver = rng.normal(-2, 0.5, size=40)
    source = rng.normal(2, 0.5, size=40)
    patched = receiver[None, :] + np.array([[0.0], [2.0], [4.0]]) + rng.normal(0, 0.1, (3, 40))
    counts = resample_counts(40, 500, seed=0)
    stats = compute_site_stats(
        patched, patched * 0, receiver, source, receiver * 0, "dataset_gap", counts, 0.95
    )
    gap = (source - receiver).mean()
    expected = (patched - receiver).mean(axis=1) / gap
    np.testing.assert_allclose(stats.effect_mean, expected)
    assert (stats.effect_lo <= stats.effect_mean).all() and (
        stats.effect_mean <= stats.effect_hi
    ).all()
    assert stats.effect_mean[2] == pytest.approx(1.0, abs=0.05)
    assert int(stats.sign_flips[2]) > 30 and int(stats.sign_flips[0]) == 0


def test_rank_correlation():
    assert rankdata(np.array([3.0, 1.0, 2.0, 2.0])).tolist() == [4.0, 1.0, 2.5, 2.5]
    assert spearman(np.arange(10.0), np.arange(10.0) ** 3) == pytest.approx(1.0)
    assert spearman(np.arange(10.0), -np.arange(10.0)) == pytest.approx(-1.0)


# -- specs -------------------------------------------------------------------------------------


def test_example_spec_is_valid_and_pinned():
    spec = Spec.from_path(EXAMPLE / "experiments" / "ioi-head-patching" / "spec.json")
    assert spec.model.revision and spec.dataset.sha256
    assert "every head" in describe_experiment(spec)
    from logogram.datasets import file_sha256

    assert file_sha256(EXAMPLE / spec.dataset.path) == spec.dataset.sha256


def test_spec_rejects_hidden_defaults_and_typos():
    base = json.loads((EXAMPLE / "experiments" / "ioi-head-patching" / "spec.json").read_text())
    with pytest.raises(ValidationError):
        Spec.model_validate({**base, "experiment": {"kind": "activation_patching"}})  # no direction
    with pytest.raises(ValidationError):
        Spec.model_validate({**base, "experiment": {"kind": "ablation"}})  # no baseline
    with pytest.raises(ValidationError):
        Spec.model_validate(
            {**base, "scope": {"kind": "sites", "sites": [{"kind": "head", "layer": 0}]}}
        )
    with pytest.raises(ValidationError):
        Spec.model_validate({**base, "unknown_field": 1})


# -- comparisons -------------------------------------------------------------------------------


def _summary(run_id, effects, his=None):
    sites = []
    for i, e in enumerate(effects):
        hi = his[i] if his else e + 0.1
        sites.append(
            {
                "index": i,
                "kind": "head",
                "layer": i // 3,
                "head": i % 3,
                "position_key": "all",
                "label": f"L{i // 3} H{i % 3}",
                "row": i // 3,
                "col": i % 3,
                "effect": {"mean": e, "lo": e - 0.1, "hi": hi},
            }
        )
    return {
        "run_id": run_id,
        "sites": sites,
        "layout": {"kind": "heads", "rows": [0, 1, 2], "cols": [0, 1, 2]},
    }


def test_compare_flags_sign_and_top_changes():
    a = _summary("a", [0.9, 0.5, 0.3, 0.2, 0.0, -0.1, 0.05, 0.02, 0.01])
    b = _summary("b", [0.9, -0.5, 0.3, 0.2, 0.6, -0.1, 0.05, 0.02, 0.01])
    out = compare_summaries(a, b, top_k=3)
    flags = {c["label"]: c["flags"] for c in out["changes"]}
    assert "sign" in flags["L0 H1"]
    assert "entered_top" in flags["L1 H1"]
    assert out["top_overlap"] == 2
    assert out["same_layout"]
    assert -1 <= out["spearman"] <= 1


def test_spec_differences_ignore_names():
    a = {"name": "x", "experiment": {"kind": "ablation", "baseline": {"kind": "zero"}}}
    b = {"name": "y", "experiment": {"kind": "ablation", "baseline": {"kind": "mean"}}}
    assert spec_differences(a, b) == [
        {"path": "experiment.baseline.kind", "a": "zero", "b": "mean"}
    ]


def test_one_prompt_has_no_confidence_interval():
    counts = resample_counts(1, 50, 0)
    stats = compute_site_stats(
        np.array([[1.0], [2.0]]),
        np.array([[0.5], [0.5]]),
        np.array([0.0]),
        np.array([2.0]),
        np.array([0.1]),
        "dataset_gap",
        counts,
        0.95,
    )
    assert np.isnan(stats.effect_lo).all() and np.isnan(stats.delta_hi).all()
    assert stats.effect_mean.tolist() == [0.5, 1.0]


def test_comparisons_need_real_intervals_and_rank_small_sweeps():
    from logogram.compare import _excludes_zero, default_top_k

    assert not _excludes_zero({"mean": 0.4, "lo": None, "hi": None})
    assert not _excludes_zero({"mean": 0.4, "lo": 0.4, "hi": 0.4})  # degenerate
    assert _excludes_zero({"mean": 0.4, "lo": 0.1, "hi": 0.6})
    assert [default_top_k(n) for n in (1, 4, 5, 6, 14, 15, 49, 50, 500)] == [
        1,
        1,
        1,
        2,
        3,
        3,
        10,
        10,
        10,
    ]


def test_compare_skips_sites_without_a_value():
    a = _summary("a", [0.9, 0.5, 0.3, 0.2])
    b = _summary("b", [0.9, 0.5, 0.3, 0.2])
    b["sites"][1]["effect"]["mean"] = None
    out = compare_summaries(a, b, top_k=2)
    assert out["n_common"] == 3


@pytest.mark.parametrize("label", ["all", "last", "3", "-1"])
def test_position_labels_cant_shadow_built_in_positions(label):
    line = json.dumps(
        {
            "clean": "a b",
            "corrupt": "a c",
            "answer": " x",
            "distractor": " y",
            "positions": {label: [0, 1]},
        }
    )
    with pytest.raises(DatasetError, match="can't be a position label"):
        parse_jsonl(line)


def test_malformed_spans_are_reported_not_crashed():
    line = json.dumps(
        {
            "clean": "a b",
            "corrupt": "a c",
            "answer": " x",
            "distractor": " y",
            "positions": {"IO": 5},
        }
    )
    with pytest.raises(DatasetError, match="line 1"):
        parse_jsonl(line)


def test_layer_profile_keeps_each_layers_strongest_signed_effect():
    from logogram.project import layer_profile

    summary = {
        "model": {"n_layers": 3},
        "sites": [
            {"layer": 0, "effect": {"mean": 0.2}},
            {"layer": 0, "effect": {"mean": -0.5}},
            {"layer": 2, "effect": {"mean": 0.3}},
            {"layer": 2, "effect": {"mean": None}},
        ],
    }
    assert layer_profile(summary) == [-0.5, 0.0, 0.3]
    assert layer_profile({"sites": []}) is None
