"""The real-weight validation script runs every method, and its identities hold on any model.

The script is meant for real models (the release check, and Apple's MPS); here it runs on the tiny
random model, where the identities must hold but agreement expected of a trained model needn't.
Its checks of GPT-2 small's published circuit (--golden) are tested on summaries made up here.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "validate_real_weights", ROOT / "scripts" / "validate_real_weights.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_real_weights"] = module
    spec.loader.exec_module(module)
    return module


def test_every_method_runs_and_the_identities_hold(tiny_backend, project):
    lines: list[str] = []
    checks = _load().validate(
        tiny_backend, project, "datasets/ioi.jsonl", prepend_bos=True, log=lines.append
    )
    names = {c.name for c in checks}
    for expected in (
        "Patching the token that differs restores everything",
        "Patching the final residual stream restores everything",
        "The same spec twice gives bit-identical results",
        "Direct effects add up to the logit difference",
        "From the last layer, a path to the logits is the whole effect",
        "The logit lens at the last layer is the model's own prediction",
        "Steering runs",
        "Path patching into heads runs",
    ):
        assert expected in names, "\n".join(lines)
    failed = [c for c in checks if c.exact and c.passed is False]
    assert not failed, "\n".join(lines)
    assert not [c for c in checks if c.name.startswith("IOI circuit")]


def test_golden_only_runs_what_the_circuit_checks_need(tiny_backend, project):
    """The tiny model isn't GPT-2 small, so the circuit checks fail; nothing else runs."""
    lines: list[str] = []
    checks = _load().validate(
        tiny_backend,
        project,
        "datasets/ioi.jsonl",
        prepend_bos=True,
        golden=True,
        golden_only=True,
        log=lines.append,
    )
    assert [c.name for c in checks] == [
        "The model reproduces itself when it loads",
        "IOI circuit: the run is a sweep of GPT-2 small's heads",
    ], "\n".join(lines)
    assert checks[-1].passed is False and "got 8 heads" in checks[-1].detail
    assert any(line.startswith("FAIL  IOI circuit") for line in lines)
    runs = [line.split(":")[0].strip() for line in lines if line.startswith("      ")]
    assert runs == ["Patching every head", "Direct effects of every head"]


# Made up, but like GPT-2 small on the bundled example: the heads of the published circuit stand
# out from many small ones. Patching 9.6 alone lowers the logit difference, as in GPT-2 small.
PATCHED = {
    (5, 5): 0.32,
    (8, 6): 0.30,
    (8, 10): 0.25,
    (7, 9): 0.21,
    (9, 9): 0.17,
    (7, 3): 0.12,
    (3, 0): 0.12,
    (10, 0): 0.11,
    (6, 9): 0.10,
    (9, 7): 0.09,
    (9, 6): -0.09,
    (11, 2): -0.15,
    (11, 10): -0.21,
    (10, 7): -0.45,
}
DIRECT = {
    (9, 9): 0.80,
    (9, 6): 0.36,
    (10, 0): 0.23,
    (10, 10): 0.12,
    (10, 1): 0.10,
    (11, 2): -0.13,
    (11, 10): -0.21,
    (10, 7): -0.49,
}
HeadMap = Callable[[int, int], tuple[int, int]]


def _summary(
    effects: dict[tuple[int, int], float],
    *,
    position: str = "all",
    measure: str = "intervention",
    receiver: str = "corrupt",
    clean: float = 2.9,
    corrupt: float = -3.0,
) -> dict[str, Any]:
    """A summary of a run over every head of a model with 12 layers of 12 heads."""
    sites = []
    for layer in range(12):
        for head in range(12):
            mean = effects.get((layer, head), 0.005 * ((layer * 7 + head * 5) % 5 - 2))
            sites.append(
                {
                    "kind": "head",
                    "layer": layer,
                    "head": head,
                    "position_key": position,
                    "label": f"L{layer} H{head}",
                    "effect": {"mean": mean, "sd": 0.1, "lo": mean - 0.02, "hi": mean + 0.02},
                }
            )
    return {
        "n_prompts": 32,
        "n_sites": len(sites),
        "receiver": receiver,
        "reference": "clean",
        "measure": measure,
        "metric": {"kind": "logit_diff", "normalization": "dataset_gap"},
        "baseline": {
            "clean": {"logit_diff": {"mean": clean, "sd": 2.0}, "prefers_answer": 29},
            "corrupt": {"logit_diff": {"mean": corrupt, "sd": 1.7}, "prefers_answer": 1},
        },
        "sites": sites,
    }


def _golden(
    patched: dict[tuple[int, int], float] = PATCHED,
    direct: dict[tuple[int, int], float] | None = DIRECT,
    **baseline: float,
):
    sweep = _summary(patched, **baseline)
    by_head = None
    if direct is not None:
        by_head = _summary(direct, position="last", measure="attribution", receiver="clean")
    return _load().golden_checks(sweep, by_head)


def _failed(checks) -> list[str]:
    return [c.name.removeprefix("IOI circuit: ") for c in checks if c.passed is False]


NAME_MOVERS_PATCHED = "patching name movers 9.9 and 10.0 restores the answer"
NAME_MOVERS_DIRECT = "name movers 9.9, 9.6 and 10.0 write the answer most directly"
NEGATIVE_PATCHED = "patching negative name movers 10.7 and 11.10 works against it"
NEGATIVE_DIRECT = "negative name movers 10.7 and 11.10 write against it most directly"
S_INHIBITION = "patching S-inhibition heads 7.3, 7.9, 8.6 and 8.10 restores it"
INDUCTION = "patching induction heads 5.5 and 6.9 restores it"
SCALE = "effects are fractions of the gap between clean and corrupt"
BASELINE = "GPT-2 small prefers the indirect object by its usual margin"


def test_the_published_circuit_passes():
    checks = _golden()
    assert [c.passed for c in checks] == [True] * 8, [f"{c.name}: {c.detail}" for c in checks]
    assert all(c.name.startswith("IOI circuit: ") and not c.exact for c in checks)


def test_a_sign_error_fails():
    flip = {k: -v for k, v in PATCHED.items()}
    assert set(_failed(_golden(flip, {k: -v for k, v in DIRECT.items()}))) >= {
        NAME_MOVERS_PATCHED,
        NAME_MOVERS_DIRECT,
        NEGATIVE_PATCHED,
        NEGATIVE_DIRECT,
        S_INHIBITION,
        INDUCTION,
    }


@pytest.mark.parametrize(
    "mapping",
    [
        lambda layer, head: (layer, 11 - head),  # heads counted from the other end
        lambda layer, head: (head, layer),  # layer and head swapped
        lambda layer, head: (layer - 1, head),  # off by one layer
    ],
)
def test_heads_mapped_to_the_wrong_index_fail(mapping: HeadMap):
    patched = {mapping(*k): v for k, v in PATCHED.items()}
    direct = {mapping(*k): v for k, v in DIRECT.items()}
    failed = _failed(_golden(patched, direct))
    assert NAME_MOVERS_PATCHED in failed and NAME_MOVERS_DIRECT in failed


@pytest.mark.parametrize("scale", [5.8, 0.1])
def test_a_broken_normalization_fails(scale: float):
    """Effects in logits rather than shares of the gap, or divided by too much."""
    assert SCALE in _failed(_golden({k: v * scale for k, v in PATCHED.items()}))


def test_leaving_out_the_earlier_positions_fails():
    """Patched at the end alone, the heads that act at the second name do nothing."""
    end_only = {k: (0.0 if k in {(5, 5), (6, 9), (3, 0)} else v) for k, v in PATCHED.items()}
    assert _failed(_golden(end_only)) == [INDUCTION]


@pytest.mark.parametrize(("clean", "corrupt"), [(0.4, -0.3), (9.0, -3.0), (2.9, 1.0)])
def test_an_implausible_baseline_fails(clean: float, corrupt: float):
    assert _failed(_golden(clean=clean, corrupt=corrupt)) == [BASELINE]


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.update(sites=s["sites"][:100]),
        lambda s: s.update(reference="corrupt", receiver="clean"),
        lambda s: s.update(measure="estimate"),
        lambda s: [site.update(position_key="last") for site in s["sites"]],
    ],
)
def test_a_different_sweep_is_refused(change: Callable[[dict[str, Any]], Any]):
    sweep = _summary(PATCHED)
    change(sweep)
    by_head = _summary(DIRECT, position="last", measure="attribution", receiver="clean")
    checks = _load().golden_checks(sweep, by_head)
    assert _failed(checks) == ["the run is a sweep of GPT-2 small's heads"]


def test_direct_effects_that_are_missing_are_skipped_and_wrong_ones_fail():
    skipped = _golden(direct=None)
    assert [c.passed for c in skipped][-2:] == [None, None]
    assert not _failed(skipped)
    module = _load()
    by_position = _summary(DIRECT, position="all", measure="attribution", receiver="clean")
    assert _failed(module.golden_checks(_summary(PATCHED), by_position)) == [
        NAME_MOVERS_DIRECT,
        NEGATIVE_DIRECT,
    ]


# -- the gallery's other examples, checked by --golden too ---------------------------------------

# Made up, but like GPT-2 small on the bundled examples.
GT_MLP = {8: 0.195, 9: 0.359, 10: 0.331, 11: 0.087, 5: 0.007}
GT_ATTN = {5: 0.044, 6: 0.027, 7: 0.074, 8: 0.074, 9: 0.164, 10: 0.028, 11: 0.005}
CAPITAL_COUNTRY = [0.94, 0.95, 0.95, 0.94, 0.94, 0.93, 0.93, 0.90, 0.89, 0.82, 0.46, 0.17]
CAPITAL_END = [0.0, 0.003, 0.007, 0.013, 0.015, 0.015, 0.022, 0.052, 0.055, 0.13, 0.48, 0.78]
PRONOUN = {(10, 9): 0.345, (9, 7): 0.224, (11, 8): 0.097, (11, 7): 0.059, (9, 5): -0.023}


def _baseline(clean: float, corrupt: float, n: int) -> dict[str, Any]:
    return {
        "clean": {
            "logit_diff": {"mean": 1.0, "sd": 1.0},
            "metric": {"mean": clean, "sd": 0.2},
            "prefers_answer": n - 2,
        },
        "corrupt": {
            "logit_diff": {"mean": -1.0, "sd": 1.0},
            "metric": {"mean": corrupt, "sd": 0.2},
            "prefers_answer": 3,
        },
    }


def _site(kind: str, layer: int, position: str, mean: float) -> dict[str, Any]:
    return {
        "kind": kind,
        "layer": layer,
        "head": None,
        "position_key": position,
        "effect": {"mean": mean, "sd": 0.1, "lo": mean - 0.02, "hi": mean + 0.02},
    }


def _greater_than(
    mlp: dict[int, float] = GT_MLP,
    attn: dict[int, float] = GT_ATTN,
    *,
    receiver: str = "clean",
) -> dict[str, Any]:
    sites = [
        _site(kind, layer, "last", values.get(layer, 0.001 * ((layer * 3) % 5 - 2)))
        for layer in range(12)
        for kind, values in (("attn_out", attn), ("mlp_out", mlp))
    ]
    return {
        "n_prompts": 96,
        "receiver": receiver,
        "reference": "corrupt" if receiver == "clean" else "clean",
        "measure": "intervention",
        "metric": {"kind": "prob_diff", "normalization": "dataset_gap"},
        "baseline": _baseline(0.8, -0.57, 96),
        "sites": sites,
    }


def _capital(
    country: list[float] = CAPITAL_COUNTRY,
    end: list[float] = CAPITAL_END,
    *,
    labels: tuple[str, str] = ("country", "end"),
) -> dict[str, Any]:
    sites = [
        _site("resid_pre", layer, label, values[layer])
        for layer in range(12)
        for label, values in zip(labels, (country, end), strict=True)
    ]
    return {
        "n_prompts": 96,
        "receiver": "corrupt",
        "reference": "clean",
        "measure": "intervention",
        "metric": {"kind": "logit_diff", "normalization": "dataset_gap"},
        "baseline": _baseline(5.8, -5.7, 96),
        "sites": sites,
    }


def _pronoun(direct: dict[tuple[int, int], float] = PRONOUN, **kw: Any) -> dict[str, Any]:
    summary = _summary(
        direct, position="last", receiver="clean", **{"measure": "attribution", **kw}
    )
    summary["baseline"] = _baseline(4.3, -4.4, 32)
    return summary


def _names(checks: list[Any]) -> list[str]:
    return [c.name.split(": ", 1)[1] for c in checks if c.passed is False]


def test_the_examples_published_findings_pass():
    module = _load()
    for checks, prefix in (
        (module.greater_than_checks(_greater_than()), "Greater-than: "),
        (module.factual_recall_checks(_capital()), "Factual recall: "),
        (module.pronoun_checks(_pronoun()), "Gendered pronoun: "),
    ):
        assert checks and all(c.passed for c in checks), [f"{c.name}: {c.detail}" for c in checks]
        assert all(c.name.startswith(prefix) and not c.exact for c in checks)
    assert set(module.GALLERY_CHECKS) == {
        "greater-than-gpt2",
        "factual-recall-gpt2",
        "gendered-pronoun-gpt2",
    }


def test_the_examples_catch_a_sign_error():
    module = _load()
    flipped = _greater_than({k: -v for k, v in GT_MLP.items()}, {k: -v for k, v in GT_ATTN.items()})
    assert set(_names(module.greater_than_checks(flipped))) >= {
        "MLPs 8 to 11 carry the largest effects",
        "MLPs 9 and 10 the most",
        "attention in layers 5 to 9 brings the start year",
    }
    capital = _capital([-v for v in CAPITAL_COUNTRY], [-v for v in CAPITAL_END])
    assert set(_names(module.factual_recall_checks(capital))) >= {
        "the country's last token restores the capital through the early layers",
        "the last token takes over in the last layers",
    }
    pronoun = _pronoun({k: -v for k, v in PRONOUN.items()})
    assert _names(module.pronoun_checks(pronoun)) == [
        "heads 10.9 and 9.7 write the pronoun most directly"
    ]


@pytest.mark.parametrize(
    "moved",
    [lambda layer: 11 - layer, lambda layer: (layer - 1) % 12, lambda layer: (layer + 1) % 12],
)
def test_the_examples_catch_layers_out_of_place(moved: Callable[[int], int]):
    module = _load()
    gt = _greater_than(
        {moved(k): v for k, v in GT_MLP.items()}, {moved(k): v for k, v in GT_ATTN.items()}
    )
    assert "MLPs 8 to 11 carry the largest effects" in _names(module.greater_than_checks(gt))
    country, end = [0.0] * 12, [0.0] * 12
    for layer in range(12):
        country[moved(layer)], end[moved(layer)] = CAPITAL_COUNTRY[layer], CAPITAL_END[layer]
    assert _names(module.factual_recall_checks(_capital(country, end)))
    pronoun = _pronoun({(moved(a), b): v for (a, b), v in PRONOUN.items()})
    assert _names(module.pronoun_checks(pronoun))


@pytest.mark.parametrize("scale", [5.8, 0.1])
def test_the_examples_catch_a_broken_normalization(scale: float):
    module = _load()
    gt = _greater_than(
        {k: v * scale for k, v in GT_MLP.items()}, {k: v * scale for k, v in GT_ATTN.items()}
    )
    assert _names(module.greater_than_checks(gt))
    capital = _capital([v * scale for v in CAPITAL_COUNTRY], [v * scale for v in CAPITAL_END])
    assert _names(module.factual_recall_checks(capital))
    assert _names(module.pronoun_checks(_pronoun({k: v * scale for k, v in PRONOUN.items()})))


def test_the_examples_refuse_a_different_experiment():
    module = _load()
    assert _names(module.greater_than_checks(_greater_than(receiver="corrupt"))) == [
        "the run patches attention and MLP outputs at the last token"
    ]
    assert _names(module.factual_recall_checks(_capital(labels=("end", "country"))))
    assert _names(module.factual_recall_checks(_capital(labels=("S2", "end")))) == [
        "the run traces the residual stream at the country and at the last token"
    ]
    assert _names(module.pronoun_checks(_pronoun(measure="intervention"))) == [
        "the run splits the clean logit difference between GPT-2 small's heads"
    ]


def test_an_example_never_runs_on_a_model_it_doesnt_pin(tiny_backend, tmp_path: Path):
    """The examples pin GPT-2 small: on another model they fail without loading anything."""
    lines: list[str] = []
    checks = _load().validate_examples(tiny_backend, tmp_path, log=lines.append)
    assert [c.name for c in checks] == [
        "The greater-than-gpt2 example runs as it ships",
        "The factual-recall-gpt2 example runs as it ships",
        "The gendered-pronoun-gpt2 example runs as it ships",
    ]
    assert all(c.passed is False and "the model loaded is tiny-gpt2" in c.detail for c in checks)
    assert not list(tmp_path.iterdir())  # nothing was copied or run
