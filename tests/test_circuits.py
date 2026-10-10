"""Sets of sites intervened on at once: identities with single-site runs, complements, and the
share and faithfulness readouts."""

from __future__ import annotations

import numpy as np
import pytest

from logogram.datasets import load_dataset, write_dataset
from logogram.engine import EngineError, run_engine, run_experiment
from logogram.prompts import prepare_prompts
from logogram.results import build_summary, compute_stats
from logogram.runner import run_spec
from logogram.sites import ScopeError
from logogram.spec import Spec

ALL = {"kind": "all"}
TOL = 1e-5


def _prompts(backend, project, name="ioi.jsonl", continuations=False):
    return prepare_prompts(
        backend, load_dataset(project.datasets_dir / name), True, continuations=continuations
    )


def _head(layer, head, position=None):
    return {"kind": "head", "layer": layer, "head": head, "position": position or ALL}


def _sets(*sets, universe=None):
    return {"kind": "site_sets", "universe": universe, "sets": list(sets)}


def _set(label, sites, complement=False):
    return {"label": label, "sites": sites, "complement": complement}


@pytest.mark.parametrize(
    "experiment",
    [
        {"kind": "activation_patching", "direction": "clean_to_corrupt"},
        {"kind": "ablation", "baseline": {"kind": "zero"}},
        {"kind": "ablation", "baseline": {"kind": "mean", "reference": "corrupt"}},
        {
            "kind": "ablation",
            "baseline": {"kind": "resample", "pool": "corrupt", "donors": 2, "seed": 3},
        },
    ],
)
def test_a_set_of_one_site_is_that_site(tiny_backend, project, spec_factory, experiment):
    """A set holding one head (or one MLP output at one position) changes the run exactly as
    patching that site alone does."""
    prompts = _prompts(tiny_backend, project)
    sites = [
        _head(1, 2),
        {"kind": "mlp_out", "layer": 0, "position": {"kind": "label", "label": "S2"}},
    ]
    for site in sites:
        alone = run_engine(
            spec_factory(experiment=experiment, scope={"kind": "sites", "sites": [site]}),
            tiny_backend,
            prompts,
        )
        mean = experiment["kind"] == "ablation" and experiment["baseline"]["kind"] == "mean"
        if mean and site["position"] != ALL:
            continue  # sets take means per position; one-position ablation takes them per prompt
        as_set = run_experiment(
            spec_factory(experiment=experiment, scope=_sets(_set("one", [site]))),
            tiny_backend,
            prompts,
        )
        np.testing.assert_allclose(as_set.patched, alone.patched, atol=TOL)


def test_patching_the_whole_stream_as_a_set_restores_everything(
    tiny_backend, project, spec_factory
):
    spec = spec_factory(
        scope=_sets(_set("stream", [{"kind": "resid_pre", "layer": 1, "position": ALL}]))
    )
    result = run_experiment(spec, tiny_backend, _prompts(tiny_backend, project))
    assert compute_stats(spec, result).effect_mean[0] == pytest.approx(1.0, abs=1e-4)


def test_heads_at_once_differ_from_heads_one_by_one(tiny_backend, project, spec_factory):
    """Patching two heads together isn't the sum of patching each: interactions are why sets are
    intervened on at once."""
    prompts = _prompts(tiny_backend, project)
    heads = [_head(0, 1), _head(1, 2)]
    together = run_experiment(spec_factory(scope=_sets(_set("both", heads))), tiny_backend, prompts)
    alone = run_engine(spec_factory(scope={"kind": "sites", "sites": heads}), tiny_backend, prompts)
    summed = alone.patched.sum(0) - alone.receiver_metric
    assert np.abs(together.patched[0] - summed).max() > 1e-3


def test_complements_and_shares(tiny_backend, project, spec_factory):
    """A complement set replaces the universe except its sites; listing every head explicitly is
    the same as replacing everything, so its share of everything is exactly 1."""
    info = tiny_backend.info
    every_head = [_head(layer, h) for layer in range(info.n_layers) for h in range(info.n_heads)]
    circuit = [_head(0, 1), _head(1, 2, {"kind": "last"})]
    spec = spec_factory(
        experiment={"kind": "ablation", "baseline": {"kind": "mean", "reference": "corrupt"}},
        scope=_sets(
            _set("everything", [], complement=True),
            _set("every head, listed", every_head),
            _set("circuit only", circuit, complement=True),
            _set("circuit removed", circuit),
            universe=["head"],
        ),
    )
    prompts = _prompts(tiny_backend, project)
    result = run_experiment(spec, tiny_backend, prompts)
    np.testing.assert_allclose(result.patched[1], result.patched[0], atol=TOL)
    stats = compute_stats(spec, result)
    summary = build_summary(spec, result, stats, "run", None)
    rows = {r["label"]: r for r in summary["circuit"]["rows"]}
    assert summary["circuit"]["everything"] == 0
    assert rows["every head, listed"]["share"]["mean"] == pytest.approx(1.0, abs=1e-6)
    assert rows["everything"]["share"] is None
    keep = rows["circuit only"]
    assert keep["faithfulness"]["mean"] == pytest.approx(1 - keep["share"]["mean"])
    assert keep["faithfulness"]["lo"] <= keep["faithfulness"]["mean"] <= keep["faithfulness"]["hi"]
    assert rows["circuit removed"]["faithfulness"] is None  # it keeps nothing
    assert summary["layout"]["kind"] == "site_sets"
    site = summary["sites"][2]
    assert site["kind"] == "site_set" and site["label"] == "circuit only"
    assert len(site["members"]) == 2


def test_a_kept_position_is_only_that_position(tiny_backend, project, spec_factory):
    """Keeping a head at the last token replaces its other positions with the rest. (A kept
    second-layer head carries the other positions to the last token's prediction.)"""
    prompts = _prompts(tiny_backend, project)
    reader = _head(1, 0)
    keep_last = _set("keep last", [_head(0, 2, {"kind": "last"}), reader], complement=True)
    keep_all = _set("keep all", [_head(0, 2), reader], complement=True)
    spec = spec_factory(
        experiment={"kind": "ablation", "baseline": {"kind": "zero"}},
        scope=_sets(keep_last, keep_all, universe=["head"]),
    )
    result = run_experiment(spec, tiny_backend, prompts)
    assert np.abs(result.patched[0] - result.patched[1]).max() > 1e-4


def test_continuations_are_covered_by_sets(tiny_backend, project, spec_factory):
    records = [
        r.model_copy(update={"answer": r.answer + r.answer})
        for r in load_dataset(project.datasets_dir / "ioi.jsonl")
    ]
    write_dataset(project.datasets_dir / "twice.jsonl", records)
    prompts = _prompts(tiny_backend, project, "twice.jsonl", continuations=True)
    spec = spec_factory(
        dataset={"path": "datasets/twice.jsonl"},
        metric={"kind": "logprob", "normalization": "dataset_gap"},
        scope=_sets(_set("stream", [{"kind": "resid_pre", "layer": 1, "position": ALL}])),
    )
    result = run_experiment(spec, tiny_backend, prompts)
    np.testing.assert_allclose(result.patched[0], result.reference_metric, atol=1e-4)
    mean = spec_factory(
        dataset={"path": "datasets/twice.jsonl"},
        metric={"kind": "logprob", "normalization": "dataset_gap"},
        experiment={"kind": "ablation", "baseline": {"kind": "mean", "reference": "corrupt"}},
        scope=_sets(_set("stream", [{"kind": "resid_pre", "layer": 1, "position": ALL}])),
    )
    with pytest.raises(EngineError, match="no such mean"):
        run_experiment(mean, tiny_backend, prompts)


def test_sets_are_checked(tiny_backend, project, spec_factory):
    prompts = _prompts(tiny_backend, project)
    with pytest.raises(ScopeError, match="layers"):
        run_experiment(spec_factory(scope=_sets(_set("x", [_head(9, 0)]))), tiny_backend, prompts)
    with pytest.raises(ScopeError, match="sum of the heads"):
        run_experiment(
            spec_factory(
                scope=_sets(
                    _set("x", [_head(0, 0), {"kind": "attn_out", "layer": 0, "position": ALL}])
                )
            ),
            tiny_backend,
            prompts,
        )
    with pytest.raises(ScopeError, match="activation patching and ablation"):
        run_experiment(
            spec_factory(
                experiment={"kind": "direct_logit_attribution", "prompts": "clean"},
                scope=_sets(_set("x", [_head(0, 0)])),
            ),
            tiny_backend,
            prompts,
        )
    for bad, match in [
        (_sets(_set("x", [], complement=True)), "universe"),
        (_sets(_set("x", [])), "no sites"),
        (_sets(_set("x", [_head(0, 0)]), _set("x", [_head(0, 1)])), "repeat"),
        (_sets(_set("x", [_head(0, 0)], complement=True), universe=["mlp_out"]), "universe"),
        (_sets(_set("x", [_head(0, 0)]), universe=["head", "attn_out"]), "sum of its heads"),
    ]:
        with pytest.raises(ValueError, match=match):
            spec_factory(scope=bad)


def test_a_circuit_run_is_saved_and_listed(tiny_backend, project, spec_factory):
    spec = spec_factory(
        experiment={"kind": "ablation", "baseline": {"kind": "zero"}},
        scope=_sets(
            _set("everything", [], complement=True),
            _set("circuit only", [_head(1, 0)], complement=True),
            universe=["head", "mlp_out"],
        ),
    )
    outcome = run_spec(spec, project, backend=tiny_backend)
    assert outcome.status == "finished", outcome.manifest.get("error")
    saved = Spec.from_path(outcome.folder / "spec.json")
    assert saved.scope.kind == "site_sets"
    assert outcome.summary["circuit"]["rows"][1]["faithfulness"] is not None
    listing = project.run_listing(outcome.run_id)
    assert listing.status == "finished" and listing.layout_kind == "site_sets"
    assert "2 sets of sites" in listing.description
