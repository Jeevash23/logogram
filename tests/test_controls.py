"""Controls a result is read against: the gate a baseline must pass, donors drawn within or
across a group of prompts, each site of a set alone beside the set, and random sets of the same
size beside it."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from logogram.circuits import random_sets, set_rows, single_sets
from logogram.datasets import load_dataset
from logogram.engine import EngineError, GateError, draw_donors, run_engine, run_experiment
from logogram.prompts import group_by_length, prepare_prompts
from logogram.results import build_summary, compute_stats
from logogram.runner import run_spec
from logogram.sites import ScopeError, site_unit
from logogram.spec import DonorGroup, SiteSetsScope, Spec, describe_experiment

ALL = {"kind": "all"}


def _prompts(backend, project, name="ioi.jsonl"):
    return prepare_prompts(backend, load_dataset(project.datasets_dir / name), True)


def _head(layer, head, position=None):
    return {"kind": "head", "layer": layer, "head": head, "position": position or ALL}


def _sets(*sets, universe=None, singles=False, controls=None):
    return {
        "kind": "site_sets",
        "universe": universe,
        "sets": list(sets),
        "singles": singles,
        "controls": controls,
    }


def _set(label, sites, complement=False):
    return {"label": label, "sites": sites, "complement": complement}


# -- the gate ------------------------------------------------------------------------------------


def test_a_gate_the_baseline_passes_changes_nothing(tiny_backend, project, spec_factory):
    """A gate is checked before anything is intervened on: a run that passes it gives the numbers
    it gives without one."""
    prompts = _prompts(tiny_backend, project)
    plain = run_engine(spec_factory(), tiny_backend, prompts)
    gap = float(np.abs((plain.baselines.clean - plain.baselines.corrupt).mean()))
    share = float((plain.baselines.clean_pref > 0).mean())
    gated = spec_factory(gate={"min_correct": max(share, 0.01), "min_gap": gap * 0.99})
    result = run_engine(gated, tiny_backend, prompts)
    np.testing.assert_array_equal(result.patched, plain.patched)
    assert "gate: " in describe_experiment(gated)


def test_a_gate_the_baseline_fails_stops_the_run(tiny_backend, project, spec_factory):
    prompts = _prompts(tiny_backend, project)
    plain = run_engine(spec_factory(), tiny_backend, prompts)
    gap = float(np.abs((plain.baselines.clean - plain.baselines.corrupt).mean()))
    correct = int((plain.baselines.clean_pref > 0).sum())
    assert correct < len(prompts)  # the tiny random model doesn't solve every prompt

    with pytest.raises(GateError, match=f"prefers the answer on {correct} of the 12 clean"):
        run_engine(spec_factory(gate={"min_correct": 1.0, "min_gap": None}), tiny_backend, prompts)
    with pytest.raises(GateError, match="mean clean–corrupt gap"):
        run_engine(
            spec_factory(gate={"min_correct": None, "min_gap": gap * 2 + 1}), tiny_backend, prompts
        )
    # Both bars, both short: one message names both.
    with pytest.raises(GateError) as both:
        run_engine(
            spec_factory(gate={"min_correct": 1.0, "min_gap": gap * 2 + 1}), tiny_backend, prompts
        )
    assert "prefers the answer" in str(both.value) and "gap" in str(both.value)

    # A whole run ends as failed, saying what it measured, and saves no results.
    outcome = run_spec(
        spec_factory(gate={"min_correct": 1.0, "min_gap": None}), project, backend=tiny_backend
    )
    assert outcome.status == "failed"
    assert (
        "The gate stopped this run before anything was intervened on" in outcome.manifest["error"]
    )


def test_a_gate_stops_sets_of_sites_too(tiny_backend, project, spec_factory):
    spec = spec_factory(
        scope=_sets(_set("a", [_head(0, 1)])), gate={"min_correct": 1.0, "min_gap": None}
    )
    with pytest.raises(GateError):
        run_experiment(spec, tiny_backend, _prompts(tiny_backend, project))


def test_gates_are_checked(spec_factory):
    with pytest.raises(ValidationError, match="min_correct or min_gap"):
        spec_factory(gate={"min_correct": None, "min_gap": None})
    with pytest.raises(ValidationError):
        spec_factory(gate={"min_correct": 1.5, "min_gap": None})
    with pytest.raises(ValidationError):
        spec_factory(gate={"min_correct": None, "min_gap": 0})
    # A version 4 spec has no gate.
    old = spec_factory().model_dump(mode="json")
    old["logogram_spec"] = 4
    del old["gate"]
    del old["statistics"]["cross"]
    spec = Spec.model_validate(old)
    assert spec.logogram_spec == 5 and spec.gate is None and spec.statistics.cross is None
    assert spec.upgraded_fields == []


# -- donors drawn within or across a group --------------------------------------------------------


def _resample(group=None, donors=2, seed=3):
    baseline = {"kind": "resample", "pool": "corrupt", "donors": donors, "seed": seed}
    return {"kind": "ablation", "baseline": {**baseline, "group": group}}


def test_donors_are_drawn_within_or_across_a_group(tiny_backend, project):
    prompts = _prompts(tiny_backend, project)
    groups = group_by_length(prompts, "clean")
    pattern = [p.record.meta["pattern"] for p in prompts]

    plain = draw_donors(prompts, groups, 2, 3, same_length=False)
    assert plain == draw_donors(prompts, groups, 2, 3, same_length=False, among=None)

    same = draw_donors(prompts, groups, 2, 3, False, DonorGroup(field="pattern", match="same"))
    other = draw_donors(
        prompts, groups, 2, 3, False, DonorGroup(field="pattern", match="different")
    )
    for i, (within, across) in enumerate(zip(same, other, strict=True)):
        assert len(set(within)) == 2 and i not in within
        assert all(pattern[d] == pattern[i] for d in within)
        assert all(pattern[d] != pattern[i] for d in across)

    # Too few prompts share the value: the draw says so.
    with pytest.raises(EngineError, match="with the same template"):
        draw_donors(prompts, groups, 5, 3, False, DonorGroup(field="template", match="same"))
    with pytest.raises(EngineError, match="has no 'nothing' in its meta"):
        draw_donors(prompts, groups, 1, 3, False, DonorGroup(field="nothing", match="same"))


def test_a_run_resamples_within_a_group(tiny_backend, project, spec_factory):
    spec = spec_factory(
        experiment=_resample({"field": "pattern", "match": "same"}),
        scope={"kind": "heads", "position": {"kind": "last"}},
    )
    assert "donors with the same pattern" in describe_experiment(spec)
    outcome = run_spec(spec, project, backend=tiny_backend)
    assert outcome.status == "finished", outcome.manifest.get("error")
    prompts = _prompts(tiny_backend, project)
    for i, donors in enumerate(outcome.summary["donors"]):
        assert all(
            prompts[d].record.meta["pattern"] == prompts[i].record.meta["pattern"] for d in donors
        )
    # The group changes which donors are drawn, and so the numbers.
    plain = run_spec(
        spec_factory(
            experiment=_resample(None), scope={"kind": "heads", "position": {"kind": "last"}}
        ),
        project,
        backend=tiny_backend,
    )
    assert plain.summary["donors"] != outcome.summary["donors"]

    # Sets of sites draw their donors the same way.
    sets = spec_factory(
        experiment=_resample({"field": "pattern", "match": "different"}),
        scope=_sets(_set("a", [_head(0, 1)])),
    )
    result = run_experiment(sets, tiny_backend, prompts)
    assert result.donors is not None
    for i, donors in enumerate(result.donors):
        assert all(
            prompts[d].record.meta["pattern"] != prompts[i].record.meta["pattern"] for d in donors
        )


# -- each site alone ------------------------------------------------------------------------------


def test_singles_add_each_site_alone_and_say_what_the_set_adds(tiny_backend, project, spec_factory):
    a, b, c = _head(0, 1), _head(1, 2), _head(1, 0)
    patching = {"kind": "activation_patching", "direction": "clean_to_corrupt"}
    prompts = _prompts(tiny_backend, project)
    spec = spec_factory(
        experiment=patching,
        scope=_sets(_set("three", [a, b, c]), _set("L0 H1", [a]), singles=True),
    )
    assert isinstance(spec.scope, SiteSetsScope)
    # The site the scope already has alone isn't added again.
    assert [s.label for s in single_sets(spec.scope)] == ["L1 H2", "L1 H0"]
    result = run_experiment(spec, tiny_backend, prompts)
    assert [s.label for s in result.sites] == ["three", "L0 H1", "L1 H2", "L1 H0"]
    stats = compute_stats(spec, result)
    summary = build_summary(spec, result, stats, "run", None)
    rows = {r["label"]: r for r in summary["circuit"]["rows"]}
    assert [rows[k]["single"] for k in ("three", "L0 H1", "L1 H2", "L1 H0")] == [
        False,
        False,
        True,
        True,
    ]
    both = rows["three"]["interaction"]
    assert set(both["parts"]) == {"L0 H1", "L1 H2", "L1 H0"} and both["b"] is None
    e = stats.effect_mean
    assert both["sum"]["mean"] == pytest.approx(e[1] + e[2] + e[3], abs=1e-9)
    assert both["effect"]["mean"] == pytest.approx(e[0] - e[1] - e[2] - e[3], abs=1e-9)
    assert both["effect"]["lo"] <= both["effect"]["mean"] <= both["effect"]["hi"]

    # The set's own numbers are those of a run that lists every set itself.
    listed = spec_factory(
        experiment=patching,
        scope=_sets(_set("three", [a, b, c]), _set("x", [a]), _set("y", [b]), _set("z", [c])),
    )
    by_hand = run_experiment(listed, tiny_backend, prompts)
    np.testing.assert_array_equal(result.patched, by_hand.patched)

    # Without singles, nothing is added; a set that keeps its sites has no sites alone.
    off = spec_factory(experiment=patching, scope=_sets(_set("three", [a, b, c])))
    assert single_sets(off.scope) == []
    kept = spec_factory(
        experiment=patching,
        scope=_sets(_set("kept", [a, b], complement=True), universe=["head"], singles=True),
    )
    assert single_sets(kept.scope) == []


# -- random sets ----------------------------------------------------------------------------------


def _controls(n=6, seed=0, match="layer"):
    return {"n": n, "seed": seed, "match": match}


def test_random_sets_match_the_set_and_never_hold_its_sites(tiny_backend, spec_factory):
    info = tiny_backend.info
    a, b, c = _head(0, 1), _head(1, 2), _head(1, 0, {"kind": "last"})
    spec = spec_factory(
        scope=_sets(_set("circuit", [a, b, c]), _set("one", [a]), controls=_controls())
    )
    scope = spec.scope
    assert isinstance(scope, SiteSetsScope)
    drawn = random_sets(scope, info)
    assert [i for i, _ in drawn] == [0] * 6 + [1] * 6
    own = {(0, 1), (1, 2), (1, 0)}
    for i, s in drawn:
        of = scope.sets[i]
        assert len(s.sites) == len(of.sites) and s.complement == of.complement
        for site, original in zip(s.sites, of.sites, strict=True):
            # Another head of the same layer, at the same position.
            assert (site.kind, site.layer, site.position) == (
                original.kind,
                original.layer,
                original.position,
            )
            assert (site.layer, site_unit(site)) not in (own if i == 0 else {(0, 1)})
        assert len({(x.layer, site_unit(x)) for x in s.sites}) == len(s.sites)
    assert [s.label for _, s in drawn[:2]] == ["circuit · random 1", "circuit · random 2"]

    # The same seed draws the same sets; another seed, others; and a set's random sets don't
    # depend on the sets listed after it.
    assert [s for _, s in random_sets(scope, info)] == [s for _, s in drawn]
    reseeded = spec_factory(
        scope=_sets(_set("circuit", [a, b, c]), controls=_controls(seed=1))
    ).scope
    assert [s.sites for _, s in random_sets(reseeded, info)] != [s.sites for _, s in drawn[:6]]
    alone = spec_factory(scope=_sets(_set("circuit", [a, b, c]), controls=_controls())).scope
    assert [s.sites for _, s in random_sets(alone, info)] == [s.sites for _, s in drawn[:6]]

    # Matched by model, a site may land in any layer.
    anywhere = spec_factory(
        scope=_sets(_set("circuit", [a, b, c]), controls=_controls(n=40, match="model"))
    ).scope
    layers = {tuple(x.layer for x in s.sites) for _, s in random_sets(anywhere, info)}
    assert len(layers) > 1

    # A head at two positions is one head of the random set, at both.
    twice = spec_factory(
        scope=_sets(
            _set(
                "twice", [_head(1, 2, {"kind": "last"}), _head(1, 2, {"kind": "index", "index": 0})]
            ),
            controls=_controls(),
        )
    ).scope
    for _, s in random_sets(twice, info):
        assert s.sites[0].head == s.sites[1].head != 2


def test_a_run_says_where_a_set_lies_among_its_random_sets(tiny_backend, project, spec_factory):
    a, b = _head(0, 1), _head(1, 2)
    spec = spec_factory(
        experiment={"kind": "ablation", "baseline": {"kind": "zero"}},
        scope=_sets(
            _set("everything", [], complement=True),
            _set("pair", [a, b]),
            _set("kept", [a, b], complement=True),
            universe=["head"],
            singles=True,
            controls=_controls(n=5),
        ),
    )
    assert "against 5 random sets each from its layers" in describe_experiment(spec)
    result = run_experiment(spec, tiny_backend, _prompts(tiny_backend, project))
    labels = [s.label for s in result.sites]
    # The scope's sets, then the sites alone, then each set's random sets (the set that replaces
    # everything has no sites to redraw).
    assert labels[:5] == ["everything", "pair", "kept", "L0 H1", "L1 H2"]
    assert labels[5:] == [f"pair · random {j}" for j in range(1, 6)] + [
        f"kept · random {j}" for j in range(1, 6)
    ]
    stats = compute_stats(spec, result)
    summary = build_summary(spec, result, stats, "run", None)
    rows = {r["label"]: r for r in summary["circuit"]["rows"]}
    assert list(rows) == labels[:5]  # random sets have no row of their own
    assert rows["everything"]["control"] is None and rows["L0 H1"]["control"] is None
    for label, at in (("pair", slice(5, 10)), ("kept", slice(10, 15))):
        control = rows[label]["control"]
        values = stats.effect_mean[at]
        own = stats.effect_mean[labels.index(label)]
        assert control["n"] == 5
        assert control["mean"] == pytest.approx(values.mean(), abs=1e-12)
        assert control["lo"] <= control["mean"] <= control["hi"]
        beyond = int((np.abs(values) >= abs(own)).sum())
        assert control["beyond"] == beyond and control["p"] == pytest.approx((beyond + 1) / 6)
    # Random sets don't change what the sets themselves are measured against.
    assert rows["kept"]["faithfulness"] is not None and rows["pair"]["share"] is not None
    assert rows["pair"]["interaction"]["parts"] == ["L0 H1", "L1 H2"]
    layout = summary["layout"]["rows"]
    assert layout[3]["single"] is True and layout[5]["control_of"] == 1
    assert layout[10]["control_of"] == 2
    # A random set is a control: it is never a run's strongest effect.
    assert [bool((s.variant or {}).get("control")) for s in result.sites] == [False] * 5 + [
        True
    ] * 10
    # The tiny model's layers have four heads: a set of one head in each leaves three others to
    # draw, enough for five draws to differ. A set that takes half a layer leaves one other set.
    assert not any("different set" in w for w in result.warnings)
    crowded = spec_factory(
        experiment={"kind": "ablation", "baseline": {"kind": "zero"}},
        scope=_sets(_set("half", [_head(0, 0), _head(0, 2)]), controls=_controls(n=6)),
    )
    few = run_experiment(crowded, tiny_backend, _prompts(tiny_backend, project))
    assert any(
        "The 6 random sets beside 'half' are only 1 different set" in w for w in few.warnings
    )

    # A whole run saves and reads back.
    outcome = run_spec(spec, project, backend=tiny_backend)
    assert outcome.status == "finished", outcome.manifest.get("error")
    assert outcome.summary["circuit"]["rows"][1]["control"]["n"] == 5

    # The same run without its controls gives the sets the same numbers.
    plain = spec_factory(
        experiment={"kind": "ablation", "baseline": {"kind": "zero"}},
        scope=_sets(
            _set("everything", [], complement=True),
            _set("pair", [a, b]),
            _set("kept", [a, b], complement=True),
            universe=["head"],
        ),
    )
    bare = run_experiment(plain, tiny_backend, _prompts(tiny_backend, project))
    np.testing.assert_array_equal(bare.patched, result.patched[:3])


def test_random_sets_are_checked(tiny_backend, spec_factory):
    info = tiny_backend.info
    every_head = [_head(0, h) for h in range(info.n_heads)]
    full = spec_factory(scope=_sets(_set("layer 0", every_head), controls=_controls()))
    with pytest.raises(ScopeError, match="leaves no other heads in layer 0"):
        random_sets(full.scope, info)
    # A layer has one attention output: matched by layer there is nothing to redraw.
    attn = {"kind": "attn_out", "layer": 0, "position": ALL}
    with pytest.raises(ValidationError, match="a layer has only one"):
        spec_factory(scope=_sets(_set("attention", [attn]), controls=_controls()))
    by_model = spec_factory(
        scope=_sets(_set("attention", [attn]), controls=_controls(n=1, match="model"))
    )
    ((_, drawn),) = random_sets(by_model.scope, info)
    assert drawn.sites[0].kind == "attn_out" and drawn.sites[0].layer != 0
    # The rows of a scope with random sets need the model's shape.
    with pytest.raises(ScopeError, match="loaded model"):
        set_rows(by_model.scope)
