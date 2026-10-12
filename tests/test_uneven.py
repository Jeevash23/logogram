"""Prompt pairs of different token lengths: labelled in both prompts, matched label by label.

The pairs here are IOI prompts whose second mention of the subject is two names in one of the two
prompts, so that prompt is a token longer: the corrupt prompt in pairs 1, 4, 7 and 10, the clean
prompt in pairs 2, 5, 8 and 11. Pairs 0, 3, 6 and 9 keep equal lengths, labelled in both prompts
all the same.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from logogram.backends.base import Patch
from logogram.datasets import DatasetError, PromptRecord, parse_jsonl, parse_record, write_dataset
from logogram.engine import compute_baselines, make_scorer, run_engine, run_experiment
from logogram.ioi import generate_ioi
from logogram.prompts import UnevenPairs, group_by_length, prepare_prompt, prepare_prompts
from logogram.results import compute_stats
from logogram.runner import run_spec
from logogram.sites import ScopeError

TOL = 1e-4
LABELS = ("IO", "S1", "S2", "end")


def _replace(
    text: str, spans: dict[str, tuple[int, int]], label: str, value: str
) -> tuple[str, dict[str, tuple[int, int]]]:
    """``text`` with the span of ``label`` replaced by ``value``, and every span moved with it."""
    start, end = spans[label]
    shift = len(value) - (end - start)
    moved = {
        k: (a, start + len(value)) if k == label else (a + shift, b + shift) if a >= end else (a, b)
        for k, (a, b) in spans.items()
    }
    return text[:start] + value + text[end:], moved


def uneven_records(n: int = 12, seed: int = 0) -> list[PromptRecord]:
    records = []
    for i, r in enumerate(generate_ioi(n, seed=seed)):
        assert r.positions is not None and isinstance(r.answer, str)
        io, s = r.answer.strip(), str(r.distractor).strip()
        extra = next(name for name in ("Bob", "Tom", "Sam") if name not in (io, s))
        clean, positions = _replace(
            r.clean, r.positions, "S2", s + (f" {extra}" if i % 3 == 2 else "")
        )
        corrupt, corrupt_positions = _replace(
            r.clean, r.positions, "S2", io + (f" {extra}" if i % 3 == 1 else "")
        )
        records.append(
            PromptRecord(
                clean=clean,
                corrupt=corrupt,
                answer=r.answer,
                distractor=r.distractor,
                positions=positions,
                corrupt_positions=corrupt_positions,
                id=r.id,
                meta=r.meta,
            )
        )
    return records


@pytest.fixture
def uneven(tiny_backend):
    return prepare_prompts(tiny_backend, uneven_records(), True)


@pytest.fixture
def uneven_project(project):
    write_dataset(project.datasets_dir / "uneven.jsonl", uneven_records())
    return project


def _spec(spec_factory, **changes):
    return spec_factory(dataset={"path": "datasets/uneven.jsonl"}, **changes)


# -- the dataset format ----------------------------------------------------------------------


def test_both_prompts_label_the_same_positions():
    clean = "When Mary and John went to the store, John gave a drink to"
    corrupt = "When Mary and John went to the store, Bartholomew gave a drink to"
    base = {"clean": clean, "corrupt": corrupt, "answer": " Mary", "distractor": " John"}
    record = parse_record(
        {
            **base,
            "positions": {"S2": [38, 42], "end": [56, 58]},
            "corrupt_positions": {"S2": [38, 49], "end": [63, 65]},
        },
        "line 1",
    )
    assert record.corrupt_positions == {"S2": (38, 49), "end": (63, 65)}
    assert corrupt[38:49] == "Bartholomew" and corrupt[63:65] == "to"
    # Written and read back as it was.
    assert parse_jsonl(json.dumps(record.model_dump(exclude_none=True)))[0] == record
    # Which labels are missing where.
    with pytest.raises(
        DatasetError, match=r"line 2: the corrupt prompt \(corrupt_positions\) has no 'end'"
    ):
        parse_record(
            {
                **base,
                "positions": {"S2": [38, 42], "end": [56, 58]},
                "corrupt_positions": {"S2": [38, 49]},
            },
            "line 2",
        )
    with pytest.raises(DatasetError, match=r"the clean prompt \(positions\) has no 'IO' or 'S1'"):
        parse_record(
            {
                **base,
                "positions": {"S2": [38, 42]},
                "corrupt_positions": {"S2": [38, 49], "IO": [5, 9], "S1": [14, 18]},
            },
            "line 3",
        )
    with pytest.raises(DatasetError, match=r"the clean prompt \(positions\) has no 'S2'"):
        parse_record({**base, "corrupt_positions": {"S2": [38, 49]}}, "line 4")
    with pytest.raises(DatasetError, match="ends at character 70, but the corrupt prompt has 65"):
        parse_record(
            {**base, "positions": {"S2": [38, 42]}, "corrupt_positions": {"S2": [38, 70]}},
            "line 5",
        )
    with pytest.raises(DatasetError, match="can't be a position label"):
        parse_record(
            {**base, "positions": {"last": [38, 42]}, "corrupt_positions": {"last": [38, 49]}},
            "line 6",
        )


def test_prompts_of_different_lengths_need_their_positions_in_both(tiny_backend):
    record = uneven_records()[1]
    prepared, issues, clean, corrupt = prepare_prompt(tiny_backend, record, 1, True)
    assert prepared is not None and not issues
    assert len(corrupt.ids) == len(clean.ids) + 1 and not prepared.even
    # Without the corrupt prompt's own positions, the pair can't be used, and the issue says why.
    bare = record.model_copy(update={"corrupt_positions": None})
    prepared, issues, _, _ = prepare_prompt(tiny_backend, bare, 1, True)
    assert prepared is None and [i.kind for i in issues] == ["length_mismatch"]
    assert "corrupt_positions" in issues[0].message
    # Pairs of equal length may label their corrupt prompts too, and are matched label by label.
    even = uneven_records()[0]
    prepared, issues, clean, corrupt = prepare_prompt(tiny_backend, even, 0, True)
    assert prepared is not None and prepared.even and prepared.corrupt_labels == prepared.labels


def test_each_prompt_has_its_own_labels(uneven):
    for p in uneven:
        assert set(p.labels) == set(p.corrupt_labels) == set(LABELS)
        assert p.clean.tokens[p.labels["end"]] == p.corrupt.tokens[p.corrupt_labels["end"]] == " to"
        assert p.labels["end"] == len(p.clean.ids) - 1
        assert p.corrupt_labels["end"] == len(p.corrupt.ids) - 1
        # The second mention of the subject: the last token of its span in each prompt.
        assert (
            p.clean.tokens[p.labels["S2"]].strip()
            == p.record.clean[slice(*p.record.positions["S2"])].split()[-1]
        )  # type: ignore[index]
        corrupt_s2 = p.record.corrupt_positions["S2"]  # type: ignore[index]
        assert (
            p.corrupt.tokens[p.corrupt_labels["S2"]].strip()
            == p.record.corrupt[slice(*corrupt_s2)].split()[-1]
        )
    lengths = {(len(p.clean.ids), len(p.corrupt.ids)) for p in uneven}
    assert lengths == {(15, 15), (15, 16), (16, 15)}
    assert [p.index for p in uneven if not p.even] == [1, 2, 4, 5, 7, 8, 10, 11]


def test_each_side_runs_in_batches_of_its_own_length(uneven):
    for side in ("clean", "corrupt"):
        groups = group_by_length(uneven, side)
        for group in groups:
            assert all(uneven[i].length_of(side) == group.length for i in group.members)
            assert group.tokens(side).shape == (len(group.members), group.length)
        assert sorted(i for g in groups for i in g.members) == list(range(len(uneven)))
    # Whole pairs line up only when both prompts have one length.
    with pytest.raises(UnevenPairs, match="Prompt pairs 1, 2, 4, 5, 7 and 3 more"):
        group_by_length(uneven)
    clean_groups = group_by_length(uneven, "clean")
    with pytest.raises(UnevenPairs, match="group the prompts by the length of their corrupt"):
        clean_groups[0].tokens("corrupt")


# -- exactness ---------------------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["clean_to_corrupt", "corrupt_to_clean"])
@pytest.mark.parametrize("normalization", ["dataset_gap", "prompt_gap"])
def test_patching_where_the_answer_is_read_gives_effect_one(
    tiny_backend, uneven, spec_factory, direction, normalization
):
    """After the last layer only the last token's residual stream is read: patching it (as the
    last token, or at the label on it) reproduces the source run exactly, however long each prompt
    is. Patching another label there changes nothing."""
    last = tiny_backend.info.n_layers - 1
    sites = [
        {"kind": "resid_post", "layer": last, "position": {"kind": "last"}},
        {"kind": "resid_post", "layer": last, "position": {"kind": "label", "label": "end"}},
        {"kind": "resid_mid", "layer": last, "position": {"kind": "last"}},
        {"kind": "resid_post", "layer": last, "position": {"kind": "label", "label": "S2"}},
    ]
    spec = spec_factory(
        experiment={"kind": "activation_patching", "direction": direction},
        scope={"kind": "sites", "sites": sites},
        metric={"kind": "logit_diff", "normalization": normalization},
    )
    result = run_engine(spec, tiny_backend, uneven)
    stats = compute_stats(spec, result)
    np.testing.assert_allclose(stats.effect_mean[:3], 1.0, atol=TOL)
    np.testing.assert_allclose(
        result.patched[:3], np.broadcast_to(result.reference_metric, (3, 12)), atol=TOL
    )
    np.testing.assert_allclose(result.patched[3], result.receiver_metric, atol=TOL)
    if normalization == "prompt_gap":
        np.testing.assert_allclose(stats.effect[:3], 1.0, atol=1e-3)


def test_patching_every_label_of_the_last_layer_at_once_gives_effect_one(
    tiny_backend, uneven, spec_factory
):
    last = tiny_backend.info.n_layers - 1
    every = [
        {"kind": "resid_post", "layer": last, "position": {"kind": "label", "label": label}}
        for label in LABELS
    ]
    spec = spec_factory(
        scope={
            "kind": "site_sets",
            "universe": None,
            "sets": [
                {"label": "every label", "sites": every, "complement": False},
                {"label": "every label but the last", "sites": every[:3], "complement": False},
            ],
        },
        metric={"kind": "logit_diff", "normalization": "prompt_gap"},
    )
    result = run_experiment(spec, tiny_backend, uneven)
    stats = compute_stats(spec, result)
    np.testing.assert_allclose(stats.effect[0], 1.0, atol=1e-3)
    np.testing.assert_allclose(result.patched[1], result.receiver_metric, atol=TOL)


def test_patching_corrupt_into_corrupt_gives_zero(tiny_backend, uneven, spec_factory):
    """A prompt patched into itself lines up with itself, at every position too."""
    s2 = {"kind": "label", "label": "S2"}
    for scope in (
        {"kind": "heads", "position": {"kind": "all"}},
        {"kind": "heads", "position": {"kind": "last"}},
        {"kind": "heads", "position": s2},
        {"kind": "layer_components", "position": {"kind": "index", "index": 3}},
        {"kind": "layer_position", "site": "resid_mid", "positions": "labels"},
    ):
        spec = spec_factory(scope=scope)
        result = run_engine(
            spec, tiny_backend, uneven, receiver_override="corrupt", source_override="corrupt"
        )
        stats = compute_stats(spec, result)
        np.testing.assert_allclose(stats.delta, 0.0, atol=TOL)
        assert int(stats.sign_flips.sum()) == 0


def test_a_label_is_patched_from_the_source_token_into_the_receiver_token(
    tiny_backend, uneven, spec_factory
):
    """Each prompt's own token at the label: the clean prompt's activation there replaces the
    corrupt prompt's, whichever is longer."""
    spec = spec_factory(
        scope={
            "kind": "sites",
            "sites": [
                {"kind": kind, "layer": 0, "position": {"kind": "label", "label": label}}
                for kind in ("resid_pre", "resid_post")
                for label in ("S2", "IO")
            ],
        },
    )
    result = run_engine(spec, tiny_backend, uneven)
    scorer = make_scorer(spec, uneven)
    for p in (uneven[1], uneven[2]):
        clean = torch.tensor([p.clean.ids])
        corrupt = torch.tensor([p.corrupt.ids])
        for row, (kind, label) in enumerate(
            (kind, label) for kind in ("resid_pre", "resid_post") for label in ("S2", "IO")
        ):
            source = tiny_backend.capture(clean, [(kind, 0)])[(kind, 0)][0, p.labels[label]]
            patch = Patch(
                kind=kind,
                layer=0,
                values=source[None],
                positions=torch.tensor([p.corrupt_labels[label]]),
            )

            def forward(tokens, keep, patch=patch):
                return tiny_backend.logits(tokens, patch, keep)

            expected = scorer.score(forward, corrupt, [p.index]).metric[0]
            assert abs(result.patched[row, p.index] - expected) < 1e-5
    # S2 is where the prompts differ: patching it moves every pair.
    assert np.abs(result.patched[0] - result.receiver_metric).min() > 1e-4


def test_the_same_spec_twice_gives_identical_results(tiny_backend, uneven_project, spec_factory):
    spec = _spec(
        spec_factory,
        experiment={
            "kind": "ablation",
            "baseline": {"kind": "resample", "pool": "corrupt", "donors": 3, "seed": 7},
        },
        scope={"kind": "heads", "position": {"kind": "label", "label": "S2"}},
    )
    first = run_spec(spec, uneven_project, backend=tiny_backend)
    second = run_spec(spec, uneven_project, backend=tiny_backend)
    assert first.status == second.status == "finished", first.manifest.get("error")
    a = pq.read_table(first.folder / "results.parquet")
    b = pq.read_table(second.folder / "results.parquet")
    assert a.equals(b)
    sa = json.loads((first.folder / "summary.json").read_text(encoding="utf-8"))
    sb = json.loads((second.folder / "summary.json").read_text(encoding="utf-8"))
    sa.pop("run_id"), sb.pop("run_id")
    assert sa == sb
    # The run says how its pairs were matched.
    assert "Prompt pairs 1, 2, 4, 5, 7 and 3 more (8 of 12) have clean and corrupt prompts of " in (
        " ".join(sa["warnings"])
    )


def test_pairs_of_equal_length_give_what_they_gave_before(tiny_backend, project, spec_factory):
    """Labelling the corrupt prompts of pairs of equal length (at the same tokens) changes no
    number, bit for bit."""
    records = generate_ioi(12, seed=0)
    labelled = []
    for r in records:
        assert r.positions is not None and isinstance(r.answer, str)
        corrupt, spans = _replace(r.clean, r.positions, "S2", r.answer.strip())
        assert corrupt == r.corrupt
        labelled.append(r.model_copy(update={"corrupt_positions": spans}))
    spec = spec_factory(scope={"kind": "heads", "position": {"kind": "label", "label": "S2"}})
    a = run_engine(spec, tiny_backend, prepare_prompts(tiny_backend, records, True))
    b = run_engine(spec, tiny_backend, prepare_prompts(tiny_backend, labelled, True))
    assert np.array_equal(a.patched, b.patched) and np.array_equal(a.patched_pref, b.patched_pref)


# -- what each method does with them -----------------------------------------------------------


REFUSED = {
    "patching every position": (
        {"experiment": {"kind": "activation_patching", "direction": "clean_to_corrupt"}},
        {"kind": "heads", "position": {"kind": "all"}},
        "Patching carries each activation .* at every position",
    ),
    "patching a token index": (
        {"experiment": {"kind": "activation_patching", "direction": "corrupt_to_clean"}},
        {
            "kind": "layer_components",
            "components": ["attn_out"],
            "position": {"kind": "index", "index": -2},
        },
        "at token index -2",
    ),
    "every token": (
        {},
        {"kind": "layer_position", "site": "resid_pre", "positions": "each"},
        "Use labelled positions instead",
    ),
    "mean ablation": (
        {"experiment": {"kind": "ablation", "baseline": {"kind": "mean", "reference": "corrupt"}}},
        {"kind": "heads", "position": {"kind": "all"}},
        "Mean ablation carries",
    ),
    "resample ablation": (
        {
            "experiment": {
                "kind": "ablation",
                "baseline": {"kind": "resample", "pool": "clean", "donors": 2, "seed": 0},
            }
        },
        {"kind": "heads", "position": {"kind": "index", "index": 4}},
        "Resample ablation carries",
    ),
    "attribution patching": (
        {"experiment": {"kind": "attribution_patching", "direction": "clean_to_corrupt"}},
        {"kind": "heads", "position": {"kind": "all"}},
        "Attribution patching compares",
    ),
    "integrated gradients": (
        {
            "experiment": {
                "kind": "attribution_patching",
                "direction": "clean_to_corrupt",
                "method": "integrated_gradients",
                "steps": 2,
            }
        },
        {"kind": "heads", "position": {"kind": "last"}},
        "Integrated gradients run on inputs between",
    ),
    "path patching": (
        {
            "experiment": {
                "kind": "path_patching",
                "direction": "clean_to_corrupt",
                "receivers": [{"kind": "logits"}],
                "freeze_mlps": False,
            }
        },
        {"kind": "heads", "position": {"kind": "all"}},
        "Path patching carries",
    ),
    "steering by the mean difference": (
        {
            "experiment": {
                "kind": "steering",
                "apply_to": "corrupt",
                "direction": {"kind": "mean_difference"},
                "coefficients": [1.0],
                "train_fraction": 0.5,
                "seed": 0,
                "control": False,
                "continuations": None,
            }
        },
        {
            "kind": "layer_components",
            "components": ["resid_pre"],
            "position": {"kind": "index", "index": 2},
        },
        "Steering by the mean difference compares",
    ),
    "a set that keeps its sites": (
        {},
        {
            "kind": "site_sets",
            "universe": ["head"],
            "sets": [{"label": "rest", "sites": [], "complement": True}],
        },
        "keeps its sites and patches every other position",
    ),
    "a set patched at every position": (
        {},
        {
            "kind": "site_sets",
            "universe": None,
            "sets": [
                {
                    "label": "L0",
                    "sites": [{"kind": "attn_out", "layer": 0, "position": {"kind": "all"}}],
                    "complement": False,
                }
            ],
        },
        "at every position that has no meaning",
    ),
    "a set mean-ablated": (
        {"experiment": {"kind": "ablation", "baseline": {"kind": "mean", "reference": "clean"}}},
        {
            "kind": "site_sets",
            "universe": None,
            "sets": [
                {
                    "label": "S2",
                    "sites": [
                        {
                            "kind": "attn_out",
                            "layer": 0,
                            "position": {"kind": "label", "label": "S2"},
                        }
                    ],
                    "complement": False,
                }
            ],
        },
        "Mean ablation of a set replaces whole activations",
    ),
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_positions_that_dont_line_up_are_refused(tiny_backend, uneven, spec_factory, case):
    """Every position and token indices have no meaning between prompts of different lengths: a
    method that carries values between them refuses them, naming the pairs and what to choose."""
    changes, scope, message = REFUSED[case]
    spec = spec_factory(scope=scope, **changes)
    with pytest.raises(ScopeError, match=message) as refused:
        run_experiment(spec, tiny_backend, uneven)
    text = str(refused.value)
    if "Use labelled positions" not in text:
        assert "Prompt pairs 1, 2, 4, 5, 7 and 3 more (8 of 12)" in text
        assert "last token" in text


def test_methods_that_read_one_prompt_take_any_position(tiny_backend, uneven, spec_factory):
    """Zero ablation carries nothing from another prompt, and direct logit attribution reads each
    prompt's own last token."""
    for scope in (
        {"kind": "heads", "position": {"kind": "all"}},
        {
            "kind": "layer_components",
            "components": ["mlp_out"],
            "position": {"kind": "index", "index": 3},
        },
        {
            "kind": "site_sets",
            "universe": ["head"],
            "sets": [{"label": "rest", "sites": [], "complement": True}],
        },
    ):
        spec = spec_factory(
            experiment={"kind": "ablation", "baseline": {"kind": "zero"}}, scope=scope
        )
        result = run_experiment(spec, tiny_backend, uneven)
        assert np.isfinite(result.patched).all()
    for which in ("clean", "corrupt"):
        spec = spec_factory(
            experiment={"kind": "direct_logit_attribution", "prompts": which},
            scope={
                "kind": "layer_components",
                "components": ["attn_out", "mlp_out"],
                "position": {"kind": "label", "label": "end"},
            },
        )
        result = run_experiment(spec, tiny_backend, uneven)
        # The terms add up to each prompt's own logit difference.
        assert not any("decomposed" in w for w in result.warnings)
        np.testing.assert_allclose(result.gap, result.baselines.metric(which))


def test_path_patching_into_the_logits_from_the_last_layer_is_patching(
    tiny_backend, uneven, spec_factory
):
    """Nothing after the last layer's heads reads them but its MLP and the logits, so a path from
    them to the logits is the whole effect of patching them."""
    last = tiny_backend.info.n_layers - 1
    scope = {
        "kind": "sites",
        "sites": [
            {"kind": "head", "layer": last, "head": h, "position": position}
            for h in range(tiny_backend.info.n_heads)
            for position in ({"kind": "last"}, {"kind": "label", "label": "end"})
        ],
    }
    paths = spec_factory(
        experiment={
            "kind": "path_patching",
            "direction": "clean_to_corrupt",
            "receivers": [{"kind": "logits"}],
            "freeze_mlps": False,
        },
        scope=scope,
    )
    patching = spec_factory(scope=scope)
    a = run_experiment(paths, tiny_backend, uneven)
    b = run_experiment(patching, tiny_backend, uneven)
    np.testing.assert_allclose(a.patched, b.patched, atol=1e-5)
    assert np.abs(b.patched - b.receiver_metric).max() > 1e-3


def test_estimates_at_labels_match_the_estimates_of_aligned_pairs(
    tiny_backend, project, spec_factory, monkeypatch
):
    """The way pairs of different lengths are estimated (each prompt read at its own token) gives
    the estimates of pairs whose positions line up, when they do."""
    from logogram.prompts import PreparedPrompt

    prompts = prepare_prompts(tiny_backend, generate_ioi(12, seed=0), True)
    for position in ({"kind": "label", "label": "S2"}, {"kind": "last"}):
        spec = spec_factory(
            experiment={"kind": "attribution_patching", "direction": "corrupt_to_clean"},
            scope={"kind": "heads", "position": position},
            metric={"kind": "logprob_diff", "normalization": "dataset_gap"},
        )
        aligned = run_experiment(spec, tiny_backend, prompts)
        with monkeypatch.context() as m:
            m.setattr(PreparedPrompt, "even", property(lambda self: False))
            at_tokens = run_experiment(spec, tiny_backend, prompts)
        np.testing.assert_allclose(at_tokens.delta, aligned.delta, rtol=1e-4, atol=1e-6)


def test_steering_compares_each_prompt_at_its_own_token(tiny_backend, uneven, spec_factory):
    spec = spec_factory(
        experiment={
            "kind": "steering",
            "apply_to": "corrupt",
            "direction": {"kind": "mean_difference"},
            "coefficients": [0.0, 1.0],
            "train_fraction": 0.5,
            "seed": 0,
            "control": False,
            "continuations": {"prompts": 2, "tokens": 3},
        },
        scope={
            "kind": "sites",
            "sites": [
                {"kind": "resid_pre", "layer": 1, "position": {"kind": "label", "label": "S2"}}
            ],
        },
    )
    result = run_experiment(spec, tiny_backend, uneven)
    train = result.extra["steering"]["train"]
    difference = torch.zeros(tiny_backend.info.d_model, dtype=torch.float64)
    for i in train:
        p = uneven[i]
        clean = tiny_backend.capture(torch.tensor([p.clean.ids]), [("resid_pre", 1)])[
            ("resid_pre", 1)
        ]
        corrupt = tiny_backend.capture(torch.tensor([p.corrupt.ids]), [("resid_pre", 1)])[
            ("resid_pre", 1)
        ]
        difference += (
            clean[0, p.labels["S2"]].double() - corrupt[0, p.corrupt_labels["S2"]].double()
        )
    difference /= len(train)
    assert result.extra["steering"]["norms"][0] == pytest.approx(float(difference.norm()), rel=1e-5)
    np.testing.assert_allclose(result.patched[0], result.receiver_metric, atol=TOL)
    continued = result.extra["continuations"]["prompts"]
    assert [len(c["prompt"]) for c in continued] == [
        len(uneven[c["index"]].corrupt.ids) for c in continued
    ]


def test_baselines_read_each_prompt_after_its_own_last_token(tiny_backend, uneven, spec_factory):
    spec = spec_factory(metric={"kind": "kl", "target": "clean", "normalization": "dataset_gap"})
    scorer = make_scorer(spec, uneven)
    baselines = compute_baselines(
        tiny_backend, uneven, group_by_length(uneven, "corrupt"), 5, scorer=scorer
    )
    np.testing.assert_allclose(baselines.clean, 0.0, atol=1e-6)  # the target's own divergence
    for p in (uneven[0], uneven[1], uneven[2]):
        for which in ("clean", "corrupt"):
            logits = tiny_backend.final_logits(torch.tensor([p.tokens_of(which).ids]))[0]
            expected = float(logits[p.answer_id] - logits[p.distractor_id])
            assert baselines.pref(which)[p.index] == pytest.approx(expected, abs=1e-4)


# -- SAE features ------------------------------------------------------------------------------


SAE_REF = {"repo": "local/exact", "path": "", "revision": "test"}


@pytest.fixture
def exact(tiny_backend, tmp_path):
    from test_sae import exact_sae

    torch.manual_seed(1)
    d = tiny_backend.info.d_model
    return exact_sae(tmp_path / "exact", d, b_dec=torch.randn(d) * 0.1)


def _feature_spec(spec_factory, experiment, position, features=(3,)):
    sites = [
        {"kind": "sae_feature", "layer": 1, "feature": f, "position": position} for f in features
    ]
    return spec_factory(sae=SAE_REF, experiment=experiment, scope={"kind": "sites", "sites": sites})


PATCH = {"kind": "activation_patching", "direction": "clean_to_corrupt"}


def test_a_feature_is_patched_from_the_source_token_into_the_receiver_token(
    tiny_backend, uneven, spec_factory, exact
):
    feature = 3
    spec = _feature_spec(spec_factory, PATCH, {"kind": "label", "label": "S2"}, (feature,))
    result = run_experiment(spec, tiny_backend, uneven, sae=exact)
    scorer = make_scorer(spec, uneven)
    key = ("resid_pre", 1)
    for p in uneven:
        corrupt_tokens = torch.tensor([p.corrupt.ids])
        clean = tiny_backend.capture(torch.tensor([p.clean.ids]), [key])[key][0]
        corrupt = tiny_backend.capture(corrupt_tokens, [key])[key][0]
        at = p.corrupt_labels["S2"]
        f_clean, _ = exact.encode(clean[p.labels["S2"]])
        f_corrupt, _ = exact.encode(corrupt[at])
        value = corrupt[at] + (f_clean - f_corrupt)[feature] * exact.params.W_dec[feature]
        patch = Patch(kind="resid_pre", layer=1, values=value[None], positions=torch.tensor([at]))

        def forward(tokens, keep, patch=patch):
            return tiny_backend.logits(tokens, patch, keep)

        expected = scorer.score(forward, corrupt_tokens, [p.index]).metric[0]
        assert abs(result.patched[0, p.index] - expected) < 1e-4
    # Every position of a feature would carry values between prompts of different lengths.
    every = _feature_spec(spec_factory, PATCH, {"kind": "all"}, (feature,))
    with pytest.raises(ScopeError, match="Patching carries"):
        run_experiment(every, tiny_backend, uneven, sae=exact)
    # Zero ablation reads the receiver alone.
    zero = _feature_spec(
        spec_factory, {"kind": "ablation", "baseline": {"kind": "zero"}}, {"kind": "all"}
    )
    assert np.isfinite(run_experiment(zero, tiny_backend, uneven, sae=exact).patched).all()


def test_feature_estimates_at_labels_match_those_of_aligned_pairs(
    tiny_backend, spec_factory, exact, monkeypatch
):
    from logogram.prompts import PreparedPrompt

    prompts = prepare_prompts(tiny_backend, generate_ioi(12, seed=0), True)
    s2 = {"kind": "label", "label": "S2"}
    estimate = {"kind": "attribution_patching", "direction": "clean_to_corrupt"}
    for spec in (
        spec_factory(
            sae=SAE_REF,
            experiment=estimate,
            scope={"kind": "features", "position": s2, "top": 5, "choose_on": None, "seed": None},
        ),
        _feature_spec(spec_factory, estimate, s2, (1, 7)),
    ):
        aligned = run_experiment(spec, tiny_backend, prompts, sae=exact)
        with monkeypatch.context() as m:
            m.setattr(PreparedPrompt, "even", property(lambda self: False))
            at_tokens = run_experiment(spec, tiny_backend, prompts, sae=exact)
        assert [rs.label for rs in at_tokens.sites] == [rs.label for rs in aligned.sites]
        np.testing.assert_allclose(at_tokens.delta, aligned.delta, rtol=1e-4, atol=1e-6)
        info, again = aligned.extra["features"], at_tokens.extra["features"]
        assert again["site_estimate"] == pytest.approx(info["site_estimate"], rel=1e-4, abs=1e-6)


def test_feature_estimates_on_uneven_pairs_add_up_to_the_site(
    tiny_backend, uneven, spec_factory, exact
):
    """Without an error term, the activation is the sum of its features' decoder rows, so their
    estimates add up to the whole site's, each prompt read at its own token."""
    estimate = {"kind": "attribution_patching", "direction": "corrupt_to_clean"}

    def every(position):
        scope = {
            "kind": "features",
            "position": position,
            "top": 5,
            "choose_on": None,
            "seed": None,
        }
        return spec_factory(sae=SAE_REF, experiment=estimate, scope=scope)

    result = run_experiment(
        every({"kind": "label", "label": "S2"}), tiny_backend, uneven, sae=exact
    )
    info = result.extra["features"]
    assert info["features_estimate"] == pytest.approx(info["site_estimate"], rel=1e-5, abs=1e-6)
    with pytest.raises(ScopeError, match="Attribution patching compares"):
        run_experiment(every({"kind": "all"}), tiny_backend, uneven, sae=exact)


# -- what the app shows ------------------------------------------------------------------------


def test_the_token_strip_lines_the_prompts_up_by_their_labels(tiny_backend):
    from logogram.analysis import align_tokens, tokenize_pair

    record = uneven_records()[1]  # the corrupt prompt names two people where the clean names one
    strip = tokenize_pair(tiny_backend, record, True)
    assert not strip["aligned"] and not strip["issues"]
    clean, corrupt = strip["clean"]["tokens"], strip["corrupt"]["tokens"]
    labels, corrupt_labels = strip["labels"], strip["corrupt_labels"]
    columns = strip["columns"]
    # Every token of each prompt, once and in order.
    assert [c for c, _ in columns if c is not None] == list(range(len(clean)))
    assert [d for _, d in columns if d is not None] == list(range(len(corrupt)))
    # Each label's two tokens share a column, as do the last tokens.
    for label in LABELS:
        assert [labels[label], corrupt_labels[label]] in columns
    assert columns[-1] == [len(clean) - 1, len(corrupt) - 1]
    # The extra name stands alone; the tokens around it line up.
    alone = [d for c, d in columns if c is None]
    assert [corrupt[d] for d in alone] == [corrupt[corrupt_labels["S2"] - 1]]
    assert all(
        clean[c] == corrupt[d]
        for c, d in columns
        if None not in (c, d) and [c, d] not in [[labels["S2"], corrupt_labels["S2"]]]
    )
    # Pairs of equal length line up position by position, as they always did.
    even = tokenize_pair(tiny_backend, uneven_records()[0], True)
    assert even["aligned"] and even["columns"] is None
    # Labels whose order differs between the prompts can't all share a column: most of them do.
    crossed = align_tokens([1, 2, 3, 4], [1, 3, 2, 9, 4], {"a": 1, "b": 2}, {"a": 2, "b": 1})
    assert crossed[-1] == [3, 4]
    assert sum(1 for c in crossed if c in ([1, 2], [2, 1])) == 1
    assert [c for c, _ in crossed if c is not None] == [0, 1, 2, 3]
    assert [d for _, d in crossed if d is not None] == [0, 1, 2, 3, 4]


def test_the_app_learns_which_pairs_differ_in_length(tiny_backend, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from logogram.server.app import create_app
    from logogram.server.security import SecurityConfig

    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    app = create_app(SecurityConfig(token="t", port=8765), serve_web=False)
    base = "http://127.0.0.1:8765"
    client = TestClient(app, base_url=base)
    auth = {"authorization": "Bearer t"}
    headers = {**auth, "origin": base}
    client.post("/api/projects", json={"name": "Uneven", "parent": str(tmp_path)}, headers=headers)
    text = "\n".join(json.dumps(r.model_dump(exclude_none=True)) for r in uneven_records()[:4])
    created = client.post(
        "/api/datasets/import", json={"name": "uneven", "text": text}, headers=headers
    )
    assert created.status_code == 200, created.text
    params = {"path": "datasets/uneven.jsonl"}
    assert client.get("/api/dataset", params=params, headers=auth).json()["uneven"] is None
    app.state.logogram.backend = tiny_backend
    app.state.logogram.model_status = {"state": "ready"}
    detail = client.get("/api/dataset", params=params, headers=auth).json()
    assert detail["issues"] == [] and detail["lengths"] == [15, 16]
    assert detail["uneven"] == [
        {"index": 1, "clean": 15, "corrupt": 16},
        {"index": 2, "clean": 16, "corrupt": 15},
    ]
    assert detail["records"][1]["corrupt_positions"]["S2"] == list(
        uneven_records()[1].corrupt_positions["S2"]
    )  # type: ignore[index]
    body = {
        "dataset": "datasets/uneven.jsonl",
        "index": 1,
        "prepend_bos": True,
        "batch_size": 4,
        "metric": {"kind": "logit_diff", "normalization": "dataset_gap"},
    }
    response = client.post("/api/tokenize", json=body, headers=headers)
    assert response.status_code == 200, response.text
    strip = response.json()
    assert strip["columns"] and strip["corrupt_labels"]["S2"] == 11
