from __future__ import annotations

import numpy as np
import pytest
import torch

from logogram.datasets import PromptRecord, load_dataset
from logogram.engine import EngineError, draw_donors, run_engine
from logogram.metrics import Target
from logogram.prompts import PreparedPrompt, PromptError, group_by_length, prepare_prompts
from logogram.results import compute_stats
from logogram.sites import ScopeError, expand_scope


def _prompts(backend, project, name="ioi.jsonl"):
    return prepare_prompts(backend, load_dataset(project.datasets_dir / name), True)


def _manual_ld(backend, tokens, prompt):
    logits = backend.final_logits(torch.tensor([tokens]))[0]
    return float(logits[prompt.answer_id] - logits[prompt.distractor_id])


def test_resample_whole_residual_reproduces_each_donor(tiny_backend, project, spec_factory):
    """Resampling resid_pre at layer 0 runs the donor prompt, scored on the receiver's answer."""
    spec = spec_factory(
        experiment={
            "kind": "ablation",
            "baseline": {"kind": "resample", "pool": "clean", "donors": 2, "seed": 3},
        },
        scope={"kind": "sites", "sites": [{"kind": "resid_pre", "layer": 0}]},
    )
    prompts = _prompts(tiny_backend, project)
    result = run_engine(spec, tiny_backend, prompts)
    assert result.donors is not None
    for i, donors in enumerate(result.donors):
        expected = np.mean(
            [_manual_ld(tiny_backend, prompts[j].clean.ids, prompts[i]) for j in donors]
        )
        assert result.patched[0, i] == pytest.approx(expected, abs=1e-4)


def test_zero_ablation_of_whole_residual_ignores_the_prompt(tiny_backend, project, spec_factory):
    spec = spec_factory(
        experiment={"kind": "ablation", "baseline": {"kind": "zero"}},
        scope={"kind": "sites", "sites": [{"kind": "resid_pre", "layer": 0}]},
    )
    prompts = _prompts(tiny_backend, project)
    result = run_engine(spec, tiny_backend, prompts)
    # Same answer pair => same output, because nothing of the prompt survives.
    by_pair: dict[tuple[int, int], set[float]] = {}
    for i, p in enumerate(prompts):
        by_pair.setdefault((p.answer_id, p.distractor_id), set()).add(
            round(result.patched[0, i], 4)
        )
    assert all(len(v) == 1 for v in by_pair.values())


def test_mean_ablation_uses_the_stated_reference(tiny_backend, project, spec_factory):
    """Mean of the clean set at all positions, applied to the whole residual stream."""
    spec = spec_factory(
        experiment={"kind": "ablation", "baseline": {"kind": "mean", "reference": "clean"}},
        scope={"kind": "sites", "sites": [{"kind": "resid_pre", "layer": 1}]},
    )
    prompts = _prompts(tiny_backend, project)
    result = run_engine(spec, tiny_backend, prompts)
    group = group_by_length(prompts)[0]
    acts = tiny_backend.capture(group.clean, [("resid_pre", 1)])[("resid_pre", 1)]
    mean = acts.float().mean(0, keepdim=True)

    def hook(act, hook=None):
        return mean.expand_as(act).to(act.dtype)

    with torch.no_grad():
        logits = tiny_backend.bridge.run_with_hooks(
            group.clean[:1], fwd_hooks=[("blocks.1.hook_resid_pre", hook)], return_type="logits"
        )[0, -1]
    p = prompts[group.members[0]]
    expected = float(logits[p.answer_id] - logits[p.distractor_id])
    assert result.patched[0, group.members[0]] == pytest.approx(expected, abs=1e-4)


def test_donors_are_deterministic_distinct_and_never_self(tiny_backend, project):
    prompts = _prompts(tiny_backend, project, "mixed.jsonl")
    groups = group_by_length(prompts)
    a = draw_donors(prompts, groups, 2, seed=5, same_length=False)
    b = draw_donors(prompts, groups, 2, seed=5, same_length=False)
    c = draw_donors(prompts, groups, 2, seed=6, same_length=False)
    assert a == b and a != c
    for i, donors in enumerate(a):
        assert i not in donors and len(set(donors)) == 2
    same = draw_donors(prompts, groups, 1, seed=5, same_length=True)
    for i, donors in enumerate(same):
        assert all(prompts[j].length == prompts[i].length for j in donors)
    with pytest.raises(EngineError, match="donor"):
        draw_donors(prompts, groups, 50, seed=0, same_length=True)


def test_label_positions_point_at_the_named_tokens(tiny_backend, project):
    for p in _prompts(tiny_backend, project, "mixed.jsonl"):
        record = p.record
        io_name = record.answer.strip()
        s_name = record.distractor.strip()
        assert p.clean.tokens[p.labels["IO"]].strip() == io_name
        assert p.clean.tokens[p.labels["S1"]].strip() == s_name
        assert p.clean.tokens[p.labels["S2"]].strip() == s_name
        assert p.corrupt.tokens[p.labels["S2"]].strip() == io_name  # flip corruption
        assert p.labels["end"] == p.length - 1


def test_layer_position_needs_equal_lengths_or_labels(tiny_backend, project, spec_factory):
    prompts = _prompts(tiny_backend, project, "mixed.jsonl")
    assert len({p.length for p in prompts}) > 1
    with pytest.raises(ScopeError, match="different token lengths"):
        expand_scope(
            spec_factory(scope={"kind": "layer_position", "positions": "each"}),
            tiny_backend.info,
            prompts,
        )
    sites, layout = expand_scope(
        spec_factory(scope={"kind": "layer_position", "positions": "labels"}),
        tiny_backend.info,
        prompts,
    )
    assert [c["key"] for c in layout["cols"]] == ["IO", "S1", "S2", "end"]
    assert len(sites) == tiny_backend.info.n_layers * 4


def test_mixed_lengths_run_in_separate_batches(tiny_backend, project, spec_factory):
    spec = spec_factory(dataset={"path": "datasets/mixed.jsonl"})
    prompts = _prompts(tiny_backend, project, "mixed.jsonl")
    result = run_engine(spec, tiny_backend, prompts)
    stats = compute_stats(spec, result)
    assert np.isfinite(stats.effect_mean).all()
    assert result.patched.shape == (tiny_backend.info.n_layers * tiny_backend.info.n_heads, 12)


def test_single_position_patch_only_touches_that_position(tiny_backend, project, spec_factory):
    """Patching resid_pre at layer 0 at a position where clean and corrupt agree changes nothing."""
    prompts = _prompts(tiny_backend, project)
    same_pos = next(
        i for i in range(prompts[0].length) if i not in prompts[0].differing_positions()
    )
    spec = spec_factory(
        scope={
            "kind": "sites",
            "sites": [
                {"kind": "resid_pre", "layer": 0, "position": {"kind": "index", "index": same_pos}}
            ],
        }
    )
    result = run_engine(spec, tiny_backend, prompts)
    np.testing.assert_allclose(result.patched[0], result.receiver_metric, atol=1e-4)


def test_unusable_prompts_are_reported_clearly(tiny_backend):
    records = [
        PromptRecord(
            clean="Hello world", corrupt="Hello world world", answer=" Mary", distractor=" John"
        ),
        PromptRecord(
            clean="Hello world", corrupt="Hello Mary", answer=" Mary Mary", distractor=" John"
        ),
    ]
    with pytest.raises(PromptError) as info:
        prepare_prompts(tiny_backend, records, True)
    kinds = {i.kind for i in info.value.issues}
    assert kinds == {"length_mismatch", "answer_tokens"}
    assert "positions can't be aligned" in str(info.value)


IOI_CLEAN = "When Mary and John went to the store, John gave a drink to"
IOI_CORRUPT = "When Mary and John went to the store, Mary gave a drink to"


def _same_prompt_twice(backend, n=2):
    """Prepared prompts whose clean and corrupt runs are identical (prepare_prompts refuses
    these, so the engine's own checks are tested directly)."""
    record = PromptRecord(clean=IOI_CLEAN, corrupt=IOI_CLEAN, answer=" Mary", distractor=" John")
    tokens = backend.tokenize(IOI_CLEAN, True)
    return [
        PreparedPrompt(
            index=i,
            record=record,
            clean=tokens,
            corrupt=tokens,
            answer=Target(ids=(backend.single_token_id(" Mary"),)),
            distractor=Target(ids=(backend.single_token_id(" John"),)),
        )
        for i in range(n)
    ]


def test_identical_clean_and_corrupt_prompts_are_refused(tiny_backend):
    record = PromptRecord(clean=IOI_CLEAN, corrupt=IOI_CLEAN, answer=" Mary", distractor=" John")
    with pytest.raises(PromptError) as info:
        prepare_prompts(tiny_backend, [record], True)
    assert [i.kind for i in info.value.issues] == ["identical"]


def test_gap_of_zero_is_refused(tiny_backend, project, spec_factory):
    prompts = _same_prompt_twice(tiny_backend)
    with pytest.raises(EngineError, match="normalized effect is undefined"):
        run_engine(spec_factory(), tiny_backend, prompts)
    prompt_gap = spec_factory(metric={"kind": "logit_diff", "normalization": "prompt_gap"})
    with pytest.raises(EngineError, match="no clean–corrupt gap"):
        run_engine(prompt_gap, tiny_backend, prompts)


def test_a_gap_within_its_noise_is_flagged(tiny_backend, spec_factory):
    """Swapping clean and corrupt negates a prompt's gap, so these gaps nearly cancel."""
    forward = PromptRecord(clean=IOI_CLEAN, corrupt=IOI_CORRUPT, answer=" Mary", distractor=" John")
    swapped = PromptRecord(clean=IOI_CORRUPT, corrupt=IOI_CLEAN, answer=" Mary", distractor=" John")
    prompts = prepare_prompts(tiny_backend, [forward, swapped] * 3 + [forward], True)
    result = run_engine(spec_factory(), tiny_backend, prompts)
    assert any("small next to its standard error" in w for w in result.warnings)
    steady = prepare_prompts(tiny_backend, [forward] * 4, True)
    result = run_engine(spec_factory(), tiny_backend, steady)
    assert not any("standard error" in w for w in result.warnings)


def test_batch_size_does_not_change_results(tiny_backend, project, spec_factory):
    """Captures and patched runs are batched; the batch size must not change any number."""
    prompts = _prompts(tiny_backend, project)
    scope = {"kind": "layer_components", "components": ["resid_pre", "attn_out", "mlp_out"]}
    results = [
        run_engine(spec_factory(scope=scope, execution={"batch_size": size}), tiny_backend, prompts)
        for size in (1, 5, 64)
    ]
    for other in results[1:]:
        np.testing.assert_allclose(other.patched, results[0].patched, rtol=0, atol=1e-5)


def test_mean_ablation_at_a_named_position_spans_lengths(tiny_backend, project, spec_factory):
    """At one position, the mean is over every reference prompt at its own S2 token."""
    spec = spec_factory(
        dataset={"path": "datasets/mixed.jsonl"},
        experiment={"kind": "ablation", "baseline": {"kind": "mean", "reference": "corrupt"}},
        scope={
            "kind": "sites",
            "sites": [
                {"kind": "resid_post", "layer": 0, "position": {"kind": "label", "label": "S2"}}
            ],
        },
    )
    prompts = _prompts(tiny_backend, project, "mixed.jsonl")
    assert len({p.length for p in prompts}) > 1
    result = run_engine(spec, tiny_backend, prompts)
    rows = []
    for p in prompts:
        acts = tiny_backend.capture(torch.tensor([p.corrupt.ids]), [("resid_post", 0)])
        rows.append(acts[("resid_post", 0)][0, p.labels["S2"]])
    mean = torch.stack(rows).float().mean(0)
    target = prompts[-1]
    pos = target.labels["S2"]

    def hook(act, hook=None):
        act = act.clone()
        act[:, pos] = mean.to(act.dtype)
        return act

    with torch.no_grad():
        logits = tiny_backend.bridge.run_with_hooks(
            torch.tensor([target.clean.ids]),
            fwd_hooks=[("blocks.0.hook_resid_post", hook)],
            return_type="logits",
        )[0, -1]
    expected = float(logits[target.answer_id] - logits[target.distractor_id])
    assert result.patched[0, len(prompts) - 1] == pytest.approx(expected, abs=1e-4)


def test_resample_at_a_named_position_uses_each_donors_own_position(
    tiny_backend, project, spec_factory
):
    spec = spec_factory(
        dataset={"path": "datasets/mixed.jsonl"},
        experiment={
            "kind": "ablation",
            "baseline": {"kind": "resample", "pool": "clean", "donors": 1, "seed": 4},
        },
        scope={
            "kind": "sites",
            "sites": [
                {"kind": "resid_pre", "layer": 1, "position": {"kind": "label", "label": "S2"}}
            ],
        },
    )
    prompts = _prompts(tiny_backend, project, "mixed.jsonl")
    result = run_engine(spec, tiny_backend, prompts)
    assert result.donors is not None
    # Donors may come from other templates (lengths), since only one position is replaced.
    i = next(k for k, d in enumerate(result.donors) if prompts[d[0]].length != prompts[k].length)
    receiver, donor = prompts[i], prompts[result.donors[i][0]]
    acts = tiny_backend.capture(torch.tensor([donor.clean.ids]), [("resid_pre", 1)])
    value = acts[("resid_pre", 1)][0, donor.labels["S2"]]

    def hook(act, hook=None):
        act = act.clone()
        act[:, receiver.labels["S2"]] = value.to(act.dtype)
        return act

    with torch.no_grad():
        logits = tiny_backend.bridge.run_with_hooks(
            torch.tensor([receiver.clean.ids]),
            fwd_hooks=[("blocks.1.hook_resid_pre", hook)],
            return_type="logits",
        )[0, -1]
    expected = float(logits[receiver.answer_id] - logits[receiver.distractor_id])
    assert result.patched[0, i] == pytest.approx(expected, abs=1e-4)
