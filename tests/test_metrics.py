"""Metrics beyond the logit difference: log-probabilities, probabilities, KL divergence, answers
of several tokens and sets of answers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from logogram.datasets import PromptRecord, load_dataset, write_dataset
from logogram.engine import compute_baselines, patched_forward, run_engine, run_experiment
from logogram.metrics import Scorer, Target, extend_tokens
from logogram.prompts import PromptError, group_by_length, prepare_prompts
from logogram.results import compute_stats
from logogram.spec import (
    KLMetric,
    LogitDiffMetric,
    LogProbDiffMetric,
    LogProbMetric,
    ProbDiffMetric,
    ProbMetric,
)

TOL = 1e-4
METRICS = {
    "logit_diff": {"kind": "logit_diff", "normalization": "dataset_gap"},
    "logprob_diff": {"kind": "logprob_diff", "normalization": "dataset_gap"},
    "logprob": {"kind": "logprob", "normalization": "dataset_gap"},
    # The tiny model's answer probabilities are tiny, so their dataset gap is below what Logogram
    # normalizes by; each prompt's own gap is large enough.
    "prob": {"kind": "prob", "normalization": "prompt_gap"},
    "prob_diff": {"kind": "prob_diff", "normalization": "prompt_gap"},
    "kl_clean": {"kind": "kl", "target": "clean", "normalization": "dataset_gap"},
    "kl_corrupt": {"kind": "kl", "target": "corrupt", "normalization": "dataset_gap"},
}


def _prompts(backend, project, name="ioi.jsonl", continuations=False):
    return prepare_prompts(
        backend, load_dataset(project.datasets_dir / name), True, continuations=continuations
    )


def _two_token_dataset(project):
    """IOI prompts whose answer and distractor are each a name said twice: continuations of two
    tokens, read by teacher forcing."""
    records = []
    for r in load_dataset(project.datasets_dir / "ioi.jsonl"):
        records.append(
            r.model_copy(update={"answer": r.answer + r.answer, "distractor": r.distractor})
        )
    # Half the distractors have two tokens too, so both continuations need their own pass.
    for i in range(0, len(records), 2):
        r = records[i]
        records[i] = r.model_copy(update={"distractor": r.distractor + r.answer})
    write_dataset(project.datasets_dir / "twice.jsonl", records)
    return "twice.jsonl"


def _log_softmax_at(backend, tokens, position):
    logits = backend.logits(tokens, None, tokens.shape[1])
    return torch.log_softmax(logits[:, position].double(), dim=-1)


def test_single_tokens_reproduce_the_logit_difference_exactly(tiny_backend, project):
    prompts = _prompts(tiny_backend, project)
    group = group_by_length(prompts)[0]
    tokens = group.clean
    scorer = Scorer(LogitDiffMetric(normalization="dataset_gap"), prompts)
    scores = scorer.score(patched_forward(tiny_backend, None), tokens, group.members)
    logits = tiny_backend.final_logits(tokens)
    rows = torch.arange(len(group.members))
    a = torch.tensor([prompts[i].answer_id for i in group.members])
    d = torch.tensor([prompts[i].distractor_id for i in group.members])
    expected = (logits[rows, a] - logits[rows, d]).double().numpy()
    np.testing.assert_array_equal(scores.metric, expected)
    np.testing.assert_array_equal(scores.pref, expected)
    prob = torch.log_softmax(logits, -1)[rows, a].exp().double().numpy()
    np.testing.assert_array_equal(scores.prob, prob)
    # The log-probability difference is the same number, read through the softmax.
    logprob = Scorer(LogProbDiffMetric(normalization="dataset_gap"), prompts)
    np.testing.assert_allclose(
        logprob.score(patched_forward(tiny_backend, None), tokens, group.members).metric,
        expected,
        atol=1e-5,
    )


def test_continuations_are_read_token_by_token(tiny_backend, project):
    name = _two_token_dataset(project)
    with pytest.raises(PromptError, match="several tokens"):
        _prompts(tiny_backend, project, name)  # the logit difference reads one position
    prompts = _prompts(tiny_backend, project, name, continuations=True)
    group = group_by_length(prompts)[0]
    rows = group.members
    for metric in (
        LogProbMetric(normalization="dataset_gap"),
        ProbMetric(normalization="dataset_gap"),
    ):
        scores = Scorer(metric, prompts).score(
            patched_forward(tiny_backend, None), group.clean, rows
        )
        for b, p in enumerate(rows):
            answer = prompts[p].answer
            assert len(answer.ids) == 2 and not answer.alternatives
            # Teacher forcing by hand: the prompt, then the answer's first token.
            tokens = torch.tensor([[*prompts[p].clean.ids, answer.ids[0]]])
            lp = torch.log_softmax(tiny_backend.logits(tokens, None, 2).double(), -1)
            expected = float(lp[0, 0, answer.ids[0]] + lp[0, 1, answer.ids[1]])
            # A batch and a pass of its own round differently in float32.
            value = expected if metric.kind == "logprob" else np.exp(expected)
            assert scores.metric[b] == pytest.approx(value, rel=1e-4, abs=1e-6)
            assert scores.prob[b] == pytest.approx(np.exp(expected), rel=1e-4)
    # The preference reads the distractor's continuation in a pass of its own.
    scores = Scorer(LogProbDiffMetric(normalization="dataset_gap"), prompts).score(
        patched_forward(tiny_backend, None), group.clean, rows
    )
    for b, p in enumerate(rows):
        logp = {}
        for which in ("answer", "distractor"):
            target = getattr(prompts[p], which)
            tokens = torch.tensor([prompts[p].clean.ids + list(target.ids[:-1])])
            lp = torch.log_softmax(tiny_backend.logits(tokens, None, len(target.ids)).double(), -1)[
                0
            ]
            logp[which] = float(sum(lp[t, i] for t, i in enumerate(target.ids)))
        assert scores.metric[b] == pytest.approx(logp["answer"] - logp["distractor"], abs=1e-4)
        assert scores.pref[b] == pytest.approx(scores.metric[b])


def test_padding_continuations_changes_nothing(tiny_backend, project):
    """Rows with shorter continuations are padded at the end, which a causal model never reads
    back: their numbers are the same as in a batch of their own."""
    prompts = _prompts(tiny_backend, project, _two_token_dataset(project), continuations=True)
    group = group_by_length(prompts)[0]
    scorer = Scorer(LogProbDiffMetric(normalization="dataset_gap"), prompts)
    together = scorer.score(patched_forward(tiny_backend, None), group.clean, group.members)
    for b, p in enumerate(group.members[:4]):
        alone = scorer.score(patched_forward(tiny_backend, None), group.clean[b : b + 1], [p])
        assert alone.metric[0] == pytest.approx(together.metric[b], abs=1e-5)


def test_sets_of_answers_sum_their_probabilities(tiny_backend, project):
    records = load_dataset(project.datasets_dir / "ioi.jsonl")
    others = [" Alice", " Bob", " Tom"]
    sets = [
        r.model_copy(
            update={
                "answer": [r.answer, *[o for o in others if o not in (r.answer, r.distractor)][:1]],
                "distractor": [r.distractor],
            }
        )
        for r in records
    ]
    prompts = prepare_prompts(tiny_backend, sets, True)
    group = group_by_length(prompts)[0]
    scores = Scorer(ProbDiffMetric(normalization="dataset_gap"), prompts).score(
        patched_forward(tiny_backend, None), group.clean, group.members
    )
    probs = torch.softmax(tiny_backend.final_logits(group.clean).double(), -1)
    for b, p in enumerate(group.members):
        answer, distractor = prompts[p].answer, prompts[p].distractor
        assert answer.alternatives and len(answer.ids) == 2
        expected = float(probs[b, list(answer.ids)].sum() - probs[b, list(distractor.ids)].sum())
        assert scores.metric[b] == pytest.approx(expected, abs=1e-6)
        assert scores.prob[b] == pytest.approx(float(probs[b, list(answer.ids)].sum()), abs=1e-6)


def test_a_set_and_a_distractor_sharing_a_token_is_refused(tiny_backend):
    record = PromptRecord(
        clean="When Mary and John went to the store, John gave a drink to",
        corrupt="When Mary and John went to the store, Mary gave a drink to",
        answer=[" Mary", " John"],
        distractor=" John",
    )
    with pytest.raises(PromptError, match="share"):
        prepare_prompts(tiny_backend, [record], True)


def test_kl_of_a_prompt_from_itself_is_exactly_zero(tiny_backend, project):
    prompts = _prompts(tiny_backend, project)
    groups = group_by_length(prompts)
    for target in ("clean", "corrupt"):
        scorer = Scorer(KLMetric(target=target, normalization="dataset_gap"), prompts)
        base = compute_baselines(tiny_backend, prompts, groups, 5, scorer=scorer)
        assert (base.metric(target) == 0).all()
        other = "corrupt" if target == "clean" else "clean"
        assert (base.metric(other) > 0).all()
        # Against a direct computation of the divergence.
        group = groups[0]
        lt = torch.log_softmax(tiny_backend.final_logits(getattr(group, target)).double(), -1)
        lx = torch.log_softmax(tiny_backend.final_logits(getattr(group, other)).double(), -1)
        expected = (lt.exp() * (lt - lx)).sum(-1).numpy()
        np.testing.assert_allclose(base.metric(other)[group.members], expected, rtol=1e-6)


@pytest.mark.parametrize("metric", list(METRICS))
@pytest.mark.parametrize("direction", ["clean_to_corrupt", "corrupt_to_clean"])
def test_patching_a_whole_layer_gives_effect_one_for_every_metric(
    tiny_backend, project, spec_factory, metric, direction
):
    """The sanity check of AGENTS.md, for each metric: the patched run is the source run."""
    spec = spec_factory(
        experiment={"kind": "activation_patching", "direction": direction},
        scope={
            "kind": "sites",
            "sites": [{"kind": "resid_pre", "layer": 1, "position": {"kind": "all"}}],
        },
        metric=METRICS[metric],
    )
    prompts = _prompts(tiny_backend, project)
    result = run_engine(spec, tiny_backend, prompts)
    np.testing.assert_allclose(compute_stats(spec, result).effect_mean, 1.0, atol=TOL)
    # And patching the receiver into itself changes nothing.
    receiver = "corrupt" if direction == "clean_to_corrupt" else "clean"
    same = run_engine(spec, tiny_backend, prompts, source_override=receiver)
    np.testing.assert_allclose(same.patched[0], same.receiver_metric, atol=1e-5)


@pytest.mark.parametrize("metric", ["logprob_diff", "logprob", "kl_clean"])
def test_continuations_pass_the_sanity_checks(tiny_backend, project, spec_factory, metric):
    name = _two_token_dataset(project)
    spec = spec_factory(
        dataset={"path": f"datasets/{name}"},
        scope={
            "kind": "sites",
            "sites": [
                {"kind": "resid_pre", "layer": 1, "position": {"kind": "all"}},
                {"kind": "head", "layer": 0, "head": 1, "position": {"kind": "all"}},
            ],
        },
        metric=METRICS[metric],
    )
    prompts = _prompts(tiny_backend, project, name, continuations=True)
    result = run_engine(spec, tiny_backend, prompts)
    stats = compute_stats(spec, result)
    assert stats.effect_mean[0] == pytest.approx(1.0, abs=TOL)
    np.testing.assert_allclose(result.patched[0], result.reference_metric, atol=1e-5)
    assert np.isfinite(stats.effect_mean[1])


def test_attribution_patching_follows_the_metric(tiny_backend, project, spec_factory):
    """The gradient of a continuation's log-probability, through both passes, matches a finite
    difference along the source - receiver direction at a small step."""
    from logogram.atp import metric_gradients

    name = _two_token_dataset(project)
    prompts = _prompts(tiny_backend, project, name, continuations=True)
    group = group_by_length(prompts)[0]
    rows = group.members[:3]
    tokens = group.corrupt[:3]
    scorer = Scorer(LogProbDiffMetric(normalization="dataset_gap"), prompts)
    key = ("resid_post", 0)
    acts, grads = metric_gradients(tiny_backend, scorer, tokens, rows, [key])
    direction = torch.randn(acts[key].shape, generator=torch.Generator().manual_seed(0))
    eps = 1e-3

    def at(step):
        def forward(toks, keep):
            def nudge(act, hook=None):
                act = act.clone()
                act[:, : tokens.shape[1]] += step * direction.to(act.dtype)
                return act

            with torch.no_grad():
                logits = tiny_backend.bridge.run_with_hooks(
                    toks, fwd_hooks=[("blocks.0.hook_resid_post", nudge)], return_type="logits"
                )
            return logits[:, -keep:].float()

        return scorer.score(forward, tokens, rows).metric

    numeric = (at(eps) - at(-eps)) / (2 * eps)
    analytic = (grads[key].double() * direction.double()).flatten(1).sum(1).numpy()
    np.testing.assert_allclose(analytic, numeric, rtol=0.05, atol=5e-3)


def test_integrated_gradients_at_the_input_add_up_to_the_whole_effect(
    tiny_backend, project, spec_factory
):
    """Integrated gradients on the input embeddings are complete: at the stream entering the first
    layer, the estimate is the whole change from receiver to source (normalized effect 1), up to
    the midpoint rule's error. A single gradient isn't."""
    sites = [{"kind": "resid_pre", "layer": 0, "position": {"kind": "all"}}]
    prompts = _prompts(tiny_backend, project)

    def estimate(method, steps):
        spec = spec_factory(
            experiment={
                "kind": "attribution_patching",
                "direction": "clean_to_corrupt",
                "method": method,
                "steps": steps,
            },
            scope={"kind": "sites", "sites": sites},
        )
        result = run_experiment(spec, tiny_backend, prompts)
        assert result.extra["attribution"] == {"method": method, "steps": steps}
        return float(compute_stats(spec, result).effect_mean[0])

    assert estimate("integrated_gradients", 32) == pytest.approx(1.0, abs=0.03)
    assert abs(estimate("gradient", None) - 1.0) > 0.05


def test_direct_effects_need_the_logit_difference(tiny_backend, project, spec_factory):
    from logogram.sites import ScopeError

    spec = spec_factory(
        experiment={"kind": "direct_logit_attribution", "prompts": "clean"},
        scope={"kind": "heads", "position": {"kind": "last"}},
        metric=METRICS["prob"],
    )
    with pytest.raises(ScopeError, match="logit difference"):
        run_experiment(spec, tiny_backend, _prompts(tiny_backend, project))


def test_extend_tokens_pads_at_the_end():
    tokens = torch.tensor([[5, 6], [7, 8]])
    out = extend_tokens(tokens, [(1, 2, 3), (4,)], 3)
    assert out.tolist() == [[5, 6, 1, 2], [7, 8, 0, 0]]
    assert (
        Target(ids=(1, 2)).positions == 2 and Target(ids=(1, 2), alternatives=True).positions == 1
    )
