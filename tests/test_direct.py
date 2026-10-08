"""Direct logit attribution: exact terms, refused where undefined, and stored like any run."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from logogram.datasets import load_dataset
from logogram.engine import run_experiment
from logogram.prompts import group_by_length, prepare_prompts
from logogram.results import compute_stats
from logogram.runner import run_spec
from logogram.sites import ScopeError

LAST = {"kind": "last"}

# The brute-force checks below hold the final normalization's scale with a hook, which TransformerLens
# warns about; holding it is the point.
pytestmark = pytest.mark.filterwarnings("ignore:A forward hook edited hook_scale")


def _prompts(backend, project):
    return prepare_prompts(backend, load_dataset(project.datasets_dir / "ioi.jsonl"), True)


def _batch(backend, project):
    prompts = _prompts(backend, project)
    group = group_by_length(prompts)[0]
    answers = torch.tensor([prompts[i].answer_id for i in group.members])
    distractors = torch.tensor([prompts[i].distractor_id for i in group.members])
    return group.clean, answers, distractors


def _logit_diff(logits, answers, distractors):
    rows = torch.arange(logits.shape[0])
    return (logits[rows, answers] - logits[rows, distractors]).double()


@pytest.mark.parametrize("arch", ["gpt2", "llama", "gpt_neox"])
def test_a_direct_effect_is_what_removing_the_output_from_the_final_residual_does(
    tiny_backend, arch_backend, project, arch
):
    """Remove one component's output from the last residual stream, with the final normalization's
    scale held: the logit difference drops by exactly that component's direct effect."""
    backend = tiny_backend if arch == "gpt2" else arch_backend(arch)
    bridge = backend.bridge
    last = backend.info.n_layers - 1
    tokens, answers, distractors = _batch(backend, project)
    terms = backend.direct_effects(tokens, answers, distractors, heads=True)

    final = f"blocks.{last}.hook_resid_post"
    with torch.no_grad():
        logits, cache = bridge.run_with_cache(
            tokens,
            names_filter=[
                f"blocks.{last}.hook_mlp_out",
                f"blocks.{last}.attn.hook_z",
                "ln_final.hook_scale",
            ],
        )
    full = _logit_diff(logits[:, -1], answers, distractors)
    scale = cache["ln_final.hook_scale"]
    z = cache[f"blocks.{last}.attn.hook_z"]
    head_1 = torch.einsum("bpd,dm->bpm", z[:, :, 1], bridge.blocks[last].attn.W_O[1])

    def without(output):
        def remove(act, hook=None):
            return act - output

        def hold(value, hook=None):
            return scale

        with torch.no_grad():
            out = bridge.run_with_hooks(
                tokens,
                fwd_hooks=[(final, remove), ("ln_final.hook_scale", hold)],
                return_type="logits",
            )
        return full - _logit_diff(out[:, -1], answers, distractors)

    np.testing.assert_allclose(
        without(cache[f"blocks.{last}.hook_mlp_out"]).numpy(),
        terms["mlp_out"][:, last].numpy(),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        without(head_1).numpy(), terms["head"][:, last, 1].numpy(), atol=1e-4
    )
    np.testing.assert_allclose(terms["logit_diff"].numpy(), full.numpy(), atol=1e-4)


def test_heads_add_up_to_the_attention_output_without_an_output_bias(arch_backend, project):
    backend = arch_backend("llama")  # no b_O, so the heads are all of the attention output
    terms = backend.direct_effects(*_batch(backend, project), heads=True)
    np.testing.assert_allclose(terms["head"].sum(2).numpy(), terms["attn_out"].numpy(), atol=1e-6)


@pytest.mark.parametrize("normalization", ["dataset_gap", "prompt_gap"])
def test_the_parts_add_up_to_the_whole_logit_difference(
    tiny_backend, project, spec_factory, normalization
):
    spec = spec_factory(
        experiment={"kind": "direct_logit_attribution", "prompts": "clean"},
        scope={"kind": "layer_components", "components": ["attn_out", "mlp_out"], "position": LAST},
        metric={"kind": "logit_diff", "normalization": normalization},
    )
    result = run_experiment(spec, tiny_backend, _prompts(tiny_backend, project))
    split = result.extra["direct"]
    stats = compute_stats(spec, result)
    assert result.measure == "attribution"
    assert np.isnan(result.patched_ld).all()
    total = stats.delta_mean.sum() + split["embeddings"] + split["biases"]
    assert total == pytest.approx(split["logit_diff"], abs=1e-6)
    assert split["logit_diff"] == pytest.approx(float(result.gap.mean()), abs=1e-4)
    if normalization == "dataset_gap":
        # Shares of the mean logit difference: with embeddings and biases, they add up to one.
        share = (split["embeddings"] + split["biases"]) / split["logit_diff"]
        assert stats.effect_mean.sum() + share == pytest.approx(1.0, abs=1e-6)
    assert int(stats.sign_flips.sum()) == 0


def test_it_is_refused_where_a_direct_effect_is_undefined(
    tiny_backend, arch_backend, project, spec_factory
):
    dla = {"kind": "direct_logit_attribution", "prompts": "clean"}
    prompts = _prompts(tiny_backend, project)
    with pytest.raises(ScopeError, match="last token"):
        run_experiment(
            spec_factory(experiment=dla, scope={"kind": "heads", "position": {"kind": "all"}}),
            tiny_backend,
            prompts,
        )
    with pytest.raises(ScopeError, match="state of the residual stream"):
        resid = {"kind": "layer_components", "components": ["resid_pre"], "position": LAST}
        run_experiment(spec_factory(experiment=dla, scope=resid), tiny_backend, prompts)
    olmo = arch_backend("olmo2")  # normalizes the attention output after combining the heads
    with pytest.raises(ScopeError, match="sum of its heads"):
        run_experiment(
            spec_factory(experiment=dla, scope={"kind": "heads", "position": LAST}),
            olmo,
            _prompts(olmo, project),
        )
    components = {
        "kind": "layer_components",
        "components": ["attn_out", "mlp_out"],
        "position": LAST,
    }
    run_experiment(spec_factory(experiment=dla, scope=components), olmo, _prompts(olmo, project))
    gemma = arch_backend("gemma2")
    with pytest.raises(ScopeError, match="soft-caps"):
        run_experiment(
            spec_factory(experiment=dla, scope=components), gemma, _prompts(gemma, project)
        )


def test_a_named_position_at_the_last_token_counts_as_the_last_token(
    tiny_backend, project, spec_factory
):
    spec = spec_factory(
        experiment={"kind": "direct_logit_attribution", "prompts": "corrupt"},
        scope={"kind": "heads", "position": {"kind": "label", "label": "end"}},
    )
    prompts = _prompts(tiny_backend, project)
    assert all(p.labels["end"] == p.length - 1 for p in prompts)
    result = run_experiment(spec, tiny_backend, prompts)
    assert result.receiver == result.reference == "corrupt"


def test_a_run_stores_the_split_and_compares_with_patching(tiny_backend, project, spec_factory):
    dla = run_spec(
        spec_factory(
            experiment={"kind": "direct_logit_attribution", "prompts": "clean"},
            scope={"kind": "heads", "position": LAST},
        ),
        project,
        backend=tiny_backend,
    )
    patching = run_spec(
        spec_factory(
            experiment={"kind": "activation_patching", "direction": "corrupt_to_clean"},
            scope={"kind": "heads", "position": LAST},
        ),
        project,
        backend=tiny_backend,
    )
    assert dla.status == patching.status == "finished"
    summary = json.loads((dla.folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["measure"] == "attribution"
    assert set(summary["direct"]) == {
        "prompts",
        "logit_diff",
        "embeddings",
        "attention",
        "mlp",
        "biases",
    }
    assert summary["sites"][0]["patched_logit_diff"] is None
    assert summary["sites"][0]["answer_prob"] is None
    assert "contribution" in summary["metric"]["normalized_effect"]

    from logogram.compare import compare_summaries
    from logogram.runs import site_detail

    other = json.loads((patching.folder / "summary.json").read_text(encoding="utf-8"))
    comparison = compare_summaries(summary, other)
    assert comparison["n_common"] == tiny_backend.info.n_layers * tiny_backend.info.n_heads
    detail = site_detail(project, dla.run_id, 0)
    assert detail["prompts"][0]["patched_logit_diff"] is None
    assert detail["prompts"][0]["flipped"] is False
