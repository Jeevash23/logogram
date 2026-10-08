"""Attribution patching: correct gradients, exactly zero for no change, and verified by patching."""

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
from logogram.stats import spearman
from logogram.verify import verification_spec

ATP = {"kind": "attribution_patching", "direction": "clean_to_corrupt"}


def _prompts(backend, project):
    return prepare_prompts(backend, load_dataset(project.datasets_dir / "ioi.jsonl"), True)


def _batch(backend, project):
    prompts = _prompts(backend, project)
    group = group_by_length(prompts)[0]
    answers = torch.tensor([prompts[i].answer_id for i in group.members])
    distractors = torch.tensor([prompts[i].distractor_id for i in group.members])
    return group.corrupt, answers, distractors


def _logit_diff(backend, tokens, answers, distractors, hooks):
    with torch.no_grad():
        logits = backend.bridge.run_with_hooks(tokens, fwd_hooks=hooks, return_type="logits")
    rows = torch.arange(tokens.shape[0])
    last = logits[:, -1].double()
    return last[rows, answers] - last[rows, distractors]


@pytest.fixture(scope="module")
def smooth_backend(tmp_path_factory):
    """Tiny models with small weights: smooth enough that finite differences measure slopes.
    (The large-weight test models are so curved that, in float32, they can't be differentiated
    numerically to a strict tolerance.)"""
    from conftest import build_tiny_architecture
    from logogram.backends.transformer_lens import TransformerLensBackend, boot_local

    def get(arch):
        folder = tmp_path_factory.mktemp(f"smooth-{arch}")
        build_tiny_architecture(folder, arch, initializer_range=0.05)
        return TransformerLensBackend.from_bridge(
            boot_local(folder, device="cpu", dtype="float32"),
            model_id=f"smooth-{arch}",
            revision="test",
            dtype="float32",
            process_weights=True,
        )

    return get


@pytest.mark.parametrize("arch", ["gpt2", "llama", "gpt_neox"])
def test_the_gradients_match_finite_differences(tiny_backend, smooth_backend, project, arch):
    """Nudge each site along a fixed direction: the logit difference changes by the gradient's
    projection on it. For resid_mid the nudge goes into the residual stream (through attn_out)."""
    backend = tiny_backend if arch == "gpt2" else smooth_backend(arch)
    tokens, answers, distractors = _batch(backend, project)
    sites = [("resid_pre", 1), ("head", 1), ("mlp_out", 0), ("resid_post", 0)]
    if "resid_mid" in backend.info.site_kinds:
        sites.append(("resid_mid", 1))
    acts, grads = backend.gradients(tokens, answers, distractors, sites)
    generator = torch.Generator().manual_seed(0)
    # Tiny random models are strongly curved; a five-point stencil with a small step measures the
    # slope rather than the curvature.
    eps = 1e-4
    for kind, layer in sites:
        direction = torch.randn(acts[(kind, layer)].shape, generator=generator)
        hook = {
            "resid_pre": f"blocks.{layer}.hook_resid_pre",
            "head": f"blocks.{layer}.attn.hook_z",
            "mlp_out": f"blocks.{layer}.hook_mlp_out",
            "resid_post": f"blocks.{layer}.hook_resid_post",
            "resid_mid": f"blocks.{layer}.hook_attn_out",
        }[kind]

        def nudge(sign, direction=direction, hook=hook):
            def fn(act, hook=None):
                return act + sign * eps * direction.to(act.dtype)

            return [(hook, fn)]

        f = {
            k: _logit_diff(backend, tokens, answers, distractors, nudge(k)) for k in (-2, -1, 1, 2)
        }
        numeric = ((-f[2] + 8 * f[1] - 8 * f[-1] + f[-2]) / (12 * eps)).numpy()
        analytic = (grads[(kind, layer)].double() * direction.double()).flatten(1).sum(1).numpy()
        scale = max(1e-3, float(np.abs(numeric).max()))
        np.testing.assert_allclose(
            analytic, numeric, rtol=1e-2, atol=1e-2 * scale, err_msg=f"{kind} {layer}"
        )


def test_no_difference_estimates_no_effect(tiny_backend, project, spec_factory):
    for scope in (
        {"kind": "heads", "position": {"kind": "all"}},
        {"kind": "layer_position", "site": "resid_mid", "positions": "each"},
        {"kind": "layer_components", "position": {"kind": "last"}},
    ):
        spec = spec_factory(experiment=ATP, scope=scope)
        result = run_experiment(
            spec,
            tiny_backend,
            _prompts(tiny_backend, project),
            receiver_override="corrupt",
            source_override="corrupt",
        )
        assert result.measure == "estimate"
        np.testing.assert_allclose(result.delta, 0.0, atol=1e-12)


@pytest.mark.parametrize("arch", ["llama", "gpt_neox", "gemma2", "olmo2", "qwen2"])
def test_it_runs_on_every_family(arch_backend, project, spec_factory, arch):
    backend = arch_backend(arch)
    spec = spec_factory(experiment=ATP, scope={"kind": "heads", "position": {"kind": "all"}})
    result = run_experiment(spec, backend, _prompts(backend, project))
    assert np.isfinite(result.delta).all()
    assert np.abs(result.delta).max() > 0


def test_estimates_track_real_patching(tiny_backend, project, spec_factory):
    """A first-order estimate isn't exact, but on this model it should rank heads like patching."""
    prompts = _prompts(tiny_backend, project)
    scope = {"kind": "heads", "position": {"kind": "last"}}
    estimated = run_experiment(spec_factory(experiment=ATP, scope=scope), tiny_backend, prompts)
    patched = run_experiment(
        spec_factory(
            experiment={"kind": "activation_patching", "direction": "clean_to_corrupt"}, scope=scope
        ),
        tiny_backend,
        prompts,
    )
    a = estimated.delta.mean(1)
    b = (patched.patched_ld - patched.receiver_ld[None, :]).mean(1)
    assert spearman(a, b) > 0.5


def test_a_run_is_stored_as_an_estimate_and_verified_by_patching(
    tiny_backend, project, spec_factory
):
    spec = spec_factory(experiment=ATP, scope={"kind": "heads", "position": {"kind": "last"}})
    outcome = run_spec(spec, project, backend=tiny_backend)
    assert outcome.status == "finished"
    summary = json.loads((outcome.folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["measure"] == "estimate"
    assert summary["metric"]["normalized_effect"].startswith("estimated")
    assert summary["sites"][0]["answer_prob"] is None
    stats = compute_stats(spec, run_experiment(spec, tiny_backend, _prompts(tiny_backend, project)))
    assert np.isfinite(stats.effect_mean).all()

    check = verification_spec(spec, summary, top=3)
    assert check.experiment.kind == "activation_patching"
    assert check.experiment.direction == "clean_to_corrupt"
    strongest = sorted(summary["sites"], key=lambda s: -abs(s["effect"]["mean"]))[:3]
    assert [(s.layer, s.head) for s in check.scope.sites] == [
        (s["layer"], s["head"]) for s in strongest
    ]
    verified = run_spec(check, project, backend=tiny_backend)
    assert verified.status == "finished"

    with pytest.raises(ValueError, match="Only attribution patching"):
        verification_spec(check, summary, top=3)
