"""SAEs on attention heads' outputs (SAELens hook_z): they read z of every head of a layer, side
by side, and a feature's change reaches the residual stream through the attention's output
projection. Checked exactly against GPT-2's Hugging Face modules on a tiny model."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from logogram.analysis import sae_fit_report, token_features_report
from logogram.atp import metric_gradients
from logogram.backends.base import BackendError, Patch
from logogram.backends.saes import read_sae
from logogram.datasets import load_dataset
from logogram.engine import make_scorer, run_experiment
from logogram.prompts import group_by_length, prepare_prompts
from logogram.sae import SAE
from test_sae import _saelens

SAE_REF = {"repo": "local/z", "path": "", "revision": "test"}
LAYER = 1


def _z_sae(folder, d: int, *, exact: bool = False, cfg: dict | None = None) -> SAE:
    """An SAELens SAE on layer 1's hook_z: exact (the positive and negative parts of each
    coordinate of z), or random, so that a feature's decoder row spans every head."""
    g = torch.Generator().manual_seed(5)
    eye = torch.eye(d)
    tensors = (
        {
            "W_enc": torch.cat([eye, -eye], 1),
            "b_enc": torch.zeros(2 * d),
            "W_dec": torch.cat([eye, -eye], 0),
            "b_dec": torch.zeros(d),
        }
        if exact
        else {
            "W_enc": torch.randn(d, 3 * d, generator=g) * 0.4,
            "b_enc": torch.randn(3 * d, generator=g) * 0.1,
            "W_dec": torch.randn(3 * d, d, generator=g) * 0.3,
            "b_dec": torch.randn(d, generator=g) * 0.1,
        }
    )
    _saelens(
        folder,
        cfg
        or {"hook_name": f"blocks.{LAYER}.attn.hook_z", "d_in": d, "apply_b_dec_to_input": True},
        tensors,
    )
    return SAE(repo="local/z", path="", revision="test", params=read_sae(folder))


def _prompts(backend, project):
    return prepare_prompts(backend, load_dataset(project.datasets_dir / "ioi.jsonl"), True)


def _heads(info) -> int:
    return info.n_heads * info.d_head


def test_hook_z_saes_are_read_as_saes_on_every_head_of_a_layer(tmp_path):
    d = 24
    sae = _z_sae(tmp_path / "flat", d)
    assert (sae.site, sae.site_in, sae.layer, sae.params.d_in) == ("head", "head", LAYER, d)
    # SAELens 6 keeps the hook under "metadata" and says it reshapes z.
    newer = _z_sae(
        tmp_path / "newer",
        d,
        cfg={
            "d_in": d,
            "d_sae": 3 * d,
            "reshape_activations": "hook_z",
            "metadata": {"hook_name": f"blocks.{LAYER}.attn.hook_z", "model_name": "gpt2"},
        },
    )
    assert (newer.site, newer.layer) == ("head", LAYER)
    weights = {
        "W_enc": torch.zeros(4, 8),
        "b_enc": torch.zeros(8),
        "W_dec": torch.zeros(8, 4),
        "b_dec": torch.zeros(4),
    }
    for cfg, message in (
        ({"hook_name": "blocks.0.attn.hook_z", "hook_head_index": 2}, "single attention head"),
        ({"hook_name": "blocks.0.hook_resid_pre", "reshape_activations": "hook_z"}, "reshapes"),
        ({"hook_name": "blocks.0.hook_resid_mid"}, "can't map"),
        ({"hook_name": "blocks.0.hook_resid_pre", "finetuning_scaling_factor": True}, "fitted"),
    ):
        folder = tmp_path / message.replace(" ", "-").replace("'", "")
        with pytest.raises(BackendError, match=message):
            read_sae(_saelens(folder, cfg, weights))
    with pytest.raises(BackendError, match="scaling"):
        read_sae(
            _saelens(
                tmp_path / "extra",
                {"hook_name": "blocks.0.hook_resid_pre"},
                {**weights, "scaling": torch.ones(8)},
            )
        )


def test_z_is_read_as_the_output_projection_reads_it(arch_backend, project, tmp_path):
    """The SAE reads every head's z side by side, head after head: exactly the input of GPT-2's
    c_proj. Its features are SAELens's encoding of that input."""
    backend = arch_backend("gpt2", process_weights=False)
    hf = backend.bridge.original_model
    sae = _z_sae(tmp_path / "z", _heads(backend.info))
    group = group_by_length(_prompts(backend, project))[0]
    seen = {}
    handle = hf.transformer.h[LAYER].attn.c_proj.register_forward_pre_hook(
        lambda m, args: seen.__setitem__("in", args[0].detach())
    )
    with torch.no_grad():
        hf(group.clean)
    handle.remove()
    z = backend.capture(group.clean, [("head", LAYER)])[("head", LAYER)]
    torch.testing.assert_close(sae.reads(z), seen["in"], rtol=0, atol=0)
    p = sae.params
    reference = torch.relu((seen["in"] - p.b_dec) @ p.W_enc + p.b_enc)
    f, _ = sae.encode(sae.reads(z))
    torch.testing.assert_close(f, reference)


def test_patching_a_feature_changes_every_head_and_reaches_attn_out_through_W_O(
    arch_backend, project, spec_factory, tmp_path
):
    backend = arch_backend("gpt2", process_weights=False)
    hf = backend.bridge.original_model
    info = backend.info
    sae = _z_sae(tmp_path / "z", _heads(info))
    prompts = _prompts(backend, project)
    group = group_by_length(prompts)[0]
    clean = backend.capture(group.clean, [("head", LAYER)])[("head", LAYER)]
    f_clean, _ = sae.encode(sae.reads(clean))
    last = group.clean.shape[1] - 1
    feature = int(f_clean[:, last].sum(0).argmax())
    spec = spec_factory(
        model={"id": info.id, "revision": "test", "device": "cpu"},
        experiment={"kind": "activation_patching", "direction": "clean_to_corrupt"},
        scope={
            "kind": "sites",
            "sites": [
                {
                    "kind": "sae_feature",
                    "layer": LAYER,
                    "feature": feature,
                    "position": {"kind": "last"},
                }
            ],
        },
        sae=SAE_REF,
    )
    result = run_experiment(spec, backend, prompts, sae=sae)
    corrupt = backend.capture(group.corrupt, [("head", LAYER), ("attn_out", LAYER)])
    z = corrupt[("head", LAYER)]
    f_corrupt, _ = sae.encode(sae.reads(z))
    change = (f_clean - f_corrupt)[:, last, feature]  # [B]
    row = sae.params.W_dec[feature]
    assert bool((row.reshape(info.n_heads, info.d_head).abs().sum(1) > 0).all())  # every head

    answers = torch.tensor([prompts[i].answer_id for i in group.members])
    distractors = torch.tensor([prompts[i].distractor_id for i in group.members])
    rows = torch.arange(len(group.members))

    def logit_diff(patch):
        logits = backend.final_logits(group.corrupt, patch)
        return (logits[rows, answers] - logits[rows, distractors]).double().numpy()

    # z of every head at the last token moves by the change times the decoder row.
    values = z.clone()
    values[:, last] += change[:, None, None] * row.reshape(info.n_heads, info.d_head)
    mask = torch.zeros(z.shape[:3], dtype=torch.bool)
    mask[:, last] = True
    np.testing.assert_allclose(
        result.patched[0, group.members],
        logit_diff(Patch(kind="head", layer=LAYER, values=values, mask=mask)),
        atol=1e-4,
    )
    # The same as moving the attention output by the change times the row through c_proj.
    through = row @ hf.transformer.h[LAYER].attn.c_proj.weight  # Conv1D: [nx, nf]
    moved = corrupt[("attn_out", LAYER)][:, last] + change[:, None] * through
    np.testing.assert_allclose(
        result.patched[0, group.members],
        logit_diff(
            Patch(
                kind="attn_out", layer=LAYER, values=moved, positions=torch.full((len(rows),), last)
            )
        ),
        atol=1e-4,
    )


def test_attribution_patching_of_z_features(tiny_backend, project, spec_factory, tmp_path):
    """Each estimate is the feature's change times the gradient with respect to z (every head,
    side by side) along its decoder row; for an exact SAE they add up to the whole site's."""
    backend = tiny_backend
    prompts = _prompts(backend, project)
    estimate = {"kind": "attribution_patching", "direction": "clean_to_corrupt"}
    exact = _z_sae(tmp_path / "exact", _heads(backend.info), exact=True)
    every = run_experiment(
        spec_factory(
            experiment=estimate,
            scope={"kind": "features", "position": {"kind": "all"}, "top": 5},
            sae=SAE_REF,
        ),
        backend,
        prompts,
        sae=exact,
    )
    info = every.extra["features"]
    assert info["features_estimate"] == pytest.approx(info["site_estimate"], rel=1e-5, abs=1e-6)
    assert info["fit"]["variance_explained"] == pytest.approx(1.0, abs=1e-5)
    assert info["skip_estimate"] is None

    sae = _z_sae(tmp_path / "z", _heads(backend.info))
    sites = [
        {"kind": "sae_feature", "layer": LAYER, "feature": i, "position": {"kind": "all"}}
        for i in range(4)
    ]
    spec = spec_factory(experiment=estimate, scope={"kind": "sites", "sites": sites}, sae=SAE_REF)
    result = run_experiment(spec, backend, prompts, sae=sae)
    group = group_by_length(prompts)[0]
    keys = [("head", LAYER)]
    src = backend.capture(group.clean, keys)
    acts, grads = metric_gradients(
        backend, make_scorer(spec, prompts), group.corrupt, group.members, keys
    )
    f_src, _ = sae.encode(sae.reads(src[("head", LAYER)].float()))
    f_rec, _ = sae.encode(sae.reads(acts[("head", LAYER)].float()))
    g = sae.flat(grads[("head", LAYER)].float())
    for i in range(4):
        expected = ((f_src[..., i] - f_rec[..., i]) * (g @ sae.params.W_dec[i])).sum(1)
        np.testing.assert_allclose(
            result.delta[i, group.members], expected.double().numpy(), rtol=1e-4, atol=1e-6
        )


def test_an_exact_z_sae_spliced_in_changes_nothing(tiny_backend, project, tmp_path):
    from logogram.datasets import load_dataset

    sae = _z_sae(tmp_path / "exact", _heads(tiny_backend.info), exact=True)
    records = load_dataset(project.datasets_dir / "ioi.jsonl")
    fit = sae_fit_report(tiny_backend, sae, records, prepend_bos=True, batch_size=8)
    assert fit["variance_explained"] == pytest.approx(1.0, abs=1e-5)
    assert fit["spliced_logit_diff"] == pytest.approx(fit["logit_diff"], abs=1e-4)
    tokens = token_features_report(
        tiny_backend, sae, records, index=1, which="clean", prepend_bos=True
    )
    assert len(tokens["features"]) == len(tokens["tokens"])
    assert all(len(row) > 0 for row in tokens["features"][1:])


def test_saelens_configs_are_read_from_json(tmp_path):
    folder = tmp_path / "json"
    _saelens(
        folder,
        {"hook_name": "blocks.3.attn.hook_z", "model_name": "gpt2-small"},
        {
            "W_enc": torch.zeros(6, 4),
            "b_enc": torch.zeros(4),
            "W_dec": torch.zeros(4, 6),
            "b_dec": torch.zeros(6),
        },
    )
    assert json.loads((folder / "cfg.json").read_text())["hook_name"].endswith("hook_z")
    params = read_sae(folder)
    assert params.note == "gpt2-small · blocks.3.attn.hook_z"
