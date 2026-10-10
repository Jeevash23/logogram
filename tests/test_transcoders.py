"""Gemma Scope 2 files, transcoders (an MLP's input to its output) and SAEs on attention heads'
outputs, checked against hand-built edits of the tiny model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from logogram.backends.base import BackendError, Patch
from logogram.backends.saes import read_sae, site_of_hf_point
from logogram.datasets import load_dataset
from logogram.engine import run_experiment
from logogram.prompts import group_by_length, prepare_prompts
from logogram.runner import run_spec
from logogram.sae import SAE
from logogram.sites import ScopeError

SAE_REF = {"repo": "local/gemma-scope", "path": "", "revision": "test"}


def _gemma_scope(folder: Path, cfg: dict, tensors: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(folder / "params.safetensors"))
    return folder


def _params(d_in: int, d_out: int, width: int, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "w_enc": torch.randn(d_in, width, generator=g) * 0.5,
        "b_enc": torch.randn(width, generator=g) * 0.1,
        "threshold": torch.full((width,), 0.05),
        "w_dec": torch.randn(width, d_out, generator=g) * 0.2,
        "b_dec": torch.randn(d_out, generator=g) * 0.1,
    }


def _load(folder: Path) -> SAE:
    return SAE(repo="local/gemma-scope", path="", revision="test", params=read_sae(folder))


def test_hugging_face_points_map_to_sites():
    assert site_of_hf_point("model.layers.12.output") == ("resid_post", 12)
    assert site_of_hf_point("model.layers.3.self_attn.o_proj.input") == ("head", 3)
    assert site_of_hf_point("model.layers.3.post_feedforward_layernorm.output") == ("mlp_out", 3)
    assert site_of_hf_point("model.layers.3.pre_feedforward_layernorm.output") == ("mlp_in", 3)
    assert site_of_hf_point("transformer.h.1.ln_2.output") == ("mlp_in", 1)
    assert site_of_hf_point("transformer.h.1.mlp.output") == ("mlp_out", 1)
    with pytest.raises(BackendError, match="can't map"):
        site_of_hf_point("model.embed_tokens.output")


def test_gemma_scope_2_files_are_read(tmp_path):
    sae = read_sae(
        _gemma_scope(
            tmp_path / "resid",
            {
                "hf_hook_point_in": "model.layers.1.output",
                "hf_hook_point_out": "model.layers.1.output",
                "width": 16,
                "model_name": "google/gemma-3-270m-pt",
                "architecture": "jump_relu",
                "l0": 4,
                "affine_connection": False,
                "type": "sae",
            },
            _params(8, 8, 16),
        )
    )
    assert (sae.site, sae.site_in, sae.layer) == ("resid_post", "resid_post", 1)
    assert sae.activation == "jumprelu" and not sae.subtract_b_dec and not sae.transcoder
    assert sae.format == "gemma_scope_2" and (sae.d_in, sae.d_sae) == (8, 16)

    transcoder = read_sae(
        _gemma_scope(
            tmp_path / "transcoder",
            {
                "hf_hook_point_in": "model.layers.2.pre_feedforward_layernorm.output",
                "hf_hook_point_out": "model.layers.2.post_feedforward_layernorm.output",
                "architecture": "jump_relu",
                "affine_connection": False,
                "type": "transcoder",
            },
            _params(8, 8, 16),
        )
    )
    assert transcoder.transcoder and (transcoder.site_in, transcoder.site) == ("mlp_in", "mlp_out")


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        ({"type": "clt"}, "cross-layer transcoder"),
        ({"type": "crosscoder"}, "crosscoder"),
        ({"type": "sae", "architecture": "topk"}, "aren't supported"),
        ({"type": "transcoder", "affine_connection": True}, "skip connection"),
        (
            {
                "type": "transcoder",
                "hf_hook_point_in": "model.layers.1.pre_feedforward_layernorm.output",
                "hf_hook_point_out": "model.layers.2.post_feedforward_layernorm.output",
            },
            "same MLP",
        ),
    ],
)
def test_gemma_scope_files_logogram_cant_reproduce_are_refused(tmp_path, cfg, message):
    base = {
        "hf_hook_point_in": "model.layers.1.pre_feedforward_layernorm.output",
        "hf_hook_point_out": "model.layers.1.post_feedforward_layernorm.output",
        "architecture": "jump_relu",
        "affine_connection": False,
    }
    with pytest.raises(BackendError, match=message):
        read_sae(_gemma_scope(tmp_path / "bad", {**base, **cfg}, _params(8, 8, 16)))


def test_split_files_are_refused_as_several_layers(tmp_path):
    folder = tmp_path / "clt"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    save_file({"x": torch.zeros(1)}, str(folder / "params_layer_0.safetensors"))
    with pytest.raises(BackendError, match="several layers"):
        read_sae(folder)


@pytest.fixture
def transcoder(tiny_backend, tmp_path):
    d = tiny_backend.info.d_model
    return _load(
        _gemma_scope(
            tmp_path / "tc",
            {
                "hf_hook_point_in": "transformer.h.0.ln_2.output",
                "hf_hook_point_out": "transformer.h.0.mlp.output",
                "architecture": "jump_relu",
                "affine_connection": False,
                "type": "transcoder",
            },
            _params(d, d, 3 * d, seed=1),
        )
    )


def _prompts(backend, project):
    return prepare_prompts(backend, load_dataset(project.datasets_dir / "ioi.jsonl"), True)


def test_patching_a_transcoder_feature_changes_the_mlps_output(
    tiny_backend, project, spec_factory, transcoder
):
    """The feature is read from the MLP's input in both prompts; the MLP's output moves by the
    change times the feature's decoder row, and the transcoder's error is kept."""
    prompts = _prompts(tiny_backend, project)
    f_in, _ = transcoder.encode(
        tiny_backend.capture(group_by_length(prompts)[0].clean, [("mlp_in", 0)])[("mlp_in", 0)]
    )
    feature = int(f_in[:, -1].sum(0).argmax())  # a feature that fires at the last token
    spec = spec_factory(
        experiment={"kind": "activation_patching", "direction": "clean_to_corrupt"},
        scope={
            "kind": "sites",
            "sites": [
                {
                    "kind": "sae_feature",
                    "layer": 0,
                    "feature": feature,
                    "position": {"kind": "last"},
                }
            ],
        },
        sae=SAE_REF,
    )
    result = run_experiment(spec, tiny_backend, prompts, sae=transcoder)
    group = group_by_length(prompts)[0]
    clean_in = tiny_backend.capture(group.clean, [("mlp_in", 0)])[("mlp_in", 0)]
    corrupt = tiny_backend.capture(group.corrupt, [("mlp_in", 0), ("mlp_out", 0)])
    last = group.clean.shape[1] - 1
    f_clean, _ = transcoder.encode(clean_in[:, last])
    f_corrupt, _ = transcoder.encode(corrupt[("mlp_in", 0)][:, last])
    values = (
        corrupt[("mlp_out", 0)][:, last]
        + (f_clean - f_corrupt)[:, feature, None] * transcoder.params.W_dec[feature]
    )
    rows = torch.arange(len(group.members))
    logits = tiny_backend.final_logits(
        group.corrupt,
        Patch(kind="mlp_out", layer=0, values=values, positions=torch.full((len(rows),), last)),
    )
    answers = torch.tensor([prompts[i].answer_id for i in group.members])
    distractors = torch.tensor([prompts[i].distractor_id for i in group.members])
    expected = (logits[rows, answers] - logits[rows, distractors]).double().numpy()
    np.testing.assert_allclose(result.patched[0, group.members], expected, atol=1e-4)
    fit = result.extra["features"]["fit"]
    assert fit["tokens"] > 0 and np.isfinite(fit["variance_explained"])


def test_transcoder_estimates_follow_the_decoder_and_the_gradient(
    tiny_backend, project, spec_factory, transcoder
):
    from logogram.atp import metric_gradients
    from logogram.engine import make_scorer

    prompts = _prompts(tiny_backend, project)
    spec = spec_factory(
        experiment={"kind": "attribution_patching", "direction": "clean_to_corrupt"},
        scope={
            "kind": "sites",
            "sites": [
                {"kind": "sae_feature", "layer": 0, "feature": i, "position": {"kind": "all"}}
                for i in range(4)
            ],
        },
        sae=SAE_REF,
    )
    result = run_experiment(spec, tiny_backend, prompts, sae=transcoder)
    group = group_by_length(prompts)[0]
    keys = [("mlp_in", 0), ("mlp_out", 0)]
    src = tiny_backend.capture(group.clean, keys)
    acts, grads = metric_gradients(
        tiny_backend, make_scorer(spec, prompts), group.corrupt, group.members, keys
    )
    f_src, _ = transcoder.encode(src[("mlp_in", 0)].float())
    f_rec, _ = transcoder.encode(acts[("mlp_in", 0)].float())
    for i in range(4):
        expected = (
            (
                (f_src[..., i] - f_rec[..., i])
                * (grads[("mlp_out", 0)].float() @ transcoder.params.W_dec[i])
            )
            .sum(1)
            .double()
            .numpy()
        )
        np.testing.assert_allclose(result.delta[i, group.members], expected, rtol=1e-4, atol=1e-6)


def test_an_sae_on_attention_heads_outputs(tiny_backend, project, spec_factory, tmp_path):
    """An SAE on the heads' outputs side by side (the input of the attention's output
    projection) reads and edits z as one vector per position."""
    info = tiny_backend.info
    d = info.n_heads * info.d_head
    eye = torch.eye(d)
    sae = _load(
        _gemma_scope(
            tmp_path / "z",
            {
                "hf_hook_point_in": "transformer.h.1.attn.c_proj.input",
                "hf_hook_point_out": "transformer.h.1.attn.c_proj.input",
                "architecture": "jump_relu",
                "affine_connection": False,
                "type": "sae",
            },
            # Exact: positive and negative parts of each coordinate.
            {
                "w_enc": torch.cat([eye, -eye], 1),
                "b_enc": torch.zeros(2 * d),
                "threshold": torch.zeros(2 * d),
                "w_dec": torch.cat([eye, -eye], 0),
                "b_dec": torch.zeros(d),
            },
        )
    )
    assert sae.site == "head" and sae.params.d_in == d
    prompts = _prompts(tiny_backend, project)
    feature = 3  # the positive part of coordinate 3: head 0, dimension 3
    spec = spec_factory(
        experiment={"kind": "ablation", "baseline": {"kind": "zero"}},
        scope={
            "kind": "sites",
            "sites": [
                {
                    "kind": "sae_feature",
                    "layer": 1,
                    "feature": feature,
                    "position": {"kind": "last"},
                }
            ],
        },
        sae=SAE_REF,
    )
    result = run_experiment(spec, tiny_backend, prompts, sae=sae)
    assert result.extra["features"]["fit"]["variance_explained"] == pytest.approx(1.0, abs=1e-5)
    group = group_by_length(prompts)[0]
    z = tiny_backend.capture(group.clean, [("head", 1)])[("head", 1)]
    last = group.clean.shape[1] - 1
    values = z[:, last, 0].clone()
    values[:, 3] = torch.clamp(values[:, 3], max=0.0)  # the positive part removed
    rows = torch.arange(len(group.members))
    logits = tiny_backend.final_logits(
        group.clean,
        Patch(
            kind="head",
            layer=1,
            values=values,
            heads=torch.zeros(len(rows), dtype=torch.long),
            positions=torch.full((len(rows),), last),
        ),
    )
    answers = torch.tensor([prompts[i].answer_id for i in group.members])
    distractors = torch.tensor([prompts[i].distractor_id for i in group.members])
    expected = (logits[rows, answers] - logits[rows, distractors]).double().numpy()
    np.testing.assert_allclose(result.patched[0, group.members], expected, atol=1e-4)


def test_an_sae_for_another_model_is_refused(tiny_backend, project, spec_factory, tmp_path):
    sae = _load(
        _gemma_scope(
            tmp_path / "wide",
            {
                "hf_hook_point_in": "model.layers.1.output",
                "hf_hook_point_out": "model.layers.1.output",
                "type": "sae",
            },
            _params(640, 640, 16),
        )
    )
    spec = spec_factory(
        scope={
            "kind": "sites",
            "sites": [
                {"kind": "sae_feature", "layer": 1, "feature": 0, "position": {"kind": "all"}}
            ],
        },
        sae=SAE_REF,
    )
    with pytest.raises(ScopeError, match="made for another model"):
        run_experiment(spec, tiny_backend, _prompts(tiny_backend, project), sae=sae)


def test_a_transcoder_run_is_stored(tiny_backend, project, spec_factory, transcoder):
    spec = spec_factory(
        experiment={"kind": "attribution_patching", "direction": "clean_to_corrupt"},
        scope={"kind": "features", "position": {"kind": "last"}, "top": 5},
        sae=SAE_REF,
    )
    outcome = run_spec(
        spec, project, backend=tiny_backend, sae_provider=lambda ref, backend: transcoder
    )
    assert outcome.status == "finished", outcome.manifest.get("error")
    features = outcome.summary["features"]
    assert features["sae"]["transcoder"] and features["sae"]["site_in"] == "mlp_in"
    assert outcome.manifest["sae"]["format"] == "gemma_scope_2"
