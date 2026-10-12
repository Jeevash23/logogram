"""Transcoders with a skip connection (EleutherAI's sparsify, Gemma Scope 2's affine ones), read
exactly: checked against reference implementations of each format, and in feature values,
fits, patching and attribution patching on the tiny model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from logogram.atp import metric_gradients
from logogram.backends.base import BackendError, Patch
from logogram.backends.saes import read_sae
from logogram.datasets import load_dataset
from logogram.engine import make_scorer, run_experiment
from logogram.prompts import group_by_length, prepare_prompts
from logogram.sae import SAE, fit_on

SAE_REF = {"repo": "local/skip", "path": "", "revision": "test"}


def sparsify_reference(cfg: dict, tensors: dict, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """sparsify's SparseCoder (1.3.3), as its source computes it, on a file it loads: b_dec is
    subtracted from the input unless cfg.json says it is a transcoder, then ReLU, the top k, the
    decoder rows they select plus b_dec, and the skip connection x @ W_skip.T of the raw input.
    Returns the dense feature activations and the output."""
    W, b = tensors["encoder.weight"], tensors["encoder.bias"]
    sae_in = x if cfg.get("transcode", False) else x - tensors["b_dec"]
    pre = torch.relu(sae_in @ W.T + b)
    values, indices = pre.topk(cfg["k"], dim=-1, sorted=False)
    dense = torch.zeros_like(pre).scatter(-1, indices, values)
    out = (values[..., None] * tensors["W_dec"][indices]).sum(-2) + tensors["b_dec"]
    if "W_skip" in tensors:
        out = out + x @ tensors["W_skip"].T
    return dense, out


def gemma_scope_reference(params: dict, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Gemma Scope 2's JumpReLU transcoder, its affine skip connection added as
    x @ affine_skip_connection (stored input by output, like w_enc and w_dec). On Gemma 3 270M's
    layer 12 this explains 91% of the MLP output's variance, more than the transcoder without
    one (89%); the transpose explains 34%, less than no skip connection at all."""
    pre = x @ params["w_enc"] + params["b_enc"]
    f = torch.relu(pre) * (pre > params["threshold"])
    out = f @ params["w_dec"] + params["b_dec"]
    if "affine_skip_connection" in params:
        out = out + x @ params["affine_skip_connection"]
    return f, out


def _skip_tensors(d: int, width: int, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "encoder.weight": torch.randn(width, d, generator=g) * 0.5,
        "encoder.bias": torch.randn(width, generator=g) * 0.1,
        "W_dec": torch.randn(width, d, generator=g) * 0.2,
        "b_dec": torch.randn(d, generator=g) * 0.1,
        # Not symmetric, so reading it the wrong way round shows.
        "W_skip": torch.randn(d, d, generator=g) * 0.3,
    }


def _sparsify(root: Path, folder: str, cfg: dict, tensors: dict, training: dict | None = None):
    path = root / folder
    path.mkdir(parents=True)
    (path / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path / "sae.safetensors"))
    if training is not None:
        (root / "config.json").write_text(json.dumps(training), encoding="utf-8")
    return path


# cfg.json as sparsify writes it since February 2025, and before (the published Llama 3.2 1B
# skip transcoders): no transcode key, which only the training run's config.json has.
CURRENT = {
    "activation": "topk",
    "expansion_factor": 4,
    "normalize_decoder": True,
    "num_latents": 0,
    "k": 6,
    "multi_topk": False,
    "skip_connection": True,
    "transcode": True,
}
BEFORE = {
    "expansion_factor": 4,
    "normalize_decoder": True,
    "num_latents": 0,
    "k": 6,
    "multi_topk": False,
    "skip_connection": True,
}
TRAINING = {
    "sae": BEFORE,
    "hookpoints": ["layers.0.mlp", "layers.1.mlp"],
    "transcode": True,
    "model": "local/tiny",
}


def _eleuther_skip(root: Path, d: int, version: str) -> tuple[SAE, dict, dict]:
    cfg = {**(CURRENT if version == "current" else BEFORE), "d_in": d}
    tensors = _skip_tensors(d, 4 * d, seed=2)
    folder = _sparsify(
        root, "layers.0.mlp", cfg, tensors, TRAINING if version == "before" else None
    )
    params = read_sae(folder, "layers.0.mlp")
    return SAE(repo="local/skip", path="", revision="test", params=params), cfg, tensors


@pytest.mark.parametrize("version", ["current", "before"])
def test_sparsify_skip_transcoders_encode_and_decode_as_sparsify_does(tmp_path, version):
    d = 16
    sae, cfg, tensors = _eleuther_skip(tmp_path, d, version)
    assert (sae.site_in, sae.site, sae.layer) == ("mlp_in", "mlp_out", 0)
    assert sae.transcoder and sae.skip
    # sparsify subtracted b_dec from a transcoder's input until it began recording transcoders.
    assert sae.params.subtract_b_dec == (version == "before")
    x = torch.randn(5, 7, d, generator=torch.Generator().manual_seed(1))
    reference_f, reference_out = sparsify_reference(cfg, tensors, x)
    f, stats = sae.encode(x)
    torch.testing.assert_close(f, reference_f)
    torch.testing.assert_close(sae.decode(f, stats, x), reference_out)
    assert "skip transcoder" in sae.describe()["note"] and sae.describe()["skip"]


def test_gemma_scope_affine_transcoders_decode_with_their_skip_connection(tmp_path):
    d, width = 12, 40
    g = torch.Generator().manual_seed(3)
    params = {
        "w_enc": torch.randn(d, width, generator=g) * 0.5,
        "b_enc": torch.randn(width, generator=g) * 0.1,
        "threshold": torch.full((width,), 0.05),
        "w_dec": torch.randn(width, d, generator=g) * 0.2,
        "b_dec": torch.randn(d, generator=g) * 0.1,
        "affine_skip_connection": torch.randn(d, d, generator=g) * 0.3,
    }
    folder = tmp_path / "affine"
    folder.mkdir()
    (folder / "config.json").write_text(
        json.dumps(
            {
                "hf_hook_point_in": "model.layers.2.pre_feedforward_layernorm.output",
                "hf_hook_point_out": "model.layers.2.post_feedforward_layernorm.output",
                "architecture": "jump_relu",
                "affine_connection": True,
                "type": "transcoder",
            }
        ),
        encoding="utf-8",
    )
    save_file(params, str(folder / "params.safetensors"))
    sae = SAE(repo="local/affine", path="", revision="test", params=read_sae(folder))
    assert sae.transcoder and sae.skip and not sae.params.subtract_b_dec
    x = torch.randn(4, 6, d, generator=g)
    reference_f, reference_out = gemma_scope_reference(params, x)
    f, stats = sae.encode(x)
    torch.testing.assert_close(f, reference_f)
    torch.testing.assert_close(sae.decode(f, stats, x), reference_out)
    # Without its input, a skip transcoder can't decode.
    with pytest.raises(ValueError, match="with its input"):
        sae.decode(f, stats)


def _prompts(backend, project):
    return prepare_prompts(backend, load_dataset(project.datasets_dir / "ioi.jsonl"), True)


@pytest.fixture(params=["current", "before"])
def skip_transcoder(request, tiny_backend, tmp_path):
    return _eleuther_skip(tmp_path, tiny_backend.info.d_model, request.param)


def test_the_fit_counts_the_skip_connection(tiny_backend, project, skip_transcoder):
    sae, cfg, tensors = skip_transcoder
    group = group_by_length(_prompts(tiny_backend, project))[0]
    captured = tiny_backend.capture(group.clean, [("mlp_in", 0), ("mlp_out", 0)])
    x = captured[("mlp_in", 0)].reshape(-1, tiny_backend.info.d_model)
    y = captured[("mlp_out", 0)].reshape(-1, tiny_backend.info.d_model)
    _, out = sparsify_reference(cfg, tensors, x)
    expected = 1 - float(((y - out).double() ** 2).sum()) / float(
        ((y - y.mean(0)).double() ** 2).sum()
    )
    assert fit_on(sae, x, y)["variance_explained"] == pytest.approx(expected, rel=1e-5)


def test_patching_a_skip_transcoder_feature_moves_only_its_decoder_row(
    tiny_backend, project, spec_factory, skip_transcoder
):
    """The input stays as it was, so the skip connection doesn't move: the MLP's output moves by
    the feature's change times its decoder row, and the transcoder's error is kept."""
    sae, _, _ = skip_transcoder
    prompts = _prompts(tiny_backend, project)
    group = group_by_length(prompts)[0]
    clean_in = tiny_backend.capture(group.clean, [("mlp_in", 0)])[("mlp_in", 0)]
    f_in, _ = sae.encode(clean_in)
    feature = int(f_in[:, -1].sum(0).argmax())  # one that fires at the last token
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
    result = run_experiment(spec, tiny_backend, prompts, sae=sae)
    corrupt = tiny_backend.capture(group.corrupt, [("mlp_in", 0), ("mlp_out", 0)])
    last = group.clean.shape[1] - 1
    f_clean, _ = sae.encode(clean_in[:, last])
    f_corrupt, _ = sae.encode(corrupt[("mlp_in", 0)][:, last])
    values = (
        corrupt[("mlp_out", 0)][:, last]
        + (f_clean - f_corrupt)[:, feature, None] * sae.params.W_dec[feature]
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


def test_the_site_estimate_splits_into_features_skip_connection_and_error(
    tiny_backend, project, spec_factory, skip_transcoder
):
    """To first order, the MLP output's change is its features' changes through their decoder
    rows, the skip connection's change and the transcoder's error's: the estimates of the three
    add up to the whole site's, exactly."""
    sae, _, _ = skip_transcoder
    prompts = _prompts(tiny_backend, project)
    spec = spec_factory(
        experiment={"kind": "attribution_patching", "direction": "clean_to_corrupt"},
        scope={"kind": "features", "position": {"kind": "all"}, "top": 5},
        sae=SAE_REF,
    )
    result = run_experiment(spec, tiny_backend, prompts, sae=sae)
    info = result.extra["features"]
    keys = [("mlp_in", 0), ("mlp_out", 0)]
    errors, skips = [], []
    scorer = make_scorer(spec, prompts)
    for group in group_by_length(prompts):
        src = tiny_backend.capture(group.clean, keys)
        acts, grads = metric_gradients(tiny_backend, scorer, group.corrupt, group.members, keys)
        g = grads[("mlp_out", 0)].float()

        def parts(captured):
            x, y = captured[("mlp_in", 0)].float(), captured[("mlp_out", 0)].float()
            f, stats = sae.encode(x)
            return x @ sae.params.W_skip, y - sae.decode(f, stats, x)

        skip_src, error_src = parts(src)
        skip_rec, error_rec = parts(acts)
        skips.append((((skip_src - skip_rec) * g).sum((1, 2))).double())
        errors.append((((error_src - error_rec) * g).sum((1, 2))).double())
    skip, error = float(torch.cat(skips).mean()), float(torch.cat(errors).mean())
    assert info["skip_estimate"] == pytest.approx(skip, rel=1e-4, abs=1e-6)
    assert info["site_estimate"] == pytest.approx(
        info["features_estimate"] + info["skip_estimate"] + error, rel=1e-4, abs=1e-5
    )
    # Every feature's own estimate is unchanged by the skip connection: Δf · (W_dec[i] · g).
    chosen = [
        {"kind": "sae_feature", "layer": 0, "feature": i, "position": {"kind": "all"}}
        for i in range(3)
    ]
    sites = run_experiment(
        spec_factory(
            experiment=spec.experiment.model_dump(),
            scope={"kind": "sites", "sites": chosen},
            sae=SAE_REF,
        ),
        tiny_backend,
        prompts,
        sae=sae,
    )
    group = group_by_length(prompts)[0]
    src = tiny_backend.capture(group.clean, keys)
    acts, grads = metric_gradients(tiny_backend, scorer, group.corrupt, group.members, keys)
    f_src, _ = sae.encode(src[("mlp_in", 0)].float())
    f_rec, _ = sae.encode(acts[("mlp_in", 0)].float())
    for i in range(3):
        expected = (
            (
                (f_src[..., i] - f_rec[..., i])
                * (grads[("mlp_out", 0)].float() @ sae.params.W_dec[i])
            )
            .sum(1)
            .double()
            .numpy()
        )
        np.testing.assert_allclose(sites.delta[i, group.members], expected, rtol=1e-4, atol=1e-6)


def test_variants_logogram_cant_reproduce_are_refused(tmp_path):
    d = 8
    tensors = _skip_tensors(d, 4 * d)
    cases = [
        # Settings and tensors from forks of sparsify, whose meaning can't be checked.
        (
            {**CURRENT, "train_post_encoder": True, "topk_coalesced": True},
            tensors,
            "sets topk_coalesced and train_post_encoder",
        ),
        (
            {**CURRENT, "dtype": "none", "n_targets": 0},
            {**tensors, "post_enc": torch.zeros(32)},
            "post_enc",
        ),
        ({**CURRENT, "activation": "groupmax"}, tensors, "groupmax"),
        ({**CURRENT, "signed": True}, tensors, "Signed SAEs"),
        # A skip connection on an autoencoder: its features don't add up to the activation.
        ({**CURRENT, "transcode": False}, tensors, "skip connection"),
        ({**BEFORE}, tensors, "skip connection"),  # without its training run's config
        # The config and the weights disagree.
        ({**CURRENT, "skip_connection": False}, tensors, "disagree"),
    ]
    for n, (cfg, weights, message) in enumerate(cases):
        folder = _sparsify(tmp_path / str(n), "layers.1.mlp", {**cfg, "d_in": d}, weights)
        with pytest.raises(BackendError, match=message):
            read_sae(folder, "layers.1.mlp")
    # A training run's config that doesn't list the hookpoint describes other files.
    folder = _sparsify(
        tmp_path / "other",
        "layers.1.mlp",
        {**BEFORE, "d_in": d},
        tensors,
        {**TRAINING, "hookpoints": ["layers.5.mlp"]},
    )
    with pytest.raises(BackendError, match=r"doesn't list layers\.1\.mlp"):
        read_sae(folder, "layers.1.mlp")
    # Hookpoints given as a pattern list it.
    folder = _sparsify(
        tmp_path / "pattern",
        "layers.1.mlp",
        {**BEFORE, "d_in": d},
        tensors,
        {**TRAINING, "hookpoints": ["layers.*.mlp"]},
    )
    assert read_sae(folder, "layers.1.mlp").transcoder
    # Shapes that don't fit together.
    folder = _sparsify(
        tmp_path / "shape",
        "layers.1.mlp",
        {**CURRENT, "d_in": d},
        {**tensors, "W_skip": torch.zeros(d, d + 1)},
    )
    with pytest.raises(BackendError, match="W_skip"):
        read_sae(folder, "layers.1.mlp")


def test_gemma_scope_skip_variants_logogram_cant_reproduce_are_refused(tmp_path):
    d, width = 8, 16
    params = {
        "w_enc": torch.zeros(d, width),
        "b_enc": torch.zeros(width),
        "threshold": torch.zeros(width),
        "w_dec": torch.zeros(width, d),
        "b_dec": torch.zeros(d),
        "affine_skip_connection": torch.zeros(d, d),
    }
    sae = {
        "hf_hook_point_in": "model.layers.1.output",
        "hf_hook_point_out": "model.layers.1.output",
        "type": "sae",
        "affine_connection": True,
    }
    transcoder = {
        "hf_hook_point_in": "model.layers.1.pre_feedforward_layernorm.output",
        "hf_hook_point_out": "model.layers.1.post_feedforward_layernorm.output",
        "type": "transcoder",
    }
    for n, (cfg, tensors, message) in enumerate(
        [
            (sae, params, "skip connection"),
            ({**transcoder, "affine_connection": False}, params, "disagree"),
            (
                {**transcoder, "affine_connection": True},
                {**params, "extra": torch.zeros(1)},
                "extra",
            ),
        ]
    ):
        folder = tmp_path / str(n)
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        save_file(tensors, str(folder / "params.safetensors"))
        with pytest.raises(BackendError, match=message):
            read_sae(folder)


def test_the_training_config_is_downloaded_beside_an_eleuther_sae(tmp_path, monkeypatch):
    """Files from before sparsify recorded transcoders say whether they are one only in the
    training run's config.json, at the top of the repository: it comes along, first, so the
    SAE's own folder is what the download returns."""
    from types import SimpleNamespace

    from logogram.backends import hub
    from logogram.backends.saes import download_sae

    info = SimpleNamespace(
        sha="c" * 40,
        gated=False,
        siblings=[
            SimpleNamespace(rfilename=name, size=1)
            for name in ("config.json", "layers.0.mlp/cfg.json", "layers.0.mlp/sae.safetensors")
        ],
    )
    monkeypatch.setattr("huggingface_hub.HfApi.model_info", lambda *args, **kwargs: info)
    asked = {}

    def download(repo, files, progress=None, cancel=None):
        asked["files"] = [name for name, _ in files.files]
        return tmp_path / "layers.0.mlp"

    monkeypatch.setattr(hub, "download", download)
    revision, folder = download_sae("org/skip", "layers.0.mlp", None)
    assert revision == "c" * 40 and folder.name == "layers.0.mlp"
    assert asked["files"] == [
        "config.json",
        "layers.0.mlp/cfg.json",
        "layers.0.mlp/sae.safetensors",
    ]
