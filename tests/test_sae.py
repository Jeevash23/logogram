"""SAEs: read in both published formats, encoded exactly, fitted, patched and estimated per feature."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from logogram.backends.base import BackendError, Patch
from logogram.backends.saes import read_sae
from logogram.datasets import load_dataset
from logogram.engine import run_experiment
from logogram.prompts import group_by_length, prepare_prompts
from logogram.runner import run_spec
from logogram.sae import SAE, fit_on
from logogram.sites import ScopeError
from logogram.verify import verification_spec


def _saelens(folder: Path, cfg: dict, tensors: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")
    save_file(
        {k: v.contiguous() for k, v in tensors.items()}, str(folder / "sae_weights.safetensors")
    )
    return folder


def exact_sae(folder: Path, d: int, layer: int = 1, b_dec: torch.Tensor | None = None) -> SAE:
    """An SAE that reconstructs every activation exactly: its 2d features are the positive and
    negative parts of each coordinate (after subtracting b_dec)."""
    eye = torch.eye(d)
    b = torch.zeros(d) if b_dec is None else b_dec
    _saelens(
        folder,
        {
            "hook_name": f"blocks.{layer}.hook_resid_pre",
            "d_in": d,
            "d_sae": 2 * d,
            "apply_b_dec_to_input": True,
        },
        {
            "W_enc": torch.cat([eye, -eye], 1),
            "b_enc": torch.zeros(2 * d),
            "W_dec": torch.cat([eye, -eye], 0),
            "b_dec": b,
        },
    )
    return SAE(repo="local/exact", path="", revision="test", params=read_sae(folder))


@pytest.fixture
def exact(tiny_backend, tmp_path):
    torch.manual_seed(1)
    return exact_sae(
        tmp_path / "exact",
        tiny_backend.info.d_model,
        b_dec=torch.randn(tiny_backend.info.d_model) * 0.1,
    )


def _prompts(backend, project):
    return prepare_prompts(backend, load_dataset(project.datasets_dir / "ioi.jsonl"), True)


SAE_REF = {"repo": "local/exact", "path": "", "revision": "test"}


def test_both_formats_are_read_and_mapped_to_sites(tmp_path):
    torch.manual_seed(0)
    d, f = 8, 16
    lens = _saelens(
        tmp_path / "lens",
        {
            "architecture": "standard",
            "hook_name": "blocks.3.hook_resid_post",
            "d_in": d,
            "d_sae": f,
            "activation_fn_str": "topk",
            "activation_fn_kwargs": {"k": 3},
            "normalize_activations": "layer_norm",
        },
        {
            "W_enc": torch.randn(d, f),
            "b_enc": torch.randn(f),
            "W_dec": torch.randn(f, d),
            "b_dec": torch.randn(d),
        },
    )
    params = read_sae(lens)
    assert (params.site, params.layer, params.activation, params.k, params.normalize) == (
        "resid_post",
        3,
        "topk",
        3,
        "layer_norm",
    )

    eleuther = tmp_path / "layers.2.mlp"
    eleuther.mkdir()
    (eleuther / "cfg.json").write_text(
        json.dumps({"k": 4, "num_latents": f, "d_in": d, "signed": False})
    )
    save_file(
        {
            "encoder.weight": torch.randn(f, d),
            "encoder.bias": torch.randn(f),
            "W_dec": torch.randn(f, d),
            "b_dec": torch.randn(d),
        },
        str(eleuther / "sae.safetensors"),
    )
    params = read_sae(eleuther)
    assert (params.site, params.layer, params.activation, params.k, params.format) == (
        "mlp_out",
        2,
        "topk",
        4,
        "eleuther",
    )
    assert params.W_enc.shape == (d, f)


def test_the_encoding_matches_the_published_definitions(tmp_path):
    """TopK with standardized inputs (SAELens): standardize, subtract b_dec, project, keep the top
    k (ReLU), decode, and restore the input's mean and spread."""
    torch.manual_seed(0)
    d, f, k = 8, 16, 3
    W_enc, b_enc, W_dec, b_dec = (
        torch.randn(d, f),
        torch.randn(f),
        torch.randn(f, d),
        torch.randn(d),
    )
    folder = _saelens(
        tmp_path / "lens",
        {
            "architecture": "standard",
            "hook_name": "blocks.0.hook_resid_pre",
            "activation_fn_str": "topk",
            "activation_fn_kwargs": {"k": k},
            "normalize_activations": "layer_norm",
            "apply_b_dec_to_input": True,
        },
        {"W_enc": W_enc, "b_enc": b_enc, "W_dec": W_dec, "b_dec": b_dec},
    )
    sae = SAE(repo="r", path="", revision="t", params=read_sae(folder))
    x = torch.randn(5, d) * 3 + 1
    mu = x.mean(-1, keepdim=True)
    std = (x - mu).std(-1, keepdim=True)
    pre = ((x - mu) / (std + 1e-5) - b_dec) @ W_enc + b_enc
    expected = torch.zeros_like(pre)
    top = pre.topk(k, dim=-1)
    expected.scatter_(-1, top.indices, torch.relu(top.values))
    f_acts, stats = sae.encode(x)
    torch.testing.assert_close(f_acts, expected)
    torch.testing.assert_close(sae.decode(f_acts, stats), (expected @ W_dec + b_dec) * std + mu)


def test_formats_logogram_cant_reproduce_are_refused(tmp_path):
    folder = tmp_path / "npz"
    folder.mkdir()
    (folder / "params.npz").write_bytes(b"")
    with pytest.raises(BackendError, match="NumPy archive"):
        read_sae(folder)
    d, f = 4, 8
    weights = {
        "W_enc": torch.zeros(d, f),
        "b_enc": torch.zeros(f),
        "W_dec": torch.zeros(f, d),
        "b_dec": torch.zeros(d),
    }
    for cfg, message in (
        (
            {
                "hook_name": "blocks.0.hook_resid_pre",
                "normalize_activations": "expected_average_only_in",
            },
            "rescales",
        ),
        ({"hook_name": "blocks.0.hook_resid_pre", "architecture": "gated"}, "gated"),
        ({"hook_name": "blocks.0.attn.hook_z"}, "can't map"),
    ):
        with pytest.raises(BackendError, match=message):
            read_sae(_saelens(tmp_path / message.replace(" ", "-").replace("'", ""), cfg, weights))
    skip = tmp_path / "layers.1.mlp"
    skip.mkdir()
    (skip / "cfg.json").write_text(
        json.dumps({"k": 2, "transcode": True, "skip_connection": True}), encoding="utf-8"
    )
    save_file(
        {
            "encoder.weight": torch.zeros(f, d),
            "encoder.bias": torch.zeros(f),
            "W_dec": torch.zeros(f, d),
            "b_dec": torch.zeros(d),
        },
        str(skip / "sae.safetensors"),
    )
    with pytest.raises(BackendError, match="skip connection"):
        read_sae(skip)


def test_an_exact_sae_explains_all_the_variance(tiny_backend, project, exact):
    prompts = _prompts(tiny_backend, project)
    group = group_by_length(prompts)[0]
    x = tiny_backend.capture(group.clean, [("resid_pre", 1)])[("resid_pre", 1)]
    fit = fit_on(exact, x.reshape(-1, x.shape[-1]))
    assert fit["variance_explained"] == pytest.approx(1.0, abs=1e-5)


def test_patching_a_feature_moves_the_activation_along_its_decoder_row(
    tiny_backend, project, spec_factory, exact
):
    prompts = _prompts(tiny_backend, project)
    feature = 3
    site = {"kind": "sae_feature", "layer": 1, "feature": feature, "position": {"kind": "last"}}
    spec = spec_factory(
        sae=SAE_REF,
        experiment={"kind": "activation_patching", "direction": "clean_to_corrupt"},
        scope={"kind": "sites", "sites": [site]},
    )
    result = run_experiment(spec, tiny_backend, prompts, sae=exact)
    # The same edit by hand: add (clean - corrupt) feature activation times the decoder row.
    group = group_by_length(prompts)[0]
    clean = tiny_backend.capture(group.clean, [("resid_pre", 1)])[("resid_pre", 1)]
    corrupt = tiny_backend.capture(group.corrupt, [("resid_pre", 1)])[("resid_pre", 1)]
    last = clean.shape[1] - 1
    f_clean, _ = exact.encode(clean[:, last])
    f_corrupt, _ = exact.encode(corrupt[:, last])
    values = (
        corrupt[:, last] + (f_clean - f_corrupt)[:, feature, None] * exact.params.W_dec[feature]
    )
    rows = torch.arange(len(group.members))
    logits = tiny_backend.final_logits(
        group.corrupt,
        Patch(kind="resid_pre", layer=1, values=values, positions=torch.full((len(rows),), last)),
    )
    answers = torch.tensor([prompts[i].answer_id for i in group.members])
    distractors = torch.tensor([prompts[i].distractor_id for i in group.members])
    expected = (logits[rows, answers] - logits[rows, distractors]).double().numpy()
    np.testing.assert_allclose(result.patched[0, group.members], expected, atol=1e-4)
    assert result.extra["features"]["fit"]["variance_explained"] == pytest.approx(1.0, abs=1e-5)

    same = run_experiment(
        spec,
        tiny_backend,
        prompts,
        sae=exact,
        receiver_override="corrupt",
        source_override="corrupt",
    )
    np.testing.assert_allclose(same.patched, same.receiver_metric[None, :], atol=1e-4)


def test_with_an_exact_sae_the_feature_estimates_add_up_to_the_site(
    tiny_backend, project, spec_factory, exact
):
    """Without an error term, the activation is the sum of its features' decoder rows, so their
    first-order estimates add up to the whole site's."""
    spec = spec_factory(
        sae=SAE_REF,
        experiment={"kind": "attribution_patching", "direction": "clean_to_corrupt"},
        scope={"kind": "features", "position": {"kind": "all"}, "top": 5},
    )
    result = run_experiment(spec, tiny_backend, _prompts(tiny_backend, project), sae=exact)
    info = result.extra["features"]
    assert info["features_estimate"] == pytest.approx(info["site_estimate"], rel=1e-5, abs=1e-6)
    assert len(result.sites) == 5 and result.layout["kind"] == "sites"
    means = np.abs(result.delta.mean(1))
    assert list(means) == sorted(means, reverse=True)
    assert result.measure == "estimate"

    # Estimating the same features as chosen sites gives the same numbers.
    chosen = [
        {"kind": "sae_feature", "layer": 1, "feature": rs.site.feature, "position": {"kind": "all"}}
        for rs in result.sites
    ]
    again = run_experiment(
        spec_factory(
            sae=SAE_REF,
            experiment=spec.experiment.model_dump(),
            scope={"kind": "sites", "sites": chosen},
        ),
        tiny_backend,
        _prompts(tiny_backend, project),
        sae=exact,
    )
    np.testing.assert_allclose(again.delta, result.delta, rtol=1e-5, atol=1e-8)


def test_features_can_be_chosen_on_some_prompts_and_reported_on_the_others(
    tiny_backend, project, spec_factory, exact
):
    """Choosing the strongest features on the prompts that report them biases their effects
    away from zero; a held-out split chooses on some prompts and reports on the rest."""
    prompts = _prompts(tiny_backend, project)
    experiment = {"kind": "attribution_patching", "direction": "clean_to_corrupt"}
    every = {"kind": "features", "position": {"kind": "all"}, "top": 4}
    full = run_experiment(
        spec_factory(sae=SAE_REF, experiment=experiment, scope={**every, "top": 500}),
        tiny_backend,
        prompts,
        sae=exact,
    )
    held = {**every, "choose_on": 0.5, "seed": 3}
    result = run_experiment(
        spec_factory(sae=SAE_REF, experiment=experiment, scope=held),
        tiny_backend,
        prompts,
        sae=exact,
    )
    info = result.extra["features"]
    chosen, reported = info["chosen_on"], info["reported_on"]
    assert len(chosen) == 6 and len(reported) == 6
    assert sorted(chosen + reported) == list(range(12))
    assert [p.index for p in result.prompts] == reported
    assert result.delta.shape == (4, 6) and result.baselines.clean.shape == (6,)
    # The features are the strongest on the choosing prompts, and their values on the reported
    # prompts are exactly those of the run that estimated every prompt.
    by_feature = {rs.site.feature: full.delta[rs.index] for rs in full.sites}
    features = sorted(by_feature)
    assert len(features) == exact.d_sae
    choosing = np.array([by_feature[f][chosen].mean() for f in features])
    strongest = [features[i] for i in np.argsort(-np.abs(choosing), kind="stable")[:4]]
    assert [rs.site.feature for rs in result.sites] == strongest
    for rs in result.sites:
        np.testing.assert_allclose(result.delta[rs.index], by_feature[rs.site.feature][reported])
    with pytest.raises(ValueError, match="both choose_on"):
        spec_factory(sae=SAE_REF, experiment=experiment, scope={**every, "choose_on": 0.5})
    with pytest.raises(ScopeError, match="at least one and two"):
        run_experiment(
            spec_factory(sae=SAE_REF, experiment=experiment, scope={**held, "choose_on": 0.95}),
            tiny_backend,
            prompts,
            sae=exact,
        )


def test_feature_runs_are_checked_against_the_sae(
    tiny_backend, project, spec_factory, exact, tmp_path
):
    prompts = _prompts(tiny_backend, project)
    patch = {"kind": "activation_patching", "direction": "clean_to_corrupt"}
    with pytest.raises(ScopeError, match="reads layer 1"):
        site = {"kind": "sae_feature", "layer": 0, "feature": 1, "position": {"kind": "all"}}
        run_experiment(
            spec_factory(sae=SAE_REF, experiment=patch, scope={"kind": "sites", "sites": [site]}),
            tiny_backend,
            prompts,
            sae=exact,
        )
    with pytest.raises(ScopeError, match="attribution patching"):
        run_experiment(
            spec_factory(
                sae=SAE_REF,
                experiment=patch,
                scope={"kind": "features", "position": {"kind": "all"}, "top": 3},
            ),
            tiny_backend,
            prompts,
            sae=exact,
        )
    other = exact_sae(tmp_path / "wide", 48)
    site = {"kind": "sae_feature", "layer": 1, "feature": 1, "position": {"kind": "all"}}
    with pytest.raises(ScopeError, match="another model"):
        run_experiment(
            spec_factory(sae=SAE_REF, experiment=patch, scope={"kind": "sites", "sites": [site]}),
            tiny_backend,
            prompts,
            sae=other,
        )
    with pytest.raises(ValueError, match="need the spec's sae"):
        spec_factory(experiment=patch, scope={"kind": "sites", "sites": [site]})


def test_a_feature_run_is_stored_and_verified(tiny_backend, project, spec_factory, exact):
    spec = spec_factory(
        sae=SAE_REF,
        experiment={"kind": "attribution_patching", "direction": "corrupt_to_clean"},
        scope={"kind": "features", "position": {"kind": "last"}, "top": 4},
    )
    outcome = run_spec(spec, project, backend=tiny_backend, sae=exact)
    assert outcome.status == "finished", outcome.manifest.get("error")
    summary = json.loads((outcome.folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["features"]["sae"]["repo"] == "local/exact"
    assert summary["features"]["fit"]["variance_explained"] == pytest.approx(1.0, abs=1e-5)
    assert (
        summary["sites"][0]["kind"] == "sae_feature" and summary["sites"][0]["feature"] is not None
    )
    manifest = json.loads((outcome.folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sae"]["revision"] == "test"

    check = verification_spec(spec, summary, top=2)
    assert check.scope.sites[0].kind == "sae_feature"
    assert check.scope.sites[0].feature == summary["sites"][0]["feature"]
    verified = run_spec(check, project, backend=tiny_backend, sae=exact)
    assert verified.status == "finished", verified.manifest.get("error")


def test_the_server_loads_fits_and_explores_an_sae(tiny_backend, tmp_path, monkeypatch, project):
    from fastapi.testclient import TestClient

    from logogram.server.app import create_app
    from logogram.server.security import SecurityConfig

    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    app = create_app(SecurityConfig(token="t", port=8765), serve_web=False)
    state = app.state.logogram
    state.backend = tiny_backend
    state.model_status = {"state": "ready"}
    state.open_project(project)
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    headers = {"authorization": "Bearer t", "origin": "http://127.0.0.1:8765"}
    exact = exact_sae(tmp_path / "exact", tiny_backend.info.d_model)
    # The download is the only part that needs the network: load the local SAE instead.
    monkeypatch.setattr(
        "logogram.sae.load_sae", lambda ref, device, progress=None, cancel=None: exact
    )

    assert client.get("/api/state", headers=headers).json()["sae"] == {"state": "none"}
    job = client.post("/api/sae/load", json={"repo": "local/exact"}, headers=headers).json()
    assert job["kind"] == "load_sae"
    for _ in range(200):
        if state.job.status != "running":
            break
        time.sleep(0.02)
    sae = client.get("/api/state", headers=headers).json()["sae"]
    assert sae["state"] == "ready" and sae["info"]["d_sae"] == 2 * tiny_backend.info.d_model

    body = {"dataset": "datasets/ioi.jsonl", "prepend_bos": True, "batch_size": 64}
    fit = client.post("/api/sae/fit", json=body, headers=headers).json()
    assert fit["variance_explained"] == pytest.approx(1.0, abs=1e-5)
    # An exact SAE spliced in changes nothing.
    assert fit["spliced_logit_diff"] == pytest.approx(fit["logit_diff"], abs=1e-4)
    tokens = client.post("/api/sae/tokens", json={**body, "index": 2}, headers=headers).json()
    assert len(tokens["features"]) == len(tokens["tokens"])
    feature = tokens["features"][-1][0]["feature"]
    report = client.post(
        "/api/sae/feature", json={**body, "index": 2, "feature": feature}, headers=headers
    ).json()
    assert report["activations"][-1] == pytest.approx(
        tokens["features"][-1][0]["activation"], rel=1e-5
    )
    assert report["top"] and report["n"] == 12

    # The SAE belongs to the model. (The tiny model is shared by every test, so it stays open.)
    monkeypatch.setattr(tiny_backend, "close", lambda: None)
    state.unload_model()
    assert client.get("/api/state", headers=headers).json()["sae"] == {"state": "none"}
    missing = client.post("/api/sae/fit", json=body, headers=headers)
    assert missing.status_code == 409
