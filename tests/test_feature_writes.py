"""What a feature writes: its decoder direction read straight into the logits through the final
normalization (its scale held) and the unembedding. Checked by hand on tiny models: from the
Hugging Face weights, and by adding the direction to the model's last residual stream."""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest
import torch

from logogram.backends.saes import read_sae
from logogram.features import feature_writes
from logogram.sae import SAE
from logogram.sites import ScopeError
from test_sae import _saelens, exact_sae

# The checks that add a direction to the model's last residual stream hold the final
# normalization's scale with a hook, which TransformerLens warns about; holding it is the point.
pytestmark = pytest.mark.filterwarnings("ignore:A forward hook edited hook_scale")

TEXT = "When Mary and John went to the store, John gave a drink to"


def _values(report: dict) -> dict[int, float]:
    """Every token's value in a report made with ``top`` the size of the vocabulary."""
    return {r["id"]: r["logit"] for r in report["promoted"]}


def _by_hand(hf, ids: list[int], at: int, direction: torch.Tensor) -> torch.Tensor:
    """From GPT-2's Hugging Face modules: the stream entering ln_f at token ``at``, its scale
    sqrt(var + eps), and the direction centered, divided by that scale, times ln_f's weight,
    through the unembedding; then relative to the mean over the vocabulary."""
    seen = {}
    ln = hf.transformer.ln_f
    handle = ln.register_forward_pre_hook(lambda m, args: seen.__setitem__("x", args[0].detach()))
    with torch.no_grad():
        hf(torch.tensor([ids]))
    handle.remove()
    x = seen["x"][0, at].double()
    scale = torch.sqrt(((x - x.mean()) ** 2).mean() + ln.eps)
    d = direction.double()
    change = ((d - d.mean()) / scale * ln.weight.double()) @ hf.lm_head.weight.double().T
    return change - change.mean()


def test_a_feature_writes_its_decoder_row_through_the_final_norm_and_the_unembedding(
    arch_backend, tmp_path
):
    backend = arch_backend("gpt2", process_weights=False)
    hf = backend.bridge.original_model
    sae = exact_sae(tmp_path / "exact", backend.info.d_model)  # reads and writes resid_pre, L1
    ids = backend.tokenize(TEXT, True).ids
    vocab = backend.info.d_vocab
    at, feature = len(ids) - 3, 7
    report = feature_writes(backend, sae, feature, ids, position=at, first=1, top=vocab)
    expected = _by_hand(hf, ids, at, sae.params.W_dec[feature])
    values = _values(report)
    assert sorted(values) == list(range(vocab))
    np.testing.assert_allclose([values[t] for t in range(vocab)], expected.numpy(), atol=1e-6)
    promoted = [r["logit"] for r in report["promoted"]]
    assert promoted == sorted(promoted, reverse=True)
    assert [r["logit"] for r in report["suppressed"]] == sorted(promoted)
    assert report["position"] == at and report["chosen"] == "given"
    assert report["token"] == backend.token_str(ids[at])
    f, _ = sae.encode(backend.capture(torch.tensor([ids]), [("resid_pre", 1)])[("resid_pre", 1)])
    assert report["activation"] == pytest.approx(float(f[0, at, feature]))
    assert report["norm"] == pytest.approx(1.0)  # a unit basis vector
    assert report["soft_cap"] is None


def test_it_is_what_adding_the_direction_to_the_last_residual_stream_does(tiny_backend, tmp_path):
    """Add a multiple of the direction to the stream entering the final normalization, with that
    normalization's scale held at its own value: every logit moves by the multiple times the
    feature's value, up to a shift shared by all tokens."""
    backend = tiny_backend
    bridge = backend.bridge
    sae = exact_sae(tmp_path / "exact", backend.info.d_model)
    ids = backend.tokenize(TEXT, True).ids
    at, feature, amount = len(ids) - 1, 12, 2.5
    report = feature_writes(
        backend, sae, feature, ids, position=at, first=1, top=backend.info.d_vocab
    )
    last = f"blocks.{backend.info.n_layers - 1}.hook_resid_post"
    tokens = torch.tensor([ids])
    with torch.no_grad():
        logits, cache = bridge.run_with_cache(tokens, names_filter=["ln_final.hook_scale"])
    scale = cache["ln_final.hook_scale"]

    def add(act, hook=None):
        act = act.clone()
        act[0, at] += amount * sae.params.W_dec[feature]
        return act

    with torch.no_grad():
        moved = bridge.run_with_hooks(
            tokens,
            fwd_hooks=[(last, add), ("ln_final.hook_scale", lambda value, hook=None: scale)],
            return_type="logits",
        )
    change = (moved[0, at] - logits[0, at]).double() / amount
    change = change - change.mean()
    values = _values(report)
    np.testing.assert_allclose(
        [values[t] for t in range(backend.info.d_vocab)], change.numpy(), atol=1e-5
    )


def test_weight_processing_doesnt_change_it(arch_backend, tmp_path):
    processed, plain = arch_backend("gpt2", True), arch_backend("gpt2", False)
    ids = processed.tokenize(TEXT, True).ids
    reports = [
        feature_writes(
            backend,
            exact_sae(tmp_path / str(i), backend.info.d_model),
            3,
            ids,
            position=4,
            top=backend.info.d_vocab,
        )
        for i, backend in enumerate((processed, plain))
    ]
    a, b = (_values(r) for r in reports)
    np.testing.assert_allclose([a[t] for t in sorted(a)], [b[t] for t in sorted(a)], atol=1e-6)


def test_an_attention_feature_writes_through_the_output_projection(arch_backend, tmp_path):
    """A feature of an SAE on the heads' outputs (z) writes through each head's slice of the
    output projection: for GPT-2, the row of c_proj's weight for that coordinate of z."""
    backend = arch_backend("gpt2", process_weights=False)
    hf = backend.bridge.original_model
    info = backend.info
    d = info.n_heads * info.d_head
    sae = _z_sae(tmp_path / "z", d, layer=1)
    ids = backend.tokenize(TEXT, True).ids
    feature = info.d_head + 2  # the positive part of head 1's third coordinate
    report = feature_writes(backend, sae, feature, ids, position=len(ids) - 1, top=info.d_vocab)
    written = hf.transformer.h[1].attn.c_proj.weight[feature]  # Conv1D: [nx, nf]
    expected = _by_hand(hf, ids, len(ids) - 1, written)
    values = _values(report)
    np.testing.assert_allclose(
        [values[t] for t in range(info.d_vocab)], expected.numpy(), atol=1e-6
    )
    assert report["site"] == "head"
    assert report["norm"] == pytest.approx(float(written.norm()), rel=1e-5)


def _z_sae(folder, d: int, layer: int) -> SAE:
    """An exact SAELens SAE on every head's output z of a layer: its features are the positive
    and negative parts of each coordinate of z, head after head."""
    eye = torch.eye(d)
    _saelens(
        folder,
        {"hook_name": f"blocks.{layer}.attn.hook_z", "d_in": d, "apply_b_dec_to_input": True},
        {
            "W_enc": torch.cat([eye, -eye], 1),
            "b_enc": torch.zeros(2 * d),
            "W_dec": torch.cat([eye, -eye], 0),
            "b_dec": torch.zeros(d),
        },
    )
    return SAE(repo="local/z", path="", revision="test", params=read_sae(folder))


def test_standardized_inputs_scale_a_unit_by_the_tokens_spread(tiny_backend, tmp_path):
    """For an SAE that standardizes its inputs, a unit of a feature is its decoder row times the
    input's standard deviation at that token."""
    backend = tiny_backend
    d = backend.info.d_model
    torch.manual_seed(3)
    folder = _saelens(
        tmp_path / "ln",
        {
            "hook_name": "blocks.1.hook_resid_pre",
            "normalize_activations": "layer_norm",
            "apply_b_dec_to_input": True,
        },
        {
            "W_enc": torch.randn(d, 64),
            "b_enc": torch.zeros(64),
            "W_dec": torch.randn(64, d),
            "b_dec": torch.zeros(d),
        },
    )
    sae = SAE(repo="local/ln", path="", revision="test", params=read_sae(folder))
    # The same SAE without standardized inputs: a unit is its decoder row alone.
    plain = SAE(
        repo="local/ln",
        path="",
        revision="test",
        params=dataclasses.replace(read_sae(folder), normalize="none"),
    )
    ids = backend.tokenize(TEXT, True).ids
    at = len(ids) - 2
    x = backend.capture(torch.tensor([ids]), [("resid_pre", 1)])[("resid_pre", 1)][0, at]
    spread = float((x - x.mean()).std())
    scaled = feature_writes(backend, sae, 9, ids, position=at, top=backend.info.d_vocab)
    unit = feature_writes(backend, plain, 9, ids, position=at, top=backend.info.d_vocab)
    a, b = _values(scaled), _values(unit)
    np.testing.assert_allclose(
        [a[t] for t in sorted(a)], [spread * b[t] for t in sorted(a)], rtol=1e-4, atol=1e-6
    )
    assert scaled["norm"] == pytest.approx(spread * unit["norm"], rel=1e-5)


def test_the_token_read_by_default(tiny_backend, tmp_path):
    """Where the feature fires most from the first real token on, or the last token where it
    fires nowhere."""
    backend = tiny_backend
    d = backend.info.d_model
    torch.manual_seed(4)
    b_enc = torch.zeros(8)
    b_enc[0] = -1e9  # feature 0 never fires
    folder = _saelens(
        tmp_path / "dead",
        {"hook_name": "blocks.1.hook_resid_pre", "apply_b_dec_to_input": True},
        {
            "W_enc": torch.randn(d, 8),
            "b_enc": b_enc,
            "W_dec": torch.randn(8, d),
            "b_dec": torch.zeros(d),
        },
    )
    sae = SAE(repo="local/dead", path="", revision="test", params=read_sae(folder))
    ids = backend.tokenize(TEXT, True).ids
    f, _ = sae.encode(backend.capture(torch.tensor([ids]), [("resid_pre", 1)])[("resid_pre", 1)])
    feature = int(f[0, 1:].amax(0).argmax())
    report = feature_writes(backend, sae, feature, ids, first=1)
    assert report["chosen"] == "fires_most"
    assert report["position"] == 1 + int(f[0, 1:, feature].argmax())
    assert report["activation"] == pytest.approx(float(f[0, 1:, feature].max()))
    dead = feature_writes(backend, sae, 0, ids, first=1)
    assert dead["chosen"] == "last" and dead["position"] == len(ids) - 1
    assert dead["activation"] == 0.0
    with pytest.raises(ValueError, match="there is no feature"):
        feature_writes(backend, sae, 8, ids)
    with pytest.raises(ValueError, match="outside this prompt"):
        feature_writes(backend, sae, 1, ids, position=len(ids))
    assert feature_writes(backend, sae, 1, ids, position=-1)["position"] == len(ids) - 1
    # Same request, same lists: ties keep the vocabulary's order.
    assert feature_writes(backend, sae, 1, ids, position=3) == feature_writes(
        backend, sae, 1, ids, position=3
    )


def test_it_is_refused_where_it_isnt_defined(arch_backend, tmp_path):
    olmo = arch_backend("olmo2")  # normalizes the attention output after combining the heads
    info = olmo.info
    sae = _z_sae(tmp_path / "z", info.n_heads * info.d_head, layer=1)
    ids = olmo.tokenize(TEXT, True).ids
    with pytest.raises(ScopeError, match="sum of its heads"):
        feature_writes(olmo, sae, 0, ids)
    gemma = arch_backend("gemma2")
    report = feature_writes(gemma, exact_sae(tmp_path / "g", gemma.info.d_model), 0, ids)
    assert report["soft_cap"] == gemma.info.extra["logit_soft_cap"] > 0


def test_the_server_and_the_python_api_read_it(tiny_backend, tmp_path, monkeypatch, project):
    from fastapi.testclient import TestClient

    import logogram as lg
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
    sae = exact_sae(tmp_path / "exact", tiny_backend.info.d_model)
    monkeypatch.setattr(
        "logogram.sae.load_sae", lambda ref, device, progress=None, cancel=None: sae
    )
    client.post("/api/sae/load", json={"repo": "local/exact"}, headers=headers)
    for _ in range(200):
        if state.job.status != "running":
            break
        time.sleep(0.02)
    body = {
        "dataset": "datasets/ioi.jsonl",
        "prepend_bos": True,
        "batch_size": 8,
        "index": 2,
        "feature": 5,
        "position": {"kind": "last"},
        "top": 4,
    }
    report = client.post("/api/sae/writes", json=body, headers=headers).json()
    tokens = report["tokens"]
    assert report["position"] == len(tokens) - 1 and report["index"] == 2
    assert len(report["promoted"]) == 4 and len(report["suppressed"]) == 4
    from logogram.datasets import load_dataset

    text = load_dataset(project.datasets_dir / "ioi.jsonl")[2].clean
    direct = lg.feature_writes(tiny_backend, sae, 5, text, prepend_bos=True, position=-1, top=4)
    assert direct["tokens"] == tokens
    assert direct["promoted"] == report["promoted"] and direct["suppressed"] == report["suppressed"]
    default = client.post(
        "/api/sae/writes", json={**body, "position": None}, headers=headers
    ).json()
    assert default["chosen"] in ("fires_most", "last")
    missing = client.post("/api/sae/writes", json={**body, "feature": 10_000}, headers=headers)
    assert missing.status_code == 400 and "no feature" in missing.json()["error"]


def test_the_python_api_loads_an_sae_checked_against_the_model(tiny_backend, tmp_path, monkeypatch):
    import logogram as lg

    sae = exact_sae(tmp_path / "exact", tiny_backend.info.d_model)
    monkeypatch.setattr("logogram.sae.load_sae", lambda ref, device, **kwargs: sae)
    assert lg.load_sae("local/exact", model=tiny_backend) is sae
    wide = exact_sae(tmp_path / "wide", tiny_backend.info.d_model + 4)
    monkeypatch.setattr("logogram.sae.load_sae", lambda ref, device, **kwargs: wide)
    with pytest.raises(ScopeError, match="another model"):
        lg.load_sae("local/wide", model=tiny_backend)
