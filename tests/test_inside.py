"""Reading any text without a dataset: predictions, the logit lens, attention and features."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from logogram.inside import look_attention, look_features, look_report
from logogram.server.app import create_app
from logogram.server.security import SecurityConfig

TEXT = "When Mary and John went to the store, John gave a drink to"


def test_predictions_match_the_model_and_the_lens_ends_where_the_model_does(tiny_backend):
    with tiny_backend.lock:
        report = look_report(tiny_backend, TEXT, prepend_bos=True, top_k=3, position=None)
    tokenized = tiny_backend.tokenize(TEXT, True)
    assert report["ids"] == tokenized.ids and report["tokens"] == tokenized.tokens
    n = len(tokenized.ids)
    assert report["position"] == n - 1 and len(report["predictions"]) == n
    logits = tiny_backend.logits(torch.tensor([tokenized.ids]), keep=n)[0]
    probs = torch.softmax(logits.double(), dim=-1)
    for pos, top in enumerate(report["predictions"]):
        assert top[0]["id"] == int(torch.argmax(logits[pos]))
        assert top[0]["prob"] == pytest.approx(float(probs[pos].max()))
        assert [t["prob"] for t in top] == sorted((t["prob"] for t in top), reverse=True)
    # The last layer's lens reads the model's own output at that token.
    assert report["lens"] is not None and len(report["lens"]) == tiny_backend.info.n_layers
    assert report["lens"][-1]["top"][0]["id"] == report["predictions"][-1][0]["id"]


def test_one_token_can_be_read_layer_by_layer_and_bad_input_says_why(tiny_backend):
    with tiny_backend.lock:
        report = look_report(tiny_backend, TEXT, prepend_bos=False, top_k=2, position=1)
        assert report["position"] == 1 and report["ids"][0] != 0
        with pytest.raises(ValueError, match="isn't in this text"):
            look_report(tiny_backend, TEXT, prepend_bos=True, top_k=2, position=500)
        with pytest.raises(ValueError, match="Type some text"):
            look_report(tiny_backend, "   ", prepend_bos=True, top_k=2, position=None)
        with pytest.raises(ValueError, match="reads at most 64"):
            look_report(
                tiny_backend, " ".join(["the"] * 80), prepend_bos=True, top_k=2, position=None
            )


def test_attention_and_features_on_the_text(tiny_backend, tmp_path):
    from test_sae import exact_sae

    with tiny_backend.lock:
        attention = look_attention(tiny_backend, TEXT, prepend_bos=True, layer=1, head=2)
        tokens = torch.tensor([tiny_backend.tokenize(TEXT, True).ids])
        expected = tiny_backend.attention_pattern(tokens, 1)[0, 2].double()
        assert torch.allclose(torch.tensor(attention["pattern"], dtype=torch.float64), expected)
        with pytest.raises(ValueError, match="doesn't exist"):
            look_attention(tiny_backend, TEXT, prepend_bos=True, layer=9, head=0)
        # The exact SAE's features are the positive and negative parts of each coordinate, so
        # every coordinate that isn't zero fires one feature.
        sae = exact_sae(tmp_path / "sae", tiny_backend.info.d_model)
        features = look_features(tiny_backend, sae, TEXT, prepend_bos=True, top_k=4)
        key = (sae.site_in, sae.layer)
        acts = tiny_backend.capture(tokens, [key])[key][0].float()
    assert features["tokens"] == tiny_backend.tokenize(TEXT, True).tokens
    assert features["active"] == [int((row != 0).sum()) for row in acts]
    strongest = features["features"][3][0]
    assert strongest["activation"] == pytest.approx(float(acts[3].abs().max()), rel=1e-5)


def test_the_endpoints_and_the_python_api(tmp_path, monkeypatch, tiny_backend):
    import logogram as lg

    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    app = create_app(SecurityConfig(token="t", port=8765), serve_web=False)
    app.state.logogram.backend = tiny_backend
    app.state.logogram.model_status = {"state": "ready"}
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    headers = {"authorization": "Bearer t", "origin": "http://127.0.0.1:8765"}
    base: dict[str, Any] = {"text": TEXT, "prepend_bos": True, "batch_size": 8}
    served = client.post("/api/look", json={**base, "top_k": 3}, headers=headers)
    assert served.status_code == 200, served.text
    assert served.json() == lg.look(tiny_backend, TEXT, top_k=3)
    attention = client.post(
        "/api/look/attention", json={**base, "layer": 0, "head": 1}, headers=headers
    )
    assert attention.status_code == 200 and attention.json()["head"] == 1
    # Features need an SAE, and say so.
    response = client.post("/api/look/features", json={**base, "top_k": 3}, headers=headers)
    assert response.status_code == 409 and "Load an SAE" in response.json()["error"]
    assert (
        client.post("/api/look", json={**base, "text": "", "top_k": 3}, headers=headers).status_code
        == 422
    )
