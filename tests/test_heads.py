"""Heads by what they attend to: the scores, their statistics, the endpoint and the Python API."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from logogram.backends.base import Tokenized
from logogram.heads import (
    HeadKindSettings,
    head_kind_scores,
    kind_scores_of_pattern,
    repeated_random_tokens,
)
from logogram.server.app import create_app
from logogram.server.security import SecurityConfig


def designed_pattern(length: int, offset: int) -> torch.Tensor:
    """[1, 3, T, T]: head 0 attends to the previous token, head 1 to the earlier copy of the current
    token, head 2 to the token after it; queries with no such key attend to themselves."""
    t = offset + 2 * length
    pattern = torch.zeros(1, 3, t, t, dtype=torch.float64)
    for q in range(t):
        targets = [q - 1, q - length, q - length + 1]
        for head, key in enumerate(targets):
            if q >= offset + length or (head == 0 and q > offset):
                pattern[0, head, q, key] = 1.0
            else:
                pattern[0, head, q, q] = 1.0
    return pattern


@pytest.mark.parametrize("offset", [0, 1])
def test_each_kind_scores_the_head_built_for_it(offset):
    scores = kind_scores_of_pattern(designed_pattern(6, offset), 6, offset)
    assert scores["previous_token"][0].tolist() == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    assert scores["duplicate_token"][0].tolist() == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    assert scores["induction"][0].tolist() == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)


class PlantedBackend:
    """Two layers of two heads; L1 H0 is an induction head and L0 H1 a previous-token head, the
    others attend uniformly. Only what head_kind_scores uses."""

    def __init__(self) -> None:
        self.info = SimpleNamespace(n_layers=2, n_heads=2, n_ctx=64, d_vocab=50)
        self.calls = 0

    def token_space(self) -> tuple[int, frozenset[int]]:
        return 50, frozenset({0})

    def tokenize(self, text: str, prepend_bos: bool) -> Tokenized:
        assert text == "" and prepend_bos
        return Tokenized(ids=[0], tokens=["<bos>"], offsets=[(0, 0)])

    def attention_patterns(
        self, tokens: torch.Tensor, layers: list[int]
    ) -> dict[int, torch.Tensor]:
        self.calls += 1
        b, t = tokens.shape
        uniform = torch.tril(torch.ones(t, t)) / torch.arange(1, t + 1).unsqueeze(1)
        out = {}
        length = (t - 1) // 2
        for layer in layers:
            pattern = uniform.expand(b, 2, t, t).clone()
            if layer == 1:
                induction = torch.zeros(t, t)
                for q in range(t):
                    key = q - length + 1 if q >= 1 + length else 0
                    induction[q, key] = 1.0
                pattern[:, 0] = induction
            else:
                previous = torch.zeros(t, t)
                for q in range(t):
                    previous[q, max(q - 1, 0)] = 1.0
                pattern[:, 1] = previous
            out[layer] = pattern.float()
        return out


def test_the_planted_heads_are_found_with_intervals():
    backend = PlantedBackend()
    settings = HeadKindSettings(
        seed=3, sequences=8, length=10, prepend_bos=True, batch_size=3, bootstrap=200
    )
    report = head_kind_scores(backend, settings)  # type: ignore[arg-type]
    assert backend.calls == 3  # batches of 3, 3 and 2
    top = report["top"]
    assert (top["induction"][0]["layer"], top["induction"][0]["head"]) == (1, 0)
    assert top["induction"][0]["mean"] == pytest.approx(1.0)
    assert (top["previous_token"][0]["layer"], top["previous_token"][0]["head"]) == (0, 1)
    for kind in report["kinds"].values():
        for row_mean, row_lo, row_hi in zip(kind["mean"], kind["lo"], kind["hi"], strict=True):
            for m, lo, hi in zip(row_mean, row_lo, row_hi, strict=True):
                assert 0.0 <= lo <= m + 1e-12 and m <= hi + 1e-12 and hi <= 1.0 + 1e-12
        assert len(kind["per_sequence"]) == 8


def test_sequences_are_seeded_distinct_and_free_of_special_tokens():
    backend = PlantedBackend()
    settings = HeadKindSettings(seed=7, sequences=5, length=12, prepend_bos=True, batch_size=4)
    a = repeated_random_tokens(backend, settings)  # type: ignore[arg-type]
    b = repeated_random_tokens(backend, settings)  # type: ignore[arg-type]
    assert torch.equal(a, b)
    assert a.shape == (5, 25) and (a[:, 0] == 0).all()
    first, second = a[:, 1:13], a[:, 13:]
    assert torch.equal(first, second)
    assert all(len(set(row.tolist())) == 12 for row in first)
    assert not (first == 0).any()
    other = repeated_random_tokens(
        backend, HeadKindSettings(seed=8, sequences=5, length=12, prepend_bos=True, batch_size=4)
    )  # type: ignore[arg-type]
    assert not torch.equal(a, other)


def test_settings_that_cannot_work_say_why():
    backend = PlantedBackend()
    with pytest.raises(ValueError, match="context of 64 tokens"):
        head_kind_scores(
            backend,
            HeadKindSettings(seed=0, sequences=4, length=40, prepend_bos=True, batch_size=4),
        )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 2 sequences"):
        head_kind_scores(
            backend, HeadKindSettings(seed=0, sequences=1, length=4, prepend_bos=True, batch_size=4)
        )  # type: ignore[arg-type]
    backend.info.n_ctx = 1000  # long enough: now the 49 ordinary tokens are too few
    with pytest.raises(ValueError, match="ordinary tokens"):
        head_kind_scores(
            backend,
            HeadKindSettings(seed=0, sequences=2, length=60, prepend_bos=False, batch_size=4),
        )  # type: ignore[arg-type]


def test_a_real_model_gives_identical_scores_twice(tiny_backend):
    settings = HeadKindSettings(seed=0, sequences=6, length=8, prepend_bos=True, batch_size=4)
    with tiny_backend.lock:
        a = head_kind_scores(tiny_backend, settings)
        b = head_kind_scores(tiny_backend, settings)
    assert a == b
    info = tiny_backend.info
    assert len(a["kinds"]["induction"]["mean"]) == info.n_layers
    assert len(a["kinds"]["induction"]["mean"][0]) == info.n_heads
    # The one-pass capture matches the per-layer one.
    tokens = repeated_random_tokens(tiny_backend, settings)
    together = tiny_backend.attention_patterns(tokens, list(range(info.n_layers)))
    for layer in range(info.n_layers):
        assert torch.equal(together[layer], tiny_backend.attention_pattern(tokens, layer))


def test_the_endpoint_and_the_python_api_agree(tmp_path, monkeypatch, tiny_backend):
    import logogram as lg

    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    app = create_app(SecurityConfig(token="t", port=8765), serve_web=False)
    app.state.logogram.backend = tiny_backend
    app.state.logogram.model_status = {"state": "ready"}
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    headers = {"authorization": "Bearer t", "origin": "http://127.0.0.1:8765"}
    body: dict[str, Any] = {
        "seed": 1,
        "sequences": 4,
        "length": 6,
        "prepend_bos": True,
        "batch_size": 2,
        "bootstrap": 200,
        "ci": 0.9,
    }
    response = client.post("/api/heads/kinds", json=body, headers=headers)
    assert response.status_code == 200, response.text
    served = response.json()
    direct = lg.head_kinds(
        tiny_backend,
        seed=1,
        sequences=4,
        length=6,
        prepend_bos=True,
        batch_size=2,
        bootstrap=200,
        ci=0.9,
    )
    assert served["kinds"]["induction"]["mean"] == direct["kinds"]["induction"]["mean"]
    assert served["settings"]["sequences"] == 4
    # A model that isn't the one the analysis names is refused, as for every analysis.
    other = {
        **body,
        "model": {
            "id": "other",
            "revision": "x",
            "dtype": "float32",
            "device": "cpu",
            "process_weights": True,
        },
    }
    assert client.post("/api/heads/kinds", json=other, headers=headers).status_code == 409
