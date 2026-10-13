"""Steering along an SAE feature, and what the steered model says.

The decoder direction W_dec[f] is added where the SAE writes, at the feature's typical activation
for strength 1. Each place it is added is checked against adding the same vector to the residual
stream through TransformerLens's own hooks (not the backend's patches), so the equivalences are
measured: an attention output and the stream between attention and MLP, an MLP output and the
stream after the layer, the heads' outputs and the attention output through W_O. Continuations
are greedy, steered as measured, and the same every time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from logogram.backends.saes import read_sae
from logogram.datasets import load_dataset
from logogram.engine import EngineError, run_experiment
from logogram.inside import continue_text, look_feature, look_features
from logogram.prompts import prepare_prompts
from logogram.runner import run_spec
from logogram.sae import SAE
from logogram.schema import ContinuationsFile
from logogram.sites import ScopeError
from logogram.steering import greedy_next, split_pairs
from test_sae import _saelens, exact_sae
from test_transcoders import _gemma_scope, _params

SAE_REF = {"repo": "local/exact", "path": "", "revision": "test"}
TEXT = "When Mary and John went to the store, John gave a drink to"


def _steering(feature: int, **changes: Any) -> dict[str, Any]:
    exp: dict[str, Any] = {
        "kind": "steering",
        "apply_to": "corrupt",
        "direction": {"kind": "sae_feature", "feature": feature},
        "coefficients": [1.0, -2.0],
        "train_fraction": 0.5,
        "seed": 0,
        "control": True,
        "continuations": None,
    }
    exp.update(changes)
    return exp


def _at(kind: str, layer: int, position: dict[str, Any] | None = None) -> dict[str, Any]:
    site = {"kind": kind, "layer": layer, "position": position or {"kind": "last"}}
    return {"kind": "sites", "sites": [site]}


def _prompts(backend, project):
    return prepare_prompts(backend, load_dataset(project.datasets_dir / "ioi.jsonl"), True)


def _random_sae(folder: Path, hook: str, d: int, **cfg: Any) -> SAE:
    """An SAELens SAE with random weights, so a feature's decoder row is no special direction."""
    g = torch.Generator().manual_seed(7)
    tensors = {
        "W_enc": torch.randn(d, 4 * d, generator=g) * 0.5,
        "b_enc": torch.randn(4 * d, generator=g) * 0.1,
        "W_dec": torch.randn(4 * d, d, generator=g) * 0.3,
        "b_dec": torch.randn(d, generator=g) * 0.1,
    }
    params = read_sae(_saelens(folder, {"hook_name": hook, "d_in": d, **cfg}, tensors))
    return SAE(repo="local/exact", path="", revision="test", params=params)


def _sae_of(kind: str, folder: Path, info: Any) -> SAE:
    d, z = info.d_model, info.n_heads * info.d_head
    if kind == "resid_pre":
        return exact_sae(folder, d, layer=1)
    if kind == "standardized":
        return _random_sae(folder, "blocks.1.hook_resid_pre", d, normalize_activations="layer_norm")
    if kind == "head":
        return _random_sae(folder, "blocks.1.attn.hook_z", z)
    if kind == "transcoder":
        cfg = {
            "hf_hook_point_in": "transformer.h.0.ln_2.output",
            "hf_hook_point_out": "transformer.h.0.mlp.output",
            "architecture": "jump_relu",
            "affine_connection": False,
            "type": "transcoder",
        }
        params = read_sae(_gemma_scope(folder, cfg, _params(d, d, 3 * d, seed=1)))
        return SAE(repo="local/exact", path="", revision="test", params=params)
    layer = {"resid_post": 0, "attn_out": 1, "mlp_out": 0}[kind]
    return _random_sae(folder, f"blocks.{layer}.hook_{kind}", d)


def _read_at_last(backend, sae: SAE, ids: list[int]) -> torch.Tensor:
    """What the SAE reads at the last token of a prompt, ``[d_in]``, one prompt at a time."""
    key = (sae.site_in, sae.layer)
    return backend.capture(torch.tensor([ids]), [key])[key][0, -1].reshape(-1).float()


def _training_activations(backend, sae: SAE, prompts, train_at) -> torch.Tensor:
    """The feature activations at the last token of the training pairs' clean and corrupt
    prompts, ``[2 * len(train_at), d_sae]``."""
    rows = []
    for at in train_at:
        for ids in (prompts[at].clean.ids, prompts[at].corrupt.ids):
            f, _ = sae.encode(_read_at_last(backend, sae, ids)[None])
            rows.append(f[0])
    return torch.stack(rows)


# Where each SAE's write enters the residual stream, as TransformerLens hooks: the vector is added
# to each at the steered token. The stream between attention and MLP is two hooks: what the MLP
# reads (hook_resid_mid, which TransformerLens hands only to the MLP) and the stream it adds to.
STREAM = {
    "resid_pre": ["blocks.{layer}.hook_resid_pre"],
    "standardized": ["blocks.{layer}.hook_resid_pre"],
    "resid_post": ["blocks.{layer}.hook_resid_post"],
    "mlp_out": ["blocks.{layer}.hook_resid_post"],
    "transcoder": ["blocks.{layer}.hook_resid_post"],
    "attn_out": ["blocks.{layer}.hook_resid_mid", "blocks.{layer}.hook_resid_post"],
    "head": ["blocks.{layer}.hook_resid_mid", "blocks.{layer}.hook_resid_post"],
}


def _with_added(backend, prompts, test_at, hooks: list[str], vectors: list[torch.Tensor]):
    """logit(answer) − logit(distractor) of each held-out corrupt prompt, with its vector added at
    its last token to every one of ``hooks``."""
    out = []
    for at, vector in zip(test_at, vectors, strict=True):
        p = prompts[at]

        def add(act: torch.Tensor, hook: Any = None, vector: torch.Tensor = vector) -> torch.Tensor:
            act = act.clone()
            act[0, -1] = act[0, -1] + vector.to(act.dtype)
            return act

        with torch.no_grad():
            logits = backend.bridge.run_with_hooks(
                torch.tensor([p.corrupt.ids]),
                fwd_hooks=[(h, add) for h in hooks],
                return_type="logits",
            )[0, -1]
        out.append(float(logits[p.answer_id] - logits[p.distractor_id]))
    return np.array(out)


@pytest.mark.parametrize(
    "kind", ["resid_pre", "resid_post", "attn_out", "mlp_out", "transcoder", "head", "standardized"]
)
def test_strength_one_adds_what_the_feature_writes_at_its_typical_activation(
    tiny_backend, project, spec_factory, tmp_path, kind
):
    backend, info = tiny_backend, tiny_backend.info
    sae = _sae_of(kind, tmp_path / kind, info)
    prompts = _prompts(backend, project)
    train_at, test_at = split_pairs(len(prompts), 0.5, 0)
    training = _training_activations(backend, sae, prompts, train_at)
    feature = int(training.sum(0).argmax())  # one that fires at the last token
    site = "attn_out" if kind == "head" else sae.site
    spec = spec_factory(sae=SAE_REF, experiment=_steering(feature), scope=_at(site, sae.layer))
    result = run_experiment(spec, backend, prompts, sae=sae)

    # Strength 1: the mean of the feature's positive activations at the steered token of the
    # training pairs' clean and corrupt prompts.
    column = training[:, feature].double()
    typical = float(column[column > 0].mean())
    record = result.extra["steering"]["feature"]
    assert record["activations"] == [pytest.approx(typical, rel=1e-5)]
    assert record["fired"] == [int((column > 0).sum())]
    assert record["measured"] == [2 * len(train_at)]
    assert (record["site"], record["layer"], record["standardized"]) == (
        sae.site,
        sae.layer,
        kind == "standardized",
    )

    # What the feature writes at that activation, for each held-out prompt, in the residual
    # stream: its decoder row (times the spread of the prompt's own input, for an SAE that
    # standardizes its inputs; through W_O, for the heads' outputs).
    row = sae.params.W_dec[feature].double()
    vectors = []
    for at in test_at:
        unit = row
        if kind == "standardized":
            x = _read_at_last(backend, sae, prompts[at].corrupt.ids).double()
            unit = row * float((x - x.mean()).std())
        if kind == "head":
            z = unit.reshape(1, info.n_heads, info.d_head)
            unit = backend.head_writes(sae.layer, z)[0]
        vectors.append(typical * unit)
    hooks = [h.format(layer=sae.layer) for h in STREAM[kind]]
    by_key = {s.variant_key: s.index for s in result.sites}
    for c, key in ((1.0, "×1"), (-2.0, "×−2")):
        expected = _with_added(backend, prompts, test_at, hooks, [c * v for v in vectors])
        np.testing.assert_allclose(result.patched[by_key[key]], expected, atol=1e-4)
    # How long the vector is at strength 1: the decoder row times the typical activation (on
    # average over the held-out prompts, when each is scaled by its spread).
    lengths = [float((typical * row).norm())] if kind != "standardized" else []
    if kind == "standardized":
        spreads = [
            float((x - x.mean()).std())
            for x in (
                _read_at_last(backend, sae, prompts[at].corrupt.ids).double() for at in test_at
            )
        ]
        lengths = [float((typical * row).norm()) * float(np.mean(spreads))]
    assert result.extra["steering"]["norms"] == [pytest.approx(lengths[0], rel=1e-5)]


@pytest.mark.parametrize("arch", ["gpt2", "gpt_neox"])
def test_adding_where_an_sae_writes_is_adding_to_the_residual_stream(arch_backend, project, arch):
    """With hooks of TransformerLens's own: a vector added to a layer's attention output is the
    same as adding it to the residual stream between attention and MLP (after the layer, when
    attention and MLP read the same stream); one added to the MLP output, to the stream after the
    layer; and a change of the heads' outputs z, the same as adding what W_O makes of it to the
    attention output."""
    backend = arch_backend(arch)
    info = backend.info
    bridge = backend.bridge
    parallel = info.extra["block_structure"] == "parallel"
    assert parallel == (arch == "gpt_neox")
    prompt = _prompts(backend, project)[0]
    tokens = torch.tensor([prompt.corrupt.ids])
    g = torch.Generator().manual_seed(3)
    v = torch.randn(info.d_model, generator=g)
    dz = torch.randn(info.n_heads, info.d_head, generator=g)
    layer = 1

    def run(*hooks: tuple[str, torch.Tensor]) -> torch.Tensor:
        def add(vector: torch.Tensor) -> Any:
            def fn(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
                act = act.clone()
                act[0, -1] = act[0, -1] + vector.to(act.dtype)
                return act

            return fn

        with torch.no_grad():
            return bridge.run_with_hooks(
                tokens,
                fwd_hooks=[(name, add(vector)) for name, vector in hooks],
                return_type="logits",
            )[0, -1].double()

    plain = run()
    between = (
        [(f"blocks.{layer}.hook_resid_post", v)]
        if parallel
        else [(f"blocks.{layer}.hook_resid_mid", v), (f"blocks.{layer}.hook_resid_post", v)]
    )
    attn = run((f"blocks.{layer}.hook_attn_out", v))
    assert float((attn - plain).abs().max()) > 1e-2  # the vector changes the output
    torch.testing.assert_close(attn, run(*between), rtol=0, atol=1e-5)
    torch.testing.assert_close(
        run((f"blocks.{layer}.hook_mlp_out", v)),
        run((f"blocks.{layer}.hook_resid_post", v)),
        rtol=0,
        atol=1e-5,
    )
    # The heads' outputs reach the attention output through W_O, where heads add up (measured
    # when the model loaded).
    assert info.extra["checks"]["heads"] <= info.extra["checks"]["tolerance"]
    through = backend.head_writes(layer, dz[None])[0]
    torch.testing.assert_close(
        run((f"blocks.{layer}.attn.hook_z", dz)),
        run((f"blocks.{layer}.hook_attn_out", through)),
        rtol=0,
        atol=1e-5,
    )


def test_the_control_is_a_random_direction_as_long_as_the_vector_at_strength_one(
    tiny_backend, project, spec_factory, tmp_path
):
    from logogram.steering import random_direction

    backend = tiny_backend
    sae = exact_sae(tmp_path / "exact", backend.info.d_model)
    prompts = _prompts(backend, project)
    train_at, test_at = split_pairs(len(prompts), 0.5, 0)
    feature = int(_training_activations(backend, sae, prompts, train_at).sum(0).argmax())
    spec = spec_factory(
        sae=SAE_REF, experiment=_steering(feature, coefficients=[3.0]), scope=_at("resid_pre", 1)
    )
    result = run_experiment(spec, backend, prompts, sae=sae)
    assert [s.variant_key for s in result.sites] == ["×3", "random ×3"]
    length = result.extra["steering"]["norms"][0]
    typical = result.extra["steering"]["feature"]["activations"][0]
    assert length == pytest.approx(typical * float(sae.params.W_dec[feature].norm()), rel=1e-6)
    # Seeded by the split's seed and the row, as for the mean difference.
    control = random_direction(torch.Size([backend.info.d_model]), 0 * 1_000_003 + 0, length)
    assert float(control.norm()) == pytest.approx(length)
    expected = _with_added(
        backend, prompts, test_at, ["blocks.1.hook_resid_pre"], [3.0 * control] * len(test_at)
    )
    np.testing.assert_allclose(result.patched[1], expected, atol=1e-4)


def test_steering_along_a_feature_is_refused_where_it_cant_be_exact(
    tiny_backend, project, spec_factory, tmp_path
):
    backend = tiny_backend
    d = backend.info.d_model
    sae = exact_sae(tmp_path / "exact", d)  # the residual stream entering layer 1
    prompts = _prompts(backend, project)

    def run(scope: dict[str, Any] | None = None, with_sae: Any = sae, **changes: Any) -> Any:
        spec = spec_factory(
            sae=SAE_REF,
            experiment=_steering(changes.pop("feature", 3), **changes),
            scope=scope or _at("resid_pre", 1),
        )
        return run_experiment(spec, backend, prompts, sae=with_sae)

    every = {"kind": "layer_components", "components": ["resid_pre"], "position": {"kind": "last"}}
    for scope, message in (
        (every, "where the SAE writes: the residual stream entering layer 1.*not a site in every"),
        (_at("resid_pre", 0), "L1 resid pre.*L0 resid pre @ last isn't it"),
        (_at("resid_post", 1), "L1 resid post @ last isn't it"),
        (_at("resid_pre", 1, {"kind": "all"}), "one token of each prompt"),
    ):
        with pytest.raises(ScopeError, match=message):
            run(scope)
    with pytest.raises(ScopeError, match="no feature 64"):
        run(feature=64)
    with pytest.raises(EngineError, match="no SAE is loaded"):
        run(with_sae=None)
    with pytest.raises(ScopeError, match="made for another model"):
        run(with_sae=exact_sae(tmp_path / "wide", 48))
    # A feature that never fires where it is steered has no typical activation.
    silent = SAE(
        repo="local/exact",
        path="",
        revision="test",
        params=read_sae(
            _saelens(
                tmp_path / "silent",
                {"hook_name": "blocks.1.hook_resid_pre", "d_in": d},
                {
                    "W_enc": torch.zeros(d, 2),
                    "b_enc": torch.full((2,), -1.0),
                    "W_dec": torch.ones(2, d),
                    "b_dec": torch.zeros(d),
                },
            )
        ),
    )
    with pytest.raises(ScopeError, match="never fires at the last token of the 6 training pairs"):
        run(with_sae=silent, feature=1)
    # The mean difference still adds to the residual stream only.
    mean = {**_steering(0), "direction": {"kind": "mean_difference"}}
    with pytest.raises(ScopeError, match="Choose residual sites, or steer along an SAE feature"):
        run_experiment(
            spec_factory(experiment=mean, scope=_at("attn_out", 1)), backend, prompts, sae=None
        )


# -- what the steered model says ---------------------------------------------------------------


def test_greedy_takes_the_most_likely_token_and_the_lowest_id_among_ties():
    logits = torch.tensor([[0.0, 2.0, 2.0, 1.0], [5.0, 5.0, 5.0, 9.0], [1.0, 0.0, 3.0, 3.0]])
    assert greedy_next(logits, 4).tolist() == [1, 3, 2]
    # Only ids the tokenizer has: the padded end of a vocabulary is never chosen.
    assert greedy_next(logits, 3).tolist() == [1, 0, 2]
    with pytest.raises(EngineError, match="aren't finite numbers"):
        greedy_next(torch.tensor([[0.0, float("nan")]]), 2)


def _assert_greedy(backend, prompt: list[int], new: list[int], hooks: list[Any] | None = None):
    """Each new token is the most likely after the tokens before it (within float32 rounding,
    and exactly where the two most likely are clearly apart), the model run with ``hooks``."""
    size, _ = backend.token_space()
    seq = list(prompt)
    for token in new:
        with torch.no_grad():
            logits = backend.bridge.run_with_hooks(
                torch.tensor([seq]), fwd_hooks=hooks or [], return_type="logits"
            )[0, -1, :size].double()
        top = torch.topk(logits, 2).values
        assert float(logits[token]) >= float(top[0]) - 1e-4
        if float(top[0] - top[1]) > 1e-3:
            assert token == int(logits.argmax())
        seq.append(token)


def _add_from(vector: torch.Tensor, start: int, stop: int | None = None) -> Any:
    def fn(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
        act = act.clone()
        act[0, start:stop] = act[0, start:stop] + vector.to(act.dtype)
        return act

    return fn


def test_continuations_are_greedy_steered_as_measured_and_the_same_every_time(
    tiny_backend, project, spec_factory, tmp_path
):
    backend = tiny_backend
    sae = exact_sae(tmp_path / "exact", backend.info.d_model)
    prompts = _prompts(backend, project)
    train_at, test_at = split_pairs(len(prompts), 0.5, 0)
    feature = int(_training_activations(backend, sae, prompts, train_at).sum(0).argmax())
    spec = spec_factory(
        sae=SAE_REF,
        experiment=_steering(
            feature, coefficients=[0.0, 4.0], continuations={"prompts": 3, "tokens": 6}
        ),
        scope=_at("resid_pre", 1),
        execution={"batch_size": 5},
    )
    first = run_spec(spec, project, backend=backend, sae=sae)
    second = run_spec(spec, project, backend=backend, sae=sae)
    assert first.status == second.status == "finished", first.manifest.get("error")
    saved = [
        json.loads((o.folder / "continuations.json").read_text("utf-8")) for o in (first, second)
    ]
    assert saved[0]["run_id"] == first.run_id
    assert {**saved[0], "run_id": ""} == {**saved[1], "run_id": ""}  # the same every time
    data = ContinuationsFile.model_validate(saved[0]).model_dump(mode="json")
    assert data["receiver"] == "corrupt" and data["tokens"] == 6 and data["decoding"] == "greedy"
    assert data["rows"] == [{"row": 0, "label": "L1 resid pre @ last"}]
    # The first held-out prompts, in split order.
    assert [p["index"] for p in data["prompts"]] == [prompts[at].index for at in test_at[:3]]
    typical = first.summary["steering"]["feature"]["activations"][0]
    for continued, at in zip(data["prompts"], test_at, strict=False):
        prompt = prompts[at]
        assert continued["text"] == prompt.record.corrupt
        assert continued["prompt"] == prompt.corrupt.tokens
        assert [s["key"] for s in continued["steered"]] == ["×0", "×4", "random ×0", "random ×4"]
        assert all(len(s["ids"]) == len(s["tokens"]) == 6 for s in continued["steered"])
        assert continued["unsteered"]["tokens"] == [
            backend.token_str(i) for i in continued["unsteered"]["ids"]
        ]
        _assert_greedy(backend, prompt.corrupt.ids, continued["unsteered"]["ids"])
        # Steered as measured: on every forward pass the vector is added at the steered token of
        # the prompt (its last), and the tokens generated after it aren't steered.
        last = prompt.length - 1
        vector = 4.0 * typical * sae.params.W_dec[feature]
        hook = ("blocks.1.hook_resid_pre", _add_from(vector, last, last + 1))
        _assert_greedy(backend, prompt.corrupt.ids, continued["steered"][1]["ids"], [hook])


def test_continuations_follow_the_mean_difference_at_every_steered_site(
    tiny_backend, project, spec_factory
):
    spec = spec_factory(
        experiment={
            **_steering(0, coefficients=[2.0], control=False),
            "direction": {"kind": "mean_difference"},
            "continuations": {"prompts": 2, "tokens": 3},
        },
        scope={
            "kind": "layer_components",
            "components": ["resid_pre"],
            "position": {"kind": "last"},
        },
    )
    result = run_experiment(spec, tiny_backend, _prompts(tiny_backend, project))
    record = result.extra["continuations"]
    assert [r["label"] for r in record["rows"]] == ["L0 resid pre @ last", "L1 resid pre @ last"]
    for prompt in record["prompts"]:
        assert [(s["row"], s["key"]) for s in prompt["steered"]] == [(0, "×2"), (1, "×2")]
    assert "feature" not in result.extra["steering"]


def test_continuations_that_wouldnt_fit_the_model_are_refused(
    tiny_backend, project, spec_factory, tmp_path
):
    sae = exact_sae(tmp_path / "exact", tiny_backend.info.d_model)
    spec = spec_factory(
        sae=SAE_REF,
        experiment=_steering(3, continuations={"prompts": 1, "tokens": 64}),
        scope=_at("resid_pre", 1),
    )
    with pytest.raises(EngineError, match="would pass the 64 tokens this model reads at once"):
        run_experiment(spec, tiny_backend, _prompts(tiny_backend, project), sae=sae)


def test_a_steering_run_keeps_what_the_model_said_and_shows_it(
    tiny_backend, project, spec_factory, tmp_path, monkeypatch
):
    import logogram as lg
    from logogram.server.app import create_app
    from logogram.server.security import SecurityConfig

    sae = exact_sae(tmp_path / "exact", tiny_backend.info.d_model)
    prompts = _prompts(tiny_backend, project)
    train_at, _ = split_pairs(len(prompts), 0.5, 0)
    # One that fires at the last token of the training pairs, whatever the tiny model's weights.
    f = int(_training_activations(tiny_backend, sae, prompts, train_at).sum(0).argmax())
    spec = spec_factory(
        sae=SAE_REF,
        experiment=_steering(f, continuations={"prompts": 2, "tokens": 4}),
        scope=_at("resid_pre", 1),
    )
    outcome = run_spec(spec, project, backend=tiny_backend, sae=sae)
    assert outcome.status == "finished", outcome.manifest.get("error")
    feature = outcome.summary["steering"]["feature"]
    assert feature["feature"] == f and feature["sae"] == {
        **SAE_REF,
        "source": "hub",
        "sha256": None,
    }

    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    app = create_app(SecurityConfig(token="t", port=8765), serve_web=False)
    state = app.state.logogram
    state.backend = tiny_backend
    state.model_status = {"state": "ready"}
    state.open_project(project)
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    headers = {"authorization": "Bearer t", "origin": "http://127.0.0.1:8765"}
    detail = client.get(f"/api/runs/{outcome.run_id}", headers=headers).json()
    said = detail["continuations"]
    assert said["prompts"][0]["steered"][0]["key"] == "×1" and len(said["prompts"]) == 2
    assert said == lg.load_run(project, outcome.run_id).continuations()

    page = client.get(f"/api/runs/{outcome.run_id}/report.html", headers=headers).text
    assert "What the steered model says" in page and f"Strength 1: feature {f}" in page
    first = said["prompts"][0]["unsteered"]["tokens"][0].replace(" ", "·")
    assert f'<span class="tok">{first}</span>' in page
    methods = client.get(f"/api/runs/{outcome.run_id}/methods", headers=headers).json()["text"]
    for words in (
        f"decoder direction W_dec[{f}] of feature {f} of the SAE local/exact at revision test",
        "the residual stream entering layer 1",
        "strength 1 is the feature's typical activation",
        f"The typical activation was {feature['activations'][0]:.4g}",
        "Greedy continuations of 4 new tokens",
        "the generated tokens weren't steered themselves",
    ):
        assert words in methods
    assert f"along SAE feature {f}" in outcome.summary["description"]


# -- continuing your text --------------------------------------------------------------------


def test_your_text_continues_greedily_with_and_without_a_feature(tiny_backend, tmp_path):
    backend = tiny_backend
    sae = exact_sae(tmp_path / "exact", backend.info.d_model)
    with backend.lock:
        plain = continue_text(backend, TEXT, prepend_bos=True, tokens=8)
        assert plain["steer"] is None and plain["steered"] is None
        assert plain["ids"] == backend.tokenize(TEXT, True).ids and plain["new"] == 8
        _assert_greedy(backend, plain["ids"], plain["plain"]["ids"])

        along = look_feature(backend, sae, TEXT, prepend_bos=True, feature=3)
        tokens = torch.tensor([plain["ids"]])
        x = backend.capture(tokens, [("resid_pre", 1)])[("resid_pre", 1)][0].float()
        f, _ = sae.encode(x)
        np.testing.assert_allclose(along["activations"], f[:, 3].double().numpy(), rtol=1e-6)
        assert along["largest"] == pytest.approx(float(f[1:, 3].max()))  # after the BOS token
        assert along["position"] == 1 + int(f[1:, 3].argmax())

        # The feature added from token 3 on, the new tokens included.
        steered = continue_text(
            backend, TEXT, prepend_bos=True, tokens=8, sae=sae, feature=3, activation=6.0, start=3
        )
        assert steered["plain"] == plain["plain"]
        assert steered["steer"] == {
            "feature": 3,
            "activation": 6.0,
            "start": 3,
            "site": "resid_pre",
            "layer": 1,
            "where": "the residual stream entering layer 1",
            "largest": along["largest"],
            "largest_at": along["position"],
        }
        hook = ("blocks.1.hook_resid_pre", _add_from(6.0 * sae.params.W_dec[3], 3))
        _assert_greedy(backend, plain["ids"], steered["steered"]["ids"], [hook])
        assert steered["steered"]["ids"] != plain["plain"]["ids"]
        # Nothing added: the plain continuation, exactly.
        zero = continue_text(
            backend, TEXT, prepend_bos=True, tokens=8, sae=sae, feature=3, activation=0.0, start=1
        )
        assert zero["steered"] == zero["plain"]

        for kwargs, message in (
            ({"tokens": 0}, "by 1 to 64 tokens"),
            ({"tokens": 65}, "by 1 to 64 tokens"),
            ({"tokens": 60}, "reads at most 64 at once: continue it by at most"),
            ({"tokens": 4, "feature": 3, "activation": 1.0}, "Load an SAE"),
            ({"tokens": 4, "sae": sae, "feature": 64, "activation": 1.0}, "no feature 64"),
            ({"tokens": 4, "sae": sae, "feature": 3, "activation": float("inf")}, "a number"),
            ({"tokens": 4, "sae": sae, "feature": 3, "activation": 1.0, "start": 99}, "Token 99"),
        ):
            with pytest.raises(ValueError, match=message):
                continue_text(backend, TEXT, prepend_bos=True, **kwargs)


def test_the_heads_outputs_are_steered_where_the_sae_writes_them(tiny_backend, tmp_path):
    """An SAE on z adds the feature to every head's output, which reaches the stream through W_O."""
    backend = tiny_backend
    info = backend.info
    sae = _sae_of("head", tmp_path / "z", info)
    with backend.lock:
        steered = continue_text(
            backend, TEXT, prepend_bos=True, tokens=5, sae=sae, feature=2, activation=5.0, start=1
        )
        # Its features are read from every head's output side by side, token by token.
        features = look_features(backend, sae, TEXT, prepend_bos=True, top_k=3)
    assert len(features["features"]) == len(steered["tokens"])
    assert steered["steer"]["where"].startswith("the outputs (z) of layer 1's attention heads")
    z = 5.0 * sae.params.W_dec[2].reshape(info.n_heads, info.d_head)
    hook = ("blocks.1.attn.hook_z", _add_from(z, 1))
    _assert_greedy(backend, steered["ids"], steered["steered"]["ids"], [hook])


def test_the_endpoints_and_the_python_api_continue_your_text(tmp_path, monkeypatch, tiny_backend):
    import logogram as lg
    from logogram.server.app import create_app
    from logogram.server.security import SecurityConfig

    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    app = create_app(SecurityConfig(token="t", port=8765), serve_web=False)
    state = app.state.logogram
    state.backend = tiny_backend
    state.model_status = {"state": "ready"}
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    headers = {"authorization": "Bearer t", "origin": "http://127.0.0.1:8765"}
    base: dict[str, Any] = {"text": TEXT, "prepend_bos": True, "batch_size": 8}
    plain = client.post("/api/look/continue", json={**base, "tokens": 5}, headers=headers)
    assert plain.status_code == 200, plain.text
    assert plain.json() == lg.continue_text(tiny_backend, TEXT, tokens=5)
    steer = {"feature": 3, "activation": 2.5, "start": 2}
    missing = client.post(
        "/api/look/continue", json={**base, "tokens": 5, "steer": steer}, headers=headers
    )
    assert missing.status_code == 409 and "Load an SAE" in missing.json()["error"]

    sae = exact_sae(tmp_path / "exact", tiny_backend.info.d_model)
    state.sae = sae
    state.sae_status = {"state": "ready"}
    served = client.post(
        "/api/look/continue", json={**base, "tokens": 5, "steer": steer}, headers=headers
    )
    assert served.status_code == 200, served.text
    assert served.json() == lg.continue_text(
        tiny_backend, TEXT, tokens=5, sae=sae, feature=3, activation=2.5, start=2
    )
    along = client.post("/api/look/feature", json={**base, "feature": 3}, headers=headers)
    assert along.status_code == 200 and along.json()["largest"] == served.json()["steer"]["largest"]
    for body in ({**base, "tokens": 65}, {**base, "tokens": 5, "steer": {**steer, "start": -1}}):
        assert client.post("/api/look/continue", json=body, headers=headers).status_code == 422
    beyond = client.post(
        "/api/look/continue",
        json={**base, "tokens": 5, "steer": {**steer, "feature": 999}},
        headers=headers,
    )
    assert beyond.status_code == 400 and "no feature 999" in beyond.json()["error"]
