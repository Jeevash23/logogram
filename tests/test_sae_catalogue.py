"""Published SAEs by name: the catalogue of SAELens's registry that ships with Logogram, what it
says Logogram reads of each SAE, and how it is used (matched to a loaded model, searched, loaded by
release and id from the app, the command line and Python), all without the network. Also the
script that writes it, run here on a registry and repositories made up in test code."""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch
from safetensors.torch import save_file
from typer.testing import CliRunner

import logogram.backends.catalogue as catalogue_module
from catalogue_fixture import REVISION, fixture_catalogue, serve_published
from logogram.backends.base import BackendError
from logogram.backends.catalogue import (
    CatalogueFile,
    load_catalogue,
    order,
    parse_catalogue,
    registry_refusal,
    search,
)
from logogram.backends.saes import read_config, read_sae, unreadable_files, weights_file
from test_sae import _saelens

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "src" / "logogram" / "backends" / "sae_catalogue.json"


@pytest.fixture
def offline(monkeypatch):
    """No connection can be opened: anything that reaches for the network fails the test. A
    connection to this machine's loopback address never leaves it, and is let through: on Windows,
    asyncio makes each event loop's socket pair with one (the test client starts event loops)."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"Logogram reached for the network: {args!r}")

    loopback = ("127.0.0.1", "::1")
    connect = socket.socket.connect

    def connect_locally(self: socket.socket, address: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(address, tuple) and address[0] in loopback:
            return connect(self, address, *args, **kwargs)
        return refuse(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect_locally)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    load_catalogue.cache_clear()
    yield
    load_catalogue.cache_clear()


@pytest.fixture
def tiny_catalogue(monkeypatch, tiny_backend, tmp_path):
    """The made-up catalogue for the tiny model in place of the real one, its SAE local/exact
    served from a folder in place of Hugging Face."""
    made = fixture_catalogue(tiny_backend.info.id, tiny_backend.info.d_model)
    monkeypatch.setattr(catalogue_module, "load_catalogue", lambda: made)
    serve_published(monkeypatch.setattr, tmp_path / "published", tiny_backend.info.d_model)
    return made


# -- the catalogue that ships ------------------------------------------------------------------


def test_the_shipped_catalogue_holds_to_its_schema_and_names_every_sae(offline):
    raw = json.loads(SHIPPED.read_text(encoding="utf-8"))
    CatalogueFile.model_validate(raw)
    shipped = load_catalogue()
    source = shipped.source
    # Where it came from, and SAELens's licence, whose notice travels with it.
    assert source.package == "sae-lens" and source.file == "sae_lens/pretrained_saes.yaml"
    assert source.version[0].isdigit() and len(source.sha256) == 64
    assert source.licence == "MIT"
    assert "Copyright" in source.notice and "Permission is hereby granted" in source.notice
    assert len(shipped.entries) > 10_000
    seen = set()
    for e in shipped.entries:
        # Every entry names a model, a repository and a folder.
        assert e.model and e.hf_models and all(e.hf_models), e
        assert "/" in e.hf_models[0], e
        assert e.repo.count("/") == 1 and e.folder.strip(), e
        assert (e.release, e.id) not in seen, e
        seen.add((e.release, e.id))
        # An SAE read exactly says where it reads; any other says why not.
        assert (e.reason is None) == e.exact, e
        if e.exact:
            assert e.site and e.layer is not None and e.width, e
        if e.verified is not None:
            assert e.verified.revision == e.revision, e
    reads = {r: sum(e.reads == r for e in shipped.entries) for r in ("exact", "refused")}
    assert reads["exact"] > 5_000 and reads["refused"] > 1_000
    # Some were checked on their files, and some of those on their models.
    checked = [e for e in shipped.entries if e.verified is not None]
    assert any(e.verified.read == "weights" and e.verified.fit for e in checked)
    assert len(checked) > 100


def test_what_the_catalogue_says_of_well_known_saes(offline):
    shipped = load_catalogue()
    res = shipped.find("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
    assert (res.repo, res.folder) == (
        "jbloom/GPT2-Small-SAEs-Reformatted",
        "blocks.8.hook_resid_pre",
    )
    assert (res.site, res.layer, res.width, res.reads) == ("resid_pre", 8, 24576, "exact")
    assert res.hf_models[0] == "openai-community/gpt2"
    # Gemma Scope 2's affine transcoders are skip transcoders, read exactly.
    affine = shipped.find(
        "gemma-scope-2-270m-pt-transcoders-all", "layer_12_width_16k_l0_small_affine"
    )
    assert (affine.kind, affine.site, affine.layer, affine.reads) == (
        "skip_transcoder",
        "mlp_out",
        12,
        "exact",
    )
    # The first Gemma Scope is published as NumPy archives, never read.
    npz = shipped.find("gemma-scope-2b-pt-res-canonical", "layer_12/width_16k/canonical")
    assert npz.reads == "refused" and "NumPy archive" in (npz.reason or "")
    # Pickled weights are never opened.
    pickled = shipped.find("gpt2-small-hook-z-kk", "blocks.5.hook_z")
    assert pickled.reads == "refused" and "pickle" in (pickled.reason or "")


def test_saes_are_matched_to_a_model_by_its_hugging_face_id(offline):
    shipped = load_catalogue()
    gpt2 = shipped.for_model("openai-community/gpt2")
    assert gpt2 and {e.model for e in gpt2} == {"gpt2-small"}
    assert ("gpt2-small-res-jb", "blocks.8.hook_resid_pre") in {(e.release, e.id) for e in gpt2}
    # The name it was published under, and any case, find the same SAEs.
    assert shipped.for_model("gpt2") == gpt2 == shipped.for_model(" OpenAI-Community/GPT2 ")
    # A base model and its instruction-tuned sibling are different models.
    base = shipped.for_model("google/gemma-3-270m")
    tuned = shipped.for_model("google/gemma-3-270m-it")
    assert base and tuned and not {e.release for e in base} & {e.release for e in tuned}
    assert all("-pt-" in e.release for e in base)
    # A model that isn't in the registry, or a local one, matches nothing.
    assert shipped.for_model("EleutherAI/pythia-70m") == []
    assert shipped.for_model("my-fine-tune") == []


def test_the_loaded_tiny_model_finds_its_published_saes(tiny_catalogue, tiny_backend, offline):
    found = catalogue_module.load_catalogue().for_model(tiny_backend.info.id)
    assert [e.release for e in order(found)] == [
        "tiny-res",
        "tiny-res",
        "tiny-transcoders",
        "tiny-transcoders",
        "tiny-attn-pickled",
    ]
    assert [e.reads for e in order(found)] == ["exact", "exact", "exact", "unknown", "refused"]
    assert catalogue_module.load_catalogue().for_model("tiny-alias") == found


def test_saes_are_searched_by_release_hook_layer_and_width(tiny_backend):
    made = fixture_catalogue(tiny_backend.info.id, tiny_backend.info.d_model)
    every = made.for_model(tiny_backend.info.id)

    def ids(query: str) -> list[str]:
        return sorted(e.id for e in search(every, query))

    assert ids("") == sorted(e.id for e in every)
    assert ids("tiny-res") == ["blocks.0.hook_resid_pre", "blocks.1.hook_resid_pre"]
    assert ids("hook_z") == ["blocks.1.hook_z"]
    assert (
        ids("layer 1")
        == ids("layer1")
        == ids("L1")
        == [
            "blocks.1.hook_resid_pre",
            "blocks.1.hook_z",
            "layer_1_width_64k",
        ]
    )
    # Widths in thousands or powers of two, or exactly.
    assert ids("64k") == ids("65k") == ids("65536") == ["layer_0_width_64k", "layer_1_width_64k"]
    assert ids("16k") == ["blocks.0.hook_resid_pre"]
    # Sites in Logogram's words and the publisher's, and words together.
    assert (
        ids("residual")
        == ids("resid_pre")
        == ["blocks.0.hook_resid_pre", "blocks.1.hook_resid_pre"]
    )
    assert ids("transcoder layer 0") == ["layer_0_width_64k"]
    assert ids("attention heads") == ["blocks.1.hook_z"]
    assert ids("nothing like this") == []


def test_saes_are_found_by_release_and_id(offline):
    shipped = load_catalogue()
    with pytest.raises(BackendError, match=r"no release 'gpt2-small-res'.*gpt2-small-res-jb"):
        shipped.find("gpt2-small-res", "blocks.8.hook_resid_pre")
    with pytest.raises(BackendError, match=r"has no SAE 'blocks.99.hook_resid_pre'.*blocks.0"):
        shipped.find("gpt2-small-res-jb", "blocks.99.hook_resid_pre")
    assert [
        e.release
        for e in shipped.at("JBLOOM/GPT2-Small-SAEs-Reformatted", "blocks.8.hook_resid_pre/")
    ] == ["gpt2-small-res-jb"]


def test_an_sae_its_registry_reads_otherwise_is_refused_before_anything_downloads(
    offline, monkeypatch
):
    """Mistral 7B's residual SAEs say nothing about rescaling their inputs, but SAELens's registry
    reads them with constant_norm_rescale: their files alone don't say what they compute."""
    from logogram.sae import load_sae
    from logogram.spec import SAERef

    entry = load_catalogue().find("mistral-7b-res-wg", "blocks.8.hook_resid_pre")
    assert entry.reads == "refused" and entry.registry == {
        "normalize_activations": "constant_norm_rescale"
    }
    reason = registry_refusal(entry.repo, entry.folder)
    assert reason == entry.reason and "constant_norm_rescale" in reason

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("downloaded an SAE the catalogue refuses")

    monkeypatch.setattr("logogram.backends.saes.download_sae", never)
    with pytest.raises(BackendError, match="constant_norm_rescale"):
        load_sae(SAERef(repo=entry.repo, path=entry.folder, revision=None), "cpu")
    assert registry_refusal("jbloom/GPT2-Small-SAEs-Reformatted", "blocks.8.hook_resid_pre") is None


# -- the app, the command line and Python ------------------------------------------------------


def _app(tmp_path, monkeypatch, backend=None):
    from fastapi.testclient import TestClient

    from logogram.server.app import create_app
    from logogram.server.security import SecurityConfig

    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    app = create_app(SecurityConfig(token="t", port=8765), serve_web=False)
    if backend is not None:
        app.state.logogram.backend = backend
        app.state.logogram.model_status = {"state": "ready"}
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    return app, client, {"authorization": "Bearer t", "origin": "http://127.0.0.1:8765"}


def test_the_app_lists_the_published_saes_for_a_model(tmp_path, monkeypatch, offline):
    _, client, headers = _app(tmp_path, monkeypatch)
    out = client.get("/api/sae/catalogue?model=openai-community/gpt2", headers=headers).json()
    shipped = load_catalogue()
    assert out["source"] == f"SAELens {shipped.source.version}"
    assert out["total"] == len(shipped.for_model("gpt2")) == out["matches"]
    assert out["exact"] == sum(e.exact for e in shipped.for_model("gpt2"))
    assert len(out["saes"]) == min(100, out["total"])
    # Those read exactly first.
    reads = [s["reads"] for s in out["saes"]]
    assert reads == sorted(reads, key=["exact", "unknown", "refused"].index)
    searched = client.get(
        "/api/sae/catalogue",
        params={"model": "gpt2", "q": "res-jb layer 8", "limit": 5},
        headers=headers,
    ).json()
    # gpt2-small-res-jb, and its feature-splitting release, all at layer 8.
    assert searched["matches"] == len(search(shipped.for_model("gpt2"), "res-jb layer 8")) > 5
    assert len(searched["saes"]) == 5
    assert all(s["layer"] == 8 and "res-jb" in s["release"] for s in searched["saes"])
    first = searched["saes"][0]
    assert (first["release"], first["id"], first["repo"], first["folder"]) == (
        "gpt2-small-res-jb",
        "blocks.8.hook_resid_pre",
        "jbloom/GPT2-Small-SAEs-Reformatted",
        "blocks.8.hook_resid_pre",
    )
    none = client.get("/api/sae/catalogue?model=my-fine-tune", headers=headers).json()
    assert (none["total"], none["saes"]) == (0, [])
    # Like every request, it needs the session token.
    assert client.get("/api/sae/catalogue?model=gpt2").status_code == 401


def test_the_app_loads_a_published_sae_by_its_repository_and_folder(
    tiny_catalogue, tiny_backend, tmp_path, monkeypatch
):
    app, client, headers = _app(tmp_path, monkeypatch, tiny_backend)
    listed = client.get(f"/api/sae/catalogue?model={tiny_backend.info.id}", headers=headers).json()
    chosen = listed["saes"][0]
    assert (chosen["release"], chosen["id"]) == ("tiny-res", "blocks.1.hook_resid_pre")
    body = {"repo": chosen["repo"], "path": chosen["folder"], "revision": None}
    assert client.post("/api/sae/load", json=body, headers=headers).json()["kind"] == "load_sae"
    state = app.state.logogram
    for _ in range(500):
        if state.job.status != "running":
            break
        time.sleep(0.02)
    sae = client.get("/api/state", headers=headers).json()["sae"]
    assert sae["state"] == "ready", sae
    # The spec records where it was loaded from, at the exact revision loading pinned.
    assert (sae["info"]["repo"], sae["info"]["path"], sae["info"]["revision"]) == (
        "local/exact",
        "exact",
        REVISION,
    )


def test_logogram_saes_lists_them_on_the_command_line(offline):
    from logogram.cli import app as cli

    result = CliRunner().invoke(cli, ["saes", "openai-community/gpt2", "res-jb", "layer", "8"])
    assert result.exit_code == 0, result.output
    shipped = load_catalogue()
    total = len(shipped.for_model("gpt2"))
    found = len(search(shipped.for_model("gpt2"), "res-jb layer 8"))
    assert (
        f"{total} SAEs for openai-community/gpt2 in SAELens {shipped.source.version}"
        in result.output
    )
    assert f"{found} matches “res-jb layer 8”." in result.output
    assert "gpt2-small-res-jb  (jbloom/GPT2-Small-SAEs-Reformatted)" in result.output
    assert "blocks.8.hook_resid_pre  layer 8" in result.output
    assert "reads exactly" in result.output and "layer 7" not in result.output

    refused = CliRunner().invoke(cli, ["saes", "gpt2", "hook-z-kk", "layer", "5"])
    assert "refused" in refused.output and "Why: Its weights are a PyTorch pickle" in refused.output
    exact = CliRunner().invoke(cli, ["saes", "gpt2", "--exact", "--json"])
    rows = json.loads(exact.output)
    assert rows and all(r["reads"] == "exact" for r in rows)
    assert {"release", "id", "repo", "folder", "hook", "layer", "width"} <= set(rows[0])

    missing = CliRunner().invoke(cli, ["saes", "gemma-3-270m"])
    assert missing.exit_code == 1
    assert "has no SAEs for gemma-3-270m" in missing.output
    assert "google/gemma-3-270m" in missing.output


def test_python_loads_an_sae_by_release_and_id(tiny_catalogue, tiny_backend, monkeypatch):
    import logogram as lg
    from logogram.api import LogogramError

    sae = lg.load_sae(release="tiny-res", sae_id="blocks.1.hook_resid_pre", model=tiny_backend)
    assert (sae.repo, sae.path, sae.revision) == ("local/exact", "exact", REVISION)
    assert (sae.site, sae.layer) == ("resid_pre", 1)

    table = lg.list_saes(tiny_backend)
    rows = table.to_dict("records") if hasattr(table, "to_dict") else table.to_pylist()
    assert [r["reads"] for r in rows] == ["exact", "exact", "exact", "unknown", "refused"]
    searched = lg.list_saes(tiny_backend.info.id, "64k")
    assert len(searched) == 2

    downloads: list[str] = []

    def record(repo: str, *args: Any, **kwargs: Any) -> Any:
        downloads.append(repo)
        raise AssertionError("downloaded an SAE the catalogue refuses")

    monkeypatch.setattr("logogram.backends.saes.download_sae", record)
    with pytest.raises(
        LogogramError, match=r"can't read tiny-attn-pickled blocks\.1\.hook_z exactly"
    ):
        lg.load_sae(release="tiny-attn-pickled", sae_id="blocks.1.hook_z", model=tiny_backend)
    assert downloads == []
    with pytest.raises(LogogramError, match="either by its repository"):
        lg.load_sae("local/exact", release="tiny-res", sae_id="x", model=tiny_backend)
    with pytest.raises(LogogramError, match="give both"):
        lg.load_sae(release="tiny-res", model=tiny_backend)
    with pytest.raises(LogogramError, match="Name the SAE"):
        lg.load_sae(model=tiny_backend)
    with pytest.raises(BackendError, match="no release 'nope'"):
        lg.load_sae(release="nope", sae_id="x", model=tiny_backend)


# -- what is read of an SAE from its configuration alone ---------------------------------------


def _same(dry: Any, read: Any) -> None:
    for name in ("site", "site_in", "layer", "d_sae", "activation", "k", "subtract_b_dec"):
        assert getattr(dry, name) == getattr(read, name), name
    assert (dry.normalize, dry.format, dry.transcoder) == (
        read.normalize,
        read.format,
        read.transcoder,
    )
    assert (dry.W_skip is None) == (read.W_skip is None)
    assert (dry.threshold is None) == (read.threshold is None)


SAELENS = {"hook_name": "blocks.1.hook_resid_pre", "d_in": 8, "d_sae": 16}


def _weights(d: int, f: int, **extra: torch.Tensor) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {
        "W_enc": torch.randn(d, f, generator=g),
        "b_enc": torch.randn(f, generator=g),
        "W_dec": torch.randn(f, d, generator=g),
        "b_dec": torch.randn(d, generator=g),
        **extra,
    }


@pytest.mark.parametrize(
    "cfg",
    [
        SAELENS,
        {**SAELENS, "architecture": "standard", "normalize_activations": "layer_norm"},
        # SAELens's configurations before version 6: TopK said by the activation function.
        {**SAELENS, "activation_fn_str": "topk", "activation_fn_kwargs": {"k": 3}},
        {**SAELENS, "architecture": "jumprelu", "apply_b_dec_to_input": False},
        {
            "hook_name": "blocks.0.attn.hook_z",
            "d_in": 8,
            "d_sae": 16,
            "reshape_activations": "hook_z",
        },
    ],
)
def test_a_configuration_alone_reads_as_the_files_do(tmp_path, cfg):
    extra = {"threshold": torch.full((16,), 0.1)} if cfg.get("architecture") == "jumprelu" else {}
    folder = _saelens(tmp_path / "sae", cfg, _weights(8, 16, **extra))
    _same(read_config("sae_weights.safetensors", cfg), read_sae(folder))
    # With the names and shapes its weights file's header lists, the same again.
    shapes = {k: tuple(v.shape) for k, v in _weights(8, 16, **extra).items()}
    _same(read_config("sae_weights.safetensors", cfg, tensors=shapes), read_sae(folder))


def test_gemma_scope_and_sparsify_configurations_alone_read_as_their_files_do(tmp_path):
    scope = {
        "hf_hook_point_in": "model.layers.2.pre_feedforward_layernorm.output",
        "hf_hook_point_out": "model.layers.2.post_feedforward_layernorm.output",
        "width": 16,
        "architecture": "jump_relu",
        "affine_connection": True,
        "type": "transcoder",
    }
    params = {
        "w_enc": torch.zeros(8, 16),
        "b_enc": torch.zeros(16),
        "w_dec": torch.zeros(16, 8),
        "b_dec": torch.zeros(8),
        "threshold": torch.zeros(16),
        "affine_skip_connection": torch.zeros(8, 8),
    }
    folder = tmp_path / "scope"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(scope), encoding="utf-8")
    save_file(params, str(folder / "params.safetensors"))
    dry = read_config("params.safetensors", scope)
    _same(dry, read_sae(folder))
    assert dry.d_sae == 16 and dry.transcoder and dry.W_skip is not None

    sparsify = {"k": 4, "d_in": 8, "num_latents": 32, "transcode": True, "skip_connection": True}
    tensors = {
        "encoder.weight": torch.zeros(32, 8),
        "encoder.bias": torch.zeros(32),
        "W_dec": torch.zeros(32, 8),
        "b_dec": torch.zeros(8),
        "W_skip": torch.zeros(8, 8),
    }
    eleuther = tmp_path / "layers.3.mlp"
    eleuther.mkdir()
    (eleuther / "cfg.json").write_text(json.dumps(sparsify), encoding="utf-8")
    save_file(tensors, str(eleuther / "sae.safetensors"))
    _same(
        read_config("sae.safetensors", sparsify, "layers.3.mlp"), read_sae(eleuther, "layers.3.mlp")
    )


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        ({**SAELENS, "normalize_activations": True}, "expected_average_only_in"),
        ({**SAELENS, "normalize_activations": "expected_average_only_in"}, "files don't hold"),
        ({**SAELENS, "normalize_activations": "constant_norm_rescale"}, "constant_norm_rescale"),
        ({**SAELENS, "architecture": "gated"}, "gated"),
        ({**SAELENS, "hook_name": "blocks.1.hook_resid_mid"}, "between attention and MLP"),
        ({**SAELENS, "finetuning_scaling_factor": True}, "fitted after training"),
        ({**SAELENS, "hook_head_index": 3}, "single attention head"),
    ],
)
def test_a_configuration_alone_is_refused_as_the_files_are(tmp_path, cfg, message):
    folder = _saelens(tmp_path / "sae", cfg, _weights(8, 16))
    with pytest.raises(BackendError, match=message) as from_files:
        read_sae(folder)
    with pytest.raises(BackendError) as from_config:
        read_config("sae_weights.safetensors", cfg)
    assert str(from_config.value) == str(from_files.value)


def test_a_scaling_factor_of_ones_changes_nothing_and_any_other_is_refused(tmp_path):
    """SAELens's earlier files hold a scaling_factor of ones that their configuration doesn't
    mention (GPT-2 small's res_*-ajt and feature-splitting SAEs among them); SAELens drops it."""
    from logogram.backends.saes import values_decide
    from logogram.sae import SAE

    ones = _saelens(tmp_path / "ones", SAELENS, _weights(8, 16, scaling_factor=torch.ones(16)))
    plain = _saelens(tmp_path / "plain", SAELENS, _weights(8, 16))
    with_ones = SAE(repo="local/ones", path="", revision="test", params=read_sae(ones))
    without = SAE(repo="local/plain", path="", revision="test", params=read_sae(plain))
    x = torch.randn(5, 8, generator=torch.Generator().manual_seed(2))
    f, stats = with_ones.encode(x)
    g, _ = without.encode(x)
    assert torch.equal(f, g) and torch.equal(with_ones.decode(f, stats), without.decode(g, None))
    other = _saelens(
        tmp_path / "other", SAELENS, _weights(8, 16, scaling_factor=torch.full((16,), 2.0))
    )
    with pytest.raises(BackendError, match="scaling_factor other than one"):
        read_sae(other)
    # From the header alone its values decide, so the catalogue reads them too.
    shapes = {k: tuple(v.shape) for k, v in _weights(8, 16, scaling_factor=torch.ones(16)).items()}
    with pytest.raises(BackendError) as unread:
        read_config("sae_weights.safetensors", SAELENS, tensors=shapes)
    assert values_decide(str(unread.value)) and "checks them when it loads" in str(unread.value)
    given = {**shapes, "scaling_factor": torch.ones(16)}
    _same(read_config("sae_weights.safetensors", SAELENS, tensors=given), read_sae(ones))
    with pytest.raises(BackendError, match="other than one"):
        read_config(
            "sae_weights.safetensors",
            SAELENS,
            tensors={**shapes, "scaling_factor": torch.zeros(16)},
        )


def test_files_logogram_cant_read_say_why():
    assert weights_file(["cfg.json", "sae_weights.safetensors"]) == "sae_weights.safetensors"
    assert weights_file(["config.json", "params.safetensors"]) == "params.safetensors"
    assert weights_file(["cfg.json", "sae.safetensors"]) == "sae.safetensors"
    assert unreadable_files(["config.json", "params.safetensors"]) is None
    assert "NumPy archive" in (unreadable_files(["params.npz"]) or "")
    assert "pickle (a .pt file)" in (unreadable_files(["config.json", "ae.pt"]) or "")
    assert "pickle (a .pth file)" in (unreadable_files(["Llama-3.1-8B-SAE-l19.pth"]) or "")
    assert "never opens" in (unreadable_files(["pytorch_model.bin"]) or "")
    assert "several layers" in (
        unreadable_files(["config.json", "params_layer_0.safetensors"]) or ""
    )
    assert "several layers" in (
        unreadable_files(["W_enc_0.safetensors", "W_dec_0.safetensors"]) or ""
    )
    llama_scope = unreadable_files(["hyperparams.json", "checkpoints/final.safetensors"])
    assert "Llama Scope" in (llama_scope or "")
    assert "OpenMOSS" in (unreadable_files(["config.json", "sae_weights.safetensors"]) or "")
    assert "temporal" in (unreadable_files(["conf.yaml", "latest_ckpt.safetensors"]) or "")
    assert "no SAE in a format" in (unreadable_files(["README.md"]) or "")


def test_loading_by_folder_says_why_its_files_cant_be_read(monkeypatch):
    from types import SimpleNamespace

    from logogram.backends.saes import download_sae

    files = ["pickled/ae.pt", "pickled/config.json", "npz/params.npz", "layer0.sae.pt"]
    info = SimpleNamespace(
        sha="a" * 40,
        gated=False,
        siblings=[SimpleNamespace(rfilename=n, size=10) for n in files],
    )
    monkeypatch.setattr("huggingface_hub.HfApi.model_info", lambda self, *a, **k: info)
    with pytest.raises(BackendError, match=r"pickle \(a \.pt file\)"):
        download_sae("someone/saes", "pickled", None)
    with pytest.raises(BackendError, match="NumPy archive"):
        download_sae("someone/saes", "npz", None)
    # A path that names one file.
    with pytest.raises(BackendError, match="pickle"):
        download_sae("someone/saes", "layer0.sae.pt", None)
    with pytest.raises(BackendError, match="has no SAE at elsewhere"):
        download_sae("someone/saes", "elsewhere", None)


# -- the script that writes the catalogue ------------------------------------------------------


def _script():
    spec = importlib.util.spec_from_file_location(
        "sae_catalogue", ROOT / "scripts" / "sae_catalogue.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["sae_catalogue"] = module
    spec.loader.exec_module(module)
    return module


REGISTRY = """
tiny-res:
  conversion_func: null
  model: gpt2-small
  repo_id: someone/tiny-res
  config_overrides:
    model_from_pretrained_kwargs:
      center_writing_weights: true
  saes:
  - id: blocks.1.hook_resid_pre
    path: blocks.1.hook_resid_pre
    l0: 12.5
  - id: blocks.2.hook_resid_mid
    path: blocks.2.hook_resid_mid
  - id: blocks.3.hook_resid_pre
    path: missing
  - id: blocks.4.hook_resid_pre
    path: blocks.4.hook_resid_pre
tiny-rescaled:
  conversion_func: null
  model: gpt2-small
  repo_id: someone/tiny-rescaled
  config_overrides:
    normalize_activations: constant_norm_rescale
  saes:
  - id: blocks.3.hook_resid_pre
    path: layer_3
tiny-scope:
  conversion_func: gemma_3
  model: google/gemma-3-270m
  repo_id: someone/tiny-scope
  saes:
  - id: layer_1_width_16k_l0_small_affine
    path: transcoder/layer_1_width_16k_l0_small_affine
tiny-scope-res-npz:
  conversion_func: gemma_2
  model: gemma-2-2b
  repo_id: someone/tiny-scope-res
  saes:
  - id: layer_4/width_16k/average_l0_10
    path: layer_4/width_16k/average_l0_10
tiny-clt:
  conversion_func: mntss_clt_layer_transcoder
  model: gemma-2-2b
  repo_id: someone/tiny-clt
  saes:
  - id: layer_0
    path: 0
"""
MIT = "MIT License\n\nCopyright (c) a maintainer\n\nPermission is hereby granted, free of charge.\n"


def _wheel(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr("sae_lens/pretrained_saes.yaml", REGISTRY)
        wheel.writestr(
            "sae_lens-9.9.9.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: sae-lens\nVersion: 9.9.9\nLicense-Expression: MIT\n",
        )
        wheel.writestr("sae_lens-9.9.9.dist-info/licenses/LICENSE", MIT)
    return path


class FakeHub:
    """Repositories and their small files, made up here: nothing goes to Hugging Face."""

    files: ClassVar[dict[str, dict[str, Any]]] = {}

    def __init__(self, cache: Path, refresh: bool) -> None:
        self.cache = cache

    def repo(self, repo_id: str) -> dict[str, Any]:
        names = self.files[repo_id]
        return {
            "id": repo_id,
            "sha": "b" * 40,
            "gated": False,
            "licence": "cc-by-4.0",
            "files": {n: 1 for n in names},
        }

    def file(self, repo_id: str, sha: str, name: str) -> Path | None:
        content = self.files[repo_id].get(name)
        if content is None:
            return None
        path = self.cache / repo_id.replace("/", "__") / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content), encoding="utf-8")
        return path

    def fetch_all(self, wanted: Any, workers: int = 8) -> None:
        pass

    def header(self, repo_id: str, sha: str, name: str) -> dict[str, Any]:
        """A weights file's header: SAELens's parameters, and a scaling_factor in some."""
        d, f = 8, 16
        shapes = {"W_enc": [d, f], "b_enc": [f], "W_dec": [f, d], "b_dec": [d]}
        if name in self.scaling:
            shapes["scaling_factor"] = [f]
        return {"size": 0, "tensors": {k: {"dtype": "F32", "shape": v} for k, v in shapes.items()}}

    def tensor(self, repo_id: str, sha: str, name: str, tensor: str) -> Any:
        return torch.full((16,), self.scaling[name])

    scaling: ClassVar[dict[str, float]] = {
        "blocks.1.hook_resid_pre/sae_weights.safetensors": 1.0,
        "blocks.4.hook_resid_pre/sae_weights.safetensors": 2.0,
    }

    def model_ids(self, model: str) -> list[str]:
        known = {"gpt2-small": ["openai-community/gpt2", "gpt2"]}
        return known.get(model, ["google/gemma-2-2b"] if "2b" in model else [model])


def test_the_script_builds_the_catalogue_from_a_sae_lens_wheel(tmp_path, monkeypatch, offline):
    script = _script()
    cfg = {
        "hook_name": "blocks.1.hook_resid_pre",
        "d_in": 8,
        "d_sae": 16,
        "architecture": "standard",
    }
    FakeHub.files = {
        "someone/tiny-res": {
            "blocks.1.hook_resid_pre/cfg.json": cfg,
            "blocks.1.hook_resid_pre/sae_weights.safetensors": None,
            "blocks.2.hook_resid_mid/cfg.json": {**cfg, "hook_name": "blocks.2.hook_resid_mid"},
            "blocks.2.hook_resid_mid/sae_weights.safetensors": None,
            "blocks.4.hook_resid_pre/cfg.json": {**cfg, "hook_name": "blocks.4.hook_resid_pre"},
            "blocks.4.hook_resid_pre/sae_weights.safetensors": None,
        },
        "someone/tiny-rescaled": {
            "layer_3/cfg.json": {**cfg, "hook_name": "blocks.3.hook_resid_pre"},
            "layer_3/sae_weights.safetensors": None,
        },
        "someone/tiny-scope": {
            "transcoder/layer_1_width_16k_l0_small_affine/config.json": {
                "hf_hook_point_in": "model.layers.1.pre_feedforward_layernorm.output",
                "hf_hook_point_out": "model.layers.1.post_feedforward_layernorm.output",
                "width": 16384,
                "architecture": "jump_relu",
                "affine_connection": True,
                "type": "transcoder",
            },
            "transcoder/layer_1_width_16k_l0_small_affine/params.safetensors": None,
        },
        "someone/tiny-scope-res": {"layer_4/width_16k/average_l0_10/params.npz": None},
        "someone/tiny-clt": {"W_enc_0.safetensors": None, "W_dec_0.safetensors": None},
    }
    monkeypatch.setattr(script, "Hub", FakeHub)
    out = tmp_path / "catalogue.json"
    args = SimpleArgs(sae_lens=str(_wheel(tmp_path / "sae_lens-9.9.9-py3-none-any.whl")))
    script.build(args.with_(cache=str(tmp_path / "cache"), out=str(out), refresh=False))
    built = parse_catalogue(json.loads(out.read_text(encoding="utf-8")))
    # Where it came from, and the licence its source asks to travel with it.
    assert (built.source.version, built.source.licence, built.source.notice) == (
        "9.9.9",
        "MIT",
        MIT,
    )
    entries = {(e.release, e.id): e for e in built.entries}
    exact = entries[("tiny-res", "blocks.1.hook_resid_pre")]
    assert (exact.reads, exact.site, exact.layer, exact.width, exact.l0) == (
        "exact",
        "resid_pre",
        1,
        16,
        12.5,
    )
    assert exact.hf_models == ("openai-community/gpt2", "gpt2") and exact.licence == "cc-by-4.0"
    mid = entries[("tiny-res", "blocks.2.hook_resid_mid")]
    assert mid.reads == "refused" and "between attention and MLP" in (mid.reason or "")
    # Its file's header was read, and the values of the scaling_factor it holds: ones change
    # nothing, other values are refused.
    assert exact.verified is not None and exact.verified.read == "header"
    scaled = entries[("tiny-res", "blocks.4.hook_resid_pre")]
    assert scaled.reads == "refused" and "scaling_factor other than one" in (scaled.reason or "")
    missing = entries[("tiny-res", "blocks.3.hook_resid_pre")]
    assert "isn't in someone/tiny-res" in (missing.reason or "")
    # The registry's own setting, which the configuration doesn't state, refuses it.
    rescaled = entries[("tiny-rescaled", "blocks.3.hook_resid_pre")]
    assert rescaled.registry == {"normalize_activations": "constant_norm_rescale"}
    assert "constant_norm_rescale" in (rescaled.reason or "")
    scope = entries[("tiny-scope", "layer_1_width_16k_l0_small_affine")]
    assert (scope.reads, scope.kind, scope.site, scope.layer, scope.width) == (
        "exact",
        "skip_transcoder",
        "mlp_out",
        1,
        16384,
    )
    assert scope.hook == "model.layers.1.pre_feedforward_layernorm.output"
    npz = entries[("tiny-scope-res-npz", "layer_4/width_16k/average_l0_10")]
    assert "NumPy archive" in (npz.reason or "")
    assert (npz.hook, npz.layer, npz.width) == ("blocks.4.hook_resid_post", 4, 16384)
    clt = entries[("tiny-clt", "layer_0")]
    assert clt.folder == "0" and "several layers" in (clt.reason or "")

    # A check on the files survives a rebuild at the same commit.
    data = json.loads(out.read_text(encoding="utf-8"))
    data["releases"][0]["saes"][0]["verified"] = {
        "revision": "b" * 40,
        "read": "header",
        "fit": None,
    }
    script.write_catalogue(out, data)
    script.build(args.with_(cache=str(tmp_path / "cache"), out=str(out), refresh=False))
    again = parse_catalogue(json.loads(out.read_text(encoding="utf-8")))
    kept = {(e.release, e.id): e.verified for e in again.entries}
    assert kept[("tiny-res", "blocks.1.hook_resid_pre")] is not None


def test_the_script_reads_the_registry_from_an_installation(tmp_path):
    script = _script()
    site = tmp_path / "site-packages"
    with zipfile.ZipFile(_wheel(tmp_path / "w.whl")) as wheel:
        wheel.extractall(site)
    from_folder = script.read_registry(site)
    from_wheel = script.read_registry(tmp_path / "w.whl")
    assert (from_folder.version, from_folder.notice, from_folder.text) == (
        from_wheel.version,
        from_wheel.notice,
        from_wheel.text,
    )
    assert set(from_folder.releases) == {
        "tiny-res",
        "tiny-rescaled",
        "tiny-scope",
        "tiny-scope-res-npz",
        "tiny-clt",
    }
    bare = tmp_path / "bare"
    (bare / "sae_lens").mkdir(parents=True)
    (bare / "sae_lens" / "pretrained_saes.yaml").write_text(REGISTRY, encoding="utf-8")
    with pytest.raises(SystemExit, match="licence"):
        script.read_registry(bare)


class SimpleArgs:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)

    def with_(self, **values: Any) -> SimpleArgs:
        return SimpleArgs(**{**self.__dict__, **values})
