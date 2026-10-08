"""Shared fixtures. Tests build a tiny random GPT-2 locally and never touch the network."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

import pytest

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_original_connect = socket.socket.connect


def _guarded_connect(self: socket.socket, address: Any) -> Any:
    if isinstance(address, tuple) and address[0] in ("127.0.0.1", "localhost", "::1"):
        return _original_connect(self, address)
    if isinstance(address, (str, bytes)):  # unix sockets
        return _original_connect(self, address)
    raise RuntimeError(f"Tests must not use the network (tried to connect to {address!r}).")


socket.socket.connect = _guarded_connect  # type: ignore[method-assign]


def _build_tiny_model(folder: Path) -> None:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from logogram.ioi import NAMES, OBJECTS, PLACES, TEMPLATES

    words: set[str] = {",", ".", "Hello", "world"}
    for template in TEMPLATES:
        for word in template.text.replace(",", " , ").split():
            if not word.startswith("{"):
                words.add(word)
    words |= set(NAMES) | set(PLACES) | set(OBJECTS)
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False)
    vocab = {"<|endoftext|>": 0, "<unk>": 1}
    for word in sorted(words):
        for form in (word, " " + word):
            for piece, _ in byte_level.pre_tokenize_str(form):
                vocab.setdefault(piece, len(vocab))
    tok_model = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok_model.pre_tokenizer = byte_level
    tok_model.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok_model,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<unk>",
        pad_token="<|endoftext|>",
    )
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=4,
        initializer_range=0.3,
        bos_token_id=0,
        eos_token_id=0,
    )
    model = GPT2LMHeadModel(config).eval()
    tokenizer.save_pretrained(folder)
    model.save_pretrained(folder)


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("tiny-gpt2")
    _build_tiny_model(folder)
    return folder


@pytest.fixture(scope="session")
def tiny_backend(tiny_model_dir: Path) -> Any:
    from logogram.backends.transformer_lens import TransformerLensBackend, boot_local
    from logogram.runner import configure_determinism

    configure_determinism()
    bridge = boot_local(tiny_model_dir, device="cpu", dtype="float32")
    return TransformerLensBackend.from_bridge(
        bridge, model_id="tiny-gpt2", revision="test", dtype="float32", process_weights=True
    )


@pytest.fixture
def project(tmp_path: Path) -> Any:
    from logogram.datasets import write_dataset
    from logogram.ioi import generate_ioi
    from logogram.project import Project

    proj = Project.create(tmp_path, "Test project")
    write_dataset(proj.datasets_dir / "ioi.jsonl", generate_ioi(12, seed=0))
    write_dataset(
        proj.datasets_dir / "mixed.jsonl",
        generate_ioi(12, seed=1, templates=["went", "working", "reached"]),
    )
    return proj


def make_spec(**overrides: Any) -> Any:
    from logogram.spec import Spec

    data: dict[str, Any] = {
        "name": "test",
        "model": {"id": "tiny-gpt2", "revision": "test", "device": "cpu"},
        "dataset": {"path": "datasets/ioi.jsonl"},
        "experiment": {"kind": "activation_patching", "direction": "clean_to_corrupt"},
        "scope": {"kind": "heads", "position": {"kind": "all"}},
        "execution": {"batch_size": 16},
        "statistics": {"bootstrap": 200, "seed": 0},
    }
    data.update(overrides)
    return Spec.model_validate(data)


@pytest.fixture
def spec_factory() -> Any:
    return make_spec
