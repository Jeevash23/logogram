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


# Tiny random models of other architecture families, built locally like the GPT-2 one. Shapes are
# deliberately not square: TransformerLens 4.0 centers a square unembedding (d_vocab == d_model)
# along the wrong axis, which no real model has.
TINY_ARCHITECTURES: dict[str, dict[str, Any]] = {
    # Llama family (also Mistral, SmolLM): RMSNorm, rotary, gated MLP, grouped-query attention.
    "llama": {
        "model_type": "llama",
        "num_key_value_heads": 2,
    },
    # Pythia: attention and MLP read the same residual stream (parallel blocks).
    "gpt_neox": {
        "model_type": "gpt_neox",
        "use_parallel_residual": True,
        "rotary_pct": 0.25,
    },
    # Qwen 2: biases on the query, key and value projections.
    "qwen2": {"model_type": "qwen2", "num_key_value_heads": 2},
    # Gemma 2: normalization before and after each sublayer, and soft-capped logits.
    "gemma2": {
        "model_type": "gemma2",
        "num_key_value_heads": 2,
        "head_dim": 12,
        "query_pre_attn_scalar": 12,
        "sliding_window": 16,
    },
    # OLMo 2: normalization after each sublayer only.
    "olmo2": {"model_type": "olmo2", "num_key_value_heads": 2},
}


def _tiny_bpe_tokenizer() -> Any:
    """A byte-level BPE trained on the IOI vocabulary, so every name is one token. Unlike a
    word-level vocabulary, every model family's tokenizer class can rebuild it from its files."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    from logogram.ioi import NAMES, OBJECTS, PLACES, TEMPLATES

    words: set[str] = {",", ".", "Hello", "world"}
    for template in TEMPLATES:
        for word in template.text.replace(",", " , ").split():
            if not word.startswith("{"):
                words.add(word)
    words |= set(NAMES) | set(PLACES) | set(OBJECTS)
    corpus = [form for word in sorted(words) for form in (word, " " + word)]
    tok_model = Tokenizer(models.BPE())
    tok_model.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok_model.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=4000,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok_model.train_from_iterator(corpus, trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tok_model,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )


def build_tiny_architecture(folder: Path, arch: str, initializer_range: float = 0.3) -> None:
    """A tiny random model of ``arch``. The large default weights give clear effects; smaller ones
    give a smoother function, for checks that need small changes to behave linearly."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    tokenizer = _tiny_bpe_tokenizer()
    options = dict(TINY_ARCHITECTURES[arch])
    model_type = options.pop("model_type")
    config = AutoConfig.for_model(
        model_type,
        vocab_size=len(tokenizer),
        hidden_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=96,
        max_position_embeddings=64,
        initializer_range=initializer_range,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        **options,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config).eval()
    tokenizer.save_pretrained(folder)
    model.save_pretrained(folder)


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


@pytest.fixture(scope="session")
def arch_backend(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """``arch_backend(name, process_weights=True)``: a backend for a tiny model of that family."""
    from logogram.backends.transformer_lens import TransformerLensBackend, boot_local
    from logogram.runner import configure_determinism

    folders: dict[str, Path] = {}
    backends: dict[tuple[str, bool], Any] = {}

    def get(arch: str, process_weights: bool = True) -> Any:
        if arch not in folders:
            folders[arch] = tmp_path_factory.mktemp(f"tiny-{arch}")
            build_tiny_architecture(folders[arch], arch)
        key = (arch, process_weights)
        if key not in backends:
            configure_determinism()
            bridge = boot_local(folders[arch], device="cpu", dtype="float32")
            backends[key] = TransformerLensBackend.from_bridge(
                bridge,
                model_id=f"tiny-{arch}",
                revision="test",
                dtype="float32",
                process_weights=process_weights,
            )
        return backends[key]

    get.folder = lambda arch: (get(arch), folders[arch])[1]  # type: ignore[attr-defined]
    return get


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
    """A complete version 2 spec. Overrides replace whole sections; the fields an override leaves
    out take version 1's values (as a convenience for tests only: real specs state them)."""
    from logogram.spec import Spec, upgrade_v1

    data: dict[str, Any] = {
        "logogram_spec": 1,
        "name": "test",
        "model": {"id": "tiny-gpt2", "revision": "test", "device": "cpu"},
        "dataset": {"path": "datasets/ioi.jsonl"},
        "experiment": {"kind": "activation_patching", "direction": "clean_to_corrupt"},
        "scope": {"kind": "heads", "position": {"kind": "all"}},
        "execution": {"batch_size": 16},
        "statistics": {"bootstrap": 200, "seed": 0},
    }
    data.update(overrides)
    data, _ = upgrade_v1(data)
    return Spec.model_validate(data)


def spec_dict(**overrides: Any) -> dict[str, Any]:
    """``make_spec`` as JSON, the way the app sends it."""
    return make_spec(**overrides).model_dump(mode="json")


@pytest.fixture
def spec_factory() -> Any:
    return make_spec
