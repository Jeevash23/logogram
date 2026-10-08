"""Loading a model: what the checks refuse, what running out of memory leaves behind, the
beginning-of-sequence token, and the memory estimate.

These regressions come from loading real models (GPT-2, Pythia, Qwen 2.5): Pythia in float16
overflows in its own forward pass, Qwen 2.5 has no beginning-of-sequence token, and processing
the weights needs several times their size for a moment. Tiny local models reproduce each case.
"""

from __future__ import annotations

import gc
import json
import math
import weakref

import pytest
import torch

from conftest import build_tiny_architecture
from logogram.backends import hub
from logogram.backends.base import BackendError
from logogram.backends.transformer_lens import (
    ModelChecks,
    TransformerLensBackend,
    _relative,
    boot_local,
    load_model,
)
from logogram.system import estimate_memory


def _from(bridge, process_weights=True, dtype="float32"):
    return TransformerLensBackend.from_bridge(
        bridge, model_id="tiny", revision="test", dtype=dtype, process_weights=process_weights
    )


def test_values_that_arent_finite_never_pass_a_check():
    ones = torch.ones(3)
    assert _relative(torch.tensor([1.0, float("nan"), 1.0]), ones) == math.inf
    assert _relative(torch.tensor([1.0, float("inf"), 1.0]), ones) == math.inf
    assert _relative(ones, torch.tensor([1.0, float("inf"), 1.0])) == math.inf
    # A NaN would compare False with every tolerance, and max() would drop it after a number.
    assert max(0.0, _relative(torch.tensor([float("nan")]), torch.ones(1))) == math.inf
    checks = ModelChecks(
        tolerance=1e-4, function=0.0, structure="components", residual=None, lens=math.inf
    )
    payload = checks.to_dict()
    assert payload["lens"] is None
    json.dumps(payload, allow_nan=False)  # the API refuses NaN and infinity


def test_a_model_whose_own_output_overflows_is_refused(arch_backend):
    bridge = boot_local(arch_backend.folder("gpt_neox"), device="cpu", dtype="float32")
    model = bridge.original_model
    forward = model.forward

    def overflowing(*args, **kwargs):
        out = forward(*args, **kwargs)
        out.logits[..., 3] = float("inf")
        return out

    model.forward = overflowing
    with pytest.raises(BackendError, match="own forward pass gives values that aren't finite"):
        _from(bridge)


@pytest.mark.parametrize("process_weights", [True, False])
def test_a_version_that_overflows_is_refused(arch_backend, process_weights):
    bridge = boot_local(arch_backend.folder("llama"), device="cpu", dtype="float32")
    original = bridge.enable_compatibility_mode

    def compatibility_mode_that_overflows(*args, **kwargs):
        original(*args, **kwargs)
        with torch.no_grad():
            bridge.original_model.get_output_embeddings().weight[3] = float("inf")

    bridge.enable_compatibility_mode = compatibility_mode_that_overflows
    what = "Processing the weights of" if process_weights else "TransformerLens's version of"
    with pytest.raises(BackendError, match=f"{what} tiny gives values that aren't finite"):
        _from(bridge, process_weights=process_weights)


def _local_hub(monkeypatch, folder):
    files = hub.RepoFiles(revision="test", files=[], n_params=None, gated=False)
    monkeypatch.setattr(hub, "resolve", lambda model_id, revision: files)
    monkeypatch.setattr(hub, "download", lambda model_id, repo, progress, cancel=None: folder)


@pytest.mark.parametrize(
    ("process_weights", "message"),
    [(True, "while its weights are processed"), (False, "doesn't fit in memory")],
)
def test_running_out_of_memory_says_what_to_do_and_frees_the_model(
    monkeypatch, tiny_model_dir, process_weights, message
):
    _local_hub(monkeypatch, tiny_model_dir)
    booted = []

    def from_bridge_out_of_memory(bridge, **kwargs):
        booted.append(weakref.ref(bridge))
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 18.00 MiB.")

    monkeypatch.setattr(TransformerLensBackend, "from_bridge", from_bridge_out_of_memory)
    with pytest.raises(BackendError, match=message) as caught:
        load_model("tiny", device="cpu", process_weights=process_weights)
    # The error doesn't keep the failed attempt's frames, so its model is freed.
    assert caught.value.__context__ is None and caught.value.__cause__ is None
    gc.collect()
    assert booted and booted[0]() is None


def test_other_load_errors_pass_through(monkeypatch, tiny_model_dir):
    _local_hub(monkeypatch, tiny_model_dir)

    def refuse(bridge, **kwargs):
        raise BackendError("Processing the weights of tiny changed its predictions.")

    monkeypatch.setattr(TransformerLensBackend, "from_bridge", refuse)
    with pytest.raises(BackendError, match="changed its predictions"):
        load_model("tiny", device="cpu")


def test_a_model_without_a_beginning_of_sequence_token_has_none(tmp_path):
    # Like Qwen 2.5: TransformerLens gives such a tokenizer its end-of-text token as a substitute,
    # which the model never saw at the start of a sequence.
    build_tiny_architecture(tmp_path, "qwen2")
    config = tmp_path / "tokenizer_config.json"
    data = json.loads(config.read_text(encoding="utf-8"))
    data["bos_token"] = None
    config.write_text(json.dumps(data), encoding="utf-8")
    backend = _from(boot_local(tmp_path, device="cpu", dtype="float32"))
    assert backend.info.extra["bos"] is False
    assert backend.bos_token_id is None
    assert backend.tokenize("When Mary and John went", False).ids
    with pytest.raises(BackendError, match="no beginning-of-sequence token"):
        backend.tokenize("When Mary and John went", True)


def test_a_model_with_a_beginning_of_sequence_token_keeps_it(tiny_backend):
    bos = tiny_backend.tokenizer.convert_tokens_to_ids("<|endoftext|>")
    assert tiny_backend.info.extra["bos"] is True
    assert tiny_backend.tokenize("Hello world", True).ids[0] == bos


def _fake_cache(root, repo_id, commits, ref=None, files=("config.json", "model.safetensors")):
    """A Hugging Face cache with ``commits`` downloaded, as Logogram downloads them: by commit,
    which records no branch (``ref`` adds one, as a download of the main branch would)."""
    folder = root / ("models--" + repo_id.replace("/", "--"))
    for commit in commits:
        for name in files:
            path = folder / "snapshots" / commit / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
    if ref is not None:
        (folder / "refs").mkdir(parents=True, exist_ok=True)
        (folder / "refs" / "main").write_text(ref, encoding="utf-8")


def test_a_model_downloaded_at_a_commit_opens_offline(tmp_path, monkeypatch):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    first, second = "a" * 40, "b" * 40
    _fake_cache(tmp_path, "org/model", [first])
    assert hub.resolve("org/model", None).revision == first
    _fake_cache(tmp_path, "org/model", [second])
    with pytest.raises(BackendError, match="several revisions of org/model"):
        hub.resolve("org/model", None)
    assert hub.resolve("org/model", second).revision == second
    _fake_cache(tmp_path, "org/model", [], ref=second)
    assert hub.resolve("org/model", None).revision == second  # a cached main branch decides


def test_offline_the_model_size_comes_from_its_files(tmp_path, monkeypatch, tiny_model_dir):
    # Not from the file size: GPT-2 is stored in float32, so halving its bytes doubles its size.
    import shutil

    from huggingface_hub import constants
    from safetensors.torch import load_file

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    _fake_cache(tmp_path, "org/model", ["d" * 40], files=("config.json",))
    snapshot = tmp_path / "models--org--model" / "snapshots" / ("d" * 40)
    shutil.copy(tiny_model_dir / "model.safetensors", snapshot)
    stored = sum(t.numel() for t in load_file(tiny_model_dir / "model.safetensors").values())
    assert hub.resolve("org/model", None).n_params == stored


def test_an_sae_downloaded_at_a_commit_opens_offline(tmp_path, monkeypatch):
    from huggingface_hub import constants

    from logogram.backends.saes import download_sae

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    commit = "c" * 40
    files = ("layers.3/cfg.json", "layers.3/sae.safetensors")
    _fake_cache(tmp_path, "org/saes", [commit], files=files)
    found, folder = download_sae("org/saes", "layers.3", None)
    assert found == commit and folder.name == "layers.3"


QWEN_05B = {
    "n_params": 494_032_768,
    "n_layers": 24,
    "n_heads": 14,
    "d_model": 896,
    "d_mlp": 4864,
    "d_vocab": 151_936,
}
GPT2_SMALL = {
    "n_params": 124_439_808,
    "n_layers": 12,
    "n_heads": 12,
    "d_model": 768,
    "d_mlp": 3072,
    "d_vocab": 50_257,
}


def test_the_memory_estimate_counts_weight_processing(monkeypatch):
    # Measured on a GPU with 6 GB: Qwen 2.5 0.5B loads only without weight processing, in any
    # dtype; GPT-2 small loads with it.
    monkeypatch.setattr("logogram.system.available_memory", lambda device: 6 * 1024**3)
    plain = estimate_memory(**QWEN_05B, dtype="float32", device="cuda", process_weights=False)
    assert plain.processing == 0 and plain.verdict == "fits"
    processed = estimate_memory(**QWEN_05B, dtype="float32", device="cuda", process_weights=True)
    assert processed.processing == 3 * 4 * QWEN_05B["n_params"]
    assert processed.verdict == "wont_fit"
    assert "weight processing off" in processed.explanation
    half = estimate_memory(**QWEN_05B, dtype="bfloat16", device="cuda", process_weights=True)
    assert half.processing == 4 * 4 * QWEN_05B["n_params"] and half.verdict == "wont_fit"
    gpt2 = estimate_memory(**GPT2_SMALL, dtype="float32", device="cuda", process_weights=True)
    assert gpt2.verdict == "fits"
    assert gpt2.total >= gpt2.weights + gpt2.processing + gpt2.margin
