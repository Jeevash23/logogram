"""Choosing one GPU of several: the system report lists each GPU, a spec's cuda:N is checked against
them, memory estimates use that GPU's free memory, the manifest names the GPU a run used, and
"cuda" keeps meaning the default GPU. Tests have no GPU: torch.cuda's functions are replaced."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from logogram.backends.base import BackendError
from logogram.backends.transformer_lens import _device_name, resolve_device
from logogram.runner import device_provenance, model_matches
from logogram.schema import Manifest
from logogram.spec import ModelRef
from logogram.system import GB, available_memory, estimate_memory, system_report


@pytest.fixture
def gpus(monkeypatch):
    """Two GPUs, the first the default: their names, memory and memory free."""
    props = [
        SimpleNamespace(name="Test GPU A", total_memory=24 * GB),
        SimpleNamespace(name="Test GPU B", total_memory=48 * GB),
    ]
    free = [20 * GB, 40 * GB]
    machine = SimpleNamespace(current=0, count=2)

    def index(device=None):
        if device is None:
            return machine.current
        if isinstance(device, torch.device):
            return machine.current if device.index is None else device.index
        if isinstance(device, str):
            return index(torch.device(device))
        return int(device)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: machine.count)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: machine.current)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: props[index(d)])
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda d=None: props[index(d)].name)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda d=None: (free[index(d)], props[index(d)].total_memory)
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    return machine


def test_the_system_report_lists_each_gpu(gpus):
    report = system_report()
    assert report.backend == "cuda" and report.default_gpu == 0
    assert [(g.index, g.name, g.memory_total, g.memory_free) for g in report.gpus] == [
        (0, "Test GPU A", 24 * GB, 20 * GB),
        (1, "Test GPU B", 48 * GB, 40 * GB),
    ]
    assert report.to_dict()["gpus"][1]["index"] == 1


def test_doctor_names_each_gpu_by_its_device(gpus):
    from typer.testing import CliRunner

    from logogram.cli import app

    out = CliRunner().invoke(app, ["doctor"]).output
    assert "cuda:0 (default) · Test GPU A" in out and "cuda:1 · Test GPU B" in out


def test_a_gpu_is_checked_against_the_machines(gpus, monkeypatch):
    assert resolve_device("cuda:1") == "cuda:1"
    # "cuda" (and "auto") is the default GPU, by its index.
    assert resolve_device("cuda") == resolve_device("auto") == "cuda:0"
    gpus.current = 1
    assert resolve_device("cuda") == "cuda:1"
    with pytest.raises(BackendError, match=r"GPU 2 \(cuda:2\).*2 GPUs, cuda:0 to cuda:1"):
        resolve_device("cuda:2")
    gpus.count = 1
    with pytest.raises(BackendError, match=r"one GPU, cuda:0.*or to cuda for the default GPU"):
        resolve_device("cuda:1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # The message the README's troubleshooting quotes, for one GPU of several too.
    with pytest.raises(BackendError, match=r"asks for CUDA, but PyTorch can't see a CUDA GPU"):
        resolve_device("cuda:1")
    assert resolve_device("auto") == "cpu"


def test_cuda_means_the_default_gpu_when_models_are_matched(gpus, tiny_backend):
    on = lambda device: SimpleNamespace(info=dataclasses.replace(tiny_backend.info, device=device))  # noqa: E731
    ref = ModelRef(
        id="tiny-gpt2",
        source="hub",
        revision="test",
        sha256=None,
        dtype="float32",
        device="cuda",
        process_weights=True,
    )

    def asks(device: str) -> ModelRef:
        return ref.model_copy(update={"device": device})

    assert model_matches(on("cuda:0"), asks("cuda"))
    assert model_matches(on("cuda:0"), asks("cuda:0"))
    assert model_matches(on("cuda:0"), asks("auto"))
    assert not model_matches(on("cuda:0"), asks("cuda:1"))
    assert model_matches(on("cuda:1"), asks("cuda:1"))
    assert not model_matches(on("cuda:1"), asks("cuda"))
    gpus.current = 1  # another default GPU: "cuda" now means GPU 1
    assert model_matches(on("cuda:1"), asks("cuda"))
    assert not model_matches(on("cuda:0"), asks("cuda"))
    assert not model_matches(on("cuda:1"), asks("cuda:5"))  # a GPU this machine lacks


def test_memory_estimates_use_the_chosen_gpus_free_memory(gpus):
    assert available_memory("cuda:1") == 40 * GB
    assert available_memory("cuda:0") == available_memory("cuda") == 20 * GB
    shape = dict(n_params=10**9, n_layers=4, n_heads=4, d_model=64, d_mlp=256, d_vocab=1000)
    one = estimate_memory(**shape, dtype="float32", device="cuda:1")
    zero = estimate_memory(**shape, dtype="float32", device="cuda:0")
    assert one.available == 40 * GB and zero.available == 20 * GB
    assert one.total == zero.total and "memory on GPU 1" in one.explanation
    # 20 GB of float32 weights fit in GPU 1's 40 GB free, not in GPU 0's 20 GB.
    big = dict(shape, n_params=5 * 10**9)
    assert estimate_memory(**big, dtype="float32", device="cuda:1").verdict == "fits"
    assert estimate_memory(**big, dtype="float32", device="cuda:0").verdict == "wont_fit"


def test_the_manifest_names_the_gpu_a_run_used(gpus, tiny_backend):
    assert _device_name("cuda:1") == "Test GPU B" and _device_name("cuda:0") == "Test GPU A"
    info = dataclasses.replace(tiny_backend.info, device="cuda:1", device_name="Test GPU B")
    record = device_provenance(info)
    assert (record["type"], record["index"], record["name"]) == ("cuda", 1, "Test GPU B")
    assert record["cuda"] == torch.version.cuda
    cpu = device_provenance(tiny_backend.info)
    assert cpu["type"] == "cpu" and "index" not in cpu and cpu["cuda"] is None
    manifest = Manifest.model_validate(
        {"run_id": "r", "status": "running", "started_at": "now", "versions": {}, "device": record}
    )
    assert manifest.device is not None and manifest.device.index == 1


def test_the_app_estimates_on_the_chosen_gpu(gpus, tmp_path, monkeypatch, tiny_model_dir):
    from fastapi.testclient import TestClient

    from logogram.local_models import add_model
    from logogram.server.app import create_app
    from logogram.server.security import SecurityConfig

    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    model = add_model(tiny_model_dir, "tiny")
    app = create_app(SecurityConfig(token="gpu-token", port=8765), serve_web=False)
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    headers = {"authorization": "Bearer gpu-token", "origin": "http://127.0.0.1:8765"}
    body = {"id": "tiny", "source": "local", "sha256": model.sha256}
    for device, free in (("cuda:1", 40 * GB), ("cuda:0", 20 * GB), ("cuda", 20 * GB)):
        out = client.post("/api/models/estimate", json={**body, "device": device}, headers=headers)
        assert out.status_code == 200, out.json()
        assert out.json()["estimate"]["available"] == free
    wrong = client.post("/api/models/estimate", json={**body, "device": "cuda:7"}, headers=headers)
    assert wrong.status_code == 400 and "cuda:0 to cuda:1" in wrong.json()["error"]
    system = client.get("/api/system", headers=headers).json()
    assert [g["index"] for g in system["gpus"]] == [0, 1] and system["default_gpu"] == 0
