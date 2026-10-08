"""Regression checks for scientific context, shared projects, caches and release workflows."""

from __future__ import annotations

import csv
import io
import json
import shutil
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from fastapi.testclient import TestClient

from logogram.backends.base import BackendError, Cancelled
from logogram.backends.hub import resolve, validate_local_weights
from logogram.backends.transformer_lens import boot_local
from logogram.datasets import file_sha256
from logogram.project import Project, ProjectError
from logogram.runner import configure_determinism, run_spec, write_manifest
from logogram.server.app import create_app
from logogram.server.security import SecurityConfig
from logogram.server.state import Job, request_project
from logogram.spec import Spec

BASE = "http://127.0.0.1:8765"
AUTH = {"authorization": "Bearer test", "origin": BASE}
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")


@pytest.fixture
def ready(project, tiny_backend, tmp_path, monkeypatch):
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    app = create_app(SecurityConfig(token="test", port=8765), serve_web=False)
    state = app.state.logogram
    state.open_project(project)
    state.backend = tiny_backend
    state.model_status = {"state": "ready"}
    with TestClient(app, base_url=BASE, headers=AUTH) as client:
        yield state, client


def test_offline_cache_refuses_pickle_weights(tmp_path, monkeypatch):
    from huggingface_hub.errors import OfflineModeIsEnabled

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "pytorch_model.bin").write_bytes(b"not a model")
    monkeypatch.setattr(
        "huggingface_hub.HfApi.model_info", Mock(side_effect=OfflineModeIsEnabled())
    )
    monkeypatch.setattr("logogram.backends.hub.cached_snapshot", lambda *args: ("pinned", tmp_path))
    with pytest.raises(BackendError, match="safetensors"):
        resolve("test/model", None)
    loader = Mock(side_effect=AssertionError("Pickle must never be opened"))
    monkeypatch.setattr(torch, "load", loader)
    with pytest.raises(BackendError, match="safetensors"):
        boot_local(tmp_path, device="cpu", dtype="float32")
    loader.assert_not_called()


@pytest.mark.parametrize(
    "shard",
    [
        "pytorch_model.bin",
        "../outside.safetensors",
        "folder/shard.safetensors",
        "missing.safetensors",
    ],
)
def test_safetensors_index_never_selects_unsafe_or_missing_shards(tmp_path, shard):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": shard}})
    )
    with pytest.raises(BackendError, match="index"):
        validate_local_weights(tmp_path)


def test_adapter_config_cannot_bypass_the_safetensors_loader(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"placeholder")
    (tmp_path / "adapter_config.json").write_text('{"base_model_name_or_path":"other/model"}')
    with pytest.raises(BackendError, match="Adapter checkpoints"):
        validate_local_weights(tmp_path)


def test_mixed_cache_explicitly_loads_safetensors(tiny_model_dir, tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM

    shutil.copytree(tiny_model_dir, tmp_path / "model")
    (tmp_path / "model" / "pytorch_model.bin").write_bytes(b"must not be read")
    original = AutoModelForCausalLM.from_pretrained
    loader = Mock(wraps=original)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", loader)
    pickle = Mock(side_effect=AssertionError("Pickle must never be opened"))
    monkeypatch.setattr(torch, "load", pickle)
    boot_local(tmp_path / "model", device="cpu", dtype="float32")
    assert loader.call_args.kwargs["use_safetensors"] is True
    assert loader.call_args.kwargs["local_files_only"] is True
    assert loader.call_args.kwargs["trust_remote_code"] is False
    pickle.assert_not_called()


def test_first_download_cancel_and_retry_reuses_complete_files(tmp_path, monkeypatch):
    from logogram.backends.hub import download

    info = SimpleNamespace(
        sha="pinned",
        safetensors=None,
        gated=False,
        siblings=[
            SimpleNamespace(rfilename="config.json", size=2),
            SimpleNamespace(rfilename="model.safetensors", size=4),
            SimpleNamespace(rfilename="pytorch_model.bin", size=4),
        ],
    )
    monkeypatch.setattr("huggingface_hub.HfApi.model_info", lambda *args, **kwargs: info)
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda model, name, **kwargs: str(tmp_path / name) if (tmp_path / name).is_file() else None,
    )
    cancel = threading.Event()
    calls = []
    interrupted = False

    def fetch(model, name, *, revision, **kwargs):
        nonlocal interrupted
        assert revision == "pinned"
        calls.append(name)
        if name == "model.safetensors" and not interrupted:
            interrupted = True
            cancel.set()
            raise Cancelled()
        (tmp_path / name).write_bytes(b"{}" if name == "config.json" else b"safe")
        return str(tmp_path / name)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fetch)
    repo = resolve("test/model", None)
    with pytest.raises(Cancelled):
        download("test/model", repo, cancel=cancel)
    assert (tmp_path / "config.json").is_file()
    cancel.clear()
    progress = []
    assert (
        download(
            "test/model",
            repo,
            lambda done, total, name: progress.append((done, total)),
            cancel=cancel,
        )
        == tmp_path
    )
    assert calls == ["config.json", "model.safetensors", "model.safetensors"]
    assert progress[-1] == (6, 6)


@posix_only
def test_saved_runs_never_read_external_links(ready, project, tmp_path, spec_factory):
    _, client = ready
    outside = tmp_path / "outside"
    outside.mkdir()
    spec = spec_factory(notes="external contents must stay unread").model_dump(mode="json")
    (outside / "spec.json").write_text(json.dumps(spec))
    (project.experiments_dir / "linked").symlink_to(outside, target_is_directory=True)
    assert client.get("/api/runs/linked").status_code == 400
    assert client.get("/api/runs").json() == []
    for name in ("spec", "summary", "manifest"):
        folder = project.prepare_run_dir(name)
        (folder / "spec.json").write_text(spec_factory().model_dump_json())
        (folder / f"{name}.json").unlink(missing_ok=True)
        (folder / f"{name}.json").symlink_to(outside / "spec.json")
        response = client.get(f"/api/runs/{name}")
        assert response.status_code == 400
        assert "external contents" not in response.text


@posix_only
def test_project_metadata_and_result_tables_stay_inside(
    project, tiny_backend, spec_factory, tmp_path, ready
):
    _, client = ready
    result = run_spec(spec_factory(), project, backend=tiny_backend)
    table = result.folder / "results.parquet"
    outside = tmp_path / "outside.parquet"
    table.replace(outside)
    table.symlink_to(outside)
    assert client.get(f"/api/runs/{result.run_id}/sites/0").status_code == 400
    assert client.get(f"/api/runs/{result.run_id}/export.csv").status_code == 409
    metadata = project.root / "project.json"
    external = tmp_path / "outside.json"
    metadata.replace(external)
    metadata.symlink_to(external)
    with pytest.raises(ProjectError, match="outside the project"):
        Project.open(project.root)


@pytest.mark.parametrize(
    "name,payload",
    [("manifest", '{"status":"surprise"}'), ("manifest", "{"), ("summary", '{"layout":42}')],
)
def test_one_bad_run_does_not_break_history(ready, project, spec_factory, name, payload):
    state, client = ready
    healthy = state.save_draft(spec_factory(name="Healthy"))
    bad = state.save_draft(spec_factory(name="Broken"))
    (project.run_dir(bad) / f"{name}.json").write_text(payload)
    response = client.get("/api/runs")
    assert response.status_code == 200, response.text
    rows = {row["id"]: row for row in response.json()}
    assert rows[healthy]["status"] == "draft"
    assert rows[bad]["status"] == "failed" and rows[bad]["error"]
    detail = client.get(f"/api/runs/{bad}")
    assert detail.status_code == 200 and detail.json()["spec"]["name"] == "Broken"


def test_stale_tab_cannot_write_into_another_project(ready, project, tmp_path, spec_factory):
    state, client = ready
    old_session = project.session_id
    other = Project.create(tmp_path, "Other")
    state.open_project(other)
    response = client.post(
        "/api/drafts",
        json={"spec": spec_factory().model_dump(mode="json")},
        headers={"x-logogram-project": old_session},
    )
    assert response.status_code == 409
    assert list(other.experiments_dir.iterdir()) == []
    assert client.get("/api/state").json()["project"]["session_id"] == other.session_id
    # A request captured just before a switch is also refused at dispatch.
    token = request_project.set((project,))
    try:
        with pytest.raises(Exception, match="project changed"):
            state.save_draft(spec_factory())
    finally:
        request_project.reset(token)
    state.job = Job(id="busy", kind="run", title="Running")
    assert client.post("/api/projects/open", json={"path": str(project.root)}).status_code == 409
    assert client.post("/api/projects/close").status_code == 409


def test_analysis_uses_the_executed_context(ready, project, tiny_backend, spec_factory):
    _, client = ready
    spec = spec_factory(
        tokenization={"prepend_bos": False},
        dataset={"path": "datasets/ioi.jsonl", "limit": 5},
        execution={"batch_size": 2},
    )
    outcome = run_spec(spec, project, backend=tiny_backend)
    assert outcome.status == "finished"
    saved = Spec.from_path(outcome.folder / "spec.json")
    body = {
        "dataset": saved.dataset.path,
        "dataset_sha256": saved.dataset.sha256,
        "model": saved.model.model_dump(mode="json"),
        "prepend_bos": False,
        "limit": 5,
        "batch_size": 2,
    }
    report = client.post("/api/baseline", json=body)
    assert report.status_code == 200, report.text
    assert report.json()["n"] == 5
    assert report.json()["summary"]["gap"] == outcome.summary["baseline"]["gap"]["mean"]
    tokens = client.post("/api/tokenize", json={**body, "index": 0}).json()
    with_bos = client.post("/api/tokenize", json={**body, "index": 0, "prepend_bos": True}).json()
    assert len(tokens["clean"]["ids"]) + 1 == len(with_bos["clean"]["ids"])
    attn = client.post("/api/attention", json={**body, "index": 0, "layer": 0, "head": 0}).json()
    assert len(attn["tokens"]) == len(tokens["clean"]["ids"])
    assert attn["n_total"] == 5
    assert client.post("/api/tokenize", json={**body, "index": 5}).status_code == 400
    for key, value in [
        ("id", "other"),
        ("revision", "different"),
        ("dtype", "float16"),
        ("process_weights", False),
    ]:
        wrong = {**body, "model": {**body["model"], key: value}}
        assert client.post("/api/baseline", json=wrong).status_code == 409
    changed = {**body, "dataset_sha256": "0" * 64}
    assert client.post("/api/baseline", json=changed).status_code == 409


def test_dataset_snapshot_survives_source_edits(project, tiny_backend, spec_factory):
    first = run_spec(spec_factory(), project, backend=tiny_backend)
    saved = Spec.from_path(first.folder / "spec.json")
    assert saved.dataset.path.startswith("datasets/snapshots/")
    assert file_sha256(project.resolve_dataset(saved.dataset.path)) == saved.dataset.sha256
    (project.datasets_dir / "ioi.jsonl").write_text("source was edited")
    second = run_spec(saved, project, backend=tiny_backend)
    assert second.status == "finished"
    assert pq.read_table(first.folder / "results.parquet").equals(
        pq.read_table(second.folder / "results.parquet")
    )
    assert (first.folder / "spec.json").read_bytes() == (second.folder / "spec.json").read_bytes()
    project.resolve_dataset(saved.dataset.path).write_text("snapshot was edited")
    failed = run_spec(saved, project, backend=tiny_backend)
    assert failed.status == "failed" and "hash" in failed.manifest["error"]


def test_interrupted_run_is_recovered_on_open(ready, project, spec_factory):
    state, client = ready
    draft = state.save_draft(spec_factory())
    write_manifest(
        project.run_dir(draft) / "manifest.json",
        {
            "run_id": draft,
            "status": "running",
            "started_at": "2026-01-01T00:00:00Z",
            "versions": {},
        },
    )
    response = client.post("/api/projects/open", json={"path": str(project.root)})
    assert response.status_code == 200
    listing = client.get("/api/runs").json()[0]
    assert listing["status"] == "failed" and "server stopped" in listing["error"]


def test_csv_keeps_every_per_prompt_value(ready, project, tiny_backend, spec_factory):
    _, client = ready
    result = run_spec(spec_factory(), project, backend=tiny_backend)
    response = client.get(f"/api/runs/{result.run_id}/export.csv")
    assert response.status_code == 200
    rows = list(csv.DictReader(io.StringIO(response.text)))
    table = pq.read_table(result.folder / "results.parquet").to_pylist()
    assert len(rows) == len(table)
    for exported, original in zip(rows, table, strict=True):
        assert int(exported["prompt"]) == original["prompt"]
        assert float(exported["effect"]) == original["effect"]


def test_determinism_is_strict_and_seeds_all_generators():
    import random

    configure_determinism(17)
    first = (random.random(), np.random.random(), torch.rand(2))
    configure_determinism(17)
    second = (random.random(), np.random.random(), torch.rand(2))
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()


def test_cancel_during_activation_capture(project, tiny_backend, monkeypatch):
    from logogram.datasets import load_dataset
    from logogram.engine import _LayerSources
    from logogram.prompts import group_by_length, prepare_prompts

    prompts = prepare_prompts(tiny_backend, load_dataset(project.datasets_dir / "ioi.jsonl"), True)
    cancel = threading.Event()
    original = tiny_backend.capture
    calls = []

    def capture(*args):
        calls.append(True)
        result = original(*args)
        cancel.set()
        return result

    monkeypatch.setattr(tiny_backend, "capture", capture)
    with pytest.raises(Cancelled):
        _LayerSources(
            tiny_backend, prompts, group_by_length(prompts), 0, ["head"], "clean", 1, cancel
        )
    assert len(calls) == 1
