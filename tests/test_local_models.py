"""Local models: folders on this machine, added under a name and found by the hash of their files.
They load like any other model (and are checked as any other is), runs name them by their name and
hash only, and a folder whose files changed, or that holds only pickles, is refused."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import _build_tiny_model
from logogram import local_models
from logogram.backends.base import BackendError
from logogram.backends.transformer_lens import load_model
from logogram.local_models import add_model, checked_folder, list_models, model_files, remove_model
from logogram.runner import model_matches, run_spec
from logogram.spec import ModelRef


@pytest.fixture(autouse=True)
def settings(tmp_path, monkeypatch):
    """Logogram's settings folder (where the local models are listed), for this test only."""
    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    return tmp_path / "config"


@pytest.fixture(scope="session")
def built_model(tmp_path_factory):
    folder = tmp_path_factory.mktemp("model-source")
    _build_tiny_model(folder)
    (folder / "README.md").write_text("A tiny model.\n", encoding="utf-8")
    return folder


@pytest.fixture
def folder(tmp_path, built_model):
    """A model folder of its own, so a test can change its files."""
    target = tmp_path / "models" / "checkpoint-folder"
    shutil.copytree(built_model, target)
    return target


def _expected_hash(folder: Path, names: list[str]) -> str:
    lines = "".join(
        f"{hashlib.sha256((folder / name).read_bytes()).hexdigest()}  {name}\n"
        for name in sorted(names)
    )
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def test_the_hash_is_of_the_files_loading_reads(folder):
    names = model_files(folder)
    assert names == sorted(names)
    assert "model.safetensors" in names and "config.json" in names
    assert "tokenizer.json" in names and "README.md" not in names
    progress: list[tuple[int, int, str]] = []
    model = add_model(folder, "my-fine-tune", lambda *p: progress.append(p))
    assert model.sha256 == _expected_hash(folder, names)
    # Progress in bytes, ending at the folder's whole size.
    total = sum((folder / n).stat().st_size for n in names)
    assert progress[-1][:2] == (total, total) and model.size == total
    # The same files elsewhere hash alike: the hash names the files, not where they are.
    moved = folder.parent / "elsewhere"
    shutil.copytree(folder, moved)
    assert local_models.folder_hash(local_models.hash_files(moved, names)) == model.sha256
    # Files loading doesn't read change nothing; one it reads changes the hash.
    (moved / "notes.md").write_text("notes", encoding="utf-8")
    assert model_files(moved) == names
    (moved / "generation_config.json").write_text("{}", encoding="utf-8")
    other = local_models.hash_files(moved, model_files(moved))
    assert local_models.folder_hash(other) != model.sha256


def test_sharded_weights_are_the_shards_the_index_names(folder):
    (folder / "model.safetensors").rename(folder / "model-00001-of-00001.safetensors")
    index = {"metadata": {}, "weight_map": {"w": "model-00001-of-00001.safetensors"}}
    (folder / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    (folder / "consolidated.safetensors").write_bytes(b"not read")
    names = model_files(folder)
    assert "model-00001-of-00001.safetensors" in names
    assert "consolidated.safetensors" not in names


def test_folders_that_cannot_be_loaded_safely_are_refused(folder, tmp_path):
    (folder / "model.safetensors").unlink()
    (folder / "pytorch_model.bin").write_bytes(b"\x80\x04pickle")
    with pytest.raises(BackendError, match=r"only pickles.*can run code"):
        add_model(folder, "pickled")
    assert list_models() == []
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(BackendError, match=r"no config\.json"):
        add_model(empty, "empty")


def test_names_are_checked_and_kept_apart(folder, built_model):
    with pytest.raises(BackendError, match="no slashes"):
        add_model(folder, "me/my-fine-tune")
    add_model(folder, "my-fine-tune")
    with pytest.raises(BackendError, match="already called my-fine-tune"):
        add_model(built_model, "my-fine-tune")
    with pytest.raises(BackendError, match="already added, as my-fine-tune"):
        add_model(built_model, "the same files")
    assert [m.name for m in list_models()] == ["my-fine-tune"]
    removed = remove_model("my-fine-tune")
    assert removed.name == "my-fine-tune" and list_models() == []
    assert (folder / "model.safetensors").is_file()  # forgotten, never deleted
    with pytest.raises(BackendError, match="No local model is called"):
        remove_model("my-fine-tune")


def test_a_local_model_loads_and_is_checked(folder):
    model = add_model(folder, "my-fine-tune")
    stages: list[str] = []
    backend = load_model(
        "my-fine-tune",
        device="cpu",
        source="local",
        sha256=model.sha256,
        on_progress=lambda p: stages.append(p["stage"]),
    )
    info = backend.info
    assert (info.id, info.source, info.sha256, info.revision) == (
        "my-fine-tune",
        "local",
        model.sha256,
        None,
    )
    assert stages[0] == "checking" and stages[-1] == "ready"
    # Checked like any model: TransformerLens reproduces its predictions.
    assert info.extra["checks"]["function"] <= info.extra["checks"]["tolerance"]
    ref = ModelRef(
        id="another name",
        source="local",
        revision=None,
        sha256=model.sha256,
        dtype="float32",
        device="cpu",
        process_weights=True,
    )
    assert model_matches(backend, ref)  # found by its files, whatever it is called
    hub = ref.model_copy(update={"source": "hub", "sha256": None, "id": "my-fine-tune"})
    assert not model_matches(backend, hub)
    other = ref.model_copy(update={"sha256": "0" * 64})
    assert not model_matches(backend, other)


def test_a_folder_whose_files_changed_is_refused_naming_them(folder):
    model = add_model(folder, "my-fine-tune")
    # The same bytes, touched: hashed again, accepted, and remembered.
    config = folder / "config.json"
    stat = config.stat()
    os.utime(config, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    assert checked_folder("my-fine-tune", model.sha256) == folder
    recorded = next(f for f in list_models()[0].files if f.path == "config.json")
    assert recorded.mtime_ns == config.stat().st_mtime_ns
    # Changed bytes, a new file loading reads, a missing one: each named.
    data = json.loads(config.read_text(encoding="utf-8"))
    data["n_layer"] = 3
    config.write_text(json.dumps(data), encoding="utf-8")
    (folder / "added_tokens.json").write_text("{}", encoding="utf-8")
    (folder / "generation_config.json").unlink()
    with pytest.raises(BackendError) as caught:
        checked_folder("my-fine-tune", model.sha256)
    message = str(caught.value)
    assert "changed: config.json" in message
    assert "new: added_tokens.json" in message
    assert "missing: generation_config.json" in message
    with pytest.raises(BackendError, match="aren't the ones it was added with"):
        load_model("my-fine-tune", device="cpu", source="local", sha256=model.sha256)


def test_a_spec_naming_a_model_this_machine_lacks_says_how_to_add_it(project, spec_factory):
    sha = "ab" * 32
    spec = spec_factory()
    spec = spec.model_copy(
        update={
            "model": spec.model.model_copy(
                update={
                    "id": "colleague-fine-tune",
                    "source": "local",
                    "revision": None,
                    "sha256": sha,
                }
            )
        }
    )
    outcome = run_spec(spec, project)
    assert outcome.status == "failed"
    error = outcome.manifest["error"]
    assert "colleague-fine-tune" in error and "abababababab…" in error
    assert "logogram models add PATH --name colleague-fine-tune" in error


def test_runs_name_a_local_model_by_its_hash_never_its_folder(folder, project, spec_factory):
    model = add_model(folder, "my-fine-tune")
    spec = spec_factory(dataset={"path": "datasets/ioi.jsonl", "limit": 4})
    spec = spec.model_copy(
        update={
            "model": spec.model.model_copy(
                update={
                    "id": "my-fine-tune",
                    "source": "local",
                    "revision": None,
                    "sha256": model.sha256,
                }
            )
        }
    )
    outcome = run_spec(spec, project)  # loaded by the run itself, from its folder
    assert outcome.status == "finished", outcome.manifest.get("error")
    assert outcome.manifest["model"]["source"] == "local"
    assert outcome.manifest["model"]["sha256"] == model.sha256
    saved = json.loads((outcome.folder / "spec.json").read_text(encoding="utf-8"))
    assert saved["model"]["sha256"] == model.sha256 and saved["model"]["revision"] is None
    from logogram.report import methods_text, run_report

    methods = methods_text(spec, outcome.summary, outcome.manifest, outcome.run_id)
    assert f"a local model whose files hash to SHA-256 {model.sha256}" in methods
    page = run_report(spec, outcome.summary, outcome.manifest, outcome.run_id, hardware=True)
    # Nothing written into the project says where the model's files are.
    written = "".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in project.root.rglob("*")
        if p.is_file() and p.suffix in (".json", ".jsonl", ".txt")
    )
    for text in (written, methods, page):
        assert str(folder) not in text and "checkpoint-folder" not in text


def test_an_error_from_a_local_model_never_names_its_folder(
    folder, project, spec_factory, monkeypatch
):
    model = add_model(folder, "my-fine-tune")
    spec = spec_factory()
    spec = spec.model_copy(
        update={
            "model": spec.model.model_copy(
                update={
                    "id": "my-fine-tune",
                    "source": "local",
                    "revision": None,
                    "sha256": model.sha256,
                }
            )
        }
    )

    def broken(ref):
        raise RuntimeError(f"Unrecognized model in {folder}/config.json")

    outcome = run_spec(spec, project, provider=broken)
    assert outcome.status == "failed"
    assert str(folder) not in outcome.manifest["error"]
    assert "<model folder>/config.json" in outcome.manifest["error"]
    assert str(folder) not in (outcome.folder / "traceback.txt").read_text(encoding="utf-8")


def test_models_are_added_listed_and_removed_on_the_command_line(folder):
    from logogram.cli import app

    runner = CliRunner()
    added = runner.invoke(app, ["models", "add", str(folder), "--name", "my-fine-tune"])
    assert added.exit_code == 0, added.output
    sha = list_models()[0].sha256
    assert f"SHA-256 of its files: {sha}" in added.output
    listed = runner.invoke(app, ["models", "list"])
    assert listed.exit_code == 0 and "my-fine-tune" in listed.output and sha[:12] in listed.output
    again = runner.invoke(app, ["models", "add", str(folder), "--name", "again"])
    assert again.exit_code == 1 and "already added, as my-fine-tune" in again.output
    removed = runner.invoke(app, ["models", "remove", "my-fine-tune"])
    assert removed.exit_code == 0 and "untouched" in removed.output
    assert "No local models" in runner.invoke(app, ["models", "list"]).output


def test_the_python_api_loads_a_local_model_by_its_name(folder):
    import logogram as lg

    added = lg.add_local_model(folder, "my-fine-tune")
    assert [m["name"] for m in lg.list_local_models()] == ["my-fine-tune"]
    model = lg.load_model("my-fine-tune", device="cpu")
    assert (model.info.source, model.info.sha256) == ("local", added["sha256"])
    with pytest.raises(lg.api.LogogramError, match="No local model is called nobody"):
        lg.load_model("nobody", source="local")
    lg.remove_local_model("my-fine-tune")
    assert lg.list_local_models() == []


def test_the_app_adds_lists_estimates_and_removes_local_models(folder, tmp_path):
    import time

    from fastapi.testclient import TestClient

    from logogram.server.app import create_app
    from logogram.server.security import SecurityConfig

    app = create_app(SecurityConfig(token="local-token", port=8765), serve_web=False)
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    headers = {"authorization": "Bearer local-token", "origin": "http://127.0.0.1:8765"}
    refused = client.post(
        "/api/models/local", json={"path": str(tmp_path), "name": "nothing"}, headers=headers
    )
    assert refused.status_code == 400 and "config.json" in refused.json()["error"]
    job = client.post(
        "/api/models/local", json={"path": str(folder), "name": "my-fine-tune"}, headers=headers
    ).json()
    assert job["kind"] == "add_model"
    for _ in range(200):
        state = client.get("/api/state", headers=headers).json()
        if state["job"]["status"] != "running":
            break
        time.sleep(0.05)
    assert state["job"]["status"] == "finished", state["job"]
    listed = client.get("/api/models/local", headers=headers).json()
    assert [m["name"] for m in listed] == ["my-fine-tune"] and listed[0]["folder"] == str(folder)
    body = {"id": "my-fine-tune", "source": "local", "sha256": listed[0]["sha256"], "device": "cpu"}
    estimate = client.post("/api/models/estimate", json=body, headers=headers).json()
    assert estimate["revision"] is None and estimate["download_bytes"] == 0
    assert estimate["architecture"]["n_layers"] == 2 and estimate["estimate"]["n_params"] > 0
    after = client.request("DELETE", "/api/models/local/my-fine-tune", headers=headers).json()
    assert after == []
