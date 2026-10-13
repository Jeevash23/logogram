"""Training SAEs and lenses from the app: the plan, the job with its progress, the listings,
loading a trained SAE and reading layer predictions through a trained lens."""

from __future__ import annotations

import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from logogram.corpora import CorpusSource, TextRecord, write_corpus
from logogram.ioi import generate_ioi
from logogram.server.app import create_app
from logogram.server.security import SecurityConfig

AUTH = {"authorization": "Bearer t", "origin": "http://127.0.0.1:8765"}
MODEL = {
    "id": "tiny-gpt2",
    "source": "hub",
    "revision": "test",
    "sha256": None,
    "dtype": "float32",
    "device": "cpu",
    "process_weights": True,
}
COMMON = {
    "model": MODEL,
    "corpus": {"name": "ioi-texts", "sha256": None},
    "tokenization": {"prepend_bos": True, "max_tokens": 32},
    "steps": 12,
    "batch_tokens": 64,
    "lr": 0.01,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "seed": 0,
    "held_out": 0.25,
    "log_every": 4,
    "model_batch": 8,
}
SAE = {
    **COMMON,
    "logogram_sae_training": 1,
    "name": "app-sae",
    "site": "resid_post",
    "layer": 0,
    "architecture": {"kind": "topk", "k": 4},
    "d_sae": 64,
    "normalize": "mean_squared_norm",
    "auxk": None,
}
LENS = {
    **COMMON,
    "logogram_lens_training": 1,
    "name": "app-lens",
    "layers": None,
    "weight_decay": None,
}


@pytest.fixture
def client(tiny_backend, project, tmp_path, monkeypatch):
    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    records = generate_ioi(30, seed=0)
    texts = [TextRecord(text=t) for r in records for t in (r.clean, r.corrupt)]
    write_corpus(project, "ioi-texts", texts, CorpusSource(kind="texts", detail="IOI"))
    app = create_app(SecurityConfig(token="t", port=8765), serve_web=False)
    state = app.state.logogram
    state.backend = tiny_backend
    state.model_status = {"state": "ready"}
    state.open_project(project)
    return TestClient(app, base_url="http://127.0.0.1:8765"), state


def wait(state: Any) -> Any:
    for _ in range(1500):
        if state.job.status != "running":
            return state.job
        time.sleep(0.02)
    raise AssertionError("the job didn't end")


def test_an_sae_trains_as_a_job_and_loads(client):
    http, state = client
    assert http.get("/api/saes/project", headers=AUTH).json() == []
    plan = http.post("/api/saes/plan", json={"spec": SAE}, headers=AUTH).json()
    assert plan["refusal"] is None and plan["tokens"]["used"] > 0 and plan["seconds"]["total"] > 0
    job = http.post("/api/saes/train", json={"spec": SAE}, headers=AUTH).json()
    assert job["kind"] == "train_sae" and job["project_session"] == state.project.session_id
    ended = wait(state)
    assert ended.status == "finished", ended.error
    assert ended.progress["stage"] == "finished" and len(ended.progress["curve"]) == 3
    listed = http.get("/api/saes/project", headers=AUTH).json()
    assert [(s["name"], s["error"]) for s in listed] == [("app-sae", None)]
    sha = listed[0]["sha256"]
    assert ended.progress["sha256"] == sha

    again = http.post("/api/saes/train", json={"spec": SAE}, headers=AUTH)
    assert again.status_code == 400 and "already exists" in again.json()["error"]

    load = {"repo": "app-sae", "source": "project", "sha256": sha}
    assert http.post("/api/sae/load", json=load, headers=AUTH).json()["kind"] == "load_sae"
    wait(state)
    sae = http.get("/api/state", headers=AUTH).json()["sae"]
    assert sae["state"] == "ready", sae
    info = sae["info"]
    assert (info["source"], info["sha256"], info["path"]) == ("project", sha, "saes/app-sae")
    assert info["warnings"] == [] and "project" not in info
    body = {"dataset": "datasets/ioi.jsonl", "prepend_bos": True, "batch_size": 64}
    fit = http.post("/api/sae/fit", json=body, headers=AUTH).json()
    assert 0 < fit["variance_explained"] <= 1

    wrong = {"repo": "app-sae", "source": "project", "sha256": "0" * 64}
    http.post("/api/sae/load", json=wrong, headers=AUTH)
    assert wait(state).status == "failed" and "has changed" in state.job.error


def test_a_cancelled_training_leaves_nothing(client):
    http, state = client
    http.post("/api/saes/train", json={"spec": {**SAE, "steps": 5000}}, headers=AUTH)
    for _ in range(500):
        if state.job.progress.get("stage") == "training":
            break
        time.sleep(0.02)
    http.post("/api/jobs/cancel", headers=AUTH)
    assert wait(state).status == "cancelled"
    assert http.get("/api/saes/project", headers=AUTH).json() == []


def test_a_lens_trains_and_reads_the_layer_predictions(client):
    http, state = client
    plan = http.post("/api/lenses/plan", json={"spec": LENS}, headers=AUTH).json()
    assert plan["refusal"] is None and plan["layers"] == [0]
    http.post("/api/lenses/train", json={"spec": LENS}, headers=AUTH)
    assert wait(state).status == "finished", state.job.error
    listed = http.get("/api/lenses", headers=AUTH).json()
    assert [(x["name"], x["fits"], x["layers"]) for x in listed] == [("app-lens", True, [0])]
    held = listed[0]["held_out"]["layers"][0]
    assert held["tuned_lens"] < held["logit_lens"]
    settings = {
        "method": "tuned_lens",
        "lens": {"name": "app-lens", "sha256": None},
        "prompt_index": 0,
        "which": "clean",
        "position": {"kind": "last"},
        "top_k": 3,
    }
    body = {
        "dataset": "datasets/ioi.jsonl",
        "index": 0,
        "settings": settings,
        "prepend_bos": True,
        "batch_size": 16,
    }
    report = http.post("/api/predictions", json=body, headers=AUTH).json()
    assert report["settings"]["lens"]["sha256"] == listed[0]["sha256"]
    assert [row["readout"] for row in report["layers"]] == ["tuned_lens", "logit_lens"]
    missing = {**settings, "lens": {"name": "nope", "sha256": None}}
    refused = http.post("/api/predictions", json={**body, "settings": missing}, headers=AUTH)
    assert refused.status_code == 400 and "no lens called nope" in refused.json()["error"]


def test_a_plan_needs_the_model_the_spec_names(client):
    http, _ = client
    other = {**SAE, "model": {**MODEL, "revision": "other"}}
    response = http.post("/api/saes/plan", json={"spec": other}, headers=AUTH)
    assert response.status_code == 409 and "isn't the one this training names" in response.text
