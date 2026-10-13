"""The app's routes for corpora, top-activating examples, representations, exported activations
and directions: what they write, what they answer, and what they refuse."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from logogram.server.app import _project_scoped, create_app
from logogram.server.security import SecurityConfig

TOKEN = "test-token"
BASE = "http://127.0.0.1:8765"
HEADERS = {"authorization": f"Bearer {TOKEN}", "origin": BASE}
READ = {"prepend_bos": True, "batch_size": 4}


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    return create_app(SecurityConfig(token=TOKEN, port=8765), serve_web=False)


@pytest.fixture
def client(app):
    return TestClient(app, base_url=BASE)


class Opened(list[tuple[str, dict[str, Any]]]):
    """The events the server sent since the project opened, and the open project's session."""

    session: str

    def said(self, kind: str) -> bool:
        return (kind, {"project_session": self.session}) in self


@pytest.fixture
def opened(app, client, project, tiny_backend) -> Opened:
    """The test project open, with the tiny model loaded."""
    state = app.state.logogram
    state.backend = tiny_backend
    state.model_status = {"state": "ready"}
    r = client.post("/api/projects/open", json={"path": str(project.root)}, headers=HEADERS)
    assert r.status_code == 200, r.text
    events = Opened()
    events.session = state.project.session_id
    publish = state.hub.publish

    def record(kind: str, data: dict[str, Any]) -> None:
        events.append((kind, data))
        publish(kind, data)

    state.hub.publish = record
    return events


def _wait_for_job(client: TestClient, timeout: float = 60) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get("/api/state", headers=HEADERS).json()["job"]
        if job is not None and job["status"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("the job did not end")


STORIES = "\n".join(
    [
        "When Mary and John went to the store, John gave a drink to Mary.",
        "When Tom and Kelly went to the park, Kelly gave a ball to Tom.",
        "Then, Sarah and Chris had a lot of fun at the school.",
        "",
        "Afterwards, Jessica and Robert went to the restaurant.",
    ]
)


def test_corpora_are_imported_paged_renamed_and_deleted(client, opened, project) -> None:
    r = client.post(
        "/api/corpora/import",
        json={"name": "stories", "text": STORIES, "filename": "stories.txt", "format": "lines"},
        headers=HEADERS,
    )
    assert r.status_code == 200, r.text
    assert r.json()["n"] == 4 and r.json()["source"]["path"] == "stories.txt"
    assert opened.said("corpora.changed")
    again = client.post(
        "/api/corpora/import",
        json={"name": "stories", "text": STORIES, "format": "lines"},
        headers=HEADERS,
    )
    assert again.status_code == 400 and "already exists" in again.json()["error"]

    page = client.get("/api/corpora/stories?q=went", headers=HEADERS).json()
    assert page["matches"] == 3 and [t["index"] for t in page["texts"]] == [0, 1, 3]

    pairs = client.post(
        "/api/corpora/import/dataset",
        json={"name": "pairs", "dataset": "datasets/ioi.jsonl", "sides": "both", "label": "side"},
        headers=HEADERS,
    ).json()
    assert pairs["n"] == 24 and pairs["labels"] == ["clean", "corrupt"]

    renamed = client.post("/api/corpora/stories/rename", json={"name": "tales"}, headers=HEADERS)
    assert renamed.status_code == 200 and renamed.json()["name"] == "tales"
    assert client.delete("/api/corpora/pairs", headers=HEADERS).json() == {"ok": True}
    listed = client.get("/api/corpora", headers=HEADERS).json()
    assert [c["name"] for c in listed] == ["tales"]
    missing = client.get("/api/corpora/pairs", headers=HEADERS)
    assert missing.status_code == 400 and "no corpus called pairs" in missing.json()["error"]


def test_corpus_routes_say_which_project_they_are_for(client, opened, project) -> None:
    for path in (
        "/api/corpora/x",
        "/api/corpus/top",
        "/api/representations/pca",
        "/api/activations/export",
    ):
        assert _project_scoped(path), path
    # A browser tab, signed in by its cookie, must say which project each request is for.
    tab = TestClient(client.app, base_url=BASE)
    tab.get(f"/?token={TOKEN}", follow_redirects=False)
    body = {"name": "x", "text": "a", "format": "lines"}
    for method, path, payload in (
        ("POST", "/api/corpora/import", body),
        ("GET", "/api/corpora/x", None),
        ("DELETE", "/api/corpora/x", None),
        ("POST", "/api/corpus/top", {}),
        ("POST", "/api/representations/pca", {}),
        ("POST", "/api/activations/export", {}),
        ("DELETE", "/api/directions/x", None),
    ):
        r = tab.request(method, path, json=payload, headers={"origin": BASE})
        assert r.status_code == 409 and "which project" in r.json()["error"], path
        stale = {"origin": BASE, "x-logogram-project": "an-older-project"}
        r = tab.request(method, path, json=payload, headers=stale)
        assert r.status_code == 409 and "changed in another tab" in r.json()["error"], path
    same = tab.post(
        "/api/corpora/import",
        json=body,
        headers={"origin": BASE, "x-logogram-project": opened.session},
    )
    assert same.status_code == 200, same.text


def test_a_hub_dataset_file_is_imported_as_a_job(client, opened, monkeypatch, tmp_path) -> None:
    from logogram.backends import hub

    local = tmp_path / "train.csv"
    local.write_text("sentence,label\nMary went home.,1\nJohn went out.,0\n", encoding="utf-8")

    def download(repo, path, revision, progress=None, *, cancel=None, limit=None):  # type: ignore[no-untyped-def]
        progress(5, 10, path)
        progress(10, 10, path)
        return "a" * 40, local

    monkeypatch.setattr(hub, "download_dataset_file", download)
    monkeypatch.setattr(
        hub,
        "dataset_files",
        lambda repo, revision: ("a" * 40, [("README.md", 10), ("data/train.csv", 40)]),
    )
    files = client.get("/api/hub/dataset-files?repo=someone/texts", headers=HEADERS).json()
    assert files["revision"] == "a" * 40
    assert [(f["path"], f["readable"]) for f in files["files"]] == [
        ("README.md", False),
        ("data/train.csv", True),
    ]
    r = client.post(
        "/api/corpora/import/hub",
        json={
            "name": "hub-texts",
            "repo": "someone/texts",
            "path": "data/train.csv",
            "text_field": "sentence",
            "label_field": "label",
        },
        headers=HEADERS,
    )
    assert r.status_code == 200, r.text
    assert r.json()["kind"] == "import_corpus"
    job = _wait_for_job(client)
    assert job["status"] == "finished", job["error"]
    assert job["progress"]["done"] == 10 and job["progress"]["corpus"] == "hub-texts"
    listed = {c["name"]: c for c in client.get("/api/corpora", headers=HEADERS).json()}
    source = listed["hub-texts"]["source"]
    assert (source["repo"], source["path"], source["revision"]) == (
        "someone/texts",
        "data/train.csv",
        "a" * 40,
    )
    refused = client.post(
        "/api/corpora/import/hub",
        json={"name": "hub-texts", "repo": "someone/texts", "path": "data/train.csv"},
        headers=HEADERS,
    )
    assert refused.status_code == 400 and "already exists" in refused.json()["error"]
    unreadable = client.post(
        "/api/corpora/import/hub",
        json={"name": "other", "repo": "someone/texts", "path": "data/train.jsonl.gz"},
        headers=HEADERS,
    )
    assert unreadable.status_code == 400 and "compressed" in unreadable.json()["error"]


def test_top_examples_representations_and_exports_answer_with_their_settings(
    client, opened, project
) -> None:
    client.post(
        "/api/corpora/import",
        json={"name": "stories", "text": STORIES, "format": "lines"},
        headers=HEADERS,
    )
    body = {
        **READ,
        "corpus": "stories",
        "unit": {"kind": "neuron", "layer": 1, "neuron": 3},
        "texts": None,
        "max_tokens": 32,
        "top_k": 3,
        "context": 2,
    }
    top = client.post("/api/corpus/top", json=body, headers=HEADERS)
    assert top.status_code == 200, top.text
    out = top.json()
    assert len(out["top"]) == 3 and out["settings"]["max_tokens"] == 32
    assert out["corpus"]["name"] == "stories" and out["corpus"]["sha256"]
    assert any(kind == "analysis.progress" for kind, _ in opened)
    assert client.post("/api/corpus/top", json=body, headers=HEADERS).json() == out
    no_sae = client.post(
        "/api/corpus/top",
        json={**body, "unit": {"kind": "sae_feature", "feature": 1}},
        headers=HEADERS,
    )
    assert no_sae.status_code == 409 and "Load an SAE" in no_sae.json()["error"]

    source = {"kind": "dataset", "path": "datasets/ioi.jsonl", "sides": "both", "mark": "side"}
    reading = {
        **READ,
        "source": source,
        "site": {"kind": "resid_post", "layer": 0},
        "position": {"kind": "last"},
    }
    pca = client.post("/api/representations/pca", json=reading, headers=HEADERS)
    assert pca.status_code == 200, pca.text
    assert pca.json()["n"] == 24 and pca.json()["marks"] == ["clean", "corrupt"]

    saved = client.post(
        "/api/representations/direction",
        json={
            **reading,
            "name": "sides",
            "save": {"kind": "mean_difference", "positive": "clean", "negative": "corrupt"},
        },
        headers=HEADERS,
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["made_by"]["method"] == "mean_difference"
    component = client.post(
        "/api/representations/direction",
        json={**reading, "name": "pc1", "save": {"kind": "component", "component": 1}},
        headers=HEADERS,
    )
    assert component.json()["made_by"]["method"] == "pca"
    names = [d["name"] for d in client.get("/api/directions", headers=HEADERS).json()]
    assert names == ["pc1", "sides"]

    direction = {**body, "unit": {"kind": "direction", "name": "sides", "sha256": None}}
    signed = client.post("/api/corpus/top", json=direction, headers=HEADERS).json()
    assert signed["signed"] and len(signed["bottom"]) == 3

    exported = client.post(
        "/api/activations/export",
        json={
            **{k: v for k, v in reading.items() if k != "site"},
            "sites": [{"kind": "resid_pre", "layer": 1}],
            "name": "acts",
        },
        headers=HEADERS,
    )
    assert exported.status_code == 200, exported.text
    assert exported.json()["path"] == "activations/acts.safetensors"
    meta = json.loads((project.root / "activations" / "acts.json").read_text(encoding="utf-8"))
    assert meta["n"] == 24 and meta["sites"][0]["tensor"] == "resid_pre:1"

    other_model = {
        **reading,
        "model": {
            "id": "another",
            "source": "hub",
            "revision": "x",
            "sha256": None,
            "dtype": "float32",
            "device": "cpu",
            "process_weights": True,
        },
    }
    refused = client.post("/api/representations/pca", json=other_model, headers=HEADERS)
    assert refused.status_code == 409 and "doesn't match" in refused.json()["error"]
    bad_label = client.post(
        "/api/representations/pca",
        json={
            **reading,
            "source": {"kind": "corpus", "name": "stories", "max_tokens": 16},
            "position": {"kind": "label", "label": "S2"},
        },
        headers=HEADERS,
    )
    assert bad_label.status_code == 400 and "no labelled positions" in bad_label.json()["error"]


def test_directions_are_renamed_and_deleted(client, opened, project, tiny_backend) -> None:
    import torch

    from logogram.directions import model_of, save_direction

    save_direction(
        project,
        "first",
        torch.ones(tiny_backend.info.d_model),
        model=model_of(tiny_backend.info),
        site={"kind": "resid_post", "layer": 1},
        method="import",
        run=None,
        detail="a test",
    )
    r = client.post("/api/directions/first/rename", json={"name": "second"}, headers=HEADERS)
    assert r.status_code == 200 and r.json()["name"] == "second" and r.json()["error"] is None
    assert opened.said("directions.changed")
    taken = client.post("/api/directions/second/rename", json={"name": "second"}, headers=HEADERS)
    assert taken.status_code == 200  # the same name: nothing to do
    bad = client.post("/api/directions/second/rename", json={"name": "../x"}, headers=HEADERS)
    assert bad.status_code == 400 and "can't name a direction" in bad.json()["error"]
    assert client.delete("/api/directions/second", headers=HEADERS).json() == {"ok": True}
    gone = client.delete("/api/directions/second", headers=HEADERS)
    assert gone.status_code == 400 and "no direction called second" in gone.json()["error"]
    assert client.get("/api/directions", headers=HEADERS).json() == []


def test_a_corpus_name_in_a_route_is_never_a_path(client, opened, project, tmp_path: Path) -> None:
    for name in ("..%2Fproject", "%2E%2E"):
        r = client.get(f"/api/corpora/{name}", headers=HEADERS)
        assert r.status_code in (400, 404)
    assert not (tmp_path / "corpora").exists()
