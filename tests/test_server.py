from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from logogram.server.app import create_app
from logogram.server.security import SecurityConfig

TOKEN = "test-token"
BASE = "http://127.0.0.1:8765"
ORIGIN = {"origin": BASE}
AUTH = {"authorization": f"Bearer {TOKEN}"}
WS_URL = "ws://127.0.0.1:8765/ws"  # a full URL, so the test client sends the right Host


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    return create_app(SecurityConfig(token=TOKEN, port=8765), serve_web=False)


@pytest.fixture
def client(app):
    return TestClient(app, base_url=BASE)


def test_every_request_needs_the_token(client):
    assert client.get("/api/state").status_code == 401
    assert client.get("/api/state", headers={"authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/state", headers=AUTH).status_code == 200


def test_launch_url_exchanges_token_for_cookie(client):
    response = client.get(f"/?token={TOKEN}", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/"
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie
    assert client.get("/api/state").status_code == 200  # cookie kept by the client
    assert client.get("/?token=nope", follow_redirects=False).status_code == 401


def test_foreign_hosts_and_origins_are_refused(client):
    assert client.get("/api/state", headers={**AUTH, "host": "evil.example"}).status_code == 421
    bad_origin = {**AUTH, "origin": "http://evil.example"}
    assert client.post("/api/projects/close", headers=bad_origin).status_code == 403
    other_port = {**AUTH, "origin": "http://127.0.0.1:9999"}
    assert client.get("/api/state", headers=other_port).status_code == 403
    cross_site = {**AUTH, "sec-fetch-site": "same-site"}
    assert client.get("/api/state", headers=cross_site).status_code == 403
    assert client.post("/api/projects/close", headers={**AUTH, **ORIGIN}).status_code == 200
    # No CORS headers, ever.
    response = client.options("/api/state", headers={**AUTH, **ORIGIN})
    assert "access-control-allow-origin" not in response.headers


def test_websocket_needs_origin_and_token(client):
    no_origin = client.websocket_connect(WS_URL, headers=AUTH)
    with pytest.raises(WebSocketDisconnect), no_origin as ws:
        ws.receive_json()
    no_token = client.websocket_connect(WS_URL, headers=ORIGIN)
    with pytest.raises(WebSocketDisconnect) as closed, no_token as ws:
        ws.receive_json()
    # A tab whose session ended learns why: the close code says the token isn't valid.
    assert closed.value.code == 4401
    with client.websocket_connect(WS_URL, headers={**AUTH, **ORIGIN}) as ws:
        assert ws.receive_json()["type"] == "hello"


def _wait_for_run(client, run_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        runs = client.get("/api/runs", headers=AUTH).json()
        status = next(r["status"] for r in runs if r["id"] == run_id)
        if status not in ("running", "draft"):
            return status
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_gui_flow_end_to_end(app, client, tiny_backend, tmp_path: Path, spec_factory):
    app.state.logogram.backend = tiny_backend
    app.state.logogram.model_status = {"state": "ready"}
    headers = {**AUTH, **ORIGIN}

    r = client.post(
        "/api/projects", json={"name": "Flow", "parent": str(tmp_path)}, headers=headers
    )
    assert r.status_code == 200, r.text
    r = client.post("/api/datasets/ioi", json={"name": "ioi", "n": 10, "seed": 1}, headers=headers)
    assert r.json()["n"] == 10
    # Refuses to overwrite silently.
    assert (
        client.post("/api/datasets/ioi", json={"name": "ioi"}, headers=headers).status_code == 400
    )

    logit_diff = {"kind": "logit_diff", "normalization": "dataset_gap"}
    context = {"prepend_bos": True, "batch_size": 64}
    strip = client.post(
        "/api/tokenize",
        json={"dataset": "datasets/ioi.jsonl", "index": 0, "metric": logit_diff, **context},
        headers=headers,
    ).json()
    assert strip["aligned"] and strip["differs"] and not strip["issues"]
    assert strip["answer"]["id"] is not None

    bad = {
        "clean": "Hello world",
        "corrupt": "Hello world world",
        "answer": " Mary Mary",
        "distractor": " John",
    }
    strip = client.post(
        "/api/tokenize", json={"record": bad, "metric": logit_diff, **context}, headers=headers
    ).json()
    assert not strip["aligned"] and {i["kind"] for i in strip["issues"]} == {
        "length_mismatch",
        "answer_tokens",
    }
    # A log-probability reads an answer of several tokens as a continuation.
    logprob = {"kind": "logprob_diff", "normalization": "dataset_gap"}
    strip = client.post(
        "/api/tokenize", json={"record": bad, "metric": logprob, **context}, headers=headers
    ).json()
    assert {i["kind"] for i in strip["issues"]} == {"length_mismatch"}

    base = client.post(
        "/api/baseline",
        json={"dataset": "datasets/ioi.jsonl", "metric": logit_diff, **context},
        headers=headers,
    ).json()
    assert base["n"] == 10 and len(base["prompts"][0]["clean_top"]) == 5
    assert base["summary"]["metric"]["clean"] == pytest.approx(base["summary"]["clean_logit_diff"])

    spec = spec_factory(dataset={"path": "datasets/ioi.jsonl"}).model_dump(mode="json")
    started = client.post("/api/runs", json={"spec": spec}, headers=headers).json()
    run_id = started["run_id"]
    assert _wait_for_run(client, run_id) == "finished"
    run = client.get(f"/api/runs/{run_id}", headers=AUTH).json()
    assert run["summary"]["n_sites"] == tiny_backend.info.n_layers * tiny_backend.info.n_heads
    assert run["manifest"]["status"] == "finished"

    detail = client.get(f"/api/runs/{run_id}/sites/3", headers=AUTH).json()
    assert len(detail["prompts"]) == 10 and detail["prompts"][0]["clean"]
    assert len(detail["strongest"]) == 3

    attention = client.post(
        "/api/attention",
        json={"dataset": "datasets/ioi.jsonl", "index": 2, "layer": 1, "head": 0, **context},
        headers=headers,
    ).json()
    n_tok = len(attention["tokens"])
    assert len(attention["pattern"]) == n_tok and len(attention["average"][0]) == n_tok

    experiment = {
        "kind": "ablation",
        "baseline": {"kind": "resample", "pool": "corrupt", "donors": 2, "seed": 0},
    }
    robust = client.post(
        f"/api/runs/{run_id}/robustness", json={"experiment": experiment}, headers=headers
    ).json()
    assert _wait_for_run(client, robust["run_id"]) == "finished"
    derived = client.get(f"/api/runs/{run_id}/derived", headers=AUTH).json()
    assert derived[0]["derived_from"]["kind"] == "robustness"
    comparison = client.get(f"/api/compare?a={run_id}&b={robust['run_id']}", headers=AUTH).json()
    assert comparison["same_layout"] and comparison["spec_differences"]


def test_every_endpoint_matches_its_schema(app, client, tiny_backend, tmp_path, spec_factory):
    """Responses are validated against their models; exercise the routes the flow test skips."""
    headers = {**AUTH, **ORIGIN}
    state = client.get("/api/state", headers=AUTH).json()
    assert state["model"]["state"] == "none" and state["first_run"] is True
    assert client.post("/api/models/unload", headers=headers).json()["state"] == "none"
    assert client.get("/api/system", headers=AUTH).json()["backend"] in ("cpu", "cuda", "mps")
    assert client.get("/api/models/presets", headers=AUTH).json()["presets"]
    assert len(client.get("/api/ioi/templates", headers=AUTH).json()) >= 3
    listing = client.get("/api/fs", params={"path": str(tmp_path)}, headers=AUTH).json()
    assert Path(listing["path"]) == tmp_path.resolve()
    settings = client.post("/api/settings", json={"theme": "dark"}, headers=headers).json()
    assert settings["theme"] == "dark"
    assert client.get("/api/state", headers=AUTH).json()["theme"] == "dark"

    app.state.logogram.backend = tiny_backend
    app.state.logogram.model_status = {"state": "ready"}
    client.post("/api/projects", json={"name": "Schemas", "parent": str(tmp_path)}, headers=headers)
    assert client.get("/api/projects/recent", headers=AUTH).json()[0]["name"] == "Schemas"
    client.post("/api/datasets/ioi", json={"name": "ioi", "n": 6}, headers=headers)
    detail = client.get("/api/datasets/ioi.jsonl", headers=AUTH).json()
    assert detail["n"] == 6 and detail["issues"] == [] and len(detail["lengths"]) >= 1
    spec = spec_factory(scope={"kind": "layer_components", "position": {"kind": "last"}})
    draft = client.post("/api/drafts", json={"spec": spec.model_dump(mode="json")}, headers=headers)
    draft_id = draft.json()["run_id"]
    runs = client.get("/api/runs", headers=AUTH).json()
    assert next(r for r in runs if r["id"] == draft_id)["status"] == "draft"
    started = client.post(
        "/api/runs",
        json={"spec": spec.model_dump(mode="json"), "draft_id": draft_id},
        headers=headers,
    ).json()
    assert started["run_id"] == draft_id  # a draft runs in its own folder
    assert _wait_for_run(client, draft_id) == "finished"
    assert client.post("/api/jobs/cancel", headers=headers).json()["job"]["status"] == "finished"
    rerun = client.post(f"/api/runs/{draft_id}/rerun", headers=headers).json()
    assert _wait_for_run(client, rerun["run_id"]) == "finished"
    detail = client.get(f"/api/runs/{rerun['run_id']}", headers=AUTH).json()
    assert detail["manifest"]["derived_from"] == {"run": draft_id, "kind": "rerun", "change": None}


def test_an_attribution_patching_run_is_verified_by_patching(
    app, client, tiny_backend, tmp_path: Path, spec_factory
):
    app.state.logogram.backend = tiny_backend
    app.state.logogram.model_status = {"state": "ready"}
    headers = {**AUTH, **ORIGIN}
    client.post("/api/projects", json={"name": "Verify", "parent": str(tmp_path)}, headers=headers)
    client.post("/api/datasets/ioi", json={"name": "ioi", "n": 10, "seed": 1}, headers=headers)
    spec = spec_factory(
        dataset={"path": "datasets/ioi.jsonl"},
        experiment={"kind": "attribution_patching", "direction": "clean_to_corrupt"},
        scope={"kind": "heads", "position": {"kind": "last"}},
    ).model_dump(mode="json")
    run_id = client.post("/api/runs", json={"spec": spec}, headers=headers).json()["run_id"]
    assert _wait_for_run(client, run_id) == "finished"

    started = client.post(f"/api/runs/{run_id}/verify", json={"top": 4}, headers=headers)
    assert started.status_code == 200, started.text
    verified = started.json()["run_id"]
    assert _wait_for_run(client, verified) == "finished"
    run = client.get(f"/api/runs/{verified}", headers=AUTH).json()
    assert run["spec"]["experiment"] == {
        "kind": "activation_patching",
        "direction": "clean_to_corrupt",
    }
    assert len(run["spec"]["scope"]["sites"]) == 4
    assert run["manifest"]["derived_from"]["kind"] == "verification"
    comparison = client.get(f"/api/compare?a={run_id}&b={verified}", headers=AUTH).json()
    assert comparison["n_common"] == 4

    # Only attribution patching runs are verified this way.
    again = client.post(f"/api/runs/{verified}/verify", json={"top": 4}, headers=headers)
    assert again.status_code == 400 and "Only attribution patching" in again.json()["error"]
