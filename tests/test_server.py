from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

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
    # A run is listed as ended once its manifest is written, a moment before its job lets go of
    # the server; until then another job can't start, so wait for both.
    deadline = time.time() + timeout
    while time.time() < deadline:
        runs = client.get("/api/runs", headers=AUTH).json()
        status = next(r["status"] for r in runs if r["id"] == run_id)
        job = client.get("/api/state", headers=AUTH).json()["job"]
        if status not in ("running", "draft") and (job is None or job["status"] != "running"):
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


def test_dataset_detail_names_answers_read_as_continuations(app, client, tiny_backend, tmp_path):
    """The app says before a run which answers only a metric that reads continuations can score."""
    headers = {**AUTH, **ORIGIN}
    client.post("/api/projects", json={"name": "Answers", "parent": str(tmp_path)}, headers=headers)
    clean = "When Mary and John went to the store, John gave a drink to"
    corrupt = "When Mary and John went to the store, Mary gave a drink to"
    lines = [
        {"clean": clean, "corrupt": corrupt, "answer": " Mary", "distractor": " John"},
        {"clean": clean, "corrupt": corrupt, "answer": " Mary Mary", "distractor": " John"},
        # A set of single tokens is read at one position, like a single token.
        {"clean": clean, "corrupt": corrupt, "answer": [" Mary", " John"], "distractor": " to"},
    ]
    text = "\n".join(json.dumps(line) for line in lines)
    created = client.post(
        "/api/datasets/import", json={"name": "answers", "text": text}, headers=headers
    )
    assert created.status_code == 200, created.text
    params = {"path": "datasets/answers.jsonl"}
    # Without a model there are no tokens to count.
    assert client.get("/api/dataset", params=params, headers=AUTH).json()["continuations"] is None
    app.state.logogram.backend = tiny_backend
    app.state.logogram.model_status = {"state": "ready"}
    detail = client.get("/api/dataset", params=params, headers=AUTH).json()
    assert detail["issues"] == [] and detail["continuations"] == [1]


def test_every_endpoint_matches_its_schema(app, client, tiny_backend, tmp_path, spec_factory):
    """Responses are validated against their models; exercise the routes the flow test skips."""
    headers = {**AUTH, **ORIGIN}
    state = client.get("/api/state", headers=AUTH).json()
    assert state["model"]["state"] == "none" and state["first_run"] is True
    assert state["theme"] == "system"  # until someone picks one
    assert state["single_key_shortcuts"] is True
    assert client.post("/api/models/unload", headers=headers).json()["state"] == "none"
    assert client.get("/api/system", headers=AUTH).json()["backend"] in ("cpu", "cuda", "mps")
    assert client.get("/api/models/presets", headers=AUTH).json()["presets"]
    assert len(client.get("/api/ioi/templates", headers=AUTH).json()) >= 3
    listing = client.get("/api/fs", params={"path": str(tmp_path)}, headers=AUTH).json()
    assert Path(listing["path"]) == tmp_path.resolve()
    settings = client.post("/api/settings", json={"theme": "dark"}, headers=headers).json()
    assert settings["theme"] == "dark"
    assert client.get("/api/state", headers=AUTH).json()["theme"] == "dark"
    client.post("/api/settings", json={"single_key_shortcuts": False}, headers=headers)
    assert client.get("/api/state", headers=AUTH).json()["single_key_shortcuts"] is False
    # What a run would need is known only for a loaded model.
    body = {"batch_size": 8, "seq_len": 16, "method": "attribution_patching"}
    assert client.post("/api/models/run-memory", json=body, headers=headers).status_code == 409

    app.state.logogram.backend = tiny_backend
    app.state.logogram.model_status = {"state": "ready"}
    memory = client.post("/api/models/run-memory", json=body, headers=headers).json()
    assert memory["all_layers"] is True and memory["batch_size"] == 8 and memory["seq_len"] == 16
    assert (
        memory["verdict"] in ("fits", "tight", "wont_fit")
        and memory["total"] > memory["activations"]
    )
    client.post("/api/projects", json={"name": "Schemas", "parent": str(tmp_path)}, headers=headers)
    assert client.get("/api/projects/recent", headers=AUTH).json()[0]["name"] == "Schemas"
    assert client.get("/api/queue", headers=AUTH).json() == []
    client.post("/api/datasets/ioi", json={"name": "ioi", "n": 6}, headers=headers)
    detail = client.get("/api/datasets/ioi.jsonl", headers=AUTH).json()
    assert detail["n"] == 6 and detail["issues"] == [] and len(detail["lengths"]) >= 1
    edits = {
        "dataset": "datasets/ioi.jsonl",
        "dataset_sha256": detail["sha256"],
        "name": "ioi-edited",
        "remove": [0],
        "replace": {"1": detail["records"][2]},
    }
    derived = client.post("/api/datasets/derive", json=edits, headers=headers).json()
    assert derived == {"name": "ioi-edited.jsonl", "path": "datasets/ioi-edited.jsonl", "n": 5}
    tasks = {t["id"]: t for t in client.get("/api/tasks", headers=AUTH).json()}
    assert {"ioi", "greater_than", "docstring", "factual_recall"} <= set(tasks)
    assert tasks["greater_than"]["metric"] == "prob_diff"
    body = {"task": "ioi", "name": "ioi-task", "n": 4, "seed": 2, "options": {}}
    created = client.post("/api/datasets/generate", json=body, headers=headers).json()
    assert created == {"name": "ioi-task.jsonl", "path": "datasets/ioi-task.jsonl", "n": 4,
                       "metric": "logit_diff"}  # fmt: skip
    again = client.post("/api/datasets/generate", json=body, headers=headers)
    assert again.status_code == 400 and "already exists" in again.json()["error"]
    bad = client.post(
        "/api/datasets/generate", json={**body, "task": "no_such_task"}, headers=headers
    )
    assert bad.status_code == 400 and "ioi" in bad.json()["error"]
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


# -- editing a dataset's prompts into a new dataset ----------------------------------------------

CLEAN = "When Mary and John went to the store, John gave a drink to"
CORRUPT = "When Mary and John went to the store, Mary gave a drink to"
DERIVE = "/api/datasets/derive"
WRITE = {**AUTH, **ORIGIN}


def _pair(i: int) -> dict[str, Any]:
    return {
        "clean": CLEAN,
        "corrupt": CORRUPT,
        "answer": " Mary",
        "distractor": " John",
        "id": f"pair-{i}",
    }


@pytest.fixture
def edits(app, client, tmp_path) -> dict[str, Any]:
    """A project with a dataset of four prompt pairs, and a request that changes none of them."""
    client.post("/api/projects", json={"name": "Edits", "parent": str(tmp_path)}, headers=WRITE)
    text = "\n".join(json.dumps(_pair(i)) for i in range(4))
    created = client.post("/api/datasets/import", json={"name": "src", "text": text}, headers=WRITE)
    assert created.status_code == 200, created.text
    detail = client.get("/api/dataset", params={"path": "datasets/src.jsonl"}, headers=AUTH).json()
    return {
        "dataset": "datasets/src.jsonl",
        "dataset_sha256": detail["sha256"],
        "name": "src-edited",
        "remove": [],
        "replace": {},
    }


def _datasets(app) -> Path:
    return app.state.logogram.project.datasets_dir


def _files(app) -> list[str]:
    return sorted(p.name for p in _datasets(app).iterdir())


def test_edited_prompts_are_saved_as_a_new_dataset(app, client, edits):
    source = _datasets(app) / "src.jsonl"
    before = source.read_bytes()
    clean = "When Anna and Ben went home, Ben gave a book to"
    edited = {
        "clean": clean,
        "corrupt": "When Anna and Ben went home, Anna gave a book to",
        "answer": [" Anna", " Ann"],  # a set of tokens, any of which counts
        "distractor": " Ben Smith",  # a continuation of several tokens
        "positions": {"IO": [5, 9], "end": [len(clean) - 2, len(clean)]},
        "id": "edited",
        "meta": {"template": "home", "checked": True},
    }
    body = {**edits, "remove": [0, 3], "replace": {"2": edited}}
    r = client.post(DERIVE, json=body, headers=WRITE)
    assert r.status_code == 200, r.text
    assert r.json() == {"name": "src-edited.jsonl", "path": "datasets/src-edited.jsonl", "n": 2}
    # The dataset edited never changes: runs record its hash.
    assert source.read_bytes() == before
    params = {"path": "datasets/src-edited.jsonl"}
    detail = client.get("/api/dataset", params=params, headers=AUTH).json()
    assert detail["records"] == [_pair(1), edited]
    listed = {d["name"]: d for d in client.get("/api/project", headers=AUTH).json()["datasets"]}
    assert listed["src-edited.jsonl"]["n"] == 2 and listed["src.jsonl"]["n"] == 4

    # A run's saved copy of its prompts is edited the same way.
    snapshot = _datasets(app) / "snapshots" / f"{edits['dataset_sha256']}.jsonl"
    snapshot.parent.mkdir()
    snapshot.write_bytes(before)
    body = {**edits, "dataset": f"datasets/snapshots/{snapshot.name}", "name": "from-run.jsonl"}
    r = client.post(DERIVE, json={**body, "remove": [1]}, headers=WRITE)
    assert r.json() == {"name": "from-run.jsonl", "path": "datasets/from-run.jsonl", "n": 3}
    assert snapshot.read_bytes() == before


def test_a_new_dataset_never_replaces_a_file(app, client, edits):
    (_datasets(app) / "taken.jsonl").write_text(json.dumps(_pair(9)) + "\n", encoding="utf-8")
    files = {p.name: p.read_bytes() for p in _datasets(app).iterdir()}
    for name in ("taken", "taken.jsonl", "src"):  # the dataset edited is no exception
        r = client.post(DERIVE, json={**edits, "name": name, "remove": [0]}, headers=WRITE)
        assert r.status_code == 400, r.text
        assert r.json()["error"] == f"{Path(name).stem}.jsonl already exists. Choose another name."
    assert {p.name: p.read_bytes() for p in _datasets(app).iterdir()} == files


@pytest.mark.parametrize(
    "record",
    [
        {"clean": "a", "corrupt": "b", "answer": "c"},
        {"clean": "", "corrupt": "b", "answer": "c", "distractor": "d"},
        {"clean": "a", "corrupt": "b", "answer": "", "distractor": "d"},
        {"clean": "a", "corrupt": "b", "answer": [" x", " x"], "distractor": "d"},
        {"clean": "a", "corrupt": "b", "answer": [], "distractor": "d"},
        {"clean": "a", "corrupt": "b", "answer": "c", "distractor": 4},
        {
            "clean": "ab",
            "corrupt": "b",
            "answer": "c",
            "distractor": "d",
            "positions": {"X": [0, 9]},
        },
        {
            "clean": "ab",
            "corrupt": "b",
            "answer": "c",
            "distractor": "d",
            "positions": {"X": [1, 1]},
        },
        {
            "clean": "ab",
            "corrupt": "b",
            "answer": "c",
            "distractor": "d",
            "positions": {"7": [0, 1]},
        },
        {"clean": "a", "corrupt": "b", "answer": "c", "distractor": "d", "meta": [1]},
        {"clean": "a", "corrupt": "b", "answer": "c", "distractor": "d", "note": "no such field"},
        [1, 2],
        "a prompt",
    ],
)
def test_edited_prompts_are_checked_as_an_import_is(app, client, edits, record):
    imported = client.post(
        "/api/datasets/import", json={"name": "check", "text": json.dumps(record)}, headers=WRITE
    )
    derived = client.post(DERIVE, json={**edits, "replace": {"1": record}}, headers=WRITE)
    assert imported.status_code == derived.status_code == 400, derived.text
    # The same check in the same words; only the place it names differs.
    expected = imported.json()["error"].replace("check.jsonl, line 1", "Prompt 1")
    assert derived.json()["error"] == expected
    assert _files(app) == ["src.jsonl"]


@pytest.mark.parametrize(
    "name", ["../outside", "a/b", "/abs", ".hidden", "-dash", "", "has space", "x" * 101]
)
def test_a_new_dataset_needs_a_plain_name(app, client, edits, tmp_path, name):
    r = client.post(DERIVE, json={**edits, "name": name, "remove": [0]}, headers=WRITE)
    assert r.status_code == 400 and r.json()["error"].startswith("Dataset names can use letters")
    assert _files(app) == ["src.jsonl"]
    assert not list(tmp_path.rglob("outside*")) and not list(tmp_path.rglob("abs*"))


@pytest.mark.parametrize(
    "change, message",
    [
        ({"remove": [0, 1, 2, 3]}, "Removing every prompt would leave an empty dataset."),
        ({"remove": [3, 2, 1, 0]}, "Removing every prompt would leave an empty dataset."),
        ({"remove": [4]}, "There is no prompt 4 to remove: the prompts are 0 to 3."),
        ({"remove": [-1]}, "There is no prompt -1 to remove"),
        ({"remove": [1, 1]}, "Prompt 1 is listed twice to remove."),
        ({"replace": {"4": _pair(0)}}, "There is no prompt 4 to edit"),
        ({"remove": [2], "replace": {"2": _pair(0)}}, "Prompt 2 is both edited and removed."),
    ],
)
def test_edits_must_name_prompts_the_dataset_has(app, client, edits, change, message):
    r = client.post(DERIVE, json={**edits, **change}, headers=WRITE)
    assert r.status_code == 400 and r.json()["error"].startswith(message), r.text
    assert _files(app) == ["src.jsonl"]


def test_edits_to_a_dataset_that_changed_since_are_refused(app, client, edits):
    source = _datasets(app) / "src.jsonl"
    source.write_text(source.read_text("utf-8") + json.dumps(_pair(4)) + "\n", "utf-8")
    r = client.post(DERIVE, json={**edits, "remove": [0]}, headers=WRITE)
    assert r.status_code == 409 and "src.jsonl changed since these edits were made" in r.text
    gone = client.post(DERIVE, json={**edits, "dataset": "datasets/gone.jsonl"}, headers=WRITE)
    assert gone.status_code == 400 and "gone.jsonl" in gone.json()["error"]
    assert _files(app) == ["src.jsonl"]


def test_a_new_dataset_stays_within_what_logogram_reads(app, client, edits, monkeypatch):
    size = (_datasets(app) / "src.jsonl").stat().st_size
    monkeypatch.setattr("logogram.datasets.DATASET_LIMIT", size + 64)
    longer = {**_pair(0), "clean": CLEAN + " and" * 40}
    r = client.post(DERIVE, json={**edits, "replace": {"0": longer}}, headers=WRITE)
    assert r.status_code == 400 and r.json()["error"] == (
        f"src-edited.jsonl would be larger than the {size + 64} bytes Logogram reads. "
        "Split the prompts into smaller files."
    )
    monkeypatch.setattr("logogram.datasets.DATASET_LIMIT", size - 1)  # nor is a larger one read
    r = client.post(DERIVE, json={**edits, "remove": [0]}, headers=WRITE)
    assert r.status_code == 400 and "src.jsonl is larger than" in r.json()["error"]
    assert _files(app) == ["src.jsonl"]


def test_editing_prompts_needs_the_token_its_origin_and_the_project(app, client, edits):
    body = {**edits, "remove": [0]}
    refused = [
        (ORIGIN, 401),
        ({**AUTH, "origin": "http://evil.example"}, 403),
        ({**AUTH, "origin": "http://127.0.0.1:9999"}, 403),
        ({**AUTH, "sec-fetch-site": "cross-site"}, 403),
        ({**WRITE, "host": "evil.example"}, 421),
    ]
    for headers, status in refused:
        assert client.post(DERIVE, json=body, headers=headers).status_code == status
    # A tab signed in by its cookie says where a change comes from, and for which project.
    client.get(f"/?token={TOKEN}", follow_redirects=False)
    assert client.post(DERIVE, json=body).status_code == 403
    assert client.post(DERIVE, json=body, headers=ORIGIN).status_code == 409
    stale = {**ORIGIN, "x-logogram-project": "an-earlier-project"}
    assert client.post(DERIVE, json=body, headers=stale).status_code == 409
    assert _files(app) == ["src.jsonl"]
    tab = {**ORIGIN, "x-logogram-project": app.state.logogram.project.session_id}
    assert client.post(DERIVE, json=body, headers=tab).json()["n"] == 3
    # With no project open there is nothing to edit.
    client.post("/api/projects/close", headers=tab)
    closed = {**ORIGIN, "x-logogram-project": "none"}
    r = client.post(DERIVE, json={**body, "name": "again"}, headers=closed)
    assert r.status_code == 409 and r.json()["error"] == "Open a project first."


def test_how_the_walkthrough_ended_is_kept_in_the_settings(client, tmp_path):
    headers = {**AUTH, **ORIGIN}
    assert client.get("/api/state", headers=AUTH).json()["walkthrough"] is None
    for ended in ("skipped", "finished"):
        settings = client.post("/api/settings", json={"walkthrough": ended}, headers=headers)
        assert settings.json()["walkthrough"] == ended
        assert client.get("/api/state", headers=AUTH).json()["walkthrough"] == ended
    # Only how it ended can be said, and it is kept with the user's settings, not in a project.
    refused = client.post("/api/settings", json={"walkthrough": "maybe"}, headers=headers)
    assert refused.status_code == 422
    settings_file = tmp_path / "config" / "settings.json"
    saved = json.loads(settings_file.read_text(encoding="utf-8"))
    assert saved["walkthrough"] == "finished"
    # A settings file edited into something else reads as never ended.
    saved["walkthrough"] = ["finished"]
    settings_file.write_text(json.dumps(saved), encoding="utf-8")
    assert client.get("/api/state", headers=AUTH).json()["walkthrough"] is None
