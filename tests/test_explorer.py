"""Scientific and project-boundary regressions for the model explorer."""

from __future__ import annotations

import json
import sys

import pytest
import torch
from fastapi.testclient import TestClient

from logogram.analysis import prediction_report, prepare_with_issues
from logogram.datasets import load_dataset
from logogram.project import ProjectError
from logogram.prompts import group_by_length
from logogram.research import NoteConflict, NoteInput, delete_note, read_notebook, save_note
from logogram.server.app import create_app
from logogram.server.security import SecurityConfig
from logogram.spec import PredictionSettings


def prediction_settings(which="clean", position=None, index=3):
    return PredictionSettings(
        method="final_norm_logit_lens",
        which=which,
        prompt_index=index,
        position=position or {"kind": "last"},
        top_k=5,
    )


@pytest.mark.parametrize("bos", [False, True])
@pytest.mark.parametrize("which", ["clean", "corrupt"])
def test_prediction_last_layer_matches_actual_model_and_repeats(tiny_backend, project, bos, which):
    records = load_dataset(project.datasets_dir / "ioi.jsonl")[:7]
    prepared, _ = prepare_with_issues(tiny_backend, records, bos)
    group = next(g for g in group_by_length(prepared) if 3 in g.members)
    within = group.members.index(3)
    start = (within // 2) * 2
    tokens = group.clean if which == "clean" else group.corrupt
    batch = tokens[start : start + 2]
    settings = prediction_settings(which)
    report = prediction_report(
        tiny_backend,
        records,
        index=3,
        settings=settings,
        prepend_bos=bos,
        batch_size=2,
    )
    again = prediction_report(
        tiny_backend,
        records,
        index=3,
        settings=settings,
        prepend_bos=bos,
        batch_size=2,
    )
    assert report == again
    assert report["batch_members"] == [prepared[i].index for i in group.members[start : start + 2]]
    actual = tiny_backend.final_logits(batch)[within - start].cpu()
    row = report["layers"][-1]
    target = prepared[3]
    assert row["logit_diff"] == pytest.approx(
        float(actual[target.answer_id] - actual[target.distractor_id]),
        abs=1e-6,
    )
    probs = torch.softmax(actual, -1)
    assert row["answer_prob"] == pytest.approx(float(probs[target.answer_id]), abs=1e-7)
    assert [t["id"] for t in row["top"]] == torch.argsort(actual, descending=True, stable=True)[
        :5
    ].tolist()
    assert tiny_backend.info.extra["block_structure"] == "sequential_pre_norm"


def test_prediction_checks_positions_and_recomputes_each_layers_norm(tiny_backend, project):
    records = load_dataset(project.datasets_dir / "ioi.jsonl")[:2]
    prepared, _ = prepare_with_issues(tiny_backend, records, False)
    tokens = group_by_length(prepared)[0].clean
    captured = tiny_backend.capture(
        tokens, [("resid_post", layer) for layer in range(tiny_backend.info.n_layers)]
    )
    lens = tiny_backend.layer_logits(tokens, 2, 0)
    with tiny_backend.lock, torch.no_grad():
        for layer in range(tiny_backend.info.n_layers):
            residual = captured[("resid_post", layer)][:, 2:3]
            expected = (
                tiny_backend.bridge.unembed(tiny_backend.bridge.ln_final(residual))[0, 0]
                .float()
                .cpu()
            )
            assert torch.equal(lens[layer], expected)
    with pytest.raises(ValueError, match="outside"):
        prediction_report(
            tiny_backend,
            records,
            index=0,
            settings=prediction_settings(position={"kind": "index", "index": 999}, index=0),
            prepend_bos=False,
            batch_size=2,
        )


def note_value():
    return NoteInput(
        title="Candidate heads",
        body="Compare across prompt templates.",
        model={"id": "tiny-gpt2", "revision": "test", "device": "cpu"},
        sites=[{"kind": "head", "layer": 0, "head": 1, "position": {"kind": "last"}}],
    )


def test_research_notes_round_trip_and_conflicting_edits(project):
    value = note_value()
    note = save_note(project, value)
    assert read_notebook(project).notes == [note]
    changed = save_note(
        project, value.model_copy(update={"body": "Updated evidence."}), note_id=note.id, revision=1
    )
    assert changed.revision == 2
    with pytest.raises(NoteConflict):
        save_note(project, value, note_id=note.id, revision=1)
    with pytest.raises(NoteConflict):
        delete_note(project, note.id, 1)
    assert read_notebook(project).notes[0].body == "Updated evidence."
    delete_note(project, note.id, 2)
    assert read_notebook(project).notes == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_research_never_reads_external_note_files(project, tmp_path):
    external = tmp_path / "outside.json"
    external.write_text('{"logogram_research":1,"notes":[]}')
    (project.root / "research.json").symlink_to(external)
    with pytest.raises(ProjectError, match="outside"):
        read_notebook(project)
    with pytest.raises(ProjectError, match="outside"):
        save_note(project, note_value())


def test_new_routes_require_session_and_preserve_prediction_context(
    project, tiny_backend, tmp_path, monkeypatch
):
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    app = create_app(SecurityConfig(token="test", port=8765), serve_web=False)
    state = app.state.logogram
    state.open_project(project)
    state.backend = tiny_backend
    auth = {
        "authorization": "Bearer test",
        "origin": "http://127.0.0.1:8765",
        "x-logogram-project": project.session_id,
    }
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        assert client.get("/api/research").status_code == 401
        assert (
            client.post(
                "/api/research",
                json=note_value().model_dump(),
                headers={**auth, "x-logogram-project": "stale"},
            ).status_code
            == 409
        )
        created = client.post("/api/research", json=note_value().model_dump(), headers=auth)
        assert created.status_code == 200, created.text
        body = {
            "dataset": "datasets/ioi.jsonl",
            "limit": 4,
            "prepend_bos": False,
            "batch_size": 2,
            "index": 3,
            "settings": prediction_settings().model_dump(),
        }
        response = client.post("/api/predictions", json=body, headers=auth)
        assert response.status_code == 200, response.text
        assert response.json()["position"] == len(response.json()["tokens"]) - 1
        assert 3 in response.json()["batch_members"]
        assert (
            client.post("/api/predictions", json={**body, "index": 4}, headers=auth).status_code
            == 400
        )
        assert (
            client.post(
                "/api/predictions", json={**body, "model": {"id": "other"}}, headers=auth
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/api/predictions", json={**body, "dataset_sha256": "changed"}, headers=auth
            ).status_code
            == 409
        )
    assert "research.json" in [p.name for p in project.root.iterdir()]
    assert (
        json.loads((project.root / "research.json").read_text())["notes"][0]["model"]["id"]
        == "tiny-gpt2"
    )


def test_saved_spec_runs_the_same_prediction_diagnostic(project, tiny_backend, spec_factory):
    from logogram.runner import run_spec
    from logogram.spec import Spec

    settings = prediction_settings(index=2, which="corrupt")
    spec = spec_factory(
        predictions=settings.model_dump(),
        tokenization={"prepend_bos": False},
        dataset={"path": "datasets/ioi.jsonl", "limit": 6},
        execution={"batch_size": 2},
    )
    assert Spec.model_validate_json(spec.to_json()).predictions == settings
    records = load_dataset(project.datasets_dir / "ioi.jsonl")[:6]
    preview = prediction_report(
        tiny_backend, records, index=2, settings=settings, prepend_bos=False, batch_size=2
    )
    first = run_spec(spec, project, backend=tiny_backend)
    second = run_spec(spec, project, backend=tiny_backend)
    assert first.status == second.status == "finished"
    assert project.read_json(first.folder / "predictions.json") == preview
    assert project.read_json(second.folder / "predictions.json") == preview
    shape = first.summary["model"]
    assert shape["d_mlp"] == tiny_backend.info.d_mlp
    assert shape["block_structure"] == "sequential_pre_norm"
    assert "resid_post" in shape["site_kinds"]


def test_prediction_requires_all_method_settings_and_consistent_prompt(tiny_backend, project):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PredictionSettings(
            method="final_norm_logit_lens", which="clean", position={"kind": "last"}, top_k=5
        )
    with pytest.raises(ValueError, match="must match"):
        prediction_report(
            tiny_backend,
            load_dataset(project.datasets_dir / "ioi.jsonl"),
            index=0,
            settings=prediction_settings(index=3),
            prepend_bos=True,
            batch_size=2,
        )


def test_note_edit_api_conflict_keeps_the_latest_text(project, tmp_path, monkeypatch):
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    app = create_app(SecurityConfig(token="test", port=8765), serve_web=False)
    app.state.logogram.open_project(project)
    auth = {
        "authorization": "Bearer test",
        "origin": "http://127.0.0.1:8765",
        "x-logogram-project": project.session_id,
    }
    with TestClient(app, base_url="http://127.0.0.1:8765", headers=auth) as client:
        note = client.post("/api/research", json=note_value().model_dump()).json()
        url = f"/api/research/{note['id']}"
        body = {**note_value().model_dump(), "revision": 1, "body": "New evidence"}
        assert client.put(url, json=body).status_code == 200
        assert client.put(url, json={**body, "body": "Old tab"}).status_code == 409
        assert client.get("/api/research").json()["notes"][0]["body"] == "New evidence"
        assert client.delete(url + "?revision=1").status_code == 409
        assert client.delete(url + "?revision=2").status_code == 200


def test_prediction_uses_original_weight_norm_and_biases(tiny_model_dir, project):
    from logogram.backends.transformer_lens import TransformerLensBackend, boot_local

    backend = TransformerLensBackend.from_bridge(
        boot_local(tiny_model_dir, device="cpu", dtype="float32"),
        model_id="tiny-gpt2",
        revision="test",
        dtype="float32",
        process_weights=False,
    )
    try:
        records = load_dataset(project.datasets_dir / "ioi.jsonl")[:2]
        prepared, _ = prepare_with_issues(backend, records, False)
        batch = group_by_length(prepared)[0].corrupt
        report = prediction_report(
            backend,
            records,
            index=0,
            settings=prediction_settings(which="corrupt", index=0),
            prepend_bos=False,
            batch_size=2,
        )
        actual = backend.final_logits(batch)[0].cpu()
        probs = torch.softmax(actual, -1)
        assert report["layers"][-1]["answer_prob"] == pytest.approx(
            float(probs[prepared[0].answer_id]), abs=1e-7
        )
        assert report == prediction_report(
            backend,
            records,
            index=0,
            settings=prediction_settings(which="corrupt", index=0),
            prepend_bos=False,
            batch_size=2,
        )
    finally:
        backend.close()
