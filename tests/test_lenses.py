"""Tuned lenses trained in the project: training brings each layer's readout closer to the model's
own prediction on held-out text than the logit lens, deterministically; the layer predictions read
through it, a run pins it, and a lens of another model is refused."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest
import torch

from logogram.analysis import prediction_report
from logogram.corpora import CorpusSource, TextRecord, write_corpus
from logogram.datasets import load_dataset
from logogram.ioi import generate_ioi
from logogram.lenses import (
    LensTrainingSpec,
    check_lens_fits,
    list_lenses,
    read_lens,
    train_lens,
)
from logogram.project import Project
from logogram.spec import LensRef, PredictionSettings, Spec
from logogram.training import TrainingError

MODEL = {
    "id": "tiny-gpt2",
    "source": "hub",
    "revision": "test",
    "sha256": None,
    "dtype": "float32",
    "device": "cpu",
    "process_weights": True,
}


@pytest.fixture
def lens_project(project: Project) -> Project:
    records = generate_ioi(30, seed=0)
    texts = [TextRecord(text=t) for r in records for t in (r.clean, r.corrupt)]
    write_corpus(project, "ioi-texts", texts, CorpusSource(kind="texts", detail="IOI"))
    return project


def lens_spec(**changes: Any) -> LensTrainingSpec:
    data: dict[str, Any] = {
        "logogram_lens_training": 1,
        "name": "tiny-lens",
        "model": MODEL,
        "corpus": {"name": "ioi-texts", "sha256": None},
        "tokenization": {"prepend_bos": True, "max_tokens": 32},
        "layers": None,
        "steps": 30,
        "batch_tokens": 64,
        "lr": 0.01,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": None,
        "seed": 0,
        "held_out": 0.25,
        "log_every": 5,
        "model_batch": 8,
    }
    data.update(changes)
    return LensTrainingSpec.model_validate(data)


def test_the_tuned_lens_beats_the_logit_lens_on_held_out_text(tiny_backend, lens_project):
    trained = train_lens(lens_spec(), lens_project, backend=tiny_backend)
    layers = trained.metrics["held_out"]["layers"]
    assert [row["layer"] for row in layers] == [0]  # every layer but the last
    assert layers[0]["tuned_lens"] < layers[0]["logit_lens"]
    curve = trained.metrics["curve"]
    assert curve[-1]["kl"] < curve[0]["kl"]
    again = train_lens(lens_spec(name="tiny-lens-2"), lens_project, backend=tiny_backend)
    root = lens_project.root
    assert (root / "lenses/tiny-lens/lens.safetensors").read_bytes() == (
        root / "lenses/tiny-lens-2/lens.safetensors"
    ).read_bytes()
    assert again.sha256 == trained.sha256
    assert again.metrics["held_out"] == trained.metrics["held_out"]
    ran = json.loads((root / "lenses/tiny-lens/training.json").read_text(encoding="utf-8"))
    assert ran["corpus"]["sha256"] is not None and ran["model"]["revision"] == "test"


def test_weight_decay_pulls_towards_the_logit_lens(tiny_backend, lens_project):
    free = train_lens(lens_spec(), lens_project, backend=tiny_backend)
    held = train_lens(
        lens_spec(name="decayed", weight_decay=1.0), lens_project, backend=tiny_backend
    )
    eye = torch.eye(tiny_backend.info.d_model)

    def moved(name: str) -> float:
        lens = read_lens(lens_project, LensRef(name=name, sha256=None))
        weight, _ = lens.translators[0]
        return float((weight - eye).norm())

    assert free.sha256 != held.sha256
    assert moved("decayed") < moved("tiny-lens")


def test_layer_predictions_read_through_the_lens(tiny_backend, lens_project):
    trained = train_lens(lens_spec(), lens_project, backend=tiny_backend)
    records = load_dataset(lens_project.datasets_dir / "ioi.jsonl")
    common = {"prompt_index": 0, "which": "clean", "position": {"kind": "last"}, "top_k": 3}
    logit = PredictionSettings.model_validate(
        {**common, "method": "final_norm_logit_lens", "lens": None}
    )
    tuned = PredictionSettings.model_validate(
        {**common, "method": "tuned_lens", "lens": {"name": "tiny-lens", "sha256": None}}
    )
    lens = read_lens(lens_project, tuned.lens)
    check_lens_fits(lens, tiny_backend.info)
    args = {"index": 0, "prepend_bos": True, "batch_size": 16}
    plain = prediction_report(tiny_backend, records, settings=logit, **args)
    read = prediction_report(tiny_backend, records, settings=tuned, lens=lens, **args)
    assert read["settings"]["lens"]["sha256"] == trained.sha256
    assert [row["readout"] for row in read["layers"]] == ["tuned_lens", "logit_lens"]
    assert read["layers"][1] == plain["layers"][1]  # the last layer is the model's own readout
    assert read["layers"][0]["top"] != plain["layers"][0]["top"] or (
        read["layers"][0]["logit_diff"] != plain["layers"][0]["logit_diff"]
    )
    assert read["lens"]["layers"] == [0]
    with pytest.raises(ValueError, match="reads the lens"):
        prediction_report(tiny_backend, records, settings=tuned, **args)


def test_a_run_pins_the_lens(tiny_backend, lens_project, spec_factory):
    from logogram.runner import run_spec

    trained = train_lens(lens_spec(), lens_project, backend=tiny_backend)
    predictions = {
        "method": "tuned_lens",
        "lens": {"name": "tiny-lens", "sha256": None},
        "prompt_index": 1,
        "which": "clean",
        "position": {"kind": "last"},
        "top_k": 3,
    }
    data = spec_factory().model_dump(mode="json")
    spec = Spec.model_validate({**data, "predictions": predictions})
    outcome = run_spec(spec, lens_project, backend=tiny_backend)
    assert outcome.status == "finished", outcome.manifest.get("error")
    saved = json.loads((outcome.folder / "spec.json").read_text(encoding="utf-8"))
    assert saved["predictions"]["lens"] == {"name": "tiny-lens", "sha256": trained.sha256}
    assert outcome.manifest["lens"]["sha256"] == trained.sha256
    report = json.loads((outcome.folder / "predictions.json").read_text(encoding="utf-8"))
    assert report["layers"][0]["readout"] == "tuned_lens"

    wrong = {**predictions, "lens": {"name": "tiny-lens", "sha256": "1" * 64}}
    failed = run_spec(
        Spec.model_validate({**data, "predictions": wrong}), lens_project, backend=tiny_backend
    )
    assert failed.status == "failed" and "has changed" in failed.manifest["error"]


def test_a_lens_of_another_model_is_refused(tiny_backend, lens_project):
    train_lens(lens_spec(), lens_project, backend=tiny_backend)
    lens = read_lens(lens_project, LensRef(name="tiny-lens", sha256=None))
    check_lens_fits(lens, tiny_backend.info)
    other = replace(tiny_backend.info, revision="other")
    with pytest.raises(TrainingError, match="not on the loaded model"):
        check_lens_fits(lens, other)
    unprocessed = replace(tiny_backend.info, process_weights=False)
    with pytest.raises(TrainingError, match="weight processing on"):
        check_lens_fits(lens, unprocessed)
    wider = replace(tiny_backend.info, d_model=64)
    with pytest.raises(TrainingError, match="32-wide"):
        check_lens_fits(lens, wider)
    listed = list_lenses(lens_project, other)
    assert listed[0]["fits"] is False and "not on the loaded model" in listed[0]["reason"]
    assert list_lenses(lens_project, tiny_backend.info)[0]["fits"] is True


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"layers": [1]}, "layers are 0 to 1"),
        ({"layers": []}, "at least 1 item"),
        ({"layers": [0, 0]}, "must not repeat"),
        ({"weight_decay": 0.0}, "greater than 0"),
    ],
)
def test_lens_specs_are_checked(tiny_backend, lens_project, change, message):
    with pytest.raises((ValueError, TrainingError), match=message):
        train_lens(lens_spec(**change), lens_project, backend=tiny_backend)


def test_lens_specs_state_every_choice():
    full = lens_spec().model_dump(mode="json")
    for key in full:
        if key == "notes":
            continue
        data = dict(full)
        del data[key]
        with pytest.raises(ValueError, match="Field required"):
            LensTrainingSpec.model_validate(data)


def test_prediction_settings_need_a_lens_exactly_for_the_tuned_lens():
    common = {"prompt_index": 0, "which": "clean", "position": {"kind": "last"}, "top_k": 3}
    with pytest.raises(ValueError, match=r"needs predictions\.lens"):
        PredictionSettings.model_validate({**common, "method": "tuned_lens", "lens": None})
    with pytest.raises(ValueError, match="only the tuned lens"):
        PredictionSettings.model_validate(
            {**common, "method": "final_norm_logit_lens", "lens": {"name": "x", "sha256": None}}
        )
