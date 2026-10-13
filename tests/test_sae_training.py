"""SAEs trained in the project: training reduces the reconstruction error and is bit-identical
twice; the saved SAE reads back through the published-SAE reader and reproduces its measurements;
a run on its features pins its hash; a cancelled training writes nothing."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest
import torch

from logogram.backends.base import Cancelled
from logogram.backends.saes import read_sae
from logogram.corpora import CorpusSource, TextRecord, write_corpus
from logogram.ioi import generate_ioi
from logogram.project import Project
from logogram.sae import SAE
from logogram.sae_training import (
    SAETrainingSpec,
    evaluate_sae,
    files_sha256,
    list_project_saes,
    load_project_sae,
    plan_sae,
    train_sae,
)
from logogram.spec import SAERef, Spec
from logogram.training import TrainingError, read_texts

MODEL = {
    "id": "tiny-gpt2",
    "source": "hub",
    "revision": "test",
    "sha256": None,
    "dtype": "float32",
    "device": "cpu",
    "process_weights": True,
}


def corpus_texts(n: int = 30, seed: int = 0) -> list[TextRecord]:
    """Texts the tiny model's tokenizer reads word by word: IOI prompts, clean and corrupt."""
    records = generate_ioi(n, seed=seed)
    return [TextRecord(text=t) for r in records for t in (r.clean, r.corrupt)]


@pytest.fixture
def trained_project(project: Project) -> Project:
    write_corpus(project, "ioi-texts", corpus_texts(), CorpusSource(kind="texts", detail="IOI"))
    return project


def sae_spec(**changes: Any) -> SAETrainingSpec:
    data: dict[str, Any] = {
        "logogram_sae_training": 1,
        "name": "tiny-sae",
        "model": MODEL,
        "corpus": {"name": "ioi-texts", "sha256": None},
        "tokenization": {"prepend_bos": True, "max_tokens": 32},
        "site": "resid_post",
        "layer": 0,
        "architecture": {"kind": "topk", "k": 4},
        "d_sae": 64,
        "steps": 60,
        "batch_tokens": 64,
        "lr": 0.01,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "seed": 0,
        "held_out": 0.25,
        "normalize": "mean_squared_norm",
        "auxk": {"k": 16, "coefficient": 0.03125, "dead_after": 256},
        "log_every": 10,
        "model_batch": 8,
    }
    data.update(changes)
    return SAETrainingSpec.model_validate(data)


def test_training_reduces_the_error_and_is_bit_identical(tiny_backend, trained_project):
    first = train_sae(sae_spec(), trained_project, backend=tiny_backend)
    again = train_sae(sae_spec(name="tiny-sae-2"), trained_project, backend=tiny_backend)
    curve = first.metrics["curve"]
    assert curve[-1]["fvu"] < 0.5 * curve[0]["fvu"]
    assert all(point["l0"] <= 4 for point in curve)
    root = trained_project.root
    for name in ("sae_weights.safetensors", "cfg.json"):
        assert (root / "saes/tiny-sae" / name).read_bytes() == (
            root / "saes/tiny-sae-2" / name
        ).read_bytes()
    assert first.sha256 == again.sha256
    assert first.metrics["held_out"] == again.metrics["held_out"]
    assert torch.are_deterministic_algorithms_enabled()
    # The spec as it ran: the corpus's hash and the model's revision pinned.
    ran = json.loads((root / "saes/tiny-sae/training.json").read_text(encoding="utf-8"))
    assert ran["corpus"]["sha256"] is not None and ran["model"]["revision"] == "test"
    assert SAETrainingSpec.model_validate(ran).corpus.sha256 == ran["corpus"]["sha256"]


def test_the_saved_sae_reads_back_and_reproduces_its_measurements(tiny_backend, trained_project):
    trained = train_sae(sae_spec(), trained_project, backend=tiny_backend)
    folder = trained_project.root / "saes/tiny-sae"
    params = read_sae(folder)
    assert (params.site, params.layer, params.activation, params.k) == ("resid_post", 0, "topk", 4)
    assert (params.normalize, params.subtract_b_dec, params.format) == ("none", True, "saelens")
    cfg = (folder / "cfg.json").read_bytes()
    weights = (folder / "sae_weights.safetensors").read_bytes()
    assert files_sha256(cfg, weights) == trained.sha256

    ref = SAERef(
        repo="tiny-sae", path="saes/tiny-sae", revision=None, source="project", sha256=None
    )
    loaded = load_project_sae(trained_project, ref, "cpu", tiny_backend.info)
    assert loaded.sha256 == trained.sha256 and loaded.warnings == []
    texts = read_texts(
        trained_project,
        tiny_backend,
        trained.spec.corpus,
        trained.spec.tokenization,
        trained.spec.held_out,
        trained.spec.seed,
    )
    for sae in (loaded, SAE(repo="x", path="", revision="r", params=params)):
        again = evaluate_sae(tiny_backend, sae, texts.held_out, texts.first, 8)
        assert again == trained.metrics["held_out"]
    held = trained.metrics["held_out"]
    assert 0 < held["variance_explained"] <= 1
    # The tiny model is random, so only the definition is checked here, not a good SAE's values.
    loss = held["loss"]
    assert loss["recovered"] == pytest.approx(
        (loss["zero"] - loss["spliced"]) / (loss["zero"] - loss["clean"])
    )

    listed = list_project_saes(trained_project)
    assert [(s["name"], s["sha256"], s["error"]) for s in listed] == [
        ("tiny-sae", trained.sha256, None)
    ]


def test_a_changed_sae_is_refused_when_pinned(tiny_backend, trained_project):
    trained = train_sae(sae_spec(), trained_project, backend=tiny_backend)
    ref = SAERef(
        repo="tiny-sae",
        path="saes/tiny-sae",
        revision=None,
        source="project",
        sha256=trained.sha256,
    )
    cfg = trained_project.root / "saes/tiny-sae/cfg.json"
    cfg.write_text(cfg.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(TrainingError, match="has changed"):
        load_project_sae(trained_project, ref, "cpu")


def test_a_cancelled_training_writes_nothing(tiny_backend, trained_project):
    cancel = threading.Event()
    seen: list[str] = []

    def progress(data: dict[str, Any]) -> None:
        seen.append(data["stage"])
        if data["stage"] == "training" and data["done"] >= 20:
            cancel.set()

    with pytest.raises(Cancelled):
        train_sae(
            sae_spec(), trained_project, backend=tiny_backend, on_progress=progress, cancel=cancel
        )
    assert "training" in seen
    saes = trained_project.root / "saes"
    assert not saes.exists() or list(saes.iterdir()) == []


def test_names_are_never_replaced(tiny_backend, trained_project):
    train_sae(sae_spec(steps=10), trained_project, backend=tiny_backend)
    with pytest.raises(TrainingError, match="already exists"):
        train_sae(sae_spec(steps=10), trained_project, backend=tiny_backend)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"name": "../x"}, "can't name"),
        ({"architecture": {"kind": "topk", "k": 65}}, "more than d_sae"),
        ({"auxk": {"k": 100, "coefficient": 0.1, "dead_after": 10}}, "more than d_sae"),
        ({"held_out": 0.0}, "greater than 0"),
        ({"site": "head"}, "Input should be"),
    ],
)
def test_training_specs_are_checked(change, message):
    with pytest.raises(ValueError, match=message):
        sae_spec(**change)


def test_training_specs_state_every_choice():
    full = sae_spec().model_dump(mode="json")
    for key in full:
        if key == "notes":
            continue
        data = dict(full)
        del data[key]
        with pytest.raises(ValueError, match="Field required"):
            SAETrainingSpec.model_validate(data)


def test_a_corpus_too_small_for_a_batch_is_refused(tiny_backend, trained_project):
    with pytest.raises(TrainingError, match="fewer than one batch"):
        train_sae(sae_spec(batch_tokens=100_000, steps=1), trained_project, backend=tiny_backend)


def test_the_plan_says_what_training_takes(tiny_backend, trained_project):
    plan = plan_sae(sae_spec(), trained_project, tiny_backend)
    assert plan["refusal"] is None
    assert plan["tokens"]["used"] > 0 and plan["texts"]["held_out"] > 0
    assert plan["memory"]["activations"] == plan["tokens"]["used"] * 32 * 4
    assert plan["seconds"]["total"] > 0
    refused = plan_sae(sae_spec(layer=9), trained_project, tiny_backend)
    assert "no layer 9" in refused["refusal"]


def project_sae_spec(
    spec_factory: Any, sha256: str | None, scope: dict[str, Any], experiment: dict[str, Any]
) -> Spec:
    sae = {
        "repo": "tiny-sae",
        "path": "saes/tiny-sae",
        "revision": None,
        "source": "project",
        "sha256": sha256,
    }
    return spec_factory(experiment=experiment, scope=scope, sae=sae)


def test_a_run_on_a_project_sae_pins_its_hash(tiny_backend, trained_project, spec_factory):
    from logogram.runner import run_spec

    trained = train_sae(sae_spec(layer=1, site="resid_pre"), trained_project, backend=tiny_backend)
    spec = project_sae_spec(
        spec_factory,
        None,
        {"kind": "features", "position": {"kind": "last"}, "top": 4},
        {"kind": "attribution_patching", "direction": "corrupt_to_clean"},
    )
    outcome = run_spec(spec, trained_project, backend=tiny_backend)
    assert outcome.status == "finished", outcome.manifest.get("error")
    saved = json.loads((outcome.folder / "spec.json").read_text(encoding="utf-8"))
    assert saved["sae"]["source"] == "project" and saved["sae"]["sha256"] == trained.sha256
    assert outcome.manifest["sae"]["sha256"] == trained.sha256
    summary = json.loads((outcome.folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["features"]["sae"]["source"] == "project"

    wrong = project_sae_spec(
        spec_factory,
        "0" * 64,
        {"kind": "features", "position": {"kind": "last"}, "top": 4},
        {"kind": "attribution_patching", "direction": "corrupt_to_clean"},
    )
    failed = run_spec(wrong, trained_project, backend=tiny_backend)
    assert failed.status == "failed" and "has changed" in failed.manifest["error"]


def feature_firing_at_the_last_token(backend: Any, project: Project, sha256: str) -> int:
    """The project SAE's feature that fires at the last token of the most prompts of
    datasets/ioi.jsonl, clean and corrupt. Which feature that is depends on the tiny model, which
    other versions of PyTorch and Transformers build differently from the same seed. It fires on
    more than half of them, so on some prompt of the pairs steering trains on, whatever the
    split."""
    from logogram.representations import DatasetRows, LastPosition, ReadSite, collect, load_rows

    ref = SAERef(
        repo="tiny-sae", path="saes/tiny-sae", revision=None, source="project", sha256=sha256
    )
    sae = load_project_sae(project, ref, "cpu", backend.info)
    rows = load_rows(backend, project, DatasetRows(path="datasets/ioi.jsonl", sides="both"), True)
    site = ReadSite(kind=sae.site_in, layer=sae.layer)
    values = collect(backend, rows, [site], LastPosition(), prepend_bos=True, batch_size=8)
    features, _ = sae.encode(torch.as_tensor(values[site.name], dtype=torch.float32))
    fired = (features > 0).sum(dim=0)
    best = int(torch.argmax(fired))
    assert int(fired[best]) > len(rows.rows) // 2
    return best


def test_steering_along_a_project_sae_feature(tiny_backend, trained_project, spec_factory):
    from logogram.runner import run_spec

    trained = train_sae(sae_spec(layer=1, site="resid_pre"), trained_project, backend=tiny_backend)
    feature = feature_firing_at_the_last_token(tiny_backend, trained_project, trained.sha256)
    steering = {
        "kind": "steering",
        "apply_to": "corrupt",
        "direction": {"kind": "sae_feature", "feature": feature},
        "coefficients": [1.0],
        "train_fraction": 0.5,
        "seed": 0,
        "control": False,
        "continuations": None,
    }
    site = {"kind": "resid_pre", "layer": 1, "position": {"kind": "last"}}
    spec = project_sae_spec(
        spec_factory, trained.sha256, {"kind": "sites", "sites": [site]}, steering
    )
    outcome = run_spec(spec, trained_project, backend=tiny_backend)
    assert outcome.status == "finished", outcome.manifest.get("error")
    summary = json.loads((outcome.folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["steering"]["feature"]["sae"]["sha256"] == trained.sha256


def test_the_files_name_no_machine(tiny_backend, trained_project):
    train_sae(sae_spec(steps=10), trained_project, backend=tiny_backend)
    for path in Path(trained_project.root / "saes/tiny-sae").iterdir():
        if path.suffix == ".json":
            text = path.read_text(encoding="utf-8")
            assert str(trained_project.root) not in text and str(Path.home()) not in text
