"""Training from the command line (a spec, or options that state every choice) and from Python."""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from logogram.cli import app as cli
from logogram.corpora import CorpusSource, TextRecord, write_corpus
from logogram.ioi import generate_ioi
from logogram.project import Project

MODEL = {
    "id": "tiny-gpt2",
    "source": "hub",
    "revision": "test",
    "sha256": None,
    "dtype": "float32",
    "device": "cpu",
    "process_weights": True,
}
SAE: dict[str, Any] = {
    "logogram_sae_training": 1,
    "name": "cli-sae",
    "model": MODEL,
    "corpus": {"name": "ioi-texts", "sha256": None},
    "tokenization": {"prepend_bos": True, "max_tokens": 32},
    "site": "resid_post",
    "layer": 0,
    "architecture": {"kind": "topk", "k": 4},
    "d_sae": 64,
    "steps": 10,
    "batch_tokens": 64,
    "lr": 0.01,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "seed": 0,
    "held_out": 0.25,
    "normalize": "none",
    "auxk": None,
    "log_every": 5,
    "model_batch": 8,
}
SAE_OPTIONS = [
    "--model", "tiny-gpt2", "--revision", "test", "--dtype", "float32", "--device", "cpu",
    "--process-weights", "--corpus", "ioi-texts", "--site", "resid_post:0", "--d-sae", "64",
    "--k", "4", "--steps", "10", "--batch-tokens", "64", "--lr", "0.01", "--betas", "0.9,0.999",
    "--eps", "1e-8", "--seed", "0", "--held-out", "0.25", "--normalize", "none", "--auxk", "none",
    "--log-every", "5", "--prepend-bos", "--max-tokens", "32", "--model-batch", "8",
]  # fmt: skip
LENS: dict[str, Any] = {
    **{k: SAE[k] for k in ("model", "corpus", "tokenization", "steps", "batch_tokens", "lr")},
    **{k: SAE[k] for k in ("betas", "eps", "seed", "held_out", "log_every", "model_batch")},
    "logogram_lens_training": 1,
    "name": "cli-lens",
    "layers": None,
    "weight_decay": None,
}


@pytest.fixture
def cli_project(tiny_backend, project: Project, monkeypatch) -> Project:
    monkeypatch.setattr("logogram.project.config_dir", lambda: project.root / "config")
    monkeypatch.setattr(
        "logogram.runner.default_provider", lambda on_event=None: lambda ref: tiny_backend
    )
    records = generate_ioi(30, seed=0)
    texts = [TextRecord(text=t) for r in records for t in (r.clean, r.corrupt)]
    write_corpus(project, "ioi-texts", texts, CorpusSource(kind="texts", detail="IOI"))
    return project


def test_an_sae_trains_from_a_spec_or_from_options_alike(cli_project):
    root = cli_project.root
    path = root / "sae.json"
    path.write_text(json.dumps(SAE), encoding="utf-8")
    result = CliRunner().invoke(cli, ["sae", "train", str(path)])
    assert result.exit_code == 0, result.output
    assert "Saved saes/cli-sae" in result.output and 'Specs name it "sae"' in result.output
    options = ["sae", "train", "--name", "opt-sae", *SAE_OPTIONS, "--project", str(root)]
    result = CliRunner().invoke(cli, options)
    assert result.exit_code == 0, result.output
    # The same choices, the same weights.
    for name in ("sae_weights.safetensors", "cfg.json"):
        assert (root / "saes/cli-sae" / name).read_bytes() == (
            root / "saes/opt-sae" / name
        ).read_bytes()
    listed = CliRunner().invoke(cli, ["sae", "list", "--project", str(root)])
    assert listed.exit_code == 0 and "cli-sae" in listed.output and "opt-sae" in listed.output


def test_a_training_states_every_choice(cli_project):
    root = cli_project.root
    result = CliRunner().invoke(
        cli, ["sae", "train", "--name", "x", "--corpus", "ioi-texts", "--project", str(root)]
    )
    assert result.exit_code == 1
    assert "these options too" in result.output and "--d-sae" in result.output
    path = root / "sae.json"
    path.write_text(json.dumps(SAE), encoding="utf-8")
    both = CliRunner().invoke(cli, ["sae", "train", str(path), "--steps", "5"])
    assert both.exit_code == 1 and "not both" in both.output
    no_layer = ["resid_post" if o == "resid_post:0" else o for o in SAE_OPTIONS]
    bad = CliRunner().invoke(
        cli, ["sae", "train", "--name", "y", *no_layer, "--project", str(root)]
    )
    assert bad.exit_code == 1 and "SITE:LAYER" in bad.output


def test_a_lens_trains_from_the_command_line(cli_project):
    root = cli_project.root
    path = root / "lens.json"
    path.write_text(json.dumps(LENS), encoding="utf-8")
    result = CliRunner().invoke(cli, ["lens", "train", str(path)])
    assert result.exit_code == 0, result.output
    assert "Saved lenses/cli-lens" in result.output and "layer   0" in result.output
    listed = CliRunner().invoke(cli, ["lens", "list", "--project", str(root)])
    assert listed.exit_code == 0 and "cli-lens" in listed.output
    assert "with the logit lens" in listed.output and "with this one" in listed.output


def test_training_from_python(cli_project, tiny_backend):
    import logogram as lg

    trained = lg.train_sae(cli_project, {**SAE, "name": "py-sae"}, model=tiny_backend)
    assert trained["path"] == "saes/py-sae" and trained["metrics"]["held_out"]["tokens"] > 0
    sae = lg.load_sae("py-sae", project=cli_project, model=tiny_backend)
    assert sae.sha256 == trained["sha256"] and sae.source == "project"
    with pytest.raises(lg.api.LogogramError, match="no SAE called nope"):
        lg.load_sae("nope", project=cli_project, model=tiny_backend)
    lens = lg.train_lens(cli_project, json.dumps({**LENS, "name": "py-lens"}), model=tiny_backend)
    assert lens["metrics"]["held_out"]["layers"][0]["tuned_lens"] is not None
    with pytest.raises(lg.api.LogogramError, match="already exists"):
        lg.train_lens(cli_project, {**LENS, "name": "py-lens"}, model=tiny_backend)
