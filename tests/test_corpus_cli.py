"""Corpora, directions and exported activations from the command line and from Python."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

import logogram as lg
from logogram.cli import app as cli
from logogram.project import Project


def invoke(project: Project, *args: str, input: str | None = None):  # type: ignore[no-untyped-def]
    return CliRunner().invoke(cli, [*args, "--project", str(project.root)], input=input)


def test_corpora_are_imported_listed_renamed_and_deleted(project: Project, tmp_path: Path) -> None:
    file = tmp_path / "notes.txt"
    file.write_text("Mary went to the store.\n\nJohn went to the park.\n", encoding="utf-8")
    done = invoke(project, "corpora", "import", "notes", str(file))
    assert done.exit_code == 0, done.output
    assert "Wrote 2 texts to corpora/notes.jsonl" in done.output
    pairs = invoke(
        project,
        "corpora",
        "import",
        "pairs",
        "--dataset",
        "datasets/ioi.jsonl",
        "--label",
        "side",
        "--limit",
        "2",
    )
    assert pairs.exit_code == 0, pairs.output
    assert "Wrote 4 texts, labelled clean, corrupt" in pairs.output
    listed = invoke(project, "corpora")
    assert listed.exit_code == 0 and "notes" in listed.output and "4 texts" in listed.output
    assert "notes.txt, one text per line" in listed.output

    both = invoke(project, "corpora", "import", "x", str(file), "--dataset", "datasets/ioi.jsonl")
    assert both.exit_code == 1 and "import from one place" in both.output
    again = invoke(project, "corpora", "import", "notes", str(file))
    assert again.exit_code == 1 and "already exists" in again.output

    assert invoke(project, "corpora", "rename", "notes", "texts").exit_code == 0
    kept = invoke(project, "corpora", "delete", "texts", input="n\n")
    assert kept.exit_code == 1 and "Nothing was deleted" in kept.output
    gone = invoke(project, "corpora", "delete", "texts", "--yes")
    assert gone.exit_code == 0 and [c["name"] for c in lg.list_corpora(project)] == ["pairs"]


def test_directions_are_listed_renamed_and_deleted(project: Project, tiny_backend) -> None:
    lg.save_direction(
        project,
        "toward",
        np.ones(tiny_backend.info.d_model),
        model=tiny_backend,
        site="resid_post:1",
        method="mean_difference",
        detail="a test",
    )
    listed = invoke(project, "directions")
    assert listed.exit_code == 0, listed.output
    assert (
        "toward" in listed.output
        and "1 dimension of 32 · resid_post:1 · tiny-gpt2" in listed.output
    )
    assert invoke(project, "directions", "rename", "toward", "away").exit_code == 0
    assert invoke(project, "directions", "delete", "away", "--yes").exit_code == 0
    empty = invoke(project, "directions")
    assert "No directions yet" in empty.output


def test_activations_are_exported_from_the_command_line(
    project: Project, tiny_backend, monkeypatch: pytest.MonkeyPatch
) -> None:
    from safetensors.numpy import load_file

    monkeypatch.setattr("logogram.api.load_model", lambda *a, **k: tiny_backend)
    done = invoke(
        project,
        "activations",
        "datasets/ioi.jsonl",
        "--site",
        "resid_post:1",
        "--site",
        "mlp_out:0",
        "--position",
        "label:S2",
        "--mark",
        "meta:pattern",
        "--limit",
        "3",
        "-o",
        "s2",
        "--model",
        "tiny-gpt2",
    )
    assert done.exit_code == 0, done.output
    assert "Wrote 6 rows of resid_post:1, mlp_out:0 to activations/s2.safetensors" in done.output
    tensors = load_file(project.root / "activations" / "s2.safetensors")
    direct = lg.activations(
        tiny_backend,
        project,
        "datasets/ioi.jsonl",
        ["resid_post:1", "mlp_out:0"],
        "label:S2",
        limit=3,
    )
    for key in ("resid_post:1", "mlp_out:0"):
        assert np.array_equal(tensors[key], direct[key].astype(np.float32))
    meta = json.loads((project.root / "activations" / "s2.json").read_text(encoding="utf-8"))
    assert (
        meta["position"] == {"kind": "label", "label": "S2"} and meta["rows"][0]["label"] == "ABBA"
    )
    bad = invoke(
        project, "activations", "datasets/ioi.jsonl", "--site", "resid:1", "-o", "x", "--model", "m"
    )
    assert bad.exit_code == 1 and "isn't a site" in bad.output


def test_the_python_api_reads_corpora_and_representations(project: Project, tiny_backend) -> None:
    texts = [
        "When Mary and John went to the store, John gave a drink to Mary",
        "When Tom and Kelly went to the park, Kelly gave a ball to Tom",
        "Then, Sarah and Chris went to the school",
        "Afterwards, Jessica and Robert went to the restaurant",
    ]
    corpus = lg.write_corpus(project, "stories", texts, labels=["a", "a", "b", "b"])
    assert corpus.texts == texts and lg.read_corpus(project, "stories").labels() == ["a", "b"]
    assert [c["name"] for c in lg.list_corpora(project)] == ["stories"]

    top = lg.top_examples(
        tiny_backend,
        project,
        "stories",
        {"kind": "neuron", "layer": 0, "neuron": 1},
        top_k=3,
        max_tokens=32,
    )
    assert len(top["top"]) == 3 and top["settings"]["max_tokens"] == 32

    pcs = lg.principal_components(
        tiny_backend, project, "stories", "resid_post:0", "mean", max_tokens=32
    )
    assert pcs["components"].shape == (4, tiny_backend.info.d_model)
    assert pcs["coords"].shape == (4, 4) and [r["mark"] for r in pcs["rows"]] == [
        "a",
        "a",
        "b",
        "b",
    ]

    difference = lg.mean_difference(
        tiny_backend, project, "stories", "resid_post:0", "mean", "a", "b", max_tokens=32
    )
    saved = lg.save_direction(
        project,
        "a-minus-b",
        difference,
        model=tiny_backend,
        site="resid_post:0",
        method="mean_difference",
    )
    read = lg.read_direction(project, "a-minus-b")
    assert torch.equal(read.vectors, saved.vectors)
    assert [d["name"] for d in lg.list_directions(project)] == ["a-minus-b"]
    signed = lg.top_examples(
        tiny_backend,
        project,
        "stories",
        {"kind": "direction", "name": "a-minus-b"},
        top_k=2,
        max_tokens=32,
    )
    assert signed["signed"] and len(signed["bottom"]) == 2

    written = lg.export_activations(
        tiny_backend,
        project,
        "pairs",
        "datasets/ioi.jsonl",
        "resid_pre:1",
        -1,
        limit=2,
        mark="side",
    )
    assert written["n"] == 4 and written["sites"] == ["resid_pre:1"]

    with pytest.raises(lg.api.LogogramError, match="no corpus called nothing"):
        lg.top_examples(
            tiny_backend, project, "nothing", {"kind": "neuron", "layer": 0, "neuron": 0}
        )
    with pytest.raises(lg.api.LogogramError, match="Import from one place"):
        lg.import_corpus(project, "x")
    imported = lg.import_corpus(
        project, "sides", dataset="datasets/ioi.jsonl", label="side", limit=2
    )
    assert imported.labels() == ["clean", "corrupt"]
