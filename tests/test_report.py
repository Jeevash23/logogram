"""A run as words and as one HTML page: the methods paragraph, the report, the CSV tables, and
the endpoints and commands that give them."""

from __future__ import annotations

import csv
import io
import re

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import logogram as lg
from logogram.cli import app as cli
from logogram.datasets import PromptRecord
from logogram.exports import results_csv, sites_csv
from logogram.report import methods_text, sites_table
from logogram.server.app import create_app
from logogram.server.security import SecurityConfig


@pytest.fixture
def finished(tiny_backend, project, spec_factory):
    spec = spec_factory(name="A <b>bold</b> run", notes="Heads in layer 1 should carry it.")
    return lg.run(spec, project, model=tiny_backend)


def test_the_methods_paragraph_says_every_choice_in_words(finished):
    text = finished.methods()
    assert not re.search(r"\btrue\b|\bfalse\b|[{}]|\bNone\b|undefined", text)
    assert "read from datasets/ioi.jsonl (SHA-256 " in text  # the source, not the snapshot
    assert "tokenized with a beginning-of-sequence token" in text
    assert "copied from the clean prompt into the corrupt one" in text
    assert "95% percentile bootstrap intervals over prompts, from 200 resamples with seed 0" in text
    assert f"The run is {finished.id}." in text
    assert text == methods_text(finished.spec, finished.summary, finished.manifest, finished.id)


def test_an_ablation_names_its_baseline(tiny_backend, project, spec_factory):
    spec = spec_factory(
        experiment={"kind": "ablation", "baseline": {"kind": "mean", "reference": "clean"}}
    )
    text = methods_text(spec, None, None, "draft")
    assert "replaced by the activation's mean over the clean prompts" in text
    assert "The prompt pairs were read from datasets/ioi.jsonl (not pinned by a checksum)" in text


def test_the_report_stands_alone_escapes_text_and_leaves_hardware_out(finished):
    page = finished.report()
    assert page.startswith("<!doctype html>") and page.rstrip().endswith("</html>")
    # Nothing to fetch and nothing to run.
    # (The SVG namespace is a name, not an address that is fetched.)
    fetched = page.replace('xmlns="http://www.w3.org/2000/svg"', "")
    assert not re.search(r"<script|<link|<img|<iframe|\bsrc=|https?://", fetched, re.I)
    assert "A &lt;b&gt;bold&lt;/b&gt; run" in page and "<b>bold</b>" not in page
    assert "Heads in layer 1 should carry it." in page
    strongest = finished.strongest(1)[0]["label"]
    assert strongest in page and "<svg" in page
    name = finished.manifest["device"]["name"]
    assert name not in page.split("<details><summary>The spec</summary>")[0]
    assert f"<dd>{name}</dd>" in finished.report(hardware=True)
    for paragraph in finished.methods().split("\n\n")[:3]:
        assert paragraph.replace("'", "&#x27;") in page


def test_csv_tables_carry_prompt_text_and_guard_formulas(finished):
    from logogram.runs import run_records

    records = run_records(finished.project, finished.spec)
    assert records is not None and len(records) == finished.summary["n_prompts"]
    rows = list(
        csv.DictReader(
            io.StringIO("".join(results_csv(finished.folder / "results.parquet", records)))
        )
    )
    first = rows[0]
    assert first["clean"] == records[int(first["prompt"])].clean
    assert {"answer", "distractor", "corrupt"} <= set(first)
    # A cell a spreadsheet would run as a formula is defused.
    formula = PromptRecord(clean="=SUM(A1)", corrupt="x", answer=" a", distractor=" b")
    rows = list(
        csv.DictReader(
            io.StringIO(
                "".join(results_csv(finished.folder / "results.parquet", [formula] * len(records)))
            )
        )
    )
    assert rows[0]["clean"] == "'=SUM(A1)"
    table = list(csv.DictReader(io.StringIO(sites_csv(sites_table(finished.summary)))))
    assert len(table) == len(finished.summary["sites"]) and "band_lo" in table[0]


def test_endpoints_and_commands(tmp_path, monkeypatch, finished, tiny_backend):
    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    server = create_app(SecurityConfig(token="t", port=8765), serve_web=False)
    server.state.logogram.backend = tiny_backend
    server.state.logogram.open_project(finished.project)
    client = TestClient(server, base_url="http://127.0.0.1:8765")
    headers = {"authorization": "Bearer t", "x-logogram-project": finished.project.session_id}
    base = f"/api/runs/{finished.id}"
    page = client.get(f"{base}/report.html", headers=headers)
    assert page.status_code == 200 and page.text == finished.report()
    assert "attachment" in page.headers["content-disposition"]
    assert client.get(f"{base}/methods", headers=headers).json()["text"] == finished.methods()
    sites = client.get(f"{base}/sites.csv", headers=headers)
    assert sites.status_code == 200 and sites.text == sites_csv(sites_table(finished.summary))
    with_text = client.get(f"{base}/export.csv", params={"text": True}, headers=headers)
    assert with_text.status_code == 200 and with_text.text.splitlines()[0].endswith(
        "clean,corrupt,answer,distractor"
    )
    without = client.get(f"{base}/export.csv", headers=headers)
    assert not without.text.splitlines()[0].endswith("distractor")

    runner = CliRunner()
    out = tmp_path / "report.html"
    result = runner.invoke(
        cli, ["report", finished.id, "-o", str(out), "--project", str(finished.project.root)]
    )
    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8") == finished.report()
    result = runner.invoke(cli, ["methods", finished.id, "--project", str(finished.project.root)])
    assert result.exit_code == 0 and finished.methods() in result.output
    result = runner.invoke(
        cli, ["export", finished.id, "--sites", "--project", str(finished.project.root)]
    )
    assert result.exit_code == 0 and result.output.startswith("index,label,kind")


def test_the_figure_is_a_standalone_svg_on_white(finished):
    from xml.etree import ElementTree

    from logogram.report import figure_svg

    svg = figure_svg(finished.spec, finished.summary, "delta", unit=False)
    root = ElementTree.fromstring(svg)  # noqa: S314 -- our own output, checked to be well-formed
    assert root.tag.endswith("svg") and "A &lt;b&gt;bold&lt;/b&gt; run" in svg
    assert "mean change in the metric" in svg
    fixed = figure_svg(finished.spec, finished.summary, "effect", unit=True)
    assert "−1.00" in fixed and "+1.00" in fixed
