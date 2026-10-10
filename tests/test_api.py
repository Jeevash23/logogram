"""The Python API and the command line's run commands: the same runs as the app makes."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import logogram as lg
from logogram.cli import app

runner = CliRunner()


def _rows(table):
    """A DataFrame or a pyarrow Table as a list of dicts."""
    if hasattr(table, "to_dict") and hasattr(table, "columns"):
        return table.to_dict("records")
    return table.to_pylist()


def test_import_stays_light():
    import subprocess
    import sys

    code = "import logogram, sys; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_runs_from_python(tiny_backend, project, spec_factory):
    run = lg.run(spec_factory(), project, model=tiny_backend)
    assert run.status == "finished" and run.summary is not None
    sites = _rows(run.sites())
    assert len(sites) == tiny_backend.info.n_layers * tiny_backend.info.n_heads
    assert {"effect", "effect_lo", "band_lo", "q", "patched_metric"} <= set(sites[0])
    per_prompt = run.per_prompt()
    assert len(per_prompt) == len(sites) * 12
    assert run.results().num_rows == len(per_prompt)
    detail = run.site(0)
    assert len(detail["prompts"]) == 12
    html = run._repr_html_()
    assert "<svg" in html and run.summary["name"] in html
    assert [r["id"] for r in lg.list_runs(project)] == [run.id]
    again = lg.load_run(project, run.id)
    assert again.summary == run.summary
    assert lg.load_spec(run.folder / "spec.json") == run.spec
    assert lg.load_spec(json.dumps(run.spec.model_dump(mode="json"))) == run.spec


def test_checks_from_python(tiny_backend, project, spec_factory):
    estimate = lg.run(
        spec_factory(
            experiment={
                "kind": "attribution_patching",
                "direction": "clean_to_corrupt",
                "method": "gradient",
                "steps": None,
            }
        ),
        project,
        model=tiny_backend,
    )
    verified = lg.verify_run(estimate, top=3)
    assert verified.manifest["derived_from"]["kind"] == "verification"
    assert len(verified.summary["sites"]) == 3
    comparison = lg.compare_runs(estimate, verified)
    assert comparison["paired"] and comparison["n_common"] == 3
    assert all(c["difference"] is not None for c in comparison["changes"])
    patched = lg.run(spec_factory(), project, model=tiny_backend)
    rerun = lg.check_robustness(
        patched, experiment={"kind": "ablation", "baseline": {"kind": "zero"}}
    )
    assert rerun.manifest["derived_from"]["change"] == "Zero-ablate"
    with pytest.raises(ValueError, match="already ran in float32"):
        lg.check_robustness(patched, dtype="float32")


def test_a_failed_run_raises_with_the_reason(tiny_backend, project, spec_factory):
    with pytest.raises(lg.api.LogogramError, match="failed"):
        lg.run(
            spec_factory(dataset={"path": "datasets/missing.jsonl"}), project, model=tiny_backend
        )


def test_the_command_line_reads_and_compares_runs(tiny_backend, project, spec_factory, tmp_path):
    a = lg.run(spec_factory(), project, model=tiny_backend)
    b = lg.run(
        spec_factory(experiment={"kind": "activation_patching", "direction": "corrupt_to_clean"}),
        project,
        model=tiny_backend,
    )
    where = ["--project", str(project.root)]

    listed = runner.invoke(app, ["list", *where])
    assert listed.exit_code == 0 and a.id in listed.output and b.id in listed.output

    shown = runner.invoke(app, ["show", a.id, *where])
    assert shown.exit_code == 0, shown.output
    assert "Normalized effect" in shown.output and "Corrected for 8 sites" in shown.output

    compared = runner.invoke(app, ["compare", a.id, b.id, *where])
    assert compared.exit_code == 0, compared.output
    assert "Rank correlation" in compared.output and "Paired prompt by prompt" in compared.output
    assert "experiment.direction" in compared.output

    diffed = runner.invoke(app, ["diff", a.id, str(b.folder / "spec.json"), *where])
    assert diffed.exit_code == 0
    assert "experiment.direction: 'clean_to_corrupt' → 'corrupt_to_clean'" in diffed.output

    csv_path = tmp_path / "a.csv"
    exported = runner.invoke(app, ["export", a.id, "-o", str(csv_path), *where])
    assert exported.exit_code == 0
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith("site,kind,layer")

    missing = runner.invoke(app, ["show", "no-such-run", *where])
    assert missing.exit_code == 1 and "there is no run no-such-run" in missing.output


def test_validate_says_what_a_spec_left_out(tmp_path):
    spec = {
        "logogram_spec": 1,
        "name": "old",
        "model": {"id": "gpt2"},
        "dataset": {"path": "datasets/ioi.jsonl"},
        "experiment": {"kind": "ablation", "baseline": {"kind": "mean"}},
        "scope": {"kind": "heads"},
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    out = runner.invoke(app, ["validate", str(path)])
    assert out.exit_code == 0
    assert (
        "version 1 spec" in out.output and 'experiment.baseline.reference = "corrupt"' in out.output
    )
    spec["logogram_spec"] = 2
    path.write_text(json.dumps(spec), encoding="utf-8")
    bad = runner.invoke(app, ["validate", str(path)])
    assert bad.exit_code == 1 and "statistics: Field required" in bad.output
