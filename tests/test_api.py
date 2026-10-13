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


def test_the_command_line_generates_task_datasets(project):
    where = ["--project", str(project.root)]
    listed = runner.invoke(app, ["tasks"])
    assert listed.exit_code == 0 and "greater_than" in listed.output
    assert "metric.kind = 'prob_diff'" in listed.output
    exists = runner.invoke(
        app, ["generate", "ioi", "-n", "5", "--seed", "1", "-O", "corruption=abc", *where]
    )
    assert exists.exit_code == 1 and "ioi.jsonl already exists" in exists.output
    made = runner.invoke(
        app,
        ["generate", "ioi", "-n", "5", "--seed", "1", "--name", "abc", "-O", "corruption=abc",
         *where],
    )  # fmt: skip
    assert made.exit_code == 0, made.output
    assert "Wrote 5 prompt pairs to datasets/abc.jsonl" in made.output
    records = (project.root / "datasets" / "abc.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 5
    same = lg.generate("ioi", 5, 1, {"corruption": "abc"})
    assert [json.loads(r)["clean"] for r in records] == [r.clean for r in same]
    bad = runner.invoke(app, ["generate", "ioi", "-n", "5", "--seed", "1", "-O", "nope=1", *where])
    assert bad.exit_code == 1 and "options are" in bad.output


def test_an_example_opens_from_python_as_a_copy_of_its_own(tmp_path):
    project = lg.open_example("ioi-gpt2", tmp_path)
    assert project.root == (tmp_path / "ioi-example").resolve()
    spec = lg.load_spec(project.root / "experiments/ioi-head-patching/spec.json")
    assert spec.model.id == "openai-community/gpt2"
    # Opened again, it is the same copy, as it was left.
    (project.root / "notes.txt").write_text("mine", encoding="utf-8")
    again = lg.open_example("ioi-gpt2", tmp_path)
    assert again.root == project.root and (again.root / "notes.txt").is_file()
    with pytest.raises(lg.api.LogogramError, match="There is no example called 'nope'"):
        lg.open_example("nope", tmp_path)


def test_look_lists_the_features_that_fire_with_an_sae(tiny_backend, tmp_path):
    from logogram.inside import look_features
    from test_sae import exact_sae

    text = "When Mary and John went to the store, John gave a drink to"
    sae = exact_sae(tmp_path / "sae", tiny_backend.info.d_model)
    plain = lg.look(tiny_backend, text, top_k=3)
    assert "features" not in plain
    report = lg.look(tiny_backend, text, top_k=3, sae=sae)
    with tiny_backend.lock:
        same = look_features(tiny_backend, sae, text, prepend_bos=True, top_k=3)
    assert report["features"] == same["features"] and report["active"] == same["active"]
    assert report["first_real_token"] == 1
    added = {"features", "active", "first_real_token"}
    assert {k: v for k, v in report.items() if k not in added} == plain


def test_a_model_that_isnt_the_specs_is_refused_naming_each_difference(
    tiny_backend, project, spec_factory, monkeypatch
):
    import torch

    def no_second_copy(*args, **kwargs):
        raise AssertionError("another copy of the model was loaded")

    monkeypatch.setattr("logogram.runner.default_provider", no_second_copy)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = {
        "id": "tiny-gpt2",
        "revision": "other",
        "dtype": "bfloat16",
        "device": "cuda",
        "process_weights": False,
    }
    spec = spec_factory(model=model)
    before = sorted(p.name for p in project.experiments_dir.iterdir())
    with pytest.raises(lg.api.LogogramError) as refused:
        lg.run(spec, project, model=tiny_backend)
    message = str(refused.value)
    assert message.startswith("The model passed to run() isn't the one the spec names: ")
    for clause in (
        "it is at revision test, and the spec pins other",
        "it runs in float32, and the spec in bfloat16",
        "it is on cpu, and the spec's device, cuda, isn't available here",
        "its weights are processed, and the spec's aren't",
        "lg.with_model(spec, model)",
        "pass no model to load the spec's own",
    ):
        assert clause in message
    # Nothing ran: no folder was made, and no model was loaded.
    assert sorted(p.name for p in project.experiments_dir.iterdir()) == before
    other = spec_factory(model={"id": "someone/else", "revision": None, "device": "auto"})
    with pytest.raises(lg.api.LogogramError, match="it is tiny-gpt2 from Hugging Face, and the "
                       "spec names someone/else from Hugging Face"):  # fmt: skip
        lg.run(other, project, model=tiny_backend)

    # The spec, on this model: its model is the one loaded, exactly as it was loaded.
    mine = lg.with_model(spec, tiny_backend)
    assert mine.model.model_dump() == {
        "id": "tiny-gpt2",
        "source": "hub",
        "revision": "test",
        "sha256": None,
        "dtype": "float32",
        "device": "cpu",
        "process_weights": True,
    }
    assert mine.model_dump(exclude={"model"}) == spec.model_dump(exclude={"model"})
    run = lg.run(mine, project, model=tiny_backend)
    assert run.status == "finished" and run.spec.model == mine.model
    # A model loaded with device "auto" is named so: the device it was loaded with.
    monkeypatch.setattr(tiny_backend, "requested_device", "auto", raising=False)
    assert lg.with_model(spec, tiny_backend).model.device == "auto"
    assert lg.with_model(spec.model_dump(mode="json"), tiny_backend).model.device == "auto"


def test_the_runs_made_from_a_run_reuse_its_model_or_refuse_another(
    tiny_backend, project, spec_factory
):
    import dataclasses

    from logogram.backends.transformer_lens import TransformerLensBackend

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
    # The same weights, but in another dtype as far as specs can tell.
    other = TransformerLensBackend(
        tiny_backend.bridge, dataclasses.replace(tiny_backend.info, dtype="bfloat16")
    )
    with pytest.raises(lg.api.LogogramError, match=r"passed to verify_run\(\).*bfloat16"):
        lg.verify_run(estimate, top=2, model=other)
    patching = {"kind": "activation_patching", "direction": "clean_to_corrupt"}
    with pytest.raises(lg.api.LogogramError, match=r"passed to check_robustness\(\)"):
        lg.check_robustness(estimate, experiment=patching, model=other)
    with pytest.raises(lg.api.LogogramError, match=r"passed to sweep\(\)"):
        lg.sweep(estimate, field="statistics.seed", values=[1, 2], model=other)
    # The model the run was made with is reused, given or not.
    assert lg.verify_run(estimate, top=2, model=tiny_backend).status == "finished"
    swept = lg.sweep(estimate, field="statistics.seed", values=[1, 2])
    assert [r.status for r in swept] == ["finished", "finished"]
    assert all(r.model is tiny_backend for r in swept)
