"""Runs that stop early keep what finished: the sites every prompt ran at (or, for methods that
measure every site at once, the prompts that finished), with the statistics a finished run gives
them, and a summary that says what is missing."""

from __future__ import annotations

import json
import threading
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

import logogram as lg
from logogram.cli import app as cli
from logogram.datasets import load_dataset, write_dataset
from logogram.runner import INTERRUPTED, run_spec

HEADS = {"kind": "heads", "position": {"kind": "all"}}
PATCHING = {"kind": "activation_patching", "direction": "clean_to_corrupt"}
ATP = {"kind": "attribution_patching", "direction": "clean_to_corrupt", "method": "gradient"}
SWEEPS = {
    "patching": PATCHING,
    "zero ablation": {"kind": "ablation", "baseline": {"kind": "zero"}},
    "mean ablation": {"kind": "ablation", "baseline": {"kind": "mean", "reference": "corrupt"}},
    "resample ablation": {
        "kind": "ablation",
        "baseline": {"kind": "resample", "pool": "corrupt", "donors": 2, "seed": 3},
    },
}
runner = CliRunner()


class Stopper:
    """Records a run's events, and cancels it once ``when(kind, data, events)`` says so."""

    def __init__(self, when: Any) -> None:
        self.cancel = threading.Event()
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.when = when

    def __call__(self, kind: str, data: dict[str, Any]) -> None:
        self.events.append((kind, data))
        if self.when(kind, data, self.events):
            self.cancel.set()


def _after_layers(n: int) -> Stopper:
    return Stopper(lambda kind, _, events: kind == "layer" and _count(events, "layer") >= n)


def _after_batches(n: int) -> Stopper:
    return Stopper(lambda kind, _, events: kind == "progress" and _count(events, "progress") >= n)


def _count(events: list[tuple[str, dict[str, Any]]], kind: str) -> int:
    return sum(1 for k, _ in events if k == kind)


def _summary(folder: Any) -> dict[str, Any]:
    return json.loads((folder / "summary.json").read_text(encoding="utf-8"))


def _rows(folder: Any) -> list[dict[str, Any]]:
    table = pq.read_table(folder / "results.parquet").sort_by(
        [("site", "ascending"), ("prompt", "ascending")]
    )
    return table.to_pylist()


def _same(a: Any, b: Any) -> bool:
    """Equal, NaN equal to NaN (values a method doesn't measure)."""
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    return a == b


def _same_rows(kept: list[dict[str, Any]], full: list[dict[str, Any]]) -> None:
    assert len(kept) == len(full) > 0
    for x, y in zip(kept, full, strict=True):
        assert x.keys() == y.keys()
        for key in x:
            assert _same(x[key], y[key]), (key, x, y)


def _without_corrections(site: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in site.items() if k not in ("band", "q")}


# -- sweeps that go layer by layer -------------------------------------------------------------


@pytest.mark.parametrize("experiment", list(SWEEPS.values()), ids=list(SWEEPS))
def test_a_cancelled_sweep_keeps_the_layers_that_finished(
    project, tiny_backend, spec_factory, experiment
):
    spec = spec_factory(experiment=experiment, scope=HEADS)
    stop = _after_layers(1)
    outcome = run_spec(spec, project, backend=tiny_backend, on_event=stop, cancel=stop.cancel)
    assert outcome.status == "cancelled"
    full = run_spec(spec, project, backend=tiny_backend)
    assert full.status == "finished"

    kept = _summary(outcome.folder)
    assert kept == outcome.summary
    partial = kept["partial"]
    assert partial["status"] == "cancelled" and partial["unit"] == "layers"
    assert (partial["done"], partial["total"]) == (1, 2)
    assert (partial["sites"], partial["sites_total"]) == (4, 8)
    assert partial["prompts"] == partial["prompts_total"] == 12
    assert partial["note"] == "Stopped after 1 of 2 layers: these are the layers that finished."
    assert any("q-values are left out" in item for item in partial["left_out"])

    # The layer that finished, exactly as the whole run gives it; nothing for the other.
    whole = _summary(full.folder)
    finished = [s for s in whole["sites"] if s["layer"] == 0]
    assert [s["index"] for s in kept["sites"]] == [s["index"] for s in finished] == [0, 1, 2, 3]
    assert [_without_corrections(s) for s in kept["sites"]] == [
        _without_corrections(s) for s in finished
    ]
    # Corrections for the number of sites need every site: left out, never computed over four.
    assert all(s["band"] is None and s["q"] is None for s in kept["sites"])
    assert any(s["q"] is not None for s in whole["sites"])
    assert kept["statistics"]["band_level"] is None
    assert kept["statistics"]["multiple_comparisons"] is None
    assert kept["n_sites"] == 4 and kept["n_prompts"] == 12
    assert kept["baseline"] == whole["baseline"] and kept["per_prompt"] == whole["per_prompt"]
    assert kept["donors"] == whole["donors"]
    _same_rows(_rows(outcome.folder), [r for r in _rows(full.folder) if r["layer"] == 0])

    # The run stays cancelled; the history says what it kept, and draws its glyph from layer 0.
    manifest = json.loads((outcome.folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "cancelled" and manifest["error"] is None
    listing = project.run_listing(outcome.run_id)
    assert listing is not None and listing.status == "cancelled"
    assert listing.partial is not None and listing.partial.done == 1
    strongest = max((s["effect"]["mean"] for s in kept["sites"]), key=abs)
    assert listing.profile == [round(strongest, 4), 0.0]
    assert ("cancelled", {"run_id": outcome.run_id, "partial": partial}) in stop.events
    assert not (outcome.folder / "owner.json").exists()


def _failing_at_layer(backend: Any, monkeypatch: Any, layer: int) -> None:
    """The backend fails (as a GPU can, mid-run) once a run patches ``layer``."""
    real = backend.logits

    def logits(tokens: Any, patch: Any, *args: Any, **kwargs: Any) -> Any:
        patches = patch if isinstance(patch, list) else [patch]
        if any(p is not None and p.layer == layer for p in patches):
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        return real(tokens, patch, *args, **kwargs)

    monkeypatch.setattr(backend, "logits", logits)


@pytest.mark.parametrize(
    "experiment", [SWEEPS["patching"], SWEEPS["mean ablation"]], ids=["patching", "ablation"]
)
def test_a_sweep_that_fails_midway_keeps_the_layers_that_finished(
    project, tiny_backend, spec_factory, experiment, monkeypatch
):
    spec = spec_factory(experiment=experiment, scope=HEADS)
    full = run_spec(spec, project, backend=tiny_backend)
    _failing_at_layer(tiny_backend, monkeypatch, 1)
    events: list[tuple[str, dict[str, Any]]] = []
    outcome = run_spec(
        spec, project, backend=tiny_backend, on_event=lambda k, d: events.append((k, d))
    )
    assert outcome.status == "failed"
    assert "illegal memory access" in outcome.manifest["error"]
    assert (outcome.folder / "traceback.txt").is_file()
    kept = _summary(outcome.folder)
    assert kept["partial"]["status"] == "failed"
    assert (
        kept["partial"]["note"]
        == "Stopped after 1 of 2 layers: these are the layers that finished."
    )
    whole = _summary(full.folder)
    assert [_without_corrections(s) for s in kept["sites"]] == [
        _without_corrections(s) for s in whole["sites"] if s["layer"] == 0
    ]
    _same_rows(_rows(outcome.folder), [r for r in _rows(full.folder) if r["layer"] == 0])
    listing = project.run_listing(outcome.run_id)
    assert listing is not None and listing.status == "failed" and listing.partial is not None
    assert "illegal memory access" in (listing.error or "")
    failed = [d for k, d in events if k == "failed"]
    assert failed and failed[0]["partial"] == kept["partial"]


def test_a_run_stopped_before_any_layer_finished_keeps_nothing(project, tiny_backend, spec_factory):
    stop = Stopper(lambda kind, *_: kind == "started")  # before the baselines have run
    spec = spec_factory(scope=HEADS)
    outcome = run_spec(spec, project, backend=tiny_backend, on_event=stop, cancel=stop.cancel)
    assert outcome.status == "cancelled" and outcome.summary is None
    for name in ("results.parquet", "summary.json"):
        assert not (outcome.folder / name).exists()
    listing = project.run_listing(outcome.run_id)
    assert listing is not None and listing.partial is None and listing.profile is None
    assert ("cancelled", {"run_id": outcome.run_id}) in stop.events


def test_ctrl_c_mid_sweep_keeps_the_layers_that_finished(project, tiny_backend, spec_factory):
    def interrupt(kind: str, data: dict[str, Any]) -> None:
        if kind == "progress" and data["layer"] == 1:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_spec(spec_factory(scope=HEADS), project, backend=tiny_backend, on_event=interrupt)
    (folder,) = [f for f in project.experiments_dir.iterdir() if f.is_dir()]
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "cancelled" and manifest["error"] == INTERRUPTED
    kept = _summary(folder)
    assert kept["partial"]["note"].startswith("Stopped after 1 of 2 layers")
    assert len(_rows(folder)) == 4 * 12
    assert not (folder / "owner.json").exists()


def test_ctrl_c_in_logogram_run_says_what_it_kept(project, tiny_backend, spec_factory, monkeypatch):
    from logogram import runner as runs

    real = runs.run_spec

    def interrupted_after_layer_0(spec: Any, project: Any, **kwargs: Any) -> Any:
        given = kwargs.pop("on_event")

        def on_event(kind: str, data: dict[str, Any]) -> None:
            given(kind, data)
            if kind == "progress" and data["layer"] == 1:
                raise KeyboardInterrupt

        return real(spec, project, **{**kwargs, "backend": tiny_backend}, on_event=on_event)

    monkeypatch.setattr(runs, "run_spec", interrupted_after_layer_0)
    spec_path = project.root / "spec.json"
    spec_path.write_text(
        json.dumps(spec_factory(scope=HEADS).model_dump(mode="json")), encoding="utf-8"
    )
    result = runner.invoke(cli, ["run", str(spec_path), "--project", str(project.root)])
    assert result.exit_code == 130, result.output
    (run_id,) = [f.name for f in project.experiments_dir.iterdir() if f.is_dir()]
    assert "Interrupted. The run is recorded as cancelled." in result.output
    assert "Stopped after 1 of 2 layers: these are the layers that finished." in result.output
    assert f"logogram show {run_id}" in result.output


# -- methods that measure every site at once -------------------------------------------------


def _batches(project: Any) -> None:
    """A dataset of prompts in several groups of the same length."""
    from logogram.ioi import generate_ioi

    write_dataset(
        project.datasets_dir / "batches.jsonl",
        generate_ioi(10, seed=1, templates=["went", "working", "reached"]),
    )


def test_attribution_patching_keeps_every_site_over_the_prompts_that_finished(
    project, tiny_backend, spec_factory
):
    _batches(project)
    spec = spec_factory(
        experiment=ATP,
        scope=HEADS,
        dataset={"path": "datasets/batches.jsonl"},
        execution={"batch_size": 3},
    )
    stop = _after_batches(1)
    outcome = run_spec(spec, project, backend=tiny_backend, on_event=stop, cancel=stop.cancel)
    assert outcome.status == "cancelled"
    kept = _summary(outcome.folder)
    partial = kept["partial"]
    assert partial["unit"] == "prompts" and partial["total"] == 10
    done = partial["done"]
    assert done == 3 and partial["prompts"] == kept["n_prompts"] == done
    assert (partial["sites"], partial["sites_total"]) == (8, 8)
    assert partial["note"] == (
        f"Stopped after {done} of 10 prompts: every site is measured on the prompts that finished."
    )
    assert any("aren't a random sample" in item for item in partial["left_out"])
    rows = _rows(outcome.folder)
    measured = sorted({r["prompt"] for r in rows})
    assert len(measured) == done and len(rows) == 8 * done

    # Each prompt's estimates are the whole run's, exactly. (Normalized effects divide by the
    # mean gap of the prompts measured, as a run of just them does.)
    full = run_spec(spec, project, backend=tiny_backend)

    def unnormalized(table: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in r.items() if k != "effect"} for r in table]

    _same_rows(
        unnormalized(rows),
        unnormalized([r for r in _rows(full.folder) if r["prompt"] in measured]),
    )

    # And its statistics and rows are those of a finished run of just those prompts (all sites:
    # so the corrections for their number are there too).
    records = load_dataset(project.datasets_dir / "batches.jsonl")
    write_dataset(project.datasets_dir / "finished.jsonl", [records[i] for i in measured])
    alone = run_spec(
        spec.model_copy(
            update={"dataset": spec.dataset.model_copy(update={"path": "datasets/finished.jsonl"})}
        ),
        project,
        backend=tiny_backend,
    )
    assert alone.status == "finished"
    renumbered = [{**r, "prompt": measured[r["prompt"]]} for r in _rows(alone.folder)]
    _same_rows(rows, sorted(renumbered, key=lambda r: (r["site"], r["prompt"])))
    other = _summary(alone.folder)
    assert kept["sites"] == other["sites"]
    assert any(s["q"] is not None for s in kept["sites"])
    assert kept["statistics"] == other["statistics"]
    assert kept["metric"] == other["metric"] and kept["baseline"] == other["baseline"]
    assert kept["warnings"] == other["warnings"]


def test_attribution_patching_that_fails_midway_keeps_the_prompts_that_finished(
    project, tiny_backend, spec_factory, monkeypatch
):
    _batches(project)
    spec = spec_factory(
        experiment=ATP,
        scope=HEADS,
        dataset={"path": "datasets/batches.jsonl"},
        execution={"batch_size": 3},
    )
    real = tiny_backend.gradients
    calls = {"n": 0}

    def gradients(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("CUDA out of memory: simulated")
        return real(*args, **kwargs)

    monkeypatch.setattr(tiny_backend, "gradients", gradients)
    outcome = run_spec(spec, project, backend=tiny_backend)
    assert outcome.status == "failed" and "ran out of memory" in outcome.manifest["error"]
    partial = _summary(outcome.folder)["partial"]
    assert partial["status"] == "failed" and partial["unit"] == "prompts"
    assert partial["done"] == len({r["prompt"] for r in _rows(outcome.folder)})
    assert 2 <= partial["done"] < 10
    listing = project.run_listing(outcome.run_id)
    assert listing is not None and listing.status == "failed" and listing.partial is not None


def test_direct_logit_attribution_keeps_the_prompts_that_finished(
    project, tiny_backend, spec_factory
):
    _batches(project)
    spec = spec_factory(
        experiment={"kind": "direct_logit_attribution", "prompts": "clean"},
        scope={
            "kind": "layer_components",
            "components": ["attn_out", "mlp_out"],
            "position": {"kind": "last"},
        },
        dataset={"path": "datasets/batches.jsonl"},
        execution={"batch_size": 3},
    )
    stop = _after_batches(1)
    outcome = run_spec(spec, project, backend=tiny_backend, on_event=stop, cancel=stop.cancel)
    assert outcome.status == "cancelled"
    kept = _summary(outcome.folder)
    assert kept["partial"]["unit"] == "prompts"
    assert any("split of the logit difference" in item for item in kept["partial"]["left_out"])
    assert kept["direct"] is not None
    done = kept["partial"]["done"]
    # The split is of the finished prompts' own logit difference.
    assert kept["direct"]["logit_diff"] == pytest.approx(
        np.mean(kept["per_prompt"]["clean_logit_diff"]), abs=1e-4
    )
    assert len(kept["per_prompt"]["clean_logit_diff"]) == done


# -- steering and sets of sites --------------------------------------------------------------


def test_steering_keeps_each_finished_site_with_its_random_control(
    project, tiny_backend, spec_factory
):
    spec = spec_factory(
        experiment={
            "kind": "steering",
            "apply_to": "corrupt",
            "coefficients": [0.0, 1.0],
            "train_fraction": 0.5,
            "seed": 0,
            "control": True,
        },
        scope={
            "kind": "layer_components",
            "components": ["resid_pre"],
            "position": {"kind": "last"},
        },
    )
    stop = _after_layers(1)
    outcome = run_spec(spec, project, backend=tiny_backend, on_event=stop, cancel=stop.cancel)
    assert outcome.status == "cancelled"
    full = run_spec(spec, project, backend=tiny_backend)
    kept, whole = _summary(outcome.folder), _summary(full.folder)
    assert (
        kept["partial"]["note"]
        == "Stopped after 1 of 2 layers: these are the layers that finished."
    )
    # One steered layer: its two strengths and their two controls.
    assert {s["row"] for s in kept["sites"]} == {0} and len(kept["sites"]) == 4
    assert [_without_corrections(s) for s in kept["sites"]] == [
        _without_corrections(s) for s in whole["sites"] if s["row"] == 0
    ]
    control = kept["steering"]["control"]
    assert control == [c for c in whole["steering"]["control"] if c["row"] == 0]
    assert any("random control covers" in item for item in kept["partial"]["left_out"])


def test_steering_stopped_while_recording_what_the_model_says_keeps_every_site(
    project, tiny_backend, spec_factory
):
    """Continuations come after the steered sites: a run stopped while it records them has
    finished every site, and keeps them all, with everything the whole run gives them."""
    spec = spec_factory(
        experiment={
            "kind": "steering",
            "apply_to": "corrupt",
            "coefficients": [0.0, 1.0],
            "train_fraction": 0.5,
            "seed": 0,
            "control": True,
            "continuations": {"prompts": 2, "tokens": 4},
        },
        scope={
            "kind": "layer_components",
            "components": ["resid_pre"],
            "position": {"kind": "last"},
        },
    )
    stop = _after_layers(2)
    outcome = run_spec(spec, project, backend=tiny_backend, on_event=stop, cancel=stop.cancel)
    assert outcome.status == "cancelled"
    full = run_spec(spec, project, backend=tiny_backend)
    assert full.status == "finished"
    kept, whole = _summary(outcome.folder), _summary(full.folder)
    partial = kept["partial"]
    assert (partial["done"], partial["total"], partial["unit"]) == (2, 2, "layers")
    assert partial["note"] == "Stopped before the run ended, once its 2 layers had finished."
    assert partial["left_out"] == ["What the steered model says wasn't recorded: it comes last."]
    # Every site, exactly as the whole run gives it, corrections for the number of sites included.
    assert (
        kept["sites"] == whole["sites"]
        and kept["steering"]["control"] == whole["steering"]["control"]
    )
    assert kept["statistics"] == whole["statistics"]
    assert not (outcome.folder / "continuations.json").exists()
    assert (full.folder / "continuations.json").exists()


def _sets_spec(spec_factory: Any, everything_first: bool) -> Any:
    head = {"kind": "head", "layer": 1, "head": 2, "position": {"kind": "all"}}
    everything = {"label": "everything", "sites": [], "complement": True}
    sets = [
        {"label": "keep L1 H2", "sites": [head], "complement": True},
        {"label": "remove L1 H2", "sites": [head], "complement": False},
    ]
    return spec_factory(
        experiment=PATCHING,
        scope={
            "kind": "site_sets",
            "universe": ["head"],
            "sets": [everything, *sets] if everything_first else [*sets, everything],
        },
    )


def test_sets_are_compared_with_everything_only_once_it_finished(
    project, tiny_backend, spec_factory
):
    # The set that replaces everything ran first: the set after it keeps its share.
    spec = _sets_spec(spec_factory, everything_first=True)
    stop = _after_layers(2)
    outcome = run_spec(spec, project, backend=tiny_backend, on_event=stop, cancel=stop.cancel)
    full = run_spec(spec, project, backend=tiny_backend)
    kept, whole = _summary(outcome.folder), _summary(full.folder)
    assert kept["partial"]["unit"] == "sets"
    assert kept["partial"]["note"] == "Stopped after 2 of 3 sets: these are the sets that finished."
    assert [s["label"] for s in kept["sites"]] == ["everything", "keep L1 H2"]
    assert kept["circuit"]["everything"] == 0
    rows = {r["label"]: r for r in kept["circuit"]["rows"]}
    assert set(rows) == {"everything", "keep L1 H2"}
    expected = {r["label"]: r for r in whole["circuit"]["rows"]}
    assert rows["keep L1 H2"]["faithfulness"] == expected["keep L1 H2"]["faithfulness"]
    assert rows["keep L1 H2"]["faithfulness"] is not None
    assert not any("Shares and faithfulness are left out" in i for i in kept["partial"]["left_out"])

    # It would have run last: no set has a share, and the summary says why.
    spec = _sets_spec(spec_factory, everything_first=False)
    stop = _after_layers(2)
    outcome = run_spec(spec, project, backend=tiny_backend, on_event=stop, cancel=stop.cancel)
    kept = _summary(outcome.folder)
    assert [s["label"] for s in kept["sites"]] == ["keep L1 H2", "remove L1 H2"]
    assert all(r["share"] is None and r["faithfulness"] is None for r in kept["circuit"]["rows"])
    assert any("Shares and faithfulness are left out" in i for i in kept["partial"]["left_out"])


# -- reading a partial run -----------------------------------------------------------------


@pytest.fixture
def stopped(project, tiny_backend, spec_factory):
    """A heads sweep cancelled after its first layer, and the same sweep finished."""
    spec = spec_factory(scope=HEADS, name="Stopped")
    stop = _after_layers(1)
    outcome = run_spec(spec, project, backend=tiny_backend, on_event=stop, cancel=stop.cancel)
    assert outcome.status == "cancelled"
    full = run_spec(spec.model_copy(update={"name": "Finished"}), project, backend=tiny_backend)
    return outcome, full


def test_the_command_line_shows_a_partial_run_as_such(project, stopped):
    outcome, full = stopped
    listed = runner.invoke(cli, ["list", "--project", str(project.root)])
    assert listed.exit_code == 0, listed.output
    assert "partial: stopped after 1 of 2 layers, kept what finished" in listed.output
    shown = runner.invoke(cli, ["show", outcome.run_id, "--project", str(project.root)])
    assert shown.exit_code == 0, shown.output
    assert "Cancelled." in shown.output
    assert (
        "Partial: Stopped after 1 of 2 layers: these are the layers that finished." in shown.output
    )
    assert "Corrected for" not in shown.output
    compared = runner.invoke(
        cli, ["compare", outcome.run_id, full.run_id, "--project", str(project.root)]
    )
    assert compared.exit_code == 0, compared.output
    assert f"{outcome.run_id} stopped after 1 of 2 layers" in compared.output
    assert "4 sites in common" in compared.output
    refused = runner.invoke(
        cli, ["robustness", outcome.run_id, "--dtype", "float16", "--project", str(project.root)]
    )
    assert refused.exit_code == 1
    assert (
        "Checking robustness needs a finished run, and this one stopped after 1 of 2 layers"
        in refused.output
    )
    report = project.root / "report.html"
    made = runner.invoke(
        cli, ["report", outcome.run_id, "-o", str(report), "--project", str(project.root)]
    )
    assert made.exit_code == 0, made.output
    page = report.read_text(encoding="utf-8")
    assert "This run was cancelled before it finished" in page
    assert "Stopped after 1 of 2 layers: these are the layers that finished." in page
    assert "The 4 sites that finished" in page
    methods = runner.invoke(cli, ["methods", outcome.run_id, "--project", str(project.root)])
    assert "The run was cancelled before it finished. Stopped after 1 of 2 layers" in methods.output


def test_the_python_api_says_a_run_is_partial(project, stopped, tiny_backend):
    outcome, full = stopped
    run = lg.load_run(project, outcome.run_id)
    assert run.status == "cancelled"
    assert run.partial is not None and run.partial["done"] == 1
    assert len(run.sites()) == 4
    assert "Stopped after 1 of 2 layers" in run._repr_html_()
    assert lg.load_run(project, full.run_id).partial is None
    comparison = lg.compare_runs(run, lg.load_run(project, full.run_id))
    assert comparison["partial_a"]["done"] == 1 and comparison["partial_b"] is None
    assert comparison["n_common"] == 4
    with pytest.raises(lg.api.LogogramError, match="needs a finished run"):
        lg.check_robustness(run, dtype="float16", model=tiny_backend)


def test_the_python_api_run_says_what_a_stopped_run_kept(
    project, tiny_backend, spec_factory, monkeypatch
):
    _failing_at_layer(tiny_backend, monkeypatch, 1)
    with pytest.raises(lg.api.LogogramError) as caught:
        lg.run(spec_factory(scope=HEADS), project, model=tiny_backend)
    message = str(caught.value)
    assert "The run failed:" in message
    assert "Stopped after 1 of 2 layers: these are the layers that finished." in message
    assert "load_run(project, " in message
