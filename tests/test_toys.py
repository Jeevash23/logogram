"""Toy models whose circuits are known (logogram.toys): ground truth for every method.

Each toy is a GPT-2 checkpoint whose weights are set by hand so that a planted circuit does the
task, loaded through exactly the code real models go through:

* the induction toy (two layers): L0 H2 is a previous-token head and L1 H1 an induction head;
* the IOI toy (four layers): L0 H2 previous-token, L1 H1 induction, L2 H3 S-inhibition and L3 H0
  a name mover, GPT-2 small's indirect-object circuit with one head per step.

The tests check that each method finds what was planted, with the effects the weights were built
to give. Interventions (activation patching, path patching, sets of sites, steering) and direct
logit attribution are exact here: the layer norms divide by a fixed scale (their epsilon dwarfs
every variance), every planted head attends to its target with all but 10^-8 of its attention,
and every other head writes where nothing reads. Designed effects of 0 and 1 come out within
about 10^-6, and the tests allow 10^-4.

Attribution patching is a first-order estimate. It is exact for a head whose output reaches the
logits linearly (the induction head, the name mover: 1.0 to 10^-4), and blind to a head that acts
through another head's saturated attention: the softmax's gradient there is about e^-20, so the
estimate is 0 where patching gives 1. Integrated gradients follow the attention as it moves, and
recover the induction toy's previous-token head once their steps are finer than the move (0.47
with 8 steps, 0.92 with 16, above 0.99 with 32); in the IOI toy two softmaxes in a row flip
within a hundredth of the way, between two steps, and stay missed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from logogram import local_models, toys
from logogram.backends.base import BackendError
from logogram.datasets import write_dataset
from logogram.heads import HeadKindSettings, head_kind_scores
from logogram.project import Project
from logogram.runner import run_spec
from logogram.spec import Spec

TOL = 1e-4


def _project(tmp_path: Path, name: str) -> Project:
    project = Project.create(tmp_path, f"{name} toy")
    for file_name, records in toys.datasets(name).items():
        write_dataset(project.datasets_dir / file_name, records)
    return project


def _spec(name: str, base: toys.ToyExperiment, **changes: Any) -> Spec:
    """A toy experiment's spec, with parts replaced: its ``dataset`` (a file name), the number of
    prompts it reads (``limit``), its ``experiment`` or its ``scope``."""
    spec = toys.spec_of(name, base)
    data = spec.model_dump(mode="json")
    for key, value in changes.items():
        if key == "dataset":
            data["dataset"]["path"] = f"datasets/{value}"
        elif key == "limit":
            data["dataset"]["limit"] = value
        else:
            data[key] = value
    return Spec.model_validate(data)


def _run(backend: Any, project: Project, spec: Spec) -> dict[str, Any]:
    outcome = run_spec(spec, project, backend=backend)
    assert outcome.status == "finished", outcome.manifest.get("error")
    assert outcome.backend is backend  # the toy was found by the hash of its files
    return json.loads((outcome.folder / "summary.json").read_text(encoding="utf-8"))


def _effects(summary: dict[str, Any]) -> dict[str, float]:
    return {site["label"]: site["effect"]["mean"] for site in summary["sites"]}


def _assert_effects(
    summary: dict[str, Any], expect: dict[str, float], tolerance: float = TOL
) -> None:
    """Every site's effect is the one expected (0 for a site not named), within ``tolerance``."""
    effects = _effects(summary)
    assert set(expect) <= set(effects), f"no sites {set(expect) - set(effects)}"
    wrong = {
        label: round(value, 6)
        for label, value in effects.items()
        if abs(value - expect.get(label, 0.0)) > tolerance
    }
    assert not wrong, f"expected {expect}, every other site 0; off: {wrong}"


def _experiment(name: str, experiment_id: str) -> toys.ToyExperiment:
    return next(e for e in toys.experiments(name) if e.id == experiment_id)


# -- the files and the model ------------------------------------------------------------------


@pytest.mark.parametrize("name", list(toys.TOYS))
def test_the_files_are_the_same_bytes_everywhere(name, tmp_path):
    """A toy's files are written byte for byte from exact numbers: built twice they are the same,
    and they hash (as local models are hashed) to the hash specs name the toy by."""
    from safetensors.numpy import load_file

    files = toys.model_files(name)
    assert files == toys.model_files(name)
    assert sorted(files) == [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    folder = toys.write_model(name, tmp_path / "model")
    names = local_models.model_files(folder)
    assert names == sorted(files)
    hashed = local_models.folder_hash(local_models.hash_files(folder, names))
    assert hashed == toys.files_sha256(files) == toys.get_toy(name).sha256
    # The weights are safetensors, as safetensors itself reads them, and exactly the numbers built.
    weights = load_file(folder / "model.safetensors")
    built = toys._build(toys.get_toy(name)).tensors()
    assert sorted(weights) == sorted(built)
    for key, array in built.items():
        assert weights[key].dtype == np.float32
        assert np.array_equal(weights[key], array.astype(np.float32)), key
    config = json.loads(files["config.json"])
    assert config["model_type"] == "gpt2" and config["n_layer"] == toys.get_toy(name).n_layers
    assert config["layer_norm_epsilon"] == 2.0**20 and not config["tie_word_embeddings"]


@pytest.mark.parametrize("name", list(toys.TOYS))
def test_a_toy_loads_and_passes_every_check_a_model_gets(name, toy_backend):
    """Loaded as any model is: TransformerLens reproduces it, its layers add into the residual
    stream in turn, the logit lens reproduces its output, and its queries, keys, values and
    neurons are sites (the MLP's neurons are there, and are always zero)."""
    info = toy_backend(name).info
    checks = info.extra["checks"]
    measured = {k: v for k, v in checks.items() if k not in ("tolerance", "structure")}
    assert all(v is not None and v <= checks["tolerance"] for v in measured.values()), checks
    assert info.extra["block_structure"] == "sequential_pre_norm"
    assert info.extra["prediction_method"] == "final_norm_logit_lens"
    assert {"head", "q", "k", "v", "neuron", "resid_mid"} <= set(info.site_kinds)
    assert (info.n_layers, info.n_heads, info.d_model, info.d_head) == (
        toys.get_toy(name).n_layers,
        4,
        256,
        64,
    )
    assert info.source == "local" and info.sha256 == toys.get_toy(name).sha256


@pytest.mark.parametrize("name", list(toys.TOYS))
def test_each_toy_does_its_task_and_its_heads_attend_as_built(name, toy_backend):
    check = toys.check_toy(toy_backend(name), name)
    assert check.ok, check.problems
    assert check.clean_correct == check.corrupt_correct == check.n_prompts == 64
    assert check.clean_logit_diff == pytest.approx(10.0, abs=1e-4)
    assert check.corrupt_logit_diff == pytest.approx(-10.0, abs=1e-4)
    assert [(h.head.layer, h.head.head) for h in check.heads] == [
        (h.layer, h.head) for h in toys.get_toy(name).heads
    ]
    for h in check.heads:
        assert h.least > 1 - 1e-6, (h.head.label, h.least)


def test_a_toy_that_is_not_as_built_is_caught(tmp_path):
    """The check at load reads the model, not its name: an induction toy whose induction head
    writes nothing solves no prompt, and the check says so."""
    from logogram.backends.transformer_lens import TransformerLensBackend, boot_local

    toy = toys.get_toy("induction")
    files = toys.model_files("induction")
    builder = toys._build(toy)
    ind = toy.head("induction")
    rows = slice(ind.head * toys.D_HEAD, (ind.head + 1) * toys.D_HEAD)
    builder.proj[ind.layer][rows] = 0.0  # its output projection: it writes nothing
    files["model.safetensors"] = toys._safetensors(builder.tensors())
    for file_name, data in files.items():
        (tmp_path / file_name).write_bytes(data)
    bridge = boot_local(tmp_path, device="cpu", dtype="float32")
    # As if its files hashed as the toy's: loading checks it as the toy, and refuses it.
    backend = TransformerLensBackend.from_bridge(
        bridge,
        model_id="toy-induction",
        revision=None,
        dtype="float32",
        process_weights=True,
        source="local",
        sha256=toy.sha256,
    )
    check = toys.check_toy(backend, "induction")
    assert not check.ok
    assert check.clean_correct == 0
    assert any("prefers the answer on 0 of 64 clean prompts" in p for p in check.problems)
    # Its heads still attend as built: only what it writes is gone.
    assert all(h.least > 0.99 for h in check.heads)
    with pytest.raises(BackendError, match="didn't behave as built when it loaded in float32"):
        toys.check_loaded_toy(backend)
    assert "toy" not in backend.info.extra
    with pytest.raises(toys.ToyError, match="2 layers of 4 heads"):
        toys.check_toy(backend, "ioi")


def test_loading_checks_a_toy_and_leaves_other_models_alone(toy_backend, tiny_backend):
    backend = toy_backend("ioi")
    check = toys.check_loaded_toy(backend)
    assert check is not None and check.ok
    assert backend.info.extra["toy"]["solved"] == {"clean": 64, "corrupt": 64}
    assert toys.check_loaded_toy(tiny_backend) is None


# -- the head finder -----------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(toys.TOYS))
def test_the_head_finder_names_the_planted_heads_and_only_them(name, toy_backend):
    """On random sequences of 32 tokens repeated, the planted previous-token and induction heads
    score 1 and no other head more than 0.06 (the heads that attend evenly score about one over
    the position, the name mover splits its attention among the names); no head is a
    duplicate-token head."""
    toy = toys.get_toy(name)
    report = head_kind_scores(
        toy_backend(name),
        HeadKindSettings(seed=0, sequences=8, length=32, prepend_bos=True, batch_size=8),
    )
    for kind in ("previous_token", "induction", "duplicate_token"):
        means = np.array(report["kinds"][kind]["mean"])
        planted = {(h.layer, h.head) for h in toy.heads if h.kind == kind}
        found = {(int(layer), int(head)) for layer, head in np.argwhere(means > 0.5)}
        assert found == planted, (kind, found)
        for layer, head in planted:
            assert means[layer, head] > 1 - 1e-6
            assert report["kinds"][kind]["lo"][layer][head] > 1 - 1e-6
        others = means.copy()
        for layer, head in planted:
            others[layer, head] = 0.0
        assert others.max() < 0.06, (kind, others.max())


# -- every experiment a toy comes with finds what was planted -----------------------------------


EXPERIMENTS = [(name, e.id) for name in toys.TOYS for e in toys.experiments(name)]


@pytest.mark.parametrize(("name", "experiment_id"), EXPERIMENTS)
def test_each_experiment_finds_what_was_planted(name, experiment_id, toy_backend, tmp_path):
    """Run each experiment a toy's project comes with, end to end (the runner, the summary), and
    compare every site with its designed effect: patching, path patching, sets of sites, direct
    logit attribution, attribution patching (exact where linear, blind through saturated
    attention) and steering."""
    experiment = _experiment(name, experiment_id)
    summary = _run(toy_backend(name), _project(tmp_path, name), toys.spec_of(name, experiment))
    _assert_effects(summary, experiment.expect, experiment.tolerance)
    for label, faithfulness in experiment.faithfulness.items():
        row = next(r for r in summary["circuit"]["rows"] if r["label"] == label)
        assert row["faithfulness"]["mean"] == pytest.approx(faithfulness, abs=TOL), label
    if experiment.experiment["kind"] == "steering":
        controls = summary["steering"]["control"]
        assert controls and all(c["beats_control"] for c in controls)


def test_the_experiments_a_toy_comes_with_are_valid_specs():
    for name in toys.TOYS:
        ids = [e.id for e in toys.experiments(name)]
        assert len(set(ids)) == len(ids)
        for experiment in toys.experiments(name):
            spec = toys.spec_of(name, experiment)
            assert spec.model.source == "local" and spec.model.sha256 == toys.get_toy(name).sha256
            assert spec.upgraded_fields == []
            assert experiment.dataset in toys.datasets(name)
            assert experiment.notes and "expect" in experiment.notes.lower()


# -- patching: what differs, and which input carries it -----------------------------------------


@pytest.mark.parametrize(
    ("corruption", "heads", "inputs"),
    [
        # A and F swap: what the previous-token head copies changes, and reaches the induction
        # head through its key.
        ("key", {"L0 H2": 1.0, "L1 H1": 1.0}, {"L0 KV2 v": 1.0, "L1 KV1 k": 1.0}),
        # B and C swap: the induction head attends where it did, and reads another token.
        ("value", {"L1 H1": 1.0}, {"L1 KV1 v": 1.0}),
        # The last word changes: the induction head looks for another word.
        ("query", {"L1 H1": 1.0}, {"L1 H1 q": 1.0}),
    ],
)
def test_patching_says_which_input_carries_the_difference(
    corruption, heads, inputs, toy_backend, tmp_path
):
    project = _project(tmp_path, "induction")
    records = toys.induction_prompts(32, seed=1, corruption=corruption)
    write_dataset(project.datasets_dir / f"{corruption}.jsonl", records)
    backend = toy_backend("induction")
    for experiment_id, expect in (("which-heads", heads), ("which-input", inputs)):
        spec = _spec(
            "induction", _experiment("induction", experiment_id), dataset=f"{corruption}.jsonl"
        )
        _assert_effects(_run(backend, project, spec), expect)


def test_ioi_corruptions_reveal_different_heads(toy_backend, tmp_path):
    """Patching shows what differs between the prompts. Flipping S2 changes what the induction
    head finds, so the previous-token head (the same in both prompts) scores 0. With three new
    names (abc) nothing repeats in the corrupt prompt: patching it into the clean one (corrupt →
    clean), each of the four upstream heads stops the repeat being found, and the name mover then
    splits over the three names, a logit difference of −10/3 against the corrupt prompt's 0: an
    effect of 4/3."""
    project = _project(tmp_path, "ioi")
    write_dataset(project.datasets_dir / "abc.jsonl", toys.ioi_prompts(32, 1, corruption="abc"))
    backend = toy_backend("ioi")
    spec = _spec(
        "ioi",
        _experiment("ioi", "which-heads"),
        dataset="abc.jsonl",
        experiment={"kind": "activation_patching", "direction": "corrupt_to_clean"},
    )
    third = 4 / 3
    _assert_effects(
        _run(backend, project, spec), {"L0 H2": third, "L1 H1": third, "L2 H3": third, "L3 H0": 1.0}
    )


# -- path patching: the planted compositions, and only them ---------------------------------------


def _receivers(toy: toys.Toy) -> list[tuple[int, int, str] | None]:
    """Every input of every planted head after layer 0, and the logits (None)."""
    heads = [(h.layer, h.head, part) for h in toy.heads if h.layer > 0 for part in "qkv"]
    return [*heads, None]


@pytest.mark.parametrize(
    ("name", "receiver"),
    [(name, r) for name in toys.TOYS for r in _receivers(toys.get_toy(name))],
)
def test_path_patching_finds_the_planted_compositions(name, receiver, toy_backend, tmp_path):
    """Path patching from every head into every input of every planted head (and the logits):
    a sender reaches a receiver's input with effect 1 exactly where the toy's compositions say
    a head writes what that input reads and the prompts differ along it, and 0 everywhere else.
    This checks the compositions logogram.toys records, which other methods' tests use."""
    toy = toys.get_toy(name)
    if receiver is None:
        receivers: list[dict[str, Any]] = [{"kind": "logits"}]
        target: tuple[tuple[int, int] | None, str | None] = (None, None)
    else:
        layer, head, part = receiver
        receivers = [{"kind": "head", "layer": layer, "head": head, "input": part}]
        target = ((layer, head), part)
    expect = {
        f"L{c.sender[0]} H{c.sender[1]}": 1.0
        for c in toy.compositions
        if c.sender is not None and (c.receiver, c.input) == target and c.changes
    }
    experiment = {
        "kind": "path_patching",
        "direction": "clean_to_corrupt",
        "receivers": receivers,
        "freeze_mlps": False,
    }
    spec = _spec(name, _experiment(name, "which-heads"), experiment=experiment, limit=16)
    _assert_effects(_run(toy_backend(name), _project(tmp_path, name), spec), expect)


def test_every_planted_head_is_in_the_compositions():
    """Every input of every planted head that reads the residual stream is listed (the
    S-inhibition head's query is a constant: it attends wherever a repeat is flagged)."""
    for toy in toys.TOYS.values():
        for h in toy.heads:
            reads = {c.input for c in toy.compositions if c.receiver == (h.layer, h.head)}
            constant = {"q"} if h.kind == "s_inhibition" else set()
            assert reads == {"q", "k", "v"} - constant, h.label
        writes = {c.sender for c in toy.compositions if c.sender is not None}
        assert writes == {(h.layer, h.head) for h in toy.heads}
        assert [c for c in toy.compositions if c.receiver is None and c.changes]


# -- attribution patching: exact where linear, blind through saturated attention ---------------


def test_integrated_gradients_recover_the_previous_token_head_with_fine_steps(
    toy_backend, tmp_path
):
    """One gradient estimates the previous-token head at 0 (its effect passes through the
    induction head's saturated attention); integrated gradients see the attention move from C to
    B as the input goes from one prompt to the other, once their steps are finer than the move
    (about a fortieth of the way)."""
    project = _project(tmp_path, "induction")
    backend = toy_backend("induction")
    estimates = {}
    for steps in (None, 8, 16, 32):
        experiment = {
            "kind": "attribution_patching",
            "direction": "clean_to_corrupt",
            "method": "gradient" if steps is None else "integrated_gradients",
            "steps": steps,
        }
        spec = _spec(
            "induction", _experiment("induction", "which-heads"), experiment=experiment, limit=8
        )
        effects = _effects(_run(backend, project, spec))
        assert effects["L1 H1"] == pytest.approx(1.0, abs=TOL)  # linear: exact at any steps
        estimates[steps] = effects["L0 H2"]
    assert abs(estimates[None]) < 1e-3
    assert estimates[8] < estimates[16] < estimates[32]
    assert estimates[8] < 0.6 and estimates[32] > 0.99


def test_ioi_upstream_heads_stay_missed_by_integrated_gradients(toy_backend, tmp_path):
    """In the IOI toy the induction and S-inhibition heads act through two saturated softmaxes in
    a row, which flip within a hundredth of the way from one prompt to the other: 32 steps of
    integrated gradients step over it and estimate them near 0, while patching gives 1. Only the
    name mover, whose output reaches the logits linearly, is estimated right."""
    experiment = {
        "kind": "attribution_patching",
        "direction": "clean_to_corrupt",
        "method": "integrated_gradients",
        "steps": 32,
    }
    spec = _spec("ioi", _experiment("ioi", "which-heads"), experiment=experiment, limit=8)
    effects = _effects(_run(toy_backend("ioi"), _project(tmp_path, "ioi"), spec))
    assert effects["L3 H0"] == pytest.approx(1.0, abs=TOL)
    assert abs(effects["L1 H1"]) < 0.05 and abs(effects["L2 H3"]) < 0.05


# -- ablation, direct logit attribution and steering ------------------------------------------


def test_zero_ablating_every_head_leaves_no_answer(toy_backend, tmp_path):
    """With every head's output zeroed nothing is copied into the logits: every logit is 0, so
    the logit difference falls from the clean prompt's 10 to 0, half the way to the corrupt
    prompt's −10 (0.5); zeroing the heads that aren't planted changes nothing."""
    toy = toys.get_toy("ioi")
    experiment = _experiment("ioi", "is-the-circuit-all")
    spec = _spec(
        "ioi",
        experiment,
        experiment={"kind": "ablation", "baseline": {"kind": "zero"}},
        limit=16,
    )
    summary = _run(toy_backend(toy.name), _project(tmp_path, toy.name), spec)
    _assert_effects(
        summary, {"every head replaced": 0.5, "only the other heads kept": 0.5}, tolerance=TOL
    )


def test_direct_logit_attribution_adds_up_to_the_logit_difference(toy_backend, tmp_path):
    """The name mover writes the whole logit difference; the other heads, the embeddings, the
    MLPs and the biases write none of it."""
    experiment = _experiment("ioi", "direct-effects")
    summary = _run(toy_backend("ioi"), _project(tmp_path, "ioi"), toys.spec_of("ioi", experiment))
    direct = summary["direct"]
    assert direct["logit_diff"] == pytest.approx(10.0, abs=1e-4)
    assert direct["attention"] == pytest.approx(10.0, abs=1e-4)
    for part in ("embeddings", "mlp", "biases"):
        assert abs(direct[part]) < 1e-4, part


def test_steering_prompts_of_both_orders_does_nothing(toy_backend, tmp_path):
    """S-inhibition carries which name slot repeats, the opposite way in ABBA and BABA prompts:
    on a dataset that mixes them the mean difference cancels, the direction moves nothing, and
    the run says it doesn't beat its random control (as in GPT-2 small)."""
    experiment = _experiment("ioi", "steer-the-slot")
    spec = _spec("ioi", experiment, dataset="ioi.jsonl")
    summary = _run(toy_backend("ioi"), _project(tmp_path, "ioi"), spec)
    assert max(abs(v) for v in _effects(summary).values()) < 0.01
    assert not any(c["beats_control"] for c in summary["steering"]["control"])
    assert any("random control" in w for w in summary["warnings"])


# -- for people: the project, the command, the Python API -------------------------------------


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Logogram's settings folder (where local models are listed) and projects folder, for this
    test only."""
    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("platformdirs.user_documents_path", lambda *a, **k: tmp_path / "docs")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    return tmp_path / "config"


def test_a_toy_project_holds_its_model_datasets_and_experiments(settings, tmp_path):
    made = toys.write_toy("ioi", tmp_path / "my-toy")
    root = made.project.root
    assert json.loads((root / "project.json").read_text(encoding="utf-8"))["toy"] == "ioi"
    assert made.registered == "added" and made.local_name == "toy-ioi"
    (model,) = local_models.list_models()
    assert model.name == "toy-ioi" and model.sha256 == toys.get_toy("ioi").sha256
    assert Path(model.folder) == made.model_folder
    assert made.datasets == ["datasets/ioi.jsonl", "datasets/ioi-abba.jsonl"]
    listed = {r.id: r for r in made.project.list_runs()}
    assert set(listed) == {e.id for e in toys.experiments("ioi")}
    assert all(r.status == "draft" and r.model_id == "toy-ioi" for r in listed.values())
    for path in made.experiments:
        spec = Spec.from_path(root / path)
        assert spec.dataset.sha256 is not None and spec.model.sha256 == model.sha256
    # Made again: up to date and nothing added twice; an experiment someone changed is kept.
    (root / made.experiments[0]).write_text(
        Spec.from_path(root / made.experiments[0]).model_copy(update={"name": "Mine"}).to_json(),
        encoding="utf-8",
    )
    again = toys.write_toy("ioi", root)
    assert again.registered == "already" and len(local_models.list_models()) == 1
    assert Spec.from_path(root / made.experiments[0]).name == "Mine"
    # A spec naming the toy on a machine without it says how to make it.
    local_models.remove_model("toy-ioi")
    assert "logogram toy ioi" in local_models.missing_message("toy-ioi", model.sha256)
    # A folder that holds something else, or another toy, is left alone.
    with pytest.raises(toys.ToyError, match="ioi toy's project, not the induction toy's"):
        toys.write_toy("induction", root)
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "notes.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(toys.ToyError, match="isn't empty"):
        toys.write_toy("induction", busy)
    with pytest.raises(toys.ToyError, match="Choose one of: induction, ioi"):
        toys.get_toy("bigram")


def test_the_toy_command_writes_checks_and_lists(settings, tmp_path):
    from logogram.cli import app

    runner = CliRunner()
    listed = runner.invoke(app, ["toy"])
    assert listed.exit_code == 0
    assert "induction: Induction toy" in listed.output and "ioi: IOI toy" in listed.output
    result = runner.invoke(app, ["toy", "induction", str(tmp_path / "toy")])
    assert result.exit_code == 0, result.output
    assert "added to the local models as toy-induction" in result.output
    assert "prefers the answer on 64 of 64 clean prompts" in result.output
    assert "L1 H1 induction head: attends 1.000" in result.output
    assert "logogram run" in result.output
    refused = runner.invoke(app, ["toy", "bigram"])
    assert refused.exit_code == 1 and "Choose one of: induction, ioi" in refused.output


def test_the_python_api_writes_loads_and_runs_a_toy(settings, tmp_path):
    import logogram as lg

    toy = lg.toy_model("induction", tmp_path / "toy")
    assert toy.check["ok"] and toy.check["accuracy"] == {"clean": 1.0, "corrupt": 1.0}
    assert toy.model.info.sha256 == toys.get_toy("induction").sha256
    assert [h["label"] for h in toy.heads] == ["L0 H2", "L1 H1"]
    assert {"sender": [0, 2], "receiver": [1, 1], "input": "k"}.items() <= next(
        c for c in toy.compositions if c["sender"] == [0, 2]
    ).items()
    run = toy.run("direct-effects")
    sites = {s["label"]: s["effect"]["mean"] for s in run.summary["sites"]}  # type: ignore[index]
    assert sites["L1 H1 @ last"] == pytest.approx(1.0, abs=TOL)
    with pytest.raises(lg.api.LogogramError, match="no experiment 'nope'"):
        toy.run("nope")


# -- the toy's own tokenizer ----------------------------------------------------------------------


def test_words_a_toy_does_not_know_are_not_single_tokens(toy_backend):
    """A toy's tokenizer is word-level: a word it doesn't know reads as its one unknown token,
    which stands for every such word, so it isn't a token of its own. Datasets generated for the
    toy keep only the words it knows, and it solves them."""
    from logogram.tasks import generate_task

    backend = toy_backend("ioi")
    assert backend.single_token_id(" Mary") is not None
    assert backend.single_token_id(" Zoe") is None
    assert backend.single_token_id(" Mary John") is None
    records = generate_task(
        "ioi",
        16,
        3,
        {"templates": ["reached", "arrived"]},
        single_token=lambda text: backend.single_token_id(text) is not None,
        token_count=lambda text: len(backend.tokenize(text, prepend_bos=False).ids),
    )
    check = toys.check_toy(backend, "ioi", records)
    assert check.ok and check.clean_correct == check.corrupt_correct == 16


def test_the_tokens_are_the_words(toy_backend):
    backend = toy_backend("induction")
    tokenized = backend.tokenize("apple river apple", prepend_bos=True)
    assert tokenized.tokens == ["<|endoftext|>", " apple", " river", " apple"]
    logits = backend.logits(torch.tensor([tokenized.ids]), None, 1)[0, -1]
    assert backend.token_str(int(logits.argmax())) == " river"  # it continues the repeat
