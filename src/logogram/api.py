"""Logogram from Python: run specs, read runs and their per-prompt values, compare and check them.

Everything here goes through the same code as the app and ``logogram run`` (the runner, the
robustness and verification specs, the comparison), so a run made from a notebook is the same run
the app would make, and it lands in the project like any other::

    import logogram as lg

    project = lg.open_project("~/Documents/Logogram/ioi-example")
    model = lg.load_model("openai-community/gpt2")   # or a local model, by the name it was added under
    spec = lg.load_spec(project.root / "experiments/ioi-head-patching/spec.json")
    run = lg.run(spec, project, model=model)
    run.sites()          # one row per site: mean effect, interval, q-value, ...
    run.per_prompt()     # one row per site and prompt: everything the statistics came from
    run                  # in a notebook: the method, the strongest sites and a heatmap
    run.rename("Name movers").star()   # as the history shows it; the spec never changes
    lg.delete_run(project, old_run)    # its folder and everything in it, for good
    seeds = lg.sweep(run, field="statistics.seed", values=[0, 1, 2, 3])  # one run per value
    seeds.table()        # the strongest sites at every value: do they keep their sign and size?
    lg.list_saes(model, "layer 8")    # published SAEs for the model, and what Logogram reads of them
    sae = lg.load_sae(release="gpt2-small-res-jb", sae_id="blocks.8.hook_resid_pre", model=model)
    toy = lg.toy_model("induction")   # a model whose circuit is known, with experiments to run
    toy.run("which-heads").sites()    # the planted heads at 1, every other head at 0
    found = lg.run(acdc_spec, project, model=model)   # ACDC over the graph scope's edges
    found.circuit()      # the edges it kept, the change each made when removed, its faithfulness
    lg.import_corpus(project, "reviews", "reviews.csv", text_field="review", label_field="stars")
    lg.top_examples(model, project, "reviews", {"kind": "neuron", "layer": 6, "neuron": 1234})
    pcs = lg.principal_components(model, project, "datasets/ioi.jsonl", "resid_post:6", mark="side")
    lg.save_direction(project, "pc1", pcs["components"][0], model=model, site="resid_post:6",
                      method="pca")
    acts = lg.activations(model, project, "reviews", ["resid_post:6"], "mean")  # NumPy arrays

Tables come back as pandas DataFrames when pandas is installed, and as pyarrow Tables otherwise.
"""

from __future__ import annotations

import html
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from logogram.project import Project, ProjectError
from logogram.schema import RunAnnotations
from logogram.spec import Spec

__all__ = [
    "Run",
    "Sweep",
    "activations",
    "add_local_model",
    "check_robustness",
    "compare_runs",
    "continue_text",
    "create_project",
    "delete_run",
    "edge_verification",
    "export_activations",
    "feature_writes",
    "generate",
    "head_kinds",
    "import_corpus",
    "list_corpora",
    "list_directions",
    "list_local_models",
    "list_runs",
    "list_saes",
    "load_model",
    "load_run",
    "load_sae",
    "load_spec",
    "load_sweep",
    "look",
    "mean_difference",
    "open_example",
    "open_project",
    "principal_components",
    "read_corpus",
    "read_direction",
    "remove_local_model",
    "run",
    "save_direction",
    "sweep",
    "top_examples",
    "toy_model",
    "verify_run",
    "write_corpus",
    "write_dataset",
]


class LogogramError(RuntimeError):
    """A run that didn't finish, or a request that can't be met; the message says why."""


def open_project(path: str | Path) -> Project:
    """Open a project folder (one with a project.json)."""
    return Project.open(path)


def create_project(parent: str | Path, name: str) -> Project:
    """Create a project folder called after ``name`` inside ``parent``."""
    return Project.create(parent, name)


def open_example(example: str = "ioi-gpt2", parent: str | Path | None = None) -> Project:
    """Copy a bundled example into a folder of its own inside ``parent`` (by default the projects
    folder the app uses, Documents/Logogram) and open it, as the app's examples do: a copy made
    before is opened as it is, never replaced. The examples are ``ioi-gpt2``,
    ``greater-than-gpt2``, ``factual-recall-gpt2`` and ``gendered-pronoun-gpt2``: a spec and a
    dataset each, made for GPT-2 small."""
    from logogram.project import open_example as copy

    try:
        return copy(None if parent is None else Path(parent).expanduser(), example=example)
    except ProjectError as exc:
        raise LogogramError(str(exc)) from exc


def load_model(
    model_id: str,
    *,
    revision: str | None = None,
    dtype: str = "float32",
    device: str = "auto",
    process_weights: bool = True,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    source: str | None = None,
) -> Any:
    """Download (once) and load a model, checked as the app checks it. Pass it to :func:`run` to
    reuse it across runs; every choice it was loaded with is recorded in each run's spec.

    ``model_id`` is a Hugging Face id, or the name a local model was added under (see
    :func:`add_local_model`): ``source`` says which ("hub" or "local"); by default a name this
    machine's local models have (it has no slash) is that local model. ``device`` is "auto",
    "cpu", "cuda" (the default GPU), "cuda:N" (one GPU of several) or "mps"."""
    from logogram.backends.transformer_lens import load_model as load
    from logogram.local_models import find_model
    from logogram.runner import configure_determinism

    sha256 = None
    if source is None:
        source = "local" if "/" not in model_id and find_model(name=model_id) else "hub"
    if source == "local":
        local = find_model(name=model_id)
        if local is None:
            raise LogogramError(
                f"No local model is called {model_id} on this machine. Add its folder first, "
                f"with add_local_model(path, {model_id!r})."
            )
        sha256 = local.sha256
    configure_determinism()
    return load(
        model_id,
        revision=revision,
        dtype=dtype,
        device=device,
        process_weights=process_weights,
        on_progress=on_progress,
        source=source,
        sha256=sha256,
    )


def add_local_model(
    path: str | Path, name: str, *, on_progress: Callable[[int, int, str], None] | None = None
) -> dict[str, Any]:
    """Add a model folder on this machine (config.json, the tokenizer's files and safetensors
    weights) under ``name``: its files are hashed (``on_progress(bytes done, bytes in all,
    file)``), and specs then name it by ``name`` and that hash, never by its folder. Load it with
    ``load_model(name)``. See :mod:`logogram.local_models` for how the hash is defined."""
    from logogram.local_models import add_model

    return add_model(path, name, on_progress).to_dict()


def list_local_models() -> list[dict[str, Any]]:
    """The models added from folders on this machine: name, folder, hash and size."""
    from logogram.local_models import list_models

    return [m.to_dict() for m in list_models()]


def remove_local_model(name: str) -> None:
    """Forget a local model. Its folder and files are left as they are."""
    from logogram.local_models import remove_model

    remove_model(name)


def toy_model(
    name: str,
    folder: str | Path | None = None,
    *,
    device: str = "cpu",
    dtype: str = "float32",
) -> ToyModel:
    """A model whose circuit is known, to learn the methods on and to check them against:
    ``"induction"`` (a previous-token head and an induction head) or ``"ioi"`` (GPT-2 small's
    indirect-object circuit, one head per step), with weights set by hand (see
    :mod:`logogram.toys`). Writes its project into ``folder`` (by default ``toy-NAME`` in the
    projects folder): the model's files in ``model/``, added to this machine's local models as
    ``toy-NAME`` so specs name it by the hash of its files, its datasets, and experiments whose
    notes say what each method should find. Then loads it, checked as any model is, and checks
    that it does its task and that its planted heads attend where they were built to; a toy that
    doesn't is refused. Run its experiments with ``toy.run("which-heads")``."""
    from logogram.backends.base import BackendError
    from logogram.toys import ToyError, load_toy, write_toy

    try:
        made = write_toy(name, folder)
        model, check = load_toy(made.toy.name, made.local_name, device=device, dtype=dtype)
    except (ToyError, BackendError) as exc:
        raise LogogramError(str(exc)) from exc
    specs = {
        Path(path).parent.name: load_spec(made.project.root / path) for path in made.experiments
    }
    return ToyModel(
        name=made.toy.name,
        project=made.project,
        model=model,
        specs=specs,
        heads=[h.to_dict() for h in made.toy.heads],
        compositions=[c.to_dict() for c in made.toy.compositions],
        check=check,
    )


@dataclass
class ToyModel:
    """A toy model, written and loaded by :func:`toy_model`: its project, the loaded model, its
    experiments' specs (by their folder in experiments/, each saying in its notes what it should
    find), the heads that were planted and the compositions between them (which head's output
    reaches which input of which head), and what the check at load found."""

    name: str
    project: Project
    model: Any = field(repr=False)
    specs: dict[str, Spec] = field(repr=False)
    heads: list[dict[str, Any]] = field(repr=False)
    compositions: list[dict[str, Any]] = field(repr=False)
    check: dict[str, Any] = field(repr=False)

    def run(self, experiment: str) -> Run:
        """Run one of the toy's experiments (``"which-heads"``, ...) on the loaded toy."""
        if experiment not in self.specs:
            raise LogogramError(
                f"The {self.name} toy has no experiment {experiment!r}. Its experiments are "
                f"{', '.join(self.specs)}."
            )
        return run(self.specs[experiment], self.project, model=self.model)


def _chat_template(value: Any) -> Any:
    """A spec's tokenization.chat_template from a dict (as a spec writes it), or None."""
    if value is None or not isinstance(value, dict):
        return value
    from logogram.spec import read_chat_template

    return read_chat_template(value)


def load_sae(
    repo: str | None = None,
    path: str = "",
    *,
    model: Any,
    revision: str | None = None,
    release: str | None = None,
    sae_id: str | None = None,
) -> Any:
    """Download (once) and read a published SAE or transcoder at an exact revision, onto
    ``model``'s device: by its Hugging Face repository and folder, or by its SAELens release and
    id (``release="gpt2-small-res-jb", sae_id="blocks.8.hook_resid_pre"``), found in the
    catalogue of SAELens's registry that ships with Logogram (see :func:`list_saes`). Either way
    the SAE is its repository, folder and revision, which is what a spec records; without a
    revision, the repository's current commit is pinned. It is refused if Logogram can't read it
    exactly, was made for another model's sizes or reads a site the model doesn't have. Pass it to
    :func:`run` with a spec that names it, or explore its features with :func:`feature_writes`."""
    from logogram.features import check_sae_fits
    from logogram.sae import load_sae as load
    from logogram.spec import SAERef

    if release is not None or sae_id is not None:
        if repo is not None or path:
            raise LogogramError(
                "Name the SAE either by its repository and folder or by its release and id."
            )
        if release is None or sae_id is None:
            raise LogogramError("Name an SAE by its release and its id: give both.")
        from logogram.backends.catalogue import load_catalogue

        entry = load_catalogue().find(release, sae_id)
        if entry.reads == "refused":
            raise LogogramError(f"Logogram can't read {release} {sae_id} exactly. {entry.reason}")
        repo, path = entry.repo, entry.folder
    elif repo is None:
        raise LogogramError(
            "Name the SAE: its Hugging Face repository (and folder), or release= and sae_id= as "
            "SAELens names it."
        )
    sae = load(SAERef(repo=repo, path=path, revision=revision), model.info.device)
    check_sae_fits(sae, model.info)
    return sae


def list_saes(model: Any, search: str = "") -> Any:
    """The published SAEs trained on a model (a loaded model, or its Hugging Face id), from the
    catalogue of SAELens's registry that ships with Logogram (nothing is fetched): one row per SAE,
    those Logogram reads exactly first, with its release and id, repository and folder, the hook
    it reads, its site, layer and width, its architecture, whether Logogram reads it exactly
    (``reads``: exact, refused or unknown) and if not why (``reason``), and how it was checked on
    its files if it was (``verified``). ``search`` keeps the ones it matches, as the app's search
    does: ``"layer 8"``, ``"16k"``, ``"resid_post"``, a release's name. Load one with
    :func:`load_sae` (``release=``, ``sae_id=``)."""
    from logogram.backends.catalogue import load_catalogue, order
    from logogram.backends.catalogue import search as matching

    model_id = model if isinstance(model, str) else model.info.id
    entries = load_catalogue().for_model(model_id)
    found = order(matching(entries, search) if search.strip() else entries)
    return _table([e.to_dict() for e in found])


def feature_writes(
    model: Any,
    sae: Any,
    feature: int,
    prompt: str,
    *,
    prepend_bos: bool,
    position: int | None = None,
    top: int = 10,
    chat_template: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The tokens a feature promotes and suppresses directly, as the app's Features view shows
    them: one unit of the feature's decoder direction, added to the residual stream entering the
    final normalization at one token of ``prompt``, read through that normalization with its
    scale held at its value there and through the unembedding, skipping every later layer. Each
    token's logit change per unit of the feature, relative to the mean change over the
    vocabulary. ``position`` is the token (negative counts from the end); by default, where the
    feature fires most in the prompt. ``chat_template`` wraps the prompt in the model's own
    template, as a spec's tokenization.chat_template does (with ``prepend_bos`` false). See
    :func:`logogram.features.feature_writes`."""
    from logogram.chat import chat_wrapper
    from logogram.features import feature_writes as writes
    from logogram.prompts import first_real_token, tokenize_prompt

    template = _chat_template(chat_template)
    with model.lock:
        tokenized = tokenize_prompt(model, prompt, prepend_bos, chat_wrapper(model, template))
        report = writes(
            model,
            sae,
            feature,
            tokenized.ids,
            position=position,
            first=first_real_token(model, tokenized.ids, prepend_bos, template is not None),
            top=top,
        )
    return {**report, "tokens": tokenized.tokens}


def load_spec(data: dict[str, Any] | str | Path) -> Spec:
    """A spec from a dict, a JSON string or a spec.json file. Version 1 specs are read too; the
    run they make lists the values they left out."""
    if isinstance(data, dict):
        return Spec.model_validate(data)
    text = str(data)
    if text.lstrip().startswith("{"):
        return Spec.model_validate_json(text)
    return Spec.from_path(Path(text).expanduser())


def run(
    experiment: Spec | dict[str, Any],
    project: Project,
    *,
    model: Any = None,
    sae: Any = None,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> Run:
    """Run a spec and write its experiment folder into ``project``, as the app does. ``model``
    is reused when it matches the spec (otherwise the spec's model is loaded)."""
    from logogram.runner import run_spec

    experiment = load_spec(experiment) if isinstance(experiment, dict) else experiment
    outcome = run_spec(experiment, project, backend=model, sae=sae, on_event=on_event)
    if outcome.status != "finished":
        raise LogogramError(_ended(outcome, "The run"))
    return load_run(project, outcome.run_id, model=outcome.backend)


def head_kinds(
    model: Any,
    *,
    seed: int = 0,
    sequences: int = 32,
    length: int = 32,
    prepend_bos: bool = True,
    batch_size: int = 16,
    bootstrap: int = 1000,
    ci: float = 0.95,
    chat_template: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score every head of ``model`` (from :func:`load_model`) as a previous-token,
    duplicate-token and induction head, on ``sequences`` seeded random sequences of ``length``
    tokens repeated twice. Returns, per kind, each head's mean score with its bootstrap interval
    over sequences, the per-sequence values, and the strongest heads; the app shows the same in
    the Attention view. ``chat_template`` (as a spec writes it, with ``prepend_bos`` false) puts
    each sequence where a prompt goes in the model's own chat template."""
    from logogram.heads import HeadKindSettings, head_kind_scores

    template = _chat_template(chat_template)
    settings = HeadKindSettings(
        seed=seed,
        sequences=sequences,
        length=length,
        prepend_bos=prepend_bos,
        batch_size=batch_size,
        bootstrap=bootstrap,
        ci=ci,
        chat_template=None if template is None else template.model_dump(mode="json"),
    )
    with model.lock:
        return head_kind_scores(model, settings)


def look(
    model: Any,
    text: str,
    *,
    prepend_bos: bool = True,
    top_k: int = 5,
    position: int | None = None,
    chat_template: dict[str, Any] | None = None,
    sae: Any = None,
) -> dict[str, Any]:
    """Read any text with ``model`` (from :func:`load_model`), without making a dataset: its
    tokens, the ``top_k`` most likely next tokens after each, and at ``position`` (the last token
    by default) what each layer would predict, as the app's Your text view shows. With
    ``chat_template`` (as a spec writes it, such as ``{"kind": "user_turn", "system": None}``,
    with ``prepend_bos`` false), the text is wrapped in the model's own chat template. With
    ``sae`` (from :func:`load_sae`), ``features`` also lists the ``top_k`` features that fire most
    on each token, with their activations, and ``active`` how many fire there (SAEs aren't
    trained on the beginning-of-sequence token, so read from ``first_real_token`` on)."""
    from logogram.inside import look_features, look_report

    template = _chat_template(chat_template)
    with model.lock:
        report = look_report(
            model,
            text,
            prepend_bos=prepend_bos,
            top_k=top_k,
            position=position,
            chat_template=template,
        )
        if sae is not None:
            found = look_features(
                model, sae, text, prepend_bos=prepend_bos, top_k=top_k, chat_template=template
            )
            report.update(
                features=found["features"],
                active=found["active"],
                first_real_token=found["first_real_token"],
            )
    return report


def continue_text(
    model: Any,
    text: str,
    *,
    tokens: int,
    prepend_bos: bool = True,
    sae: Any = None,
    feature: int | None = None,
    activation: float | None = None,
    start: int | None = None,
) -> dict[str, Any]:
    """What ``model`` writes after ``text``: ``tokens`` new tokens (1 to 64), each the most likely
    one (ties to the lowest token id), as the app's Your text view continues it (``plain``). With
    ``sae`` (from :func:`load_sae`) and ``feature``, also a continuation with that feature added
    where the SAE writes (``steered``): ``activation`` units of it, in the feature's own units (its
    decoder row times ``activation``), at every position from token ``start`` on (the first real
    token by default), the new tokens included. ``steer`` says what was added, with the feature's
    largest activation on the text for reference."""
    from logogram.inside import continue_text as go

    with model.lock:
        return go(
            model,
            text,
            prepend_bos=prepend_bos,
            tokens=tokens,
            sae=sae,
            feature=feature,
            activation=activation,
            start=start,
        )


def _ended(outcome: Any, what: str) -> str:
    """Why a run didn't finish, what it kept of what finished, and where its traceback is."""
    message = f"{what} {outcome.status}: {outcome.manifest.get('error')}"
    partial = (outcome.summary or {}).get("partial")
    if partial:
        message += f" {partial['note']} load_run(project, {outcome.run_id!r}) reads them."
    if outcome.traceback:
        message += (
            f" Its traceback is in experiments/{outcome.run_id}/traceback.txt "
            f"(load_run(project, {outcome.run_id!r}).traceback)."
        )
    return message


def list_runs(project: Project) -> list[dict[str, Any]]:
    """Every run and draft in the project, newest first, as the history lists them: archived runs
    too (``archived``), and runs whose spec can't be read (``damaged``, with ``error`` saying why)."""
    return [r.to_dict() for r in project.list_runs()]


def delete_run(project: Project, run: Run | str) -> None:
    """Delete a finished, failed or cancelled run, or a draft (a :class:`Run` or a run id): its
    folder and everything in it, for good. Archive a run to keep it out of the way instead. A run
    in progress can't be deleted. Runs derived from it stay, and keep its id in their manifest."""
    from logogram.history import delete_run as delete

    if isinstance(run, Run) and run.project.root != project.root:
        raise LogogramError(f"{run.id} belongs to another project.")
    try:
        delete(project, run.id if isinstance(run, Run) else run)
    except ProjectError as exc:
        raise LogogramError(str(exc)) from exc


def load_run(project: Project, run_id: str, *, model: Any = None) -> Run:
    """A run of ``project`` by its id (the name of its experiments/ folder)."""
    folder = project.run_dir(run_id)
    try:
        specification = Spec.from_path(project.readable(folder / "spec.json"))
    except ProjectError as exc:
        raise LogogramError(f"There is no run {run_id} in this project.") from exc
    summary = project.read_json(folder / "summary.json", optional=True)
    manifest = project.read_json(folder / "manifest.json", optional=True) or {}
    return Run(
        project=project,
        id=run_id,
        spec=specification,
        summary=summary,
        manifest=manifest,
        model=model,
    )


def compare_runs(a: Run, b: Run) -> dict[str, Any]:
    """Compare two finished runs of one project: rank correlation, top-k overlap, conclusions
    that changed, and paired differences where they measured the same prompts."""
    from logogram import compare

    if a.project.root != b.project.root:
        raise LogogramError("Both runs must belong to the same project.")
    return compare.compare_runs(a.project, a.id, b.id)


def verify_run(estimated: Run, top: int = 10, *, model: Any = None) -> Run:
    """Patch, for real, the ``top`` sites an attribution patching run estimated to matter most;
    for an edge attribution run, path patch its ``top`` strongest edges one at a time, at most 50
    (compare the two with :func:`edge_verification`)."""
    from logogram.robustness import require_finished
    from logogram.runner import run_spec
    from logogram.verify import verification_change, verification_spec

    try:
        require_finished(estimated.project, estimated.id, "Verifying by patching")
    except ValueError as exc:
        raise LogogramError(str(exc)) from exc
    if estimated.summary is None:
        raise LogogramError("Only a finished run can be verified.")
    try:
        variant = verification_spec(estimated.spec, estimated.summary, top)
    except ValueError as exc:
        raise LogogramError(str(exc)) from exc
    outcome = run_spec(
        variant,
        estimated.project,
        backend=model or estimated.model,
        derived_from={
            "run": estimated.id,
            "kind": "verification",
            "change": verification_change(variant),
        },
    )
    if outcome.status != "finished":
        raise LogogramError(_ended(outcome, "The verification"))
    return load_run(estimated.project, outcome.run_id, model=outcome.backend)


def edge_verification(estimated: Run, verified: Run) -> dict[str, Any]:
    """An edge attribution run's estimates beside their measurements by path patching (the run
    :func:`verify_run` made from it), edge by edge, ranked both ways, with Spearman's rank
    correlation of the two and how many have the same sign."""
    from logogram.edges import verification_table

    if not (estimated.summary or {}).get("edges"):
        raise LogogramError(f"{estimated.id} isn't an edge attribution run.")
    if verified.summary is None:
        raise LogogramError(f"{verified.id} has no results ({verified.status}).")
    return verification_table(estimated.summary or {}, verified.summary)


def check_robustness(
    original: Run,
    *,
    experiment: dict[str, Any] | None = None,
    dtype: str | None = None,
    model: Any = None,
) -> Run:
    """Rerun with one choice changed (the experiment, or the dtype), as Check robustness does."""
    from logogram.robustness import require_finished, robustness_spec
    from logogram.runner import run_spec

    try:
        require_finished(original.project, original.id, "Checking robustness")
    except ValueError as exc:
        raise LogogramError(str(exc)) from exc
    variant, change = robustness_spec(original.spec, experiment=experiment, dtype=dtype)
    outcome = run_spec(
        variant,
        original.project,
        backend=model or original.model,
        derived_from={"run": original.id, "kind": "robustness", "change": change},
    )
    if outcome.status != "finished":
        raise LogogramError(_ended(outcome, "The rerun"))
    return load_run(original.project, outcome.run_id, model=outcome.backend)


def sweep(
    experiment: Run | Spec | dict[str, Any] | str | Path,
    project: Project | None = None,
    *,
    field: str,
    values: Sequence[Any],
    model: Any = None,
    sae: Any = None,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> Sweep:
    """Run a run's spec (or a spec, into ``project``) across ``values`` of one ``field``, one run
    after another, as ``logogram sweep`` does: a seed the spec states (``statistics.seed``,
    ``experiment.baseline.seed``, ``experiment.seed`` or ``scope.seed``), the dataset
    (``dataset.path``: dataset files of the project) or the model's revision (``model.revision``:
    commits, tags or branches, each run pinning its commit). A finished run stands in for its own
    value, which isn't run again. Returns the runs, one per value in the order given; a run that
    doesn't finish is kept with its status, and the next value runs."""
    from logogram.sweeps import SweepError, plan_sweep, run_sweep

    original = None
    if isinstance(experiment, Run):
        if project is not None and project.root != experiment.project.root:
            raise LogogramError(f"{experiment.id} belongs to another project.")
        project = experiment.project
        specification = experiment.spec
        original = project.run_listing(experiment.id)
        model = model or experiment.model
    else:
        if project is None:
            raise LogogramError(
                "Name the project the runs go in: lg.sweep(spec, project, field=..., values=...)."
            )
        specification = experiment if isinstance(experiment, Spec) else load_spec(experiment)
    try:
        plan = plan_sweep(project, specification, field, values, original=original)
    except SweepError as exc:
        raise LogogramError(str(exc)) from exc
    outcomes, last = run_sweep(plan, project, backend=model, on_event=on_event, sae=sae)
    runs = []
    for outcome in outcomes:
        if outcome is None:  # the run it was made from stands in for this value
            assert isinstance(experiment, Run)
            runs.append(experiment)
            continue
        # Only a model that is still loaded: one a revision sweep moved on from is closed.
        kept = outcome.backend if outcome.backend is last or outcome.backend is model else None
        runs.append(load_run(project, outcome.run_id, model=kept))
    return Sweep(runs, project=project, id=plan.id, field=plan.field.path, values=plan.values)


def load_sweep(project: Project, sweep_id: str, *, model: Any = None) -> Sweep:
    """A sweep of ``project`` by its id, as its runs record it: its runs in the order of its values
    (a value whose run is gone, or never ran, has none)."""
    from logogram.sweeps import SweepNotFound, get_sweep

    try:
        found = get_sweep(project, sweep_id)
    except SweepNotFound as exc:
        raise LogogramError(str(exc)) from exc
    runs = [load_run(project, found.runs[i].id, model=model) for i in sorted(found.runs)]
    return Sweep(runs, project=project, id=found.id, field=found.field.path, values=found.values)


def generate(
    task: str,
    n: int,
    seed: int,
    options: dict[str, Any] | None = None,
    *,
    model: Any = None,
) -> list[Any]:
    """Prompt pairs for a task (see ``logogram.tasks.TASKS``). With ``model``, only words that are
    single tokens for it are used, and clean and corrupt prompts have the same length in it."""
    from logogram.tasks import generate_task

    single = None
    count = None
    if model is not None:
        single = lambda text: model.single_token_id(text) is not None  # noqa: E731
        count = lambda text: len(model.tokenize(text, prepend_bos=False).ids)  # noqa: E731
    return generate_task(task, n, seed, options or {}, single_token=single, token_count=count)


def write_dataset(
    project: Project, name: str, records: list[Any], *, overwrite: bool = False
) -> str:
    """Write prompt pairs into the project's datasets/ folder; returns its path in the project."""
    from logogram.datasets import check_dataset_name
    from logogram.datasets import write_dataset as write

    path = project.dataset_file(check_dataset_name(name))
    if path.exists() and not overwrite:
        raise LogogramError(f"{path.name} already exists. Pass overwrite=True to replace it.")
    write(path, records)
    return path.relative_to(project.root).as_posix()


# -- corpora, top activations, representations and directions -----------------------------------


def list_corpora(project: Project) -> list[dict[str, Any]]:
    """The project's corpora (plain texts in its ``corpora/`` folder), by name: how many texts
    each has, its labels, the hash of its file and where its texts came from; ``error`` says why
    one can't be read. Import one with :func:`import_corpus` or write one with
    :func:`write_corpus`."""
    from logogram.corpora import list_corpora as listed

    return listed(project)


def read_corpus(project: Project, name: str, sha256: str | None = None) -> Any:
    """A corpus of the project: its ``texts``, its ``records`` (each text with its label, id and
    meta), ``labels()`` (the distinct labels, as text), the ``sha256`` of its file and ``meta``
    (where its texts came from). With ``sha256``, it is refused unless its file still has that
    hash."""
    from logogram.corpora import CorpusError
    from logogram.corpora import read_corpus as read

    try:
        return read(project, name, sha256)
    except CorpusError as exc:
        raise LogogramError(str(exc)) from exc


def write_corpus(
    project: Project,
    name: str,
    texts: Sequence[str],
    labels: Sequence[Any] | None = None,
    *,
    overwrite: bool = False,
) -> Any:
    """Write ``texts`` as the corpus ``name``, each with the label at the same place in
    ``labels`` (a string, a whole number or true/false) when they are given; empty texts are left
    out. Returns the corpus (as :func:`read_corpus` does)."""
    from logogram.corpora import CorpusError
    from logogram.corpus_import import write_texts

    try:
        given = None if labels is None else list(labels)
        return write_texts(project, name, list(texts), given, overwrite=overwrite).corpus
    except CorpusError as exc:
        raise LogogramError(str(exc)) from exc


def import_corpus(
    project: Project,
    name: str,
    path: str | Path | None = None,
    *,
    repo: str | None = None,
    file: str | None = None,
    revision: str | None = None,
    dataset: str | None = None,
    sides: str = "both",
    label: str | None = None,
    format: str | None = None,
    split: str = "lines",
    text_field: str = "text",
    label_field: str | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> Any:
    """Import texts as the corpus ``name``, from one of three places:

    * ``path``, a file on this machine: Parquet, JSON Lines, a JSON array, CSV, TSV or text
      (``format`` says which when its name doesn't; a text file is split into ``"lines"`` or
      ``"paragraphs"``, as ``split`` says). The corpus records the file's name and hash, never
      its folder.
    * ``repo`` and ``file``, a file of a Hugging Face dataset, downloaded once at the exact
      commit ``revision`` resolves to (the main branch when None), which the corpus records.
    * ``dataset``, a project dataset's prompt pairs (``"datasets/ioi.jsonl"``): their
      ``sides`` (``"clean"``, ``"corrupt"`` or ``"both"``), labelled by ``label``: ``"side"``,
      or ``"meta:FIELD"`` for a field of their meta.

    ``text_field`` and ``label_field`` name the field or column that holds each text and its
    label (no label when None); rows without text are left out. ``limit`` keeps the first texts
    (the first pairs, for a dataset). Returns the corpus (as :func:`read_corpus` does)."""
    from logogram.backends.base import BackendError
    from logogram.corpora import CorpusError
    from logogram.corpus_import import import_file, import_hub, import_prompts

    given = [path is not None, repo is not None, dataset is not None]
    if sum(given) != 1:
        raise LogogramError(
            "Import from one place: a file's path, a Hugging Face dataset's repo and file, or a "
            "project dataset."
        )
    try:
        if path is not None:
            done = import_file(
                project,
                name,
                Path(path),
                fmt=format,  # type: ignore[arg-type]
                split=split,
                text_field=text_field,
                label_field=label_field,
                limit=limit,
                overwrite=overwrite,
            )
        elif repo is not None:
            if not file:
                raise LogogramError("Name the file of the dataset to import: file=...")
            done = import_hub(
                project,
                name,
                repo=repo,
                path=file,
                revision=revision,
                text_field=text_field,
                label_field=label_field,
                limit=limit,
                split=split,
                overwrite=overwrite,
            )
        else:
            done = import_prompts(
                project,
                name,
                dataset=str(dataset),
                sides=sides,  # type: ignore[arg-type]
                label=label,
                limit=limit,
                overwrite=overwrite,
            )
    except (CorpusError, BackendError, ProjectError) as exc:
        raise LogogramError(str(exc)) from exc
    return done.corpus


def _rows(
    source: Any,
    *,
    sides: str,
    mark: str | None,
    limit: int | None,
    max_tokens: int,
) -> Any:
    """What to read, from a dict as the app sends it, a dataset's path (``"datasets/x.jsonl"``)
    or a corpus's name."""
    from pydantic import TypeAdapter

    from logogram.representations import CorpusRows, DatasetRows, Rows

    if isinstance(source, DatasetRows | CorpusRows):
        return source
    if isinstance(source, dict):
        return TypeAdapter(Rows).validate_python(source)
    text = str(source).replace("\\", "/")
    if text.startswith("corpora/") and text.endswith(".jsonl"):
        return CorpusRows(
            name=text[len("corpora/") : -len(".jsonl")], limit=limit, max_tokens=max_tokens
        )
    if text.endswith(".jsonl"):
        return DatasetRows(path=text, sides=sides, mark=mark, limit=limit)  # type: ignore[arg-type]
    return CorpusRows(name=text, limit=limit, max_tokens=max_tokens)


def _sites(sites: Any) -> list[Any]:
    """Sites as ``"KIND:LAYER"``, one or a list of them (or dicts, as the app sends them)."""
    from logogram.representations import parse_site

    try:
        return [parse_site(s) for s in ([sites] if isinstance(sites, str | dict) else sites)]
    except ValueError as exc:
        raise LogogramError(str(exc)) from exc


def _read(model: Any, project: Project, source: Any, prepend_bos: bool, **choices: Any) -> Any:
    from logogram.corpora import CorpusError
    from logogram.representations import load_rows

    try:
        with model.lock:
            return load_rows(model, project, _rows(source, **choices), prepend_bos)
    except (CorpusError, ProjectError, ValueError) as exc:
        raise LogogramError(str(exc)) from exc


def top_examples(
    model: Any,
    project: Project,
    corpus: str,
    unit: dict[str, Any],
    *,
    sae: Any = None,
    texts: int | None = None,
    max_tokens: int = 128,
    prepend_bos: bool = True,
    batch_size: int = 16,
    top_k: int = 20,
    context: int = 8,
) -> dict[str, Any]:
    """Where a unit is most active in the corpus ``corpus``, as the app's Corpus view shows it.
    ``unit`` is ``{"kind": "neuron", "layer": 3, "neuron": 12}``, ``{"kind": "sae_feature",
    "feature": 1234}`` (a feature of ``sae``, from :func:`load_sae`), ``{"kind": "direction",
    "name": "sides"}`` (a saved direction, read at its site, or at ``"site": {"kind":
    "resid_post", "layer": 6}``) or ``{"kind": "head", "layer": 9, "head": 9}`` (the length of
    what the head writes into the residual stream).

    The first ``texts`` texts (every text when None) are read, each cut to its first
    ``max_tokens`` tokens, after the beginning-of-sequence token with ``prepend_bos`` (that token
    is never counted). Returns the ``top_k`` tokens where the unit is largest (``top``), each with
    ``context`` tokens on either side and the unit's value at each, for a signed unit (a direction
    of one dimension) the ``top_k`` most negative (``bottom``), a ``histogram`` of its values, the
    share of tokens where it is above zero (``active_share``), and the ``settings`` that made
    them. Ties go to the earlier text, then the earlier token."""
    from pydantic import TypeAdapter

    from logogram.corpora import CorpusError
    from logogram.directions import DirectionError, read_direction
    from logogram.top_activations import DirectionUnit, Unit
    from logogram.top_activations import top_examples as top

    try:
        chosen = TypeAdapter(Unit).validate_python(unit)
        direction = None
        if isinstance(chosen, DirectionUnit):
            direction = read_direction(project, chosen.name)
            if chosen.sha256 is not None and chosen.sha256 != direction.sha256:
                raise LogogramError(f"The direction {chosen.name} has changed (its hash differs).")
        found = read_corpus(project, corpus)
        with model.lock:
            return top(
                model,
                found,
                chosen,
                sae=sae,
                direction=direction,
                texts=texts,
                max_tokens=max_tokens,
                prepend_bos=prepend_bos,
                batch_size=batch_size,
                top_k=top_k,
                context=context,
            )
    except (CorpusError, DirectionError, ValueError) as exc:
        raise LogogramError(str(exc)) from exc


def activations(
    model: Any,
    project: Project,
    source: Any,
    sites: str | Sequence[str],
    position: Any = "last",
    *,
    prepend_bos: bool = True,
    batch_size: int = 16,
    sides: str = "both",
    mark: str | None = None,
    limit: int | None = None,
    max_tokens: int = 128,
) -> dict[str, Any]:
    """The activations at ``sites`` and ``position``, one row per prompt or text, as NumPy
    arrays ``[rows, width]`` in float64 by site (``"resid_post:6"``); nothing is written.

    ``source`` is a project dataset's path (``"datasets/ioi.jsonl"``: its ``sides``,
    ``"clean"``, ``"corrupt"`` or ``"both"``, each pair's clean prompt then its corrupt one) or a
    corpus's name (each text cut to its first ``max_tokens`` tokens); ``limit`` reads the first
    pairs or texts only. ``sites`` are ``"KIND:LAYER"`` (``resid_pre``, ``resid_mid``,
    ``resid_post``, ``attn_out`` or ``mlp_out``). ``position`` is ``"last"``, a token index
    (negative counts from the end), ``"label:NAME"`` (a labelled position of the prompts) or
    ``"mean"`` (the mean over the tokens after the beginning-of-sequence token). The same rows
    always run in the same batches, so they give the same numbers. Write them to the project
    with :func:`export_activations`."""
    from logogram.representations import collect, parse_position

    loaded = _read(
        model,
        project,
        source,
        prepend_bos,
        sides=sides,
        mark=mark,
        limit=limit,
        max_tokens=max_tokens,
    )
    chosen = _sites(sites)
    try:
        with model.lock:
            return collect(
                model,
                loaded,
                chosen,
                parse_position(position),
                prepend_bos=prepend_bos,
                batch_size=batch_size,
            )
    except ValueError as exc:
        raise LogogramError(str(exc)) from exc


def export_activations(
    model: Any,
    project: Project,
    name: str,
    source: Any,
    sites: str | Sequence[str],
    position: Any = "last",
    *,
    prepend_bos: bool = True,
    batch_size: int = 16,
    sides: str = "both",
    mark: str | None = None,
    limit: int | None = None,
    max_tokens: int = 128,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write the activations :func:`activations` reads to ``activations/NAME.safetensors`` in
    the project (one float32 tensor per site, ``[rows, width]``, named ``"kind:layer"``) and what
    they are to ``activations/NAME.json``: the model, the sites, the position, the dataset or
    corpus read and its hash, each row's index, side and class (``mark``: ``"side"`` or
    ``"meta:FIELD"`` for a dataset; a corpus's labels), and how texts were tokenized. Returns where
    they were written."""
    from logogram.representations import export_activations as export
    from logogram.representations import parse_position

    loaded = _read(
        model,
        project,
        source,
        prepend_bos,
        sides=sides,
        mark=mark,
        limit=limit,
        max_tokens=max_tokens,
    )
    chosen = _sites(sites)
    try:
        with model.lock:
            return export(
                project,
                name,
                model,
                loaded,
                chosen,
                parse_position(position),
                prepend_bos=prepend_bos,
                batch_size=batch_size,
                overwrite=overwrite,
            )
    except (ValueError, ProjectError) as exc:
        raise LogogramError(str(exc)) from exc


def principal_components(
    model: Any,
    project: Project,
    source: Any,
    site: Any,
    position: Any = "last",
    *,
    prepend_bos: bool = True,
    batch_size: int = 16,
    sides: str = "both",
    mark: str | None = None,
    limit: int | None = None,
    max_tokens: int = 128,
) -> dict[str, Any]:
    """The principal components of the activations at ``site`` and ``position`` (read as
    :func:`activations` reads them), as the app's Representations view shows them: the rows
    centered on their mean and decomposed in float64, each component signed so its
    largest-magnitude loading is positive. Returns ``components`` (``[k, width]``, at most ten),
    ``explained`` (each one's share of the variance), ``coords`` (``[rows, k]``), ``mean``,
    ``rows`` (each row's index, side, class and text) and ``settings``. Save a component as a
    direction with :func:`save_direction`."""
    from logogram.representations import (
        collect,
        parse_position,
        parse_site,
        position_words,
        settings_of,
        site_words,
        source_words,
    )
    from logogram.representations import principal_components as decompose

    loaded = _read(
        model,
        project,
        source,
        prepend_bos,
        sides=sides,
        mark=mark,
        limit=limit,
        max_tokens=max_tokens,
    )
    chosen, where = parse_site(site), parse_position(position)
    try:
        with model.lock:
            values = collect(
                model, loaded, [chosen], where, prepend_bos=prepend_bos, batch_size=batch_size
            )[chosen.name]
        found = decompose(values)
    except ValueError as exc:
        raise LogogramError(str(exc)) from exc
    return {
        "components": found.components,
        "explained": found.explained,
        "variances": found.variances,
        "coords": found.coords,
        "mean": found.mean,
        "rows": [
            {"index": r.index, "side": r.side, "mark": r.mark, "text": r.text} for r in loaded.rows
        ],
        "settings": settings_of(loaded, chosen, where, prepend_bos, batch_size),
        "description": f"{site_words(chosen.kind, chosen.layer)} at {position_words(where)}, "
        f"over {source_words(loaded.source)}",
    }


def mean_difference(
    model: Any,
    project: Project,
    source: Any,
    site: Any,
    position: Any,
    positive: str,
    negative: str,
    *,
    prepend_bos: bool = True,
    batch_size: int = 16,
    sides: str = "both",
    mark: str | None = None,
    limit: int | None = None,
    max_tokens: int = 128,
) -> Any:
    """The mean activation at ``site`` and ``position`` over the rows of class ``positive``
    minus the mean over those of class ``negative`` (read as :func:`activations` reads them; a
    dataset's rows take their class from ``mark``, ``"side"`` or ``"meta:FIELD"``, a corpus's from
    their labels), as a NumPy vector. Save it as a direction with :func:`save_direction`
    (``method="mean_difference"``)."""
    from logogram.representations import collect, parse_position, parse_site
    from logogram.representations import mean_difference as difference

    loaded = _read(
        model,
        project,
        source,
        prepend_bos,
        sides=sides,
        mark=mark,
        limit=limit,
        max_tokens=max_tokens,
    )
    chosen = parse_site(site)
    try:
        with model.lock:
            values = collect(
                model,
                loaded,
                [chosen],
                parse_position(position),
                prepend_bos=prepend_bos,
                batch_size=batch_size,
            )[chosen.name]
        vector, _, _ = difference(values, loaded, positive, negative)
    except ValueError as exc:
        raise LogogramError(str(exc)) from exc
    return vector


def list_directions(project: Project) -> list[dict[str, Any]]:
    """The directions saved in the project (its ``directions/`` folder), by name: the model and
    site each was found on, how many dimensions it has, how it was made (``made_by``: the method,
    the run and in words) and the hash of its tensor file; ``error`` says why one can't be
    read."""
    from logogram.directions import list_directions as listed

    return listed(project)


def read_direction(project: Project, name: str) -> Any:
    """A saved direction: its ``vectors`` (``[k, width]``, orthonormal rows, float32) and ``meta``
    (the model and site it was found on, how it was made, its hash)."""
    from logogram.directions import DirectionError
    from logogram.directions import read_direction as read

    try:
        return read(project, name)
    except DirectionError as exc:
        raise LogogramError(str(exc)) from exc


def save_direction(
    project: Project,
    name: str,
    vectors: Any,
    *,
    model: Any,
    site: Any,
    method: str = "import",
    detail: str = "",
    notes: str = "",
    overwrite: bool = False,
) -> Any:
    """Save ``vectors`` (one vector ``[width]``, or ``[k, width]`` for a subspace; a NumPy array,
    a tensor or a list) as the direction ``name``, made orthonormal, with the length of each
    vector before kept. ``model`` is the loaded model it was found on (from :func:`load_model`),
    ``site`` where (``"resid_post:6"``, or None), ``method`` how (``"pca"``,
    ``"mean_difference"``, ``"probe"``, ``"das"``, ``"sae_feature"`` or ``"import"``) and
    ``detail`` in words. Steering, ablation along it and the analyses that read directions use
    it by name. An existing direction of that name is replaced only with ``overwrite``."""
    import torch

    from logogram.backends.base import float64
    from logogram.directions import DirectionError, model_of
    from logogram.directions import save_direction as save
    from logogram.representations import parse_site

    chosen = None if site is None else parse_site(site)
    tensor = vectors if isinstance(vectors, torch.Tensor) else torch.as_tensor(vectors)
    try:
        return save(
            project,
            name,
            float64(tensor.detach()),
            model=model if isinstance(model, dict) else model_of(model.info),
            site=None if chosen is None else {"kind": chosen.kind, "layer": chosen.layer},
            method=method,  # type: ignore[arg-type]
            run=None,
            detail=detail or f"vectors saved from Python ({method})",
            notes=notes,
            replace=overwrite,
        )
    except (DirectionError, ValueError) as exc:
        raise LogogramError(str(exc)) from exc


def _table(rows: list[dict[str, Any]]) -> Any:
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except ImportError:
        import pyarrow as pa

        return pa.Table.from_pylist(rows)


@dataclass
class Run:
    """A run in a project: its spec, summary and manifest as written, and its per-prompt values.
    ``model`` is the loaded model it was run with, when there is one, for the runs made from
    it (:func:`verify_run`, :func:`check_robustness`, :func:`sweep`)."""

    project: Project
    id: str
    spec: Spec
    summary: dict[str, Any] | None
    manifest: dict[str, Any]
    model: Any = field(default=None, repr=False)

    @property
    def folder(self) -> Path:
        """The run's folder: experiments/ and its id, in the project."""
        return self.project.run_dir(self.id)

    @property
    def status(self) -> str:
        """``finished``, ``failed``, ``cancelled``, ``running``, or ``draft`` for a spec that
        hasn't run. A run whose process stopped before it finished reads as ``failed``."""
        status = str(self.manifest.get("status") or "draft")
        if status == "running":
            # Still going, here or in another process, or interrupted: the history knows which.
            listing = self.project.run_listing(self.id)
            return listing.status if listing is not None else status
        return status

    @property
    def traceback(self) -> str | None:
        """What a failed run kept of the error that stopped it (this machine's paths scrubbed)."""
        return self.project.run_traceback(self.id)

    @property
    def partial(self) -> dict[str, Any] | None:
        """For a run that was cancelled or failed mid-sweep and kept what finished: how far it
        got (``note``, such as "Stopped after 7 of 12 layers: these are the layers that
        finished.") and what it doesn't keep (``left_out``). Its sites, statistics and per-prompt
        values are those of what finished, as a finished run's are. None for a finished run."""
        return (self.summary or {}).get("partial")

    @property
    def title(self) -> str:
        """What the run is called: the title it was given (see :meth:`rename`), or its spec's
        name."""
        return self._annotations().title or self.spec.name

    @property
    def starred(self) -> bool:
        """Whether the run is starred (see :meth:`star`)."""
        return self._annotations().starred

    @property
    def archived(self) -> bool:
        """Whether the run is archived (see :meth:`archive`)."""
        return self._annotations().archived

    def rename(self, title: str | None) -> Run:
        """Give the run a title, shown instead of its spec's name (None or "" clears it). The spec
        never changes: the title is kept beside it, in annotations.json."""
        return self._annotate(title=title)

    def star(self, starred: bool = True) -> Run:
        """Star the run, so the history finds it first; ``star(False)`` removes the star."""
        return self._annotate(starred=starred)

    def archive(self, archived: bool = True) -> Run:
        """Keep the run out of the history until it is asked to show archived runs; nothing is
        deleted. ``archive(False)`` brings it back."""
        return self._annotate(archived=archived)

    def _annotations(self) -> RunAnnotations:
        from logogram.history import read_annotations

        return read_annotations(self.project, self.folder)[0]

    def _annotate(self, **changes: Any) -> Run:
        from logogram.history import annotate_run, reason

        try:
            annotate_run(self.project, self.id, **changes)
        except ValueError as exc:  # ProjectError, or a title that isn't one line of text
            raise LogogramError(reason(exc)) from exc
        return self

    def _summary(self) -> dict[str, Any]:
        if self.summary is None:
            raise LogogramError(f"{self.id} has no results ({self.status}).")
        return self.summary

    def sites(self) -> Any:
        """One row per site: what it is, its mean effect and interval, the simultaneous band and
        q-value, sign flips and the metric patched. An edge attribution run's summary lists its
        strongest edges only: :meth:`edges` has every one."""
        from logogram.report import sites_table

        return _table(sites_table(self._summary()))

    def methods(self) -> str:
        """The methods paragraph: every choice that can change a number, in sentences, as the app
        copies it."""
        from logogram.report import methods_text

        return methods_text(self.spec, self.summary, self.manifest, self.id)

    def report(self, path: str | Path | None = None, *, hardware: bool = False) -> str:
        """The run as one self-contained HTML page anyone can open: its map, strongest sites,
        every site's numbers, the methods and where it came from. Written to ``path`` when given;
        hardware model names are left out unless ``hardware`` is true."""
        from logogram.edges import latest_verification
        from logogram.report import run_report

        summary = self._summary()
        page = run_report(
            self.spec,
            summary,
            self.manifest,
            self.id,
            hardware=hardware,
            continuations=self.continuations(),
            verification=latest_verification(self.project, self.id, summary),
        )
        if path is not None:
            Path(path).expanduser().write_text(page, encoding="utf-8")
        return page

    def continuations(self) -> dict[str, Any] | None:
        """What a steering run's model went on to say (its continuations.json): for each of the
        first held-out prompts, the continuation without steering and one per steered site and
        strength (and random control). None for a run that recorded none."""
        from logogram.runs import load_continuations

        return load_continuations(self.project, self.id)

    def results(self) -> Any:
        """results.parquet as a pyarrow Table: one row per site and prompt."""
        import pyarrow.parquet as pq

        self._summary()
        return pq.read_table(self.project.readable(self.folder / "results.parquet"))

    def per_prompt(self) -> Any:
        """The per-prompt values every statistic came from, as a DataFrame (or a pyarrow Table)."""
        table = self.results()
        try:
            return table.to_pandas()
        except ImportError:
            return table

    def site(self, index: int) -> dict[str, Any]:
        """One site's evidence: each prompt's effect and values, the strongest and weakest."""
        from logogram.runs import site_detail

        return site_detail(self.project, self.id, index)

    def strongest(self, k: int = 10) -> list[dict[str, Any]]:
        """The ``k`` sites with the largest mean effects, by magnitude, as summary.json holds
        them: each site's label, effect with its interval, and the rest of its statistics."""
        sites = [s for s in self._summary()["sites"] if s["effect"]["mean"] is not None]
        return sorted(sites, key=lambda s: -abs(s["effect"]["mean"]))[:k]

    def edges(self) -> Any:
        """An edge attribution or ACDC run's every edge, one row each (its edges.parquet): the
        sender and receiver, the mean effect (estimated, or for ACDC the change its removal made
        when it was tested) and its interval, the simultaneous band and q-value; for ACDC also
        ``kept`` (in the circuit) and ``step`` (when it was tested, from 0). The summary lists
        only the strongest."""
        import pyarrow.parquet as pq

        from logogram.schema import EDGES_FILE

        self._summary()
        path = self.folder / EDGES_FILE
        if not path.is_file():
            raise LogogramError(
                f"{self.id} isn't an edge attribution or ACDC run: it has no edges."
            )
        table = pq.read_table(self.project.readable(path))
        try:
            return table.to_pandas()
        except ImportError:
            return table

    def circuit(self) -> dict[str, Any]:
        """An ACDC run's circuit: ``edges``, the kept edges as the spec writes them (sender and
        receiver) with their ``label`` and the change their removal made when tested (``change``,
        normalized, and ``delta``, in the metric's units); ``n_kept`` of ``n_edges``; the
        ``threshold`` and ``rule`` they were kept by; the circuit's ``faithfulness`` with its
        interval (0 is the source run, 1 the receiver run); and the mean metric of the circuit
        (``circuit``) and with every edge kept (``full``)."""
        import pyarrow.parquet as pq

        from logogram.schema import EDGES_FILE

        acdc = self._summary().get("acdc")
        if not acdc:
            raise LogogramError(f"{self.id} isn't an ACDC run: it found no circuit.")
        table = pq.read_table(self.project.readable(self.folder / EDGES_FILE))
        by_index = {int(r["edge"]): r for r in table.to_pylist()}
        edges = []
        for index in acdc["kept"]:
            row = by_index[index]
            edges.append(
                {
                    "index": index,
                    "label": row["label"],
                    "edge": _edge_of(row),
                    "change": _number(row["effect_mean"]),
                    "delta": _number(row["delta_mean"]),
                }
            )
        return {
            "edges": edges,
            "n_kept": acdc["n_kept"],
            "n_edges": acdc["n_edges"],
            "threshold": acdc["threshold"],
            "rule": acdc["rule"],
            "faithfulness": acdc["faithfulness"],
            "circuit": acdc["circuit"],
            "full": acdc["full"],
        }

    def curve(self) -> dict[str, Any] | None:
        """An edge attribution run's faithfulness curve: for each k, the faithfulness of the k
        strongest edges and of k edges chosen at random, with intervals; None without one."""
        return (self._summary().get("edges") or {}).get("curve")

    def probes(self) -> Any:
        """A probe run's sites, one row each: the held-out accuracy with its interval, the
        control's (trained on shuffled labels) and their difference (selectivity) with its
        interval, the training accuracy, and chance."""
        summary = self._summary()
        probe = summary.get("probe")
        if not probe:
            raise LogogramError(f"{self.id} isn't a probe run.")
        sites = {s["index"]: s for s in summary["sites"]}
        trained = {s["index"]: s for s in probe["sites"]}
        chosen = {c["index"]: c for c in probe.get("selectivity") or []}
        rows = []
        for index, site in sites.items():
            if (site.get("variant") or {}).get("control"):
                continue
            sel = chosen.get(index) or {}
            control = sites.get(sel.get("control_index", -1))
            rows.append(
                {
                    "site": index,
                    "label": site["label"],
                    "kind": site["kind"],
                    "layer": site["layer"],
                    "accuracy": site["effect"]["mean"],
                    "accuracy_lo": site["effect"]["lo"],
                    "accuracy_hi": site["effect"]["hi"],
                    "control": None if control is None else control["effect"]["mean"],
                    "selectivity": sel.get("difference"),
                    "selectivity_lo": sel.get("lo"),
                    "selectivity_hi": sel.get("hi"),
                    "train_accuracy": (trained.get(index) or {}).get("train_accuracy"),
                    "chance": probe["chance"],
                }
            )
        return _table(rows)

    def subspaces(self) -> Any:
        """A distributed alignment search run's sites, one row each: the normalized effect on
        held-out pairs of patching within the learned subspace, of the whole site and (with the
        control) within a random subspace, the learned subspace's share of the whole site's
        effect, and its effect on the training pairs."""
        summary = self._summary()
        das = summary.get("das")
        if not das:
            raise LogogramError(f"{self.id} isn't a distributed alignment search run.")
        by_row: dict[int, dict[str, Any]] = {}
        for site in summary["sites"]:
            key = str((site.get("variant") or {}).get("subspace") or "")
            row = by_row.setdefault(
                site["row"], {"label": None, "kind": site["kind"], "layer": site["layer"]}
            )
            row[key] = site["effect"]["mean"]
            if key == "learned":
                row["site"] = site["index"]
                row["label"] = site["label"].removesuffix(" learned")
        trained = {s["row"]: s for s in das["sites"]}
        rows = []
        for r, row in sorted(by_row.items()):
            learned, whole = row.get("learned"), row.get("full")
            rows.append(
                {
                    "site": row.get("site"),
                    "label": row["label"],
                    "kind": row["kind"],
                    "layer": row["layer"],
                    "learned": learned,
                    "whole": whole,
                    "random": row.get("random"),
                    "share": None if learned is None or not whole else learned / whole,
                    "train_effect": (trained.get(r) or {}).get("train_effect"),
                }
            )
        return _table(rows)

    def save_direction(self, site: int, name: str, *, row: int | None = None) -> dict[str, Any]:
        """Save what this probe or learned subspace run found at its site ``site`` (an index of
        :meth:`sites`) as the project's direction ``name``: a subspace's rows, or a probe's
        weights (with two classes, the direction toward the second; with more, class ``row``'s).
        Steering, directional ablation, subspace patching and readouts can then use it."""
        from logogram.directions import DirectionError, save_run_direction

        try:
            return save_run_direction(self.project, self.id, site, name, row=row).to_dict()
        except DirectionError as exc:
            raise LogogramError(str(exc)) from None

    def _repr_html_(self) -> str:
        if self.summary is None:
            return f"<p><b>{html.escape(self.title)}</b> · {html.escape(self.status)}</p>"
        s = self.summary
        rows = "".join(
            f"<tr><td>{html.escape(x['label'])}</td><td style='text-align:right'>"
            f"{_num(x['effect']['mean'])}</td><td>[{_num(x['effect']['lo'])}, "
            f"{_num(x['effect']['hi'])}]</td></tr>"
            for x in self.strongest(8)
        )
        partial = self.partial
        stopped = (
            f"<p><b>{html.escape(self.status.capitalize())}.</b> {html.escape(partial['note'])}</p>"
            if partial
            else ""
        )
        return (
            f"<div style='font-family:sans-serif'><p><b>{html.escape(self.title)}</b><br>"
            f"{html.escape(s['description'])} · n = {s['n_prompts']}</p>{stopped}"
            f"<p style='color:#555'>{html.escape(s['metric']['normalized_effect'])}</p>"
            f"{heatmap_svg(s)}<table><tr><th>site</th><th>effect</th>"
            f"<th>{round(s['statistics']['ci'] * 100)}% CI</th></tr>{rows}</table></div>"
        )


class Sweep(list[Run]):
    """A sweep's runs, one per value in the order given (see :func:`sweep`): a list of
    :class:`Run`, with the sweep's ``id``, ``field`` and ``values``."""

    def __init__(
        self, runs: list[Run], *, project: Project, id: str, field: str, values: list[Any]
    ) -> None:
        super().__init__(runs)
        self.project = project
        self.id = id
        self.field = field
        self.values = values

    def summary(self, choose: str = "first", top: int = 10) -> dict[str, Any]:
        """The sweep as the app's sweep view shows it: for the ``top`` strongest sites (0: all),
        chosen on the ``"first"`` value or on ``"all"`` values, each site's effect and interval
        at every value, the spread of its means across values, and whether it keeps its sign and
        size (``verdict``: holds, size, sign or unknown)."""
        from logogram.sweeps import SweepError, sweep_table

        try:
            return sweep_table(self.project, self.id, choose=choose, top=top)  # type: ignore[arg-type]
        except SweepError as exc:
            raise LogogramError(str(exc)) from exc

    def table(self, choose: str = "first", top: int = 10) -> Any:
        """:meth:`summary` as a table, one row per site and value (as the app exports it). A
        value without results has no numbers, never zeros, and says why in ``status``."""
        from logogram.sweeps import table_rows

        return _table(table_rows(self.summary(choose, top)))

    def __repr__(self) -> str:
        return f"Sweep({self.id!r}, field={self.field!r}, values={self.values!r}, runs={len(self)})"

    def _repr_html_(self) -> str:
        table = self.summary()
        head = "".join(f"<th>{html.escape(v['label'])}</th>" for v in table["values"])
        rows = "".join(
            f"<tr><td>{html.escape(s['label'])}</td>"
            + "".join(
                f"<td style='text-align:right'>{_num((s['points'][v['index']] or {}).get('mean'))}</td>"
                for v in table["values"]
            )
            + f"<td>{html.escape(s['text'])}</td></tr>"
            for s in table["sites"]
        )
        return (
            f"<div style='font-family:sans-serif'><p><b>{html.escape(table['headline'])}</b><br>"
            f"{html.escape(table['name'])}: {html.escape(table['plural'])} "
            f"{html.escape(', '.join(v['label'] for v in table['values']))}</p>"
            f"<p style='color:#555'>{html.escape(table['problem'] or table['chosen'])}</p>"
            f"<table><tr><th>site</th>{head}<th>across values</th></tr>{rows}</table></div>"
        )


def _num(x: float | None) -> str:
    return "—" if x is None else f"{x:+.3f}"


def _number(x: Any) -> float | None:
    """A float, or None for a value that isn't a finite number (as summary.json writes it)."""
    import math

    value = float(x)
    return value if math.isfinite(value) else None


def _edge_of(row: dict[str, Any]) -> dict[str, Any]:
    """An edge of edges.parquet as the spec writes it: its sender and receiver."""
    from logogram.spec import Edge

    sender: dict[str, Any] = {"kind": row["sender"]}
    if row["sender"] != "embed":
        sender["layer"] = int(row["sender_layer"])
    if row["sender"] == "head":
        sender["head"] = int(row["sender_head"])
    kind = row["receiver"]
    receiver: dict[str, Any] = (
        {"kind": "logits"}
        if kind == "logits"
        else {"kind": "mlp", "layer": int(row["receiver_layer"])}
        if kind == "mlp"
        else {
            "kind": "head",
            "layer": int(row["receiver_layer"]),
            "head": int(row["receiver_head"]),
            "input": kind,
        }
    )
    return Edge.model_validate({"sender": sender, "receiver": receiver}).model_dump(mode="json")


def heatmap_svg(summary: dict[str, Any], cell: int = 18) -> str:
    """A run's results as an SVG grid in the effect colors, with its scale: for notebooks, with
    no plotting library (the same map as the HTML report's)."""
    from logogram.report import heatmap_svg as svg

    return svg(summary, cell=cell)


def dumps(value: Any) -> str:
    return json.dumps(value, indent=2, default=str)
