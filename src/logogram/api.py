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

Tables come back as pandas DataFrames when pandas is installed, and as pyarrow Tables otherwise.
"""

from __future__ import annotations

import html
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from logogram.project import Project, ProjectError
from logogram.schema import RunAnnotations
from logogram.spec import Spec

__all__ = [
    "Run",
    "add_local_model",
    "check_robustness",
    "compare_runs",
    "create_project",
    "delete_run",
    "feature_writes",
    "generate",
    "head_kinds",
    "list_local_models",
    "list_runs",
    "load_model",
    "load_run",
    "load_sae",
    "load_spec",
    "look",
    "open_project",
    "remove_local_model",
    "run",
    "verify_run",
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


def _chat_template(value: Any) -> Any:
    """A spec's tokenization.chat_template from a dict (as a spec writes it), or None."""
    if value is None or not isinstance(value, dict):
        return value
    from pydantic import TypeAdapter

    from logogram.spec import ChatTemplate

    return TypeAdapter(ChatTemplate).validate_python(value)


def load_sae(repo: str, path: str = "", *, model: Any, revision: str | None = None) -> Any:
    """Download (once) and read a published SAE or transcoder at an exact revision, onto
    ``model``'s device. It is refused if it was made for another model's sizes or reads a site the
    model doesn't have. Pass it to :func:`run` with a spec that names it, or explore its features
    with :func:`feature_writes`."""
    from logogram.features import check_sae_fits
    from logogram.sae import load_sae as load
    from logogram.spec import SAERef

    sae = load(SAERef(repo=repo, path=path, revision=revision), model.info.device)
    check_sae_fits(sae, model.info)
    return sae


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
) -> dict[str, Any]:
    """Read any text with ``model`` (from :func:`load_model`), without making a dataset: its
    tokens, the ``top_k`` most likely next tokens after each, and at ``position`` (the last token
    by default) what each layer would predict, as the app's Your text view shows. With
    ``chat_template`` (as a spec writes it, such as ``{"kind": "user_turn", "system": None}``,
    with ``prepend_bos`` false), the text is wrapped in the model's own chat template."""
    from logogram.inside import look_report

    with model.lock:
        return look_report(
            model,
            text,
            prepend_bos=prepend_bos,
            top_k=top_k,
            position=position,
            chat_template=_chat_template(chat_template),
        )


def _ended(outcome: Any, what: str) -> str:
    """Why a run didn't finish, and where its traceback is."""
    message = f"{what} {outcome.status}: {outcome.manifest.get('error')}"
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
    """Patch, for real, the ``top`` sites an attribution patching run estimated to matter most."""
    from logogram.runner import run_spec
    from logogram.verify import verification_spec

    if estimated.summary is None:
        raise LogogramError("Only a finished run can be verified.")
    variant = verification_spec(estimated.spec, estimated.summary, top)
    n = len(variant.scope.sites)  # type: ignore[union-attr]
    outcome = run_spec(
        variant,
        estimated.project,
        backend=model or estimated.model,
        derived_from={
            "run": estimated.id,
            "kind": "verification",
            "change": f"the {n} strongest estimated site{'s' if n != 1 else ''}, patched",
        },
    )
    if outcome.status != "finished":
        raise LogogramError(_ended(outcome, "The verification"))
    return load_run(estimated.project, outcome.run_id, model=outcome.backend)


def check_robustness(
    original: Run,
    *,
    experiment: dict[str, Any] | None = None,
    dtype: str | None = None,
    model: Any = None,
) -> Run:
    """Rerun with one choice changed (the experiment, or the dtype), as Check robustness does."""
    from logogram.robustness import robustness_spec
    from logogram.runner import run_spec

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


def _table(rows: list[dict[str, Any]]) -> Any:
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except ImportError:
        import pyarrow as pa

        return pa.Table.from_pylist(rows)


@dataclass
class Run:
    """A run in a project: its spec, summary and manifest as written, and its per-prompt values."""

    project: Project
    id: str
    spec: Spec
    summary: dict[str, Any] | None
    manifest: dict[str, Any]
    model: Any = field(default=None, repr=False)

    @property
    def folder(self) -> Path:
        return self.project.run_dir(self.id)

    @property
    def status(self) -> str:
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
    def title(self) -> str:
        """What the run is called: the title it was given (see :meth:`rename`), or its spec's
        name."""
        return self._annotations().title or self.spec.name

    @property
    def starred(self) -> bool:
        return self._annotations().starred

    @property
    def archived(self) -> bool:
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
        q-value, sign flips and the metric patched."""
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
        from logogram.report import run_report

        page = run_report(self.spec, self._summary(), self.manifest, self.id, hardware=hardware)
        if path is not None:
            Path(path).expanduser().write_text(page, encoding="utf-8")
        return page

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
        sites = [s for s in self._summary()["sites"] if s["effect"]["mean"] is not None]
        return sorted(sites, key=lambda s: -abs(s["effect"]["mean"]))[:k]

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
        return (
            f"<div style='font-family:sans-serif'><p><b>{html.escape(self.title)}</b><br>"
            f"{html.escape(s['description'])} · n = {s['n_prompts']}</p>"
            f"<p style='color:#555'>{html.escape(s['metric']['normalized_effect'])}</p>"
            f"{heatmap_svg(s)}<table><tr><th>site</th><th>effect</th>"
            f"<th>{round(s['statistics']['ci'] * 100)}% CI</th></tr>{rows}</table></div>"
        )


def _num(x: float | None) -> str:
    return "—" if x is None else f"{x:+.3f}"


def heatmap_svg(summary: dict[str, Any], cell: int = 18) -> str:
    """A run's results as an SVG grid in the effect colors, with its scale: for notebooks, with
    no plotting library (the same map as the HTML report's)."""
    from logogram.report import heatmap_svg as svg

    return svg(summary, cell=cell)


def dumps(value: Any) -> str:
    return json.dumps(value, indent=2, default=str)
