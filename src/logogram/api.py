"""Logogram from Python: run specs, read runs and their per-prompt values, compare and check them.

Everything here goes through the same code as the app and ``logogram run`` (the runner, the
robustness and verification specs, the comparison), so a run made from a notebook is the same run
the app would make, and it lands in the project like any other::

    import logogram as lg

    project = lg.open_project("~/Documents/Logogram/ioi-example")
    model = lg.load_model("openai-community/gpt2")
    spec = lg.load_spec(project.root / "experiments/ioi-head-patching/spec.json")
    run = lg.run(spec, project, model=model)
    run.sites()          # one row per site: mean effect, interval, q-value, ...
    run.per_prompt()     # one row per site and prompt: everything the statistics came from
    run                  # in a notebook: the method, the strongest sites and a heatmap

Tables come back as pandas DataFrames when pandas is installed, and as pyarrow Tables otherwise.
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from logogram.project import Project, ProjectError
from logogram.spec import Spec

__all__ = [
    "Run",
    "check_robustness",
    "compare_runs",
    "create_project",
    "generate",
    "list_runs",
    "load_model",
    "load_run",
    "load_spec",
    "open_project",
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
) -> Any:
    """Download (once) and load a model, checked as the app checks it. Pass it to :func:`run` to
    reuse it across runs; every choice it was loaded with is recorded in each run's spec."""
    from logogram.backends.transformer_lens import load_model as load
    from logogram.runner import configure_determinism

    configure_determinism()
    return load(
        model_id,
        revision=revision,
        dtype=dtype,
        device=device,
        process_weights=process_weights,
        on_progress=on_progress,
    )


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
        raise LogogramError(f"The run {outcome.status}: {outcome.manifest.get('error')}")
    return load_run(project, outcome.run_id, model=outcome.backend)


def list_runs(project: Project) -> list[dict[str, Any]]:
    """Every run and draft in the project, newest first, as the history lists them."""
    return [r.to_dict() for r in project.list_runs()]


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
        raise LogogramError(f"The verification {outcome.status}: {outcome.manifest.get('error')}")
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
        raise LogogramError(f"The rerun {outcome.status}: {outcome.manifest.get('error')}")
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
        return str(self.manifest.get("status") or "draft")

    def _summary(self) -> dict[str, Any]:
        if self.summary is None:
            raise LogogramError(f"{self.id} has no results ({self.status}).")
        return self.summary

    def sites(self) -> Any:
        """One row per site: what it is, its mean effect and interval, the simultaneous band and
        q-value, sign flips and the metric patched."""
        rows = []
        for s in self._summary()["sites"]:
            band = s.get("band") or {}
            rows.append(
                {
                    "index": s["index"],
                    "label": s["label"],
                    "kind": s["kind"],
                    "layer": s["layer"],
                    "head": s.get("head"),
                    "feature": s.get("feature"),
                    "position": s["position_key"],
                    "variant": s.get("variant_key"),
                    "n": s["n"],
                    "effect": s["effect"]["mean"],
                    "effect_sd": s["effect"]["sd"],
                    "effect_lo": s["effect"]["lo"],
                    "effect_hi": s["effect"]["hi"],
                    "band_lo": band.get("lo"),
                    "band_hi": band.get("hi"),
                    "q": s.get("q"),
                    "delta": s["delta"]["mean"],
                    "patched_metric": s.get("patched_metric"),
                    "sign_flips": s["sign_flips"],
                    "opposite_sign": s["opposite_sign"],
                }
            )
        return _table(rows)

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
            return f"<p><b>{html.escape(self.spec.name)}</b> · {html.escape(self.status)}</p>"
        s = self.summary
        rows = "".join(
            f"<tr><td>{html.escape(x['label'])}</td><td style='text-align:right'>"
            f"{_num(x['effect']['mean'])}</td><td>[{_num(x['effect']['lo'])}, "
            f"{_num(x['effect']['hi'])}]</td></tr>"
            for x in self.strongest(8)
        )
        return (
            f"<div style='font-family:sans-serif'><p><b>{html.escape(s['name'])}</b><br>"
            f"{html.escape(s['description'])} · n = {s['n_prompts']}</p>"
            f"<p style='color:#555'>{html.escape(s['metric']['normalized_effect'])}</p>"
            f"{heatmap_svg(s)}<table><tr><th>site</th><th>effect</th>"
            f"<th>{round(s['statistics']['ci'] * 100)}% CI</th></tr>{rows}</table></div>"
        )


def _num(x: float | None) -> str:
    return "—" if x is None else f"{x:+.3f}"


# The effect scale: cobalt for positive, amber for negative, neutral at zero (as in the app).
_POSITIVE = (37, 87, 222)
_NEGATIVE = (200, 120, 20)
_NEUTRAL = (236, 236, 236)


def _color(value: float | None, bound: float) -> str:
    if value is None or not math.isfinite(value):
        return "#ffffff"
    t = max(-1.0, min(1.0, value / bound)) if bound > 0 else 0.0
    end = _POSITIVE if t >= 0 else _NEGATIVE
    mix = [round(n + abs(t) * (e - n)) for n, e in zip(_NEUTRAL, end, strict=True)]
    return "#" + "".join(f"{c:02x}" for c in mix)


def heatmap_svg(summary: dict[str, Any], cell: int = 18) -> str:
    """A run's results as an SVG grid in the effect colors, with its scale's bound: for notebooks,
    with no plotting library."""
    layout = summary["layout"]
    rows, cols = layout["rows"], layout["cols"]
    values = {(s["row"], s["col"]): s["effect"]["mean"] for s in summary["sites"]}
    finite = [abs(v) for v in values.values() if v is not None and math.isfinite(v)]
    bound = max(finite) if finite else 1.0
    left, top = 60, 20
    width, height = left + cell * len(cols) + 10, top + cell * len(rows) + 30
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        "font-family='sans-serif' font-size='10'>"
    ]
    for r, row in enumerate(rows):
        label = html.escape(str(row.get("label", r)))[:10]
        parts.append(
            f"<text x='{left - 4}' y='{top + r * cell + cell * 0.7}' text-anchor='end'>{label}</text>"
        )
        for c in range(len(cols)):
            fill = _color(values.get((r, c)), bound)
            parts.append(
                f"<rect x='{left + c * cell}' y='{top + r * cell}' width='{cell - 1}' "
                f"height='{cell - 1}' fill='{fill}'/>"
            )
    parts.append(
        f"<text x='{left}' y='{height - 8}'>{html.escape(layout['row_title'])} × "
        f"{html.escape(layout['col_title'] or 'site')} · scale ±{bound:.3g}</text></svg>"
    )
    return "".join(parts)


def dumps(value: Any) -> str:
    return json.dumps(value, indent=2, default=str)
