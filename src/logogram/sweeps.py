"""Sweeps: one spec run across several values of one choice.

A finding that holds for one seed, one sample of prompts or one checkpoint may not hold for
others. A sweep runs the same spec once for each value of one field: a seed the spec states, the
dataset, or the model's revision. Each run is an ordinary spec that differs from the original in
that field alone (a dataset's hash, which pins its file, follows its path), and its manifest's
``derived_from`` records the sweep (:class:`logogram.schema.SweepMember`): the sweep's id, the
field, the run's value and its place, and every value in the order given. The run a sweep is made
from stands in for its own value when that value is one of the sweep's, so it isn't run again.

The app queues a sweep's runs (server/state.py); ``logogram sweep`` and ``lg.sweep`` run them in
turn (:func:`run_sweep`). All three plan them here, run them with logogram.runner.run_spec, and
read them back with :func:`sweep_table`: for the strongest sites, each site's effect with its
interval at every value, the spread of the means across values, and which sites keep their sign
and size from value to value.
"""

from __future__ import annotations

import hashlib
import math
import secrets
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from logogram.compare import site_key, spec_differences
from logogram.datasets import DatasetError, decode_dataset, parse_jsonl, read_dataset_bytes
from logogram.project import Project, ProjectError, slugify
from logogram.schema import RunListing, SweepMember
from logogram.spec import (
    NAME_MAX,
    SEED_MAX,
    Ablation,
    FeaturesScope,
    ResampleBaseline,
    Spec,
    Steering,
    check_relative_path,
    check_revision,
)

if TYPE_CHECKING:
    from logogram.backends.base import ModelBackend
    from logogram.runner import EventFn, ModelProvider, RunOutcome

FieldKind = Literal["seed", "dataset", "revision"]
Choose = Literal["first", "all"]
Verdict = Literal["holds", "size", "sign", "unknown"]

# A sweep runs at least two values, and at most this many.
MAX_VALUES = 256
# How many of the strongest sites the sweep view shows unless asked for another number.
TOP = 10


class SweepError(ValueError):
    """A sweep can't be made or read as asked. The message says what went wrong and how to fix it."""


class SweepNotFound(SweepError):
    """No run of the project belongs to a sweep of that id."""


@dataclass(frozen=True)
class SweepField:
    """A field a sweep can vary: its dotted path in the spec, and how it is named."""

    path: str
    kind: FieldKind
    label: str
    plural: str
    description: str


FIELDS: dict[str, SweepField] = {
    f.path: f
    for f in (
        SweepField(
            "statistics.seed",
            "seed",
            "bootstrap seed",
            "bootstrap seeds",
            "The seed of the bootstrap's resamples: it moves the intervals, never the means.",
        ),
        SweepField(
            "experiment.baseline.seed",
            "seed",
            "donor seed",
            "donor seeds",
            "The seed that draws each prompt's donors for resample ablation.",
        ),
        SweepField(
            "experiment.seed",
            "seed",
            "split seed",
            "split seeds",
            "The seed that splits the pairs into those that compute the steering direction and "
            "those it is measured on.",
        ),
        SweepField(
            "scope.seed",
            "seed",
            "feature split seed",
            "feature split seeds",
            "The seed that splits the prompts into those that choose the features and those they "
            "are reported on.",
        ),
        SweepField(
            "dataset.path",
            "dataset",
            "dataset",
            "datasets",
            "The prompts: a dataset file of the project for each value, such as the same task "
            "generated with other seeds.",
        ),
        SweepField(
            "model.revision",
            "revision",
            "revision",
            "revisions",
            "The model's revision: commits, tags or branches of its Hugging Face repository, such "
            "as training checkpoints.",
        ),
    )
}


def get_field(path: str) -> SweepField:
    field = FIELDS.get(path) if isinstance(path, str) else None
    if field is None:
        raise SweepError(f"Logogram can't sweep {path!r}. It sweeps {', '.join(FIELDS)}.")
    return field


def _unknown_field(path: str) -> SweepField:
    """How a sweep over a field this version doesn't know (made by a newer one) is shown."""
    return SweepField(path, "seed", path, path, path)


# -- the field in a spec -------------------------------------------------------------------------


def field_value(spec: Spec, field: SweepField) -> Any:
    """The spec's own value of ``field``, or a SweepError saying why the spec has none to vary."""
    path = field.path
    if path == "statistics.seed":
        return spec.statistics.seed
    if path == "experiment.baseline.seed":
        e = spec.experiment
        if isinstance(e, Ablation) and isinstance(e.baseline, ResampleBaseline):
            return e.baseline.seed
        raise SweepError(
            "This spec has no donor seed: only resample ablation draws donors. Sweep another field, "
            "such as statistics.seed."
        )
    if path == "experiment.seed":
        if isinstance(spec.experiment, Steering):
            return spec.experiment.seed
        raise SweepError(
            "This spec has no split seed: only steering splits its pairs. Sweep another field, "
            "such as statistics.seed."
        )
    if path == "scope.seed":
        scope = spec.scope
        if not isinstance(scope, FeaturesScope):
            raise SweepError(
                "This spec has no feature split seed: only a scope over every SAE feature splits "
                "its prompts. Sweep another field, such as statistics.seed."
            )
        if scope.seed is None:
            raise SweepError(
                "This spec's features are chosen and reported on every prompt (scope.choose_on is "
                "null), so there is no split seed to vary. Set choose_on and seed to split them."
            )
        return scope.seed
    if path == "dataset.path":
        return spec.dataset.path
    if path == "model.revision":
        if spec.model.source == "local":
            raise SweepError(
                "A local model has no revisions to sweep: its files are pinned by their hash. Sweep "
                "a model from Hugging Face."
            )
        return spec.model.revision
    raise SweepError(f"Logogram can't sweep {path!r}. It sweeps {', '.join(FIELDS)}.")


def sweepable_fields(spec: Spec) -> list[tuple[SweepField, Any]]:
    """The fields ``spec`` can be swept over, each with the spec's own value."""
    out = []
    for field in FIELDS.values():
        try:
            out.append((field, field_value(spec, field)))
        except SweepError:
            continue
    return out


def value_label(field: SweepField, value: Any) -> str:
    """A value as the sweep shows it: a seed, a dataset's file name, a revision (a commit short)."""
    if field.kind == "dataset" and isinstance(value, str):
        return Path(value).name
    if field.kind == "revision" and isinstance(value, str) and _is_commit(value):
        return value[:7]
    return str(value)


def value_name(field: SweepField, value: Any) -> str:
    """A value in words, such as "bootstrap seed 2" or "dataset ioi-seed1.jsonl"."""
    return f"{field.label} {value_label(field, value)}"


def _is_commit(value: str) -> bool:
    return len(value) == 40 and all(c in "0123456789abcdef" for c in value)


# -- values --------------------------------------------------------------------------------------


def _seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < SEED_MAX:
        raise SweepError(
            f"Seeds are whole numbers from 0 to {SEED_MAX - 1:,}; {value!r} isn't one."
        )
    return value


def _revision(value: Any) -> str:
    text = value.strip() if isinstance(value, str) else ""
    try:
        check_revision(text, "A revision")
    except ValueError as exc:
        raise SweepError(
            f"{exc}, such as main, step1000 or a commit; {value!r} isn't one."
        ) from exc
    return text


def dataset_path(project: Project, value: Any) -> str:
    """A dataset value as the spec names it: a path inside the project with forward slashes. A
    bare file name is looked for in datasets/. The file must be a dataset Logogram can read."""
    if not isinstance(value, str) or not value.strip():
        raise SweepError(
            "Name each dataset by its path in the project, such as datasets/ioi.jsonl."
        )
    text = value.strip().replace("\\", "/")
    if "/" not in text and not (project.root / text).exists():
        text = f"datasets/{text}"
    try:
        check_relative_path(text, "A dataset")
    except ValueError as exc:
        raise SweepError(str(exc)) from exc
    try:
        path = project.resolve_dataset(text)
    except ProjectError as exc:
        raise SweepError(
            f"There is no dataset {text} in this project. Choose one of its datasets, or generate "
            "one first."
        ) from exc
    try:
        parse_jsonl(decode_dataset(read_dataset_bytes(path), text), source=text)
    except DatasetError as exc:
        raise SweepError(str(exc)) from exc
    return text


def _sha256(project: Project, path: str) -> str:
    return hashlib.sha256(read_dataset_bytes(project.resolve_dataset(path))).hexdigest()


def check_values(project: Project, field: SweepField, values: Any) -> list[Any]:
    """The values to sweep, checked, in the order given: seeds in range, datasets that exist in
    the project and can be read, revisions that are names. Each value is listed once."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise SweepError(f"Give the {field.plural} as a list, in the order to run them.")
    if len(values) < 2:
        raise SweepError(f"Give at least two {field.plural} to sweep over.")
    if len(values) > MAX_VALUES:
        raise SweepError(
            f"A sweep runs at most {MAX_VALUES} values; this one has {len(values)}. Split it into "
            "several sweeps."
        )
    if field.kind == "seed":
        out: list[Any] = [_seed(v) for v in values]
    elif field.kind == "dataset":
        out = [dataset_path(project, v) for v in values]
    else:
        out = [_revision(v) for v in values]
    seen: set[Any] = set()
    for value in out:
        if value in seen:
            named = value_name(field, value)
            raise SweepError(
                f"{named[:1].upper()}{named[1:]} is listed twice. List each {field.label} once."
            )
        seen.add(value)
    if field.kind == "dataset":
        hashes: dict[str, str] = {}
        for value in out:
            sha = _sha256(project, value)
            if sha in hashes:
                raise SweepError(
                    f"{hashes[sha]} and {value} hold the same prompts, so their runs would be the "
                    "same. Leave one of them out."
                )
            hashes[sha] = value
    return out


# -- planning ------------------------------------------------------------------------------------


def _set(data: dict[str, Any], path: str, value: Any) -> None:
    *parents, last = path.split(".")
    for key in parents:
        data = data[key]
    data[last] = value


def _name(name: str, field: SweepField, value: Any) -> str:
    suffix = f" · {value_name(field, value)}"
    if len(suffix) >= NAME_MAX:
        return (name + suffix)[:NAME_MAX]
    return name[: NAME_MAX - len(suffix)].rstrip() + suffix


def variant_spec(spec: Spec, field: SweepField, value: Any, project: Project | None = None) -> Spec:
    """``spec`` with ``field`` set to ``value`` and nothing else changed but its name. A dataset's
    path brings its hash, pinned now, so its run reads the file as the sweep found it."""
    data = spec.model_dump(mode="json")
    _set(data, field.path, value)
    allowed = {field.path}
    if field.kind == "dataset":
        if project is None:
            raise SweepError("A sweep over datasets needs the project they are in.")
        data["dataset"]["sha256"] = _sha256(project, value)
        allowed.add("dataset.sha256")
    data["name"] = _name(spec.name, field, value)
    variant = Spec.model_validate(data)
    # A version 1 spec's runs say what it left out, as the spec's own run does.
    variant._upgraded = spec.upgraded_fields
    changed = {d["path"] for d in spec_differences(spec.model_dump(mode="json"), data)}
    if not changed <= allowed:  # a field this module doesn't know how to set alone
        raise SweepError(f"Setting {field.path} would also change {', '.join(sorted(changed))}.")
    return variant


def _matches(project: Project, spec: Spec, field: SweepField, value: Any) -> bool:
    """Whether ``value`` is the spec's own: the same seed or revision, or the same prompts."""
    if field.kind != "dataset":
        return field_value(spec, field) == value
    own = spec.dataset.sha256
    if own is None:
        try:
            own = _sha256(project, spec.dataset.path)
        except (ProjectError, OSError, DatasetError):
            return False
    return _sha256(project, value) == own


@dataclass(frozen=True)
class PlannedRun:
    index: int
    value: Any
    # None: the run the sweep is made from stands in for this value.
    spec: Spec | None
    derived_from: dict[str, Any] | None


@dataclass(frozen=True)
class SweepPlan:
    id: str
    field: SweepField
    values: list[Any]
    # The run the sweep is made from, and the value it stands in for (if it is one of them).
    original: str | None
    stands_in: int | None
    runs: list[PlannedRun]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sweep": self.id,
            "field": self.field.path,
            "kind": self.field.kind,
            "label": self.field.label,
            "plural": self.field.plural,
            "values": self.values,
            "labels": [value_label(self.field, v) for v in self.values],
            "original": self.original,
            "stands_in": self.stands_in,
            "runs": [
                {
                    "index": r.index,
                    "value": r.value,
                    "label": value_label(self.field, r.value),
                    "name": r.spec.name if r.spec is not None else None,
                    "original": r.spec is None,
                }
                for r in self.runs
            ],
        }


def new_sweep_id(field: SweepField) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{slugify(field.label, 24)}-{secrets.token_hex(2)}"


def plan_sweep(
    project: Project,
    spec: Spec,
    field: str,
    values: Any,
    *,
    original: RunListing | None = None,
    sweep_id: str | None = None,
) -> SweepPlan:
    """The runs of a sweep of ``spec`` over ``values`` of ``field``, in the order given.

    ``original`` is the run the spec came from, if it came from one: when it has finished and its
    value is one of the sweep's, it stands in for that value and that value isn't run again. Each
    planned run's ``derived_from`` records the sweep in its manifest."""
    chosen = get_field(field)
    field_value(spec, chosen)  # the spec has the field to vary, or says why not
    checked = check_values(project, chosen, values)
    identity = sweep_id or new_sweep_id(chosen)
    stands_in = None
    if original is not None and original.status == "finished":
        stands_in = next(
            (i for i, v in enumerate(checked) if _matches(project, spec, chosen, v)), None
        )
    runs = []
    for i, value in enumerate(checked):
        if i == stands_in:
            runs.append(PlannedRun(i, value, None, None))
            continue
        record = SweepMember(
            kind="sweep",
            sweep=identity,
            field=chosen.path,
            value=value,
            index=i,
            of=len(checked),
            values=checked,
            run=original.id if original is not None else None,
            original=stands_in,
            change=value_name(chosen, value),
        )
        runs.append(
            PlannedRun(
                i,
                value,
                variant_spec(spec, chosen, value, project),
                record.model_dump(mode="json"),
            )
        )
    return SweepPlan(identity, chosen, checked, original.id if original else None, stands_in, runs)


def suggested_values(
    project: Project, spec: Spec, field: SweepField, listing: RunListing | None
) -> list[Any]:
    """What the app offers to sweep at first: the spec's own value first (so its run stands in for
    it), and for a seed four more after it."""
    current = field_value(spec, field)
    if field.kind == "seed":
        return [(current + k) % SEED_MAX for k in range(5)]
    if field.kind == "revision":
        return [current] if current else []
    # A finished run read a saved copy of its dataset: offer the file it came from, or another of
    # the project's datasets, while it holds the same prompts; the saved copy itself last.
    try:
        own = spec.dataset.sha256 or _sha256(project, current)
    except (ProjectError, OSError, DatasetError):
        return []
    source = listing.dataset_source if listing is not None else None
    same = [d["path"] for d in project.list_datasets() if d.get("sha256") == own]
    for path in (*(p for p in same if p == source), *same, source, current):
        if not path:
            continue
        try:
            if _sha256(project, path) == own:
                return [path]
        except (ProjectError, OSError, DatasetError):
            continue
    return []


# -- running in turn -----------------------------------------------------------------------------


def run_sweep(
    plan: SweepPlan,
    project: Project,
    *,
    backend: ModelBackend | None = None,
    provider: ModelProvider | None = None,
    on_event: EventFn | None = None,
    on_run: Callable[[PlannedRun], None] | None = None,
    sae: Any = None,
) -> tuple[list[RunOutcome | None], ModelBackend | None]:
    """Run a sweep's runs one after another (``logogram sweep`` and ``lg.sweep``; the app queues
    them instead). Returns each value's outcome (None where the original run stands in) and the
    model the last run used. A model the sweep loads for one revision is closed before the next
    is loaded; ``backend``, the caller's, is never closed."""
    from logogram.runner import default_provider, run_spec

    outcomes: list[RunOutcome | None] = []
    current = backend
    loaded: list[ModelBackend] = []

    def provide(ref: Any) -> ModelBackend:
        while loaded:  # the previous revision's model, which no later run needs: make room
            loaded.pop().close()
        model = (provider or default_provider(on_event))(ref)
        loaded.append(model)
        return model

    for item in plan.runs:
        if item.spec is None:
            outcomes.append(None)
            continue
        if on_run is not None:
            on_run(item)
        outcome = run_spec(
            item.spec,
            project,
            backend=current,
            provider=provide,
            on_event=on_event,
            derived_from=item.derived_from,
            sae=sae,
        )
        if outcome.backend is not None:
            current = outcome.backend
        outcomes.append(outcome)
    return outcomes, current


# -- reading sweeps back ---------------------------------------------------------------------------


@dataclass
class Sweep:
    """A sweep as its runs record it."""

    id: str
    record: SweepMember
    # Its runs by value index, the run it was made from among them where it stands in.
    runs: dict[int, RunListing]
    created: str | None

    @property
    def field(self) -> SweepField:
        return FIELDS.get(self.record.field) or _unknown_field(self.record.field)

    @property
    def values(self) -> list[Any]:
        return list(self.record.values)


def find_sweeps(project: Project, listings: list[RunListing] | None = None) -> dict[str, Sweep]:
    """The project's sweeps, newest first, from the runs that record them. A sweep's values,
    field and original run are read from its first run; a run that disagrees (edited by hand) is
    left out. Where two runs claim one value, a finished one is kept."""
    listings = project.list_runs() if listings is None else listings
    by_id = {r.id: r for r in listings}
    groups: dict[str, list[tuple[SweepMember, RunListing]]] = {}
    for listing in listings:
        record = listing.derived_from
        if isinstance(record, SweepMember):
            groups.setdefault(record.sweep, []).append((record, listing))
    sweeps = []
    for sweep_id, members in groups.items():
        members.sort(key=lambda m: (m[0].index, m[1].created or ""))
        record = members[0][0]
        runs: dict[int, RunListing] = {}
        for member, listing in members:
            if member.field != record.field or member.values != record.values:
                continue
            kept = runs.get(member.index)
            if kept is None or (kept.status != "finished" and listing.status == "finished"):
                runs[member.index] = listing
        original = by_id.get(record.run) if record.run is not None else None
        if record.original is not None and original is not None:
            runs.setdefault(record.original, original)
        created = min((m[1].created or "" for m in members), default="") or None
        sweeps.append(Sweep(sweep_id, record, runs, created))
    sweeps.sort(key=lambda s: (s.created or "", s.id), reverse=True)
    return {s.id: s for s in sweeps}


def get_sweep(project: Project, sweep_id: str) -> Sweep:
    sweep = find_sweeps(project).get(sweep_id)
    if sweep is None:
        raise SweepNotFound(
            f"There is no sweep {sweep_id} in this project. Its runs may have been deleted."
        )
    return sweep


def _finite(x: Any) -> float | None:
    return float(x) if isinstance(x, int | float) and math.isfinite(x) else None


def _point(site: dict[str, Any]) -> dict[str, Any]:
    effect = site.get("effect") or {}
    return {
        "mean": _finite(effect.get("mean")),
        "lo": _finite(effect.get("lo")),
        "hi": _finite(effect.get("hi")),
        "sd": _finite(effect.get("sd")),
        "n": site.get("n"),
    }


def _key(site: dict[str, Any]) -> str:
    return "|".join("" if part is None else str(part) for part in site_key(site))


def _interval(p: dict[str, Any] | None) -> tuple[float, float] | None:
    """A point's interval, when it has one that says anything (lo < hi)."""
    if p is None or p["lo"] is None or p["hi"] is None or p["lo"] >= p["hi"]:
        return None
    return p["lo"], p["hi"]


def _names(field: SweepField, labels: list[str], indices: Sequence[int]) -> str:
    """Values in words: "bootstrap seed 1", "bootstrap seeds 1 and 3", "datasets a, b and c"."""
    shown = [labels[i] for i in indices]
    noun = field.label if len(shown) == 1 else field.plural
    if len(shown) <= 1:
        return f"{noun} {''.join(shown)}"
    return f"{noun} {', '.join(shown[:-1])} and {shown[-1]}"


def _site_row(
    key: str,
    meta: dict[str, Any],
    points: list[dict[str, Any] | None],
    reasons: list[str | None],
    field: SweepField,
    labels: list[str],
    rank: int,
) -> dict[str, Any]:
    n = len(points)
    measured = [i for i, p in enumerate(points) if p is not None and p["mean"] is not None]
    means = [points[i]["mean"] for i in measured]  # type: ignore[index]
    spread = None
    if means:
        spread = {
            "mean": statistics.fmean(means),
            "sd": statistics.stdev(means) if len(means) > 1 else None,
            "min": min(means),
            "max": max(means),
        }
    halves = [(iv[1] - iv[0]) / 2 for i in measured if (iv := _interval(points[i])) is not None]
    above = [i for i in measured if (iv := _interval(points[i])) is not None and iv[0] > 0]
    below = [i for i in measured if (iv := _interval(points[i])) is not None and iv[1] < 0]
    sign = sorted([above[0], below[0]]) if above and below else None
    apart = None
    for a_pos, a in enumerate(measured):
        ia = _interval(points[a])
        if ia is None:
            continue
        for b in measured[a_pos + 1 :]:
            ib = _interval(points[b])
            if ib is not None and (ia[1] < ib[0] or ib[1] < ia[0]):
                apart = [a, b]
                break
        if apart:
            break
    missing = [i for i in range(n) if i not in measured]
    verdict: Verdict
    if len(measured) < 2:
        verdict = "unknown"
    elif sign:
        verdict = "sign"
    elif apart:
        verdict = "size"
    else:
        verdict = "holds"

    every = (
        f"all {n} {field.plural}"
        if not missing
        else f"the {len(measured)} {field.plural} that measured it"
    )
    if verdict == "unknown":
        text = (
            "Measured at no value yet."
            if not measured
            else f"Measured at {_names(field, labels, measured)} only: nothing to compare it with."
        )
    elif verdict == "sign":
        text = (
            f"Flips sign: above zero at {_names(field, labels, [above[0]])} and below zero at "
            f"{_names(field, labels, [below[0]])}, with both intervals excluding zero."
        )
    elif verdict == "size":
        text = (
            f"Changes size: its intervals at {_names(field, labels, apart or [])} exclude each "
            "other."
        )
    else:
        confident = len(above) + len(below)
        zero = (
            "Its interval excludes zero at every one."
            if confident == len(measured)
            else "Its interval includes zero at every one, so no value shows an effect."
            if confident == 0
            else f"Its interval excludes zero at {confident} of them."
        )
        text = f"Keeps its sign and size at {every}: no two intervals exclude each other. {zero}"
    # Where its intervals exclude zero: at every value that measured it, at some, or at none.
    confident = len(above) + len(below)
    excludes = (
        "every" if measured and confident == len(measured) else "some" if confident else "none"
    )
    if missing and measured:
        why = sorted({reasons[i] or "not measured" for i in missing})
        text += f" Not measured at {_names(field, labels, missing)} ({'; '.join(why)})."
    return {
        "key": key,
        "rank": rank,
        "label": meta["label"],
        "kind": meta["kind"],
        "layer": meta["layer"],
        "head": meta.get("head"),
        "feature": meta.get("feature"),
        "neuron": meta.get("neuron"),
        "position": meta.get("position"),
        "position_key": meta["position_key"],
        "variant": meta.get("variant"),
        "variant_key": meta.get("variant_key"),
        "points": points,
        "measured": len(measured),
        "spread": spread,
        "half_width": statistics.median(halves) if halves else None,
        "above": above,
        "below": below,
        "sign": sign,
        "apart": apart,
        "missing": missing,
        "excludes_zero": excludes,
        "verdict": verdict,
        "text": text,
    }


def _status_words(status: str) -> str:
    return {
        "failed": "the run failed",
        "cancelled": "the run was cancelled",
        "running": "the run hasn't finished",
        "draft": "the run hasn't run",
        "missing": "no run",
        "deleted": "the run it was made from was deleted",
    }.get(status, status)


def sweep_table(
    project: Project, sweep_id: str, *, choose: Choose = "first", top: int = TOP
) -> dict[str, Any]:
    """A sweep read back: its values and their runs, and for the ``top`` strongest sites (0: every
    site), chosen on the first value or on all values, each site's effect and interval at every
    value, the spread of its means across values, and whether it keeps its sign and size.

    A value whose run failed, hasn't finished or never ran has no results: it is missing, never
    zero. Only finished runs' summaries are read."""
    if choose not in ("first", "all"):
        raise SweepError("Choose the sites on the first value or on all values.")
    if isinstance(top, bool) or not isinstance(top, int) or not 0 <= top <= 500:
        raise SweepError("Show from 1 to 500 sites, or 0 for every site.")
    sweep = get_sweep(project, sweep_id)
    field = sweep.field
    values = sweep.values
    labels = [value_label(field, v) for v in values]
    n = len(values)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any] | None] = []
    first_spec: Spec | None = None
    for i, value in enumerate(values):
        listing = sweep.runs.get(i)
        status = listing.status if listing is not None else "missing"
        if listing is None and i == sweep.record.original:
            status = "deleted"  # the run it was made from, which stood in for this value
        summary = None
        spec = None
        error = listing.error if listing is not None else None
        if listing is not None:
            folder = project.run_dir(listing.id)
            try:
                spec = Spec.from_path(project.readable(folder / "spec.json"))
                if status == "finished":
                    summary = project.read_json(folder / "summary.json")
            except (ValueError, OSError) as exc:  # ProjectError and invalid files are ValueErrors
                status, error, summary = "failed", f"Its files can't be read: {exc}", None
        if summary is not None and first_spec is None:
            first_spec = spec
        summaries.append(summary)
        rows.append(
            {
                "index": i,
                "value": value,
                "label": labels[i],
                "run": listing.id if listing is not None else None,
                "status": status,
                "original": listing is not None and i == sweep.record.original,
                "title": (listing.title or listing.name) if listing is not None else None,
                "error": error,
                "n_prompts": (summary or {}).get("n_prompts"),
                # What a revision or a dataset resolved to: the commit it ran, the prompts it read.
                "revision": spec.model.revision if spec is not None else None,
                "dataset_sha256": spec.dataset.sha256 if spec is not None else None,
            }
        )

    # Every site any run measured, keyed as comparisons match them, in the order first measured.
    meta: dict[str, dict[str, Any]] = {}
    points: dict[str, list[dict[str, Any] | None]] = {}
    for i, summary in enumerate(summaries):
        for site in (summary or {}).get("sites") or []:
            key = _key(site)
            if key not in meta:
                meta[key] = site
                points[key] = [None] * n
            points[key][i] = _point(site)
    reasons = [
        None if s is not None else _status_words(rows[i]["status"]) for i, s in enumerate(summaries)
    ]

    def control(key: str) -> bool:
        return bool((meta[key].get("variant") or {}).get("control"))

    def magnitude(key: str, i: int) -> float | None:
        p = points[key][i]
        return abs(p["mean"]) if p is not None and p["mean"] is not None else None

    problem = None
    first = f"the first value, {value_name(field, values[0])}"
    if choose == "first":
        chosen_text = (
            f"Chosen on {first}: the sites with the largest |effect| there. Sites chosen on one "
            "value tend to look weaker at the others (the winner's curse)."
        )
        if summaries[0] is None:
            problem = (
                f"Sites can't be chosen on {first}: {_status_words(rows[0]['status'])}. Choose "
                "them on all values instead."
            )
            ranked = []
        else:
            ranked = [k for k in meta if not control(k) and magnitude(k, 0) is not None]
            ranked.sort(key=lambda k: -(magnitude(k, 0) or 0.0))
    else:
        chosen_text = (
            "Chosen on all values: the sites with the largest mean |effect| over the values that "
            "measured them."
        )

        def mean_magnitude(key: str) -> float:
            found = [m for i in range(n) if (m := magnitude(key, i)) is not None]
            return statistics.fmean(found) if found else -1.0

        ranked = [k for k in meta if not control(k) and mean_magnitude(k) >= 0]
        ranked.sort(key=lambda k: -mean_magnitude(k))
    chosen = ranked if top == 0 else ranked[:top]
    reasons_of = {
        key: [
            reasons[i] if summaries[i] is None else "not among that run's sites" for i in range(n)
        ]
        for key in chosen
    }
    sites = [
        _site_row(key, meta[key], points[key], reasons_of[key], field, labels, rank)
        for rank, key in enumerate(chosen, start=1)
    ]

    finished = sum(1 for s in summaries if s is not None)
    counts = {
        v: sum(1 for s in sites if s["verdict"] == v) for v in ("holds", "size", "sign", "unknown")
    }
    first_summary = next((s for s in summaries if s is not None), None)
    statistics_info = (first_summary or {}).get("statistics") or {}
    ci = statistics_info.get("ci")
    level = f"{round(ci * 100)}%" if isinstance(ci, int | float) else "confidence"
    notes = []
    if field.path == "statistics.seed":
        notes.append(
            "The bootstrap seed moves only the intervals: every run has the same means, so their "
            "spread across seeds is zero."
        )
    if first_spec is not None and isinstance(first_spec.scope, FeaturesScope):
        notes.append(
            "Each run keeps only its own strongest features, so a feature can be missing where a "
            "run didn't keep it."
        )
    if finished < n:
        notes.append(
            f"{n - finished} of {n} values have no results (a run failed, was cancelled or hasn't "
            "finished): they are missing, not zero."
        )
    if field.kind == "revision":
        spread_text = (
            f"The {field.plural} in the order given, such as checkpoints over training: the line "
            f"joins each run's mean, and each bar is that run's {level} interval over its prompts."
        )
    else:
        spread_text = (
            f"Each bar is one run's {level} interval: it resamples that run's prompts, so it leaves "
            f"out what changes with the {field.label}. The band is the spread across "
            f"{field.plural}: the mean of the runs' means ± 1 SD, which no single run's interval "
            "includes."
        )
    held = counts["holds"]
    if not finished:
        headline = "No run of this sweep has finished yet"
    elif problem is not None:
        headline = f"Sites can't be chosen on {first}"
    elif not sites:
        headline = "The runs of this sweep measured no sites"
    else:
        keep = "site keeps its" if len(sites) == 1 else "sites keep their"
        across = field.label if finished == 1 else field.plural
        headline = f"{held} of {len(sites)} {keep} sign and size across {finished} {across}"
    return {
        "sweep": sweep.id,
        "field": field.path,
        "kind": field.kind,
        "label": field.label,
        "plural": field.plural,
        "description": field.description,
        "name": _sweep_name(sweep, field),
        "created": sweep.created,
        "original": sweep.record.run,
        "stands_in": sweep.record.original,
        "values": rows,
        "finished": finished,
        "choose": choose,
        "top": top,
        "chosen": chosen_text,
        "problem": problem,
        "n_sites": len(meta),
        "sites": sites,
        "counts": counts,
        "headline": headline,
        "spread": spread_text,
        "notes": notes,
        "ci": ci,
        "measure": (first_summary or {}).get("measure", "intervention") if first_summary else None,
        "metric": (first_summary or {}).get("metric"),
        "experiment": first_spec.experiment.model_dump(mode="json") if first_spec else None,
    }


def _sweep_name(sweep: Sweep, field: SweepField) -> str:
    """What the sweep's spec is called: the original run's title or name, or a run's name without
    the value it adds."""
    original = sweep.runs.get(sweep.record.original) if sweep.record.original is not None else None
    if original is not None:
        return original.title or original.name
    if not sweep.runs:
        return sweep.id
    index, listing = min(sweep.runs.items())
    suffix = f" · {value_name(field, sweep.values[index])}"
    return listing.name.removesuffix(suffix) or listing.name


TABLE_COLUMNS = [
    "sweep",
    "field",
    "value_index",
    "value",
    "run",
    "status",
    "site_rank",
    "site",
    "kind",
    "layer",
    "head",
    "feature",
    "neuron",
    "position",
    "variant",
    "effect",
    "effect_lo",
    "effect_hi",
    "effect_sd",
    "n",
    "verdict",
    "spread_mean",
    "spread_sd",
]


def table_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    """The sweep table, one row per site and value (TABLE_COLUMNS: the CSV export, and
    ``Sweep.table()``). A value without results has no numbers and says why in ``status``."""
    rows = []
    for site in table["sites"]:
        spread = site.get("spread") or {}
        for value in table["values"]:
            point = site["points"][value["index"]] or {}
            rows.append(
                {
                    "sweep": table["sweep"],
                    "field": table["field"],
                    "value_index": value["index"],
                    "value": value["value"],
                    "run": value["run"],
                    "status": value["status"],
                    "site_rank": site["rank"],
                    "site": site["label"],
                    "kind": site["kind"],
                    "layer": site["layer"],
                    "head": site["head"],
                    "feature": site["feature"],
                    "neuron": site.get("neuron"),
                    "position": site["position_key"],
                    "variant": site["variant_key"],
                    "effect": point.get("mean"),
                    "effect_lo": point.get("lo"),
                    "effect_hi": point.get("hi"),
                    "effect_sd": point.get("sd"),
                    "n": point.get("n"),
                    "verdict": site["verdict"],
                    "spread_mean": spread.get("mean"),
                    "spread_sd": spread.get("sd"),
                }
            )
    return rows
