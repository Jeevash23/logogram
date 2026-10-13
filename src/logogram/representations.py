"""Representations: the activations at a site, read over a dataset's prompts or a corpus's texts.

Each prompt or text gives one row: the activation at one of the residual stream's sites, an
attention output or an MLP output, at one layer and one position (the last token, a token index, a
labelled position of a dataset's prompts, or the mean over the tokens after the
beginning-of-sequence token). From the rows:

* their principal components: the rows centered on their mean, decomposed by a singular value
  decomposition in float64, each component's sign fixed so its largest-magnitude loading is
  positive, so the same rows always give the same components and coordinates;
* the mean difference between two classes of rows (a dataset's prompts marked by side or by a
  field of their meta, a corpus's texts by their labels);
* the activations themselves, exported to ``activations/<name>.safetensors`` (one tensor per
  site, ``[rows, width]`` in float32) with ``activations/<name>.json`` saying what they are.

A principal component or a mean difference can be saved as a direction (logogram.directions).

Texts and prompts are read as they are, after the beginning-of-sequence token when it is chosen;
a corpus's texts are cut to their first ``max_tokens`` tokens. Rows run in batches of one token
length (shortest first, each in the order of its rows), so no padding is needed and the same
settings always give the same batches.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from logogram.backends.base import Cancelled, ModelBackend, Tokenized, float64
from logogram.corpora import label_text
from logogram.fileio import atomic_output, write_text_atomic
from logogram.project import Project, now_iso
from logogram.spec import IndexPosition, LabelPosition, LastPosition, StreamSiteKind

ACTIVATIONS_DIR = "activations"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
# The most tokens one analysis reads (rows times their tokens), so it ends in reasonable time and
# memory; and the most rows of activations exported or decomposed at once.
TOKENS_LIMIT = 4_000_000
ROWS_LIMIT = 200_000
# The largest export written: every site's tensor together.
EXPORT_LIMIT = 2 * 1024**3
# How many principal components are reported.
COMPONENTS = 10
# Characters of each row's text an answer carries, for hovering over a point.
SHOWN_CHARS = 300

SITE_WORDS = {
    "resid_pre": "the residual stream entering layer {layer}",
    "resid_mid": "the residual stream between layer {layer}'s attention and MLP",
    "resid_post": "the residual stream leaving layer {layer}",
    "attn_out": "layer {layer}'s attention output",
    "mlp_out": "layer {layer}'s MLP output",
}

ProgressFn = Callable[[int, int], None]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReadSite(_Strict):
    """Where activations are read: a site of the residual stream, or an attention or MLP output,
    at a layer."""

    kind: StreamSiteKind
    layer: int = Field(ge=0)

    @property
    def name(self) -> str:
        """``kind:layer``, as the command line writes it and an export names its tensor."""
        return f"{self.kind}:{self.layer}"


class MeanPosition(_Strict):
    """The mean over a prompt's or text's tokens, after the beginning-of-sequence token."""

    kind: Literal["mean"] = "mean"


ReadPosition = Annotated[
    LastPosition | IndexPosition | LabelPosition | MeanPosition, Field(discriminator="kind")
]


class DatasetRows(_Strict):
    """A project dataset's prompts: the clean ones, the corrupt ones, or both (each pair's clean
    prompt, then its corrupt one), marked by side, by a field of their meta, or not at all."""

    kind: Literal["dataset"] = "dataset"
    path: str = Field(min_length=1)
    sha256: str | None = None
    sides: Literal["clean", "corrupt", "both"]
    mark: str | None = None
    """None, "side", or "meta:FIELD" (such as "meta:template")."""
    limit: int | None = Field(default=None, ge=1)
    """The first n prompt pairs only."""


class CorpusRows(_Strict):
    """A corpus's texts, marked by their labels, each cut to its first ``max_tokens`` tokens."""

    kind: Literal["corpus"] = "corpus"
    name: str = Field(min_length=1)
    sha256: str | None = None
    limit: int | None = Field(default=None, ge=1)
    """The first n texts only."""
    max_tokens: int = Field(ge=1, le=100_000)


Rows = Annotated[DatasetRows | CorpusRows, Field(discriminator="kind")]


@dataclass
class TextRow:
    """One prompt or text, as the model reads it."""

    index: int
    """Its index among the dataset's prompt pairs, or the corpus's texts."""
    side: str | None
    """clean or corrupt, for a prompt."""
    mark: str | None
    """Its class: its side, its meta's field or its label, as text."""
    text: str
    tokens: Tokenized
    labels: dict[str, int] = field(default_factory=dict)

    @property
    def where(self) -> str:
        return f"prompt {self.index} ({self.side})" if self.side else f"text {self.index}"


@dataclass
class LoadedRows:
    rows: list[TextRow]
    source: dict[str, Any]
    """What was read, as an answer records it: the file, its hash and the choices."""
    truncated: int = 0
    """Texts cut at max_tokens."""

    @property
    def marks(self) -> list[str]:
        seen: dict[str, None] = {}
        for row in self.rows:
            if row.mark is not None:
                seen.setdefault(row.mark, None)
        return list(seen)


def site_words(kind: str, layer: int) -> str:
    return SITE_WORDS.get(kind, f"{kind} of layer {{layer}}").format(layer=layer)


def position_words(position: Any) -> str:
    if isinstance(position, LastPosition):
        return "the last token"
    if isinstance(position, IndexPosition):
        return f"token {position.index}" + (" from the end" if position.index < 0 else "")
    if isinstance(position, LabelPosition):
        return f"the token labelled {position.label}"
    return "the mean over the tokens after the beginning-of-sequence token"


def source_words(source: dict[str, Any]) -> str:
    if source["kind"] == "dataset":
        sides = {
            "clean": "clean prompts",
            "corrupt": "corrupt prompts",
            "both": "clean and corrupt prompts",
        }
        first = f", the first {source['limit']:,} pairs" if source.get("limit") else ""
        return f"{source['path']} ({sides[source['sides']]}{first})"
    first = f"the first {source['limit']:,} texts" if source.get("limit") else "its texts"
    return f"the corpus {source['name']} ({first}, at most {source['max_tokens']:,} tokens each)"


def check_sites(backend: ModelBackend, sites: list[ReadSite]) -> None:
    info = backend.info
    if not sites:
        raise ValueError("Choose a site to read.")
    for site in sites:
        if not 0 <= site.layer < info.n_layers:
            raise ValueError(
                f"Layer {site.layer} isn't in this model, which has layers 0 to {info.n_layers - 1}."
            )
        if site.kind not in info.site_kinds:
            raise ValueError(
                f"{site_words(site.kind, site.layer).capitalize()} isn't a site of {info.id}: when "
                "it loaded, its layers didn't add attention and the MLP in turn. Choose another "
                "site."
            )


# -- prompts and texts as rows -------------------------------------------------------------------


def _tokenize(backend: ModelBackend, text: str, prepend_bos: bool) -> Tokenized:
    if prepend_bos and backend.bos_token_id is None:
        raise ValueError(
            f"{backend.info.id} has no beginning-of-sequence token: read the texts without one."
        )
    return backend.tokenize(text, prepend_bos)


def _cut(tokens: Tokenized, n: int) -> Tokenized:
    return Tokenized(ids=tokens.ids[:n], tokens=tokens.tokens[:n], offsets=tokens.offsets[:n])


def _labels(tokens: Tokenized, spans: dict[str, tuple[int, int]] | None) -> dict[str, int]:
    from logogram.prompts import label_position

    out: dict[str, int] = {}
    for label, span in (spans or {}).items():
        pos = label_position(tokens, span)
        if pos is not None:
            out[label] = pos
    return out


def load_rows(
    backend: ModelBackend, project: Project, source: DatasetRows | CorpusRows, prepend_bos: bool
) -> LoadedRows:
    """The rows ``source`` names, tokenized as the model reads them, with each one's class."""
    if isinstance(source, CorpusRows):
        return _corpus_rows(backend, project, source, prepend_bos)
    return _dataset_rows(backend, project, source, prepend_bos)


def _dataset_rows(
    backend: ModelBackend, project: Project, source: DatasetRows, prepend_bos: bool
) -> LoadedRows:
    from logogram.corpus_import import _label, check_label
    from logogram.datasets import decode_dataset, parse_jsonl, read_dataset_bytes
    from logogram.spec import check_relative_path

    check_relative_path(source.path, "the dataset")
    content = read_dataset_bytes(project.resolve_dataset(source.path))
    sha = hashlib.sha256(content).hexdigest()
    if source.sha256 is not None and sha != source.sha256:
        raise ValueError(
            f"{source.path} has changed since these settings were made (its hash no longer "
            "matches). Choose it again as it is now."
        )
    records = parse_jsonl(decode_dataset(content, source.path), source=source.path)
    if source.limit is not None:
        records = records[: source.limit]
    by = check_label(source.mark)
    sides = ("clean", "corrupt") if source.sides == "both" else (source.sides,)
    rows: list[TextRow] = []
    for i, record in enumerate(records):
        clean = _tokenize(backend, record.clean, prepend_bos)
        corrupt = _tokenize(backend, record.corrupt, prepend_bos)
        clean_labels = _labels(clean, record.positions)
        if record.corrupt_positions is not None:
            corrupt_labels = _labels(corrupt, record.corrupt_positions)
        else:  # the prompts line up only when they have the same length
            corrupt_labels = dict(clean_labels) if len(clean.ids) == len(corrupt.ids) else {}
        value: str | None = None
        if by is not None and by[0] == "meta":
            name = by[1]
            if record.meta is None or name not in record.meta:
                raise ValueError(
                    f"{source.path}, prompt {i} has no meta field {name!r}. Mark the prompts by a "
                    "field every prompt has, or by side."
                )
            value = label_text(_label(record.meta[name], f"{source.path}, prompt {i}"))
        for side in sides:
            mark = side if by is not None and by[0] == "side" else value
            tokens, labels = (clean, clean_labels) if side == "clean" else (corrupt, corrupt_labels)
            text = record.clean if side == "clean" else record.corrupt
            rows.append(TextRow(i, side, mark, text, tokens, labels))
    described = {
        "kind": "dataset",
        "path": source.path,
        "sha256": sha,
        "sides": source.sides,
        "mark": source.mark,
        "limit": source.limit,
        "n": len(rows),
    }
    return LoadedRows(_checked(backend, rows), described)


def _corpus_rows(
    backend: ModelBackend, project: Project, source: CorpusRows, prepend_bos: bool
) -> LoadedRows:
    from logogram.corpora import read_corpus

    check_max_tokens(backend, source.max_tokens, prepend_bos)
    corpus = read_corpus(project, source.name, source.sha256)
    records = corpus.records[: source.limit] if source.limit is not None else corpus.records
    rows: list[TextRow] = []
    truncated = 0
    for i, record in enumerate(records):
        tokens = _tokenize(backend, record.text, prepend_bos)
        if len(tokens.ids) > source.max_tokens:
            tokens = _cut(tokens, source.max_tokens)
            truncated += 1
        rows.append(TextRow(i, None, label_text(record.label), record.text, tokens))
    described = {
        **corpus.describe(),
        "kind": "corpus",
        "limit": source.limit,
        "max_tokens": source.max_tokens,
        "truncated": truncated,
        "n": len(rows),
    }
    return LoadedRows(_checked(backend, rows), described, truncated)


def check_max_tokens(backend: ModelBackend, max_tokens: int, prepend_bos: bool) -> None:
    """Refuse a number of tokens per text the model can't read at once, or that leaves no token
    after the beginning-of-sequence token."""
    n_ctx = backend.info.n_ctx
    if max_tokens > n_ctx:
        raise ValueError(
            f"{backend.info.id} reads at most {n_ctx:,} tokens at once: read at most that many "
            "tokens of each text."
        )
    if max_tokens < (2 if prepend_bos else 1):
        raise ValueError(
            "Read at least one token of each text after the beginning-of-sequence token."
        )


def _checked(backend: ModelBackend, rows: list[TextRow]) -> list[TextRow]:
    if not rows:
        raise ValueError("There is nothing to read: choose prompts or texts.")
    if len(rows) > ROWS_LIMIT:
        raise ValueError(
            f"That is {len(rows):,} prompts or texts; Logogram reads at most {ROWS_LIMIT:,} at "
            "once. Read fewer of them."
        )
    total = sum(len(r.tokens.ids) for r in rows)
    if total > TOKENS_LIMIT:
        raise ValueError(
            f"These settings read {total:,} tokens; an analysis reads at most {TOKENS_LIMIT:,} at "
            "once. Read fewer texts, or fewer tokens of each."
        )
    n_ctx = backend.info.n_ctx
    for row in rows:
        if len(row.tokens.ids) > n_ctx:
            raise ValueError(
                f"{row.where.capitalize()} is {len(row.tokens.ids):,} tokens, and "
                f"{backend.info.id} reads at most {n_ctx:,}. Shorten it."
            )
    return rows


def length_batches(lengths: list[int], batch_size: int) -> Iterator[list[int]]:
    """Rows grouped by token length, shortest first, each group in the order of its rows, in
    batches of at most ``batch_size``: no batch mixes lengths, so none needs padding."""
    by_length: dict[int, list[int]] = defaultdict(list)
    for i, n in enumerate(lengths):
        by_length[n].append(i)
    for n in sorted(by_length):
        members = by_length[n]
        for start in range(0, len(members), batch_size):
            yield members[start : start + batch_size]


def resolve_position(position: Any, row: TextRow, first: int) -> int | None:
    """The token of ``row`` to read at ``position``, or None for the mean over its tokens from
    ``first`` on."""
    n = len(row.tokens.ids)
    if isinstance(position, LastPosition):
        return n - 1
    if isinstance(position, IndexPosition):
        at = position.index if position.index >= 0 else n + position.index
        if not 0 <= at < n:
            raise ValueError(
                f"{row.where.capitalize()} has {n} tokens, so token {position.index} isn't in it. "
                "Choose a smaller index, the last token, or the mean."
            )
        return at
    if isinstance(position, LabelPosition):
        if row.side is None:
            raise ValueError(
                "A corpus's texts have no labelled positions: read the last token, an index, or "
                "the mean."
            )
        if position.label not in row.labels:
            raise ValueError(
                f"{row.where.capitalize()} has no token labelled {position.label!r}. Choose a "
                "label every prompt has (a corrupt prompt of another length needs "
                "corrupt_positions), or another position."
            )
        return row.labels[position.label]
    if n <= first:
        raise ValueError(
            f"{row.where.capitalize()} has no tokens after the beginning-of-sequence token to "
            "average."
        )
    return None


def collect(
    backend: ModelBackend,
    loaded: LoadedRows,
    sites: list[ReadSite],
    position: Any,
    *,
    prepend_bos: bool,
    batch_size: int,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> dict[str, np.ndarray]:
    """Each site's activations at ``position``, one row per prompt or text: ``[rows, width]`` in
    float64, by the site's name (``kind:layer``)."""
    from logogram.prompts import first_real_token

    check_sites(backend, sites)
    rows = loaded.rows
    first = first_real_token(backend, rows[0].tokens.ids, prepend_bos, False)
    where = [resolve_position(position, row, first) for row in rows]
    keys = list(dict.fromkeys((s.kind, s.layer) for s in sites))
    width = backend.info.d_model
    out = {f"{k}:{layer}": np.zeros((len(rows), width), dtype=np.float64) for k, layer in keys}
    done = 0
    for batch in length_batches([len(r.tokens.ids) for r in rows], batch_size):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        tokens = torch.tensor([rows[i].tokens.ids for i in batch], dtype=torch.long)
        captured = backend.capture(tokens, keys)
        for key in keys:
            acts = float64(captured[key]).cpu()  # [b, length, width]
            if not torch.isfinite(acts).all():
                raise ValueError(
                    "The model produced activations that aren't finite numbers. Load it in "
                    "float32 and try again."
                )
            target = out[f"{key[0]}:{key[1]}"]
            for r, i in enumerate(batch):
                at = where[i]
                vector = acts[r, first:].mean(dim=0) if at is None else acts[r, at]
                target[i] = vector.numpy()
        done += len(batch)
        if progress is not None:
            progress(done, len(rows))
    return out


# -- principal components and mean differences ---------------------------------------------------


@dataclass
class Components:
    """The principal components of rows of activations."""

    components: np.ndarray
    """``[k, width]``, orthonormal rows, each signed so its largest-magnitude loading is positive."""
    explained: np.ndarray
    """The share of the rows' variance each component explains."""
    variances: np.ndarray
    """Each component's variance (over rows, with n - 1 in the denominator)."""
    coords: np.ndarray
    """``[rows, k]``: each row's coordinate on each component, after centering."""
    mean: np.ndarray
    total_variance: float


def principal_components(values: np.ndarray, k: int = COMPONENTS) -> Components:
    """The top ``k`` principal components of ``values`` (``[rows, width]``): centered on their
    mean, decomposed in float64 by a singular value decomposition."""
    x = np.asarray(values, dtype=np.float64)
    n = x.shape[0]
    if n < 2:
        raise ValueError("Principal components need at least two prompts or texts.")
    mean = x.mean(axis=0)
    centered = x - mean
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    squares = s**2
    total = float(squares.sum())
    if not total > 0:
        raise ValueError("Every prompt or text has the same activation here: there is no variance.")
    k = min(k, len(s))
    components = vt[:k].copy()
    for j in range(k):
        if components[j, int(np.argmax(np.abs(components[j])))] < 0:
            components[j] = -components[j]
    return Components(
        components=components,
        explained=squares[:k] / total,
        variances=squares[:k] / (n - 1),
        coords=centered @ components.T,
        mean=mean,
        total_variance=total / (n - 1),
    )


def class_rows(loaded: LoadedRows, mark: str) -> list[int]:
    return [i for i, row in enumerate(loaded.rows) if row.mark == mark]


def mean_difference(
    values: np.ndarray, loaded: LoadedRows, positive: str, negative: str
) -> tuple[np.ndarray, int, int]:
    """The mean of the rows marked ``positive`` minus the mean of those marked ``negative``, and
    how many rows each class has."""
    if positive == negative:
        raise ValueError("Choose two different classes.")
    marks = loaded.marks
    a, b = class_rows(loaded, positive), class_rows(loaded, negative)
    for name, members in ((positive, a), (negative, b)):
        if not members:
            shown = ", ".join(repr(m) for m in marks[:12]) or "none"
            raise ValueError(f"No prompt or text is marked {name!r}. The classes are {shown}.")
    x = np.asarray(values, dtype=np.float64)
    return x[a].mean(axis=0) - x[b].mean(axis=0), len(a), len(b)


# -- answers -------------------------------------------------------------------------------------


def settings_of(
    loaded: LoadedRows, site: ReadSite | None, position: Any, prepend_bos: bool, batch_size: int
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "source": loaded.source,
        "position": position.model_dump(mode="json"),
        "prepend_bos": prepend_bos,
        "batch_size": batch_size,
    }
    if site is not None:
        out["site"] = site.model_dump(mode="json")
    return out


def pca_report(
    backend: ModelBackend,
    loaded: LoadedRows,
    site: ReadSite,
    position: Any,
    *,
    prepend_bos: bool,
    batch_size: int,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """The rows' principal components at ``site``, as the Representations view shows them: the
    share of variance each of the first ten explains, and each row's coordinates on them, with
    its class and the start of its text."""
    values = collect(
        backend,
        loaded,
        [site],
        position,
        prepend_bos=prepend_bos,
        batch_size=batch_size,
        progress=progress,
    )[site.name]
    found = principal_components(values)
    points = [
        {
            "row": i,
            "index": row.index,
            "side": row.side,
            "mark": row.mark,
            "text": row.text[:SHOWN_CHARS],
            "coords": [float(c) for c in found.coords[i]],
        }
        for i, row in enumerate(loaded.rows)
    ]
    return {
        **settings_of(loaded, site, position, prepend_bos, batch_size),
        "n": len(loaded.rows),
        "width": int(values.shape[1]),
        "components": len(found.explained),
        "explained": [float(x) for x in found.explained],
        "variances": [float(x) for x in found.variances],
        "total_variance": found.total_variance,
        "marks": loaded.marks,
        "points": points,
        "description": f"{site_words(site.kind, site.layer)} at {position_words(position)}, over "
        f"{source_words(loaded.source)}",
    }


# -- directions from components and classes ------------------------------------------------------


def save_component(
    project: Project,
    backend: ModelBackend,
    loaded: LoadedRows,
    site: ReadSite,
    position: Any,
    component: int,
    name: str,
    *,
    prepend_bos: bool,
    batch_size: int,
    replace: bool = False,
) -> Any:
    """Save principal component ``component`` (from 1) of the rows at ``site`` as the direction
    ``name``, saying what it was found on."""
    from logogram.directions import model_of, save_direction

    values = collect(
        backend, loaded, [site], position, prepend_bos=prepend_bos, batch_size=batch_size
    )[site.name]
    found = principal_components(values)
    if not 1 <= component <= len(found.explained):
        raise ValueError(f"Choose a component from 1 to {len(found.explained)}.")
    share = float(found.explained[component - 1])
    detail = (
        f"principal component {component} of {site_words(site.kind, site.layer)} at "
        f"{position_words(position)}, over {source_words(loaded.source)}: {len(loaded.rows):,} "
        f"rows, centered; it explains {share:.1%} of their variance"
    )
    return save_direction(
        project,
        name,
        torch.from_numpy(found.components[component - 1].copy()),
        model=model_of(backend.info),
        site={"kind": site.kind, "layer": site.layer},
        method="pca",
        run=None,
        detail=detail,
        replace=replace,
    )


def save_mean_difference(
    project: Project,
    backend: ModelBackend,
    loaded: LoadedRows,
    site: ReadSite,
    position: Any,
    positive: str,
    negative: str,
    name: str,
    *,
    prepend_bos: bool,
    batch_size: int,
    replace: bool = False,
) -> Any:
    """Save the mean of the rows marked ``positive`` minus the mean of those marked ``negative``,
    at ``site``, as the direction ``name`` (its length before it was made unit length is kept)."""
    from logogram.directions import model_of, save_direction

    values = collect(
        backend, loaded, [site], position, prepend_bos=prepend_bos, batch_size=batch_size
    )[site.name]
    vector, n_pos, n_neg = mean_difference(values, loaded, positive, negative)
    detail = (
        f"the mean over {n_pos:,} rows marked {positive!r} minus the mean over {n_neg:,} marked "
        f"{negative!r}, of {site_words(site.kind, site.layer)} at {position_words(position)}, "
        f"over {source_words(loaded.source)}"
    )
    return save_direction(
        project,
        name,
        torch.from_numpy(vector),
        model=model_of(backend.info),
        site={"kind": site.kind, "layer": site.layer},
        method="mean_difference",
        run=None,
        detail=detail,
        replace=replace,
    )


# -- exported activations ------------------------------------------------------------------------


def check_export_name(name: str) -> str:
    if not NAME_RE.match(name or ""):
        raise ValueError(
            f"{name!r} can't name an export: use letters, digits, dots, dashes and underscores, "
            "starting with a letter or digit, at most 80 characters."
        )
    return name


def export_activations(
    project: Project,
    name: str,
    backend: ModelBackend,
    loaded: LoadedRows,
    sites: list[ReadSite],
    position: Any,
    *,
    prepend_bos: bool,
    batch_size: int,
    overwrite: bool = False,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Write the activations at ``sites`` and ``position`` to ``activations/<name>.safetensors``
    (one tensor per site, named ``kind:layer``, ``[rows, width]`` in float32) and what they are to
    ``activations/<name>.json``: the model, the sites, the position, the dataset or corpus read and
    its hash, each row's index, side and class, and how the texts were tokenized. Returns where
    they were written, in the project."""
    from safetensors.numpy import save

    from logogram.directions import model_of

    check_export_name(name)
    check_sites(backend, sites)
    sites = list({s.name: s for s in sites}.values())
    size = len(loaded.rows) * backend.info.d_model * 4 * len(sites)
    if size > EXPORT_LIMIT:
        raise ValueError(
            f"The export would be {size / 1e9:.1f} GB; Logogram writes at most "
            f"{EXPORT_LIMIT / 1e9:.0f} GB at once. Export fewer rows or sites."
        )
    folder = project.root / ACTIVATIONS_DIR
    tensor_path, json_path = folder / f"{name}.safetensors", folder / f"{name}.json"
    if not overwrite and (os.path.lexists(tensor_path) or os.path.lexists(json_path)):
        raise ValueError(f"An export called {name} already exists. Choose another name.")
    values = collect(
        backend,
        loaded,
        sites,
        position,
        prepend_bos=prepend_bos,
        batch_size=batch_size,
        progress=progress,
    )
    tensors = {key: np.ascontiguousarray(v, dtype=np.float32) for key, v in values.items()}
    data = save(tensors, metadata={"format": "np"})
    info = backend.info
    meta = {
        "logogram_activations": 1,
        "name": name,
        "created": now_iso(),
        "model": {**model_of(info), "dtype": info.dtype, "process_weights": info.process_weights},
        "sites": [
            {"kind": s.kind, "layer": s.layer, "tensor": s.name, "width": int(info.d_model)}
            for s in sites
        ],
        "position": position.model_dump(mode="json"),
        "source": loaded.source,
        "tokenization": {
            "prepend_bos": prepend_bos,
            "max_tokens": loaded.source.get("max_tokens"),
        },
        "batch_size": batch_size,
        "dtype": "float32",
        "n": len(loaded.rows),
        "rows": [
            {"row": i, "index": row.index, "side": row.side, "label": row.mark}
            for i, row in enumerate(loaded.rows)
        ],
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    folder.mkdir(exist_ok=True)
    # The tensors first: a JSON file always describes tensors that are there.
    with atomic_output(project.writable(tensor_path), replace=overwrite) as out:
        out.write(data)
    write_text_atomic(
        project.writable(json_path),
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        replace=overwrite,
    )
    return {
        "name": name,
        "path": f"{ACTIVATIONS_DIR}/{name}.safetensors",
        "json_path": f"{ACTIVATIONS_DIR}/{name}.json",
        "n": len(loaded.rows),
        "sites": [s.name for s in sites],
        "sha256": meta["sha256"],
        "bytes": len(data),
    }


# -- the command line's and Python's ways of naming sites and positions ---------------------------


def parse_site(value: str | dict[str, Any] | ReadSite) -> ReadSite:
    """A site from ``"resid_post:6"`` (as the command line writes it), a dict, or a ReadSite."""
    if isinstance(value, ReadSite):
        return value
    if isinstance(value, dict):
        return ReadSite.model_validate(value)
    kind, _, layer = str(value).partition(":")
    if kind not in SITE_WORDS or not layer.strip().isdigit():
        raise ValueError(
            f"{value!r} isn't a site: write KIND:LAYER, such as resid_post:6, with KIND one of "
            f"{', '.join(SITE_WORDS)}."
        )
    return ReadSite(kind=kind, layer=int(layer))  # type: ignore[arg-type]


def parse_position(value: str | int | dict[str, Any] | Any) -> Any:
    """A position from ``"last"``, ``"mean"``, an index (``3``, ``-1``), ``"label:S2"``, a dict
    as a spec writes positions, or a position itself."""
    if isinstance(value, LastPosition | IndexPosition | LabelPosition | MeanPosition):
        return value
    if isinstance(value, dict):
        kind = value.get("kind")
        models = {"last": LastPosition, "index": IndexPosition, "label": LabelPosition}
        return models.get(str(kind), MeanPosition).model_validate(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return IndexPosition(index=value)
    text = str(value).strip()
    if text == "last":
        return LastPosition()
    if text == "mean":
        return MeanPosition()
    if re.fullmatch(r"-?\d+", text):
        return IndexPosition(index=int(text))
    if text.startswith("label:") and text[6:].strip():
        return LabelPosition(label=text[6:].strip())
    raise ValueError(
        f"{value!r} isn't a position: write last, mean, a token index (negative counts from the "
        "end), or label:NAME for a labelled position of the prompts."
    )
