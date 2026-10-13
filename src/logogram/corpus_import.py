"""Corpora from files, from Hugging Face datasets and from a project's prompts (logogram.corpora).

A file is read the way the person says it is laid out:

* plain text: one text per line (``lines``), or texts separated by blank lines (``paragraphs``);
* JSON Lines (``jsonl``), or a JSON array (``json``), of objects: a text field and, optionally, a
  label field;
* CSV or TSV with a header row (``csv``, ``tsv``): a text column and, optionally, a label column;
* Parquet (``parquet``): the same columns, read with pyarrow.

Rows whose text is empty (an empty string or null) are left out, and counted; a text that isn't a
string, or a label that isn't a string, a number or true/false, is refused with its row. Texts
keep their characters exactly, spaces included: a text file's lines lose only their line breaks.

A file of a Hugging Face dataset is downloaded once, when the person asks, at the exact commit its
revision resolves to (logogram.backends.hub, which respects HF_HUB_OFFLINE); the corpus records
the repository, the file and that commit. A project dataset's prompt pairs give their clean or
corrupt prompts (or both) as texts, labelled by side or by a field of their meta.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from logogram.corpora import (
    CORPUS_LIMIT,
    Corpus,
    CorpusError,
    CorpusSource,
    Label,
    TextRecord,
    write_corpus,
)
from logogram.project import Project

Format = Literal["lines", "paragraphs", "jsonl", "json", "csv", "tsv", "parquet"]
FORMATS: tuple[str, ...] = ("lines", "paragraphs", "jsonl", "json", "csv", "tsv", "parquet")
# The formats whose files can be pasted or uploaded as text (Parquet is binary).
TEXT_FORMATS = ("lines", "paragraphs", "jsonl", "json", "csv", "tsv")
FORMAT_WORDS = {
    "lines": "one text per line",
    "paragraphs": "texts separated by blank lines",
    "jsonl": "JSON Lines",
    "json": "a JSON array",
    "csv": "CSV",
    "tsv": "TSV",
    "parquet": "Parquet",
}
# A file of a Hugging Face dataset is read up to this size (large datasets come in several).
HUB_FILE_LIMIT = 2 * 1024**3
SIDES = ("clean", "corrupt")

ProgressFn = Callable[[int, int, str], None]


@dataclass
class Imported:
    """A corpus just written, and how many rows were left out for having no text."""

    corpus: Corpus
    skipped: int

    def to_dict(self) -> dict[str, Any]:
        corpus = self.corpus
        return {
            **corpus.describe(),
            "labels": corpus.labels(),
            "skipped": self.skipped,
            "source": None if corpus.meta is None else corpus.meta.source.model_dump(mode="json"),
        }


def format_of(filename: str, split: str = "lines") -> Format:
    """The format a file's name says it has; a text file's texts are split as ``split`` says."""
    lower = filename.lower()
    for ending, found in (
        (".jsonl", "jsonl"),
        (".ndjson", "jsonl"),
        (".json", "json"),
        (".csv", "csv"),
        (".tsv", "tsv"),
        (".parquet", "parquet"),
    ):
        if lower.endswith(ending):
            return found  # type: ignore[return-value]
    if lower.endswith((".gz", ".zst", ".bz2", ".xz", ".zip", ".arrow")):
        raise CorpusError(
            f"{_name(filename)} is compressed or in a format Logogram doesn't read. Choose a "
            "Parquet, JSON Lines, JSON, CSV, TSV or text file."
        )
    if split not in ("lines", "paragraphs"):
        raise CorpusError("Split a text file into lines or paragraphs.")
    return split  # type: ignore[return-value]


def _name(path: str) -> str:
    """A file's name without its folders, which corpora never record."""
    return path.replace("\\", "/").rsplit("/", 1)[-1]


# -- reading rows --------------------------------------------------------------------------------


def _lines(source: Iterable[str]) -> Iterator[str]:
    """Lines without their line breaks."""
    for line in source:
        yield line.rstrip("\n").rstrip("\r")


def _texts_by_line(source: Iterable[str]) -> Iterator[tuple[int, str]]:
    for number, line in enumerate(_lines(source), start=1):
        if line.strip():
            yield number, line


def _texts_by_paragraph(source: Iterable[str]) -> Iterator[tuple[int, str]]:
    lines: list[str] = []
    first = 0
    number = 0
    for number, line in enumerate(_lines(source), start=1):
        if line.strip():
            if not lines:
                first = number
            lines.append(line)
        elif lines:
            yield first, "\n".join(lines)
            lines = []
    if lines:
        yield first, "\n".join(lines)


def _json_rows(source: Iterable[str], where: str) -> Iterator[tuple[int, Any]]:
    for number, line in enumerate(_lines(source), start=1):
        if not line.strip():
            continue
        try:
            yield number, json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorpusError(
                f"{where}, line {number}: not valid JSON ({exc.msg}). Each line must be one JSON "
                "object."
            ) from exc


def _json_array(text: str, where: str) -> Iterator[tuple[int, Any]]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # Many datasets' .json files are JSON Lines.
        yield from _json_rows(io.StringIO(text), where)
        return
    if isinstance(value, dict):
        lists = [k for k, v in value.items() if isinstance(v, list)]
        raise CorpusError(
            f"{where} holds one JSON object, not a list of them"
            + (f" (its list is under {lists[0]!r})." if len(lists) == 1 else ".")
            + " Save the rows as a JSON array or as JSON Lines."
        )
    if not isinstance(value, list):
        raise CorpusError(f"{where} must hold a list of JSON objects, one for each text.")
    yield from enumerate(value, start=1)


def _csv_rows(source: Iterable[str], delimiter: str, where: str) -> Iterator[tuple[int, Any]]:
    reader = csv.reader(source, delimiter=delimiter)
    header: list[str] | None = None
    try:
        for row in reader:
            if header is None:
                header = row
                if not header or not any(h.strip() for h in header):
                    raise CorpusError(f"{where} needs a header row naming its columns.")
                continue
            if not row:
                continue
            yield reader.line_num, dict(zip(header, row, strict=False))
    except csv.Error as exc:
        raise CorpusError(f"{where}, line {reader.line_num}: {exc}.") from exc
    if header is None:
        raise CorpusError(f"{where} is empty.")


def _parquet_rows(
    path: Path, columns: list[str], where: str
) -> Iterator[tuple[int, dict[str, Any]]]:
    import pyarrow.parquet as pq

    try:
        file = pq.ParquetFile(path)
    except Exception as exc:  # any malformed file
        raise CorpusError(f"{where} isn't a Parquet file Logogram can read: {exc}") from exc
    names = list(file.schema_arrow.names)
    for column in columns:
        if column not in names:
            raise CorpusError(_missing_column(where, column, names))
    number = 0
    for batch in file.iter_batches(batch_size=4096, columns=columns):
        for row in batch.to_pylist():
            number += 1
            yield number, row


def _missing_column(where: str, column: str, names: Iterable[str]) -> str:
    shown = ", ".join(repr(n) for n in list(names)[:20])
    return f"{where} has no column {column!r}. Its columns are {shown}: choose one of them."


# -- rows into texts -----------------------------------------------------------------------------


def _label(value: Any, where: str) -> Label | None:
    """A label as a corpus keeps it: a string, a whole number or true/false; None when missing.
    Numbers stored as decimals that are whole (1.0) are whole numbers; others keep their text."""
    if type(value).__module__ == "numpy":
        value = value.item()  # a NumPy scalar, from Python
    if value is None:
        return None
    if isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value) if value.is_integer() else str(value)
    if isinstance(value, str):
        return value if value != "" else None
    raise CorpusError(
        f"{where}: the label is {type(value).__name__}, not a string, a number or true/false. "
        "Choose another label field, or none."
    )


class _Collector:
    """Texts gathered row by row, checked as they come, up to a limit and to what a corpus holds."""

    def __init__(self, where: str, limit: int | None) -> None:
        self.where = where
        self.limit = limit
        self.records: list[TextRecord] = []
        self.skipped = 0
        self.size = 0

    @property
    def full(self) -> bool:
        return self.limit is not None and len(self.records) >= self.limit

    def add(
        self,
        row: int,
        text: Any,
        label: Any = None,
        *,
        id: str | None = None,
        meta: dict[str, Any] | None = None,
        what: str = "row",
    ) -> None:
        where = f"{self.where}, {what} {row}"
        if text is None or (isinstance(text, str) and not text.strip()):
            self.skipped += 1
            return
        if not isinstance(text, str):
            raise CorpusError(
                f"{where}: the text is {type(text).__name__}, not a string. Choose the field that "
                "holds the text."
            )
        try:
            record = TextRecord(text=text, label=_label(label, where), id=id, meta=meta)
        except ValidationError as exc:
            first = exc.errors()[0]
            field = ".".join(str(p) for p in first["loc"])
            raise CorpusError(f"{where}: {field}: {first['msg']}.") from exc
        self.size += len(text.encode("utf-8")) + 16
        if self.size > CORPUS_LIMIT:
            raise CorpusError(
                f"The texts would be larger than the {CORPUS_LIMIT // 1024**2} MB a corpus holds. "
                "Keep fewer of them (a limit), or import them as several corpora."
            )
        self.records.append(record)

    def done(self) -> tuple[list[TextRecord], int]:
        if not self.records:
            empty = f" ({self.skipped} rows had no text)" if self.skipped else ""
            raise CorpusError(f"{self.where} has no texts{empty}. Check the file and the field.")
        return self.records, self.skipped


def _from_objects(
    rows: Iterable[tuple[int, Any]],
    text_field: str,
    label_field: str | None,
    collector: _Collector,
    what: str = "row",
) -> None:
    for number, row in rows:
        if collector.full:
            return
        if not isinstance(row, dict):
            raise CorpusError(
                f"{collector.where}, {what} {number}: expected an object with fields, such as "
                f'{{"{text_field}": "..."}}.'
            )
        if text_field not in row:
            raise CorpusError(
                _missing_column(f"{collector.where}, {what} {number}", text_field, row.keys())
            )
        if label_field is not None and label_field not in row:
            raise CorpusError(
                _missing_column(f"{collector.where}, {what} {number}", label_field, row.keys())
            )
        label = row.get(label_field) if label_field is not None else None
        collector.add(number, row[text_field], label, what=what)


def read_rows(
    source: Iterable[str] | Path,
    fmt: Format,
    *,
    where: str,
    text_field: str = "text",
    label_field: str | None = None,
    limit: int | None = None,
    whole: str | None = None,
) -> tuple[list[TextRecord], int]:
    """The texts of a file laid out as ``fmt``: its lines (``source``, an iterable of lines; for
    Parquet, its path), or, for a JSON array, its whole text (``whole``). Returns the texts and how
    many rows were left out for having none. ``where`` names the file in errors."""
    if limit is not None and limit < 1:
        raise CorpusError("Keep at least one text.")
    collector = _Collector(where, limit)
    if fmt in ("lines", "paragraphs"):
        if label_field is not None:
            raise CorpusError("A text file has no labels: leave out the label field.")
        lines = source if not isinstance(source, Path) else _open_lines(source)
        split = _texts_by_line if fmt == "lines" else _texts_by_paragraph
        what = "line" if fmt == "lines" else "the paragraph at line"
        for number, text in split(lines):
            if collector.full:
                break
            collector.add(number, text, what=what)
    elif fmt == "parquet":
        if not isinstance(source, Path):
            raise CorpusError("A Parquet file is read from its path.")
        columns = [text_field] + ([label_field] if label_field is not None else [])
        _from_objects(_parquet_rows(source, columns, where), text_field, label_field, collector)
    elif fmt == "json":
        text = whole if whole is not None else _read_text(source)
        _from_objects(_json_array(text, where), text_field, label_field, collector)
    elif fmt == "jsonl":
        lines = source if not isinstance(source, Path) else _open_lines(source)
        _from_objects(_json_rows(lines, where), text_field, label_field, collector, "line")
    elif fmt in ("csv", "tsv"):
        lines = source if not isinstance(source, Path) else _open_lines(source)
        delimiter = "," if fmt == "csv" else "\t"
        _from_objects(
            _csv_rows(lines, delimiter, where), text_field, label_field, collector, "line"
        )
    else:
        raise CorpusError(
            f"Logogram doesn't read {fmt!r} files. Choose one of {', '.join(FORMATS)}."
        )
    return collector.done()


def _open_lines(path: Path) -> Iterator[str]:
    try:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            yield from fh
    except UnicodeDecodeError as exc:
        raise CorpusError(
            f"{path.name} isn't UTF-8 text (byte {exc.start} can't be decoded). Save it as UTF-8."
        ) from exc


def _read_text(source: Iterable[str] | Path) -> str:
    if isinstance(source, Path):
        return "".join(_open_lines(source))
    return "".join(source)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _detail(
    name: str,
    fmt: Format,
    text_field: str,
    label_field: str | None,
    limit: int | None,
    skipped: int,
) -> str:
    """Where the texts came from, in words: '<name>, JSON Lines, field text, labels from label,
    the first 1,000 texts'."""
    parts = [name, FORMAT_WORDS[fmt]]
    if fmt not in ("lines", "paragraphs"):
        parts.append(f"text from {text_field!r}")
        if label_field is not None:
            parts.append(f"labels from {label_field!r}")
    if limit is not None:
        parts.append(f"the first {limit:,} texts")
    if skipped:
        parts.append(f"{skipped:,} rows without text left out")
    return ", ".join(parts)


# -- the three sources ---------------------------------------------------------------------------


def import_text(
    project: Project,
    name: str,
    content: str,
    *,
    filename: str,
    fmt: Format,
    text_field: str = "text",
    label_field: str | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> Imported:
    """A corpus from a file's text, as uploaded in the app or pasted: ``filename`` names it (its
    folder, if any, is left out)."""
    if fmt == "parquet":
        raise CorpusError(
            "A Parquet file can't be uploaded as text. Import it with `logogram corpora import`, "
            "from Python, or from a Hugging Face dataset."
        )
    shown = _name(filename) or "the text"
    content = content.lstrip("﻿")
    records, skipped = read_rows(
        io.StringIO(content, newline=None if fmt in ("lines", "paragraphs") else ""),
        fmt,
        where=shown,
        text_field=text_field,
        label_field=label_field,
        limit=limit,
        whole=content if fmt == "json" else None,
    )
    source = CorpusSource(
        kind="file",
        detail=_detail(shown, fmt, text_field, label_field, limit, skipped),
        path=shown,
        sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
    return Imported(write_corpus(project, name, records, source, replace=overwrite), skipped)


def import_file(
    project: Project,
    name: str,
    path: Path,
    *,
    fmt: Format | None = None,
    split: str = "lines",
    text_field: str = "text",
    label_field: str | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> Imported:
    """A corpus from a file on this machine (the command line's and Python's): its format from
    its name unless ``fmt`` says, a text file split into ``lines`` or ``paragraphs``. The corpus
    records the file's name and hash, never its folder."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise CorpusError(f"{path.name} isn't a file Logogram can read.")
    fmt = fmt or format_of(path.name, split)
    records, skipped = read_rows(
        path,
        fmt,
        where=path.name,
        text_field=text_field,
        label_field=label_field,
        limit=limit,
    )
    source = CorpusSource(
        kind="file",
        detail=_detail(path.name, fmt, text_field, label_field, limit, skipped),
        path=path.name,
        sha256=file_sha256(path),
    )
    return Imported(write_corpus(project, name, records, source, replace=overwrite), skipped)


def import_hub(
    project: Project,
    name: str,
    *,
    repo: str,
    path: str,
    revision: str | None,
    text_field: str = "text",
    label_field: str | None = None,
    limit: int | None = None,
    split: str = "lines",
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
    overwrite: bool = False,
) -> Imported:
    """A corpus from one file of a Hugging Face dataset (``repo``, ``path`` in it), downloaded at
    the exact commit ``revision`` resolves to (the main branch when None), with progress in bytes
    and cancelled by ``cancel``. The corpus records the repository, the file and the commit."""
    from logogram.backends import hub

    repo = repo.strip()
    path = path.strip().lstrip("/")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?", repo):
        raise CorpusError(
            f"{repo!r} isn't a Hugging Face dataset id: they look like owner/name, such as "
            "stanfordnlp/imdb."
        )
    if not path:
        raise CorpusError("Choose a file of the dataset: a Parquet, JSON Lines, CSV or text file.")
    if "\\" in path or any(part in ("", ".", "..") for part in path.split("/")):
        raise CorpusError(
            f"{path!r} isn't a file of a dataset: give its path in the repository, with forward "
            "slashes, such as data/train-00000-of-00001.parquet."
        )
    fmt = format_of(path, split)
    commit, local = hub.download_dataset_file(
        repo, path, revision, progress, cancel=cancel, limit=HUB_FILE_LIMIT
    )
    if local.stat().st_size > HUB_FILE_LIMIT:
        raise CorpusError(
            f"{path} is larger than the {HUB_FILE_LIMIT // 1024**3} GB Logogram reads of a "
            "dataset's file. Choose a smaller file."
        )
    where = _name(path)
    records, skipped = read_rows(
        local,
        fmt,
        where=where,
        text_field=text_field,
        label_field=label_field,
        limit=limit,
    )
    detail = _detail(
        f"{repo}, {path} at {commit[:7]}", fmt, text_field, label_field, limit, skipped
    )
    source = CorpusSource(
        kind="hub",
        detail=detail,
        repo=repo,
        path=path,
        revision=commit,
        sha256=file_sha256(local),
    )
    return Imported(write_corpus(project, name, records, source, replace=overwrite), skipped)


def check_label(label: str | None) -> tuple[str, str | None] | None:
    """A prompts' label choice: None, "side" (clean or corrupt), or "meta:FIELD" (a field of the
    prompts' meta). Returns ("side", None), ("meta", FIELD) or None."""
    if label is None or label == "":
        return None
    if label == "side":
        return "side", None
    if label.startswith("meta:") and label[5:].strip():
        return "meta", label[5:].strip()
    raise CorpusError(
        f"{label!r} isn't a way to label prompts: use 'side' (clean or corrupt), or "
        "'meta:FIELD' for a field of their meta, such as 'meta:template'."
    )


def import_prompts(
    project: Project,
    name: str,
    *,
    dataset: str,
    sides: Literal["clean", "corrupt", "both"],
    label: str | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> Imported:
    """A corpus from a project dataset's prompt pairs: their clean or corrupt prompts, or both
    (each pair's clean prompt, then its corrupt one), labelled by side, by a field of their meta
    (``label="meta:template"``), or not at all. The first ``limit`` pairs only, when given. Each
    text's meta says which prompt and side it is."""
    from logogram.datasets import decode_dataset, parse_jsonl, read_dataset_bytes
    from logogram.spec import check_relative_path

    if sides not in ("clean", "corrupt", "both"):
        raise CorpusError("Take the clean prompts, the corrupt prompts, or both.")
    by = check_label(label)
    try:
        check_relative_path(dataset, "the dataset")
    except ValueError as exc:
        raise CorpusError(str(exc)) from None
    file = project.resolve_dataset(dataset)
    content = read_dataset_bytes(file)
    pairs = parse_jsonl(decode_dataset(content, dataset), source=dataset)
    if limit is not None:
        if limit < 1:
            raise CorpusError("Take at least one prompt pair.")
        pairs = pairs[:limit]
    taken = SIDES if sides == "both" else (sides,)
    collector = _Collector(dataset, None)
    for i, pair in enumerate(pairs):
        value: Any = None
        if by is not None and by[0] == "meta":
            field = by[1]
            if pair.meta is None or field not in pair.meta:
                raise CorpusError(
                    f"{dataset}, prompt {i} has no meta field {field!r}. Label the texts by a "
                    "field every prompt has, or by side."
                )
            value = pair.meta[field]
        for side in taken:
            text = pair.clean if side == "clean" else pair.corrupt
            mark = side if by is not None and by[0] == "side" else value
            collector.add(
                i, text, mark, id=pair.id, meta={"prompt": i, "side": side}, what="prompt"
            )
    records, _ = collector.done()
    words = {
        "clean": "clean prompts",
        "corrupt": "corrupt prompts",
        "both": "clean and corrupt prompts",
    }
    labelled = (
        ", unlabelled"
        if by is None
        else ", labelled by side"
        if by[0] == "side"
        else f", labelled by their meta's {by[1]!r}"
    )
    first = f", the first {limit:,} pairs" if limit is not None else ""
    source = CorpusSource(
        kind="dataset",
        detail=f"{dataset}: {words[sides]}{first}{labelled}",
        path=dataset,
        sha256=hashlib.sha256(content).hexdigest(),
    )
    return Imported(write_corpus(project, name, records, source, replace=overwrite), 0)


def write_texts(
    project: Project,
    name: str,
    texts: list[str],
    labels: list[Any] | None = None,
    *,
    detail: str = "texts written from Python",
    overwrite: bool = False,
) -> Imported:
    """A corpus of ``texts``, each with the label at the same place in ``labels`` (if given)."""
    if labels is not None and len(labels) != len(texts):
        raise CorpusError(
            f"There are {len(texts):,} texts and {len(labels):,} labels: give one label for each "
            "text, or none."
        )
    collector = _Collector("the texts", None)
    for i, text in enumerate(texts):
        collector.add(i, text, None if labels is None else labels[i], what="text")
    records, skipped = collector.done()
    source = CorpusSource(kind="texts", detail=detail)
    return Imported(write_corpus(project, name, records, source, replace=overwrite), skipped)
