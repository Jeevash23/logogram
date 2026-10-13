"""Corpora: plain texts in the project, for what reads text rather than prompt pairs.

Prompt pairs measure one behavior; a corpus is text the model reads as it would any text. The
analyses that look for what a neuron, a feature or a direction responds to, the representations of
labelled texts, and the SAEs and lenses trained in the project all read corpora.

A corpus is ``corpora/<name>.jsonl`` in the project, one JSON object per line::

    {"text": "The Eiffel Tower is in Paris.", "label": "true", "meta": {"topic": "places"}}

``text`` is required. ``label`` (a string, a number or true/false) names a class for analyses that
compare classes of text, such as a probe or a principal component coloured by class. ``id`` and
``meta`` are optional and carried through. Beside it, ``corpora/<name>.json`` says where the texts
came from (a file, a Hugging Face dataset at an exact commit, or a project's prompts), how many
there are and the hash of the JSONL file, which whatever reads the corpus records.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from logogram.fileio import (
    DATASET_LIMIT,
    JSON_LIMIT,
    read_limited,
    remove_file,
    size_text,
    write_text_atomic,
)
from logogram.project import Project, ProjectError, now_iso

CORPORA_DIR = "corpora"
CORPUS_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
# The largest corpus file read, as for datasets: about a million short texts.
CORPUS_LIMIT = DATASET_LIMIT
# The longest single text kept, in characters. Analyses read a bounded number of tokens of each.
TEXT_MAX = 1_000_000

Label = str | int | bool


class CorpusError(ValueError):
    """A corpus can't be read or written. The message says where and how to fix it."""


class TextRecord(BaseModel):
    """One text of a corpus."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=TEXT_MAX)
    label: Label | None = None
    id: str | None = None
    meta: dict[str, Any] | None = None


class CorpusSource(BaseModel):
    """Where a corpus's texts came from: a file (by its name only, never its folder), a Hugging
    Face dataset file at an exact commit, a project dataset's prompts, or texts written in the
    app or from Python."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["file", "hub", "dataset", "texts"]
    detail: str
    """In words, for example 'reviews.txt, one text per line'."""
    repo: str | None = None
    path: str | None = None
    revision: str | None = None
    sha256: str | None = None
    """The hash of the file the texts were read from: a file imported, a Hugging Face dataset's
    file, or a project dataset."""


class CorpusMeta(BaseModel):
    """``corpora/<name>.json``."""

    model_config = ConfigDict(extra="forbid")

    logogram_corpus: Literal[1] = 1
    name: str
    n: int = Field(ge=1)
    sha256: str
    labels: list[str]
    """The distinct labels, as text, in order of first appearance; empty when no text has one."""
    source: CorpusSource
    created: str


class Corpus:
    """A corpus read from the project: its texts, the hash of its file and what its JSON says."""

    def __init__(self, name: str, records: list[TextRecord], sha256: str, meta: CorpusMeta | None):
        self.name = name
        self.records = records
        self.sha256 = sha256
        self.meta = meta

    @property
    def path(self) -> str:
        return f"{CORPORA_DIR}/{self.name}.jsonl"

    @property
    def texts(self) -> list[str]:
        return [r.text for r in self.records]

    def labels(self) -> list[str]:
        return label_values(self.records)

    def describe(self) -> dict[str, Any]:
        """What an analysis records about the corpus it read."""
        return {"name": self.name, "path": self.path, "sha256": self.sha256, "n": len(self.records)}


def label_text(value: Label | None) -> str | None:
    """A label as the text analyses group by: true and false as 'true' and 'false'."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def label_values(records: list[TextRecord]) -> list[str]:
    seen: dict[str, None] = {}
    for record in records:
        text = label_text(record.label)
        if text is not None:
            seen.setdefault(text, None)
    return list(seen)


def check_corpus_name(name: str) -> str:
    stem = name[:-6] if name.endswith(".jsonl") else name
    if not CORPUS_NAME_RE.match(stem or ""):
        raise CorpusError(
            "Corpus names can use letters, digits, '.', '_' and '-', start with a letter or "
            "digit, and have at most 80 characters."
        )
    return stem


def corpora_dir(project: Project) -> Path:
    return project.root / CORPORA_DIR


def _files(project: Project, name: str) -> tuple[Path, Path]:
    folder = corpora_dir(project)
    return folder / f"{name}.jsonl", folder / f"{name}.json"


def parse_corpus(text: str, *, source: str = "corpus") -> list[TextRecord]:
    """The texts of a corpus file, each line checked; errors name the line and how to fix it."""
    records: list[TextRecord] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{source}, line {line_no}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorpusError(
                f"{where}: not valid JSON ({exc.msg}). Each line must be one JSON object with "
                "a text."
            ) from exc
        records.append(parse_text_record(raw, where))
    if not records:
        raise CorpusError(f"{source} has no texts. Add one JSON object with a text per line.")
    return records


def parse_text_record(raw: Any, where: str) -> TextRecord:
    if not isinstance(raw, dict):
        raise CorpusError(f"{where}: expected a JSON object with a text.")
    if "text" not in raw:
        raise CorpusError(f"{where}: missing 'text'. Each line needs the text itself.")
    try:
        return TextRecord.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(p) for p in first["loc"])
        if first["type"] == "extra_forbidden":
            raise CorpusError(
                f"{where}: {field} isn't a field of a text. Use text, and optionally label, id "
                "and meta (anything else goes in meta)."
            ) from exc
        raise CorpusError(f"{where}: {field}: {first['msg']}.") from exc


def to_jsonl(records: list[TextRecord]) -> str:
    return (
        "\n".join(
            json.dumps(r.model_dump(mode="json", exclude_none=True), ensure_ascii=False)
            for r in records
        )
        + "\n"
    )


def write_corpus(
    project: Project,
    name: str,
    records: list[TextRecord],
    source: CorpusSource,
    *,
    replace: bool = False,
) -> Corpus:
    """Write ``records`` as the corpus ``name`` with where they came from. An existing corpus of
    that name is replaced only with ``replace``."""
    name = check_corpus_name(name)
    if not records:
        raise CorpusError("A corpus needs at least one text.")
    text = to_jsonl(records)
    data = text.encode("utf-8")
    if len(data) > CORPUS_LIMIT:
        raise CorpusError(
            f"The corpus would be larger than the {size_text(CORPUS_LIMIT)} Logogram reads. Keep "
            "fewer texts, or split them into several corpora."
        )
    folder = corpora_dir(project)
    folder.mkdir(exist_ok=True)
    jsonl, info = _files(project, name)
    project.writable(jsonl)
    if not replace and (os.path.lexists(jsonl) or os.path.lexists(info)):
        raise CorpusError(f"A corpus called {name} already exists. Choose another name.")
    sha = hashlib.sha256(data).hexdigest()
    meta = CorpusMeta(
        name=name,
        n=len(records),
        sha256=sha,
        labels=label_values(records),
        source=source,
        created=now_iso(),
    )
    write_text_atomic(project.writable(jsonl), text, replace=replace)
    write_text_atomic(
        project.writable(info),
        json.dumps(meta.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        replace=replace,
    )
    _remember(sha, records)
    return Corpus(name, records, sha, meta)


# Corpora parsed lately, by the hash of their file: paging through one, or reading it again for
# another analysis, skips checking every line again. The hash is of the bytes, so a file that
# changed is parsed anew.
_parsed: dict[str, list[TextRecord]] = {}
PARSED_KEPT = 4


def _remember(sha: str, records: list[TextRecord]) -> None:
    while len(_parsed) >= PARSED_KEPT:
        _parsed.pop(next(iter(_parsed)), None)
    _parsed[sha] = records


def read_corpus(project: Project, name: str, sha256: str | None = None) -> Corpus:
    """The corpus ``name``; with ``sha256``, refused unless its file still has that hash."""
    name = check_corpus_name(name)
    jsonl, info = _files(project, name)
    if not os.path.lexists(jsonl):
        raise CorpusError(
            f"This project has no corpus called {name}. Import one first: a text or JSONL file, "
            "a Hugging Face dataset, or a dataset's prompts."
        )
    try:
        content = read_limited(
            project.readable(jsonl), CORPUS_LIMIT, "Split the texts into smaller corpora."
        )
    except (ProjectError, OSError, ValueError) as exc:
        raise CorpusError(f"The corpus {name} can't be read: {exc}") from exc
    sha = hashlib.sha256(content).hexdigest()
    if sha256 is not None and sha != sha256:
        raise CorpusError(
            f"The corpus {name} has changed since this was set up (its hash no longer matches). "
            "Import the original texts again, or choose the corpus as it is now."
        )
    records = _parsed.get(sha)
    if records is None:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CorpusError(
                f"{name}.jsonl isn't UTF-8 text (byte {exc.start} can't be decoded). Save it as "
                "UTF-8."
            ) from exc
        records = parse_corpus(text, source=f"{name}.jsonl")
        _remember(sha, records)
    meta = None
    if os.path.lexists(info):
        try:
            raw = read_limited(project.readable(info), JSON_LIMIT, "Import the corpus again.")
            meta = CorpusMeta.model_validate_json(raw)
        except (ProjectError, OSError, ValueError, ValidationError):
            meta = None  # the texts are what count; their description is best effort
    return Corpus(name, records, sha, meta)


# Texts on a page of a corpus, as the app pages a dataset's prompts; and the characters of each
# text a page shows (the rest is read by the analyses, not shown).
PAGE_SIZE = 100
SHOWN_CHARS = 2000


def search_text(record: TextRecord) -> str:
    """What a search looks through for one text: the text, its label, id and meta, lowercase, a
    line between fields so a search never matches across two of them."""
    fields = [record.text]
    label = label_text(record.label)
    if label is not None:
        fields.append(label)
    if record.id is not None:
        fields.append(record.id)
    for key, value in (record.meta or {}).items():
        fields.append(f"{key}: {value if isinstance(value, str) else json.dumps(value)}")
    return "\n".join(fields).lower()


def corpus_page(
    corpus: Corpus, *, query: str = "", page: int = 0, size: int = PAGE_SIZE
) -> dict[str, Any]:
    """One page of a corpus's texts, those containing ``query`` (ignoring case; every text when
    it is empty), each with its index in the corpus. A page past the last shows the last. Long
    texts are cut to their first SHOWN_CHARS characters, and say so."""
    q = query.strip().lower()
    matches = [i for i, r in enumerate(corpus.records) if not q or q in search_text(r)]
    pages = max(1, -(-len(matches) // size))
    shown = min(max(0, page), pages - 1)
    start = shown * size
    rows = []
    for i in matches[start : start + size]:
        record = corpus.records[i]
        rows.append(
            {
                "index": i,
                "text": record.text[:SHOWN_CHARS],
                "length": len(record.text),
                "label": label_text(record.label),
                "id": record.id,
                "meta": record.meta,
            }
        )
    meta = corpus.meta
    return {
        **corpus.describe(),
        "labels": corpus.labels(),
        "source": None if meta is None else meta.source.model_dump(mode="json"),
        "created": None if meta is None else meta.created,
        "query": query,
        "matches": len(matches),
        "page": shown,
        "pages": pages,
        "size": size,
        "start": start,
        "texts": rows,
    }


def list_corpora(project: Project) -> list[dict[str, Any]]:
    """The project's corpora by name, with what their JSON files say; one that can't be read
    says why (``error``)."""
    folder = corpora_dir(project)
    if not project.inside(folder) or not folder.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(folder.glob("*.jsonl"), key=lambda p: p.name.lower()):
        name = path.name[: -len(".jsonl")]
        if not CORPUS_NAME_RE.match(name):
            continue
        try:
            corpus = read_corpus(project, name)
        except CorpusError as exc:
            out.append({"name": name, "path": f"{CORPORA_DIR}/{name}.jsonl", "error": str(exc)})
            continue
        meta = corpus.meta
        out.append(
            {
                "name": name,
                "path": corpus.path,
                "n": len(corpus.records),
                "sha256": corpus.sha256,
                "labels": corpus.labels(),
                "source": None if meta is None else meta.source.model_dump(mode="json"),
                "created": None if meta is None else meta.created,
                "error": None,
            }
        )
    return out


def delete_corpus(project: Project, name: str) -> None:
    """Delete the corpus ``name`` and its description."""
    name = check_corpus_name(name)
    jsonl, info = _files(project, name)
    if not os.path.lexists(jsonl) and not os.path.lexists(info):
        raise CorpusError(f"This project has no corpus called {name}.")
    for path in (jsonl, info):
        if os.path.lexists(path):
            remove_file(project.writable(path))


def rename_corpus(project: Project, name: str, new: str) -> Corpus:
    """Give the corpus ``name`` the name ``new``; refuses a name that is taken."""
    corpus = read_corpus(project, name)
    new = check_corpus_name(new)
    if new == corpus.name:
        return corpus
    jsonl, info = _files(project, new)
    if os.path.lexists(jsonl) or os.path.lexists(info):
        raise CorpusError(f"A corpus called {new} already exists. Choose another name.")
    source = (
        corpus.meta.source
        if corpus.meta is not None
        else CorpusSource(kind="texts", detail="renamed from a corpus without a description")
    )
    written = write_corpus(project, new, corpus.records, source)
    delete_corpus(project, name)
    return written
