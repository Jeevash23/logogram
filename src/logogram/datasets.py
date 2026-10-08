"""Prompt datasets: JSONL files of clean/corrupt pairs with an answer and a distractor.

Each line is one JSON object::

    {"clean": "When Mary and John went to the store, John gave a drink to",
     "corrupt": "When Mary and John went to the store, Mary gave a drink to",
     "answer": " Mary", "distractor": " John",
     "positions": {"IO": [5, 9], "S1": [14, 18], "S2": [38, 42]}}

``positions`` is optional. It names character spans in the clean prompt; a label points to the
last token that overlaps its span. ``id`` and ``meta`` are optional and carried through.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from logogram.fileio import write_text_atomic

DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


class DatasetError(ValueError):
    """A dataset file can't be read. The message says where and how to fix it."""


class PromptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clean: str = Field(min_length=1)
    corrupt: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    distractor: str = Field(min_length=1)
    positions: dict[str, tuple[int, int]] | None = None
    id: str | None = None
    meta: dict[str, Any] | None = None

    @field_validator("positions")
    @classmethod
    def _spans_ordered(
        cls, value: dict[str, tuple[int, int]] | None
    ) -> dict[str, tuple[int, int]] | None:
        if value is None:
            return value
        for label, (start, end) in value.items():
            if not label or not label.strip():
                raise ValueError("position labels must be non-empty")
            if label in ("all", "last") or re.fullmatch(r"-?\d+", label):
                raise ValueError(
                    f"{label!r} can't be a position label: 'all', 'last' and numbers already "
                    "name positions"
                )
            if start < 0 or end <= start:
                raise ValueError(f"position {label!r} must be a span [start, end) with start < end")
        return value


def parse_jsonl(text: str, *, source: str = "dataset") -> list[PromptRecord]:
    records: list[PromptRecord] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(
                f"{source}, line {line_no}: not valid JSON ({exc.msg}). "
                "Each line must be one JSON object."
            ) from exc
        if not isinstance(raw, dict):
            raise DatasetError(f"{source}, line {line_no}: expected a JSON object.")
        missing = [k for k in ("clean", "corrupt", "answer", "distractor") if k not in raw]
        if missing:
            raise DatasetError(
                f"{source}, line {line_no}: missing {', '.join(repr(m) for m in missing)}. "
                "Each line needs clean, corrupt, answer and distractor."
            )
        if isinstance(raw.get("positions"), dict):
            raw["positions"] = {
                k: tuple(v) if isinstance(v, list) else v for k, v in raw["positions"].items()
            }
        try:
            record = PromptRecord.model_validate(raw)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first["loc"])
            raise DatasetError(f"{source}, line {line_no}: {where}: {first['msg']}.") from exc
        _check_spans(record, line_no, source)
        records.append(record)
    if not records:
        raise DatasetError(f"{source} has no prompts. Add one JSON object per line.")
    return records


def _check_spans(record: PromptRecord, line_no: int, source: str) -> None:
    if not record.positions:
        return
    for label, (_start, end) in record.positions.items():
        if end > len(record.clean):
            raise DatasetError(
                f"{source}, line {line_no}: position {label!r} ends at character {end}, but "
                f"the clean prompt has {len(record.clean)} characters."
            )


def load_dataset(path: Path) -> list[PromptRecord]:
    if not path.exists():
        raise DatasetError(f"Dataset not found: {path.name}. Check the path in the spec.")
    if not path.is_file():  # a folder, device or pipe could fail, block or never end
        raise DatasetError(f"{path.name} isn't a regular file.")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetError(
            f"{path.name} isn't UTF-8 text (byte {exc.start} can't be decoded). Save it as UTF-8."
        ) from exc
    except OSError as exc:
        raise DatasetError(f"{path.name} can't be read: {exc.strerror or exc}.") from exc
    return parse_jsonl(text, source=path.name)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_jsonl(records: list[PromptRecord]) -> str:
    lines = []
    for record in records:
        data = record.model_dump(mode="json", exclude_none=True)
        if "positions" in data:
            data["positions"] = {k: list(v) for k, v in data["positions"].items()}
        lines.append(json.dumps(data, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def write_dataset(path: Path, records: list[PromptRecord]) -> None:
    """Write a dataset file. Callers in a project get ``path`` from ``Project.dataset_file``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, to_jsonl(records))


def check_dataset_name(name: str) -> str:
    stem = name[:-6] if name.endswith(".jsonl") else name
    if not DATASET_NAME_RE.match(stem):
        raise DatasetError(
            "Dataset names can use letters, digits, '.', '_' and '-', and must start with a "
            "letter or digit."
        )
    return stem + ".jsonl"
