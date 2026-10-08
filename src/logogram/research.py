"""Portable research notes and named selections, with optimistic edit revisions."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from logogram.fileio import write_text_atomic
from logogram.project import Project, ProjectError, now_iso
from logogram.spec import ModelRef, Site


class NoteInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    body: str = Field(default="", max_length=50_000)
    model: ModelRef
    sites: list[Site] = Field(min_length=1, max_length=512)
    run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ResearchNote(NoteInput):
    id: str
    revision: int = Field(ge=1)
    created: str
    updated: str


class Notebook(BaseModel):
    model_config = ConfigDict(extra="forbid")

    logogram_research: Literal[1] = 1
    notes: list[ResearchNote] = Field(default_factory=list)


class NoteConflict(ValueError):
    pass


def read_notebook(project: Project) -> Notebook:
    data = project.read_json(project.root / "research.json", optional=True)
    return Notebook.model_validate(data) if data is not None else Notebook()


def save_note(
    project: Project, value: NoteInput, *, note_id: str | None = None, revision: int | None = None
) -> ResearchNote:
    book = read_notebook(project)
    old = next((n for n in book.notes if n.id == note_id), None)
    if note_id is not None and (old is None or old.revision != revision):
        raise NoteConflict("This note changed in another tab. Reload the note before saving again.")
    if value.run_id is not None:
        project.readable(project.run_dir(value.run_id) / "spec.json")
    now = now_iso()
    note = ResearchNote(
        **value.model_dump(),
        id=old.id if old else uuid.uuid4().hex,
        revision=old.revision + 1 if old else 1,
        created=old.created if old else now,
        updated=now,
    )
    book.notes = [note if n.id == note.id else n for n in book.notes]
    if old is None:
        book.notes.append(note)
    write_text_atomic(
        project.writable(project.root / "research.json"), book.model_dump_json(indent=2) + "\n"
    )
    return note


def delete_note(project: Project, note_id: str, revision: int) -> None:
    book = read_notebook(project)
    note = next((n for n in book.notes if n.id == note_id), None)
    if note is None:
        raise ProjectError("This note no longer exists. Refresh the notebook.")
    if note.revision != revision:
        raise NoteConflict("This note changed in another tab. Reload it before removing it.")
    book.notes = [n for n in book.notes if n.id != note_id]
    write_text_atomic(
        project.writable(project.root / "research.json"), book.model_dump_json(indent=2) + "\n"
    )
