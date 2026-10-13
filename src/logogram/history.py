"""The history's own records: what a person says about a run, and deleting runs.

A run's spec.json records what ran, and never changes once it has run. What someone says about the
run afterwards lives beside it, in annotations.json::

    experiments/<id>/annotations.json
    {"logogram_annotations": 1, "title": "Name movers, clean run", "starred": true,
     "archived": false}

The title is shown instead of the spec's name, starred runs are found first, and archived runs are
kept out of the history until it is asked to show them. The file is read like any project file (a
regular file inside the project, size-limited and validated). One that can't be read is ignored,
with a note in the run's listing, never fatally, and it is never overwritten: it may hold what
someone wrote, so changing the annotations asks for it to be corrected or deleted first.

Deleting a run removes its folder for good: see :func:`delete_run` for why there is no trash.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from logogram.fileio import read_limited, remove_file, write_text_atomic
from logogram.project import Project, ProjectError, _mtime_iso
from logogram.schema import Manifest, RunAnnotations, RunListing

ANNOTATIONS = "annotations.json"
# Logogram writes a few hundred bytes: a file far larger than that isn't an annotations file.
ANNOTATIONS_LIMIT = 64 * 1024
TITLE_LIMIT = 200
# The statuses a run can be deleted in. Anything else, "running" or a status a later version gives
# a run in progress, is refused.
DELETABLE = ("finished", "failed", "cancelled", "draft")
_UNSET: Any = object()


class RunNotFound(ProjectError):
    """There is no run of that id in the project."""


class RunInUse(ProjectError):
    """The run is running or waiting to run, so its folder can't be deleted."""


def run_folder(project: Project, run_id: str) -> Path:
    """The folder of the run ``run_id``: a plain name in experiments/ whose folder is inside the
    project. Its spec may be unreadable: a damaged run can still be renamed or deleted."""
    folder = project.run_dir(run_id)
    if not folder.is_dir():
        raise RunNotFound(f"There is no run {run_id} in this project. It may have been deleted.")
    return folder


def reason(exc: BaseException) -> str:
    """What went wrong reading a file, as one sentence."""
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        text = f"{where}: {first['msg']}" if where else first["msg"]
    elif isinstance(exc, RecursionError):
        text = "it is nested too deeply"
    elif isinstance(exc, OSError):
        text = exc.strerror or type(exc).__name__
    else:
        text = str(exc) or type(exc).__name__
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else f"{text}."


def read_annotations(project: Project, folder: Path) -> tuple[RunAnnotations, str | None]:
    """A run's annotations, and, when its annotations.json can't be read, why (it is then
    ignored). A run without the file has nothing said about it."""
    path = folder / ANNOTATIONS
    if not os.path.lexists(path):
        return RunAnnotations(), None
    try:
        raw = read_limited(
            project.readable(path), ANNOTATIONS_LIMIT, "Logogram never writes one that large."
        )
        return RunAnnotations.model_validate_json(raw), None
    except (ValueError, OSError, RecursionError) as exc:  # ProjectError is a ValueError
        return RunAnnotations(), reason(exc)


def annotation_fields(project: Project, folder: Path) -> dict[str, Any]:
    """The listing's fields for what is said about a run, with a note when it was ignored."""
    annotations, problem = read_annotations(project, folder)
    note = None
    if problem is not None:
        note = (
            f"Its {ANNOTATIONS} can't be read, so its title, star and archive mark are ignored: "
            f"{problem} Correct the file, or delete it."
        )
    return {
        "title": annotations.title,
        "starred": annotations.starred,
        "archived": annotations.archived,
        "annotations_error": note,
    }


def write_annotations(project: Project, folder: Path, annotations: RunAnnotations) -> None:
    """Write a run's annotations.json through the project's helpers (never through a link at the
    file), or remove it once nothing is said about the run."""
    path = project.writable(folder / ANNOTATIONS)
    if annotations == RunAnnotations():
        remove_file(path)  # a link is removed itself, never followed
        return
    write_text_atomic(path, annotations.model_dump_json(indent=2) + "\n")


def annotate_run(
    project: Project,
    run_id: str,
    *,
    title: str | None = _UNSET,
    starred: bool | None = None,
    archived: bool | None = None,
) -> RunListing:
    """Change what is said about a run: its title (None or "" clears it, and so does the spec's
    own name), its star, whether it is archived. What is left out stays as it is, and the spec
    never changes. Returns the run's listing."""
    folder = run_folder(project, run_id)
    current, problem = read_annotations(project, folder)
    if problem is not None:
        raise ProjectError(
            f"This run's {ANNOTATIONS} can't be read: {problem} Logogram won't replace it, since it "
            "may hold what someone wrote: correct it, or delete it, then try again."
        )
    changes: dict[str, Any] = {}
    if title is not _UNSET:
        text = " ".join((title or "").split())  # one line, however it was typed
        if len(text) > TITLE_LIMIT:
            raise ProjectError(
                f"A title is at most {TITLE_LIMIT} characters long; this one has {len(text)}."
            )
        listing = project.run_listing(run_id)
        same = listing is not None and text == listing.name
        changes["title"] = None if not text or same else text
    if starred is not None:
        changes["starred"] = bool(starred)
    if archived is not None:
        changes["archived"] = bool(archived)
    updated = RunAnnotations.model_validate({**current.model_dump(), **changes})
    if updated != current:
        try:
            write_annotations(project, folder, updated)
        except OSError as exc:  # a read-only project, say
            raise ProjectError(
                f"Logogram can't write {ANNOTATIONS} in experiments/{run_id}: "
                f"{exc.strerror or type(exc).__name__}. Check that the project folder can be "
                "written to."
            ) from exc
    listing = project.run_listing(run_id)
    if listing is None:
        raise RunNotFound(f"There is no run {run_id} in this project. It may have been deleted.")
    return listing


def display_name(listing: RunListing) -> str:
    return f"“{listing.title or listing.name}”"


def queued_drafts(project: Project) -> set[str]:
    """The saved drafts waiting in the project's queue to run as they were saved (none when the
    saved queue can't be read: clearing it is what is offered then)."""
    from logogram import runqueue

    try:
        return {item.draft_id for item in runqueue.read_queue(project) if item.draft_id}
    except (runqueue.QueueError, OSError):
        return set()


def check_deletable(project: Project, run_id: str) -> RunListing | None:
    """Refuse (:class:`RunInUse`, saying why and what to do) a run that can't be deleted now: one
    in progress, here or in another process, or a draft waiting in the queue. Returns its listing
    (None for a folder whose run can't be listed)."""
    run_folder(project, run_id)
    listing = project.run_listing(run_id)
    if listing is not None and listing.status not in DELETABLE:
        if listing.status == "running":
            raise RunInUse(
                f"{display_name(listing)} is running, so its folder is in use. Wait for it to "
                "end, or cancel it, then delete it."
            )
        raise RunInUse(
            f"{display_name(listing)} can't be deleted while it is "
            f"{listing.status.replace('_', ' ')}."
        )
    if run_id in queued_drafts(project):
        name = display_name(listing) if listing is not None else run_id
        raise RunInUse(
            f"{name} is waiting in the queue to run. Take it out of the queue first, then "
            "delete it."
        )
    return listing


def delete_run(project: Project, run_id: str) -> None:
    """Delete a finished, failed or cancelled run, or a draft: its folder and everything in it.
    A run in progress can't be deleted, nor a draft waiting in the queue (its turn would find its
    folder gone).

    There is no trash. Files kept out of sight would still travel with the project when it is
    copied, zipped or shared (a run's manifest records the device it ran on), and take up space
    unseen. Archive a run to keep it out of the way and keep it. Runs derived from a deleted run
    stay, with its id in their manifest; research notes keep theirs too.

    Nothing outside the run's folder is touched, and no link is followed: a run folder that is a
    link is removed as a link (what it points to stays), links inside the folder are removed
    without being followed, and no run is deleted through an experiments folder that is a link.
    """
    experiments = project.experiments_dir
    if experiments.is_symlink() or experiments.resolve() != project.root.resolve() / "experiments":
        raise ProjectError(
            "This project's experiments folder links to another folder, so Logogram won't delete "
            "runs through it. Delete the run's folder yourself."
        )
    folder = run_folder(project, run_id)
    check_deletable(project, run_id)
    try:
        if folder.is_symlink() or _is_junction(folder):
            os.unlink(folder)  # the link itself: what it points to stays
        else:
            shutil.rmtree(folder)  # never follows a link, at the top or inside
    except OSError as exc:
        what = exc.strerror or type(exc).__name__
        name = Path(exc.filename).name if isinstance(exc.filename, str) else None
        raise ProjectError(
            f"Logogram couldn't delete all of experiments/{run_id}: {what}"
            f"{f' ({name})' if name else ''}. Check that no other program is using its files, "
            "then delete it again."
        ) from exc
    project.interrupted_runs.discard(run_id)


def _is_junction(path: Path) -> bool:
    check = getattr(os.path, "isjunction", None)  # Windows directory junctions, Python 3.12+
    return bool(check is not None and check(path))


def damaged_listing(project: Project, run_id: str, folder: Path, exc: BaseException) -> RunListing:
    """The listing of a run whose spec can't be read: a failed run whose error says what is wrong
    and how to fix it, so it never disappears from the history unexplained. It can be deleted."""
    if not os.path.lexists(folder / "spec.json"):
        problem = (
            "Its spec.json is missing, so Logogram can't tell what it ran. Restore the file from a "
            "backup or version control, or delete the run."
        )
    else:
        problem = (
            f"Its spec.json can't be read: {reason(exc)} Restore the file from a backup or "
            "version control, or delete the run."
        )
    manifest = None
    try:  # the manifest, when it can still be read, says when it ran and where it came from
        data = project.read_json(folder / "manifest.json", optional=True)
        manifest = Manifest.model_validate(data) if data is not None else None
    except Exception:  # noqa: BLE001, S110 - the listing says what matters: the spec
        pass
    return RunListing(
        id=run_id,
        name=run_id,
        description="A run whose spec can't be read",
        status="failed",
        created=(manifest.started_at if manifest else None) or _mtime_iso(folder),
        finished=manifest.finished_at if manifest else None,
        wall_time_s=None,
        n_prompts=None,
        layout_kind=None,
        model_id="",
        dataset="",
        dataset_source=manifest.dataset.source_path if manifest and manifest.dataset else None,
        experiment={},
        scope={},
        derived_from=manifest.derived_from if manifest else None,
        error=problem,
        damaged=True,
        **annotation_fields(project, folder),
    )
