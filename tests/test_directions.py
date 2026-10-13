"""Directions saved in a project: written exactly, read back checked, pinned by runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from logogram.directions import (
    DirectionError,
    check_fits,
    delete_direction,
    is_orthonormal,
    list_directions,
    orthonormal,
    pin_direction,
    read_direction,
    rename_direction,
    save_direction,
)
from logogram.project import Project
from logogram.spec import DirectionRef

MODEL = {"id": "tiny", "source": "hub", "revision": "abc123", "sha256": None}


def _save(project: Project, name: str, vectors: torch.Tensor, **kwargs: object):
    return save_direction(
        project,
        name,
        vectors,
        model=MODEL,
        site={"kind": "resid_post", "layer": 1},
        method="probe",
        run=None,
        detail="a test",
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.fixture
def empty_project(tmp_path: Path) -> Project:
    return Project.create(tmp_path, "Directions")


def test_one_vector_is_saved_at_unit_length_and_read_back_exactly(empty_project: Project) -> None:
    vector = torch.tensor([3.0, 4.0, 0.0, 0.0])
    saved = _save(empty_project, "toward-answer", vector)
    assert saved.meta.norms == [5.0]
    assert torch.equal(saved.vectors, torch.tensor([[0.6, 0.8, 0.0, 0.0]]))
    read = read_direction(empty_project, "toward-answer")
    assert torch.equal(read.vectors, saved.vectors)
    assert read.sha256 == saved.sha256 and read.dimensions == 1 and read.width == 4
    assert read.meta.made_by.method == "probe" and read.meta.site is not None
    assert read.path == "directions/toward-answer.safetensors"


def test_a_subspace_is_made_orthonormal_keeping_each_vector_s_side() -> None:
    vectors = torch.tensor([[2.0, 0.0, 0.0], [1.0, -3.0, 0.0]])
    rows, norms = orthonormal(vectors)
    assert is_orthonormal(rows)
    assert torch.allclose(rows[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(rows[1], torch.tensor([0.0, -1.0, 0.0]))  # the second vector's side
    assert norms == pytest.approx([2.0, 10**0.5])


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        (torch.zeros(3), "zeros"),
        (torch.tensor([[1.0, 0.0], [2.0, 0.0]]), "span"),
        (torch.tensor([float("nan"), 1.0]), "finite"),
        (torch.ones(3, 2), "more dimensions"),
    ],
)
def test_vectors_that_make_no_direction_are_refused(vectors: torch.Tensor, message: str) -> None:
    with pytest.raises(DirectionError, match=message):
        orthonormal(vectors)


def test_a_name_is_never_replaced_without_asking(empty_project: Project) -> None:
    first = _save(empty_project, "d", torch.tensor([1.0, 0.0]))
    with pytest.raises(DirectionError, match="already exists"):
        _save(empty_project, "d", torch.tensor([0.0, 1.0]))
    second = _save(empty_project, "d", torch.tensor([0.0, 1.0]), replace=True)
    assert second.sha256 != first.sha256
    assert torch.equal(read_direction(empty_project, "d").vectors, torch.tensor([[0.0, 1.0]]))


@pytest.mark.parametrize("name", ["", "../x", ".hidden", "a/b", "snapshots", "x" * 81])
def test_names_that_could_leave_the_folder_are_refused(empty_project: Project, name: str) -> None:
    with pytest.raises(DirectionError):
        _save(empty_project, name, torch.tensor([1.0, 0.0]))
    with pytest.raises(ValueError):
        DirectionRef(name=name, sha256=None)


def test_the_list_says_why_a_direction_can_t_be_read(empty_project: Project) -> None:
    _save(empty_project, "good", torch.tensor([1.0, 0.0]))
    _save(empty_project, "edited", torch.tensor([0.0, 1.0]))
    tensor = empty_project.root / "directions" / "edited.safetensors"
    tensor.write_bytes(tensor.read_bytes()[:-4] + b"\0\0\0\0")
    listed = {d["name"]: d for d in list_directions(empty_project)}
    assert listed["good"]["error"] is None and listed["good"]["dimensions"] == 1
    assert "has changed since it was saved" in listed["edited"]["error"]


def test_rename_and_delete(empty_project: Project) -> None:
    saved = _save(empty_project, "old", torch.tensor([1.0, 2.0]))
    _save(empty_project, "taken", torch.tensor([2.0, 1.0]))
    with pytest.raises(DirectionError, match="already exists"):
        rename_direction(empty_project, "old", "taken")
    renamed = rename_direction(empty_project, "old", "new")
    assert renamed.sha256 == saved.sha256 and renamed.name == "new"
    assert [d["name"] for d in list_directions(empty_project)] == ["new", "taken"]
    delete_direction(empty_project, "new")
    with pytest.raises(DirectionError, match="no direction called new"):
        read_direction(empty_project, "new")


def test_a_run_keeps_the_direction_it_pinned(empty_project: Project) -> None:
    first = _save(empty_project, "d", torch.tensor([1.0, 0.0, 0.0]))
    pinned = pin_direction(empty_project, "d", None)
    assert pinned.sha256 == first.sha256
    assert pinned.path == f"directions/snapshots/{first.sha256}.safetensors"
    # Saved again under the same name: the run's own copy is still there.
    _save(empty_project, "d", torch.tensor([0.0, 1.0, 0.0]), replace=True)
    again = pin_direction(empty_project, "d", first.sha256)
    assert torch.equal(again.vectors, first.vectors)
    # Without a snapshot, a changed direction is refused rather than used silently.
    os.remove(empty_project.root / "directions" / "snapshots" / f"{first.sha256}.json")
    with pytest.raises(DirectionError, match="has changed since this spec was written"):
        pin_direction(empty_project, "d", first.sha256)


def test_a_direction_must_fit_the_model(empty_project: Project) -> None:
    saved = _save(empty_project, "d", torch.tensor([1.0, 0.0, 0.0, 0.0]))
    same = SimpleNamespace(id="tiny", source="hub", revision="abc123", sha256=None, d_model=4)
    assert check_fits(saved, same, "the residual stream") == []
    other = SimpleNamespace(id="other", source="hub", revision="r", sha256=None, d_model=4)
    assert "may not mean the same thing" in check_fits(saved, other, "the residual stream")[0]
    narrow = SimpleNamespace(**{**vars(same), "d_model": 3})
    with pytest.raises(DirectionError, match="4 wide"):
        check_fits(saved, narrow, "the residual stream")


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_a_directions_folder_that_links_outside_is_never_written(
    empty_project: Project, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (empty_project.root / "directions").symlink_to(outside, target_is_directory=True)
    with pytest.raises(Exception, match="outside the project"):
        _save(empty_project, "d", torch.tensor([1.0, 0.0]))
    assert list(outside.iterdir()) == []


def test_the_json_file_is_plain_and_names_no_machine(empty_project: Project) -> None:
    _save(empty_project, "d", torch.tensor([1.0, 0.0]))
    data = json.loads((empty_project.root / "directions" / "d.json").read_text(encoding="utf-8"))
    assert data["logogram_direction"] == 1 and data["model"] == MODEL
    assert str(empty_project.root) not in json.dumps(data)
