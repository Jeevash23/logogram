"""Directions: vectors in a model's activation space, saved in the project to be used again.

One method finds a direction and others use it. A probe's weights, a subspace learned by
distributed alignment search, a principal component and the mean difference between two sets of
prompts are all directions; steering adds one, directional ablation removes one, subspace
patching patches along one and a readout metric measures one.

A direction is two files in the project's ``directions/`` folder:

* ``<name>.safetensors`` holds one tensor, ``directions``, ``[k, width]`` in float32: one row for a
  direction, k rows for a subspace of k dimensions. Rows are orthonormal (each of length 1, and at
  right angles to the others), so the component of an activation along them is ``x @ D.T @ D``.
* ``<name>.json`` says what it is: the model it was found on (by id and revision, or a local
  model's files by their hash), the site and layer, how it was made, the hash of the tensor file,
  and the lengths of the vectors before they were made unit length.

A run pins a direction by the hash of its tensor file and keeps a copy in
``directions/snapshots/``, as runs keep their prompts, so saving another direction under the same
name later never changes a run, and a run's spec still finds the direction it used.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from logogram.fileio import JSON_LIMIT, atomic_output, read_limited, remove_file, write_text_atomic
from logogram.project import Project, ProjectError, now_iso

DIRECTIONS_DIR = "directions"
SNAPSHOTS = "snapshots"
TENSOR = "directions"
# A direction's name, which is also its files' name: letters, digits, dots, dashes and underscores.
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# The largest tensor file read: a subspace of 64 dimensions in a 16,384-wide model is 4 MB.
TENSOR_LIMIT = 64 * 1024**2
# The most dimensions a saved subspace has.
MAX_DIMENSIONS = 64
# Rows count as orthonormal when their Gram matrix is within this of the identity.
ORTHONORMAL_TOLERANCE = 1e-4

Method = Literal["probe", "das", "pca", "mean_difference", "import", "sae_feature"]

METHOD_WORDS: dict[str, str] = {
    "probe": "a linear probe's weights",
    "das": "a subspace learned by distributed alignment search",
    "pca": "a principal component",
    "mean_difference": "the mean difference between two sets of prompts",
    "import": "vectors imported from a safetensors file",
    "sae_feature": "an SAE feature's decoder direction",
}


class DirectionError(ValueError):
    """A direction can't be saved, found or read. The message says why and how to fix it."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DirectionModel(_Strict):
    """The model a direction was found on: a Hugging Face model by id and revision, or a local
    model by the hash of its files."""

    id: str
    source: Literal["hub", "local"]
    revision: str | None
    sha256: str | None


class DirectionSite(_Strict):
    """Where in the model the direction was found: a stream site (or head output) and a layer."""

    kind: str
    layer: int = Field(ge=0)


class MadeBy(_Strict):
    """How a direction was made: the method, the run it came from (if any) and in words."""

    method: Method
    run: str | None
    detail: str


class DirectionMeta(_Strict):
    """``directions/<name>.json``."""

    logogram_direction: Literal[1] = 1
    name: str
    model: DirectionModel
    site: DirectionSite | None
    width: int = Field(ge=1)
    dimensions: int = Field(ge=1, le=MAX_DIMENSIONS)
    made_by: MadeBy
    created: str
    sha256: str
    norms: list[float]
    notes: str = ""


class Direction:
    """A direction read from the project: what its JSON file says, and its rows ``[k, width]``
    in float32 on the CPU."""

    def __init__(self, meta: DirectionMeta, vectors: torch.Tensor, path: str) -> None:
        self.meta = meta
        self.vectors = vectors
        # Where it was read from, relative to the project: its own file or its snapshot.
        self.path = path

    @property
    def name(self) -> str:
        return self.meta.name

    @property
    def sha256(self) -> str:
        return self.meta.sha256

    @property
    def dimensions(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def width(self) -> int:
        return int(self.vectors.shape[1])

    def to_dict(self) -> dict[str, Any]:
        return {**self.meta.model_dump(mode="json"), "path": self.path}


def check_name(name: str) -> str:
    if not NAME_RE.match(name or ""):
        raise DirectionError(
            f"{name!r} can't name a direction: use letters, digits, dots, dashes and underscores, "
            "starting with a letter or digit, at most 80 characters."
        )
    if name == SNAPSHOTS:
        raise DirectionError(f"{SNAPSHOTS!r} is where runs keep their copies: choose another name.")
    return name


def directions_dir(project: Project) -> Path:
    return project.root / DIRECTIONS_DIR


def _files(project: Project, name: str, snapshot: bool = False) -> tuple[Path, Path]:
    folder = directions_dir(project) / SNAPSHOTS if snapshot else directions_dir(project)
    return folder / f"{name}.json", folder / f"{name}.safetensors"


def orthonormal(vectors: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
    """``vectors`` ``[k, width]`` as orthonormal rows spanning the same subspace, in float32, and
    the length of each original vector. A single vector is divided by its length; several are
    orthonormalized in order (Gram-Schmidt, through a QR decomposition in float64), so the first
    row keeps the first vector's direction. Vectors of length zero, or that lie in the span of the
    ones before them, are refused."""
    if vectors.dim() == 1:
        vectors = vectors[None, :]
    if vectors.dim() != 2 or vectors.shape[0] < 1 or vectors.shape[1] < 1:
        raise DirectionError("A direction is one vector, or a few vectors of the same width.")
    if vectors.shape[0] > MAX_DIMENSIONS:
        raise DirectionError(
            f"A saved subspace has at most {MAX_DIMENSIONS} dimensions; this has "
            f"{vectors.shape[0]}."
        )
    if vectors.shape[0] > vectors.shape[1]:
        raise DirectionError("A subspace can't have more dimensions than the space it lives in.")
    v = vectors.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(v).all():
        raise DirectionError("A direction's values must be finite numbers.")
    norms = v.norm(dim=1)
    if bool((norms <= 1e-12).any()):
        raise DirectionError("A direction can't be a vector of zeros.")
    q, r = torch.linalg.qr(v.T)  # v.T = q r, q [width, k] with orthonormal columns
    diag = torch.diagonal(r)
    if bool((diag.abs() <= 1e-9 * norms.max()).any()):
        raise DirectionError(
            "These vectors don't span as many dimensions as there are vectors: one lies in the "
            "span of the others."
        )
    rows = (q * torch.sign(diag)[None, :]).T  # each row points the way its vector did
    return rows.to(torch.float32).contiguous(), [float(n) for n in norms]


def is_orthonormal(vectors: torch.Tensor) -> bool:
    v = vectors.to(torch.float64)
    gram = v @ v.T
    return bool(
        (gram - torch.eye(v.shape[0], dtype=torch.float64)).abs().max() <= ORTHONORMAL_TOLERANCE
    )


def tensor_bytes(vectors: torch.Tensor) -> bytes:
    from safetensors.torch import save

    return save({TENSOR: vectors.to(torch.float32).contiguous()}, metadata={"format": "pt"})


def save_direction(
    project: Project,
    name: str,
    vectors: torch.Tensor,
    *,
    model: dict[str, Any],
    site: dict[str, Any] | None,
    method: Method,
    run: str | None,
    detail: str,
    notes: str = "",
    replace: bool = False,
) -> Direction:
    """Save ``vectors`` (one vector ``[width]``, or ``[k, width]``) as the direction ``name``, made
    orthonormal (see :func:`orthonormal`). ``model`` names the model it was found on (``id``,
    ``source``, ``revision``, ``sha256``), ``site`` where (``kind``, ``layer``), or None.

    An existing direction of that name is replaced only with ``replace``; runs that used it keep
    their own copy either way."""
    check_name(name)
    rows, norms = orthonormal(vectors)
    data = tensor_bytes(rows)
    sha = hashlib.sha256(data).hexdigest()
    meta = DirectionMeta(
        name=name,
        model=DirectionModel.model_validate(model),
        site=None if site is None else DirectionSite.model_validate(site),
        width=int(rows.shape[1]),
        dimensions=int(rows.shape[0]),
        made_by=MadeBy(method=method, run=run, detail=detail),
        created=now_iso(),
        sha256=sha,
        norms=norms,
        notes=notes,
    )
    folder = directions_dir(project)
    folder.mkdir(exist_ok=True)
    json_path, tensor_path = _files(project, name)
    project.writable(json_path)
    if not replace and (os.path.lexists(json_path) or os.path.lexists(tensor_path)):
        raise DirectionError(
            f"A direction called {name} already exists. Choose another name, or replace it."
        )
    # The tensor first: a JSON file always describes a tensor that is there.
    with atomic_output(project.writable(tensor_path), replace=replace) as out:
        out.write(data)
    write_text_atomic(
        project.writable(json_path),
        json.dumps(meta.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        replace=replace,
    )
    return Direction(meta, rows, f"{DIRECTIONS_DIR}/{name}.safetensors")


def _read_files(project: Project, json_path: Path, tensor_path: Path, label: str) -> Direction:
    try:
        raw = read_limited(project.readable(json_path), JSON_LIMIT, "Save the direction again.")
        meta = DirectionMeta.model_validate_json(raw)
    except (ProjectError, OSError, ValueError, ValidationError) as exc:
        raise DirectionError(f"The direction {label} can't be read: {_first_line(exc)}") from exc
    try:
        data = read_limited(
            project.readable(tensor_path), TENSOR_LIMIT, "Save the direction again."
        )
    except (ProjectError, OSError, ValueError) as exc:
        raise DirectionError(
            f"The direction {label} has no tensor file: {_first_line(exc)}"
        ) from exc
    sha = hashlib.sha256(data).hexdigest()
    if sha != meta.sha256:
        raise DirectionError(
            f"The direction {label}'s tensor file has changed since it was saved (its hash no "
            "longer matches its JSON file). Save it again."
        )
    vectors = _vectors(data, label)
    if tuple(vectors.shape) != (meta.dimensions, meta.width):
        raise DirectionError(
            f"The direction {label}'s tensor is {list(vectors.shape)}, but its JSON file says "
            f"{meta.dimensions} × {meta.width}. Save it again."
        )
    if not is_orthonormal(vectors):
        raise DirectionError(
            f"The direction {label}'s rows aren't orthonormal, which Logogram always saves. Save "
            "it again."
        )
    rel = tensor_path.relative_to(project.root).as_posix()
    return Direction(meta, vectors, rel)


def _vectors(data: bytes, label: str) -> torch.Tensor:
    from safetensors.torch import load

    try:
        tensors = load(data)
    except Exception as exc:  # any malformed file
        raise DirectionError(f"The direction {label}'s tensor file isn't safetensors.") from exc
    if set(tensors) != {TENSOR}:
        raise DirectionError(
            f"The direction {label}'s tensor file must hold one tensor called {TENSOR!r}."
        )
    vectors = tensors[TENSOR]
    if vectors.dim() != 2 or not vectors.is_floating_point():
        raise DirectionError(f"The direction {label} must be a 2-D tensor of floats.")
    vectors = vectors.to(torch.float32)
    if not torch.isfinite(vectors).all():
        raise DirectionError(f"The direction {label} holds values that aren't finite numbers.")
    return vectors


def _first_line(exc: BaseException) -> str:
    return (str(exc).strip().splitlines() or [type(exc).__name__])[0]


def read_direction(project: Project, name: str) -> Direction:
    """The direction ``name`` as saved now."""
    check_name(name)
    json_path, tensor_path = _files(project, name)
    if not os.path.lexists(json_path):
        raise DirectionError(
            f"This project has no direction called {name}. Save one first (a probe or a DAS run "
            "saves its directions, and so does a principal component in Representations)."
        )
    return _read_files(project, json_path, tensor_path, name)


def list_directions(project: Project) -> list[dict[str, Any]]:
    """Every direction saved in the project, by name; one that can't be read says why
    (``error``)."""
    folder = directions_dir(project)
    if not project.inside(folder) or not folder.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(folder.glob("*.json"), key=lambda p: p.name.lower()):
        name = path.name[: -len(".json")]
        if not NAME_RE.match(name):
            continue
        try:
            out.append({**read_direction(project, name).to_dict(), "error": None})
        except DirectionError as exc:
            out.append({"name": name, "error": str(exc)})
    return out


def delete_direction(project: Project, name: str) -> None:
    """Delete the direction ``name``. Runs that used it keep their snapshot."""
    check_name(name)
    json_path, tensor_path = _files(project, name)
    if not os.path.lexists(json_path) and not os.path.lexists(tensor_path):
        raise DirectionError(f"This project has no direction called {name}.")
    for path in (json_path, tensor_path):
        if os.path.lexists(path):
            remove_file(project.writable(path))


def rename_direction(project: Project, name: str, new: str) -> Direction:
    """Give the direction ``name`` the name ``new``; refuses a name that is taken."""
    direction = read_direction(project, name)
    check_name(new)
    if new == name:
        return direction
    json_new, tensor_new = _files(project, new)
    if os.path.lexists(json_new) or os.path.lexists(tensor_new):
        raise DirectionError(f"A direction called {new} already exists. Choose another name.")
    meta = direction.meta.model_copy(update={"name": new})
    with atomic_output(project.writable(tensor_new), replace=False) as out:
        out.write(tensor_bytes(direction.vectors))
    written = Direction(meta, direction.vectors, f"{DIRECTIONS_DIR}/{new}.safetensors")
    # The bytes are the same, so the hash is too.
    data = read_limited(project.readable(tensor_new), TENSOR_LIMIT, "Save the direction again.")
    if hashlib.sha256(data).hexdigest() != meta.sha256:  # pragma: no cover - same bytes
        remove_file(project.writable(tensor_new))
        raise DirectionError("The renamed direction couldn't be written exactly. Try again.")
    write_text_atomic(
        project.writable(json_new),
        json.dumps(meta.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        replace=False,
    )
    delete_direction(project, name)
    return written


def pin_direction(project: Project, name: str, sha256: str | None) -> Direction:
    """The direction a run uses: with ``sha256``, the one with exactly that hash (the project's
    snapshot of it, or the saved direction if it still has that hash); without, the direction as
    saved now. Either way the run's copy is kept in ``directions/snapshots/<sha256>``, and the
    direction returned is read from there, so a rerun finds it however the named one changes."""
    check_name(name)
    if sha256 is not None:
        if not SHA256_RE.match(sha256):
            raise DirectionError("A direction's sha256 is 64 lowercase hex digits.")
        snap_json, snap_tensor = _files(project, sha256, snapshot=True)
        if os.path.lexists(snap_json):
            direction = _read_files(project, snap_json, snap_tensor, name)
            if direction.sha256 != sha256:
                raise DirectionError(
                    f"The snapshot of the direction {name} has changed. Restore it from a backup."
                )
            return direction
    direction = read_direction(project, name)
    if sha256 is not None and direction.sha256 != sha256:
        raise DirectionError(
            f"The direction {name} has changed since this spec was written (its hash no longer "
            "matches), and the project has no snapshot of the one it used. Save the original "
            "again, or remove the direction's sha256 to use it as it is now."
        )
    snapshots = directions_dir(project) / SNAPSHOTS
    snapshots.mkdir(parents=True, exist_ok=True)
    snap_json, snap_tensor = _files(project, direction.sha256, snapshot=True)
    if not os.path.lexists(snap_tensor):
        with atomic_output(project.writable(snap_tensor)) as out:
            out.write(tensor_bytes(direction.vectors))
    if not os.path.lexists(snap_json):
        write_text_atomic(
            project.writable(snap_json),
            json.dumps(direction.meta.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        )
    return _read_files(project, snap_json, snap_tensor, name)


def model_of(info: Any) -> dict[str, Any]:
    """A loaded model as a direction's JSON file names it."""
    return {
        "id": info.id,
        "source": info.source,
        "revision": info.revision if info.source == "hub" else None,
        "sha256": info.sha256 if info.source == "local" else None,
    }


def check_fits(direction: Direction, info: Any, what: str) -> list[str]:
    """Refuse a direction whose width isn't the model's ``d_model``; warn (return a sentence) when
    it was found on another model or revision, where it may mean something else."""
    if direction.width != info.d_model:
        raise DirectionError(
            f"The direction {direction.name} is {direction.width} wide, and {what} is "
            f"{info.d_model} wide ({info.id}'s residual stream). Use a direction found on this "
            "model."
        )
    found = direction.meta.model
    same = found.source == info.source and (
        found.sha256 == info.sha256
        if found.source == "local"
        else found.id == info.id and (found.revision in (None, info.revision))
    )
    if same:
        return []
    where = found.id if found.source == "hub" else "a local model"
    revision = f" at {found.revision[:7]}" if found.source == "hub" and found.revision else ""
    return [
        f"The direction {direction.name} was found on {where}{revision}, not on the model this "
        "run uses. It has the right width, but it may not mean the same thing here."
    ]
