"""Projects are plain folders::

    my-project/
      project.json
      datasets/*.jsonl
      experiments/<id>/spec.json
      experiments/<id>/results.parquet
      experiments/<id>/summary.json
      experiments/<id>/manifest.json
      .gitignore

Nothing in a project folder refers to the machine it was made on: paths are project-relative.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shutil
import uuid
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

import platformdirs
from pydantic import BaseModel, ConfigDict, Field

from logogram.datasets import DatasetError, file_sha256, load_dataset
from logogram.fileio import write_text_atomic
from logogram.schema import Manifest, RunListing, Summary
from logogram.spec import Spec, describe_experiment

PROJECT_FILE = "project.json"
GITIGNORE = "# Logogram caches\n.logogram/\n__pycache__/\n"
APP_NAME = "logogram"


class ProjectError(ValueError):
    pass


class ProjectMeta(BaseModel):
    model_config = ConfigDict(extra="allow")

    logogram_project: int = 1
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    created: str = ""


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:max_len].rstrip("-")) or "run"


class Project:
    def __init__(self, root: Path, meta: ProjectMeta):
        self.root = root
        self.meta = meta
        self.session_id = uuid.uuid4().hex
        self.interrupted_runs: set[str] = set()

    # -- locations ---------------------------------------------------------------------------

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def experiments_dir(self) -> Path:
        return self.root / "experiments"

    def run_dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
            raise ProjectError(f"Not a valid run id: {run_id!r}")
        folder = self.experiments_dir / run_id
        if not self.inside(folder):
            raise ProjectError("The run folder links outside the project.")
        return folder

    def resolve_dataset(self, rel: str) -> Path:
        return self.readable(self.root / rel)

    def readable(self, path: Path) -> Path:
        """Resolve a regular project file before reading it, including shared run metadata."""
        if not self.inside(path):
            raise ProjectError(f"{path.name} links to a file outside the project.")
        resolved = path.resolve()
        if not resolved.is_file():
            raise ProjectError(f"{path.name} isn't a regular file inside the project.")
        return resolved

    def read_json(self, path: Path, *, optional: bool = False) -> dict[str, Any] | None:
        if optional and not os.path.lexists(path):
            return None
        data = json.loads(self.readable(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ProjectError(f"{path.name} must contain a JSON object.")
        return data

    def inside(self, path: Path) -> bool:
        """Whether ``path``, after following symlinks, is inside the project folder."""
        try:
            return path.resolve().is_relative_to(self.root.resolve())
        except (OSError, RuntimeError):  # a symlink loop
            return False

    def writable(self, path: Path) -> Path:
        """Return ``path`` if writing it stays inside the project.

        Projects are shared through git and archives, which keep symlinks, so a folder in a
        project could point anywhere. Files are only written into folders that really are part of
        the project (and through ``fileio``, which never follows a symlink at the file itself).
        """
        if not self.inside(path.parent):
            where = path.parent.relative_to(self.root).as_posix()
            raise ProjectError(
                f"{where} links to a folder outside the project, so Logogram won't write there."
            )
        return path

    def prepare_run_dir(self, run_id: str, *, new: bool = False) -> Path:
        """Create (or reuse) ``experiments/<run_id>`` and check that it is inside the project."""
        folder = self.run_dir(run_id)
        self.experiments_dir.mkdir(exist_ok=True)
        self.writable(folder)  # experiments/ itself is part of the project
        folder.mkdir(exist_ok=not new)
        return self.writable(folder / "spec.json").parent  # and so is the run's folder

    def new_run_dir(self, name: str) -> tuple[str, Path]:
        """Create a fresh ``experiments/<id>`` folder for a run or draft called ``name``."""
        for _ in range(100):
            run_id = self.new_run_id(name)
            try:
                return run_id, self.prepare_run_dir(run_id, new=True)
            except FileExistsError:
                continue  # taken in the meantime
        raise ProjectError("Couldn't create a folder for the run.")

    def dataset_file(self, name: str) -> Path:
        """Where a dataset called ``name`` is written (``name`` is checked by the caller)."""
        self.datasets_dir.mkdir(exist_ok=True)
        return self.writable(self.datasets_dir / name)

    # -- open / create -----------------------------------------------------------------------

    @classmethod
    def open(cls, path: str | Path) -> Project:
        root = Path(path).expanduser().resolve()
        meta_path = root / PROJECT_FILE
        if not meta_path.is_file():
            raise ProjectError(
                f"{root.name or root} isn't a Logogram project (there's no project.json). "
                "Create a project there instead."
            )
        try:
            cls(root, ProjectMeta(name="Project")).readable(meta_path)
            meta = ProjectMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ProjectError(f"project.json can't be read: {exc}") from exc
        project = cls(root, meta)
        project.ensure_layout()
        return project

    @classmethod
    def create(cls, parent: str | Path, name: str, folder: str | None = None) -> Project:
        name = name.strip()
        if not name:
            raise ProjectError("Give the project a name.")
        parent_path = Path(parent).expanduser().resolve()
        root = parent_path / (folder or slugify(name, 60))
        if (root / PROJECT_FILE).exists():
            raise ProjectError(f"A project already exists in {root.name}. Open it instead.")
        try:
            if root.exists() and any(root.iterdir()):
                raise ProjectError(f"The folder {root.name} already exists and isn't empty.")
            root.mkdir(parents=True, exist_ok=True)
            meta = ProjectMeta(name=name, created=now_iso())
            write_text_atomic(root / PROJECT_FILE, json.dumps(meta.model_dump(), indent=2) + "\n")
            project = cls(root, meta)
            project.ensure_layout()
        except OSError as exc:
            raise ProjectError(
                f"Logogram can't create a project in {parent_path}: {exc.strerror or exc}. "
                "Choose a folder you can write to."
            ) from exc
        return project

    @staticmethod
    def find_root(start: Path) -> Path | None:
        start = start.resolve()
        for candidate in [start, *start.parents]:
            if (candidate / PROJECT_FILE).is_file():
                return candidate
        return None

    def ensure_layout(self) -> None:
        """Add missing folders and the .gitignore. A read-only project still opens."""
        with contextlib.suppress(OSError):
            self.datasets_dir.mkdir(exist_ok=True)
            self.experiments_dir.mkdir(exist_ok=True)
            gitignore = self.root / ".gitignore"
            if not os.path.lexists(gitignore):  # not even a dangling symlink
                write_text_atomic(gitignore, GITIGNORE)

    # -- contents ----------------------------------------------------------------------------

    def list_datasets(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not self.inside(self.datasets_dir) or not self.datasets_dir.is_dir():
            return out
        for path in sorted(self.datasets_dir.glob("*.jsonl")):
            entry: dict[str, Any] = {
                "name": path.name,
                "path": f"datasets/{path.name}",
                "modified": _mtime_iso(path),
            }
            try:
                records = load_dataset(self.readable(path))
                entry["n"] = len(records)
                entry["has_positions"] = all(r.positions for r in records)
                entry["sha256"] = file_sha256(path)
            except (DatasetError, ProjectError, OSError) as exc:
                entry["error"] = str(exc)
            out.append(entry)
        return out

    def new_run_id(self, name: str) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{stamp}-{slugify(name)}"
        candidate, i = base, 2
        while (self.experiments_dir / candidate).exists():
            candidate = f"{base}-{i}"
            i += 1
        return candidate

    def list_runs(self) -> list[RunListing]:
        runs = []
        if not self.inside(self.experiments_dir) or not self.experiments_dir.is_dir():
            return runs
        for folder in self.experiments_dir.iterdir():
            if self.inside(folder) and folder.is_dir():
                listing = self.run_listing(folder.name)
                if listing is not None:
                    runs.append(listing)
        runs.sort(key=lambda r: (r.created or "", r.id), reverse=True)
        return runs

    def run_listing(self, run_id: str) -> RunListing | None:
        try:
            folder = self.run_dir(run_id)
            spec = Spec.from_path(self.readable(folder / "spec.json"))
        except Exception:  # noqa: BLE001 - unreadable specs are skipped in history
            return None
        error = None
        try:
            manifest = self.read_json(folder / "manifest.json", optional=True) or {}
            if manifest:
                manifest = Manifest.model_validate(manifest).model_dump(mode="json")
            summary_meta = self.read_json(folder / "summary.json", optional=True) or {}
            if summary_meta:
                summary_meta = Summary.model_validate(summary_meta).model_dump(mode="json")
            n = summary_meta.get("n_prompts")
            layout = summary_meta.get("layout") or {}
            if (n is not None and (type(n) is not int or n < 1)) or not isinstance(layout, dict):
                raise ValueError("Invalid summary metadata")
            if layout.get("kind") is not None and not isinstance(layout["kind"], str):
                raise ValueError("Invalid layout kind")
        except (ValueError, OSError):
            manifest, summary_meta = {}, {}
            error = "Run metadata is unreadable or invalid. Restore its manifest and summary from a backup, or rerun the spec."
        status = manifest.get("status") or "draft"
        if run_id in self.interrupted_runs:
            status = "failed"
            error = "The server stopped before this run finished. Rerun its saved spec."
        if error or (
            status == "finished"
            and (
                not self.inside(folder / "results.parquet")
                or not (folder / "results.parquet").is_file()
            )
        ):
            status = "failed"
            error = error or "Run results are missing or outside the project. Rerun the spec."
        return RunListing(
            id=run_id,
            name=spec.name,
            description=describe_experiment(spec),
            status=status,
            created=manifest.get("started_at") or _mtime_iso(folder / "spec.json"),
            finished=manifest.get("finished_at"),
            wall_time_s=manifest.get("wall_time_s"),
            n_prompts=summary_meta.get("n_prompts"),
            layout_kind=(summary_meta.get("layout") or {}).get("kind"),
            model_id=spec.model.id,
            dataset=spec.dataset.path,
            experiment=spec.experiment.model_dump(),
            scope=spec.scope.model_dump(),
            derived_from=manifest.get("derived_from"),
            error=error or manifest.get("error"),
            profile=layer_profile(summary_meta) if status == "finished" else None,
        )

    def last_modified(self) -> str:
        latest = (self.root / PROJECT_FILE).stat().st_mtime
        for sub in (self.datasets_dir, self.experiments_dir):
            try:
                latest = max(latest, sub.stat().st_mtime)
                for child in sub.iterdir() if self.inside(sub) else ():
                    with contextlib.suppress(OSError):  # e.g. a dangling symlink
                        latest = max(latest, child.lstat().st_mtime)
            except OSError:
                continue
        return datetime.fromtimestamp(latest, UTC).replace(microsecond=0).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.meta.name,
            "description": self.meta.description,
            "path": str(self.root),
            "session_id": self.session_id,
            "datasets": self.list_datasets(),
        }


def layer_profile(summary: dict[str, Any]) -> list[float] | None:
    """Per layer, the measured site with the largest absolute normalized effect, with its sign.

    The app writes each run's logogram from this: layers clockwise, swelling where effects are
    strong. Layers without a measured value are 0.
    """
    sites = summary.get("sites") or []
    layers = [s.get("layer") for s in sites if isinstance(s.get("layer"), int)]
    if not layers:
        return None
    model = summary.get("model") or {}
    n_layers = model.get("n_layers") if isinstance(model.get("n_layers"), int) else max(layers) + 1
    out = [0.0] * max(n_layers, max(layers) + 1)
    for site in sites:
        layer = site.get("layer")
        value = (site.get("effect") or {}).get("mean")
        if not isinstance(layer, int) or layer < 0 or not isinstance(value, int | float):
            continue
        if math.isfinite(value) and abs(value) > abs(out[layer]):
            out[layer] = float(value)
    return [round(v, 4) for v in out]


def _mtime_iso(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.lstat().st_mtime, UTC).replace(microsecond=0).isoformat()
    except OSError:
        return None


def _read_json(path: Path, keys: tuple[str, ...] | None = None) -> dict[str, Any] | None:
    try:
        if not path.is_file():  # also skips devices and pipes, which could block or never end
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if keys is not None and isinstance(data, dict):
        return {k: data.get(k) for k in keys}
    return data if isinstance(data, dict) else None


# -- recent projects and the example -----------------------------------------------------------


def config_dir() -> Path:
    return platformdirs.user_config_path(APP_NAME)


def recent_file() -> Path:
    return config_dir() / "recent.json"


def load_recent() -> list[dict[str, Any]]:
    data = _read_json(recent_file())
    items = data.get("projects", []) if data else []
    out = []
    for item in items:
        path = Path(item.get("path", ""))
        try:
            if not (path / PROJECT_FILE).is_file():
                continue
            project = Project.open(path)
            out.append(
                {
                    "path": str(path),
                    "name": project.meta.name,
                    "opened": item.get("opened"),
                    "modified": project.last_modified(),
                }
            )
        except (ProjectError, OSError):
            continue  # moved, deleted or unreadable: leave it out of the list
    return out


def remember_recent(project: Project) -> None:
    items = [i for i in load_recent() if i["path"] != str(project.root)]
    items.insert(0, {"path": str(project.root), "opened": now_iso()})
    recent_file().parent.mkdir(parents=True, exist_ok=True)
    payload = {"projects": [{"path": i["path"], "opened": i.get("opened")} for i in items[:12]]}
    write_text_atomic(recent_file(), json.dumps(payload, indent=2) + "\n")


def default_projects_parent() -> Path:
    return platformdirs.user_documents_path() / "Logogram"


def example_source() -> Path:
    return Path(str(resources.files("logogram") / "examples" / "ioi-gpt2"))


def open_example(parent: Path | None = None) -> Project:
    """Copy the bundled example into a writable folder (once) and open it."""
    target = (parent or default_projects_parent()) / "ioi-example"
    if (target / PROJECT_FILE).is_file():
        return Project.open(target)
    try:
        if target.exists() and any(target.iterdir()):
            raise ProjectError(
                f"{target} exists and isn't a Logogram project. Move it aside first."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(example_source(), target, dirs_exist_ok=True)
    except OSError as exc:
        raise ProjectError(
            f"Logogram can't copy the example to {target}: {exc.strerror or exc}."
        ) from exc
    return Project.open(target)
