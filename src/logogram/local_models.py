"""Models from folders on this machine, found by the hash of their files.

A local model is a folder of a model made or downloaded outside Hugging Face's cache (a
fine-tune, say): its config.json, its tokenizer's files and its weights in safetensors (folders
whose weights are only pickles, .bin, .pt or .pth, are refused: opening a pickle can run code).
It is added once, under a name; specs then name it by that name and by its hash
(``model.source: "local"``, ``model.sha256``), never by its folder, so a spec, its runs and their
reports say nothing about where the files are, and the same spec finds the same files on any
machine they were added to.

**The hash.** A folder's hash is the SHA-256 of a text that lists the files loading reads, one
line each, sorted by name: the file's own SHA-256 in lowercase hexadecimal, two spaces, and its
name, with a newline after each line (as ``sha256sum`` prints them). The files are those in the
folder itself (not in a subfolder) that Logogram would download for a model on Hugging Face: every
``*.json``, ``*.txt``, ``*.model``, ``*.tiktoken`` and ``*.jinja`` file (config.json, the
tokenizer's files and its chat template; not hidden files, nor those of ONNX, Flax or TensorFlow
exports, whose names start with onnx, flax or tf_), and the
weights loading reads: model.safetensors, or else the shards model.safetensors.index.json names.
Names are compared in Unicode's composed form (NFC), so the hash is the same on every machine and
file system. On Linux, with the files' names in ``files.txt``:
``sha256sum $(cat files.txt) | LC_ALL=C sort -k2 | sha256sum``.

**The registry.** Added models are listed in ``models.json`` in Logogram's settings folder (beside
the recent projects), with each file's size, modification time and hash. Loading checks that the
files are still the ones hashed: a file whose size or modification time changed is hashed again,
and a folder whose files changed (or that gained or lost one) is refused, naming the files.
"""

from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import json
import os
import threading
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from logogram.backends.base import BackendError, Cancelled

REGISTRY_FILE = "models.json"
REGISTRY_VERSION = 1

# What Logogram downloads for a model on Hugging Face beside its weights (backends.hub).
SUPPORT_PATTERNS = ("*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja")
SKIPPED_PREFIXES = ("onnx", "flax", "tf_")
PICKLES = (".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle")
SINGLE = "model.safetensors"
INDEX = "model.safetensors.index.json"

NAME_MAX = 200
CHUNK = 8 * 1024 * 1024

# (bytes hashed so far, bytes to hash, the file being hashed)
ProgressFn = Callable[[int, int, str], None]


@dataclass
class LocalFile:
    path: str  # its name in the folder (NFC)
    size: int
    mtime_ns: int
    sha256: str


@dataclass
class LocalModel:
    name: str
    folder: str  # absolute; kept in the settings folder only, never in a project
    sha256: str
    added: str
    files: list[LocalFile] = field(default_factory=list)

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)

    def to_dict(self) -> dict[str, Any]:
        """For the app and the command line (on this machine only): with the folder."""
        return {
            "name": self.name,
            "folder": self.folder,
            "sha256": self.sha256,
            "added": self.added,
            "size": self.size,
            "files": len(self.files),
        }


# -- the files and their hash ------------------------------------------------------------------


def _nfc(name: str) -> str:
    return unicodedata.normalize("NFC", name)


def model_files(folder: Path) -> list[str]:
    """The names of the files loading reads in ``folder``, sorted (see the module's docstring).
    Refuses a folder that isn't a model Logogram can load from safetensors, saying why."""
    try:
        if not folder.is_dir():
            raise BackendError(
                f"{folder} isn't a folder. Choose the folder that holds config.json."
            )
        names = {_nfc(p.name): p.name for p in folder.iterdir() if p.is_file()}
    except OSError as exc:
        raise BackendError(f"Logogram can't read {folder}: {exc.strerror or exc}.") from exc
    if "config.json" not in names:
        raise BackendError(
            f"{folder} has no config.json, so it isn't a model folder. Choose the folder that "
            "holds config.json, the tokenizer's files and the safetensors weights."
        )
    if "adapter_config.json" in names:
        raise BackendError(
            "This folder holds an adapter, not a whole model. Merge the adapter into its base "
            "model and save the result with safetensors weights, then add that folder."
        )
    if SINGLE in names:
        weights = [SINGLE]
    elif INDEX in names:
        weights = _shards(folder / names[INDEX], names)
    else:
        if any(name.lower().endswith(PICKLES) for name in names):
            raise BackendError(
                "This folder's weights are only pickles (.bin, .pt or .pth), which can run code "
                "when they are opened, so Logogram doesn't load them. Save the model with "
                "safetensors weights (model.save_pretrained(folder, safe_serialization=True) in "
                "Transformers), then add that folder."
            )
        raise BackendError(
            "This folder has no model.safetensors (or model.safetensors.index.json with its "
            "shards). Save the model with safetensors weights, then add the folder."
        )
    support = [
        name
        for name in names
        if any(fnmatch.fnmatch(name, pattern) for pattern in SUPPORT_PATTERNS)
        and not name.startswith(SKIPPED_PREFIXES)
        and not name.startswith(".")
    ]
    return sorted(set(support) | set(weights))


def _shards(index: Path, names: dict[str, str]) -> list[str]:
    try:
        mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted({_nfc(str(name)) for name in mapping.values()})
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise BackendError(
            f"{INDEX} can't be read ({exc}). Save the model's weights again with safetensors."
        ) from exc
    for shard in shards:
        if "/" in shard or "\\" in shard or not shard.endswith(".safetensors"):
            raise BackendError(
                f"{INDEX} names {shard!r}, which isn't a safetensors file beside it."
            )
        if shard not in names:
            raise BackendError(
                f"{INDEX} names {shard}, which isn't in the folder. Copy every shard of the "
                "weights into it."
            )
    if not shards:
        raise BackendError(f"{INDEX} names no weights. Save the model's weights again.")
    return shards


def folder_hash(files: list[LocalFile]) -> str:
    """The SHA-256 of the lines "<sha256 of file>  <name>", sorted by name (see the module)."""
    lines = "".join(f"{f.sha256}  {f.path}\n" for f in sorted(files, key=lambda f: f.path))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def _hash_file(
    path: Path,
    done: int,
    total: int,
    name: str,
    progress: ProgressFn | None,
    cancel: threading.Event | None,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            digest.update(chunk)
            done += len(chunk)
            if progress is not None:
                progress(done, total, name)
    return digest.hexdigest()


def hash_files(
    folder: Path,
    names: list[str],
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> list[LocalFile]:
    """Each file's size, modification time and SHA-256, with progress in bytes."""
    on_disk = {_nfc(p.name): p for p in folder.iterdir()}
    stats = {name: on_disk[name].stat() for name in names}
    total = sum(st.st_size for st in stats.values())
    done = 0
    out = []
    for name in names:
        st = stats[name]
        sha = _hash_file(on_disk[name], done, total, name, progress, cancel)
        done += st.st_size
        out.append(LocalFile(path=name, size=st.st_size, mtime_ns=st.st_mtime_ns, sha256=sha))
    if progress is not None:
        progress(total, total, "")
    return out


# -- the registry ------------------------------------------------------------------------------


def registry_path() -> Path:
    from logogram import project  # read at call time: tests point the settings folder elsewhere

    return project.config_dir() / REGISTRY_FILE


def list_models() -> list[LocalModel]:
    """The local models added on this machine, by name."""
    path = registry_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as exc:
        raise BackendError(
            f"The list of local models ({path.name} in Logogram's settings folder) can't be read: "
            f"{exc}. Fix or delete the file, then add the models again."
        ) from exc
    out = []
    for item in data.get("models", []) if isinstance(data, dict) else []:
        try:
            files = [LocalFile(**f) for f in item.get("files", [])]
            out.append(
                LocalModel(
                    name=str(item["name"]),
                    folder=str(item["folder"]),
                    sha256=str(item["sha256"]),
                    added=str(item.get("added", "")),
                    files=files,
                )
            )
        except (KeyError, TypeError):
            continue  # a damaged entry: left out, and rewritten without it on the next change
    return sorted(out, key=lambda m: m.name.lower())


def _save(models: list[LocalModel]) -> None:
    from logogram.fileio import write_text_atomic

    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "logogram_models": REGISTRY_VERSION,
        "models": [asdict(m) for m in sorted(models, key=lambda m: m.name.lower())],
    }
    write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def check_name(name: str) -> str:
    """A local model's name, as specs write it in model.id: one line, without slashes."""
    clean = name.strip()
    if not clean:
        raise BackendError("Give the model a name, such as my-fine-tune.")
    if len(clean) > NAME_MAX:
        raise BackendError(f"A model's name is at most {NAME_MAX} characters.")
    if "/" in clean or "\\" in clean:
        raise BackendError(
            "A local model's name has no slashes: those are how Hugging Face models are named. "
            "Use a name such as my-fine-tune."
        )
    if any(ord(c) < 32 or ord(c) == 127 for c in clean):
        raise BackendError("A model's name is one line of text.")
    return clean


def find_model(sha256: str | None = None, name: str | None = None) -> LocalModel | None:
    """The local model with these files (by their hash), or else with this name."""
    models = list_models()
    if sha256 is not None:
        return next((m for m in models if m.sha256 == sha256), None)
    return next((m for m in models if m.name == name), None)


def add_model(
    path: str | Path,
    name: str,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> LocalModel:
    """Hash a model folder and add it to this machine's local models under ``name``."""
    from logogram.project import now_iso

    name = check_name(name)
    folder = Path(path).expanduser().resolve()
    names = model_files(folder)
    existing = list_models()
    if any(m.name == name for m in existing):
        raise BackendError(
            f"A local model is already called {name}. Choose another name, or remove that one "
            f"first (logogram models remove {name})."
        )
    files = hash_files(folder, names, progress, cancel)
    sha = folder_hash(files)
    same = next((m for m in existing if m.sha256 == sha), None)
    if same is not None:
        raise BackendError(
            f"These files are already added, as {same.name}: a spec finds them by their hash, "
            "whatever their name. Use that name."
        )
    model = LocalModel(name=name, folder=str(folder), sha256=sha, added=now_iso(), files=files)
    _save([*list_models(), model])
    return model


def remove_model(name: str) -> LocalModel:
    """Forget a local model. Its folder and files are left as they are."""
    models = list_models()
    model = next((m for m in models if m.name == name), None)
    if model is None:
        known = ", ".join(m.name for m in models) or "none"
        raise BackendError(f"No local model is called {name} (local models: {known}).")
    _save([m for m in models if m.name != name])
    return model


def missing_message(name: str, sha256: str | None) -> str:
    """What a spec that names a local model this machine doesn't have says."""
    start = (sha256 or "")[:12]
    return (
        f"This machine has no local model {name} with the files the spec names (SHA-256 "
        f"{start}…). Add its folder with `logogram models add PATH --name {name}`, or with "
        "Add a model folder in the app's model dialog; a folder whose files hash to that is the "
        "one the spec ran on."
    )


def checked_folder(
    name: str,
    sha256: str | None,
    on_progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> Path:
    """The folder of the local model whose files hash to ``sha256``, once its files are checked
    to be the ones hashed: a file whose size or modification time changed is hashed again, and a
    folder whose files changed is refused, naming them."""
    if sha256 is None:
        raise BackendError("A local model is found by the hash of its files: give model.sha256.")
    model = find_model(sha256=sha256)
    if model is None:
        raise BackendError(missing_message(name, sha256))
    folder = Path(model.folder)
    if not folder.is_dir():
        raise BackendError(
            f"The folder {model.name} was added from is gone ({folder}). Move it back, or add "
            f"the model again where it is now (logogram models add PATH --name {model.name}, "
            f"after logogram models remove {model.name})."
        )
    current = model_files(folder)
    recorded = {f.path: f for f in model.files}
    added = [n for n in current if n not in recorded]
    removed = [n for n in recorded if n not in current]
    on_disk = {_nfc(p.name): p for p in folder.iterdir()}
    stale: list[tuple[str, os.stat_result]] = []
    for name_ in current:
        if name_ in recorded:
            st = on_disk[name_].stat()
            if st.st_size != recorded[name_].size or st.st_mtime_ns != recorded[name_].mtime_ns:
                stale.append((name_, st))
    total = sum(st.st_size for _, st in stale)
    done = 0
    changed = []
    touched = []
    for name_, st in stale:
        sha = _hash_file(on_disk[name_], done, total, name_, on_progress, cancel)
        done += st.st_size
        if sha != recorded[name_].sha256:
            changed.append(name_)
        else:
            touched.append(LocalFile(name_, st.st_size, st.st_mtime_ns, sha))
    if added or removed or changed:
        parts = []
        if changed:
            parts.append(f"changed: {', '.join(changed)}")
        if added:
            parts.append(f"new: {', '.join(added)}")
        if removed:
            parts.append(f"missing: {', '.join(removed)}")
        raise BackendError(
            f"The files of {model.name} aren't the ones it was added with ({'; '.join(parts)}), "
            "so they aren't the model specs name by its hash. Restore them, or add the folder "
            "again under a new name: changed files make a new model."
        )
    if touched:
        # The same bytes, touched: remember the new times, so the next load needn't hash them.
        with contextlib.suppress(OSError, BackendError):
            fresh = {f.path: f for f in touched}
            models = list_models()
            for m in models:
                if m.sha256 == model.sha256:
                    m.files = [fresh.get(f.path, f) for f in m.files]
            _save(models)
    return folder


def folder_of(sha256: str | None) -> str | None:
    """The folder a local model was added from, if this machine has it (to keep its path out of
    what a run writes)."""
    if sha256 is None:
        return None
    with contextlib.suppress(BackendError, OSError):
        model = find_model(sha256=sha256)
        return model.folder if model is not None else None
    return None
