"""Hugging Face Hub access: resolve an exact revision, download with progress, read metadata.

This is the only module that talks to the network, and only when the user starts it (loading a
model or asking for a memory estimate). Weights are loaded only from safetensors files, which
cannot execute code; pickle-based ``.bin`` checkpoints are refused.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import math
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from logogram.backends.base import BackendError, Cancelled

log = logging.getLogger(__name__)

# Top-level files needed to build the model and tokenizer. Weights are added separately.
_SUPPORT_PATTERNS = ("*.json", "*.txt", "*.model", "*.tiktoken")
_SKIP_FILES = {"README.md", ".gitattributes"}


@dataclass
class RepoFiles:
    revision: str
    files: list[tuple[str, int]]  # (path, size in bytes) to download
    n_params: int | None
    gated: bool


ProgressFn = Callable[[int, int, str], None]  # (bytes done, bytes total, current file)
T = TypeVar("T")


def _is_offline_error(exc: Exception) -> bool:
    from huggingface_hub.errors import OfflineModeIsEnabled

    if isinstance(exc, OfflineModeIsEnabled):
        return True
    name = type(exc).__name__
    return (
        any(k in name for k in ("Connect", "Timeout", "NameResolution"))
        or "connect" in str(exc).lower()
    )


def _friendly_hub_error(model_id: str, exc: Exception) -> BackendError:
    from huggingface_hub.errors import (
        GatedRepoError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    if isinstance(exc, GatedRepoError):
        return BackendError(
            f"{model_id} is gated. Accept its license on huggingface.co, then log in on this "
            "machine with `hf auth login` and try again."
        )
    if isinstance(exc, RepositoryNotFoundError):
        return BackendError(
            f"No model called {model_id} was found on Hugging Face. Check the spelling (ids look "
            "like owner/name). Private models need `hf auth login` first."
        )
    if isinstance(exc, RevisionNotFoundError):
        return BackendError(
            f"That revision of {model_id} doesn't exist. Check the commit or branch."
        )
    return BackendError(f"Couldn't reach Hugging Face for {model_id}: {exc}")


def _select_files(siblings: list[tuple[str, int]]) -> list[tuple[str, int]]:
    top_level = [(f, s) for f, s in siblings if "/" not in f and f not in _SKIP_FILES]
    weights = [(f, s) for f, s in top_level if f.endswith(".safetensors")]
    if not weights:
        has_pickle = any(f.endswith((".bin", ".pt", ".pth", ".ckpt")) for f, _ in top_level)
        if has_pickle:
            raise BackendError(
                "This model only publishes pickle weights (.bin). Logogram loads safetensors "
                "weights only, because pickle files can run code when opened. Choose a model "
                "with model.safetensors, or convert the weights yourself."
            )
        raise BackendError("This repository has no safetensors weights at its top level.")
    support = [
        (f, s)
        for f, s in top_level
        if any(fnmatch.fnmatch(f, p) for p in _SUPPORT_PATTERNS)
        and not f.startswith(("onnx", "flax", "tf_"))
    ]
    return sorted(support) + sorted(weights)


def resolve(model_id: str, revision: str | None) -> RepoFiles:
    """Ask the Hub for the exact commit and the files to fetch. Falls back to the local cache."""
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    except Exception as exc:
        if _is_offline_error(exc):
            cached = cached_snapshot(model_id, revision)
            if cached is not None:
                sha, path = cached
                files = _select_files(
                    [(p.name, p.stat().st_size) for p in path.iterdir() if p.is_file()]
                )
                weights = [name for name, _ in files if name.endswith(".safetensors")]
                return RepoFiles(
                    revision=sha,
                    files=files,
                    n_params=count_stored_values(path, weights),
                    gated=False,
                )
            raise BackendError(
                f"Couldn't reach Hugging Face, and {model_id} isn't in your local cache. "
                "Connect to the internet to download it once."
            ) from exc
        raise _friendly_hub_error(model_id, exc) from exc
    siblings = [(s.rfilename, int(s.size or 0)) for s in (info.siblings or [])]
    n_params = None
    stored = getattr(info, "safetensors", None)
    if stored is not None:
        n_params = int(stored.total)
    if not info.sha:
        raise BackendError(
            f"Hugging Face didn't say which commit of {model_id} it describes. Try again, or give "
            "the revision in the model dialog."
        )
    return RepoFiles(
        revision=info.sha,
        files=_select_files(siblings),
        n_params=n_params,
        gated=bool(info.gated),
    )


def count_stored_values(folder: Path, names: list[str]) -> int | None:
    """How many values safetensors files hold, as the Hub reports for a model it knows, read from
    their headers (no tensor is loaded). None if a file can't be read: loading it will say why."""
    from safetensors import safe_open

    total = 0
    try:
        for name in names:
            with safe_open(str(folder / name), framework="pt", device="cpu") as f:
                total += sum(math.prod(f.get_slice(key).get_shape()) for key in f.keys())  # noqa: SIM118
    except Exception:  # noqa: BLE001
        return None
    return total


def cached_snapshot(model_id: str, revision: str | None) -> tuple[str, Path] | None:
    """The (commit, folder) of a cached snapshot containing config.json, if any."""
    found = cached_file(model_id, "config.json", revision)
    return None if found is None else (found[0], found[1].parent)


def cached_file(repo_id: str, filename: str, revision: str | None) -> tuple[str, Path] | None:
    """The commit and local path of ``filename`` in the Hugging Face cache, at ``revision``.

    Without a revision, the cached main branch; failing that, the only cached commit that has the
    file. Logogram downloads exact commits, which records no branch in the cache, so this is how a
    model it downloaded opens offline. Several commits and no branch would be a guess, so that is
    refused with the commits to choose from.
    """
    from huggingface_hub import constants, try_to_load_from_cache
    from huggingface_hub.file_download import repo_folder_name

    depth = len(Path(filename).parts)
    found = try_to_load_from_cache(repo_id, filename, revision=revision or "main")
    if isinstance(found, str):
        return Path(found).parents[depth - 1].name, Path(found)
    snapshots = (
        Path(constants.HF_HUB_CACHE) / repo_folder_name(repo_id=repo_id, repo_type="model")
    ) / "snapshots"
    if revision is not None or not snapshots.is_dir():
        return None
    commits = sorted(d for d in snapshots.iterdir() if (d / filename).is_file())
    if len(commits) > 1:
        names = ", ".join(d.name[:12] for d in commits)
        raise BackendError(
            f"Hugging Face can't be reached, and several revisions of {repo_id} are in your local "
            f"cache ({names}). Enter the one to use as the revision."
        )
    return (commits[0].name, commits[0] / filename) if commits else None


def validate_local_weights(folder: Path) -> None:
    """Refuse pickle-only caches and unsafe or incomplete safetensors shard indexes."""
    if (folder / "adapter_config.json").exists():
        raise BackendError(
            "Adapter checkpoints are not supported. Choose a complete base or merged model published as safetensors, without a separate adapter configuration."
        )
    single = folder / "model.safetensors"
    index = folder / "model.safetensors.index.json"
    if single.is_file():
        return
    if not index.is_file():
        raise BackendError(
            "No model.safetensors weights are available. Download a model with safetensors weights."
        )
    try:
        data = json.loads(index.read_text(encoding="utf-8"))
        mapping = data["weight_map"]
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Empty weight map")
        for name in mapping.values():
            if (
                not isinstance(name, str)
                or "/" in name
                or "\\" in name
                or ":" in name
                or not name.endswith(".safetensors")
            ):
                raise ValueError("Unsafe shard name")
            if not (folder / name).is_file():
                raise ValueError("Missing shard")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise BackendError(
            "The safetensors index is invalid or has missing shards. Download the complete safetensors checkpoint again."
        ) from exc


def _make_bar_class(on_value: Callable[[int], None], size: int) -> type:
    """A silent progress bar that reports how much of one file has arrived.

    Plain HTTP downloads report bytes as they arrive. Xet downloads report network bytes often
    and bytes written to disk rarely; the larger of the two, capped at the file size, gives
    smooth progress that still ends exactly at the file size.
    """
    from tqdm.std import tqdm

    class _Bar(tqdm):  # type: ignore[misc]
        monitor_interval = 0  # no display, so no tqdm monitor thread per file

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            self._written = 0
            self._received = 0

        def _report(self) -> None:
            on_value(min(size, max(self._written, self._received)))

        def update(self, n: float | None = 1) -> bool | None:
            if n:
                self._written += int(n)
                self._report()
            return None

        def update_transfer(self, n: float | None = 1) -> None:
            if n:
                self._received += int(n)
                self._report()

        def set_transfer_postfix_str(self, *args: Any, **kwargs: Any) -> None:
            return None

        def __getattr__(self, name: str) -> Any:
            # Display hooks that newer huggingface_hub versions may call: nothing to display.
            if name.startswith(("set_", "update_")):
                return lambda *args, **kwargs: None
            raise AttributeError(name)

    return _Bar


def download(
    model_id: str,
    repo: RepoFiles,
    progress: ProgressFn | None = None,
    *,
    cancel: threading.Event | None = None,
) -> Path:
    """Download the selected files at the pinned revision and return the snapshot folder.

    Setting ``cancel`` raises ``Cancelled`` within a fraction of a second and stops the transfer;
    a partial file stays in the cache, where the next attempt resumes it.
    """
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    total = sum(size for _, size in repo.files)
    finished = 0  # bytes of files already complete
    folder: Path | None = None
    for filename, size in repo.files:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        cached = try_to_load_from_cache(model_id, filename, revision=repo.revision)
        if isinstance(cached, str):
            finished += size
            folder = Path(cached).parent
            if progress:
                progress(finished, total, filename)
            continue

        def on_value(value: int, _name: str = filename, _base: int = finished) -> None:
            if cancel is not None and cancel.is_set():
                # Raising stops a plain HTTP transfer. Xet transfers are aborted from outside
                # (see _abort_xet); an exception raised in their callback would only be printed.
                if not _called_from_xet():
                    raise Cancelled()
                return
            if progress:
                progress(_base + value, total, _name)

        def fetch(_name: str = filename, _size: int = size, _on: Any = on_value) -> str:
            return hf_hub_download(
                model_id,
                _name,
                revision=repo.revision,
                tqdm_class=_make_bar_class(_on, _size),
            )

        try:
            path = _cancellable(fetch, cancel)
        except Cancelled:
            raise
        except Exception as exc:
            raise _friendly_hub_error(model_id, exc) from exc
        finished += size
        folder = Path(path).parent
        if progress:
            progress(finished, total, filename)
    if folder is None:
        raise BackendError(f"Nothing to download for {model_id}.")
    return folder


def _cancellable(fn: Callable[[], T], cancel: threading.Event | None) -> T:
    """Run ``fn`` on a helper thread, so waiting for it can stop as soon as ``cancel`` is set."""
    if cancel is None:
        return fn()
    box: dict[str, Any] = {}
    done = threading.Event()

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - handed to the waiting thread
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=target, name="logogram-download", daemon=True).start()
    while not done.wait(0.1):
        if cancel.is_set():
            _abort_xet()
            raise Cancelled()
    if "error" in box:
        raise box["error"]
    return box["value"]


def _called_from_xet() -> bool:
    frame = sys._getframe(1)
    while frame is not None:
        if "xet" in frame.f_globals.get("__name__", ""):
            return True
        frame = frame.f_back
    return False


def _abort_xet() -> None:
    """Stop Xet transfers in flight, as huggingface_hub does on Ctrl+C. Without this (if the
    private helper moves), a cancelled Xet download finishes in the background, into the cache."""
    try:
        from huggingface_hub.utils._xet import abort_xet_session

        abort_xet_session()
    except Exception:
        log.debug("couldn't abort Xet transfers", exc_info=True)


@dataclass
class ArchitectureSummary:
    n_layers: int
    n_heads: int
    d_model: int
    d_mlp: int
    d_vocab: int
    n_ctx: int
    architecture: str


def read_architecture(config: dict[str, Any]) -> ArchitectureSummary:
    """Read the shape of a transformer from a Hugging Face config.json (best effort)."""

    def first(*keys: str, default: int | None = None) -> int:
        for key in keys:
            value = config.get(key)
            if isinstance(value, int) and value > 0:
                return value
        text = config.get("text_config")
        if isinstance(text, dict):
            for key in keys:
                value = text.get(key)
                if isinstance(value, int) and value > 0:
                    return value
        if default is None:
            raise BackendError(f"config.json has none of {', '.join(keys)}.")
        return default

    d_model = first("hidden_size", "n_embd", "d_model")
    return ArchitectureSummary(
        n_layers=first("num_hidden_layers", "n_layer", "num_layers"),
        n_heads=first("num_attention_heads", "n_head", "num_heads"),
        d_model=d_model,
        d_mlp=first("intermediate_size", "n_inner", "ffn_dim", default=4 * d_model),
        d_vocab=first("vocab_size"),
        n_ctx=first("max_position_embeddings", "n_positions", "n_ctx", default=2048),
        architecture=(config.get("architectures") or ["unknown"])[0],
    )


def fetch_config(model_id: str, revision: str) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(model_id, "config.json", revision=revision)
    except Exception as exc:
        raise _friendly_hub_error(model_id, exc) from exc
    return json.loads(Path(path).read_text(encoding="utf-8"))
