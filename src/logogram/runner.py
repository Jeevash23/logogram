"""Run a spec end to end and write its experiment folder. The GUI and the CLI both call this."""

from __future__ import annotations

import contextlib
import getpass
import hashlib
import json
import logging
import os
import platform
import random
import re
import socket
import sys
import sysconfig
import tempfile
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from logogram import __version__
from logogram.backends.base import ModelBackend, device_type
from logogram.chat import chat_wrapper
from logogram.datasets import decode_dataset, parse_jsonl, read_dataset_bytes
from logogram.engine import Cancelled, run_experiment
from logogram.fileio import atomic_output, write_text_atomic
from logogram.owner import claim_run, release_run
from logogram.partial import Progress, keep_finished
from logogram.project import TRACEBACK_FILE, Project, now_iso
from logogram.prompts import prepare_prompts
from logogram.results import (
    bootstrap_counts,
    build_summary,
    compute_stats,
    results_table,
    site_payload,
    write_results,
)
from logogram.schema import CONTINUATIONS_FILE, EDGES_FILE, ContinuationsFile, Manifest
from logogram.spec import SINGLE_POSITION_METRICS, EdgeAttribution, ModelRef, SAERef, Spec

EventFn = Callable[[str, dict[str, Any]], None]
ModelProvider = Callable[[ModelRef], ModelBackend]
SAEProvider = Callable[[SAERef, ModelBackend], Any]


class RunError(RuntimeError):
    pass


log = logging.getLogger(__name__)

# A cancelled run's reason, when Ctrl+C (or a notebook's interrupt) stopped it.
INTERRUPTED = (
    "The run was interrupted (Ctrl+C, or a notebook's interrupt) before it finished. Rerun its "
    "saved spec."
)
# A failed run keeps this much of its traceback: the end, where the error is.
TRACEBACK_CHARS = 64 * 1024

# Account and machine names that identify nobody, as scripts/check_privacy.py treats them. Names
# shorter than four letters aren't replaced either: they would match ordinary words.
_GENERIC_NAMES = {
    "root", "user", "users", "admin", "runner", "runneradmin", "ubuntu", "debian", "vagrant",
    "docker", "build", "builder", "jenkins", "localhost", "home", "codespace", "codespaces",
}  # fmt: skip
# Characters that end a path: space, quotes, brackets, and characters no file name has.
_STOP = r"\s\"'<>|*?"
# A folder's name on Windows, where names hold spaces (a first and last name, in C:\Users):
# words with single spaces between them, so the words after a path aren't taken for part of it.
_WINDOWS_NAME = r"[^\\/:\s\"'<>|*?]+(?: [^\\/:\s\"'<>|*?]+)*"
# Anyone's home folder: /home/<name> (also under /var and /usr), /root, /Users/<name> (also as
# WSL and Git Bash show C:\Users), and C:\Users\<name>.
_HOMES = re.compile(
    r"(?<![\w.-])(?:"
    rf"(?:/var|/usr)?/home/[^/\\{_STOP}]+"
    rf"|/root(?=[/\\{_STOP}]|$)"
    rf"|(?:/mnt/[A-Za-z]/[Uu]sers|/[A-Za-z]/Users|/Users)/[^/\\{_STOP}]+"
    rf"|[A-Za-z]:[\\/]+[Uu]sers[\\/]+{_WINDOWS_NAME}"
    r")"
)
# The folders of any other absolute path, on POSIX, on Windows, and on a network share.
_POSIX_FOLDERS = re.compile(rf"(?<![\w.~<>/\\:-])(?:/[^/\\:{_STOP}]+)+/")
_WINDOWS_FOLDERS = re.compile(rf"(?<![\w.~<>/\\-])[A-Za-z]:[\\/]+(?:{_WINDOWS_NAME}[\\/]+)*")
_SHARE_FOLDERS = re.compile(rf"(?<![\w\\:])\\\\{_WINDOWS_NAME}(?:\\{_WINDOWS_NAME})*\\")


def _known_places(roots: tuple[Path | str | None, ...]) -> list[tuple[str, str]]:
    """Folders of this machine that tracebacks and library errors name, with their placeholders:
    the project, the Hugging Face cache, where Python and its libraries are installed,
    Logogram's own source, the temporary folder and the home folder."""
    places: list[tuple[str, str]] = [(str(root), "<project>") for root in roots if root]
    try:
        from huggingface_hub import constants

        places.append((str(constants.HF_HUB_CACHE), "<huggingface cache>"))
    except Exception:  # noqa: BLE001, S110 - no cache to name
        pass
    paths = sysconfig.get_paths()
    libraries = {paths.get("purelib"), paths.get("platlib")}
    libraries |= {p for p in sys.path if Path(p).name in ("site-packages", "dist-packages")}
    places += [(p, "<site-packages>") for p in libraries if p]
    places += [(p, "<python>") for p in (paths.get("stdlib"), paths.get("platstdlib")) if p]
    places.append((str(Path(__file__).parent), "<logogram>"))
    places.append((tempfile.gettempdir(), "<temp>"))
    with contextlib.suppress(RuntimeError):  # no home folder to name
        places.append((str(Path.home()), "~"))
    out: list[tuple[str, str]] = []
    for path, label in places:
        out.append((path, label))
        with contextlib.suppress(OSError, RuntimeError):
            out.append((str(Path(path).resolve()), label))
    return out


def _place_pattern(path: str) -> re.Pattern[str] | None:
    """``path`` as written with either separator (or doubled ones, as in JSON), as a whole name:
    /tmp in /tmp/x, not in /var/tmp or /tmpfiles. Case-insensitive where file names are."""
    parts = re.split(r"[\\/]+", path)
    if len([p for p in parts if p]) < 1 or path in ("/", "\\", "."):
        return None
    body = r"[\\/]+".join(re.escape(p) for p in parts)
    flags = re.IGNORECASE if sys.platform in ("win32", "darwin") else 0
    return re.compile(rf"(?<![\w.-]){body}(?![\w.-])", flags)


def _identity_names() -> list[tuple[re.Pattern[str], str]]:
    """This person's account name and this machine's name, as whole words."""
    candidates: list[tuple[str, str]] = []
    for read, label in (
        (getpass.getuser, "<user>"),
        (lambda: Path.home().name, "<user>"),
        (lambda: socket.gethostname().split(".")[0], "<host>"),
        (lambda: platform.node().split(".")[0], "<host>"),
    ):
        with contextlib.suppress(Exception):
            candidates.append((read(), label))
    out = []
    for name, label in candidates:
        if len(name) >= 4 and name.lower() not in _GENERIC_NAMES:
            word = rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])"
            out.append((re.compile(word, re.IGNORECASE), label))
    return out


def scrub_paths(
    message: str, *roots: Path | str | None, places: tuple[tuple[str, str], ...] = ()
) -> str:
    """``message`` (an error, or a whole traceback) without what identifies this machine or its
    user. Run files are shared, and an error from a library often names a file by its full path,
    which includes the user's name.

    Known folders become placeholders, longest first, so a project inside the home folder reads
    as <project>, not ~/...: the project (``roots``), other folders given as (path, placeholder)
    in ``places`` (a local model's), the Hugging Face cache, <site-packages>, <python> (the
    standard library), <logogram>, <temp> and the home folder (~). Anyone else's home folder
    becomes ~ too. Any other absolute path keeps only its file name, after <path>. Last, the
    account's and the machine's names become <user> and <host> wherever they appear.
    """
    known = _known_places(roots)
    for path, label in places:
        known.append((path, label))
        with contextlib.suppress(OSError, RuntimeError):
            known.append((str(Path(path).resolve()), label))
    for path, label in sorted(set(known), key=lambda p: -len(p[0])):
        pattern = _place_pattern(path)
        if pattern is not None:
            message = pattern.sub(lambda _, label=label: label, message)
    message = _HOMES.sub("~", message)
    for folders in (_WINDOWS_FOLDERS, _SHARE_FOLDERS, _POSIX_FOLDERS):
        message = folders.sub(lambda m: "<path>" + m.group(0)[-1], message)
    for pattern, label in _identity_names():
        message = pattern.sub(label, message)
    return message


def failure_detail(
    exc: BaseException, project_root: Path, places: tuple[tuple[str, str], ...] = ()
) -> str:
    """The traceback of ``exc``, scrubbed for sharing, its end kept when it is very long."""
    text = scrub_paths("".join(traceback.format_exception(exc)), project_root, places=places)
    if len(text) > TRACEBACK_CHARS:
        left_out = len(text) - TRACEBACK_CHARS
        text = f"[The first {left_out:,} characters are left out.]\n" + text[-TRACEBACK_CHARS:]
    return text


def configure_determinism(seed: int = 0) -> None:
    """Make repeated runs on the same machine bit-identical."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def environment_versions() -> dict[str, Any]:
    return {
        "logogram": __version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformer_lens": _pkg_version("transformer-lens"),
        "transformers": _pkg_version("transformers"),
        "numpy": _pkg_version("numpy"),
        "pyarrow": _pkg_version("pyarrow"),
    }


def model_matches(backend: ModelBackend | None, ref: ModelRef) -> bool:
    """Whether the loaded model is the one ``ref`` asks for: the same files (a Hugging Face model
    by id and revision, a local model by the hash of its files, whatever name it was added
    under), dtype, weight processing and device, the device as it resolves here ("cuda" is the
    default GPU, so it matches a model on that GPU however it was asked for)."""
    if backend is None:
        return False
    from logogram.backends.transformer_lens import resolve_device

    info = backend.info
    try:
        device = resolve_device(ref.device)
    except Exception:  # noqa: BLE001
        return False
    if info.source != ref.source:
        return False
    if ref.source == "local":
        same_files = ref.sha256 is not None and info.sha256 == ref.sha256
    else:
        same_files = info.id == ref.id and (ref.revision is None or ref.revision == info.revision)
    return (
        same_files
        and info.dtype == ref.dtype
        and info.device == device
        and info.process_weights == ref.process_weights
    )


def device_provenance(info: Any) -> dict[str, Any]:
    """Where a run ran, as its manifest records it: the device's type (cuda, cpu or mps), which
    of the machine's GPUs (its index, for cuda:N) and its name, and what makes it deterministic."""
    on_gpu = device_type(info.device) == "cuda"
    return {
        "type": device_type(info.device),
        **({"index": int(info.device.split(":", 1)[1])} if ":" in info.device else {}),
        "name": info.device_name,
        "cuda": torch.version.cuda if on_gpu else None,
        "threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _model_places(spec: Spec) -> tuple[tuple[str, str], ...]:
    """The folder of the spec's local model, for scrubbing it from errors."""
    if spec.model.source != "local":
        return ()
    from logogram.local_models import folder_of

    folder = folder_of(spec.model.sha256)
    return ((folder, "<model folder>"),) if folder else ()


def default_provider(on_event: EventFn | None = None) -> ModelProvider:
    def provide(ref: ModelRef) -> ModelBackend:
        from logogram.backends.transformer_lens import load_model

        progress = None
        if on_event is not None:
            emit = on_event
            progress = lambda p: emit("model", p)  # noqa: E731
        return load_model(
            ref.id,
            revision=ref.revision,
            dtype=ref.dtype,
            device=ref.device,
            process_weights=ref.process_weights,
            on_progress=progress,
            source=ref.source,
            sha256=ref.sha256,
        )

    return provide


@dataclass
class RunOutcome:
    run_id: str
    folder: Path
    status: str
    summary: dict[str, Any] | None
    manifest: dict[str, Any]
    backend: ModelBackend | None
    # A failed run's traceback, as saved in its folder (scrubbed of this machine's paths).
    traceback: str | None = None


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    write_json(path, Manifest.model_validate(manifest).model_dump(mode="json"))


def write_json(path: Path, data: Any) -> None:
    """Write atomically, so readers (the history panel) never see a half-written file."""
    write_text_atomic(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def run_spec(
    spec: Spec,
    project: Project,
    *,
    run_id: str | None = None,
    backend: ModelBackend | None = None,
    provider: ModelProvider | None = None,
    on_event: EventFn | None = None,
    cancel: threading.Event | None = None,
    derived_from: dict[str, Any] | None = None,
    sae: Any = None,
    sae_provider: SAEProvider | None = None,
) -> RunOutcome:
    """Execute ``spec`` and write ``experiments/<run_id>/`` in ``project``.

    ``sae`` is an already loaded SAE, used when it is the one the spec names; otherwise
    ``sae_provider`` (by default, a download from Hugging Face) loads it."""
    emit = on_event or (lambda kind, data: None)
    upgraded = spec.upgraded_fields
    configure_determinism(spec.statistics.seed)
    run_id = run_id or project.new_run_id(spec.name)
    folder = project.prepare_run_dir(run_id)
    started_at = now_iso()
    t0 = time.perf_counter()

    manifest: dict[str, Any] = {
        "logogram_manifest": 1,
        "run_id": run_id,
        "status": "running",
        "started_at": started_at,
        "versions": environment_versions(),
        "derived_from": derived_from,
    }
    # The spec as executed: written first so a failed run still records what was attempted.
    write_json(folder / "spec.json", spec.model_dump(mode="json"))
    with contextlib.suppress(OSError):  # an earlier attempt's, in a reused folder
        (folder / TRACEBACK_FILE).unlink(missing_ok=True)
    # What the sweep finishes, so a run that stops early keeps it (see logogram.partial).
    progress = Progress()

    def keep(status: str, again: list[BaseException]) -> dict[str, Any] | None:
        """Keep what finished; None when nothing did, or it can't be kept."""
        try:
            return keep_finished(progress, folder, run_id, status)
        except Exception:  # the run's own error is what it reports
            log.info("The run %s couldn't keep what finished.", run_id, exc_info=True)
        except BaseException as exc:  # noqa: BLE001 - Ctrl+C again: stop once the run is recorded
            again.append(exc)
        return None

    def fail(
        status: str, message: str | None, detail: str | None = None, *, reraise: bool = True
    ) -> RunOutcome:
        manifest.update(
            {
                "status": status,
                "error": message,
                "finished_at": now_iso(),
                "wall_time_s": round(time.perf_counter() - t0, 3),
            }
        )
        again: list[BaseException] = []
        kept = keep(status, again)
        # A run that kept what finished keeps the predictions it measured before the sweep too.
        # Continuations come after the sweep, so a run that stopped never has them.
        names = (
            ("results.parquet", "summary.json", "predictions.json", EDGES_FILE)
            if kept is None
            else ()
        )
        for name in (*names, CONTINUATIONS_FILE):
            (folder / name).unlink(missing_ok=True)
        if detail is not None:
            # Before the manifest says the run failed, so whoever reads that finds why.
            with contextlib.suppress(OSError):
                write_text_atomic(project.writable(folder / TRACEBACK_FILE), detail)
        write_manifest(folder / "manifest.json", manifest)
        if again and reraise:
            raise again[0]
        return RunOutcome(run_id, folder, status, kept, manifest, backend, detail)

    def ended(outcome: RunOutcome) -> dict[str, Any]:
        """What the cancelled or failed event says: the run, and what it kept, if anything."""
        partial = (outcome.summary or {}).get("partial")
        return {"run_id": run_id, **({"partial": partial} if partial else {})}

    # Recorded before the manifest says "running", and removed once it says how the run ended:
    # whoever reads a running manifest can tell whether this process is still going.
    owner = claim_run(project, folder)
    try:
        write_manifest(folder / "manifest.json", manifest)
        dataset_path = project.resolve_dataset(spec.dataset.path)
        # Hash, parse and snapshot the same bytes, even if the source is subsequently edited.
        content = read_dataset_bytes(dataset_path)
        sha = hashlib.sha256(content).hexdigest()
        if spec.dataset.sha256 and sha != spec.dataset.sha256:
            raise RunError(
                f"{spec.dataset.path} has changed since this spec was written (its hash no longer "
                "matches). Restore the original file, or remove dataset.sha256 to accept the new one."
            )
        records = parse_jsonl(decode_dataset(content, spec.dataset.path), source=spec.dataset.path)
        if spec.dataset.limit is not None:
            records = records[: spec.dataset.limit]
        source_path = spec.dataset.path
        snapshots = project.writable(project.datasets_dir / "snapshots")
        snapshots.mkdir(exist_ok=True)
        snapshot = project.writable(snapshots / f"{sha}.jsonl")
        if os.path.lexists(snapshot):
            if read_dataset_bytes(project.readable(snapshot)) != content:
                raise RunError(
                    "The pinned dataset snapshot has changed. Restore it from a backup before rerunning."
                )
        else:
            with atomic_output(snapshot) as output:
                output.write(content)
        spec = spec.model_copy(
            update={
                "dataset": spec.dataset.model_copy(
                    update={
                        "path": snapshot.relative_to(project.root).as_posix(),
                        "sha256": sha,
                    }
                )
            }
        )
        write_json(folder / "spec.json", spec.model_dump(mode="json"))
        if cancel is not None and cancel.is_set():
            raise Cancelled()

        if not model_matches(backend, spec.model):
            emit("status", {"message": f"Loading {spec.model.id}"})
            backend = (provider or default_provider(on_event))(spec.model)
        assert backend is not None
        info = backend.info

        # Pin exactly what ran, so `logogram run` on this file reproduces it.
        spec = spec.model_copy(
            update={
                "model": spec.model.model_copy(update={"revision": info.revision}),
                "dataset": spec.dataset.model_copy(update={"sha256": sha}),
            }
        )
        write_json(folder / "spec.json", spec.model_dump(mode="json"))
        manifest.update(
            {
                "model": {
                    "id": info.id,
                    "revision": info.revision,
                    # A local model: by the hash of its files, never by its folder.
                    **(
                        {"source": "local", "sha256": info.sha256} if info.source == "local" else {}
                    ),
                    "architecture": info.architecture,
                    "process_weights": info.process_weights,
                    "backend": info.backend,
                    "backend_version": info.backend_version,
                    "chat": info.extra.get("chat"),
                },
                "device": device_provenance(info),
                "dtype": info.dtype,
                "dataset": {
                    "path": spec.dataset.path,
                    "sha256": sha,
                    "n": len(records),
                    "source_path": source_path,
                },
            }
        )
        write_manifest(folder / "manifest.json", manifest)

        if spec.sae is not None:
            from logogram.sae import load_sae, sae_matches

            if not sae_matches(sae, spec.sae):
                emit("status", {"message": f"Loading the SAE {spec.sae.repo}"})
                sae = (
                    sae_provider(spec.sae, backend)
                    if sae_provider is not None
                    else load_sae(spec.sae, info.device, cancel=cancel)
                )
            spec = spec.model_copy(
                update={"sae": spec.sae.model_copy(update={"revision": sae.revision})}
            )
            write_json(folder / "spec.json", spec.model_dump(mode="json"))
            manifest["sae"] = {
                "repo": sae.repo,
                "path": sae.path,
                "revision": sae.revision,
                "format": sae.params.format,
                "site": sae.site,
                "layer": sae.layer,
                "d_sae": sae.d_sae,
            }
            write_manifest(folder / "manifest.json", manifest)

        # The model's own chat template, when the spec wraps prompts in it: the same text around
        # every prompt, recorded so the methods can say what the model read.
        chat = chat_wrapper(backend, spec.tokenization.chat_template)
        if chat is not None:
            manifest["chat_template"] = {"kind": chat.kind, "sha256": chat.sha256}
            write_manifest(folder / "manifest.json", manifest)
        prompts = prepare_prompts(
            backend,
            records,
            spec.tokenization.prepend_bos,
            continuations=spec.metric.kind not in SINGLE_POSITION_METRICS,
            chat=chat,
        )
        # Bootstrap resamples over the prompts a method measures (steering measures held-out
        # prompts only), drawn once per run so every site shares them.
        resamples: dict[tuple[int, ...], np.ndarray] = {}

        def counts_for(result: Any) -> np.ndarray:
            key = tuple(p.index for p in result.prompts)
            if key not in resamples:
                resamples[key] = bootstrap_counts(spec, result)
            return resamples[key]

        model_shape = {
            "id": info.id,
            "n_layers": info.n_layers,
            "n_heads": info.n_heads,
            "d_model": info.d_model,
            "d_head": info.d_head,
            "d_mlp": info.d_mlp,
            "site_kinds": list(info.site_kinds),
            "block_structure": info.extra.get("block_structure"),
            "normalization": info.extra.get("normalization"),
            "activation": info.extra.get("activation"),
        }

        if spec.predictions is not None:
            from logogram.analysis import prediction_report

            emit("status", {"message": "Measuring the configured layer predictions"})
            with backend.lock:
                predictions = prediction_report(
                    backend,
                    records,
                    index=spec.predictions.prompt_index,
                    settings=spec.predictions,
                    prepend_bos=spec.tokenization.prepend_bos,
                    batch_size=spec.execution.batch_size,
                    chat_template=spec.tokenization.chat_template,
                )
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            write_json(folder / "predictions.json", predictions)

        notes = (
            [
                "This spec is version 1, which left some choices out; they took version 1's "
                f"values: {'; '.join(upgraded)}. The saved spec states every value."
            ]
            if upgraded
            else []
        )
        progress.prepare(
            spec,
            model_shape,
            counts_for,
            notes,
            len(prompts),
            chat_template=chat.to_dict() if chat is not None else None,
        )

        def on_start(sites: list[Any], layout: dict[str, Any]) -> None:
            progress.started(sites, layout)
            emit(
                "started",
                {
                    "run_id": run_id,
                    "n_prompts": len(prompts),
                    "model": model_shape,
                    "layout": layout,
                    # An edge attribution run's edges (tens of thousands) arrive with its
                    # summary, which lists the strongest.
                    "sites": []
                    if isinstance(spec.experiment, EdgeAttribution)
                    else [s.to_dict() for s in sites],
                },
            )

        def on_layer(layer: int, indices: list[int], partial: Any) -> None:
            progress.finished_sites(indices, partial)
            # A finished site's statistics depend only on its own rows, so they are final.
            stats = compute_stats(spec, partial, counts_for(partial), indices)
            emit(
                "layer",
                {"run_id": run_id, "layer": layer, "sites": site_payload(partial, stats, indices)},
            )

        def on_progress(done: int, total: int, layer: int) -> None:
            emit(
                "progress",
                {
                    "run_id": run_id,
                    "done": done,
                    "total": total,
                    "layer": layer,
                    "memory": backend.memory_in_use() if backend else None,
                    "elapsed_s": round(time.perf_counter() - t0, 2),
                },
            )

        result = run_experiment(
            spec,
            backend,
            prompts,
            on_progress=on_progress,
            on_layer=on_layer,
            on_start=on_start,
            on_prompts=progress.finished_prompts,
            cancel=cancel,
            sae=sae,
        )
        progress.complete = True
        result.warnings[:0] = notes
        # What a steered model went on to say: kept beside the results, not in the summary.
        continuations = result.extra.pop("continuations", None)
        stats = compute_stats(spec, result, counts_for(result))
        summary = build_summary(
            spec,
            result,
            stats,
            run_id,
            model_shape,
            chat_template=chat.to_dict() if chat is not None else None,
        )
        write_results(folder / "results.parquet", results_table(result, stats))
        if "edges" in result.extra:
            from logogram.edges import write_edge_table

            write_edge_table(folder / EDGES_FILE, result, stats)
        if continuations is not None:
            record = ContinuationsFile.model_validate({"run_id": run_id, **continuations})
            write_json(folder / CONTINUATIONS_FILE, record.model_dump(mode="json"))
        write_json(folder / "summary.json", summary)
        manifest.update(
            {
                "status": "finished",
                "finished_at": now_iso(),
                "wall_time_s": round(time.perf_counter() - t0, 3),
            }
        )
        write_manifest(folder / "manifest.json", manifest)
        emit("finished", {"run_id": run_id})
        return RunOutcome(run_id, folder, "finished", summary, manifest, backend)
    except Cancelled:
        outcome = fail("cancelled", None)
        emit("cancelled", ended(outcome))
        return outcome
    except Exception as exc:  # recorded in the manifest and reported
        from logogram.backends.transformer_lens import is_out_of_memory, release_memory

        # Shown with --verbose (the server's log); the run's folder keeps a scrubbed copy.
        log.info("The run %s failed.", run_id, exc_info=exc)
        # A local model's folder is this machine's business: never in what the run writes.
        places = _model_places(spec)
        detail = failure_detail(exc, project.root, places)
        if is_out_of_memory(exc):
            device = device_type(backend.info.device) if backend is not None else "the device"
            where = {"cuda": "GPU memory", "mps": "unified memory"}.get(device, "memory")
            message = (
                f"The run ran out of {where} at a batch of {spec.execution.batch_size} rows. Lower "
                "execution.batch_size in the spec (it is part of the spec because it can change "
                "results in the last digits), use fewer prompts, or a smaller model."
            )
            del exc
            release_memory()
        else:
            message = str(exc) or type(exc).__name__
        message = scrub_paths(message, project.root, places=places)
        outcome = fail("failed", message, detail)
        emit("failed", {**ended(outcome), "error": message})
        return outcome
    except BaseException:
        # Ctrl+C, a notebook's interrupt, or the process exiting: say so, keeping what finished,
        # then let it stop.
        outcome = fail("cancelled", INTERRUPTED, reraise=False)
        with contextlib.suppress(Exception):
            emit("cancelled", {**ended(outcome), "error": INTERRUPTED})
        raise
    finally:
        release_run(project, folder, owner)
