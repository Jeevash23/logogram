"""Run a spec end to end and write its experiment folder. The GUI and the CLI both call this."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from logogram import __version__
from logogram.backends.base import ModelBackend
from logogram.datasets import parse_jsonl
from logogram.engine import Cancelled, run_experiment
from logogram.fileio import atomic_output, write_text_atomic
from logogram.project import Project, now_iso
from logogram.prompts import prepare_prompts
from logogram.results import (
    build_summary,
    compute_stats,
    results_table,
    site_payload,
    write_results,
)
from logogram.schema import Manifest
from logogram.spec import ModelRef, SAERef, Spec
from logogram.stats import resample_counts

EventFn = Callable[[str, dict[str, Any]], None]
ModelProvider = Callable[[ModelRef], ModelBackend]
SAEProvider = Callable[[SAERef, ModelBackend], Any]


class RunError(RuntimeError):
    pass


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
    if backend is None:
        return False
    from logogram.backends.transformer_lens import resolve_device

    info = backend.info
    try:
        device = resolve_device(ref.device)
    except Exception:  # noqa: BLE001
        return False
    return (
        info.id == ref.id
        and (ref.revision is None or ref.revision == info.revision)
        and info.dtype == ref.dtype
        and info.device == device
        and info.process_weights == ref.process_weights
    )


def default_provider(on_event: EventFn | None = None) -> ModelProvider:
    def provide(ref: ModelRef) -> ModelBackend:
        from logogram.backends.transformer_lens import load_model

        return load_model(
            ref.id,
            revision=ref.revision,
            dtype=ref.dtype,
            device=ref.device,
            process_weights=ref.process_weights,
            on_progress=(lambda p: on_event("model", p)) if on_event else None,
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
    write_manifest(folder / "manifest.json", manifest)

    def fail(status: str, message: str | None) -> RunOutcome:
        manifest.update(
            {
                "status": status,
                "error": message,
                "finished_at": now_iso(),
                "wall_time_s": round(time.perf_counter() - t0, 3),
            }
        )
        for name in ("results.parquet", "summary.json", "predictions.json"):
            (folder / name).unlink(missing_ok=True)
        write_manifest(folder / "manifest.json", manifest)
        return RunOutcome(run_id, folder, status, None, manifest, backend)

    try:
        dataset_path = project.resolve_dataset(spec.dataset.path)
        # Hash, parse and snapshot the same bytes, even if the source is subsequently edited.
        content = dataset_path.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        if spec.dataset.sha256 and sha != spec.dataset.sha256:
            raise RunError(
                f"{spec.dataset.path} has changed since this spec was written (its hash no longer "
                "matches). Restore the original file, or remove dataset.sha256 to accept the new one."
            )
        records = parse_jsonl(content.decode("utf-8"), source=spec.dataset.path)
        if spec.dataset.limit is not None:
            records = records[: spec.dataset.limit]
        source_path = spec.dataset.path
        snapshots = project.writable(project.datasets_dir / "snapshots")
        snapshots.mkdir(exist_ok=True)
        snapshot = project.writable(snapshots / f"{sha}.jsonl")
        if os.path.lexists(snapshot):
            if project.readable(snapshot).read_bytes() != content:
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
                    "architecture": info.architecture,
                    "process_weights": info.process_weights,
                    "backend": info.backend,
                    "backend_version": info.backend_version,
                },
                "device": {
                    "type": info.device,
                    "name": info.device_name,
                    "cuda": torch.version.cuda if info.device == "cuda" else None,
                    "threads": torch.get_num_threads(),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                },
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

        prompts = prepare_prompts(backend, records, spec.tokenization.prepend_bos)
        # Bootstrap resamples over the prompts a method measures (steering measures held-out
        # prompts only), drawn once per run so every site shares them.
        resamples: dict[int, np.ndarray] = {}

        def counts_for(result: Any) -> np.ndarray:
            n = len(result.prompts)
            if n not in resamples:
                resamples[n] = resample_counts(n, spec.statistics.bootstrap, spec.statistics.seed)
            return resamples[n]

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
                )
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            write_json(folder / "predictions.json", predictions)

        def on_start(sites: list[Any], layout: dict[str, Any]) -> None:
            emit(
                "started",
                {
                    "run_id": run_id,
                    "n_prompts": len(prompts),
                    "model": model_shape,
                    "layout": layout,
                    "sites": [s.to_dict() for s in sites],
                },
            )

        def on_layer(layer: int, indices: list[int], partial: Any) -> None:
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
            cancel=cancel,
            sae=sae,
        )
        stats = compute_stats(spec, result, counts_for(result))
        summary = build_summary(spec, result, stats, run_id, model_shape)
        write_results(folder / "results.parquet", results_table(result, stats))
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
        emit("cancelled", {"run_id": run_id})
        return outcome
    except Exception as exc:  # noqa: BLE001 - recorded in the manifest and reported
        message = str(exc) or type(exc).__name__
        outcome = fail("failed", message)
        emit("failed", {"run_id": run_id, "error": message})
        return outcome
