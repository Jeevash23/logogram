"""The local web server: JSON API, WebSocket event stream and the bundled web app."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import threading
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import anyio
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, ValidationError, model_validator

from logogram import __version__
from logogram.backends.base import BackendError
from logogram.datasets import (
    DatasetError,
    PromptRecord,
    check_dataset_name,
    file_sha256,
    load_dataset,
    parse_jsonl,
    write_dataset,
)
from logogram.project import (
    Project,
    ProjectError,
    default_projects_parent,
    load_recent,
    open_example,
)
from logogram.research import (
    Notebook,
    NoteConflict,
    NoteInput,
    ResearchNote,
    delete_note,
    read_notebook,
    save_note,
)
from logogram.schema import Manifest, RunListing, Summary
from logogram.server import models as M
from logogram.server.security import SecurityConfig, SecurityMiddleware
from logogram.server.state import AppState, Conflict, EventHub, Missing, request_project
from logogram.spec import (
    NAME_MAX,
    SINGLE_POSITION_METRICS,
    Metric,
    ModelRef,
    PredictionSettings,
    Spec,
    describe_intervention,
)
from logogram.system import SystemReport

log = logging.getLogger(__name__)

# Starting points that load through TransformerLens and fit an ordinary machine. Every model is
# checked again when it loads (backends/transformer_lens.check_model); "tested" ones have run every
# method end to end with the bundled example.
MODEL_PRESETS = [
    {
        "id": "openai-community/gpt2",
        "label": "GPT-2 small",
        "detail": "124M · the model the IOI example was made for",
        "tested": True,
        "gated": False,
    },
    {
        "id": "openai-community/gpt2-medium",
        "label": "GPT-2 medium",
        "detail": "355M · 24 layers",
        "tested": False,
        "gated": False,
    },
    {
        "id": "EleutherAI/pythia-160m",
        "label": "Pythia 160M",
        "detail": "Attention and MLP in parallel · training checkpoints as revisions, like step3000",
        "tested": False,
        "gated": False,
    },
    {
        "id": "EleutherAI/pythia-410m",
        "label": "Pythia 410M",
        "detail": "Attention and MLP in parallel · training checkpoints as revisions",
        "tested": False,
        "gated": False,
    },
    {
        "id": "HuggingFaceTB/SmolLM2-135M",
        "label": "SmolLM2 135M",
        "detail": "Llama architecture, small enough for a CPU",
        "tested": False,
        "gated": False,
    },
    {
        "id": "Qwen/Qwen2.5-0.5B",
        "label": "Qwen2.5 0.5B",
        "detail": "No beginning-of-sequence token",
        "tested": True,
        "gated": False,
    },
    {
        "id": "allenai/OLMo-2-0425-1B",
        "label": "OLMo 2 1B",
        "detail": "Normalization after each sublayer",
        "tested": False,
        "gated": False,
    },
    {
        "id": "meta-llama/Llama-3.2-1B",
        "label": "Llama 3.2 1B",
        "detail": "Gated: request access on Hugging Face first",
        "tested": False,
        "gated": True,
    },
    {
        "id": "google/gemma-2-2b",
        "label": "Gemma 2 2B",
        "detail": "Gated: request access on Hugging Face first · soft-capped logits",
        "tested": False,
        "gated": True,
    },
]
MODEL_SUGGESTIONS = [
    *(p["id"] for p in MODEL_PRESETS),
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-1b",
    "HuggingFaceTB/SmolLM2-360M",
    "Qwen/Qwen3-0.6B-Base",
    "google/gemma-3-270m",
    "google/gemma-3-1b-pt",
    "microsoft/phi-1_5",
]


# Published SAEs for the suggested models, by model id. Each is checked against the loaded model when
# it loads and measured on the project's prompts before its features are trusted.
SAE_SUGGESTIONS: dict[str, list[dict[str, str]]] = {
    "openai-community/gpt2": [
        {
            "repo": "jbloom/GPT2-Small-SAEs-Reformatted",
            "detail": "Residual stream before each layer · 24,576 features",
        },
        {
            "repo": "jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs",
            "detail": "OpenAI's TopK SAEs · residual stream after each layer · 32,768 features",
        },
    ],
    "EleutherAI/pythia-70m": [
        {
            "repo": "EleutherAI/sae-pythia-70m-32k",
            "detail": "TopK SAEs · each layer's output, attention and MLP · 32,768 features",
        }
    ],
    "EleutherAI/pythia-70m-deduped": [
        {
            "repo": "EleutherAI/sae-pythia-70m-deduped-32k",
            "detail": "TopK SAEs · each layer's output, attention and MLP · 32,768 features",
        }
    ],
    "HuggingFaceTB/SmolLM2-135M": [
        {"repo": "EleutherAI/sae-smollm2-135m-64x", "detail": "TopK SAEs · MLP outputs"}
    ],
    "meta-llama/Llama-3.2-1B": [
        {
            "repo": "EleutherAI/sae-llama-3.2-1b-131k",
            "detail": "TopK SAEs · MLP outputs · 131,072 features",
        }
    ],
    "google/gemma-3-270m": [
        {
            "repo": "google/gemma-scope-2-270m-pt",
            "detail": "Gemma Scope 2 · SAEs on each layer's output, attention heads and MLP, and "
            "transcoders · 16k to 262k features · load the model with weight processing off",
        }
    ],
    "google/gemma-3-270m-it": [
        {
            "repo": "google/gemma-scope-2-270m-it",
            "detail": "Gemma Scope 2 · SAEs and transcoders · load the model with weight "
            "processing off",
        }
    ],
    "google/gemma-3-1b-pt": [
        {
            "repo": "google/gemma-scope-2-1b-pt",
            "detail": "Gemma Scope 2 · SAEs on each layer's output, attention heads and MLP, and "
            "transcoders · 16k to 262k features · load the model with weight processing off",
        }
    ],
    "google/gemma-3-1b-it": [
        {
            "repo": "google/gemma-scope-2-1b-it",
            "detail": "Gemma Scope 2 · SAEs and transcoders · load the model with weight "
            "processing off",
        }
    ],
}


def web_dist() -> Path:
    return Path(str(resources.files("logogram") / "web_dist"))


# -- request bodies ----------------------------------------------------------------------------


class CreateProject(BaseModel):
    name: str
    parent: str | None = None


class OpenProject(BaseModel):
    path: str


class IOIRequest(BaseModel):
    name: str = "ioi"
    n: int = Field(default=32, ge=1, le=100_000)
    seed: int = 0
    templates: list[str] | None = None
    patterns: list[Literal["ABBA", "BABA"]] = ["ABBA", "BABA"]
    corruption: Literal["flip", "abc"] = "flip"
    overwrite: bool = False


class ImportRequest(BaseModel):
    name: str
    text: str
    overwrite: bool = False


class PairRequest(BaseModel):
    name: str = "pair"
    clean: str
    corrupt: str
    answer: str
    distractor: str
    overwrite: bool = False


class AnalysisRequest(BaseModel):
    """An interactive analysis. The choices that change its numbers are stated, as in a spec."""

    model: ModelRef | None = None
    dataset_sha256: str | None = None
    prepend_bos: bool
    limit: int | None = Field(default=None, ge=1, le=1_000_000)
    batch_size: int = Field(ge=1, le=4096)


class TokenizeRequest(AnalysisRequest):
    dataset: str | None = None
    index: int = 0
    record: PromptRecord | None = None
    # The metric the prompts are for: it decides whether answers of several tokens can be read.
    metric: Metric


class EstimateRequest(BaseModel):
    id: str
    revision: str | None = None
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    process_weights: bool = True


class BaselineRequest(AnalysisRequest):
    dataset: str
    metric: Metric


class AttentionRequest(AnalysisRequest):
    dataset: str
    index: int = 0
    layer: int
    head: int
    which: Literal["clean", "corrupt"] = "clean"


class PredictionRequest(AnalysisRequest):
    dataset: str
    index: int = Field(ge=0)
    settings: PredictionSettings


class SAELoadRequest(BaseModel):
    repo: str = Field(min_length=1)
    path: str = ""
    revision: str | None = None


class SAEAnalysisRequest(AnalysisRequest):
    dataset: str


class TokenFeaturesRequest(SAEAnalysisRequest):
    index: int = Field(default=0, ge=0)
    which: Literal["clean", "corrupt"] = "clean"
    top_k: int = Field(default=8, ge=1, le=32)


class FeatureRequest(SAEAnalysisRequest):
    index: int = Field(default=0, ge=0)
    which: Literal["clean", "corrupt"] = "clean"
    feature: int = Field(ge=0)


class NoteUpdate(NoteInput):
    revision: int = Field(ge=1)


class RunRequest(BaseModel):
    spec: dict[str, Any]
    draft_id: str | None = None  # run a saved, not-yet-run experiment in its own folder


class RobustnessRequest(BaseModel):
    """One methodological choice changed: the experiment, or the precision the model runs in."""

    experiment: dict[str, Any] | None = None
    dtype: Literal["float32", "float16", "bfloat16"] | None = None

    @model_validator(mode="after")
    def _one_change(self) -> RobustnessRequest:
        if (self.experiment is None) == (self.dtype is None):
            raise ValueError("Change either the experiment or the dtype.")
        return self


class VerifyRequest(BaseModel):
    top: int = Field(default=10, ge=1, le=200)


class SettingsRequest(BaseModel):
    system_check_seen: bool | None = None
    theme: Literal["light", "dark", "system"] | None = None
    # Whether Logogram may ask PyPI about new versions once a day. Off until the user says yes.
    update_check: bool | None = None


# -- app factory -------------------------------------------------------------------------------


def create_app(
    security: SecurityConfig,
    *,
    state: AppState | None = None,
    initial_project: Path | None = None,
    serve_web: bool = True,
    check_updates: bool = False,
) -> FastAPI:
    hub = state.hub if state else EventHub()
    state = state or AppState(hub)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        hub.bind(asyncio.get_running_loop())
        if check_updates:
            state.start_update_checks()
        yield
        state.stop()

    app = FastAPI(
        title="Logogram",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.logogram = state
    if initial_project is not None:
        state.open_project(Project.open(initial_project))

    @app.exception_handler(Missing)
    async def _missing(_: Request, exc: Missing) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=409)

    @app.exception_handler(Conflict)
    @app.exception_handler(NoteConflict)
    async def _conflict(_: Request, exc: Conflict) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=409)

    for exc_type in (ProjectError, DatasetError, BackendError, ValueError):

        @app.exception_handler(exc_type)
        async def _bad(_: Request, exc: Exception) -> JSONResponse:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.exception_handler(OSError)
    async def _os_error(_: Request, exc: OSError) -> JSONResponse:
        name = Path(exc.filename).name if isinstance(exc.filename, str) else None
        what = exc.strerror or type(exc).__name__
        return JSONResponse({"error": f"{what}: {name}" if name else what}, status_code=400)

    @app.exception_handler(ValidationError)
    async def _invalid(_: Request, exc: ValidationError) -> JSONResponse:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        return JSONResponse({"error": f"{where}: {first['msg']}"}, status_code=400)

    # -- state -------------------------------------------------------------------------------

    @app.get("/api/state", response_model=M.ServerState)
    def get_state() -> dict[str, Any]:
        settings = state.settings()
        return {
            "version": __version__,
            "project": state.project.to_dict() if state.project else None,
            "model": state.model_payload(),
            "job": state.job.to_dict() if state.job else None,
            "first_run": not settings.get("system_check_seen", False),
            "theme": settings.get("theme", "light"),
            "projects_parent": str(default_projects_parent()),
            "update": state.update_status(),
            "sae": state.sae_payload(),
        }

    @app.post("/api/settings", response_model=M.Settings)
    def post_settings(body: SettingsRequest) -> dict[str, Any]:
        values = {k: v for k, v in body.model_dump().items() if v is not None}
        state.update_settings(**values)
        if values.get("update_check") is True:
            # Allowed just now: check right away rather than at the next hourly look.
            threading.Thread(
                target=state.check_updates, name="logogram-update", daemon=True
            ).start()
        elif "update_check" in values:
            state.hub.publish("update", state.update_status())
        return state.settings()

    @app.get("/api/update", response_model=M.UpdateStatus)
    def get_update() -> dict[str, Any]:
        """What is known about newer versions. Never contacts the network."""
        return state.update_status()

    @app.post("/api/update/check", response_model=M.UpdateStatus)
    def check_update() -> dict[str, Any]:
        """Ask PyPI now: the user pressed Check now."""
        return state.check_updates()

    @app.get("/api/system", response_model=SystemReport)
    def get_system() -> dict[str, Any]:
        from logogram.system import system_report

        return system_report().to_dict()

    # -- files and projects -----------------------------------------------------------------

    def _is_project(folder: Path) -> bool:
        try:
            return (folder / "project.json").is_file()
        except OSError:  # e.g. a folder we may list but not enter
            return False

    @app.get("/api/fs", response_model=M.FolderListing)
    def list_dirs(path: str | None = None) -> dict[str, Any]:
        base = Path(path).expanduser() if path else Path.home()
        base = base.resolve()
        try:
            if not base.is_dir():
                raise ProjectError(f"{base} isn't a folder.")
            children = sorted(base.iterdir(), key=lambda p: p.name.lower())
        except OSError as exc:
            raise ProjectError(f"Logogram isn't allowed to read {base}.") from exc
        entries = []
        for child in children:
            try:
                if child.name.startswith(".") or not child.is_dir():
                    continue
            except OSError:
                continue
            entries.append(
                {"name": child.name, "path": str(child), "is_project": _is_project(child)}
            )
        return {
            "path": str(base),
            "parent": str(base.parent) if base.parent != base else None,
            "is_project": _is_project(base),
            "entries": entries,
        }

    @app.get("/api/projects/recent", response_model=list[M.RecentProject])
    def recent() -> list[dict[str, Any]]:
        return load_recent()

    def _enter(project: Project) -> dict[str, Any]:
        info = project.to_dict()  # read it first: a project that can't be listed isn't opened
        state.open_project(project)
        return info

    @app.post("/api/projects", response_model=M.ProjectInfo)
    def create_project(body: CreateProject) -> dict[str, Any]:
        parent = Path(body.parent).expanduser() if body.parent else default_projects_parent()
        return _enter(Project.create(parent, body.name))

    @app.post("/api/projects/open", response_model=M.ProjectInfo)
    def open_project(body: OpenProject) -> dict[str, Any]:
        return _enter(Project.open(body.path))

    @app.post("/api/projects/example", response_model=M.ProjectInfo)
    def example() -> dict[str, Any]:
        return _enter(open_example())

    @app.post("/api/projects/close", response_model=M.OkOut)
    def close_project() -> dict[str, Any]:
        state.close_project()
        return {"ok": True}

    @app.get("/api/project", response_model=M.ProjectInfo)
    def get_project() -> dict[str, Any]:
        return state.require_project().to_dict()

    @app.get("/api/research", response_model=Notebook)
    def research_notes() -> Notebook:
        return read_notebook(state.require_project())

    @app.post("/api/research", response_model=ResearchNote)
    def new_research_note(body: NoteInput) -> ResearchNote:
        with state._job_lock:
            project = state.require_project()
            note = save_note(project, body)
        state.hub.publish("research.updated", {"project_session": project.session_id})
        return note

    @app.put("/api/research/{note_id}", response_model=ResearchNote)
    def update_research_note(note_id: str, body: NoteUpdate) -> ResearchNote:
        with state._job_lock:
            project = state.require_project()
            value = NoteInput.model_validate(body.model_dump(exclude={"revision"}))
            note = save_note(project, value, note_id=note_id, revision=body.revision)
        state.hub.publish("research.updated", {"project_session": project.session_id})
        return note

    @app.delete("/api/research/{note_id}", response_model=M.OkOut)
    def remove_research_note(note_id: str, revision: int) -> dict[str, bool]:
        with state._job_lock:
            project = state.require_project()
            delete_note(project, note_id, revision)
        state.hub.publish("research.updated", {"project_session": project.session_id})
        return {"ok": True}

    # -- datasets ----------------------------------------------------------------------------

    def _dataset_path(project: Project, name: str) -> Path:
        """A dataset to read: it must really be inside the project (not linked from elsewhere)."""
        return project.readable(project.datasets_dir / check_dataset_name(Path(name).name))

    def _new_dataset_path(project: Project, name: str) -> Path:
        return project.dataset_file(check_dataset_name(Path(name).name))

    def _write_new(path: Path, records: list[PromptRecord], overwrite: bool) -> dict[str, Any]:
        if os.path.lexists(path) and not overwrite:
            raise DatasetError(
                f"{path.name} already exists. Choose another name, or replace it explicitly."
            )
        write_dataset(path, records)
        return {"name": path.name, "path": f"datasets/{path.name}", "n": len(records)}

    def _records(
        dataset: str, limit: int | None = None, sha256: str | None = None
    ) -> list[PromptRecord]:
        project = state.require_project()
        path = project.resolve_dataset(dataset)
        content = path.read_bytes()
        if sha256 is not None and hashlib.sha256(content).hexdigest() != sha256:
            raise Conflict(
                "The dataset changed since these settings were saved. Restore the original dataset or choose the updated dataset for a new experiment."
            )
        records = parse_jsonl(content.decode("utf-8"), source=dataset)
        return records[:limit] if limit else records

    @app.get("/api/datasets/{name}", response_model=M.DatasetDetail)
    @app.get("/api/dataset", response_model=M.DatasetDetail)
    def get_dataset(
        name: str = "", path: str | None = None, prepend_bos: bool = True, limit: int | None = None
    ) -> dict[str, Any]:
        project = state.require_project()
        file = project.resolve_dataset(path) if path else _dataset_path(project, name)
        records = load_dataset(file)
        if limit is not None:
            if limit < 1:
                raise ValueError("Prompt limit must be positive.")
            records = records[:limit]
        out: dict[str, Any] = {
            "name": file.name,
            "path": file.relative_to(project.root).as_posix(),
            "sha256": file_sha256(file),
            "n": len(records),
            "records": [r.model_dump(mode="json", exclude_none=True) for r in records],
        }
        if state.backend is not None:
            from logogram.analysis import prepare_with_issues

            prepared, issues = prepare_with_issues(state.backend, records, prepend_bos)
            out["issues"] = [i.to_dict() for i in issues]
            out["lengths"] = sorted({p.length for p in prepared})
        return out

    @app.get("/api/ioi/templates", response_model=list[M.IOITemplateOut])
    def ioi_templates() -> list[dict[str, Any]]:
        from logogram.ioi import TEMPLATES

        return [{"id": t.id, "text": t.text, "default": t.default} for t in TEMPLATES]

    @app.post("/api/datasets/ioi", response_model=M.DatasetCreated)
    def make_ioi(body: IOIRequest) -> dict[str, Any]:
        from logogram.ioi import generate_ioi

        project = state.require_project()
        backend = state.backend
        single = (lambda w: backend.single_token_id(w) is not None) if backend else None
        records = generate_ioi(
            body.n,
            seed=body.seed,
            templates=body.templates,
            patterns=body.patterns,
            corruption=body.corruption,
            single_token=single,
        )
        return _write_new(_new_dataset_path(project, body.name), records, body.overwrite)

    @app.post("/api/datasets/import", response_model=M.DatasetCreated)
    def import_dataset(body: ImportRequest) -> dict[str, Any]:
        project = state.require_project()
        path = _new_dataset_path(project, body.name)
        records = parse_jsonl(body.text, source=path.name)
        return _write_new(path, records, body.overwrite)

    @app.post("/api/datasets/pair", response_model=M.DatasetCreated)
    def make_pair(body: PairRequest) -> dict[str, Any]:
        project = state.require_project()
        record = PromptRecord(
            clean=body.clean, corrupt=body.corrupt, answer=body.answer, distractor=body.distractor
        )
        return _write_new(_new_dataset_path(project, body.name), [record], body.overwrite)

    @app.post("/api/tokenize", response_model=M.TokenStrip)
    def tokenize(body: TokenizeRequest) -> dict[str, Any]:
        from logogram.analysis import tokenize_pair

        backend = _analysis_backend(body)
        if body.record is not None:
            record = body.record
        elif body.dataset:
            records = _records(body.dataset, body.limit, body.dataset_sha256)
            if not 0 <= body.index < len(records):
                raise ValueError(f"There is no prompt {body.index}.")
            record = records[body.index]
        else:
            raise ValueError("Send a dataset and index, or a prompt pair.")
        return tokenize_pair(
            backend,
            record,
            body.prepend_bos,
            continuations=body.metric.kind not in SINGLE_POSITION_METRICS,
        )

    # -- models ------------------------------------------------------------------------------

    @app.get("/api/models/presets", response_model=M.Presets)
    def presets() -> dict[str, Any]:
        return {"presets": MODEL_PRESETS, "suggestions": MODEL_SUGGESTIONS}

    @app.post("/api/models/estimate", response_model=M.EstimateOut)
    def estimate(body: EstimateRequest) -> dict[str, Any]:
        from logogram.backends import hub
        from logogram.backends.transformer_lens import architecture_support, resolve_device
        from logogram.system import estimate_memory

        device = resolve_device(body.device)
        repo = hub.resolve(body.id, body.revision)
        config = hub.fetch_config(body.id, repo.revision)
        arch = hub.read_architecture(config)
        support_note = architecture_support(arch.architecture)
        n_params = repo.n_params
        if n_params is None:
            weights = sum(s for f, s in repo.files if f.endswith(".safetensors"))
            n_params = weights // 2  # stored size is a fallback; most checkpoints are 16-bit
        download = 0
        from huggingface_hub import try_to_load_from_cache

        for filename, size in repo.files:
            if not isinstance(
                try_to_load_from_cache(body.id, filename, revision=repo.revision), str
            ):
                download += size
        loaded = 0
        if state.backend is not None and state.backend.info.device == device:
            loaded = state.backend.memory_in_use() or 0
        est = estimate_memory(
            n_params=n_params,
            n_layers=arch.n_layers,
            n_heads=arch.n_heads,
            d_model=arch.d_model,
            d_mlp=arch.d_mlp,
            d_vocab=arch.d_vocab,
            dtype=body.dtype,
            device=device,
            loaded_bytes=loaded,
            process_weights=body.process_weights,
        )
        return {
            "id": body.id,
            "revision": repo.revision,
            "architecture": arch.__dict__,
            "download_bytes": download,
            "total_bytes": sum(s for _, s in repo.files),
            "gated": repo.gated,
            "supported": support_note is None,
            "support_note": support_note,
            "estimate": est.to_dict(),
        }

    @app.post("/api/models/load", response_model=M.JobInfo)
    def load(body: ModelRef) -> dict[str, Any]:
        return state.load_model_job(body).to_dict()

    @app.post("/api/models/unload", response_model=M.ModelStatus)
    def unload() -> dict[str, Any]:
        state.unload_model()
        return state.model_payload()

    # -- analyses ----------------------------------------------------------------------------

    def _analysis_backend(body: AnalysisRequest):  # type: ignore[no-untyped-def]
        from logogram.runner import model_matches

        backend = state.require_backend()
        if body.model is not None and not model_matches(backend, body.model):
            raise Conflict(
                "The loaded model doesn't match these experiment settings. Load the experiment's model, revision, dtype and weight processing before inspecting it."
            )
        return backend

    @app.post("/api/baseline", response_model=M.BaselineReport)
    def baseline(body: BaselineRequest) -> dict[str, Any]:
        from logogram.analysis import baseline_report

        backend = _analysis_backend(body)
        records = _records(body.dataset, body.limit, body.dataset_sha256)
        with backend.lock:  # unloading waits until the analysis is done
            return baseline_report(
                backend,
                records,
                prepend_bos=body.prepend_bos,
                batch_size=body.batch_size,
                metric=body.metric,
            )

    @app.post("/api/attention", response_model=M.AttentionData)
    def attention(body: AttentionRequest) -> dict[str, Any]:
        from logogram.analysis import attention_report

        backend = _analysis_backend(body)
        records = _records(body.dataset, body.limit, body.dataset_sha256)
        with backend.lock:
            return attention_report(
                backend,
                records,
                index=body.index,
                layer=body.layer,
                head=body.head,
                which=body.which,
                prepend_bos=body.prepend_bos,
                batch_size=body.batch_size,
            )

    @app.post("/api/predictions", response_model=M.PredictionReport)
    def predictions(body: PredictionRequest) -> dict[str, Any]:
        from logogram.analysis import prediction_report

        backend = _analysis_backend(body)
        records = _records(body.dataset, body.limit, body.dataset_sha256)
        with backend.lock:
            return prediction_report(
                backend,
                records,
                index=body.index,
                settings=body.settings,
                prepend_bos=body.prepend_bos,
                batch_size=body.batch_size,
            )

    # -- sparse autoencoders -----------------------------------------------------------------

    @app.get("/api/sae/suggestions")
    def sae_suggestions(model: str) -> list[dict[str, str]]:
        return SAE_SUGGESTIONS.get(model, [])

    @app.get("/api/sae/folders")
    def sae_folders(repo: str, revision: str | None = None) -> dict[str, Any]:
        """The SAEs in a Hugging Face repository (one per folder), at its exact revision."""
        from logogram.backends.saes import list_saes

        sha, folders = list_saes(repo, revision)
        if not folders:
            raise ValueError(
                f"{repo} has no SAE that Logogram can read: it looks for cfg.json with "
                "sae_weights.safetensors or sae.safetensors."
            )
        return {"repo": repo, "revision": sha, "folders": folders}

    @app.post("/api/sae/load", response_model=M.JobInfo)
    def sae_load(body: SAELoadRequest) -> dict[str, Any]:
        from logogram.spec import SAERef

        state.require_backend()
        ref = SAERef(repo=body.repo, path=body.path, revision=body.revision)
        return state.load_sae_job(ref).to_dict()

    @app.post("/api/sae/unload")
    def sae_unload() -> dict[str, Any]:
        state.unload_sae()
        return state.sae_payload()

    def _sae() -> Any:
        if state.sae is None:
            raise Missing("Load an SAE first.")
        return state.sae

    @app.post("/api/sae/fit")
    def sae_fit(body: SAEAnalysisRequest) -> dict[str, Any]:
        from logogram.analysis import sae_fit_report

        backend, sae = _analysis_backend(body), _sae()
        records = _records(body.dataset, body.limit, body.dataset_sha256)
        with backend.lock:
            fit = sae_fit_report(
                backend, sae, records, prepend_bos=body.prepend_bos, batch_size=body.batch_size
            )
        state.hub.publish("sae", state.sae_payload())
        return fit

    @app.post("/api/sae/tokens")
    def sae_tokens(body: TokenFeaturesRequest) -> dict[str, Any]:
        from logogram.analysis import token_features_report

        backend, sae = _analysis_backend(body), _sae()
        records = _records(body.dataset, body.limit, body.dataset_sha256)
        with backend.lock:
            return token_features_report(
                backend,
                sae,
                records,
                index=body.index,
                which=body.which,
                prepend_bos=body.prepend_bos,
                top_k=body.top_k,
            )

    @app.post("/api/sae/feature")
    def sae_feature(body: FeatureRequest) -> dict[str, Any]:
        from logogram.analysis import feature_report

        backend, sae = _analysis_backend(body), _sae()
        records = _records(body.dataset, body.limit, body.dataset_sha256)
        with backend.lock:
            return feature_report(
                backend,
                sae,
                records,
                feature=body.feature,
                index=body.index,
                which=body.which,
                prepend_bos=body.prepend_bos,
                batch_size=body.batch_size,
            )

    # -- runs --------------------------------------------------------------------------------

    def _read_run(project: Project, run_id: str) -> dict[str, Any]:
        folder = project.run_dir(run_id)
        if not (folder / "spec.json").is_file():
            raise Missing(f"There is no run {run_id}.")
        out: dict[str, Any] = {"id": run_id}
        for name in ("spec", "summary", "manifest", "predictions"):
            path = folder / f"{name}.json"
            try:
                data = project.read_json(path, optional=name != "spec")
                schema = {
                    "spec": Spec,
                    "summary": Summary,
                    "manifest": Manifest,
                    "predictions": M.PredictionReport,
                }[name]
                out[name] = (
                    schema.model_validate(data).model_dump(mode="json")
                    if data is not None
                    else None
                )
            except ProjectError:
                raise
            except ValueError:
                if name == "spec":
                    raise
                out[name] = None
        listing = project.run_listing(run_id)
        out["listing"] = listing.to_dict() if listing else None
        out["folder"] = f"experiments/{run_id}"
        return out

    @app.get("/api/runs", response_model=list[RunListing])
    def runs() -> list[dict[str, Any]]:
        project = state.require_project()
        out = [r.to_dict() for r in project.list_runs()]
        job = state.job
        if job is not None and job.status == "running" and job.run_id:
            for r in out:
                # The job outlives its run by a moment (it writes the manifest, then reports);
                # a run whose manifest is written has ended, whatever the job says.
                if r["id"] == job.run_id and r["status"] == "draft":
                    r["status"] = "running"
        return out

    @app.get("/api/runs/{run_id}", response_model=M.RunDetail)
    def get_run(run_id: str) -> dict[str, Any]:
        return _read_run(state.require_project(), run_id)

    @app.get("/api/runs/{run_id}/export.csv")
    def export_run(run_id: str) -> StreamingResponse:
        from logogram.exports import results_csv

        project = state.require_project()
        listing = project.run_listing(run_id)
        if listing is None or listing.status != "finished":
            raise Conflict("Only finished runs can be exported. Select a finished run first.")
        path = project.readable(project.run_dir(run_id) / "results.parquet")
        return StreamingResponse(
            results_csv(path),
            media_type="text/csv",
            headers={
                "content-disposition": f'attachment; filename="{run_id}.csv"',
            },
        )

    @app.post("/api/runs", response_model=M.StartedRun)
    def start_run(body: RunRequest) -> dict[str, Any]:
        spec = Spec.model_validate(body.spec)
        state.require_project()
        job = state.run_spec_job(spec, draft_id=body.draft_id)
        return {"run_id": job.run_id, "job": job.to_dict()}

    @app.post("/api/drafts", response_model=M.DraftSaved)
    def save_draft(body: RunRequest) -> dict[str, Any]:
        """Save a spec without running it, for `logogram run` or later. With ``draft_id``, update
        that saved experiment, as long as it hasn't run."""
        spec = Spec.model_validate(body.spec)
        run_id = state.save_draft(spec, body.draft_id)
        return {"run_id": run_id, "path": f"experiments/{run_id}/spec.json"}

    @app.post("/api/runs/{run_id}/rerun", response_model=M.StartedRun)
    def rerun(run_id: str) -> dict[str, Any]:
        project = state.require_project()
        spec = Spec.from_path(project.readable(project.run_dir(run_id) / "spec.json"))
        job = state.run_spec_job(spec, derived_from={"run": run_id, "kind": "rerun"})
        return {"run_id": job.run_id, "job": job.to_dict()}

    @app.post("/api/runs/{run_id}/robustness", response_model=M.StartedRun)
    def robustness(run_id: str, body: RobustnessRequest) -> dict[str, Any]:
        project = state.require_project()
        original = Spec.from_path(project.readable(project.run_dir(run_id) / "spec.json"))
        data = original.model_dump(mode="json")
        if body.dtype is not None:
            if body.dtype == original.model.dtype:
                raise ValueError(f"This run already ran in {body.dtype}.")
            # The same run in another precision: what rounding changed.
            data["model"]["dtype"] = body.dtype
        else:
            data["experiment"] = body.experiment
        suffix = " · robustness"
        data["name"] = original.name[: NAME_MAX - len(suffix)].rstrip() + suffix
        variant = Spec.model_validate(data)
        change = (
            f"In {body.dtype} instead of {original.model.dtype}"
            if body.dtype is not None
            else describe_intervention(variant.experiment)
        )
        job = state.run_spec_job(
            variant, derived_from={"run": run_id, "kind": "robustness", "change": change}
        )
        return {"run_id": job.run_id, "job": job.to_dict()}

    @app.post("/api/runs/{run_id}/verify", response_model=M.StartedRun)
    def verify(run_id: str, body: VerifyRequest) -> dict[str, Any]:
        """Patch, for real, the sites an attribution patching run estimated to matter most."""
        from logogram.verify import verification_spec

        project = state.require_project()
        run = _read_run(project, run_id)
        if not run["summary"]:
            raise ValueError("Only a finished run can be verified.")
        spec = verification_spec(Spec.model_validate(run["spec"]), run["summary"], body.top)
        n = len(spec.scope.sites)  # type: ignore[union-attr]
        job = state.run_spec_job(
            spec,
            derived_from={
                "run": run_id,
                "kind": "verification",
                "change": f"the {n} strongest estimated site{'s' if n != 1 else ''}, patched",
            },
        )
        return {"run_id": job.run_id, "job": job.to_dict()}

    @app.get("/api/runs/{run_id}/derived", response_model=list[RunListing])
    def derived(run_id: str) -> list[dict[str, Any]]:
        project = state.require_project()
        return [
            r.to_dict()
            for r in project.list_runs()
            if r.derived_from and r.derived_from.run == run_id
        ]

    @app.get("/api/runs/{run_id}/sites/{site}", response_model=M.SiteDetail)
    def site_detail(run_id: str, site: int) -> dict[str, Any]:
        from logogram.runs import site_detail as detail

        return detail(state.require_project(), run_id, site)

    @app.get("/api/compare", response_model=M.Comparison)
    def compare(a: str, b: str) -> dict[str, Any]:
        from logogram.compare import compare_summaries

        project = state.require_project()
        ra, rb = _read_run(project, a), _read_run(project, b)
        if not ra["summary"] or not rb["summary"]:
            raise ValueError("Both runs need to have finished.")
        return compare_summaries(
            ra["summary"],
            rb["summary"],
            Spec.model_validate(ra["spec"]),
            Spec.model_validate(rb["spec"]),
        )

    @app.post("/api/jobs/cancel", response_model=M.CancelOut)
    async def cancel() -> dict[str, Any]:
        # On the event loop, not the thread pool: queued analyses can't hold up a cancel.
        job = state.cancel_job()
        return {"job": job.to_dict() if job else None}

    # -- events ------------------------------------------------------------------------------

    @app.websocket("/ws")
    async def events(ws: WebSocket) -> None:
        await ws.accept()
        # Subscribe before saying hello: the tab refetches state when the stream opens, and every
        # event after this point (and the current run so far) is queued for it.
        queue = hub.subscribe()

        async def send_events() -> None:
            while True:
                message = await queue.get()
                if message is None:  # fell too far behind: close, so the tab reconnects
                    await ws.close(code=1013)
                    return
                await ws.send_text(message)

        async def until_closed() -> None:
            # Returns when the tab closes or the server shuts down, so nothing lingers.
            while (await ws.receive())["type"] != "websocket.disconnect":
                pass

        async def first_to_finish(job: Any, scope: anyio.CancelScope) -> None:
            await job()
            scope.cancel()

        try:
            await ws.send_json({"type": "hello", "version": __version__})
            # A task group, not bare asyncio tasks, so a server-side cancel (shutdown, or the
            # test client closing) stays inside this handler's scope.
            async with anyio.create_task_group() as group:
                group.start_soon(first_to_finish, send_events, group.cancel_scope)
                group.start_soon(first_to_finish, until_closed, group.cancel_scope)
        except* (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.unsubscribe(queue)

    # -- web app -----------------------------------------------------------------------------

    if serve_web:
        dist = web_dist()

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> Response:
            if path.startswith("api/"):
                raise HTTPException(status_code=404)
            target = (dist / path).resolve()
            if path and target.is_file() and target.is_relative_to(dist.resolve()):
                headers = (
                    {"cache-control": "public, max-age=31536000, immutable"}
                    if path.startswith("assets/")
                    else {}
                )
                return FileResponse(target, headers=headers)
            index = dist / "index.html"
            if not index.is_file():
                return Response(
                    "The web app hasn't been built. Run `npm run build` in web/.",
                    media_type="text/plain",
                    status_code=503,
                )
            return FileResponse(index, headers={"cache-control": "no-store"})

    @app.middleware("http")
    async def headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        project = state.project
        expected = request.headers.get("x-logogram-project")
        if (
            expected is None
            and _project_scoped(request.url.path)
            and "authorization" not in request.headers
        ):
            # A browser tab (authenticated by its cookie) must say which project it is showing,
            # so a stale tab can't read or change the project another tab opened since.
            return JSONResponse(
                {"error": "This request didn't say which project it is for. Reload the page."},
                status_code=409,
            )
        # Browser requests carry an ephemeral project identity. Capture the object as well:
        # an open/close concurrent with dispatch must never retarget a pending operation.
        if expected is not None and expected != (project.session_id if project else "none"):
            return JSONResponse(
                {
                    "error": "The project changed in another tab. Wait for this tab to refresh, then try again."
                },
                status_code=409,
            )
        token = request_project.set((project,))
        try:
            response = await call_next(request)
        finally:
            request_project.reset(token)
        response.headers.setdefault("x-content-type-options", "nosniff")
        response.headers.setdefault("referrer-policy", "no-referrer")
        response.headers.setdefault("x-frame-options", "DENY")
        host = request.headers.get("host", "")
        response.headers.setdefault(
            "content-security-policy",
            f"default-src 'self'; connect-src 'self' ws://{host}; img-src 'self' data: blob:; "
            "style-src 'self' 'unsafe-inline'; font-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'",
        )
        return response

    return SecuredApp(app, security)  # type: ignore[return-value]


# Routes that read or change the open project. A browser tab sends the project it shows with each
# of them (``x-logogram-project``); scripts using the bearer token address the open project.
PROJECT_SCOPED = (
    "/api/project",
    "/api/research",
    "/api/dataset",
    "/api/datasets/",
    "/api/tokenize",
    "/api/baseline",
    "/api/attention",
    "/api/predictions",
    "/api/sae/fit",
    "/api/sae/tokens",
    "/api/sae/feature",
    "/api/runs",
    "/api/drafts",
    "/api/compare",
)


def _project_scoped(path: str) -> bool:
    for prefix in PROJECT_SCOPED:
        if prefix.endswith("/"):
            if path.startswith(prefix):
                return True
        elif path == prefix or path.startswith(prefix + "/"):
            return True
    return False


class SecuredApp:
    """The FastAPI app wrapped in the security middleware (outermost, so it sees everything)."""

    def __init__(self, app: FastAPI, security: SecurityConfig) -> None:
        self.inner = app
        self.state = app.state
        self._app = SecurityMiddleware(app, security)

    async def __call__(self, scope, receive, send):  # type: ignore[no-untyped-def]
        await self._app(scope, receive, send)
