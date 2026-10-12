"""The local web server: JSON API, WebSocket event stream and the bundled web app."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import threading
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import anyio
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from logogram import __version__
from logogram.backends.base import BackendError
from logogram.datasets import (
    DatasetError,
    PromptRecord,
    check_dataset_name,
    decode_dataset,
    derive_records,
    file_sha256,
    load_dataset,
    parse_jsonl,
    read_dataset_bytes,
    write_dataset,
)
from logogram.gallery import DEFAULT_EXAMPLE, list_examples
from logogram.history import RunInUse, RunNotFound
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
from logogram.schema import (
    ContinuationsFile,
    Manifest,
    QueueChanged,
    QueuedRunInfo,
    RunListing,
    Summary,
)
from logogram.server import models as M
from logogram.server.security import SecurityConfig, SecurityMiddleware
from logogram.server.state import AppState, Conflict, EventHub, Job, Missing, request_project
from logogram.spec import (
    DEVICE,
    SHA256,
    SINGLE_POSITION_METRICS,
    ChatTemplate,
    Metric,
    ModelRef,
    Position,
    PredictionSettings,
    Scope,
    Spec,
)
from logogram.sweeps import SweepNotFound
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


class OpenExample(BaseModel):
    # One of logogram.gallery.EXAMPLES; without it, the example there was before the gallery.
    example: str = Field(default=DEFAULT_EXAMPLE, max_length=100)


class IOIRequest(BaseModel):
    name: str = "ioi"
    n: int = Field(default=32, ge=1, le=100_000)
    seed: int = Field(default=0, ge=0, lt=2**32)
    templates: list[str] | None = None
    patterns: list[Literal["ABBA", "BABA"]] = ["ABBA", "BABA"]
    corruption: Literal["flip", "abc"] = "flip"
    overwrite: bool = False


class TaskRequest(BaseModel):
    task: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=120)
    n: int = Field(ge=1, le=100_000)
    seed: int = Field(ge=0, lt=2**32)
    options: dict[str, Any] = {}
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


class DeriveRequest(BaseModel):
    """A dataset's prompts, some edited and some left out, saved as a new dataset. The dataset
    they come from never changes: runs record its hash."""

    # The dataset edited, as specs name it (a run's saved copy too), and the version edited.
    dataset: str
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # The new dataset's name, checked as is: a name that is really a path is refused.
    name: str
    # Indices of the prompts to leave out, and the new value of each edited prompt by index.
    remove: list[int] = []
    replace: dict[int, Any] = {}


class AnalysisRequest(BaseModel):
    """An interactive analysis. The choices that change its numbers are stated, as in a spec."""

    model: ModelRef | None = None
    dataset_sha256: str | None = None
    prepend_bos: bool
    # The chat template prompts are wrapped in, as a spec's tokenization.chat_template says.
    chat_template: ChatTemplate | None = None
    limit: int | None = Field(default=None, ge=1, le=1_000_000)
    batch_size: int = Field(ge=1, le=4096)

    @model_validator(mode="after")
    def _one_beginning(self) -> AnalysisRequest:
        if self.chat_template is not None and self.prepend_bos:
            raise ValueError(
                "with a chat template, the template writes the beginning of the text: set "
                "prepend_bos to false"
            )
        return self


class TokenizeRequest(AnalysisRequest):
    dataset: str | None = None
    index: int = 0
    record: PromptRecord | None = None
    # The metric the prompts are for: it decides whether answers of several tokens can be read.
    metric: Metric


class EstimateRequest(BaseModel):
    id: str
    # A local model is found by the hash of its files (logogram.local_models).
    source: Literal["hub", "local"] = "hub"
    sha256: str | None = Field(default=None, pattern=SHA256.pattern)
    revision: str | None = None
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    device: str = Field(default="auto", pattern=DEVICE.pattern)
    process_weights: bool = True


class ChatWrapRequest(BaseModel):
    """A prompt in the loaded model's chat template, to show what the model reads."""

    model: ModelRef | None = None
    chat_template: ChatTemplate
    text: str = Field(min_length=1, max_length=4000)


class AddModelRequest(BaseModel):
    """A model folder to hash and add to this machine's local models, under a name."""

    path: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=200)


class RunMemoryRequest(BaseModel):
    batch_size: int = Field(ge=1, le=4096)
    # The longest prompt, in tokens of the loaded model.
    seq_len: int = Field(ge=1, le=1_000_000)
    method: Literal[
        "activation_patching",
        "ablation",
        "attribution_patching",
        "direct_logit_attribution",
        "path_patching",
        "steering",
    ]
    # The spec's scope, the prompts it runs and its bootstrap resamples: what its sites hold
    # (thousands of neurons hold more than a batch). Left out, only a batch is counted.
    scope: Scope | None = None
    prompts: int | None = Field(default=None, ge=1, le=10_000_000)
    bootstrap: int | None = Field(default=None, ge=1, le=100_000)


class BaselineRequest(AnalysisRequest):
    dataset: str
    metric: Metric


class AttentionRequest(AnalysisRequest):
    dataset: str
    index: int = 0
    layer: int
    head: int
    which: Literal["clean", "corrupt"] = "clean"


class HeadKindsRequest(AnalysisRequest):
    """Score every head as a previous-token, duplicate-token and induction head on seeded random
    sequences repeated twice. Every choice that changes the scores is stated."""

    seed: int = Field(ge=0, lt=2**32)
    sequences: int = Field(ge=2, le=512)
    length: int = Field(ge=2, le=4096)
    bootstrap: int = Field(ge=100, le=100_000)
    ci: float = Field(gt=0.5, lt=1.0)


class LookRequest(AnalysisRequest):
    """Any text, read without making a dataset."""

    text: str = Field(min_length=1, max_length=4000)


class LookPredictionsRequest(LookRequest):
    top_k: int = Field(ge=1, le=20)
    # The token whose layer-by-layer predictions to read; the last one when absent.
    position: int | None = Field(default=None, ge=0)


class LookAttentionRequest(LookRequest):
    layer: int = Field(ge=0)
    head: int = Field(ge=0)


class LookFeaturesRequest(LookRequest):
    top_k: int = Field(ge=1, le=20)


class LookFeatureRequest(LookRequest):
    """One feature of the loaded SAE, along the text."""

    feature: int = Field(ge=0)


class LookSteer(BaseModel):
    """A feature of the loaded SAE to add while the text continues: ``activation`` units of it
    (in the feature's own units), at every position from token ``start`` on."""

    feature: int = Field(ge=0)
    activation: float = Field(allow_inf_nan=False)
    start: int = Field(ge=0)


class LookContinueRequest(LookRequest):
    tokens: int = Field(ge=1, le=64)
    steer: LookSteer | None = None


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


class FeatureWritesRequest(FeatureRequest):
    # The token to read it at, as a spec names positions; by default where it fires most.
    position: Position | None = None
    top: int = Field(default=10, ge=1, le=50)


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


class SweepRequest(BaseModel):
    """A sweep: a run (or a spec) and the values of one of its fields to run it across, in the
    order given. logogram.sweeps checks the field and the values."""

    model_config = ConfigDict(extra="forbid")

    run: str | None = None
    spec: dict[str, Any] | None = None
    field: str = Field(min_length=1, max_length=64)
    values: list[int | str] = Field(max_length=256)

    @model_validator(mode="after")
    def _one_source(self) -> SweepRequest:
        if (self.run is None) == (self.spec is None):
            raise ValueError("Sweep either a run or a spec.")
        return self


class AnnotateRequest(BaseModel):
    """What to change in what is said about a run (its annotations.json: the spec never changes).
    Fields left out stay as they are; a title of null or "", or the spec's own name, clears it."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=10_000)  # 200 once on one line
    starred: bool | None = None
    archived: bool | None = None


class SettingsRequest(BaseModel):
    system_check_seen: bool | None = None
    theme: Literal["light", "dark", "system"] | None = None
    # Whether Logogram may ask PyPI about new versions once a day. Off until the user says yes.
    update_check: bool | None = None
    # Whether single keys (1 to 5, [ and ], P, B, A, C and ?) act as shortcuts. On until turned
    # off, for someone who uses speech input or presses keys by accident.
    single_key_shortcuts: bool | None = None


class ReportRequest(BaseModel):
    """A problem report: with the hardware's details or without, and the last error the app
    showed, as it showed it (paths and names are scrubbed from it here)."""

    include_hardware: bool = False
    last_error: str | None = Field(default=None, max_length=20_000)


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
    @app.exception_handler(RunNotFound)
    @app.exception_handler(SweepNotFound)
    async def _missing(_: Request, exc: Missing) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=409)

    @app.exception_handler(Conflict)
    @app.exception_handler(NoteConflict)
    @app.exception_handler(RunInUse)
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
        # Before the job: a run leaves the queue only once its job has started.
        queue = state.queue_status()
        return {
            "version": __version__,
            "project": state.project.to_dict() if state.project else None,
            "model": state.model_payload(),
            "job": state.job.to_dict() if state.job else None,
            "queue": queue["runs"],
            "queue_status": queue,
            "first_run": not settings.get("system_check_seen", False),
            # Until someone picks one, the app follows the computer's light or dark setting.
            "theme": settings.get("theme", "system"),
            "single_key_shortcuts": settings.get("single_key_shortcuts", True),
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

    @app.get("/api/environment", response_model=M.EnvironmentOut)
    def get_environment() -> dict[str, Any]:
        """The versions and devices a rerun here would use, to compare with a run's manifest."""
        from logogram.system import environment

        return environment()

    @app.post("/api/report", response_model=M.ProblemReport)
    def problem_report(body: ReportRequest) -> dict[str, Any]:
        """A report to paste into an issue, built from what this server knows. Nothing is sent:
        the app shows it to copy."""
        from logogram.support import build_report

        project = state.project
        return build_report(
            model=state.model_payload(),
            last_error=body.last_error,
            include_hardware=body.include_hardware,
            roots=[project.root] if project else [],
            secrets=[security.token],
        ).to_dict()

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

    @app.get("/api/examples", response_model=list[M.ExampleInfo])
    def examples() -> list[dict[str, Any]]:
        """The bundled examples, and whether each has been copied into the projects folder."""
        return list_examples(default_projects_parent())

    @app.post("/api/projects/example", response_model=M.ProjectInfo)
    def example(body: OpenExample | None = None) -> dict[str, Any]:
        """Copy an example into the projects folder (once) and open it; a copy is never
        overwritten."""
        chosen = (body or OpenExample()).example
        return _enter(open_example(default_projects_parent(), example=chosen))

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

    def _write_new(
        path: Path,
        records: list[PromptRecord],
        overwrite: bool,
        fix: str = "Choose another name, or replace it explicitly.",
    ) -> dict[str, Any]:
        exists = f"{path.name} already exists. {fix}"
        if os.path.lexists(path) and not overwrite:
            raise DatasetError(exists)
        try:
            write_dataset(path, records, replace=overwrite)
        except FileExistsError:  # created since the check: it is never replaced
            raise DatasetError(exists) from None
        return {"name": path.name, "path": f"datasets/{path.name}", "n": len(records)}

    def _records(
        dataset: str, limit: int | None = None, sha256: str | None = None
    ) -> list[PromptRecord]:
        project = state.require_project()
        path = project.resolve_dataset(dataset)
        content = read_dataset_bytes(path)
        if sha256 is not None and hashlib.sha256(content).hexdigest() != sha256:
            raise Conflict(
                "The dataset changed since these settings were saved. Restore the original dataset or choose the updated dataset for a new experiment."
            )
        records = parse_jsonl(decode_dataset(content, dataset), source=dataset)
        return records[:limit] if limit else records

    @app.get("/api/datasets/{name}", response_model=M.DatasetDetail)
    @app.get("/api/dataset", response_model=M.DatasetDetail)
    def get_dataset(
        name: str = "",
        path: str | None = None,
        prepend_bos: bool = True,
        limit: int | None = None,
        chat_template: str | None = None,
    ) -> dict[str, Any]:
        """A dataset's prompts, and with a model loaded what it makes of them: tokenized with or
        without the beginning-of-sequence token, or wrapped in the chat template that
        ``chat_template`` (a spec's tokenization.chat_template, as JSON) describes."""
        project = state.require_project()
        template = None
        if chat_template:
            from pydantic import TypeAdapter

            template = TypeAdapter(ChatTemplate).validate_json(chat_template)
            prepend_bos = False
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

            prepared, issues = prepare_with_issues(
                state.backend, records, prepend_bos, chat_template=template
            )
            out["issues"] = [i.to_dict() for i in issues]
            out["lengths"] = sorted({p.length for p in prepared})
            # Answers this model reads as continuations of several tokens: only the metrics that
            # read continuations can score them, which the app says before a run is started.
            out["continuations"] = [
                p.index for p in prepared if p.answer.positions > 1 or p.distractor.positions > 1
            ]
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

    @app.get("/api/tasks", response_model=list[M.TaskOut])
    def tasks() -> list[dict[str, Any]]:
        from logogram.tasks import TASKS

        return [t.to_dict() for t in TASKS.values()]

    @app.post("/api/datasets/generate", response_model=M.TaskDatasetCreated)
    def generate_dataset(body: TaskRequest) -> dict[str, Any]:
        from logogram.tasks import generate_task, get_task

        project = state.require_project()
        backend = state.backend
        single: Callable[[str], bool] | None = None
        count: Callable[[str], int] | None = None
        if backend is not None:  # words this model splits are left out, and pairs keep one length
            model = backend

            def single_token(text: str) -> bool:
                return model.single_token_id(text) is not None

            def token_count(text: str) -> int:
                return len(model.tokenize(text, prepend_bos=False).ids)

            single, count = single_token, token_count

        records = generate_task(
            body.task, body.n, body.seed, body.options, single_token=single, token_count=count
        )
        created = _write_new(_new_dataset_path(project, body.name), records, body.overwrite)
        return {**created, "metric": get_task(body.task).recommended_metric(body.options)}

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

    @app.post("/api/datasets/derive", response_model=M.DatasetCreated)
    def derive_dataset(body: DeriveRequest) -> dict[str, Any]:
        """Save a dataset's prompts, some edited and some left out, as a new dataset. Nothing is
        ever replaced: neither the dataset edited nor a file already called ``name``."""
        project = state.require_project()
        path = project.dataset_file(check_dataset_name(body.name))
        fix = "Choose another name."
        if os.path.lexists(path):  # say so before reading the prompts
            raise DatasetError(f"{path.name} already exists. {fix}")
        source = project.resolve_dataset(body.dataset)
        content = read_dataset_bytes(source)
        if hashlib.sha256(content).hexdigest() != body.dataset_sha256:
            raise Conflict(
                f"{source.name} changed since these edits were made, so they no longer match its "
                "prompts. Discard them and edit the file as it is now."
            )
        records = parse_jsonl(decode_dataset(content, body.dataset), source=body.dataset)
        derived = derive_records(records, remove=body.remove, replace=body.replace)
        return _write_new(path, derived, overwrite=False, fix=fix)

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
            chat_template=body.chat_template,
        )

    @app.post("/api/chat/wrap", response_model=M.ChatWrapped)
    def chat_wrap(body: ChatWrapRequest) -> dict[str, Any]:
        """A prompt as the loaded model reads it in its own chat template: the whole text, where
        the prompt is in it, and the template's SHA-256."""
        from logogram.chat import WrapError, chat_wrapper

        backend = _analysis_backend(body)
        with backend.lock:
            chat = chat_wrapper(backend, body.chat_template)
            assert chat is not None
            try:
                text, start = chat.wrap(backend, body.text)
                problem = None
            except WrapError as exc:
                text, start, problem = exc.text, None, str(exc)
        return {
            "text": text,
            "start": start,
            "end": None if start is None else start + len(body.text),
            "before": chat.before,
            "after": chat.after,
            "sha256": chat.sha256,
            "problem": problem,
        }

    # -- models ------------------------------------------------------------------------------

    @app.get("/api/models/presets", response_model=M.Presets)
    def presets() -> dict[str, Any]:
        return {"presets": MODEL_PRESETS, "suggestions": MODEL_SUGGESTIONS}

    @app.post("/api/models/estimate", response_model=M.EstimateOut)
    def estimate(body: EstimateRequest) -> dict[str, Any]:
        """What loading a model would take: its size, and whether it fits in the memory free on
        the device chosen (one GPU's, for cuda:N). A local model is read from its folder."""
        from logogram.backends import hub
        from logogram.backends.transformer_lens import architecture_support, resolve_device
        from logogram.system import estimate_memory

        device = resolve_device(body.device)
        if body.source == "local":
            files, revision, n_params, config = _local_estimate(body)
            download = 0
            gated = False
        else:
            repo = hub.resolve(body.id, body.revision)
            config = hub.fetch_config(body.id, repo.revision)
            files, revision, n_params, gated = repo.files, repo.revision, repo.n_params, repo.gated
            download = 0
            from huggingface_hub import try_to_load_from_cache

            for filename, size in repo.files:
                if not isinstance(
                    try_to_load_from_cache(body.id, filename, revision=repo.revision), str
                ):
                    download += size
        arch = hub.read_architecture(config)
        support_note = architecture_support(arch.architecture)
        if n_params is None:
            weights = sum(s for f, s in files if f.endswith(".safetensors"))
            n_params = weights // 2  # stored size is a fallback; most checkpoints are 16-bit
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
            "revision": revision,
            "architecture": arch.__dict__,
            "download_bytes": download,
            "total_bytes": sum(s for _, s in files),
            "gated": gated,
            "supported": support_note is None,
            "support_note": support_note,
            "estimate": est.to_dict(),
        }

    def _local_estimate(
        body: EstimateRequest,
    ) -> tuple[list[tuple[str, int]], None, int | None, dict[str, Any]]:
        """A local model's files, its stored values and its config, read from its folder."""
        import json

        from logogram.backends.hub import count_stored_values
        from logogram.local_models import find_model, missing_message

        model = find_model(sha256=body.sha256) if body.sha256 else find_model(name=body.id)
        if model is None:
            raise Missing(missing_message(body.id, body.sha256))
        folder = Path(model.folder)
        try:
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BackendError(
                f"The config.json of {model.name} can't be read ({exc}). Check its folder."
            ) from exc
        files = [(f.path, f.size) for f in model.files]
        weights = [name for name, _ in files if name.endswith(".safetensors")]
        return files, None, count_stored_values(folder, weights), config

    @app.get("/api/models/revisions", response_model=M.RevisionsOut)
    def model_revisions(id: str) -> dict[str, Any]:
        """A model's branches and tags on Hugging Face, to sweep over (one request, when asked)."""
        from logogram.backends import hub

        revisions = hub.list_revisions(id)
        return {"id": id, "revisions": [r.__dict__ for r in revisions]}

    @app.post("/api/models/run-memory", response_model=M.RunMemoryOut)
    def run_memory(body: RunMemoryRequest) -> dict[str, Any]:
        """What a run of this batch size needs on the loaded model, against the memory free."""
        from logogram.system import ALL_LAYER_METHODS, estimate_run_memory, scope_footprint

        backend = state.backend
        if backend is None:
            raise Conflict("Load a model to estimate what a run on it needs.")
        info = backend.info
        footprint = (
            scope_footprint(
                body.scope,
                n_layers=info.n_layers,
                n_heads=info.n_heads,
                n_kv_heads=int(info.extra.get("n_key_value_heads") or info.n_heads),
                d_model=info.d_model,
                d_head=info.d_head,
                d_mlp=info.d_mlp or 0,
                seq_len=body.seq_len,
            )
            if body.scope is not None
            else None
        )
        return estimate_run_memory(
            n_layers=info.n_layers,
            n_heads=info.n_heads,
            d_model=info.d_model,
            d_mlp=info.d_mlp or 0,
            d_vocab=info.d_vocab,
            dtype=info.dtype,
            device=info.device,
            batch_size=body.batch_size,
            seq_len=body.seq_len,
            all_layers=body.method in ALL_LAYER_METHODS,
            footprint=footprint,
            method=body.method,
            prompts=body.prompts or 0,
            bootstrap=body.bootstrap or 0,
        ).to_dict()

    @app.post("/api/models/load", response_model=M.JobInfo)
    def load(body: ModelRef) -> dict[str, Any]:
        return state.load_model_job(body).to_dict()

    # -- local models: folders on this machine, found by the hash of their files ---------------

    @app.get("/api/models/local", response_model=list[M.LocalModelOut])
    def local_models() -> list[dict[str, Any]]:
        """The models added from folders on this machine (their folders are shown here only)."""
        from logogram.local_models import list_models

        return [m.to_dict() for m in list_models()]

    @app.post("/api/models/local", response_model=M.JobInfo)
    def add_local_model(body: AddModelRequest) -> dict[str, Any]:
        """Hash a model folder and add it, as a job with progress (big folders take a while)."""
        from logogram.local_models import check_name, model_files

        name = check_name(body.name)
        folder = Path(body.path).expanduser().resolve()
        model_files(folder)  # refuse a folder that isn't a model before starting a job
        return state.add_model_job(folder, name).to_dict()

    @app.delete("/api/models/local/{name}", response_model=list[M.LocalModelOut])
    def remove_local_model(name: str) -> list[dict[str, Any]]:
        """Forget a local model; its folder is left as it is."""
        from logogram.local_models import list_models, remove_model

        remove_model(name)
        return [m.to_dict() for m in list_models()]

    @app.post("/api/models/unload", response_model=M.ModelStatus)
    def unload() -> dict[str, Any]:
        state.unload_model()
        return state.model_payload()

    # -- analyses ----------------------------------------------------------------------------

    def _analysis_backend(body: AnalysisRequest | ChatWrapRequest):  # type: ignore[no-untyped-def]
        from logogram.runner import model_matches

        backend = state.require_backend()
        if body.model is not None and not model_matches(backend, body.model):
            raise Conflict(
                "The loaded model doesn't match these experiment settings. Load the experiment's model, revision, dtype, device and weight processing before inspecting it."
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
                chat_template=body.chat_template,
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
                chat_template=body.chat_template,
            )

    @app.post("/api/heads/kinds", response_model=M.HeadKinds)
    def head_kinds(body: HeadKindsRequest) -> dict[str, Any]:
        from logogram.heads import HeadKindSettings, head_kind_scores

        backend = _analysis_backend(body)
        settings = HeadKindSettings(
            seed=body.seed,
            sequences=body.sequences,
            length=body.length,
            prepend_bos=body.prepend_bos,
            batch_size=body.batch_size,
            bootstrap=body.bootstrap,
            ci=body.ci,
            chat_template=(
                None if body.chat_template is None else body.chat_template.model_dump(mode="json")
            ),
        )
        with backend.lock:
            return head_kind_scores(backend, settings)

    @app.post("/api/look", response_model=M.LookReport)
    def look(body: LookPredictionsRequest) -> dict[str, Any]:
        from logogram.inside import look_report

        backend = _analysis_backend(body)
        with backend.lock:
            return look_report(
                backend,
                body.text,
                prepend_bos=body.prepend_bos,
                top_k=body.top_k,
                position=body.position,
                chat_template=body.chat_template,
            )

    @app.post("/api/look/attention", response_model=M.LookAttention)
    def look_attention(body: LookAttentionRequest) -> dict[str, Any]:
        from logogram.inside import look_attention as attention

        backend = _analysis_backend(body)
        with backend.lock:
            return attention(
                backend,
                body.text,
                prepend_bos=body.prepend_bos,
                layer=body.layer,
                head=body.head,
                chat_template=body.chat_template,
            )

    @app.post("/api/look/features", response_model=M.LookFeatures)
    def look_features(body: LookFeaturesRequest) -> dict[str, Any]:
        from logogram.inside import look_features as features

        backend, sae = _analysis_backend(body), _sae()
        with backend.lock:
            return features(
                backend,
                sae,
                body.text,
                prepend_bos=body.prepend_bos,
                top_k=body.top_k,
                chat_template=body.chat_template,
            )

    @app.post("/api/look/feature", response_model=M.LookFeatureAlong)
    def look_feature(body: LookFeatureRequest) -> dict[str, Any]:
        """One feature's activation at each token of the text, and its largest."""
        from logogram.inside import look_feature as along

        backend, sae = _analysis_backend(body), _sae()
        with backend.lock:
            return along(
                backend,
                sae,
                body.text,
                prepend_bos=body.prepend_bos,
                feature=body.feature,
                chat_template=body.chat_template,
            )

    @app.post("/api/look/continue", response_model=M.LookContinuation)
    def look_continue(body: LookContinueRequest) -> dict[str, Any]:
        """What the model writes after the text, greedily; with ``steer``, also with a feature of
        the loaded SAE added where the SAE writes."""
        from logogram.inside import continue_text

        backend = _analysis_backend(body)
        steer = body.steer
        sae = _sae() if steer is not None else None
        with backend.lock:
            return continue_text(
                backend,
                body.text,
                prepend_bos=body.prepend_bos,
                tokens=body.tokens,
                sae=sae,
                feature=None if steer is None else steer.feature,
                activation=None if steer is None else steer.activation,
                start=None if steer is None else steer.start,
                chat_template=body.chat_template,
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
                chat_template=body.chat_template,
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
                backend,
                sae,
                records,
                prepend_bos=body.prepend_bos,
                batch_size=body.batch_size,
                chat_template=body.chat_template,
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
                chat_template=body.chat_template,
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
                chat_template=body.chat_template,
            )

    @app.post("/api/sae/writes")
    def sae_writes(body: FeatureWritesRequest) -> dict[str, Any]:
        """The tokens a feature promotes and suppresses directly, at one token of one prompt."""
        from logogram.analysis import feature_writes_report

        backend, sae = _analysis_backend(body), _sae()
        records = _records(body.dataset, body.limit, body.dataset_sha256)
        with backend.lock:
            return feature_writes_report(
                backend,
                sae,
                records,
                feature=body.feature,
                index=body.index,
                which=body.which,
                prepend_bos=body.prepend_bos,
                position=body.position,
                top=body.top,
                chat_template=body.chat_template,
            )

    # -- runs --------------------------------------------------------------------------------

    def _read_run(project: Project, run_id: str) -> dict[str, Any]:
        folder = project.run_dir(run_id)
        if not (folder / "spec.json").is_file():
            raise Missing(f"There is no run {run_id}.")
        out: dict[str, Any] = {"id": run_id}
        for name in ("spec", "summary", "manifest", "predictions", "continuations"):
            path = folder / f"{name}.json"
            try:
                data = project.read_json(path, optional=name != "spec")
                schema = {
                    "spec": Spec,
                    "summary": Summary,
                    "manifest": Manifest,
                    "predictions": M.PredictionReport,
                    "continuations": ContinuationsFile,
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
        try:
            out["traceback"] = project.run_traceback(run_id)
        except (ValueError, OSError):  # not a regular file in the project, or far too large
            out["traceback"] = None
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

    def _finished(project: Project, run_id: str) -> None:
        """A run with results: finished, or stopped early with what finished kept (the files say
        it is partial)."""
        listing = project.run_listing(run_id)
        if listing is None or (listing.status != "finished" and listing.partial is None):
            raise Conflict(
                "Only a run with results can be exported: a finished run, or one that stopped "
                "early and kept what finished. Select one first."
            )

    @app.get("/api/runs/{run_id}/export.csv")
    def export_run(run_id: str, text: bool = False) -> StreamingResponse:
        """Every site and prompt, one row each; with ``text``, each row carries its prompt."""
        from logogram.exports import results_csv
        from logogram.runs import run_records

        project = state.require_project()
        _finished(project, run_id)
        folder = project.run_dir(run_id)
        path = project.readable(folder / "results.parquet")
        records = None
        if text:
            records = run_records(project, Spec.from_path(project.readable(folder / "spec.json")))
            if records is None:
                raise Conflict(
                    "This run's prompts are gone or have changed since it ran, so its rows can't "
                    "carry their text. Export without prompt text."
                )
        return StreamingResponse(
            results_csv(path, records),
            media_type="text/csv",
            headers={
                "content-disposition": f'attachment; filename="{run_id}.csv"',
            },
        )

    @app.get("/api/runs/{run_id}/sites.csv")
    def export_sites(run_id: str) -> Response:
        """One row per site: its mean, intervals, q-value and counts."""
        from logogram.exports import sites_csv
        from logogram.report import sites_table
        from logogram.runs import load_summary

        project = state.require_project()
        _finished(project, run_id)
        return Response(
            sites_csv(sites_table(load_summary(project, run_id))),
            media_type="text/csv",
            headers={"content-disposition": f'attachment; filename="{run_id}-sites.csv"'},
        )

    @app.get("/api/runs/{run_id}/report.html")
    def report_run(run_id: str, hardware: bool = False) -> Response:
        """The run as one self-contained HTML page, to download and pass on."""
        from logogram.report import run_report
        from logogram.runs import load_continuations, load_summary

        project = state.require_project()
        _finished(project, run_id)
        folder = project.run_dir(run_id)
        spec = Spec.from_path(project.readable(folder / "spec.json"))
        manifest = project.read_json(folder / "manifest.json", optional=True) or {}
        page = run_report(
            spec,
            load_summary(project, run_id),
            manifest,
            run_id,
            hardware=hardware,
            continuations=load_continuations(project, run_id),
        )
        return Response(
            page,
            media_type="text/html; charset=utf-8",
            headers={"content-disposition": f'attachment; filename="{run_id}.html"'},
        )

    @app.get("/api/runs/{run_id}/figure.svg")
    def figure_run(
        run_id: str, value: Literal["effect", "delta"] = "effect", unit: bool = False
    ) -> Response:
        """The run's map as a vector figure: the normalized effect or the change in the metric,
        on a fitted scale or fixed at ±1."""
        from logogram.report import figure_svg
        from logogram.runs import load_summary

        project = state.require_project()
        _finished(project, run_id)
        summary = load_summary(project, run_id)
        if (summary.get("layout") or {}).get("kind") in ("sites", "site_sets"):
            raise Conflict("This run has no map to draw: its results are a list. Export a CSV.")
        spec = Spec.from_path(project.readable(project.run_dir(run_id) / "spec.json"))
        return Response(
            figure_svg(spec, summary, value, unit),
            media_type="image/svg+xml",
            headers={"content-disposition": f'attachment; filename="{run_id}.svg"'},
        )

    @app.get("/api/runs/{run_id}/methods", response_model=M.MethodsText)
    def run_methods(run_id: str) -> dict[str, Any]:
        """The methods paragraph, as the app copies it and the report prints it."""
        from logogram.report import methods_text

        project = state.require_project()
        folder = project.run_dir(run_id)
        spec = Spec.from_path(project.readable(folder / "spec.json"))
        summary = project.read_json(folder / "summary.json", optional=True)
        manifest = project.read_json(folder / "manifest.json", optional=True) or {}
        return {"text": methods_text(spec, summary, manifest, run_id)}

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
        from logogram.robustness import require_finished, robustness_spec

        project = state.require_project()
        original = Spec.from_path(project.readable(project.run_dir(run_id) / "spec.json"))
        require_finished(project, run_id, "Checking robustness")
        variant, change = robustness_spec(original, experiment=body.experiment, dtype=body.dtype)
        job = state.run_spec_job(
            variant, derived_from={"run": run_id, "kind": "robustness", "change": change}
        )
        return {"run_id": job.run_id, "job": job.to_dict()}

    @app.post("/api/runs/{run_id}/verify", response_model=M.StartedRun)
    def verify(run_id: str, body: VerifyRequest) -> dict[str, Any]:
        """Patch, for real, the sites an attribution patching run estimated to matter most."""
        from logogram.robustness import require_finished
        from logogram.verify import verification_spec

        project = state.require_project()
        run = _read_run(project, run_id)
        require_finished(project, run_id, "Verifying by patching")
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

    # -- sweeps: one spec across several values of one field (logogram.sweeps) ------------------

    def _swept_run(project: Project, run_id: str) -> tuple[Spec, RunListing]:
        listing = project.run_listing(run_id)
        if listing is None:
            raise Missing(f"There is no run {run_id}. It may have been deleted.")
        if listing.damaged:
            raise ValueError(listing.error or "This run's spec can't be read.")
        return Spec.from_path(project.readable(project.run_dir(run_id) / "spec.json")), listing

    def _sweep_source(project: Project, body: SweepRequest) -> tuple[Spec, RunListing | None]:
        if body.run is None:
            return Spec.model_validate(body.spec), None
        return _swept_run(project, body.run)

    @app.get("/api/runs/{run_id}/sweep-fields", response_model=list[M.SweepFieldOut])
    def sweep_fields(run_id: str) -> list[dict[str, Any]]:
        """The fields this run's spec can be swept over, with its own value of each."""
        from logogram.sweeps import suggested_values, sweepable_fields, value_label

        project = state.require_project()
        spec, listing = _swept_run(project, run_id)
        out = []
        for f, current in sweepable_fields(spec):
            suggested = suggested_values(project, spec, f, listing)
            # A run's dataset is a saved copy: name it by the file of the project it matches.
            shown = suggested[0] if f.kind == "dataset" and suggested else current
            out.append(
                {
                    "field": f.path,
                    "kind": f.kind,
                    "label": f.label,
                    "plural": f.plural,
                    "description": f.description,
                    "current": current,
                    "current_label": None if shown is None else value_label(f, shown),
                    "suggested": suggested,
                }
            )
        return out

    @app.post("/api/sweeps/plan", response_model=M.SweepPlanOut)
    def plan_sweep(body: SweepRequest) -> dict[str, Any]:
        """The runs a sweep would make, without making them: what the app shows before Queue."""
        from logogram.sweeps import plan_sweep as plan

        project = state.require_project()
        spec, original = _sweep_source(project, body)
        return plan(project, spec, body.field, body.values, original=original).to_dict()

    @app.post("/api/sweeps", response_model=M.SweepStarted)
    def start_sweep(body: SweepRequest) -> dict[str, Any]:
        """Queue a sweep's runs, in the order of its values: the first starts at once when nothing
        is running. The run it is made from stands in for its own value."""
        from logogram.sweeps import plan_sweep as plan

        project = state.require_project()
        spec, original = _sweep_source(project, body)
        made = plan(project, spec, body.field, body.values, original=original)
        started: Job | None = None
        # One hold for all of them, so they line up together, in order.
        with state._job_lock:
            for item in made.runs:
                if item.spec is not None:
                    out = state.queue_run(item.spec, derived_from=item.derived_from)
                    if isinstance(out, Job):
                        started = out
        return {
            "sweep": made.id,
            "plan": made.to_dict(),
            "run_id": started.run_id if started else None,
            "job": started.to_dict() if started else None,
            "queue": state.queued_runs(),
        }

    @app.get("/api/sweeps/{sweep_id}", response_model=M.SweepTable)
    def get_sweep(
        sweep_id: str, choose: Literal["first", "all"] = "first", top: int = 10
    ) -> dict[str, Any]:
        from logogram.sweeps import sweep_table

        return sweep_table(state.require_project(), sweep_id, choose=choose, top=top)

    @app.get("/api/sweeps/{sweep_id}/table.csv")
    def export_sweep(
        sweep_id: str, choose: Literal["first", "all"] = "first", top: int = 10
    ) -> Response:
        """The sweep's table, one row per site and value."""
        from logogram.exports import sweep_csv
        from logogram.sweeps import sweep_table, table_rows

        table = sweep_table(state.require_project(), sweep_id, choose=choose, top=top)
        return Response(
            sweep_csv(table_rows(table)),
            media_type="text/csv",
            headers={"content-disposition": f'attachment; filename="sweep-{table["sweep"]}.csv"'},
        )

    @app.get("/api/runs/{run_id}/derived", response_model=list[RunListing])
    def derived(run_id: str) -> list[dict[str, Any]]:
        project = state.require_project()
        return [
            r.to_dict()
            for r in project.list_runs()
            if r.derived_from and r.derived_from.run == run_id
        ]

    @app.patch("/api/runs/{run_id}", response_model=RunListing)
    def annotate_run(run_id: str, body: AnnotateRequest) -> dict[str, Any]:
        """Rename, star or archive a run (or a draft, or a damaged run)."""
        given = body.model_dump(exclude_unset=True)
        changes = {k: v for k, v in given.items() if k == "title" or v is not None}
        return state.annotate_run(run_id, changes).to_dict()

    @app.delete("/api/runs/{run_id}", response_model=M.OkOut)
    def delete_run(run_id: str) -> dict[str, bool]:
        """Delete a finished, failed or cancelled run, or a draft, folder and all (there is no
        trash: see logogram.history). Refused while it runs or waits in the queue."""
        state.delete_run(run_id)
        return {"ok": True}

    @app.get("/api/runs/{run_id}/sites/{site}", response_model=M.SiteDetail)
    def site_detail(run_id: str, site: int) -> dict[str, Any]:
        from logogram.runs import site_detail as detail

        return detail(state.require_project(), run_id, site)

    @app.get("/api/compare", response_model=M.Comparison)
    def compare(a: str, b: str) -> dict[str, Any]:
        from logogram.compare import compare_runs

        return compare_runs(state.require_project(), a, b)

    @app.post("/api/jobs/cancel", response_model=M.CancelOut)
    async def cancel() -> dict[str, Any]:
        # On the event loop, not the thread pool: queued analyses can't hold up a cancel.
        job = state.cancel_job()
        return {"job": job.to_dict() if job else None}

    # -- the run queue -----------------------------------------------------------------------

    @app.get("/api/queue", response_model=list[QueuedRunInfo])
    def queued_runs() -> list[dict[str, Any]]:
        state.require_project()
        return state.queued_runs()

    @app.post("/api/queue", response_model=M.QueueAdded)
    def queue_run(body: RunRequest) -> dict[str, Any]:
        """Run a spec when the job in progress ends, or now if nothing is running."""
        spec = Spec.model_validate(body.spec)
        state.require_project()
        out = state.queue_run(spec, draft_id=body.draft_id)
        started, queued = (out, None) if isinstance(out, Job) else (None, out)
        return {
            "run_id": started.run_id if started else None,
            "job": started.to_dict() if started else None,
            "queued": queued.to_dict() if queued else None,
            "queue": state.queued_runs(),
        }

    @app.delete("/api/queue/{queued_id}", response_model=list[QueuedRunInfo])
    def remove_queued_run(queued_id: str) -> list[dict[str, Any]]:
        return state.remove_queued_run(queued_id)

    @app.get("/api/queue/state", response_model=QueueChanged)
    def queue_status() -> dict[str, Any]:
        """The queue and how it stands: waiting, running (here or in another process, which it
        names), paused or idle. It is kept in the project, so another process can change it."""
        state.require_project()
        return state.queue_status()

    @app.post("/api/queue/start", response_model=QueueChanged)
    def start_queue() -> dict[str, Any]:
        """Run the queue here: its next run starts now if nothing is running."""
        return state.start_queue()

    @app.post("/api/queue/pause", response_model=QueueChanged)
    def pause_queue() -> dict[str, Any]:
        """Start no more of the queue's runs; the run in progress goes on."""
        return state.pause_queue()

    @app.delete("/api/queue", response_model=QueueChanged)
    def clear_queue() -> dict[str, Any]:
        """Take every run out of the queue."""
        return state.clear_queue()

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
    "/api/sae/writes",
    "/api/runs",
    "/api/drafts",
    "/api/compare",
    "/api/queue",
    "/api/sweeps",
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
