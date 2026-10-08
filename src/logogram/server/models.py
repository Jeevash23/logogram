"""Response models for the JSON API. Request bodies are defined beside their routes in app.py."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from logogram.schema import RunListing, SiteSummary
from logogram.spec import ModelRef, PredictionSettings


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


# -- models and jobs ---------------------------------------------------------------------------


class ModelInfoOut(_Out):
    id: str
    revision: str | None
    architecture: str
    n_layers: int
    n_heads: int
    d_model: int
    d_head: int
    d_mlp: int | None
    d_vocab: int
    n_ctx: int
    n_params: int | None
    dtype: str
    device: str
    device_name: str
    process_weights: bool
    site_kinds: list[str]
    backend: str
    backend_version: str
    extra: dict[str, Any]


class ModelStatus(_Out):
    state: Literal["none", "loading", "ready", "error"]
    id: str | None = None
    model_id: str | None = None
    stage: str | None = None
    done: int | None = None
    total: int | None = None
    file: str | None = None
    revision: str | None = None
    error: str | None = None
    info: ModelInfoOut | None = None
    memory: int | None = None
    ref: ModelRef | None = None


class JobInfo(_Out):
    id: str
    kind: Literal["load_model", "load_sae", "run"]
    title: str
    status: Literal["running", "finished", "failed", "cancelled"]
    run_id: str | None
    project_session: str | None = None
    progress: dict[str, Any]
    error: str | None
    started: float
    version: int
    cancelling: bool | None = None


class StartedRun(_Out):
    run_id: str
    job: JobInfo


class CancelOut(_Out):
    job: JobInfo | None


class DraftSaved(_Out):
    run_id: str
    path: str


# -- projects and files ------------------------------------------------------------------------


class DatasetListing(_Out):
    name: str
    path: str
    modified: str | None
    n: int | None = None
    has_positions: bool | None = None
    sha256: str | None = None
    error: str | None = None


class ProjectInfo(_Out):
    name: str
    description: str
    path: str
    session_id: str
    datasets: list[DatasetListing]


class RecentProject(_Out):
    path: str
    name: str
    opened: str | None
    modified: str


class FolderEntry(_Out):
    name: str
    path: str
    is_project: bool


class FolderListing(_Out):
    path: str
    parent: str | None
    is_project: bool
    entries: list[FolderEntry]


class UpdateStatus(_Out):
    current: str
    released: str | None
    automatic: bool | None
    latest: str | None
    available: bool
    old: bool
    checked_at: str | None
    error: str | None
    notes_url: str | None
    command: str


class ServerState(_Out):
    version: str
    project: ProjectInfo | None
    model: ModelStatus
    job: JobInfo | None
    first_run: bool
    theme: Literal["light", "dark", "system"]
    projects_parent: str
    update: UpdateStatus
    sae: dict[str, Any] = {"state": "none"}


class Settings(BaseModel):
    model_config = ConfigDict(extra="allow")

    system_check_seen: bool | None = None
    theme: str | None = None
    update_check: bool | None = None


class OkOut(_Out):
    ok: bool


# -- datasets and prompts ----------------------------------------------------------------------


class PromptIssue(_Out):
    index: int
    kind: str
    message: str


class DatasetDetail(_Out):
    name: str
    path: str
    sha256: str
    n: int
    records: list[dict[str, Any]]
    issues: list[PromptIssue] | None = None
    lengths: list[int] | None = None


class DatasetCreated(_Out):
    name: str
    path: str
    n: int


class IOITemplateOut(_Out):
    id: str
    text: str
    default: bool


class TokenRow(_Out):
    tokens: list[str]
    ids: list[int]


class AnswerTokens(_Out):
    text: str
    tokens: list[str]
    id: int | None


class TokenStrip(_Out):
    clean: TokenRow
    corrupt: TokenRow
    aligned: bool
    differs: list[int]
    labels: dict[str, int]
    answer: AnswerTokens
    distractor: AnswerTokens
    issues: list[PromptIssue]


# -- analyses ----------------------------------------------------------------------------------


class TopToken(_Out):
    token: str
    id: int
    prob: float


class BaselinePrompt(_Out):
    index: int
    clean: str
    corrupt: str
    answer: str
    distractor: str
    clean_logit_diff: float | None
    corrupt_logit_diff: float | None
    clean_answer_prob: float | None
    corrupt_answer_prob: float | None
    clean_top: list[TopToken]
    corrupt_top: list[TopToken]


class BaselineSummaryOut(_Out):
    clean_logit_diff: float | None
    corrupt_logit_diff: float | None
    gap: float | None
    clean_prefers_answer: int
    corrupt_prefers_answer: int
    clean_answer_prob: float | None
    corrupt_answer_prob: float | None


class BaselineReport(_Out):
    n: int
    prompts: list[BaselinePrompt]
    issues: list[PromptIssue]
    summary: BaselineSummaryOut | None


class AttentionData(_Out):
    layer: int
    head: int
    which: Literal["clean", "corrupt"]
    index: int
    tokens: list[str]
    labels: dict[str, int]
    pattern: list[list[float]]
    average: list[list[float]]
    average_tokens: list[str | None]
    average_labels: dict[str, int]
    n_average: int
    n_total: int
    length: int


class PredictionLayer(_Out):
    layer: int
    top: list[TopToken]
    answer_prob: float
    distractor_prob: float
    logit_diff: float


class PredictionReport(_Out):
    index: int
    position: int
    tokens: list[str]
    settings: PredictionSettings
    layers: list[PredictionLayer]
    batch_members: list[int]
    answer: str
    distractor: str
    issues: list[PromptIssue]


class Presets(_Out):
    presets: list[dict[str, Any]]
    suggestions: list[str]


class ArchitectureOut(_Out):
    n_layers: int
    n_heads: int
    d_model: int
    d_mlp: int
    d_vocab: int
    n_ctx: int
    architecture: str


class MemoryEstimateOut(_Out):
    device: str
    dtype: str
    n_params: int
    weights: int
    activations: int
    processing: int
    margin: int
    total: int
    available: int
    verdict: Literal["fits", "tight", "wont_fit"]
    explanation: str


class EstimateOut(_Out):
    id: str
    revision: str
    architecture: ArchitectureOut
    download_bytes: int
    total_bytes: int
    gated: bool
    # Whether TransformerLens can load this architecture, and if not, why (before downloading).
    supported: bool
    support_note: str | None
    estimate: MemoryEstimateOut


# -- runs --------------------------------------------------------------------------------------


class RunDetail(_Out):
    """A run's files as stored. They were validated when written; older files are passed through."""

    id: str
    spec: dict[str, Any] | None
    summary: dict[str, Any] | None
    manifest: dict[str, Any] | None
    predictions: PredictionReport | None = None
    listing: RunListing | None
    folder: str


class PromptEvidence(_Out):
    index: int
    clean: str | None
    corrupt: str | None
    answer: str | None
    distractor: str | None
    effect: float | None
    delta: float | None
    patched_logit_diff: float | None
    receiver_logit_diff: float | None
    reference_logit_diff: float | None
    patched_answer_prob: float | None
    receiver_answer_prob: float | None
    flipped: bool


class SiteDetail(_Out):
    run_id: str
    site: SiteSummary
    receiver: Literal["clean", "corrupt"]
    reference: Literal["clean", "corrupt"]
    metric: dict[str, Any]
    statistics: dict[str, Any]
    prompts: list[PromptEvidence]
    strongest: list[int]
    weakest: list[int]
    dataset_changed: bool
    dataset_available: bool


class ComparisonChange(_Out):
    label: str
    kind: str
    layer: int
    head: int | None
    feature: int | None = None
    position_key: str
    variant_key: str | None = None
    index_a: int
    index_b: int
    row: int
    col: int
    effect_a: dict[str, float | None]
    effect_b: dict[str, float | None]
    rank_a: int
    rank_b: int
    flags: list[Literal["sign", "left_top", "entered_top"]]


class DiffCell(_Out):
    index_a: int
    row: int
    col: int
    value: float


class SpecDifference(_Out):
    path: str
    a: Any
    b: Any


class Comparison(_Out):
    run_a: str
    run_b: str
    n_common: int
    n_a: int
    n_b: int
    spearman: float | None
    top_k: int
    top_overlap: int
    top_a: list[str]
    top_b: list[str]
    changes: list[ComparisonChange]
    flagged: list[int]
    n_sign_changes: int
    diff: list[DiffCell]
    same_layout: bool
    spec_differences: list[SpecDifference]
