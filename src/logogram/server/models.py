"""Response models for the JSON API. Request bodies are defined beside their routes in app.py."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from logogram.schema import (
    ContinuationsFile,
    GeneratedTokens,
    PartialRun,
    QueueChanged,
    QueuedRunInfo,
    RunListing,
    SiteSummary,
)
from logogram.spec import ModelRef, PredictionSettings


class _Out(BaseModel):
    # protected_namespaces: pydantic before 2.10 warns about a field named model_id otherwise.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


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
    # Hugging Face ("hub"), or a folder on this machine ("local", by the hash of its files).
    source: Literal["hub", "local"] = "hub"
    sha256: str | None = None


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
    kind: Literal["load_model", "load_sae", "run", "add_model"]
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


class QueueAdded(_Out):
    """A run sent to the queue. With nothing running it started at once (``run_id`` and ``job``);
    otherwise it waits its turn (``queued``). ``queue`` lists every run still waiting."""

    run_id: str | None
    job: JobInfo | None
    queued: QueuedRunInfo | None
    queue: list[QueuedRunInfo]


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


class ExampleInfo(_Out):
    """A bundled example (logogram.gallery): what it asks, the result to expect and its spec."""

    id: str
    name: str
    question: str
    expect: str
    source: str
    # Where its copy goes in the projects folder, and whether one is there already.
    path: str
    copied: bool
    experiment: str
    n_prompts: int
    spec: dict[str, Any]


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
    single_key_shortcuts: bool = True
    projects_parent: str
    update: UpdateStatus
    sae: dict[str, Any] = {"state": "none"}
    # The runs waiting for the job in progress, first in line first.
    queue: list[QueuedRunInfo] = []
    # The same runs, and how the queue stands (waiting, running, paused), and who runs it.
    queue_status: QueueChanged | None = None


class Settings(BaseModel):
    model_config = ConfigDict(extra="allow")

    system_check_seen: bool | None = None
    theme: str | None = None
    update_check: bool | None = None
    single_key_shortcuts: bool | None = None


class OkOut(_Out):
    ok: bool


class ReportRow(_Out):
    label: str
    value: str


class ProblemReport(_Out):
    """A problem report (logogram.support): its rows, and the same as text to paste."""

    rows: list[ReportRow]
    hardware: bool
    text: str
    new_issue_url: str


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
    # With a model loaded: the prompts whose answer or distractor it reads as a continuation of
    # several tokens (dataset indices), which the logit difference can't score.
    continuations: list[int] | None = None


class DatasetCreated(_Out):
    name: str
    path: str
    n: int


class IOITemplateOut(_Out):
    id: str
    text: str
    default: bool


class TaskOptionOut(_Out):
    name: str
    type: Literal["bool", "choice", "choices"]
    default: Any
    description: str
    allowed: list[str]


class TaskMetricWhen(_Out):
    option: str
    value: Any
    metric: str


class TaskOut(_Out):
    """A task Logogram generates prompts for (logogram.tasks)."""

    id: str
    name: str
    description: str
    # The metric that reads its answers, and the options that change it.
    metric: str
    metric_when: list[TaskMetricWhen]
    options: list[TaskOptionOut]
    templates: list[IOITemplateOut]


class TaskDatasetCreated(DatasetCreated):
    # The metric that reads the dataset's answers.
    metric: str


class TokenRow(_Out):
    tokens: list[str]
    ids: list[int]


class AnswerTokens(_Out):
    text: str
    tokens: list[str]
    id: int | None
    # A set of single tokens, any of which counts (its tokens are the members).
    alternatives: bool = False


class TokenStrip(_Out):
    clean: TokenRow
    corrupt: TokenRow
    # With a chat template: the clean tokens that hold the prompt (first, one past the last);
    # the others are the template's.
    prompt: list[int] | None = None
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
    # log P(answer) - log P(distractor): the logit difference for single tokens.
    clean_logit_diff: float | None
    corrupt_logit_diff: float | None
    clean_answer_prob: float | None
    corrupt_answer_prob: float | None
    clean_metric: float | None = None
    corrupt_metric: float | None = None
    clean_top: list[TopToken]
    corrupt_top: list[TopToken]


class BaselineMetric(_Out):
    kind: str
    target: Literal["clean", "corrupt"] | None
    label: str
    description: str
    clean: float | None
    corrupt: float | None
    gap: float | None


class BaselineSummaryOut(_Out):
    clean_logit_diff: float | None
    corrupt_logit_diff: float | None
    gap: float | None
    clean_prefers_answer: int
    corrupt_prefers_answer: int
    clean_answer_prob: float | None
    corrupt_answer_prob: float | None
    metric: BaselineMetric | None = None


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


class HeadKindScores(_Out):
    description: str
    # [layer][head]: the mean over sequences and its bootstrap interval.
    mean: list[list[float]]
    lo: list[list[float]]
    hi: list[list[float]]
    # [sequence][layer][head]: the values the statistics come from.
    per_sequence: list[list[list[float]]]


class TopHead(_Out):
    layer: int
    head: int
    mean: float
    lo: float
    hi: float


class HeadKinds(_Out):
    settings: dict[str, Any]
    n_layers: int
    n_heads: int
    kinds: dict[Literal["previous_token", "duplicate_token", "induction"], HeadKindScores]
    top: dict[Literal["previous_token", "duplicate_token", "induction"], list[TopHead]]


class MethodsText(_Out):
    text: str


class ChatWrapped(_Out):
    """A prompt in the model's own chat template: the whole text, where the prompt is in it
    (characters; None when the template changes the prompt, as ``problem`` says), the template's
    text before and after a prompt, and its SHA-256."""

    text: str
    start: int | None
    end: int | None
    before: str
    after: str
    sha256: str
    problem: str | None


class LocalModelOut(_Out):
    """A model added from a folder on this machine. The folder is shown on this machine only;
    specs and runs name the model by its name and the hash of its files."""

    name: str
    folder: str
    sha256: str
    added: str
    size: int
    files: int


class LookToken(_Out):
    id: int
    token: str
    prob: float


class LookLayer(_Out):
    layer: int
    top: list[LookToken]


class LookReport(_Out):
    tokens: list[str]
    ids: list[int]
    prepend_bos: bool
    # The chat template the text was wrapped in, if any, and the tokens that hold the text
    # itself (first, one past the last).
    chat_template: dict[str, Any] | None = None
    text_tokens: list[int] | None = None
    top_k: int
    position: int
    # After each token: the model's most likely next tokens.
    predictions: list[list[LookToken]]
    # At ``position``: what each layer's residual stream would predict, when the lens holds.
    lens: list[LookLayer] | None
    lens_note: str | None


class LookAttention(_Out):
    tokens: list[str]
    layer: int
    head: int
    pattern: list[list[float]]


class LookFeature(_Out):
    feature: int
    activation: float


class LookFeatures(_Out):
    tokens: list[str]
    features: list[list[LookFeature]]
    active: list[int]
    first_real_token: int


class LookFeatureAlong(_Out):
    """One SAE feature along the text: its activation at each token, and its largest from the
    first real token on."""

    tokens: list[str]
    feature: int
    activations: list[float]
    largest: float
    position: int
    first_real_token: int


class LookSteerOut(_Out):
    """The feature added while the text continued: how much of it, from which token, where the
    SAE writes, and its largest activation on the text for reference."""

    feature: int
    activation: float
    start: int
    site: str
    layer: int
    where: str
    largest: float
    largest_at: int


class LookContinuation(_Out):
    """What the model writes after the text: greedily, and with a feature added if asked."""

    tokens: list[str]
    ids: list[int]
    prepend_bos: bool
    new: int
    plain: GeneratedTokens
    steer: LookSteerOut | None
    steered: GeneratedTokens | None


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
    # How the lens reads the answer: one token, a set (summed), or a continuation's first token.
    answer_reading: Literal["token", "set", "first_token"] = "token"
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
    # What gradient methods and path patching hold: every layer's activations at once.
    all_layers: int = 0
    all_layers_verdict: Literal["fits", "tight", "wont_fit"] = "fits"


class RunMemoryOut(_Out):
    device: str
    batch_size: int
    seq_len: int
    # Whether the method holds every layer's activations at once.
    all_layers: bool
    activations: int
    total: int
    available: int
    verdict: Literal["fits", "tight", "wont_fit"]
    explanation: str


class EstimateOut(_Out):
    id: str
    # None for a local model, which has no revisions.
    revision: str | None
    architecture: ArchitectureOut
    download_bytes: int
    total_bytes: int
    gated: bool
    # Whether TransformerLens can load this architecture, and if not, why (before downloading).
    supported: bool
    support_note: str | None
    estimate: MemoryEstimateOut


# -- published SAEs ----------------------------------------------------------------------------


class CatalogueFitOut(_Out):
    model: str
    process_weights: bool
    tokens: int
    variance_explained: float
    l0: float


class CatalogueCheckOut(_Out):
    revision: str
    read: Literal["header", "weights"]
    fit: CatalogueFitOut | None


class CatalogueSAEOut(_Out):
    """One SAE of SAELens's registry, as the bundled catalogue describes it."""

    release: str
    id: str
    model: str
    hf_model: str
    repo: str
    folder: str
    # The commit whose files the catalogue read; loading pins the repository's current one.
    revision: str | None
    licence: str | None
    gated: bool
    hook: str | None
    site: str | None
    layer: int | None
    width: int | None
    architecture: str | None
    kind: Literal["sae", "transcoder", "skip_transcoder"]
    reads: Literal["exact", "refused", "unknown"]
    reason: str | None
    l0: float | None
    verified: CatalogueCheckOut | None


class CatalogueOut(_Out):
    model: str
    # Where the catalogue came from: SAELens's registry, in this version of sae-lens.
    source: str
    total: int
    exact: int
    matches: int
    saes: list[CatalogueSAEOut]


# -- runs --------------------------------------------------------------------------------------


class RunDetail(_Out):
    """A run's files as stored. They were validated when written; older files are passed through."""

    id: str
    spec: dict[str, Any] | None
    summary: dict[str, Any] | None
    manifest: dict[str, Any] | None
    predictions: PredictionReport | None = None
    # A steering run's continuations.json: what the steered model went on to say.
    continuations: ContinuationsFile | None = None
    listing: RunListing | None
    folder: str
    # A failed run's traceback, as kept in its folder (with the machine's paths scrubbed).
    traceback: str | None = None


class PromptEvidence(_Out):
    index: int
    clean: str | None
    corrupt: str | None
    answer: str | None
    distractor: str | None
    effect: float | None
    delta: float | None
    patched_metric: float | None = None
    receiver_metric: float | None = None
    reference_metric: float | None = None
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
    flags: list[Literal["sign", "left_top", "entered_top", "differs"]]
    # b - a in per-prompt effects, when both runs measured the same prompts.
    difference: dict[str, float] | None = None


class DiffCell(_Out):
    index_a: int
    row: int
    col: int
    value: float


class SpecDifference(_Out):
    path: str
    a: Any
    b: Any


class EnvironmentDifference(_Out):
    """Something the two runs' manifests record differently about where they ran."""

    key: str
    label: str
    a: str
    b: str


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
    paired: bool = False
    n_differs: int = 0
    # Versions or devices that differ: these alone can change results in the last digits.
    environment_differences: list[EnvironmentDifference] = []
    # A run that stopped early: what it kept (only the sites both runs have are compared).
    partial_a: PartialRun | None = None
    partial_b: PartialRun | None = None


class EnvironmentOut(_Out):
    """What the server's numbers can depend on beyond a spec (logogram.system.environment)."""

    logogram: str
    python: str
    implementation: str
    os: str
    architecture: str
    versions: dict[str, str | None]
    torch_cuda: str | None
    devices: list[str]
    default_device: str


# -- sweeps ------------------------------------------------------------------------------------

# A swept value: a seed, or a dataset's path or a revision as given.
SweepValueOut = int | str
SweepKind = Literal["seed", "dataset", "revision"]


class SweepFieldOut(_Out):
    """A field a run's spec can be swept over, with the spec's own value and the values the app
    offers at first."""

    field: str
    kind: SweepKind
    label: str
    plural: str
    description: str
    current: SweepValueOut | None
    current_label: str | None
    suggested: list[SweepValueOut]


class SweepPlanRun(_Out):
    index: int
    value: SweepValueOut
    label: str
    # The run's spec name; None where the run the sweep is made from stands in for the value.
    name: str | None
    original: bool


class SweepPlanOut(_Out):
    """The runs a sweep makes, one per value in the order given (logogram.sweeps.plan_sweep)."""

    sweep: str
    field: str
    kind: SweepKind
    label: str
    plural: str
    values: list[SweepValueOut]
    labels: list[str]
    original: str | None
    stands_in: int | None
    runs: list[SweepPlanRun]


class SweepStarted(_Out):
    """A sweep sent to the queue: its first run started at once when nothing was running (``run_id``
    and ``job``); the rest wait their turn in ``queue``."""

    sweep: str
    plan: SweepPlanOut
    run_id: str | None
    job: JobInfo | None
    queue: list[QueuedRunInfo]


class SweepValueRow(_Out):
    index: int
    value: SweepValueOut
    label: str
    run: str | None
    # missing: no run (it never started, or was taken out of the queue); deleted: the run the
    # sweep was made from, which stood in for this value, was deleted.
    status: Literal["finished", "failed", "cancelled", "running", "draft", "missing", "deleted"]
    original: bool
    title: str | None
    error: str | None
    n_prompts: int | None
    # What a revision or a dataset resolved to: the commit the run pinned, the prompts it read.
    revision: str | None
    dataset_sha256: str | None


class SweepPoint(_Out):
    """One site at one value: the run's mean effect, its interval and SD over prompts, and n."""

    mean: float | None
    lo: float | None
    hi: float | None
    sd: float | None
    n: int | None


class SweepSpread(_Out):
    """The spread of a site's means across the values that measured it."""

    mean: float
    sd: float | None
    min: float
    max: float


class SweepSite(_Out):
    key: str
    rank: int
    label: str
    kind: str
    layer: int
    head: int | None
    feature: int | None
    position: dict[str, Any] | None
    position_key: str
    variant: dict[str, Any] | None
    variant_key: str | None
    # One per value; None where the value has no results for this site (never zero).
    points: list[SweepPoint | None]
    measured: int
    spread: SweepSpread | None
    # The median half-width of the runs' own intervals.
    half_width: float | None
    # Values whose interval lies above zero, below it; a pair of values with confident opposite
    # signs; a pair whose intervals exclude each other; values without results.
    above: list[int]
    below: list[int]
    sign: list[int] | None
    apart: list[int] | None
    missing: list[int]
    # Where its intervals exclude zero: at every value that measured it, at some, or at none.
    excludes_zero: Literal["every", "some", "none"]
    verdict: Literal["holds", "size", "sign", "unknown"]
    text: str


class SweepCounts(_Out):
    holds: int
    size: int
    sign: int
    unknown: int


class SweepTable(_Out):
    """A sweep read back (logogram.sweeps.sweep_table)."""

    sweep: str
    field: str
    kind: SweepKind
    label: str
    plural: str
    description: str
    name: str
    created: str | None
    original: str | None
    stands_in: int | None
    values: list[SweepValueRow]
    finished: int
    choose: Literal["first", "all"]
    top: int
    chosen: str
    problem: str | None
    n_sites: int
    sites: list[SweepSite]
    counts: SweepCounts
    headline: str
    spread: str
    notes: list[str]
    ci: float | None
    measure: str | None
    metric: dict[str, Any] | None
    experiment: dict[str, Any] | None


class RevisionOut(_Out):
    name: str
    commit: str
    kind: Literal["branch", "tag"]


class RevisionsOut(_Out):
    """A Hugging Face model's branches and tags, each with its commit."""

    id: str
    revisions: list[RevisionOut]
