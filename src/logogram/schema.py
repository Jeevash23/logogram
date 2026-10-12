"""Schemas for what Logogram writes and sends: run files, listings and live events.

The experiment spec lives in :mod:`logogram.spec`; request bodies are defined next to their
endpoints. Everything here is validated when it is produced, so files on disk and messages to
the web app always have these shapes.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Model(BaseModel):
    # protected_namespaces: pydantic before 2.10 warns about a field named model_id otherwise.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


# -- per-site statistics -----------------------------------------------------------------------


class Stat(_Model):
    mean: float | None
    sd: float | None
    lo: float | None
    hi: float | None


class SiteRef(_Model):
    index: int
    kind: Literal[
        "resid_pre",
        "resid_mid",
        "resid_post",
        "attn_out",
        "mlp_out",
        "head",
        "sae_feature",
        "site_set",
        "edge",
    ]
    layer: int
    head: int | None
    feature: int | None = None
    position: dict[str, Any]
    position_key: str
    row: int
    col: int
    label: str
    # A variant of the site, such as a steering strength; None for one measurement per site.
    variant: dict[str, Any] | None = None
    variant_key: str | None = None
    # The sites of a set intervened on together (kind site_set).
    members: list[dict[str, Any]] | None = None
    # An edge of the model's graph (kind edge): its sender and receiver, as the spec writes them.
    edge: dict[str, Any] | None = None


class Band(_Model):
    lo: float
    hi: float


class SiteSummary(SiteRef):
    n: int
    effect: Stat
    delta: Stat
    # The spec's metric, patched (older runs measured the logit difference only).
    patched_metric: float | None = None
    # log P(answer) - log P(distractor), patched: the logit difference for single tokens.
    patched_logit_diff: float | None
    answer_prob: float | None
    answer_prob_delta: float | None
    sign_flips: int
    opposite_sign: int
    # Corrected for the number of sites: a band that holds for all sites together, and the
    # Benjamini-Hochberg q-value. Only in a finished run's summary.
    band: Band | None = None
    q: float | None = None


class LayoutAxis(BaseModel):
    model_config = ConfigDict(extra="allow")

    key: str
    label: str


class Layout(_Model):
    kind: Literal[
        "heads", "layer_position", "layer_components", "sites", "steering", "site_sets", "edges"
    ]
    site: str | None = None
    row_title: str
    col_title: str
    rows: list[LayoutAxis]
    cols: list[LayoutAxis]
    # Sets of sites: the components a complement set replaces.
    universe: list[str] | None = None


# -- summary.json ------------------------------------------------------------------------------


class GroupStats(_Model):
    mean: float | None
    sd: float | None


class PromptSetStats(_Model):
    # log P(answer) - log P(distractor): the logit difference for single tokens.
    logit_diff: GroupStats
    answer_prob: GroupStats
    prefers_answer: int
    metric: GroupStats | None = None


class BaselineSummary(_Model):
    clean: PromptSetStats
    corrupt: PromptSetStats
    gap: GroupStats


class MetricInfo(_Model):
    kind: str
    target: Literal["clean", "corrupt"] | None = None
    label: str | None = None
    normalization: Literal["dataset_gap", "prompt_gap"]
    denominator: float | None
    description: str
    normalized_effect: str


class StatisticsInfo(_Model):
    bootstrap: int
    ci: float
    seed: int
    method: str
    cluster: str | None = None
    clusters: int | None = None
    # The two-sided tail each site's simultaneous band keeps, and what the bands and q-values mean.
    band_level: float | None = None
    multiple_comparisons: str | None = None


class ModelShape(_Model):
    id: str
    n_layers: int
    n_heads: int
    d_model: int | None = None
    d_head: int | None = None
    d_mlp: int | None = None
    site_kinds: list[str] | None = None
    block_structure: str | None = None
    normalization: str | None = None
    activation: str | None = None


class DirectSplit(_Model):
    """Direct logit attribution: the mean logit difference and the parts it splits into."""

    prompts: Literal["clean", "corrupt"]
    logit_diff: float | None
    embeddings: float | None
    attention: float | None
    mlp: float | None
    biases: float | None


class ControlComparison(_Model):
    """A steered site and strength against its random control, from the same resamples: the
    difference of their effects' magnitudes, |direction| - |control|."""

    row: int
    coefficient: float
    index: int
    control_index: int
    difference: float | None
    lo: float | None
    hi: float | None
    beats_control: bool


class SAESource(_Model):
    repo: str
    path: str
    revision: str


class SteeringFeature(_Model):
    """Steering along an SAE feature: the feature, its SAE, where the SAE writes (the site its
    decoder direction is added to: ``site`` of ``layer``, the heads' outputs z for "head"), and
    what strength 1 is at each row of the layout: the feature's typical activation, the mean of
    its positive activations at the row's token of the training pairs' prompts, clean and
    corrupt (``fired`` of the ``measured`` activations were positive)."""

    feature: int
    sae: SAESource
    site: str
    layer: int
    transcoder: bool
    # Whether the SAE standardizes its inputs, so each prompt's vector is also scaled by the
    # standard deviation of its own input at the token.
    standardized: bool
    activations: list[float]
    fired: list[int]
    measured: list[int]


class SteeringInfo(_Model):
    """Which prompts trained the steering directions and which measured them, and how long each
    site's vector is at strength 1 (one per row of the layout)."""

    train: list[int]
    test: list[int]
    norms: list[float]
    control: list[ControlComparison] | None = None
    # Along an SAE feature; None for the mean difference.
    feature: SteeringFeature | None = None


class FeatureRun(_Model):
    """Runs on SAE features: the SAE, how well it fits these prompts, and for attribution
    patching how much of the site's estimated effect its features account for."""

    sae: dict[str, Any]
    fit: dict[str, Any]
    site_estimate: float | None = None
    features_estimate: float | None = None
    # Through a transcoder's skip connection (input · W_skip), which no feature carries.
    skip_estimate: float | None = None
    evaluated: int | None = None
    # Every feature, chosen on some prompts and reported on the others: their dataset indices.
    chosen_on: list[int] | None = None
    reported_on: list[int] | None = None


class CircuitWithout(_Model):
    """A set that keeps one site fewer than another: how much faithfulness that site adds."""

    of: str  # the set this one is, less one site
    site: str  # the site it leaves out
    drop: Stat  # faithfulness(of) - faithfulness(this set)


class CircuitInteraction(_Model):
    """A set of two sites, each also intervened on alone: what intervening on both does beyond
    the sum of the two, normalized like the effects."""

    a: str  # the sets holding each site alone
    b: str
    effect: Stat  # effect(both) - effect(a) - effect(b)


class CircuitRow(_Model):
    """One set of a site_sets run against the set that replaces the whole universe: its share of
    that set's effect, and for sets that keep their sites (complements), the faithfulness:
    1 - share, the part of the model's behavior the kept sites carry alone."""

    index: int
    label: str
    complement: bool
    size: int
    share: Stat | None = None
    faithfulness: Stat | None = None
    without: CircuitWithout | None = None
    interaction: CircuitInteraction | None = None


class CircuitInfo(_Model):
    universe: list[str] | None
    # The set that replaces the whole universe, if the scope has one (the "everything" set).
    everything: int | None
    rows: list[CircuitRow] = []


class AttributionInfo(_Model):
    """How attribution patching estimated: one gradient, or integrated gradients over steps."""

    method: Literal["gradient", "integrated_gradients"]
    steps: int | None = None


class EdgeCurvePoint(_Model):
    """One point of a faithfulness curve: k edges, and how much of the behavior they carry, from
    the run's resamples. ``metric`` is the mean metric of the patched runs; for the random control,
    ``draws`` holds each draw's mean faithfulness (the point averages them prompt by prompt)."""

    k: int
    faithfulness: Stat
    metric: float | None
    draws: list[float | None] | None = None


class EdgeCurve(_Model):
    """The k strongest edges as a circuit, for each k, beside edges chosen at random: see
    logogram.edges. ``every`` patches every edge of the scope (the end of the curve that patches
    most), and ``source_difference`` is the largest difference, over prompts, between its metric
    and the source prompt's when the scope is the whole graph (it measures the source run then)."""

    patch: Literal["chosen", "rest"]
    counts: list[int]
    dropped: list[int] = []
    seed: int
    draws: int
    strongest: list[EdgeCurvePoint]
    random: list[EdgeCurvePoint]
    every: EdgeCurvePoint
    source_difference: float | None = None
    # The patched metric per prompt: one list per point, in the order of ``counts``.
    per_prompt: dict[str, list[list[float | None]]]


class EdgeRun(_Model):
    """An edge attribution run: the graph, how the estimates were taken, what the model was
    measured to do that they rely on, what the run cost, and the faithfulness curve. The summary
    lists the ``reported`` strongest edges; edges.parquet beside it holds every edge's
    statistics, and results.parquet every edge's per-prompt values."""

    method: Literal["gradient", "integrated_gradients"]
    steps: int | None = None
    n_edges: int
    n_senders: int
    n_receivers: int
    reported: int
    structure: Literal["sequential", "parallel"]
    gradient: str
    checks: dict[str, float]
    cost: dict[str, Any]
    curve: EdgeCurve | None = None


class ChatTemplateInfo(_Model):
    """How the prompts were wrapped in the model's own chat template (logogram.chat): as the
    person's message, or as the start of the model's reply to a fixed message; the template's
    SHA-256, and its text before and after each prompt."""

    kind: Literal["user_turn", "assistant_prefix"]
    system: str | None
    user: str | None
    sha256: str
    before: str
    after: str


class PartialRun(_Model):
    """What a run that stopped before the end (cancelled, or failed mid-sweep) kept.

    A sweep that goes layer by layer (or site by site, or set by set) keeps the sites at which
    every prompt ran: ``unit`` is layers, sites or sets. A method that measures every site in one
    pass per batch of prompts keeps every site, over the prompts that finished: ``unit`` is
    prompts. Statistics come from the kept rows exactly as for a finished run (see
    logogram.partial)."""

    status: Literal["cancelled", "failed"]
    unit: Literal["layers", "sites", "sets", "prompts"]
    done: int = Field(ge=1)
    total: int = Field(ge=1)
    # The sites the summary reports and the sites the sweep has; the prompts measured at each.
    sites: int = Field(ge=1)
    sites_total: int = Field(ge=1)
    prompts: int = Field(ge=1)
    prompts_total: int = Field(ge=1)
    # "Stopped after 7 of 12 layers: these are the layers that finished."
    note: str
    # What the run doesn't keep, or keeps over the finished part only, in sentences.
    left_out: list[str] = []


def stopped_text(partial: PartialRun | dict[str, Any]) -> str:
    """How far a run that stopped early got, in words: "stopped after 7 of 12 layers"."""
    data = partial.model_dump() if isinstance(partial, PartialRun) else partial
    total, unit = int(data["total"]), str(data["unit"])
    return f"stopped after {int(data['done']):,} of {total:,} {unit if total != 1 else unit[:-1]}"


class Summary(_Model):
    logogram_summary: Literal[1] = 1
    run_id: str
    name: str
    description: str
    model: ModelShape | None
    n_prompts: int
    n_sites: int
    layout: Layout
    receiver: Literal["clean", "corrupt"]
    reference: Literal["clean", "corrupt"]
    # What each per-prompt value is: a patched run, an estimate of one, or a term of a split.
    measure: Literal["intervention", "estimate", "attribution"] = "intervention"
    metric: MetricInfo
    statistics: StatisticsInfo
    baseline: BaselineSummary
    sites: list[SiteSummary]
    per_prompt: dict[str, list[float | None]]
    donors: list[list[int]] | None
    warnings: list[str]
    direct: DirectSplit | None = None
    steering: SteeringInfo | None = None
    features: FeatureRun | None = None
    circuit: CircuitInfo | None = None
    attribution: AttributionInfo | None = None
    edges: EdgeRun | None = None
    chat_template: ChatTemplateInfo | None = None
    # A run that stopped before the end: what it kept. None for a finished run.
    partial: PartialRun | None = None


# -- manifest.json -----------------------------------------------------------------------------


class ModelProvenance(_Model):
    id: str
    revision: str | None
    # A local model (logogram.local_models), by the hash of its files; never its folder.
    source: Literal["hub", "local"] | None = None
    sha256: str | None = None
    architecture: str
    process_weights: bool
    backend: str
    backend_version: str
    # Whether the model is made for chat (prompts ran as plain text, without its chat template),
    # whether its tokenizer has a chat template, and why it counts as made for chat.
    chat: dict[str, Any] | None = None


class DeviceInfo(_Model):
    type: str
    # Which of the machine's GPUs, for a model on a GPU (as cuda:N names it).
    index: int | None = None
    name: str
    cuda: str | None
    threads: int
    deterministic_algorithms: bool


class DatasetProvenance(_Model):
    path: str
    sha256: str | None
    n: int
    source_path: str | None = None


class ChatTemplateProvenance(_Model):
    kind: Literal["user_turn", "assistant_prefix"]
    sha256: str


class DerivedFrom(_Model):
    run: str
    # robustness: one methodological choice changed; rerun: the same spec again; verification:
    # the strongest estimated sites of an attribution patching run, patched for real.
    kind: Literal["robustness", "rerun", "verification"]
    change: str | None = None


SweepValue = Annotated[int, Field(strict=True)] | Annotated[str, Field(min_length=1)]


class SweepMember(_Model):
    """A run of a sweep: the same spec run once for each of several values of one field (see
    logogram.sweeps). Every run of the sweep records it whole, so any of them tells the rest."""

    kind: Literal["sweep"]
    # The sweep's id, which every run of it shares.
    sweep: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
    # The field the sweep varies, dotted as in the spec ("statistics.seed"), and this run's value.
    field: str = Field(min_length=1, max_length=64)
    value: SweepValue
    # This run's place among the sweep's values, and every value in the order given.
    index: int = Field(ge=0)
    of: int = Field(ge=2)
    values: list[SweepValue]
    # The run the sweep was made from (None for a sweep of a spec), and the value it stands in
    # for, when that value is one of the sweep's: it isn't run again.
    run: str | None = None
    original: int | None = Field(default=None, ge=0)
    # This run's value in words, such as "bootstrap seed 2".
    change: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> SweepMember:
        if len(self.values) != self.of or self.index >= self.of:
            raise ValueError("a sweep's values must number `of`, and its index be one of them")
        if self.values[self.index] != self.value:
            raise ValueError("a sweep run's value must be its sweep's value at its index")
        if self.original is not None and (
            self.run is None or self.original >= self.of or self.original == self.index
        ):
            raise ValueError(
                "the value a sweep's original run stands in for is another value of the sweep, "
                "and needs that run"
            )
        return self


# Where a run came from: another run, or a sweep (told apart by ``kind``).
Derivation = Annotated[DerivedFrom | SweepMember, Field(discriminator="kind")]


class Manifest(_Model):
    logogram_manifest: Literal[1] = 1
    run_id: str
    status: Literal["running", "finished", "failed", "cancelled"]
    started_at: str
    finished_at: str | None = None
    wall_time_s: float | None = None
    error: str | None = None
    versions: dict[str, str | None]
    derived_from: Derivation | None = None
    model: ModelProvenance | None = None
    device: DeviceInfo | None = None
    dtype: str | None = None
    dataset: DatasetProvenance | None = None
    sae: dict[str, Any] | None = None
    # The chat template the prompts were wrapped in: how, and its SHA-256.
    chat_template: ChatTemplateProvenance | None = None


# -- continuations.json ------------------------------------------------------------------------

CONTINUATIONS_FILE = "continuations.json"


class GeneratedTokens(_Model):
    """Tokens a model generated: their ids, and each as its tokenizer writes it."""

    ids: list[int]
    tokens: list[str]


class SteeredContinuation(GeneratedTokens):
    """A continuation with one row's vector (a steered site of the layout) at one strength, or
    along its random control. ``key`` names the variant as the results do, such as "×2"."""

    row: int
    coefficient: float
    control: bool
    key: str


class PromptContinuations(_Model):
    """One held-out prompt continued: the prompt steered (the receiver) as written and as the
    model read it, its continuation without steering, and one per row and variant."""

    index: int  # the prompt's index in the dataset
    text: str
    prompt: list[str]
    unsteered: GeneratedTokens
    steered: list[SteeredContinuation]


class ContinuationRow(_Model):
    row: int
    label: str


class ContinuationsFile(_Model):
    """continuations.json: what a steered model goes on to say. Greedy decoding (the most likely
    token each time, ties to the lowest token id) of ``tokens`` new tokens after the first
    held-out prompts, in split order: without steering, and with each row's vector at each
    strength (and its random control). Each forward pass adds the vector at the steered site and
    token of the prompt, as when it was measured; the generated tokens aren't steered."""

    logogram_continuations: Literal[1] = 1
    run_id: str
    receiver: Literal["clean", "corrupt"]
    tokens: int = Field(ge=1)
    decoding: Literal["greedy"]
    rows: list[ContinuationRow]
    prompts: list[PromptContinuations]


# -- owner.json --------------------------------------------------------------------------------


class RunOwner(_Model):
    """The process running a run, kept in its folder while it runs (see logogram.owner). A
    running manifest whose owner is gone is a run that was interrupted."""

    logogram_owner: Literal[1] = 1
    pid: int = Field(ge=1, lt=2**32, strict=True)
    # The process's start time, in seconds since the epoch, as psutil reports it: with the pid,
    # it tells the process apart from a later one that reuses the pid.
    started: float = Field(gt=0, allow_inf_nan=False)
    # Random, so the run removes only its own record.
    token: str = Field(pattern=r"^[0-9a-f]{32}$")


# -- .logogram/queue.json and queue-runner.json ------------------------------------------------


class QueueEntry(_Model):
    """A run waiting in a project's queue (see logogram.runqueue)."""

    id: str = Field(pattern=r"^[0-9a-f]{12}$")
    # When it was added, in UTC (ISO 8601).
    queued_at: str = Field(pattern=r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
    # The spec as it was queued; checked again when its turn comes.
    spec: dict[str, Any]
    # A saved draft it runs in (its folder), and where it comes from (a rerun, a check, a sweep).
    draft_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    derived_from: Derivation | None = None


class QueueFile(_Model):
    """A project's run queue, first in line first."""

    logogram_queue: Literal[1] = 1
    runs: list[QueueEntry] = Field(default_factory=list, max_length=10_000)


class QueueRunner(_Model):
    """The process running a project's queue: the app, or ``logogram queue run``. Like a run's
    owner record (RunOwner): a record whose process has ended holds nothing."""

    logogram_queue_runner: Literal[1] = 1
    program: Literal["app", "cli"]
    pid: int = Field(ge=1, lt=2**32, strict=True)
    started: float = Field(gt=0, allow_inf_nan=False)
    token: str = Field(pattern=r"^[0-9a-f]{32}$")


# -- annotations.json --------------------------------------------------------------------------


class RunAnnotations(_Model):
    """What a person says about a run, kept beside its spec, which records what ran and never
    changes once it has run (see logogram.history)."""

    # Strict: a hand-edited "starred": "yes" is a mistake to point out, not a yes.
    model_config = ConfigDict(extra="forbid", strict=True)

    logogram_annotations: Literal[1] = 1
    # Shown instead of the spec's name; None shows the spec's name.
    title: str | None = Field(default=None, min_length=1, max_length=200)
    starred: bool = False
    # Kept out of the history unless it is asked to show archived runs.
    archived: bool = False

    @field_validator("title")
    @classmethod
    def _one_line(cls, title: str | None) -> str | None:
        if title is not None and any(ord(c) < 32 or ord(c) == 127 for c in title):
            raise ValueError("a title is one line of text, without control characters")
        return title


# -- live events (WebSocket) -------------------------------------------------------------------


class RunStarted(_Model):
    run_id: str
    n_prompts: int
    model: ModelShape
    layout: Layout
    sites: list[SiteRef]


class RunLayer(_Model):
    run_id: str
    layer: int
    sites: list[SiteSummary]


class RunProgress(_Model):
    run_id: str
    done: int
    total: int
    layer: int
    memory: int | None
    elapsed_s: float


class RunEnded(_Model):
    run_id: str
    error: str | None = None
    # A cancelled or failed run that kept what finished.
    partial: PartialRun | None = None


class QueuedRunInfo(_Model):
    """A run waiting for the job in progress to end, in the open project's queue."""

    id: str
    # The spec's name.
    title: str
    spec: dict[str, Any]
    draft_id: str | None
    derived_from: Derivation | None
    project_session: str
    # Seconds since the epoch, like a job's start.
    queued_at: float


QueueState = Literal["idle", "waiting", "running", "paused"]


class QueueChanged(_Model):
    """The whole queue, first in line first, after any change, and how it stands: ``waiting``
    (its next run starts when the job in progress ends), ``running`` (one of its runs is in
    progress, here or in another process, and the next follows), ``paused`` (nothing starts until
    it is started: a queue that comes back with its project is paused) or ``idle`` (empty)."""

    project_session: str | None
    runs: list[QueuedRunInfo]
    state: QueueState = "idle"
    # Who runs it, when another process does: "`logogram queue run` (process 1234)".
    runner: str | None = None
    # Why the project's saved queue can't be read, and how to fix it.
    error: str | None = None


class QueueSkipped(_Model):
    """A queued run that couldn't start when its turn came, and left the queue."""

    project_session: str
    id: str
    title: str
    error: str


class RunsChanged(_Model):
    """A run renamed, starred, archived or deleted: tabs showing its project read the history
    again (not a "run." event: those are replayed to tabs that connect mid-run)."""

    project_session: str
    run_id: str
    deleted: bool = False


EVENT_SCHEMAS: dict[str, type[BaseModel]] = {
    "run.started": RunStarted,
    "run.layer": RunLayer,
    "run.progress": RunProgress,
    "run.finished": RunEnded,
    "run.failed": RunEnded,
    "run.cancelled": RunEnded,
    "queue": QueueChanged,
    "queue.skipped": QueueSkipped,
    "runs.changed": RunsChanged,
}


def validate_event(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    """Check an event against its schema before it is sent."""
    schema = EVENT_SCHEMAS.get(kind)
    if schema is None:
        return data
    return schema.model_validate(data).model_dump(mode="json")


# -- history -----------------------------------------------------------------------------------


class RunListing(_Model):
    id: str
    name: str
    description: str
    status: Literal["finished", "failed", "cancelled", "draft", "running"]
    created: str | None
    finished: str | None
    wall_time_s: float | None
    n_prompts: int | None
    layout_kind: str | None
    model_id: str
    dataset: str
    experiment: dict[str, Any]
    scope: dict[str, Any]
    derived_from: Derivation | None
    error: str | None
    # Per layer, the largest normalized effect (signed): what writes the run's logogram in the app.
    profile: list[float] | None = None
    # A running run whose process is another one (logogram run, a notebook, another server): its
    # progress isn't streamed here, so the app reads the listing again to see it end.
    elsewhere: bool = False
    # The dataset the run read, as it was named when it ran (``dataset`` is then its snapshot).
    dataset_source: str | None = None
    # What a person says about the run (its annotations.json): a title shown instead of the spec's
    # name, a star, archived; and why the file was ignored, if it can't be read.
    title: str | None = None
    starred: bool = False
    archived: bool = False
    annotations_error: str | None = None
    # The run's spec can't be read: error says what is wrong and how to fix it. It can be deleted.
    damaged: bool = False
    # A cancelled or failed run that kept what finished (its summary's partial).
    partial: PartialRun | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
