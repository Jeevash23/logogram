"""Schemas for what Logogram writes and sends: run files, listings and live events.

The experiment spec lives in :mod:`logogram.spec`; request bodies are defined next to their
endpoints. Everything here is validated when it is produced, so files on disk and messages to
the web app always have these shapes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    kind: Literal["heads", "layer_position", "layer_components", "sites", "steering", "site_sets"]
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


class SteeringInfo(_Model):
    """Which prompts trained the steering directions and which measured them, and how long each
    site's direction is (one per row of the layout)."""

    train: list[int]
    test: list[int]
    norms: list[float]
    control: list[ControlComparison] | None = None


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
    # A run that stopped before the end: what it kept. None for a finished run.
    partial: PartialRun | None = None


# -- manifest.json -----------------------------------------------------------------------------


class ModelProvenance(_Model):
    id: str
    revision: str | None
    architecture: str
    process_weights: bool
    backend: str
    backend_version: str
    # Whether the model is made for chat (prompts ran as plain text, without its chat template),
    # whether its tokenizer has a chat template, and why it counts as made for chat.
    chat: dict[str, Any] | None = None


class DeviceInfo(_Model):
    type: str
    name: str
    cuda: str | None
    threads: int
    deterministic_algorithms: bool


class DatasetProvenance(_Model):
    path: str
    sha256: str | None
    n: int
    source_path: str | None = None


class DerivedFrom(_Model):
    run: str
    # robustness: one methodological choice changed; rerun: the same spec again; verification:
    # the strongest estimated sites of an attribution patching run, patched for real.
    kind: Literal["robustness", "rerun", "verification"]
    change: str | None = None


class Manifest(_Model):
    logogram_manifest: Literal[1] = 1
    run_id: str
    status: Literal["running", "finished", "failed", "cancelled"]
    started_at: str
    finished_at: str | None = None
    wall_time_s: float | None = None
    error: str | None = None
    versions: dict[str, str | None]
    derived_from: DerivedFrom | None = None
    model: ModelProvenance | None = None
    device: DeviceInfo | None = None
    dtype: str | None = None
    dataset: DatasetProvenance | None = None
    sae: dict[str, Any] | None = None


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
    derived_from: DerivedFrom | None
    project_session: str
    # Seconds since the epoch, like a job's start.
    queued_at: float


class QueueChanged(_Model):
    """The whole queue, first in line first, after any change."""

    project_session: str | None
    runs: list[QueuedRunInfo]


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
    derived_from: DerivedFrom | None
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
