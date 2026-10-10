"""The experiment spec: the single description of an experiment that the GUI and CLI both run.

A spec is stored as ``experiments/<id>/spec.json``. Every methodological choice that can change
a number (direction, baseline, position, metric, normalization, seeds, batch size, dtype) lives
here explicitly, so a run can always be traced back to how it was produced.

Version 2 specs state every such choice: no field that can change a number has a default. Version
1 specs (Logogram 0.1) are still read: the fields they leave out take the values version 1 gave
them, and :attr:`Spec.upgraded_fields` lists which, so a run can say so instead of assuming
silently.

Specs refer to abstract sites (``resid_pre``, ``attn_out``, ``head``, ...). Backend-specific hook
names never appear in a spec.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from pydantic.functional_validators import ModelWrapValidatorHandler

NAME_MAX = 200

SPEC_VERSION = 2

# Seeds are drawn into 32-bit generators on some paths: keep every seed in that range.
SEED_MAX = 2**32

StreamSiteKind = Literal["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"]
SiteKind = Literal[
    "resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out", "head", "sae_feature"
]

SITE_KIND_LABELS: dict[str, str] = {
    "resid_pre": "residual stream before the layer",
    "resid_mid": "residual stream between attention and MLP",
    "resid_post": "residual stream after the layer",
    "attn_out": "attention output",
    "mlp_out": "MLP output",
    "head": "attention head output (z)",
    "sae_feature": "SAE feature",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


Seed = Annotated[int, Field(ge=0, lt=SEED_MAX)]

# ---------------------------------------------------------------------------------------------
# Positions


class AllPositions(_Strict):
    """Intervene on every token position at once."""

    kind: Literal["all"] = "all"


class LastPosition(_Strict):
    """The final token of each prompt (the position that predicts the answer)."""

    kind: Literal["last"] = "last"


class IndexPosition(_Strict):
    """A fixed token index. Negative values count from the end (-1 is the last token)."""

    kind: Literal["index"] = "index"
    index: int


class LabelPosition(_Strict):
    """A named position from the dataset, such as ``S2`` in IOI prompts."""

    kind: Literal["label"] = "label"
    label: str = Field(min_length=1)


Position = Annotated[
    AllPositions | LastPosition | IndexPosition | LabelPosition, Field(discriminator="kind")
]

ALL_POSITIONS = AllPositions(kind="all")


def describe_position(position: AllPositions | LastPosition | IndexPosition | LabelPosition) -> str:
    if isinstance(position, AllPositions):
        return "all positions"
    if isinstance(position, LastPosition):
        return "the last token"
    if isinstance(position, IndexPosition):
        return f"token index {position.index}"
    return f"position {position.label}"


# ---------------------------------------------------------------------------------------------
# Sites and sweep scopes


class Site(_Strict):
    kind: SiteKind
    layer: int = Field(ge=0)
    head: int | None = Field(default=None, ge=0)
    # A feature of the spec's SAE (kind sae_feature): its index among the SAE's features.
    feature: int | None = Field(default=None, ge=0)
    position: Position

    @model_validator(mode="after")
    def _head_matches_kind(self) -> Site:
        if self.kind == "head" and self.head is None:
            raise ValueError("a head site needs a head index")
        if self.kind != "head" and self.head is not None:
            raise ValueError(f"a {self.kind} site has no head index")
        if self.kind == "sae_feature" and self.feature is None:
            raise ValueError("an SAE feature site needs a feature index")
        if self.kind != "sae_feature" and self.feature is not None:
            raise ValueError(f"a {self.kind} site has no feature index")
        return self


class HeadsScope(_Strict):
    """Every attention head in every layer, at one position: a layer x head grid."""

    kind: Literal["heads"] = "heads"
    position: Position


class LayerPositionScope(_Strict):
    """One stream site at every layer and every position: a layer x position grid.

    ``positions="each"`` uses every token index and needs prompts of equal length.
    ``positions="labels"`` uses the dataset's named positions, which works across templates.
    """

    kind: Literal["layer_position"] = "layer_position"
    site: StreamSiteKind
    positions: Literal["each", "labels"]


class LayerComponentsScope(_Strict):
    """Attention and MLP outputs of every layer, at one position: a layer x component grid."""

    kind: Literal["layer_components"] = "layer_components"
    components: list[StreamSiteKind]
    position: Position

    @model_validator(mode="after")
    def _non_empty(self) -> LayerComponentsScope:
        if not self.components:
            raise ValueError("choose at least one component")
        if len(set(self.components)) != len(self.components):
            raise ValueError("components must not repeat")
        return self


class SitesScope(_Strict):
    """An explicit list of sites, for example a single head chosen on the map."""

    kind: Literal["sites"] = "sites"
    sites: list[Site] = Field(min_length=1)


class FeaturesScope(_Strict):
    """Every feature of the spec's SAE, at one position (or summed over all of them).

    Only attribution patching can sweep every feature: it estimates them all from one gradient
    and keeps the ``top`` with the largest estimated effects (by magnitude) as the run's sites,
    ready to verify by patching.
    """

    kind: Literal["features"] = "features"
    position: Position
    top: int = Field(ge=1, le=500)


# What "the rest of the model" is made of, for sets that intervene on everything but their sites.
UniverseKind = Literal["head", "attn_out", "mlp_out"]
# Sites a set can hold: components that write into the residual stream, and the stream itself
# before or after a layer (between attention and MLP it isn't a single hook, so not in a set).
SET_SITE_KINDS = ("head", "attn_out", "mlp_out", "resid_pre", "resid_post")


class SiteSet(_Strict):
    """Sites intervened on together, in one forward pass.

    With ``complement``, the intervention covers every component of the scope's universe except
    these sites: a set of heads then reads as "keep this circuit, replace the rest". A site at one
    position keeps only that position of its component; the component's other positions are
    replaced with the rest. A complement set with no sites replaces the whole universe.
    """

    label: str = Field(min_length=1, max_length=80)
    sites: list[Site] = Field(max_length=4096)
    complement: bool

    @model_validator(mode="after")
    def _usable(self) -> SiteSet:
        if not self.complement and not self.sites:
            raise ValueError(f"the set {self.label!r} has no sites to intervene on")
        for site in self.sites:
            if site.kind not in SET_SITE_KINDS:
                raise ValueError(
                    f"a set can't hold {site.kind} sites; use heads, attention or MLP outputs, "
                    "or the residual stream before or after a layer"
                )
        return self


class SiteSetsScope(_Strict):
    """Sets of sites, each intervened on at once: one result per set.

    A circuit is evaluated with a few sets: the circuit alone (``complement``: everything else is
    replaced), the circuit removed, and everything removed. Each set's effect is also reported as
    a share of the set that replaces the whole universe, when the scope has one.
    """

    kind: Literal["site_sets"] = "site_sets"
    universe: list[UniverseKind] | None
    sets: list[SiteSet] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def _consistent(self) -> SiteSetsScope:
        labels = [s.label for s in self.sets]
        if len(set(labels)) != len(labels):
            raise ValueError("set labels must not repeat")
        complements = [s for s in self.sets if s.complement]
        if complements and not self.universe:
            raise ValueError(
                "a set that replaces everything but its sites needs the scope's universe: the "
                "components that make up the rest of the model"
            )
        if self.universe is not None:
            if not self.universe or len(set(self.universe)) != len(self.universe):
                raise ValueError("the universe must list each component kind once")
            if "head" in self.universe and "attn_out" in self.universe:
                raise ValueError(
                    "the universe can't hold both heads and attention outputs: a layer's "
                    "attention output is the sum of its heads"
                )
        for s in complements:
            for site in s.sites:
                if site.kind not in (self.universe or []):
                    raise ValueError(
                        f"the set {s.label!r} keeps a {site.kind} site, which isn't part of the "
                        "universe it replaces"
                    )
        return self


Scope = Annotated[
    HeadsScope
    | LayerPositionScope
    | LayerComponentsScope
    | SitesScope
    | FeaturesScope
    | SiteSetsScope,
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------------------------
# Experiments


Direction = Literal["clean_to_corrupt", "corrupt_to_clean"]


class ActivationPatching(_Strict):
    """Copy activations from one prompt of each pair into the run on the other.

    ``clean_to_corrupt`` runs the corrupt prompt and patches in clean activations
    ("does this restore the behavior?"). ``corrupt_to_clean`` runs the clean prompt and patches
    in corrupt activations ("does this break it?").
    """

    kind: Literal["activation_patching"] = "activation_patching"
    direction: Direction


class ZeroBaseline(_Strict):
    kind: Literal["zero"] = "zero"


class MeanBaseline(_Strict):
    """Replace the activation with its mean over a stated reference set of prompts.

    With an all-positions intervention the mean is taken per position, over reference prompts
    of the same token length. With a single position it is taken at each reference prompt's own
    resolved position.
    """

    kind: Literal["mean"] = "mean"
    reference: Literal["clean", "corrupt"]


class ResampleBaseline(_Strict):
    """Replace the activation with the same activation from other prompts (donors).

    Each prompt draws ``donors`` donors from the pool without replacement, never itself, using
    ``seed``. The patched metric is averaged over donors. The same donors are used at every site.
    """

    kind: Literal["resample"] = "resample"
    pool: Literal["clean", "corrupt"]
    donors: int = Field(ge=1, le=1000)
    seed: Seed


Baseline = Annotated[ZeroBaseline | MeanBaseline | ResampleBaseline, Field(discriminator="kind")]


class Ablation(_Strict):
    """Run the clean prompt and replace the activation with a baseline value."""

    kind: Literal["ablation"] = "ablation"
    baseline: Baseline


class DirectLogitAttribution(_Strict):
    """Split the logit difference of the chosen prompts into what each component writes directly.

    At the last position, the residual stream is the sum of the embeddings and every attention and
    MLP output. With the final normalization's scale held at its value in the run, the logit
    difference is a sum of one term per component plus a constant from biases. A component's term
    is its direct effect: it leaves out everything the component does through later components.
    This is a decomposition of one forward pass, not an intervention.
    """

    kind: Literal["direct_logit_attribution"] = "direct_logit_attribution"
    prompts: Literal["clean", "corrupt"]


class AttributionPatching(_Strict):
    """Estimate activation patching at every site from gradients (attribution patching).

    For each site, the change patching would cause is estimated as (source activation - receiver
    activation) · a gradient of the metric. With ``method="gradient"`` the gradient is taken at the
    receiver run: one forward and backward pass per batch of prompts instead of one patched run
    per site. It is a first-order estimate: it misses saturation (in attention, normalization and
    the softmax) and can miss or even invert an effect.

    With ``method="integrated_gradients"`` the gradient is averaged over ``steps`` runs whose input
    embeddings lie evenly between the receiver's and the source's (midpoints of equal intervals),
    as in EAP-IG. It follows the metric along the way from one prompt to the other, which corrects
    much of the saturation a single gradient misses, at ``steps`` times the cost. Verify the
    strongest sites with activation patching either way.
    """

    kind: Literal["attribution_patching"] = "attribution_patching"
    direction: Direction
    method: Literal["gradient", "integrated_gradients"]
    steps: int | None = Field(ge=2, le=64)

    @model_validator(mode="after")
    def _steps_match_method(self) -> AttributionPatching:
        if self.method == "integrated_gradients" and self.steps is None:
            raise ValueError("integrated gradients need a number of steps (2 to 64)")
        if self.method == "gradient" and self.steps is not None:
            raise ValueError("a single gradient takes no steps; set steps to null")
        return self


class Steering(_Strict):
    """Add a direction to the residual stream at several strengths, and measure what it does.

    At each steered site, the direction is the mean difference between the two prompts of each
    pair, from the receiver (``apply_to``) toward the other prompt. It is computed on a training
    split of the prompts and applied to the held-out rest, so no prompt receives its own
    difference. A strength of 1 adds the whole mean difference. The effect is normalized like
    patching: 1 means the steered prompt moves as far as switching to the other prompt. With
    ``control``, a random direction of the same length, at the same strengths, runs alongside.
    """

    kind: Literal["steering"] = "steering"
    apply_to: Literal["clean", "corrupt"]
    coefficients: list[float] = Field(min_length=1, max_length=16)
    train_fraction: float = Field(gt=0.0, lt=1.0)
    seed: Seed
    control: bool

    @model_validator(mode="after")
    def _distinct_finite(self) -> Steering:
        if any(c != c or c in (float("inf"), float("-inf")) for c in self.coefficients):
            raise ValueError("steering strengths must be finite numbers")
        if len(set(self.coefficients)) != len(self.coefficients):
            raise ValueError("steering strengths must not repeat")
        return self


class HeadReceiver(_Strict):
    """A later attention head whose query, key or value input receives the path."""

    kind: Literal["head"] = "head"
    layer: int = Field(ge=0)
    head: int = Field(ge=0)
    input: Literal["q", "k", "v"]


class LogitsReceiver(_Strict):
    """The residual stream at the end of the model, read directly by the unembedding."""

    kind: Literal["logits"] = "logits"


PathReceiver = Annotated[HeadReceiver | LogitsReceiver, Field(discriminator="kind")]


class PathPatching(_Strict):
    """Patch a component's effect along chosen paths only (path patching).

    For each sender (a site of the scope), the receiver prompt runs with the sender's activation
    from the source prompt and every other attention head held at its own value, so the change
    travels only through the residual stream (and through the MLPs, unless ``freeze_mlps``). The
    receivers' inputs from that run are recorded, then patched into an otherwise unchanged
    receiver run, and the metric is read there. The result is the sender's effect through those
    receivers alone. Senders must come before at least one receiver.
    """

    kind: Literal["path_patching"] = "path_patching"
    direction: Direction
    receivers: list[PathReceiver] = Field(min_length=1, max_length=64)
    freeze_mlps: bool

    @model_validator(mode="after")
    def _distinct(self) -> PathPatching:
        keys = [r.model_dump_json() for r in self.receivers]
        if len(set(keys)) != len(keys):
            raise ValueError("path patching receivers must not repeat")
        return self


def receiver_text(receiver: HeadReceiver | LogitsReceiver) -> str:
    if isinstance(receiver, LogitsReceiver):
        return "logits"
    return f"L{receiver.layer} H{receiver.head} {receiver.input}"


Experiment = Annotated[
    ActivationPatching
    | Ablation
    | DirectLogitAttribution
    | AttributionPatching
    | Steering
    | PathPatching,
    Field(discriminator="kind"),
]


def strength_text(coefficient: float) -> str:
    """A steering strength as written on the map, for example ×2 or ×−0.5."""
    text = f"{coefficient:g}".replace("-", "−")
    return f"×{text}"


# ---------------------------------------------------------------------------------------------
# Model, data, metric, statistics


def check_revision(value: str | None, what: str) -> str | None:
    """A commit, tag or branch name. Revisions name folders in the Hugging Face cache, so one that
    climbs out of it ('..'), or is a path, is refused."""
    if value is None:
        return value
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or any(part in ("", "..") for part in value.split("/"))
    ):
        raise ValueError(f"{what} must be a commit, tag or branch name")
    return value


class ModelRef(_Strict):
    id: str = Field(min_length=1, description="Hugging Face model id")
    revision: str | None = Field(
        description="Exact commit. None resolves the current main branch at run time."
    )
    dtype: Literal["float32", "float16", "bfloat16"]
    device: Literal["auto", "cpu", "cuda", "mps"]
    process_weights: bool = Field(
        description="Fold LayerNorm weights and center writing weights and the unembedding, as "
        "TransformerLens does by default. Logit differences are unchanged by this.",
    )

    @model_validator(mode="after")
    def _revision(self) -> ModelRef:
        check_revision(self.revision, "model.revision")
        return self


DATASET_PATH_HELP = (
    "a path inside the project, written with forward slashes and without '..', such as "
    "datasets/ioi.jsonl"
)


def check_relative_path(value: str, what: str) -> str:
    """Refuse absolute paths, drive letters and '..': specs travel between machines, and a path
    that names the machine it was made on, or leaves the project, has no place in one."""
    text = value.replace("\\", "/")
    if (
        text.startswith("/")
        or re.match(r"^[A-Za-z]:", text)
        or any(part == ".." for part in text.split("/"))
    ):
        raise ValueError(f"{what} must be {DATASET_PATH_HELP}")
    return value


class DatasetRef(_Strict):
    path: str = Field(min_length=1, description="Project-relative path to a JSONL dataset")
    sha256: str | None = Field(description="Expected hash of the file, if pinned")
    limit: int | None = Field(ge=1, description="Use only the first n prompts")

    @model_validator(mode="after")
    def _relative(self) -> DatasetRef:
        check_relative_path(self.path, "dataset.path")
        return self


class SAERef(_Strict):
    """A published sparse autoencoder (or transcoder) on Hugging Face: the repository, the folder
    holding it (empty for the top level) and the exact commit, pinned when a run starts."""

    repo: str = Field(min_length=1)
    path: str
    revision: str | None

    @model_validator(mode="after")
    def _relative(self) -> SAERef:
        if self.path:
            check_relative_path(self.path, "sae.path")
        check_revision(self.revision, "sae.revision")
        return self


class Tokenization(_Strict):
    prepend_bos: bool


Normalization = Literal["dataset_gap", "prompt_gap"]


class LogitDiffMetric(_Strict):
    """logit(answer) - logit(distractor) at the last position. Answers and distractors are single
    tokens (or sets of single tokens, read through the log of their summed probability)."""

    kind: Literal["logit_diff"] = "logit_diff"
    normalization: Normalization


class LogProbDiffMetric(_Strict):
    """log P(answer) - log P(distractor): with answers of several tokens, each continuation's
    log-probability is the sum over its tokens, each predicted from the prompt and the tokens
    before it. For single tokens it equals the logit difference."""

    kind: Literal["logprob_diff"] = "logprob_diff"
    normalization: Normalization


class LogProbMetric(_Strict):
    """log P(answer): the answer's log-probability (summed over its tokens)."""

    kind: Literal["logprob"] = "logprob"
    normalization: Normalization


class ProbMetric(_Strict):
    """P(answer): the answer's probability (the product over its tokens; the sum over a set)."""

    kind: Literal["prob"] = "prob"
    normalization: Normalization


class ProbDiffMetric(_Strict):
    """P(answer) - P(distractor), as in the greater-than task's probability difference."""

    kind: Literal["prob_diff"] = "prob_diff"
    normalization: Normalization


class KLMetric(_Strict):
    """KL(P_target || P): how far the next-token distribution at the last position is from the
    ``target`` prompt's own. It reads the whole distribution, not only the answer.

    Normalized like the others, it becomes the share of the divergence an intervention closes or
    opens: with the target being the source prompt, 1 means the patched run predicts exactly what
    the source prompt predicts."""

    kind: Literal["kl"] = "kl"
    target: Literal["clean", "corrupt"]
    normalization: Normalization


Metric = Annotated[
    LogitDiffMetric | LogProbDiffMetric | LogProbMetric | ProbMetric | ProbDiffMetric | KLMetric,
    Field(discriminator="kind"),
]

METRIC_LABELS: dict[str, str] = {
    "logit_diff": "logit difference",
    "logprob_diff": "log-probability difference",
    "logprob": "answer log-probability",
    "prob": "answer probability",
    "prob_diff": "probability difference",
    "kl": "KL divergence",
}

# Metrics read at the last position only: they can't score answers of several tokens.
SINGLE_POSITION_METRICS = ("logit_diff", "kl")


def describe_metric(metric: Any) -> str:
    """The metric in words, for example 'KL divergence from the clean prompt's prediction'."""
    if metric.kind == "kl":
        return f"KL divergence from the {metric.target} prompt's next-token distribution"
    return {
        "logit_diff": "logit(answer) − logit(distractor) at the last position",
        "logprob_diff": "log P(answer) − log P(distractor)",
        "logprob": "log P(answer)",
        "prob": "P(answer)",
        "prob_diff": "P(answer) − P(distractor)",
    }[metric.kind]


CLUSTER_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")


class Statistics(_Strict):
    """Percentile bootstrap over prompts. With ``cluster``, the bootstrap resamples groups of
    prompts that share a value of that ``meta`` field (for example ``template``) instead of single
    prompts, since prompts from one template aren't independent."""

    bootstrap: int = Field(ge=100, le=100_000)
    ci: float = Field(gt=0.5, lt=1.0)
    seed: Seed
    cluster: str | None

    @model_validator(mode="after")
    def _cluster_key(self) -> Statistics:
        if self.cluster is not None and not CLUSTER_KEY.match(self.cluster):
            raise ValueError("statistics.cluster must name a field of the prompts' meta")
        return self


class Execution(_Strict):
    batch_size: int = Field(
        ge=1,
        le=4096,
        description="Rows per forward pass. Part of the spec because batch shape can change "
        "floating-point results in the last digits.",
    )


class PredictionSettings(_Strict):
    """An optional, explicitly configured diagnostic; it never changes an intervention."""

    method: Literal["final_norm_logit_lens"]
    prompt_index: int = Field(ge=0)
    which: Literal["clean", "corrupt"]
    position: Annotated[LastPosition | IndexPosition, Field(discriminator="kind")]
    top_k: int = Field(ge=1, le=20)


class Spec(_Strict):
    logogram_spec: Literal[2]
    name: str = Field(min_length=1, max_length=NAME_MAX)
    notes: str = ""
    model: ModelRef
    dataset: DatasetRef
    tokenization: Tokenization
    experiment: Experiment
    scope: Scope
    metric: Metric
    statistics: Statistics
    execution: Execution
    predictions: PredictionSettings | None = None
    # The SAE (or transcoder) whose features sae_feature sites and the features scope refer to.
    sae: SAERef | None = None

    # Fields a version 1 spec left out, which took the values version 1 gave them.
    _upgraded: list[str] = PrivateAttr(default_factory=list)

    @model_validator(mode="wrap")
    @classmethod
    def _upgrade(cls, data: Any, handler: ModelWrapValidatorHandler[Spec]) -> Spec:
        filled: list[str] = []
        if isinstance(data, dict) and data.get("logogram_spec", 1) == 1:
            data, filled = upgrade_v1(data)
        spec = handler(data)
        if filled:
            spec._upgraded = filled
        return spec

    @property
    def upgraded_fields(self) -> list[str]:
        return list(self._upgraded)

    @model_validator(mode="after")
    def _features_have_an_sae(self) -> Spec:
        uses = isinstance(self.scope, FeaturesScope) or (
            isinstance(self.scope, SitesScope)
            and any(s.kind == "sae_feature" for s in self.scope.sites)
        )
        if uses and self.sae is None:
            raise ValueError("SAE features need the spec's sae: the SAE they belong to")
        return self

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2) + "\n"

    @classmethod
    def from_path(cls, path: str | Path) -> Spec:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------
# Version 1 specs


def upgrade_v1(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """A version 1 spec as version 2, and the fields that took version 1's values because the
    spec left them out. Fields new in version 2 take the value that reproduces version 1 (no
    clustering, a single gradient) and aren't listed: they didn't exist to be chosen."""
    data = copy.deepcopy(data)
    filled: list[str] = []

    def fill(obj: Any, key: str, value: Any, path: str, report: bool = True) -> None:
        if isinstance(obj, dict) and key not in obj:
            obj[key] = copy.deepcopy(value)
            if report:
                filled.append(f"{path} = {_show(value)}")

    all_positions = {"kind": "all"}
    data["logogram_spec"] = 2
    model = data.get("model")
    if isinstance(model, dict):
        fill(model, "revision", None, "model.revision")
        fill(model, "dtype", "float32", "model.dtype")
        fill(model, "device", "auto", "model.device")
        fill(model, "process_weights", True, "model.process_weights")
    dataset = data.get("dataset")
    if isinstance(dataset, dict):
        fill(dataset, "sha256", None, "dataset.sha256", report=False)
        fill(dataset, "limit", None, "dataset.limit")
    fill(data, "tokenization", {"prepend_bos": True}, "tokenization")
    if isinstance(data.get("tokenization"), dict):
        fill(data["tokenization"], "prepend_bos", True, "tokenization.prepend_bos")
    fill(data, "metric", {"kind": "logit_diff", "normalization": "dataset_gap"}, "metric")
    if isinstance(data.get("metric"), dict):
        fill(data["metric"], "kind", "logit_diff", "metric.kind")
        fill(data["metric"], "normalization", "dataset_gap", "metric.normalization")
    fill(data, "statistics", {"bootstrap": 1000, "ci": 0.95, "seed": 0}, "statistics")
    stats = data.get("statistics")
    if isinstance(stats, dict):
        fill(stats, "bootstrap", 1000, "statistics.bootstrap")
        fill(stats, "ci", 0.95, "statistics.ci")
        fill(stats, "seed", 0, "statistics.seed")
        fill(stats, "cluster", None, "statistics.cluster", report=False)
    fill(data, "execution", {"batch_size": 64}, "execution")
    if isinstance(data.get("execution"), dict):
        fill(data["execution"], "batch_size", 64, "execution.batch_size")
    sae = data.get("sae")
    if isinstance(sae, dict):
        fill(sae, "path", "", "sae.path")
        fill(sae, "revision", None, "sae.revision", report=False)

    exp = data.get("experiment")
    if isinstance(exp, dict):
        kind = exp.get("kind")
        if kind == "attribution_patching":
            fill(exp, "method", "gradient", "experiment.method", report=False)
            fill(exp, "steps", None, "experiment.steps", report=False)
        baseline = exp.get("baseline") if kind == "ablation" else None
        if isinstance(baseline, dict):
            if baseline.get("kind") == "mean":
                fill(baseline, "reference", "corrupt", "experiment.baseline.reference")
            elif baseline.get("kind") == "resample":
                fill(baseline, "pool", "corrupt", "experiment.baseline.pool")
                fill(baseline, "donors", 10, "experiment.baseline.donors")
                fill(baseline, "seed", 0, "experiment.baseline.seed")

    scope = data.get("scope")
    if isinstance(scope, dict):
        kind = scope.get("kind")
        if kind in ("heads", "layer_components", "features"):
            fill(scope, "position", all_positions, "scope.position")
        if kind == "layer_position":
            fill(scope, "site", "resid_pre", "scope.site")
            fill(scope, "positions", "each", "scope.positions")
        if kind == "layer_components":
            fill(scope, "components", ["attn_out", "mlp_out"], "scope.components")
        if kind == "sites":
            for i, site in enumerate(scope.get("sites") or []):
                fill(site, "position", all_positions, f"scope.sites[{i}].position")
    return data, filled


def _show(value: Any) -> str:
    return json.dumps(value)


# ---------------------------------------------------------------------------------------------
# Descriptions


def describe_intervention(
    exp: ActivationPatching
    | Ablation
    | DirectLogitAttribution
    | AttributionPatching
    | Steering
    | PathPatching,
) -> str:
    """The intervention in a few words, for example 'Resample-ablate (10 corrupt donors, seed 0)'."""
    if isinstance(exp, ActivationPatching):
        return (
            "Patch clean → corrupt"
            if exp.direction == "clean_to_corrupt"
            else "Patch corrupt → clean"
        )
    if isinstance(exp, DirectLogitAttribution):
        return f"Direct logit attribution ({exp.prompts} prompts)"
    if isinstance(exp, AttributionPatching):
        arrow = "clean → corrupt" if exp.direction == "clean_to_corrupt" else "corrupt → clean"
        if exp.method == "integrated_gradients":
            return f"Attribution patching, estimated with integrated gradients ({exp.steps} steps, {arrow})"
        return f"Attribution patching, estimated ({arrow})"
    if isinstance(exp, PathPatching):
        arrow = "clean → corrupt" if exp.direction == "clean_to_corrupt" else "corrupt → clean"
        names = ", ".join(receiver_text(r) for r in exp.receivers[:4])
        more = f" and {len(exp.receivers) - 4} more" if len(exp.receivers) > 4 else ""
        frozen = "heads and MLPs" if exp.freeze_mlps else "other heads"
        return f"Path patching {arrow} into {names}{more} ({frozen} frozen)"
    if isinstance(exp, Steering):
        toward = "corrupt" if exp.apply_to == "clean" else "clean"
        strengths = ", ".join(strength_text(c) for c in exp.coefficients)
        control = ", random control" if exp.control else ""
        return (
            f"Steer {exp.apply_to} prompts toward {toward} ({strengths}; "
            f"{exp.train_fraction:.0%} of pairs train the direction, seed {exp.seed}{control})"
        )
    b = exp.baseline
    if isinstance(b, ZeroBaseline):
        return "Zero-ablate"
    if isinstance(b, MeanBaseline):
        return f"Mean-ablate ({b.reference} mean)"
    return f"Resample-ablate ({b.donors} {b.pool} donors, seed {b.seed})"


def describe_experiment(spec: Spec) -> str:
    """A one-line plain-language description, used in history and summaries."""
    what = describe_intervention(spec.experiment)
    scope = spec.scope
    if isinstance(scope, HeadsScope):
        where = f"every head, {describe_position(scope.position)}"
    elif isinstance(scope, LayerPositionScope):
        unit = "every position" if scope.positions == "each" else "each labelled position"
        where = f"{scope.site} at every layer and {unit}"
    elif isinstance(scope, LayerComponentsScope):
        where = f"{' and '.join(scope.components)} per layer, {describe_position(scope.position)}"
    elif isinstance(scope, FeaturesScope):
        where = (
            f"every SAE feature, {describe_position(scope.position)}, keeping the top {scope.top}"
        )
    elif isinstance(scope, SiteSetsScope):
        n = len(scope.sets)
        where = f"{n} set{'s' if n != 1 else ''} of sites, each at once"
    else:
        where = f"{len(scope.sites)} chosen site{'s' if len(scope.sites) != 1 else ''}"
    metric = "" if spec.metric.kind == "logit_diff" else f" · {METRIC_LABELS[spec.metric.kind]}"
    return f"{what} · {where}{metric}"
