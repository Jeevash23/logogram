"""The experiment spec: the single description of an experiment that the GUI and CLI both run.

A spec is stored as ``experiments/<id>/spec.json``. Every methodological choice that can change
a number (direction, baseline, position, metric, normalization, seeds, batch size, dtype) lives
here explicitly, so a run can always be traced back to how it was produced.

Specs refer to abstract sites (``resid_pre``, ``attn_out``, ``head``, ...). Backend-specific hook
names never appear in a spec.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

NAME_MAX = 200

SPEC_VERSION = 1

StreamSiteKind = Literal["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"]
SiteKind = Literal["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out", "head"]

SITE_KIND_LABELS: dict[str, str] = {
    "resid_pre": "residual stream before the layer",
    "resid_mid": "residual stream between attention and MLP",
    "resid_post": "residual stream after the layer",
    "attn_out": "attention output",
    "mlp_out": "MLP output",
    "head": "attention head output (z)",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


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
    position: Position = Field(default_factory=AllPositions)

    @model_validator(mode="after")
    def _head_matches_kind(self) -> Site:
        if self.kind == "head" and self.head is None:
            raise ValueError("a head site needs a head index")
        if self.kind != "head" and self.head is not None:
            raise ValueError(f"a {self.kind} site has no head index")
        return self


class HeadsScope(_Strict):
    """Every attention head in every layer, at one position: a layer x head grid."""

    kind: Literal["heads"] = "heads"
    position: Position = Field(default_factory=AllPositions)


class LayerPositionScope(_Strict):
    """One stream site at every layer and every position: a layer x position grid.

    ``positions="each"`` uses every token index and needs prompts of equal length.
    ``positions="labels"`` uses the dataset's named positions, which works across templates.
    """

    kind: Literal["layer_position"] = "layer_position"
    site: StreamSiteKind = "resid_pre"
    positions: Literal["each", "labels"] = "each"


class LayerComponentsScope(_Strict):
    """Attention and MLP outputs of every layer, at one position: a layer x component grid."""

    kind: Literal["layer_components"] = "layer_components"
    components: list[StreamSiteKind] = Field(default_factory=lambda: ["attn_out", "mlp_out"])
    position: Position = Field(default_factory=AllPositions)

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


Scope = Annotated[
    HeadsScope | LayerPositionScope | LayerComponentsScope | SitesScope,
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------------------------
# Experiments


class ActivationPatching(_Strict):
    """Copy activations from one prompt of each pair into the run on the other.

    ``clean_to_corrupt`` runs the corrupt prompt and patches in clean activations
    ("does this restore the behavior?"). ``corrupt_to_clean`` runs the clean prompt and patches
    in corrupt activations ("does this break it?").
    """

    kind: Literal["activation_patching"] = "activation_patching"
    direction: Literal["clean_to_corrupt", "corrupt_to_clean"]


class ZeroBaseline(_Strict):
    kind: Literal["zero"] = "zero"


class MeanBaseline(_Strict):
    """Replace the activation with its mean over a stated reference set of prompts.

    With an all-positions intervention the mean is taken per position, over reference prompts
    of the same token length. With a single position it is taken at each reference prompt's own
    resolved position.
    """

    kind: Literal["mean"] = "mean"
    reference: Literal["clean", "corrupt"] = "corrupt"


class ResampleBaseline(_Strict):
    """Replace the activation with the same activation from other prompts (donors).

    Each prompt draws ``donors`` donors from the pool without replacement, never itself, using
    ``seed``. The patched metric is averaged over donors. The same donors are used at every site.
    """

    kind: Literal["resample"] = "resample"
    pool: Literal["clean", "corrupt"] = "corrupt"
    donors: int = Field(default=10, ge=1, le=1000)
    seed: int = 0


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


Experiment = Annotated[
    ActivationPatching | Ablation | DirectLogitAttribution, Field(discriminator="kind")
]


# ---------------------------------------------------------------------------------------------
# Model, data, metric, statistics


class ModelRef(_Strict):
    id: str = Field(min_length=1, description="Hugging Face model id")
    revision: str | None = Field(
        default=None, description="Exact commit. None resolves the current main branch at run time."
    )
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    process_weights: bool = Field(
        default=True,
        description="Fold LayerNorm weights and center writing weights and the unembedding, as "
        "TransformerLens does by default. Logit differences are unchanged by this.",
    )


class DatasetRef(_Strict):
    path: str = Field(min_length=1, description="Project-relative path to a JSONL dataset")
    sha256: str | None = Field(default=None, description="Expected hash of the file, if pinned")
    limit: int | None = Field(default=None, ge=1, description="Use only the first n prompts")


class Tokenization(_Strict):
    prepend_bos: bool = True


class Metric(_Strict):
    """Logit difference at the last position: logit(answer) - logit(distractor).

    The normalized effect for each prompt is (patched - receiver) divided by either the
    dataset's mean gap (``dataset_gap``) or that prompt's own gap (``prompt_gap``). The gap is
    source - receiver for patching, and corrupt - clean for ablation. 0 means no change and 1
    means a change as large as swapping to the other prompt.
    """

    kind: Literal["logit_diff"] = "logit_diff"
    normalization: Literal["dataset_gap", "prompt_gap"] = "dataset_gap"


class Statistics(_Strict):
    bootstrap: int = Field(default=1000, ge=100, le=100_000)
    ci: float = Field(default=0.95, gt=0.5, lt=1.0)
    seed: int = 0


class Execution(_Strict):
    batch_size: int = Field(
        default=64,
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
    logogram_spec: Literal[1] = SPEC_VERSION
    name: str = Field(min_length=1, max_length=NAME_MAX)
    notes: str = ""
    model: ModelRef
    dataset: DatasetRef
    tokenization: Tokenization = Field(default_factory=Tokenization)
    experiment: Experiment
    scope: Scope
    metric: Metric = Field(default_factory=Metric)
    statistics: Statistics = Field(default_factory=Statistics)
    execution: Execution = Field(default_factory=Execution)
    predictions: PredictionSettings | None = None

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2) + "\n"

    @classmethod
    def from_path(cls, path: str | Path) -> Spec:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def describe_intervention(exp: ActivationPatching | Ablation | DirectLogitAttribution) -> str:
    """The intervention in a few words, for example 'Resample-ablate (10 corrupt donors, seed 0)'."""
    if isinstance(exp, ActivationPatching):
        return (
            "Patch clean → corrupt"
            if exp.direction == "clean_to_corrupt"
            else "Patch corrupt → clean"
        )
    if isinstance(exp, DirectLogitAttribution):
        return f"Direct logit attribution ({exp.prompts} prompts)"
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
    else:
        where = f"{len(scope.sites)} chosen site{'s' if len(scope.sites) != 1 else ''}"
    return f"{what} · {where}"
