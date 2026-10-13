"""The experiment spec: the single description of an experiment that the GUI and CLI both run.

A spec is stored as ``experiments/<id>/spec.json``. Every methodological choice that can change
a number (direction, baseline, position, metric, normalization, seeds, batch size, dtype) lives
here explicitly, so a run can always be traced back to how it was produced.

Specs since version 2 state every such choice: no field that can change a number has a default.
Version 1 specs (Logogram 0.1) are still read: the fields they leave out take the values version 1
gave them, and :attr:`Spec.upgraded_fields` lists which, so a run can say so instead of assuming
silently. Version 3 (Logogram 0.5) added chat templates, local models, a steering direction and
steered continuations; a version 2 spec reads as version 3 with each of them off, which is what
version 2 did. Version 4 (Logogram 0.7) added methods that find and use directions in the model
(probes, learned subspaces, directional ablation, readouts), automated circuit discovery, and SAEs
and lenses trained in the project; a version 3 spec reads as version 4 unchanged.

Specs refer to abstract sites (``resid_pre``, ``attn_out``, ``head``, ...). Backend-specific hook
names never appear in a spec.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, RootModel, model_validator
from pydantic.functional_validators import ModelWrapValidatorHandler

from logogram.fileio import JSON_LIMIT, read_limited

NAME_MAX = 200

SPEC_VERSION = 4

# Seeds are drawn into 32-bit generators on some paths: keep every seed in that range.
SEED_MAX = 2**32

StreamSiteKind = Literal["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"]
SiteKind = Literal[
    "resid_pre",
    "resid_mid",
    "resid_post",
    "attn_out",
    "mlp_out",
    "head",
    "sae_feature",
    "q",
    "k",
    "v",
    "neuron",
]
# A head's query, key or value. With grouped-query attention (Llama, Qwen, Gemma) a model has
# fewer key and value heads than query heads, and a k or v site names a key/value head: every
# query head of its group reads it.
HeadInput = Literal["q", "k", "v"]
HEAD_INPUTS = ("q", "k", "v")

SITE_KIND_LABELS: dict[str, str] = {
    "resid_pre": "residual stream before the layer",
    "resid_mid": "residual stream between attention and MLP",
    "resid_post": "residual stream after the layer",
    "attn_out": "attention output",
    "mlp_out": "MLP output",
    "head": "attention head output (z)",
    "sae_feature": "SAE feature",
    "q": "attention head query",
    "k": "attention key (of a key/value head)",
    "v": "attention value (of a key/value head)",
    "neuron": "MLP neuron (its activation, the input to the MLP's output projection)",
}


# A string just below a field describes it: docs/reference/spec.md is written from them.
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)


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
    """The token's index in each prompt, from 0; negative counts from the end."""


class LabelPosition(_Strict):
    """A named position from the dataset, such as ``S2`` in IOI prompts."""

    kind: Literal["label"] = "label"
    label: str = Field(min_length=1)
    """A name in the prompts' ``positions``: the last token that overlaps its span."""


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


def describe_sae_site(site: str, layer: int, transcoder: bool) -> str:
    """Where an SAE's output is read into the model, in words: the site it writes, where steering
    adds one of its features."""
    if site == "head":
        return (
            f"the outputs (z) of layer {layer}'s attention heads, side by side, before the "
            "output projection"
        )
    if transcoder:
        return f"layer {layer}'s MLP output, which the transcoder writes"
    return {
        "resid_pre": f"the residual stream entering layer {layer}",
        "resid_post": f"the residual stream leaving layer {layer}",
        "attn_out": f"layer {layer}'s attention output",
        "mlp_out": f"layer {layer}'s MLP output",
    }.get(site, f"the {site} of layer {layer}")


# ---------------------------------------------------------------------------------------------
# Sites and sweep scopes


class Site(_Strict):
    """One place in the model, at a layer and a position of each prompt."""

    kind: SiteKind
    """A residual stream state, an attention or MLP output, one head's output (z), a head's query,
    key or value, one MLP neuron, or a feature of the spec's SAE."""
    layer: int = Field(ge=0)
    """The layer, from 0."""
    head: int | None = Field(default=None, ge=0)
    """The head's index in its layer: the head of a head site (its output z), the query head of a
    q site, the key/value head of a k or v site; null for any other site."""
    feature: int | None = Field(default=None, ge=0)
    """The feature's index in the spec's SAE, for an sae_feature site; null for any other."""
    neuron: int | None = Field(default=None, ge=0)
    """The neuron's index among its layer's MLP neurons, for a neuron site: its activation after
    the nonlinearity (for a gated MLP, the gate's activation times the up projection), the input
    to the MLP's output projection. Null for any other site."""
    position: Position
    """The token or tokens of each prompt intervened on."""

    @model_validator(mode="after")
    def _head_matches_kind(self) -> Site:
        headed = self.kind == "head" or self.kind in HEAD_INPUTS
        if headed and self.head is None:
            what = {"q": "a query head", "k": "a key/value head", "v": "a key/value head"}
            raise ValueError(f"a {self.kind} site needs {what.get(self.kind, 'a head')} index")
        if not headed and self.head is not None:
            raise ValueError(f"a {self.kind} site has no head index")
        if self.kind == "sae_feature" and self.feature is None:
            raise ValueError("an SAE feature site needs a feature index")
        if self.kind != "sae_feature" and self.feature is not None:
            raise ValueError(f"a {self.kind} site has no feature index")
        if self.kind == "neuron" and self.neuron is None:
            raise ValueError("a neuron site needs a neuron index")
        if self.kind != "neuron" and self.neuron is not None:
            raise ValueError(f"a {self.kind} site has no neuron index")
        return self


class HeadsScope(_Strict):
    """Every attention head in every layer, at one position: a layer x head grid."""

    kind: Literal["heads"] = "heads"
    position: Position
    """The token or tokens of each prompt where every head is intervened on."""


class HeadInputsScope(_Strict):
    """Every head's query, key or value in every layer, at one position: a layer x head grid.

    Keys and values belong to key/value heads: with grouped-query attention a model has fewer of
    them than query heads, and each is read by every query head of its group, so the grid of
    ``input="k"`` or ``"v"`` has a column per key/value head."""

    kind: Literal["head_inputs"] = "head_inputs"
    input: HeadInput
    """The input intervened on: each query head's query (q), or each key/value head's key (k) or
    value (v)."""
    position: Position
    """The token or tokens of each prompt whose query, key or value is intervened on."""


# Layers a neurons scope can list: more than any model has, few enough to check quickly.
NEURON_LAYERS_MAX = 512


class NeuronsScope(_Strict):
    """Every neuron of the MLPs of the chosen layers, at one position. A neuron's activation is
    the MLP's hidden activation after its nonlinearity (for a gated MLP, the gate's activation
    times the up projection): the input to the MLP's output projection."""

    kind: Literal["neurons"] = "neurons"
    layers: list[Annotated[int, Field(ge=0)]] = Field(min_length=1, max_length=NEURON_LAYERS_MAX)
    """The layers whose MLP neurons are measured, each listed once."""
    position: Position
    """The token or tokens of each prompt where every neuron is intervened on."""

    @model_validator(mode="after")
    def _distinct(self) -> NeuronsScope:
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("the layers of a neurons scope must not repeat")
        return self


class LayerPositionScope(_Strict):
    """One stream site at every layer and every position: a layer x position grid.

    ``positions="each"`` uses every token index and needs prompts of equal length.
    ``positions="labels"`` uses the dataset's named positions, which works across templates.
    """

    kind: Literal["layer_position"] = "layer_position"
    site: StreamSiteKind
    """The site measured at every layer and position."""
    positions: Literal["each", "labels"]
    """``each``: every token index (prompts of one length); ``labels``: the named positions."""


class LayerComponentsScope(_Strict):
    """Attention and MLP outputs of every layer, at one position: a layer x component grid."""

    kind: Literal["layer_components"] = "layer_components"
    components: list[StreamSiteKind]
    """The sites measured at every layer, each once, such as attn_out and mlp_out."""
    position: Position
    """The token or tokens of each prompt intervened on."""

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
    """The sites, each intervened on alone: one result per site."""


class FeaturesScope(_Strict):
    """Every feature of the spec's SAE, at one position (or summed over all of them).

    Only attribution patching can sweep every feature: it estimates them all from one gradient
    and keeps the ``top`` with the largest estimated effects (by magnitude) as the run's sites,
    ready to verify by patching.

    Choosing the strongest features and reporting them on the same prompts biases their effects
    away from zero (the winner's curse). With ``choose_on``, a seeded split of the prompts
    chooses the features and only the other prompts report them. ``None`` chooses and reports on
    every prompt.
    """

    kind: Literal["features"] = "features"
    position: Position
    """The token of each prompt where the features are read, or all of them, summed."""
    top: int = Field(ge=1, le=500)
    """How many features, largest estimated effects first, the run keeps as its sites."""
    choose_on: float | None = Field(gt=0.0, lt=1.0)
    """The share of prompts that choose the features (the rest report them), or null for all."""
    seed: Seed | None
    """The seed of the split into prompts that choose and prompts that report; null without it."""

    @model_validator(mode="after")
    def _split(self) -> FeaturesScope:
        if (self.choose_on is None) != (self.seed is None):
            raise ValueError(
                "choosing features on some prompts needs both choose_on (the share of prompts "
                "that choose them) and seed (for the split); give neither to use every prompt"
            )
        return self


# What "the rest of the model" is made of, for sets that intervene on everything but their sites:
# components that write into the residual stream (heads' outputs, attention and MLP outputs), or
# finer parts of them (every head's query, key or value, every MLP neuron).
UniverseKind = Literal["head", "attn_out", "mlp_out", "q", "k", "v", "neuron"]
# Sites a set can hold: components that write into the residual stream and their parts, and the
# stream itself before or after a layer (between attention and MLP it isn't a single hook, so not
# in a set).
SET_SITE_KINDS = ("head", "attn_out", "mlp_out", "resid_pre", "resid_post", "q", "k", "v", "neuron")
# A component and what it is made of can't both be in a universe: the attention output is the sum
# of its heads' outputs, each head's output is made from its query, key and value, and the MLP
# output from its neurons.
_MADE_OF = {"attn_out": ("head", "q", "k", "v"), "head": ("q", "k", "v"), "mlp_out": ("neuron",)}
UNIVERSE_WORDS = {
    "attn_out": "a layer's attention output",
    "head": "a head's output",
    "mlp_out": "a layer's MLP output",
    "q": "the heads' queries",
    "k": "the heads' keys",
    "v": "the heads' values",
    "neuron": "its neurons",
}


class SiteSet(_Strict):
    """Sites intervened on together, in one forward pass.

    With ``complement``, the intervention covers every component of the scope's universe except
    these sites: a set of heads then reads as "keep this circuit, replace the rest". A site at one
    position keeps only that position of its component; the component's other positions are
    replaced with the rest. A complement set with no sites replaces the whole universe.
    """

    label: str = Field(min_length=1, max_length=80)
    """The set's name in the results."""
    sites: list[Site] = Field(max_length=4096)
    """The sites intervened on together, or, with complement, the sites kept."""
    complement: bool
    """Intervene on every component of the scope's universe except these sites."""

    @model_validator(mode="after")
    def _usable(self) -> SiteSet:
        if not self.complement and not self.sites:
            raise ValueError(f"the set {self.label!r} has no sites to intervene on")
        for site in self.sites:
            if site.kind not in SET_SITE_KINDS:
                raise ValueError(
                    f"a set can't hold {site.kind} sites; use heads (their outputs, queries, keys "
                    "or values), attention or MLP outputs, MLP neurons, or the residual stream "
                    "before or after a layer"
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
    """The components, at every layer and position, that make up the rest of the model for
    complement sets; null when no set is one."""
    sets: list[SiteSet] = Field(min_length=1, max_length=64)
    """The sets, each intervened on at once: one result per set."""

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
            for whole, parts in _MADE_OF.items():
                inside = [p for p in parts if p in self.universe]
                if whole in self.universe and inside:
                    raise ValueError(
                        f"the universe can't hold both {whole} and {', '.join(inside)}: "
                        f"{UNIVERSE_WORDS[whole]} is made from {UNIVERSE_WORDS[inside[0]]}"
                    )
        for s in complements:
            for site in s.sites:
                if site.kind not in (self.universe or []):
                    raise ValueError(
                        f"the set {s.label!r} keeps a {site.kind} site, which isn't part of the "
                        "universe it replaces"
                    )
        return self


class HeadReceiver(_Strict):
    """A later attention head whose query, key or value input receives the path."""

    kind: Literal["head"] = "head"
    layer: int = Field(ge=0)
    """The receiving head's layer, from 0."""
    head: int = Field(ge=0)
    """The receiving head's index in its layer."""
    input: Literal["q", "k", "v"]
    """The input the path reaches: the head's query, key or value."""


class LogitsReceiver(_Strict):
    """The residual stream at the end of the model, read directly by the unembedding."""

    kind: Literal["logits"] = "logits"


class MlpReceiver(_Strict):
    """A later MLP's input: the residual stream as that MLP reads it, before its normalization."""

    kind: Literal["mlp"] = "mlp"
    layer: int = Field(ge=0)
    """The receiving MLP's layer, from 0."""


PathReceiver = Annotated[HeadReceiver | LogitsReceiver | MlpReceiver, Field(discriminator="kind")]


def receiver_text(receiver: HeadReceiver | LogitsReceiver | MlpReceiver) -> str:
    if isinstance(receiver, LogitsReceiver):
        return "logits"
    if isinstance(receiver, MlpReceiver):
        return f"L{receiver.layer} MLP"
    return f"L{receiver.layer} H{receiver.head} {receiver.input}"


class GraphScope(_Strict):
    """The edges of the model's graph, for edge attribution.

    The graph's nodes are the embeddings, every attention head (its output, z through its slice of
    the output projection), every MLP and the logits. An edge runs from a node's output to a later
    node's input: a head's query, key or value input (each its own edge), an MLP's input, or the
    logits. ``layers`` chooses the heads and MLPs whose inputs receive the edges (from every
    earlier node), None for every layer; ``logits`` adds the edges into the logits.
    """

    kind: Literal["graph"] = "graph"
    layers: list[Annotated[int, Field(ge=0)]] | None
    """The layers whose heads and MLPs receive edges, in order; null for every layer."""
    logits: bool
    """Include the edges into the logits."""

    @model_validator(mode="after")
    def _receives(self) -> GraphScope:
        if self.layers is not None:
            if not self.layers:
                raise ValueError("choose at least one layer, or null for every layer")
            if len(set(self.layers)) != len(self.layers):
                raise ValueError("layers must not repeat")
            if self.layers != sorted(self.layers):
                raise ValueError("list the layers in order")
        return self


class EdgeSender(_Strict):
    """The node an edge starts from: the embeddings, an attention head's output, or an MLP's."""

    kind: Literal["embed", "head", "mlp"]
    layer: int | None = Field(default=None, ge=0)
    """The head's or MLP's layer, from 0; null for the embeddings."""
    head: int | None = Field(default=None, ge=0)
    """The head's index in its layer; null for the embeddings and for an MLP."""

    @model_validator(mode="after")
    def _fields_match_kind(self) -> EdgeSender:
        if self.kind == "embed" and (self.layer is not None or self.head is not None):
            raise ValueError("the embeddings have no layer or head")
        if self.kind != "embed" and self.layer is None:
            raise ValueError(f"a {self.kind} sender needs its layer")
        if (self.kind == "head") != (self.head is not None):
            raise ValueError("a head sender needs its head index, and only a head has one")
        return self


class Edge(_Strict):
    """A sender's output read directly by one receiver's input, through the residual stream."""

    sender: EdgeSender
    """Where the edge starts."""
    receiver: PathReceiver
    """Where the edge ends: a later head's query, key or value, a later MLP's input, or the
    logits."""

    @model_validator(mode="after")
    def _forward(self) -> Edge:
        r = self.receiver
        s = self.sender
        if s.kind == "embed" or isinstance(r, LogitsReceiver):
            return self
        assert s.layer is not None
        # A head reads its layer's input; an MLP also reads its own layer's heads, where layers
        # add attention before the MLP (checked against the model when it runs).
        if isinstance(r, HeadReceiver) and s.layer >= r.layer:
            raise ValueError("an edge into a head starts in an earlier layer")
        if isinstance(r, MlpReceiver) and (
            s.layer > r.layer or (s.layer == r.layer and s.kind == "mlp")
        ):
            raise ValueError("an edge into an MLP starts before it")
        return self


def sender_text(sender: EdgeSender) -> str:
    if sender.kind == "embed":
        return "embeddings"
    if sender.kind == "head":
        return f"L{sender.layer} H{sender.head}"
    return f"L{sender.layer} MLP"


def edge_text(edge: Edge) -> str:
    """An edge as Logogram names it, for example "L5 H5 → L9 H6 q"."""
    return f"{sender_text(edge.sender)} → {receiver_text(edge.receiver)}"


class EdgesScope(_Strict):
    """Edges measured one at a time, each its own result: path patching from the edge's sender
    into its receiver alone, with every head and MLP held (see PathPatching)."""

    kind: Literal["edges"] = "edges"
    edges: list[Edge] = Field(min_length=1, max_length=64)
    """The edges, each measured on its own, none repeated."""

    @model_validator(mode="after")
    def _distinct(self) -> EdgesScope:
        keys = [e.model_dump_json() for e in self.edges]
        if len(set(keys)) != len(keys):
            raise ValueError("edges must not repeat")
        return self


Scope = Annotated[
    HeadsScope
    | LayerPositionScope
    | LayerComponentsScope
    | SitesScope
    | FeaturesScope
    | SiteSetsScope
    | HeadInputsScope
    | NeuronsScope
    | GraphScope
    | EdgesScope,
    Field(discriminator="kind"),
]


DIRECTION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


class DirectionRef(_Strict):
    """A direction (or a subspace of a few) saved in the project's directions folder: a probe's
    weights, a learned subspace, a principal component or a mean difference. A run pins it by the
    hash of its tensor file and keeps a copy, so saving another direction under the same name
    later never changes the run."""

    name: str
    """The direction's name in the project's directions folder."""
    sha256: str | None
    """The hash of its tensor file; null uses the direction as saved when a run starts, and the
    run's spec pins it."""

    @model_validator(mode="after")
    def _named(self) -> DirectionRef:
        if not DIRECTION_NAME.match(self.name) or self.name == "snapshots":
            raise ValueError(
                "a direction's name is letters, digits, dots, dashes and underscores, starting "
                "with a letter or digit"
            )
        if self.sha256 is not None and not SHA256.match(self.sha256):
            raise ValueError("a direction's sha256 is 64 lowercase hex digits")
        return self


# ---------------------------------------------------------------------------------------------
# Experiments


Direction = Annotated[
    Literal["clean_to_corrupt", "corrupt_to_clean"],
    Field(
        description="clean_to_corrupt runs the corrupt prompt with clean activations (does this "
        "restore the behavior?); corrupt_to_clean runs the clean prompt with corrupt ones (does "
        "this break it?)."
    ),
]


class ActivationPatching(_Strict):
    """Copy activations from one prompt of each pair into the run on the other.

    ``clean_to_corrupt`` runs the corrupt prompt and patches in clean activations
    ("does this restore the behavior?"). ``corrupt_to_clean`` runs the clean prompt and patches
    in corrupt activations ("does this break it?").
    """

    kind: Literal["activation_patching"] = "activation_patching"
    direction: Direction


class ZeroBaseline(_Strict):
    """Replace the activation with zeros."""

    kind: Literal["zero"] = "zero"


class MeanBaseline(_Strict):
    """Replace the activation with its mean over a stated reference set of prompts.

    With an all-positions intervention the mean is taken per position, over reference prompts
    of the same token length. With a single position it is taken at each reference prompt's own
    resolved position.
    """

    kind: Literal["mean"] = "mean"
    reference: Literal["clean", "corrupt"]
    """The prompts the mean is taken over: the dataset's clean or corrupt prompts."""


class ResampleBaseline(_Strict):
    """Replace the activation with the same activation from other prompts (donors).

    Each prompt draws ``donors`` donors from the pool without replacement, never itself, using
    ``seed``. The patched metric is averaged over donors. The same donors are used at every site.
    """

    kind: Literal["resample"] = "resample"
    pool: Literal["clean", "corrupt"]
    """The prompts donors are drawn from: the dataset's clean or corrupt prompts."""
    donors: int = Field(ge=1, le=1000)
    """How many donors each prompt draws."""
    seed: Seed
    """The seed of the draw."""


class ScaleBaseline(_Strict):
    """Multiply the activation by a factor: 0 removes it, as zero ablation does; 2 doubles it;
    -1 turns it around. Appended tokens read to score an answer of several tokens are scaled
    too, when every position is."""

    kind: Literal["scale"] = "scale"
    factor: float
    """What the activation is multiplied by."""

    @model_validator(mode="after")
    def _finite(self) -> ScaleBaseline:
        if self.factor != self.factor or self.factor in (float("inf"), float("-inf")):
            raise ValueError("the scale factor must be a finite number")
        return self


class ProjectionBaseline(_Strict):
    """Directional ablation: replace the component of the activation along a saved direction
    (or within a saved subspace) and keep everything at right angles to it. With no reference the
    component is removed (set to zero); with one, it is set to its mean over those prompts (at
    the same position, as mean ablation takes it). Only activations as wide as the residual
    stream have such a component: stream sites and attention and MLP outputs."""

    kind: Literal["projection"] = "projection"
    direction: DirectionRef
    """The direction or subspace whose component is replaced."""
    reference: Literal["clean", "corrupt"] | None
    """The prompts whose mean component replaces it, or null to remove it."""


Baseline = Annotated[
    ZeroBaseline | MeanBaseline | ResampleBaseline | ScaleBaseline | ProjectionBaseline,
    Field(discriminator="kind"),
]


class Ablation(_Strict):
    """Run the clean prompt and replace the activation with a baseline value."""

    kind: Literal["ablation"] = "ablation"
    baseline: Baseline
    """What replaces the activation. It changes the result, so there is no default."""


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
    """Whose logit difference is split: the clean prompts' or the corrupt prompts'."""


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
    """One gradient at the receiver run, or gradients averaged along the way between the two
    prompts' input embeddings (integrated gradients)."""
    steps: int | None = Field(ge=2, le=64)
    """The points along the way, for integrated gradients; null for a single gradient."""

    @model_validator(mode="after")
    def _steps_match_method(self) -> AttributionPatching:
        if self.method == "integrated_gradients" and self.steps is None:
            raise ValueError("integrated gradients need a number of steps (2 to 64)")
        if self.method == "gradient" and self.steps is not None:
            raise ValueError("a single gradient takes no steps; set steps to null")
        return self


class RandomEdges(_Strict):
    """The control: edges chosen at random, ``draws`` times. Each draw orders every edge of the
    graph by a seeded shuffle and takes the first k, as the strongest are taken from the
    estimates."""

    seed: Seed
    """The seed of the shuffles."""
    draws: int = Field(ge=1, le=16)
    """How many random orders are drawn: each point of the control averages them, prompt by
    prompt, and keeps each draw's own faithfulness."""


class FaithfulnessCurve(_Strict):
    """The k strongest edges as a circuit, for each k of ``counts``.

    ``patch`` says which edges carry the source prompt's values in the receiver run: with
    ``"chosen"``, the chosen edges do and every other edge keeps the receiver's own; with
    ``"rest"``, the chosen edges keep the receiver's and every other edge carries the source's.
    Faithfulness is the share of the gap between the run with no edge chosen and the run with
    every edge chosen that the chosen edges close: 0 with none, 1 with every edge of the graph.
    """

    counts: list[Annotated[int, Field(ge=1)]] = Field(min_length=1, max_length=24)
    """The numbers of edges chosen, increasing: the curve's points."""
    patch: Literal["chosen", "rest"]
    """Which edges carry the source prompt's values: the chosen ones, or every other one."""
    control: RandomEdges
    """The same counts of edges chosen at random, the curve to compare with."""

    @model_validator(mode="after")
    def _increasing(self) -> FaithfulnessCurve:
        if any(b <= a for a, b in zip(self.counts, self.counts[1:], strict=False)):
            raise ValueError("the curve's edge counts must increase")
        return self


class EdgeAttribution(_Strict):
    """Estimate the effect of every edge of the model's graph from gradients (edge attribution
    patching, EAP, as in Syed et al. 2023; with integrated gradients, EAP-IG, as in Hanna et al.
    2024).

    The estimate for the edge from u to v is (u's output on the source prompt − on the receiver
    prompt) · the gradient of the metric with respect to v's own input, summed over positions.
    v's own input is the copy of the residual stream that only v reads, before v's normalization;
    the gradient goes through that normalization as the model computes it, its scale recomputed
    from the input (see logogram.edges). With ``method="integrated_gradients"`` the gradient is
    averaged over ``steps`` runs whose input embeddings lie evenly between the receiver's and the
    source's, as in attribution patching. With ``curve``, the strongest edges are then patched
    together, edge by edge, for a faithfulness curve beside edges chosen at random.
    """

    kind: Literal["edge_attribution"] = "edge_attribution"
    direction: Direction
    method: Literal["gradient", "integrated_gradients"]
    """One gradient at the receiver run, or gradients averaged along the way between the two
    prompts' input embeddings (integrated gradients)."""
    steps: int | None = Field(ge=2, le=64)
    """The points along the way, for integrated gradients; null for a single gradient."""
    curve: FaithfulnessCurve | None
    """Patch the strongest edges together for a faithfulness curve; null to only estimate."""

    @model_validator(mode="after")
    def _steps_match_method(self) -> EdgeAttribution:
        if self.method == "integrated_gradients" and self.steps is None:
            raise ValueError("integrated gradients need a number of steps (2 to 64)")
        if self.method == "gradient" and self.steps is not None:
            raise ValueError("a single gradient takes no steps; set steps to null")
        return self


class MeanDifference(_Strict):
    """At each steered site, the mean difference between the two prompts of each training pair,
    from the receiver toward the other prompt. A strength of 1 adds the whole mean difference."""

    kind: Literal["mean_difference"] = "mean_difference"


class FeatureDirection(_Strict):
    """The decoder direction of one feature of the spec's SAE, added where the SAE reads."""

    kind: Literal["sae_feature"] = "sae_feature"
    feature: int = Field(ge=0)
    """The feature's index in the spec's SAE."""


class SavedDirection(_Strict):
    """A direction saved in the project (one dimension), added where the steered site is. A
    strength of 1 moves an activation's component along it as far as the mean difference between
    the two prompts of each training pair does."""

    kind: Literal["direction"] = "direction"
    direction: DirectionRef
    """The saved direction; it must have one dimension and the residual stream's width."""


SteeringDirection = Annotated[
    MeanDifference | FeatureDirection | SavedDirection, Field(discriminator="kind")
]


class Continuations(_Strict):
    """Greedy continuations of the first held-out prompts, without steering and at each strength
    (and with the control, when it runs): what the steered model says, beside what it scores."""

    prompts: int = Field(ge=1, le=8)
    """How many held-out prompts are continued, in the split's order."""
    tokens: int = Field(ge=1, le=64)
    """How many tokens each continuation adds, each the most likely one."""


class Steering(_Strict):
    """Add a direction to the residual stream at several strengths, and measure what it does.

    The direction is computed on a training split of the prompts and applied to the held-out
    rest, so no prompt receives its own difference. The effect is normalized like patching: 1
    means the steered prompt moves as far as switching from the receiver (``apply_to``) to the
    other prompt. With ``control``, a random direction of the same length, at the same strengths,
    runs alongside. With ``continuations``, the run also records what the model goes on to say.
    """

    kind: Literal["steering"] = "steering"
    apply_to: Literal["clean", "corrupt"]
    """The prompts the direction is added to; it steers them toward the other prompts."""
    direction: SteeringDirection
    """What is added: the mean difference between the prompts of each training pair, or one SAE
    feature's decoder direction."""
    coefficients: list[float] = Field(min_length=1, max_length=16)
    """The strengths: 1 adds the whole mean difference (or the feature at its typical
    activation), negative values push the other way."""
    train_fraction: float = Field(gt=0.0, lt=1.0)
    """The share of pairs, drawn with seed, that compute the direction; the rest measure it."""
    seed: Seed
    """The seed of the split into training and held-out pairs, and of the random control."""
    control: bool
    """Also add a random direction of the same length, at the same strengths."""
    continuations: Continuations | None
    """Also record what the model goes on to say; null records nothing."""

    @model_validator(mode="after")
    def _distinct_finite(self) -> Steering:
        if any(c != c or c in (float("inf"), float("-inf")) for c in self.coefficients):
            raise ValueError("steering strengths must be finite numbers")
        if len(set(self.coefficients)) != len(self.coefficients):
            raise ValueError("steering strengths must not repeat")
        return self


class PathPatching(_Strict):
    """Patch a component's effect along chosen paths only (path patching).

    For each sender (a site of the scope), the receiver prompt runs with the sender's activation
    from the source prompt and every other attention head held at its own value, so the change
    travels only through the residual stream (and through the MLPs, unless ``freeze_mlps``). The
    receivers' inputs from that run are recorded, then patched into an otherwise unchanged
    receiver run, and the metric is read there. The result is the sender's effect through those
    receivers alone. Senders must come before at least one receiver.

    With an ``edges`` scope, each edge is its own result: its sender's effect through its own
    receiver alone, with every head and MLP held (``freeze_mlps``), so the change reaches the
    receiver only directly, as the edge carries it. ``receivers`` then lists each edge's receiver.
    """

    kind: Literal["path_patching"] = "path_patching"
    direction: Direction
    receivers: list[PathReceiver] = Field(min_length=1, max_length=64)
    """Where the paths end: later heads' queries, keys or values, or the logits."""
    freeze_mlps: bool
    """Hold the MLPs at their own values too, so the change travels through the residual stream
    alone."""

    @model_validator(mode="after")
    def _distinct(self) -> PathPatching:
        keys = [r.model_dump_json() for r in self.receivers]
        if len(set(keys)) != len(keys):
            raise ValueError("path patching receivers must not repeat")
        return self


# ---------------------------------------------------------------------------------------------
# Methods that find directions, and patching along them (version 4)


class SubspacePatching(_Strict):
    """Patch only the component of each site's activation along a saved direction (or within a
    saved subspace) from the other prompt of each pair; everything at right angles to it keeps
    the receiver's own value. Beside activation patching of the whole site, it says how much of
    the site's effect that direction carries."""

    kind: Literal["subspace_patching"] = "subspace_patching"
    direction: Direction
    subspace: DirectionRef
    """The direction or subspace patched."""


class DistributedAlignmentSearch(_Strict):
    """Learn, at each site, a subspace of a few dimensions whose patching restores (or breaks)
    the behavior (distributed alignment search, DAS: Geiger et al. 2023), on a training split of
    the pairs, and measure it on the held-out rest.

    At each site, an orthonormal basis of ``dimensions`` vectors is trained by Adam to maximize
    the normalized effect of patching only the component within it, from the source prompt into
    the receiver run, over the training pairs (all of them at each step, in their batches). The
    held-out pairs are then patched within the learned subspace, beside patching the whole site
    and, with ``control``, a random subspace of the same size. A learned subspace can find an
    effect that the model doesn't use as a variable (an interpretability illusion, Makelov et al.
    2023): compare it with the whole site, and check it on other prompts."""

    kind: Literal["das"] = "das"
    direction: Direction
    dimensions: int = Field(ge=1, le=64)
    """How many dimensions the learned subspace has."""
    train_fraction: float = Field(gt=0.0, lt=1.0)
    """The share of pairs, drawn with seed, that train the subspace; the rest measure it."""
    seed: Seed
    """The seed of the split, of the subspace's starting point and of the random control."""
    steps: int = Field(ge=1, le=10_000)
    """How many steps of Adam train each site's subspace, each over every training pair."""
    learning_rate: float = Field(gt=0.0, le=1.0)
    """Adam's learning rate."""
    control: bool
    """Also patch a random subspace of the same number of dimensions."""


class SideLabels(_Strict):
    """Each pair gives two examples: its clean prompt, labelled clean, and its corrupt prompt,
    labelled corrupt. The probe reads which of the two prompts the model is reading."""

    kind: Literal["side"] = "side"


class MetaLabels(_Strict):
    """Each prompt is labelled by a field of its pair's ``meta``, such as a template, a number
    or a gender."""

    kind: Literal["meta"] = "meta"
    field: str
    """The field of each pair's meta whose value is the label."""
    prompts: Literal["clean", "corrupt", "both"]
    """Which prompts of each pair are examples, each with the pair's label."""

    @model_validator(mode="after")
    def _field(self) -> MetaLabels:
        if not CLUSTER_KEY.match(self.field):
            raise ValueError("labels.field must name a field of the prompts' meta")
        return self


ProbeLabels = Annotated[SideLabels | MetaLabels, Field(discriminator="kind")]


class Probe(_Strict):
    """Train a linear probe at each site: logistic regression that reads a label from the
    activation, on a training split of the pairs, then measure its accuracy on the held-out rest.

    Each site's activations are standardized with the training examples' mean and spread, and
    the probe (a weight vector per class and a bias) is fitted by L-BFGS to the mean cross-entropy
    plus ``regularization`` / 2 times the squared length of the weights, from zero, for
    ``iterations`` iterations, in float64. Pairs are split, not examples, so the two prompts of a
    pair are never on both sides. A probe shows that a label can be read from a site, not that the
    model reads it: patch the site, or patch along the probe's direction, to test that. With
    ``control``, a second probe is trained on the training labels shuffled with ``seed`` (a
    control task, as in Hewitt and Liang 2019), and measured on the true held-out labels: how far a
    probe gets from what it can memorize."""

    kind: Literal["probe"] = "probe"
    labels: ProbeLabels
    """What each example is labelled with."""
    train_fraction: float = Field(gt=0.0, lt=1.0)
    """The share of pairs, drawn with seed, that train the probes; the rest measure them."""
    seed: Seed
    """The seed of the split, and of the control's shuffle."""
    regularization: float = Field(ge=0.0, le=1e6)
    """The weight of the squared length of the weights in the loss (L2); 0 for none."""
    iterations: int = Field(ge=1, le=1000)
    """How many iterations of L-BFGS fit each probe."""
    control: bool
    """Also fit a probe on shuffled labels, a control for what a probe can memorize."""


Experiment = Annotated[
    ActivationPatching
    | Ablation
    | DirectLogitAttribution
    | AttributionPatching
    | Steering
    | PathPatching
    | EdgeAttribution
    | SubspacePatching
    | DistributedAlignmentSearch
    | Probe,
    Field(discriminator="kind"),
]

# Methods whose per-prompt values aren't read from the model's output: no metric.
NO_METRIC_KINDS = ("probe",)
# Methods that can read a readout metric (an internal value) instead of the output.
READOUT_KINDS = ("activation_patching", "ablation", "subspace_patching")


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


SHA256 = re.compile(r"^[0-9a-f]{64}$")
DEVICE = re.compile(r"^(auto|cpu|cuda|mps|cuda:\d{1,2})$")


class ModelRef(_Strict):
    """The model, exactly: which files, in which precision, on which device."""

    id: str = Field(
        min_length=1,
        max_length=200,
        description="The Hugging Face model id, or for a local model the name it was added under",
    )
    source: Literal["hub", "local"] = Field(
        description="Where the weights come from: Hugging Face, or a folder on this machine that "
        "was added to Logogram's local models, found by the hash of its files"
    )
    revision: str | None = Field(
        description="Exact commit. None resolves the current main branch at run time. Always "
        "None for a local model."
    )
    sha256: str | None = Field(
        description="For a local model, the hash of its files, which finds its folder on any "
        "machine it was added to. None for Hugging Face, where the revision pins the files."
    )
    dtype: Literal["float32", "float16", "bfloat16"]
    """The precision the model runs in. float16 and bfloat16 halve the memory, but round."""
    device: str = Field(description="auto, cpu, cuda, cuda:N (one GPU of several) or mps")
    process_weights: bool = Field(
        description="Fold LayerNorm weights and center writing weights and the unembedding, as "
        "TransformerLens does by default. Logit differences are unchanged by this.",
    )

    @model_validator(mode="after")
    def _source(self) -> ModelRef:
        check_revision(self.revision, "model.revision")
        if not DEVICE.match(self.device):
            raise ValueError("model.device must be auto, cpu, cuda, cuda:N (such as cuda:1) or mps")
        if self.source == "local":
            if self.sha256 is None or not SHA256.match(self.sha256):
                raise ValueError(
                    "a local model needs model.sha256: the hash of its files, 64 lowercase hex "
                    "digits"
                )
            if self.revision is not None:
                raise ValueError("a local model has no revisions: set model.revision to null")
            if "/" in self.id or "\\" in self.id:
                raise ValueError(
                    "a local model's id is the name it was added under, without slashes; its "
                    "folder is found by model.sha256"
                )
        elif self.sha256 is not None:
            raise ValueError(
                "model.sha256 is for local models: a Hugging Face model is pinned by its revision"
            )
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
    """The prompt pairs: a JSONL file in the project, one pair per line."""

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
    """The Hugging Face repository, such as jbloom/GPT2-Small-SAEs-Reformatted."""
    path: str
    """The folder in the repository that holds the SAE, such as blocks.8.hook_resid_pre; empty
    for the top level."""
    revision: str | None
    """The exact commit; null resolves the main branch when a run starts, and the run's spec
    pins it."""

    @model_validator(mode="after")
    def _relative(self) -> SAERef:
        if self.path:
            check_relative_path(self.path, "sae.path")
        check_revision(self.revision, "sae.revision")
        return self


class UserTurn(_Strict):
    """The prompt is the person's message. The template then opens the model's reply, so the
    answer is the first token of the reply."""

    kind: Literal["user_turn"] = "user_turn"
    system: str | None = Field(max_length=4000)
    """A system message before the prompt, or null for none (some templates then write their
    own)."""


class AssistantPrefix(_Strict):
    """The prompt is the beginning of the model's own reply to a fixed message from the person,
    so the model goes on with it as its reply."""

    kind: Literal["assistant_prefix"] = "assistant_prefix"
    system: str | None = Field(max_length=4000)
    """A system message before the person's, or null for none (some templates then write their
    own)."""
    user: str = Field(min_length=1, max_length=4000)
    """The person's message that every prompt replies to."""


ChatTemplate = Annotated[UserTurn | AssistantPrefix, Field(discriminator="kind")]


class _ChatTemplateValue(RootModel[ChatTemplate]):
    pass


def read_chat_template(value: Any) -> UserTurn | AssistantPrefix:
    """A chat template on its own (an analysis request's, a stored form's), checked as a spec's
    tokenization.chat_template is: from a dict, or from its JSON."""
    if isinstance(value, str | bytes):
        return _ChatTemplateValue.model_validate_json(value).root
    return _ChatTemplateValue.model_validate(value).root


class Tokenization(_Strict):
    """How each prompt becomes the tokens the model reads."""

    prepend_bos: bool
    """Put the beginning-of-sequence token before each prompt. False with a chat template, which
    writes the beginning itself, and for tokenizers that have no such token."""
    chat_template: ChatTemplate | None = Field(
        description="Wrap each prompt in the model's own chat template (the one its tokenizer "
        "ships at the model's revision), as instruct models are used; None feeds the raw text."
    )

    @model_validator(mode="after")
    def _one_beginning(self) -> Tokenization:
        if self.chat_template is not None and self.prepend_bos:
            raise ValueError(
                "with a chat template, the template writes the beginning of the text: set "
                "tokenization.prepend_bos to false"
            )
        return self


Normalization = Annotated[
    Literal["dataset_gap", "prompt_gap"],
    Field(
        description="What an effect is divided by: the gap between the two prompts' metric, "
        "averaged over the dataset (dataset_gap) or each prompt's own (prompt_gap)."
    ),
]


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
    """The prompt whose own next-token distribution the divergence is measured from."""
    normalization: Normalization


class NeuronReadout(_Strict):
    """One MLP neuron's activation, after its nonlinearity (for a gated MLP, the gate's
    activation times the up projection)."""

    kind: Literal["neuron"] = "neuron"
    layer: int = Field(ge=0)
    """The neuron's layer, from 0."""
    neuron: int = Field(ge=0)
    """The neuron's index among its layer's MLP neurons."""


class DirectionReadout(_Strict):
    """An activation's component along a saved direction of one dimension: its dot product with
    the direction's unit vector, at a stream site and layer you choose."""

    kind: Literal["direction"] = "direction"
    direction: DirectionRef
    """The saved direction, of one dimension and the residual stream's width."""
    site: StreamSiteKind
    """The activation it is read from."""
    layer: int = Field(ge=0)
    """The layer, from 0."""


class FeatureReadout(_Strict):
    """One feature of the spec's SAE: its activation, as the SAE encodes the activation it reads
    (a transcoder, its MLP's input)."""

    kind: Literal["sae_feature"] = "sae_feature"
    feature: int = Field(ge=0)
    """The feature's index in the spec's SAE."""


Readout = Annotated[NeuronReadout | DirectionReadout | FeatureReadout, Field(discriminator="kind")]


class ReadoutMetric(_Strict):
    """A value inside the model instead of its output: a neuron's activation, an activation's
    component along a saved direction, or an SAE feature's activation, at one position of each
    prompt. Patching and ablation then ask what drives that value.

    Normalized like the other metrics, an effect of 1 means the intervention moves the value as
    far as switching from the receiver prompt to the other prompt does. A site later in the model
    than the value (a later layer, or a later part of the same layer) can't change it, so its
    effect is exactly 0."""

    kind: Literal["readout"] = "readout"
    unit: Readout
    """What is read."""
    position: Annotated[LastPosition | IndexPosition | LabelPosition, Field(discriminator="kind")]
    """The token of each prompt where it is read."""
    normalization: Normalization


Metric = Annotated[
    LogitDiffMetric
    | LogProbDiffMetric
    | LogProbMetric
    | ProbMetric
    | ProbDiffMetric
    | KLMetric
    | ReadoutMetric,
    Field(discriminator="kind"),
]

METRIC_LABELS: dict[str, str] = {
    "logit_diff": "logit difference",
    "logprob_diff": "log-probability difference",
    "logprob": "answer log-probability",
    "prob": "answer probability",
    "prob_diff": "probability difference",
    "kl": "KL divergence",
    "readout": "readout",
}


def readout_text(unit: NeuronReadout | DirectionReadout | FeatureReadout) -> str:
    """What a readout reads, in words, for example 'neuron 1234 of L5's MLP'."""
    if isinstance(unit, NeuronReadout):
        return f"neuron {unit.neuron} of L{unit.layer}'s MLP"
    if isinstance(unit, DirectionReadout):
        site = unit.site.replace("_", " ")
        return f"L{unit.layer} {site} along the direction {unit.direction.name}"
    return f"SAE feature {unit.feature}"


# Metrics that read answers at the last position only: they can't score continuations of several
# tokens. (The KL divergence reads no answer; the preference it reports beside it reads them.)
SINGLE_POSITION_METRICS = ("logit_diff",)


def reads_continuations(metric: Any) -> bool:
    """Whether a run with this metric reads answers of several tokens: every metric but the logit
    difference does. A probe has no metric: the baselines beside it read answers as
    log-probabilities, which read any answer."""
    return metric is None or metric.kind not in SINGLE_POSITION_METRICS


def describe_metric(metric: Any) -> str:
    """The metric in words, for example 'KL divergence from the clean prompt's prediction'."""
    if metric is None:
        return "no metric: the probes read the activations, not the output"
    if metric.kind == "kl":
        return f"KL divergence from the {metric.target} prompt's next-token distribution"
    if metric.kind == "readout":
        return f"{readout_text(metric.unit)} at {describe_position(metric.position)}"
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
    """How many bootstrap resamples the intervals are computed from."""
    ci: float = Field(gt=0.5, lt=1.0)
    """The confidence level of the intervals and simultaneous bands, such as 0.95."""
    seed: Seed
    """Seeds the bootstrap, and Python, NumPy and PyTorch for the whole run."""
    cluster: str | None
    """A field of the prompts' meta whose values group them, or null to resample prompts."""

    @model_validator(mode="after")
    def _cluster_key(self) -> Statistics:
        if self.cluster is not None and not CLUSTER_KEY.match(self.cluster):
            raise ValueError("statistics.cluster must name a field of the prompts' meta")
        return self


class Execution(_Strict):
    """How the work is batched."""

    batch_size: int = Field(
        ge=1,
        le=4096,
        description="Rows per forward pass. Part of the spec because batch shape can change "
        "floating-point results in the last digits.",
    )


class PredictionSettings(_Strict):
    """An optional, explicitly configured diagnostic; it never changes an intervention."""

    method: Literal["final_norm_logit_lens"]
    """Each layer's resid_post read through the final normalization and the unembedding."""
    prompt_index: int = Field(ge=0)
    """The prompt pair's index in the dataset, from 0."""
    which: Literal["clean", "corrupt"]
    """The pair's clean or corrupt prompt."""
    position: Annotated[LastPosition | IndexPosition, Field(discriminator="kind")]
    """The token whose next-token predictions are read."""
    top_k: int = Field(ge=1, le=20)
    """How many of the most likely tokens are listed at each layer."""


class Spec(_Strict):
    """An experiment: what runs, on which model and prompts, and how its numbers are measured."""

    logogram_spec: Literal[4]
    """The spec's version. Version 1, 2 and 3 specs are read too, as version 4."""
    name: str = Field(min_length=1, max_length=NAME_MAX)
    """What the experiment is called, in the history and the results."""
    notes: str = ""
    """What you expect and why, saved with the spec and shown beside the results."""
    model: ModelRef
    """The model, exactly: its files, precision, device and weight processing."""
    dataset: DatasetRef
    """The prompt pairs."""
    tokenization: Tokenization
    """How each prompt becomes tokens."""
    experiment: Experiment
    """The method, with every choice that changes its numbers."""
    scope: Scope
    """The sites the method measures: one result per site."""
    metric: Metric | None
    """What is read from the model's output (or, as a readout, from inside it), and how effects
    are normalized; null for probes, which read activations instead."""
    statistics: Statistics
    """The intervals' bootstrap, confidence level and seed."""
    execution: Execution
    """How the work is batched."""
    predictions: PredictionSettings | None = None
    """An optional logit lens saved beside the results; it never changes an intervention."""
    # The SAE (or transcoder) whose features sae_feature sites and the features scope refer to.
    sae: SAERef | None = None
    """The SAE or transcoder that feature sites, the features scope and steering along a feature
    refer to; null otherwise."""

    # Fields a version 1 spec left out, which took the values version 1 gave them.
    _upgraded: list[str] = PrivateAttr(default_factory=list)

    @model_validator(mode="wrap")
    @classmethod
    def _upgrade(cls, data: Any, handler: ModelWrapValidatorHandler[Spec]) -> Spec:
        filled: list[str] = []
        if isinstance(data, dict) and data.get("logogram_spec", 1) == 1:
            data, filled = upgrade_v1(data)
        if isinstance(data, dict) and data.get("logogram_spec") == 2:
            data = upgrade_v2(data)
        if isinstance(data, dict) and data.get("logogram_spec") == 3:
            data = upgrade_v3(data)
        spec = handler(data)
        if filled:
            spec._upgraded = filled
        return spec

    @property
    def upgraded_fields(self) -> list[str]:
        return list(self._upgraded)

    @property
    def scored(
        self,
    ) -> (
        LogitDiffMetric
        | LogProbDiffMetric
        | LogProbMetric
        | ProbMetric
        | ProbDiffMetric
        | KLMetric
        | ReadoutMetric
    ):
        """The metric of a method that reads one: every method but probes."""
        if self.metric is None:
            raise ValueError("A probe has no metric: it reads activations, not the output.")
        return self.metric

    @property
    def normalization(self) -> Literal["dataset_gap", "prompt_gap"]:
        """How effects are normalized. A probe's held-out accuracy isn't normalized: its values
        go through the statistics as they are (divided by a gap of 1)."""
        return "dataset_gap" if self.metric is None else self.metric.normalization

    @model_validator(mode="after")
    def _features_have_an_sae(self) -> Spec:
        uses = (
            isinstance(self.scope, FeaturesScope)
            or (
                isinstance(self.scope, SitesScope)
                and any(s.kind == "sae_feature" for s in self.scope.sites)
            )
            or (
                isinstance(self.experiment, Steering)
                and isinstance(self.experiment.direction, FeatureDirection)
            )
        )
        if uses and self.sae is None:
            raise ValueError("SAE features need the spec's sae: the SAE they belong to")
        return self

    @model_validator(mode="after")
    def _sets_fit(self) -> Spec:
        """Sets of sites are replaced at once with values computed before the forward pass: the
        methods that compute their values from the run's own activations, or that learn or read
        one site at a time, act on single sites."""
        if not isinstance(self.scope, SiteSetsScope):
            return self
        exp = self.experiment
        if isinstance(exp, Ablation) and isinstance(
            exp.baseline, ScaleBaseline | ProjectionBaseline
        ):
            raise ValueError(
                "sets of sites are ablated with a zero, mean or resample baseline; scaling and "
                "directional ablation act on one site at a time"
            )
        if isinstance(exp, Probe | DistributedAlignmentSearch | SubspacePatching):
            raise ValueError(
                "probes, distributed alignment search and subspace patching act on one site at a "
                "time: use a scope of single sites, not sets of sites"
            )
        return self

    @model_validator(mode="after")
    def _metric_fits(self) -> Spec:
        kind = self.experiment.kind
        if kind in NO_METRIC_KINDS and self.metric is not None:
            raise ValueError(
                "a probe reads activations, not the model's output: set metric to null"
            )
        if kind not in NO_METRIC_KINDS and self.metric is None:
            raise ValueError("this method reads the model's output: the spec needs a metric")
        metric = self.metric
        if isinstance(metric, ReadoutMetric):
            if kind not in READOUT_KINDS:
                raise ValueError(
                    "a readout metric is read by activation patching, ablation and subspace "
                    "patching; use another metric with this method"
                )
            if isinstance(self.scope, FeaturesScope | SiteSetsScope):
                raise ValueError(
                    "a readout metric is read site by site: use a scope of single sites, not "
                    "sets of sites or SAE features"
                )
            if isinstance(metric.unit, FeatureReadout) and self.sae is None:
                raise ValueError("an SAE feature readout needs the spec's sae")
        return self

    @model_validator(mode="after")
    def _edges_fit(self) -> Spec:
        exp, scope = self.experiment, self.scope
        if isinstance(exp, EdgeAttribution) != isinstance(scope, GraphScope):
            raise ValueError(
                "edge attribution estimates the edges of the model's graph: use the graph scope "
                "with it, and the graph scope with it alone"
            )
        if isinstance(scope, EdgesScope):
            if not isinstance(exp, PathPatching):
                raise ValueError("edges are measured one at a time by path patching: choose it")
            if not exp.freeze_mlps:
                raise ValueError(
                    "an edge reaches its receiver directly, so path patching it holds every head "
                    "and MLP: set freeze_mlps to true"
                )
            receivers = list(dict.fromkeys(e.receiver.model_dump_json() for e in scope.edges))
            if [r.model_dump_json() for r in exp.receivers] != receivers:
                raise ValueError(
                    "with an edges scope, experiment.receivers lists each edge's receiver once, "
                    "in the order the edges first name them"
                )
        return self

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2) + "\n"

    @classmethod
    def from_path(cls, path: str | Path) -> Spec:
        raw = read_limited(Path(path), JSON_LIMIT, "A spec is a few kilobytes: check the file.")
        return cls.model_validate_json(raw)


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
        if kind == "features":
            fill(scope, "choose_on", None, "scope.choose_on", report=False)
            fill(scope, "seed", None, "scope.seed", report=False)
        if kind == "layer_position":
            fill(scope, "site", "resid_pre", "scope.site")
            fill(scope, "positions", "each", "scope.positions")
        if kind == "layer_components":
            fill(scope, "components", ["attn_out", "mlp_out"], "scope.components")
        if kind == "sites":
            for i, site in enumerate(scope.get("sites") or []):
                fill(site, "position", all_positions, f"scope.sites[{i}].position")
    return data, filled


# ---------------------------------------------------------------------------------------------
# Version 2 specs


def upgrade_v2(data: dict[str, Any]) -> dict[str, Any]:
    """A version 2 spec as version 3. What version 3 added is off, as it was in version 2: no
    chat template, a model from Hugging Face, steering along the mean difference without
    continuations. These didn't exist to be chosen, so they aren't listed as filled."""
    data = copy.deepcopy(data)
    data["logogram_spec"] = 3
    tokenization = data.get("tokenization")
    if isinstance(tokenization, dict):
        tokenization.setdefault("chat_template", None)
    model = data.get("model")
    if isinstance(model, dict):
        model.setdefault("source", "hub")
        model.setdefault("sha256", None)
    experiment = data.get("experiment")
    if isinstance(experiment, dict) and experiment.get("kind") == "steering":
        experiment.setdefault("direction", {"kind": "mean_difference"})
        experiment.setdefault("continuations", None)
    return data


# ---------------------------------------------------------------------------------------------
# Version 3 specs


def upgrade_v3(data: dict[str, Any]) -> dict[str, Any]:
    """A version 3 spec as version 4. Version 4 added methods, metrics and sources that a version
    3 spec can't name, and fields that are off when it doesn't: an SAE from Hugging Face, the
    final-norm logit lens. These didn't exist to be chosen, so they aren't listed as filled."""
    data = copy.deepcopy(data)
    data["logogram_spec"] = 4
    return data


def _show(value: Any) -> str:
    return json.dumps(value)


# ---------------------------------------------------------------------------------------------
# Descriptions


def describe_curve(curve: FaithfulnessCurve) -> str:
    """The faithfulness curve in words, for example 'curve over the 1 to 500 strongest edges'."""
    counts = curve.counts
    span = f"{counts[0]:,}" if len(counts) == 1 else f"{counts[0]:,} to {counts[-1]:,}"
    patched = "patching them" if curve.patch == "chosen" else "patching every other edge"
    draws = curve.control.draws
    control = f"{draws} random draw{'s' if draws != 1 else ''}, seed {curve.control.seed}"
    return f"faithfulness of the {span} strongest edges, {patched} ({control})"


def labels_text(labels: SideLabels | MetaLabels) -> str:
    """What a probe reads, for example 'which prompt (clean or corrupt)'."""
    if isinstance(labels, SideLabels):
        return "which prompt it is (clean or corrupt)"
    prompts = {"clean": "clean prompts", "corrupt": "corrupt prompts", "both": "both prompts"}
    return f"meta.{labels.field} ({prompts[labels.prompts]})"


def describe_intervention(
    exp: ActivationPatching
    | Ablation
    | DirectLogitAttribution
    | AttributionPatching
    | Steering
    | PathPatching
    | EdgeAttribution
    | SubspacePatching
    | DistributedAlignmentSearch
    | Probe,
) -> str:
    """The intervention in a few words, for example 'Resample-ablate (10 corrupt donors, seed 0)'."""
    if isinstance(exp, Probe):
        control = ", against shuffled labels" if exp.control else ""
        return (
            f"Linear probe for {labels_text(exp.labels)} ({exp.train_fraction:.0%} of pairs "
            f"train, seed {exp.seed}, L2 {exp.regularization:g}{control})"
        )
    if isinstance(exp, SubspacePatching):
        arrow = "clean → corrupt" if exp.direction == "clean_to_corrupt" else "corrupt → clean"
        return f"Patch {arrow} along the direction {exp.subspace.name}"
    if isinstance(exp, DistributedAlignmentSearch):
        arrow = "clean → corrupt" if exp.direction == "clean_to_corrupt" else "corrupt → clean"
        dims = f"{exp.dimensions} dimension{'s' if exp.dimensions != 1 else ''}"
        control = ", random control" if exp.control else ""
        return (
            f"Learn a subspace of {dims} and patch {arrow} within it ({exp.train_fraction:.0%} "
            f"of pairs train, {exp.steps} steps, seed {exp.seed}{control})"
        )
    if isinstance(exp, ActivationPatching):
        return (
            "Patch clean → corrupt"
            if exp.direction == "clean_to_corrupt"
            else "Patch corrupt → clean"
        )
    if isinstance(exp, DirectLogitAttribution):
        return f"Direct logit attribution ({exp.prompts} prompts)"
    if isinstance(exp, EdgeAttribution):
        arrow = "clean → corrupt" if exp.direction == "clean_to_corrupt" else "corrupt → clean"
        how = (
            f"integrated gradients, {exp.steps} steps"
            if exp.method == "integrated_gradients"
            else "one gradient"
        )
        curve = "" if exp.curve is None else f"; {describe_curve(exp.curve)}"
        return f"Edge attribution, estimated with {how} ({arrow}{curve})"
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
        if isinstance(exp.direction, SavedDirection):
            return (
                f"Steer {exp.apply_to} prompts along the direction {exp.direction.direction.name} "
                f"({strengths}; strength 1 is the pairs' mean difference along it on "
                f"{exp.train_fraction:.0%} of the pairs, seed {exp.seed}{control})"
            )
        if isinstance(exp.direction, FeatureDirection):
            return (
                f"Steer {exp.apply_to} prompts along SAE feature {exp.direction.feature} "
                f"({strengths}; strength 1 is its typical activation on {exp.train_fraction:.0%} "
                f"of the pairs, seed {exp.seed}{control})"
            )
        return (
            f"Steer {exp.apply_to} prompts toward {toward} ({strengths}; "
            f"{exp.train_fraction:.0%} of pairs train the direction, seed {exp.seed}{control})"
        )
    b = exp.baseline
    if isinstance(b, ZeroBaseline):
        return "Zero-ablate"
    if isinstance(b, MeanBaseline):
        return f"Mean-ablate ({b.reference} mean)"
    if isinstance(b, ScaleBaseline):
        return f"Scale by {b.factor:g}".replace("-", "−")
    if isinstance(b, ProjectionBaseline):
        to = "removed" if b.reference is None else f"set to its {b.reference} mean"
        return f"Ablate the direction {b.direction.name} ({to})"
    return f"Resample-ablate ({b.donors} {b.pool} donors, seed {b.seed})"


def layers_text(layers: list[int]) -> str:
    """Layers as runs, for example "0–3, 7"."""
    runs: list[list[int]] = []
    for layer in layers:
        if runs and layer == runs[-1][-1] + 1:
            runs[-1].append(layer)
        else:
            runs.append([layer])
    return ", ".join(f"{r[0]}" if len(r) == 1 else f"{r[0]}–{r[-1]}" for r in runs)


def graph_text(scope: GraphScope) -> str:
    """A graph scope in words: 'every edge of the graph', or the edges into some layers."""
    if scope.layers is None:
        return "every edge of the graph" if scope.logits else "every edge but those into the logits"
    one = len(scope.layers) == 1
    where = f"layer{'' if one else 's'} {layers_text(scope.layers)}"
    logits = " and the logits" if scope.logits else ""
    return f"the edges into {where}{logits}"


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
        if scope.choose_on is not None:
            where += (
                f" chosen on {scope.choose_on:.0%} of the prompts (seed {scope.seed}) and "
                "reported on the rest"
            )
    elif isinstance(scope, SiteSetsScope):
        n = len(scope.sets)
        where = f"{n} set{'s' if n != 1 else ''} of sites, each at once"
    elif isinstance(scope, HeadInputsScope):
        inputs = {"q": "every head's query", "k": "every key", "v": "every value"}[scope.input]
        where = f"{inputs}, {describe_position(scope.position)}"
    elif isinstance(scope, NeuronsScope):
        layers = ", ".join(str(layer) for layer in scope.layers)
        named = "layer" if len(scope.layers) == 1 else "layers"
        where = f"every MLP neuron of {named} {layers}, {describe_position(scope.position)}"
    elif isinstance(scope, GraphScope):
        where = graph_text(scope)
    elif isinstance(scope, EdgesScope):
        n = len(scope.edges)
        where = f"{n} edge{'s' if n != 1 else ''}, one at a time"
    else:
        where = f"{len(scope.sites)} chosen site{'s' if len(scope.sites) != 1 else ''}"
    if spec.metric is None or spec.metric.kind == "logit_diff":
        metric = ""
    elif isinstance(spec.metric, ReadoutMetric):
        metric = f" · reading {readout_text(spec.metric.unit)}"
    else:
        metric = f" · {METRIC_LABELS[spec.metric.kind]}"
    chat = spec.tokenization.chat_template
    wrapped = (
        ""
        if chat is None
        else " · in the chat template, as the person's message"
        if chat.kind == "user_turn"
        else " · in the chat template, as the start of the reply"
    )
    return f"{what} · {where}{metric}{wrapped}"
