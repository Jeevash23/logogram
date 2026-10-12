"""Expand a spec's scope into concrete sites, and lay them out as a grid.

Every result is a grid of rows x columns (layer x head, layer x position, layer x component, or
one row per chosen site). The model map and the heatmaps draw from this layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from logogram.backends.base import ModelInfo
from logogram.prompts import PreparedPrompt, common_labels, uneven_text
from logogram.spec import (
    HEAD_INPUTS,
    AllPositions,
    FeaturesScope,
    HeadInputsScope,
    HeadsScope,
    IndexPosition,
    LabelPosition,
    LastPosition,
    LayerComponentsScope,
    LayerPositionScope,
    NeuronsScope,
    Site,
    SiteSet,
    SiteSetsScope,
    SitesScope,
    Spec,
)

COMPONENT_LABELS = {
    "resid_pre": "resid pre",
    "resid_mid": "resid mid",
    "resid_post": "resid post",
    "attn_out": "attn",
    "mlp_out": "mlp",
}


class ScopeError(ValueError):
    pass


@dataclass
class ResolvedSite:
    index: int
    site: Site
    row: int
    col: int
    label: str
    # A variant of the same site, for methods that measure one site several ways (a steering
    # strength, or its random control). ``variant_key`` names it, for example "×2".
    variant: dict[str, Any] | None = None
    variant_key: str | None = None
    # A set of sites intervened on together (a row of a site_sets scope); ``site`` is then its
    # first site, and the row reads as the whole set.
    site_set: SiteSet | None = None

    @property
    def kind(self) -> str:
        return "site_set" if self.site_set is not None else self.site.kind

    @property
    def layer(self) -> int:
        return -1 if self.site_set is not None else self.site.layer

    @property
    def head(self) -> int | None:
        return None if self.site_set is not None else self.site.head

    def position_key(self) -> str:
        return "set" if self.site_set is not None else position_key(self.site.position)

    def to_dict(self) -> dict[str, Any]:
        members = None
        if self.site_set is not None:
            members = [s.model_dump(mode="json") for s in self.site_set.sites]
        return {
            "index": self.index,
            "kind": self.kind,
            "layer": self.layer,
            "head": self.head,
            "feature": None if self.site_set is not None else self.site.feature,
            "neuron": None if self.site_set is not None else self.site.neuron,
            "position": self.site.position.model_dump(),
            "position_key": self.position_key(),
            "row": self.row,
            "col": self.col,
            "label": self.label,
            "variant": self.variant,
            "variant_key": self.variant_key,
            "members": members,
        }


def position_key(position: AllPositions | LastPosition | IndexPosition | LabelPosition) -> str:
    if isinstance(position, AllPositions):
        return "all"
    if isinstance(position, LastPosition):
        return "last"
    if isinstance(position, IndexPosition):
        return str(position.index)
    return position.label


def site_label(site: Site) -> str:
    pos = "" if isinstance(site.position, AllPositions) else f" @ {position_key(site.position)}"
    if site.kind == "head":
        return f"L{site.layer} H{site.head}{pos}"
    if site.kind == "sae_feature":
        return f"L{site.layer} F{site.feature}{pos}"
    if site.kind == "q":
        return f"L{site.layer} H{site.head} q{pos}"
    if site.kind in ("k", "v"):
        # A key/value head, which every query head of its group reads.
        return f"L{site.layer} KV{site.head} {site.kind}{pos}"
    if site.kind == "neuron":
        return f"L{site.layer} N{site.neuron}{pos}"
    return f"L{site.layer} {COMPONENT_LABELS[site.kind]}{pos}"


def site_unit(site: Site) -> int | None:
    """The site's unit of its layer's activation: its head (query head, key/value head) for head,
    q, k and v sites, its neuron for a neuron site; None for a site that is a whole vector."""
    if site.kind == "neuron":
        return site.neuron
    if site.kind == "head" or site.kind in HEAD_INPUTS:
        return site.head
    return None


def kv_heads(info: ModelInfo) -> int:
    """The model's key/value heads: as many as its query heads without grouped-query attention."""
    return int(info.extra.get("n_key_value_heads") or info.n_heads)


def unit_count(kind: str, info: ModelInfo) -> int | None:
    """How many units a layer's activation of ``kind`` has: heads for heads' outputs and queries,
    key/value heads for keys and values, neurons for neurons; None for a whole vector."""
    if kind in ("head", "q"):
        return info.n_heads
    if kind in ("k", "v"):
        return kv_heads(info)
    if kind == "neuron":
        return info.d_mlp or 0
    return None


def activation_shape(kind: str, info: ModelInfo) -> tuple[int, ...]:
    """The shape of a layer's activation of ``kind`` at one position."""
    units = unit_count(kind, info)
    if units is None:
        return (info.d_model,)
    return (units,) if kind == "neuron" else (units, info.d_head)


def query_heads(info: ModelInfo, kv_head: int) -> list[int]:
    """The query heads that read a key/value head: query head h reads key/value head
    h // (query heads per key/value head), as the backend measured when the model loaded."""
    group = max(1, info.n_heads // kv_heads(info))
    return list(range(kv_head * group, (kv_head + 1) * group))


def missing_kind(kind: str, info: ModelInfo) -> ScopeError:
    """Why the loaded model has no ``kind`` sites, as the backend measured it."""
    note = (info.extra.get("site_notes") or {}).get(kind)
    if note:
        return ScopeError(note)
    return ScopeError(f"This model has no {kind} site in TransformerLens.")


def resolve_position(
    position: AllPositions | LastPosition | IndexPosition | LabelPosition,
    prompt: PreparedPrompt,
    side: str = "clean",
) -> int | None:
    """The token index a position refers to in this prompt's ``side`` (its clean or corrupt
    prompt), or None for all positions. In a pair of equal length both sides agree, unless the
    dataset labels the corrupt prompt's positions itself."""
    if isinstance(position, AllPositions):
        return None
    length = prompt.length_of(side)
    if isinstance(position, LastPosition):
        return length - 1
    if isinstance(position, IndexPosition):
        idx = position.index if position.index >= 0 else length + position.index
        if not 0 <= idx < length:
            raise ScopeError(
                f"Token index {position.index} is outside prompt {prompt.index}, which has "
                f"{length} tokens."
            )
        return idx
    labels = prompt.labels_of(side)
    if position.label not in labels:
        raise ScopeError(
            f"Prompt {prompt.index} has no position named {position.label!r}. Add it to the "
            "dataset's positions, or choose another position."
        )
    return labels[position.label]


# What carries a value from one prompt into another, in words, for refusing positions that don't
# line up between prompts of different lengths.
CARRIED = {
    "patch": "Patching carries each activation from one prompt of a pair into the other",
    "estimate": "Attribution patching compares each activation with the other prompt's",
    "path": "Path patching carries the sender's activation from one prompt of a pair into the "
    "other",
    "mean": "Mean ablation carries each activation's mean over other prompts into the prompt",
    "resample": "Resample ablation carries each activation from other prompts into the prompt",
    "steer": "Steering by the mean difference compares each prompt with the other of its pair",
}


def check_lined_up(
    positions: list[AllPositions | LastPosition | IndexPosition | LabelPosition],
    prompts: list[PreparedPrompt],
    carried: str,
) -> None:
    """Refuse positions that need the prompts' positions to line up (every position, or a token
    index) when some pairs have prompts of different token lengths: a position is then matched
    from one prompt to another by its label, or as the last token, and only so.

    ``carried`` says how a value goes from one prompt into another (a key of :data:`CARRIED`)."""
    uneven = [p.index for p in prompts if not p.even]
    if not uneven:
        return
    bad = next((p for p in positions if isinstance(p, AllPositions | IndexPosition)), None)
    if bad is None:
        return
    what = "every position" if isinstance(bad, AllPositions) else f"token index {bad.index}"
    labels = common_labels(prompts)
    named = f" or a labelled position ({', '.join(labels)})" if labels else ""
    raise ScopeError(
        f"{uneven_text(uneven, len(prompts))}, so their positions don't line up by index. "
        f"{CARRIED[carried]}, and at {what} that has no meaning across different lengths. "
        f"Choose the last token{named}, which are matched from one prompt to the other, or use "
        "pairs of equal length."
    )


def _check_site(site: Site, info: ModelInfo) -> None:
    if site.layer >= info.n_layers:
        raise ScopeError(
            f"Layer {site.layer} doesn't exist; this model has {info.n_layers} layers."
        )
    if site.kind == "sae_feature":
        return  # checked against the SAE, which knows its layer and features
    if site.kind not in info.site_kinds:
        raise missing_kind(site.kind, info)
    if site.kind in ("head", "q") and site.head is not None and site.head >= info.n_heads:
        raise ScopeError(f"Head {site.head} doesn't exist; this model has {info.n_heads} heads.")
    if site.kind in ("k", "v") and site.head is not None and site.head >= kv_heads(info):
        raise ScopeError(
            f"Key/value head {site.head} doesn't exist; this model has {kv_heads(info)} key/value "
            f"heads{_shared(info)}."
        )
    if site.kind == "neuron" and site.neuron is not None and site.neuron >= (info.d_mlp or 0):
        raise ScopeError(
            f"Neuron {site.neuron} doesn't exist; this model's MLPs have {info.d_mlp or 0} neurons."
        )


def _shared(info: ModelInfo) -> str:
    group = info.n_heads // kv_heads(info)
    return f", each read by {group} of its {info.n_heads} query heads" if group > 1 else ""


def expand_scope(
    spec: Spec, info: ModelInfo, prompts: list[PreparedPrompt]
) -> tuple[list[ResolvedSite], dict[str, Any]]:
    scope = spec.scope
    sites: list[ResolvedSite] = []

    def add(site: Site, row: int, col: int) -> None:
        sites.append(ResolvedSite(len(sites), site, row, col, site_label(site)))

    if isinstance(scope, HeadsScope):
        for layer in range(info.n_layers):
            for head in range(info.n_heads):
                add(Site(kind="head", layer=layer, head=head, position=scope.position), layer, head)
        layout = {
            "kind": "heads",
            "row_title": "Layer",
            "col_title": "Head",
            "rows": [{"key": str(r), "label": str(r)} for r in range(info.n_layers)],
            "cols": [{"key": str(c), "label": str(c)} for c in range(info.n_heads)],
        }
    elif isinstance(scope, LayerPositionScope):
        if scope.site not in info.site_kinds:
            raise ScopeError(f"This model has no {scope.site} site in TransformerLens.")
        cols: list[dict[str, Any]] = []
        positions: list[AllPositions | LastPosition | IndexPosition | LabelPosition] = []
        if scope.positions == "each":
            lengths = sorted({p.length_of(side) for p in prompts for side in ("clean", "corrupt")})
            if len(lengths) > 1:
                raise ScopeError(
                    f"Prompts have different token lengths ({lengths[0]}–{lengths[-1]}), so "
                    "positions don't line up. Use labelled positions instead, or a dataset "
                    "whose prompts share one template length."
                )
            first = prompts[0]
            differs = set(first.differing_positions())
            for j in range(first.length):
                positions.append(IndexPosition(index=j))
                cols.append(
                    {
                        "key": str(j),
                        "label": first.clean.tokens[j],
                        "position": j,
                        "clean": first.clean.tokens[j],
                        "corrupt": first.corrupt.tokens[j],
                        "differs": j in differs,
                    }
                )
        else:
            labels = common_labels(prompts)
            if not labels:
                raise ScopeError(
                    "The dataset has no named positions shared by every prompt. Generate a "
                    "task's dataset (they name their positions), or add positions to your JSONL."
                )
            for label in labels:
                positions.append(LabelPosition(label=label))
                cols.append({"key": label, "label": label})
        for layer in range(info.n_layers):
            for j, pos in enumerate(positions):
                add(Site(kind=scope.site, layer=layer, position=pos), layer, j)
        layout = {
            "kind": "layer_position",
            "site": scope.site,
            "row_title": "Layer",
            "col_title": "Position",
            "rows": [{"key": str(r), "label": str(r)} for r in range(info.n_layers)],
            "cols": cols,
        }
    elif isinstance(scope, LayerComponentsScope):
        for kind in scope.components:
            if kind not in info.site_kinds:
                raise ScopeError(f"This model has no {kind} site in TransformerLens.")
        for layer in range(info.n_layers):
            for j, kind in enumerate(scope.components):
                add(Site(kind=kind, layer=layer, position=scope.position), layer, j)
        layout = {
            "kind": "layer_components",
            "row_title": "Layer",
            "col_title": "Component",
            "rows": [{"key": str(r), "label": str(r)} for r in range(info.n_layers)],
            "cols": [{"key": k, "label": COMPONENT_LABELS[k]} for k in scope.components],
        }
    elif isinstance(scope, SitesScope):
        for i, site in enumerate(scope.sites):
            add(site, i, 0)
        layout = {
            "kind": "sites",
            "row_title": "Site",
            "col_title": "",
            "rows": [{"key": str(i), "label": site_label(s)} for i, s in enumerate(scope.sites)],
            "cols": [{"key": "effect", "label": "effect"}],
        }
    elif isinstance(scope, HeadInputsScope):
        kind = scope.input
        if kind not in info.site_kinds:
            raise missing_kind(kind, info)
        count = info.n_heads if kind == "q" else kv_heads(info)
        for layer in range(info.n_layers):
            for head in range(count):
                add(Site(kind=kind, layer=layer, head=head, position=scope.position), layer, head)
        cols = []
        for head in range(count):
            col: dict[str, Any] = {"key": str(head), "label": str(head)}
            if kind != "q":
                col["query_heads"] = query_heads(info, head)
            cols.append(col)
        layout = {
            "kind": "head_inputs",
            "site": kind,
            "row_title": "Layer",
            "col_title": "Head" if kind == "q" else "Key/value head",
            "rows": [{"key": str(r), "label": str(r)} for r in range(info.n_layers)],
            "cols": cols,
        }
    elif isinstance(scope, NeuronsScope):
        if "neuron" not in info.site_kinds or not info.d_mlp:
            raise missing_kind("neuron", info)
        for layer in scope.layers:
            if layer >= info.n_layers:
                raise ScopeError(
                    f"Layer {layer} doesn't exist; this model has {info.n_layers} layers."
                )
        for row, layer in enumerate(scope.layers):
            for neuron in range(info.d_mlp):
                # Thousands of sites per layer, every field known to be valid: built directly.
                site = Site.model_construct(
                    kind="neuron", layer=layer, neuron=neuron, position=scope.position
                )
                add(site, row, neuron)
        layout = {
            "kind": "neurons",
            "site": "neuron",
            "row_title": "Layer",
            "col_title": "Neuron",
            # Each layer's neurons are its columns, too many to list: rows say how many.
            "rows": [
                {"key": str(layer), "label": str(layer), "neurons": info.d_mlp}
                for layer in scope.layers
            ],
            "cols": [],
        }
    elif isinstance(scope, SiteSetsScope):
        raise ScopeError(
            "Sets of sites are intervened on together, which activation patching and ablation do. "
            "Choose one of those."
        )
    elif isinstance(scope, FeaturesScope):
        raise ScopeError(
            "Sweeping every SAE feature needs attribution patching, which estimates them all at "
            "once. To patch features for real, choose them as sites."
        )
    else:  # pragma: no cover - exhaustive
        raise ScopeError(f"Unknown scope {scope!r}")

    checked: set[str] = set()
    for rs in sites:
        _check_site(rs.site, info)
        key = rs.site.position.model_dump_json()
        if key not in checked:
            checked.add(key)
            for prompt in prompts:
                for side in ("clean", "corrupt"):
                    resolve_position(rs.site.position, prompt, side)
    return sites, layout
