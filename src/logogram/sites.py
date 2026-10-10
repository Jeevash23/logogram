"""Expand a spec's scope into concrete sites, and lay them out as a grid.

Every result is a grid of rows x columns (layer x head, layer x position, layer x component, or
one row per chosen site). The model map and the heatmaps draw from this layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from logogram.backends.base import ModelInfo
from logogram.prompts import PreparedPrompt, common_labels
from logogram.spec import (
    AllPositions,
    FeaturesScope,
    HeadsScope,
    IndexPosition,
    LabelPosition,
    LastPosition,
    LayerComponentsScope,
    LayerPositionScope,
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
    return f"L{site.layer} {COMPONENT_LABELS[site.kind]}{pos}"


def resolve_position(
    position: AllPositions | LastPosition | IndexPosition | LabelPosition, prompt: PreparedPrompt
) -> int | None:
    """The token index a position refers to in this prompt, or None for all positions."""
    if isinstance(position, AllPositions):
        return None
    if isinstance(position, LastPosition):
        return prompt.length - 1
    if isinstance(position, IndexPosition):
        idx = position.index if position.index >= 0 else prompt.length + position.index
        if not 0 <= idx < prompt.length:
            raise ScopeError(
                f"Token index {position.index} is outside prompt {prompt.index}, which has "
                f"{prompt.length} tokens."
            )
        return idx
    if position.label not in prompt.labels:
        raise ScopeError(
            f"Prompt {prompt.index} has no position named {position.label!r}. Add it to the "
            "dataset's positions, or choose another position."
        )
    return prompt.labels[position.label]


def _check_site(site: Site, info: ModelInfo) -> None:
    if site.layer >= info.n_layers:
        raise ScopeError(
            f"Layer {site.layer} doesn't exist; this model has {info.n_layers} layers."
        )
    if site.kind == "sae_feature":
        return  # checked against the SAE, which knows its layer and features
    if site.kind not in info.site_kinds:
        raise ScopeError(f"This model has no {site.kind} site in TransformerLens.")
    if site.kind == "head" and site.head is not None and site.head >= info.n_heads:
        raise ScopeError(f"Head {site.head} doesn't exist; this model has {info.n_heads} heads.")


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
            lengths = sorted({p.length for p in prompts})
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
                    "The dataset has no named positions shared by every prompt. Generate an "
                    "IOI dataset, or add positions to your JSONL."
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
                resolve_position(rs.site.position, prompt)
    return sites, layout
