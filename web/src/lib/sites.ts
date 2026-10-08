// Selections and how they map onto sites of a run.

import type { Layout, SiteBase, SiteKind } from "../api/types";

/** A component on the model map, optionally at a position (for layer × position runs). */
export interface Selection {
  layer: number;
  part: MapPart;
  head?: number;
  positionKey?: string;
  /** Which residual stream site (before, between or after the layer's blocks), when known. The
   * map draws the three in one column, so a run can have several on one map cell. */
  kind?: ResidKind;
  /** Which variant of the site, such as a steering strength ("×2" or "random ×2"). */
  variantKey?: string;
  /** A feature of the run's SAE (part "feature"). */
  feature?: number;
}

export type MapPart = "resid" | "head" | "attn" | "mlp" | "feature";
export type ResidKind = "resid_pre" | "resid_mid" | "resid_post";

export function partOfKind(kind: SiteKind): MapPart {
  if (kind === "head") return "head";
  if (kind === "sae_feature") return "feature";
  if (kind === "attn_out") return "attn";
  if (kind === "mlp_out") return "mlp";
  return "resid";
}

export function isResidKind(kind: SiteKind): kind is ResidKind {
  return kind === "resid_pre" || kind === "resid_mid" || kind === "resid_post";
}

export function selectionOfSite(site: SiteBase): Selection {
  return {
    layer: site.layer,
    part: partOfKind(site.kind),
    head: site.head ?? undefined,
    positionKey: site.position_key === "all" ? undefined : site.position_key,
    kind: isResidKind(site.kind) ? site.kind : undefined,
    variantKey: site.variant_key ?? undefined,
    feature: site.feature ?? undefined,
  };
}

export function sameSelection(a: Selection | null, b: Selection | null): boolean {
  if (!a || !b) return a === b;
  return (
    a.layer === b.layer &&
    a.part === b.part &&
    (a.head ?? -1) === (b.head ?? -1) &&
    (a.positionKey ?? "") === (b.positionKey ?? "") &&
    (a.kind ?? "") === (b.kind ?? "") &&
    (a.variantKey ?? "") === (b.variantKey ?? "") &&
    (a.feature ?? -1) === (b.feature ?? -1)
  );
}

/** Sites of a run that sit on the same map component as the selection. */
export function sitesOnComponent<T extends SiteBase>(sites: T[], sel: Selection): T[] {
  return sites.filter(
    (s) =>
      s.layer === sel.layer &&
      partOfKind(s.kind) === sel.part &&
      (sel.part !== "head" || s.head === sel.head) &&
      (sel.part !== "feature" || s.feature === sel.feature) &&
      (sel.kind === undefined || s.kind === sel.kind) &&
      (sel.variantKey === undefined || (s.variant_key ?? undefined) === sel.variantKey),
  );
}

/** The site a selection points at, if the run has one. */
export function findSite<T extends SiteBase>(sites: T[], sel: Selection | null): T | null {
  if (!sel) return null;
  const candidates = sitesOnComponent(sites, sel);
  if (candidates.length === 0) return null;
  if (sel.positionKey !== undefined) {
    return candidates.find((s) => s.position_key === sel.positionKey) ?? null;
  }
  return candidates.length === 1 ? candidates[0] : null;
}

/** Keep as much of a selection as another run can show: the same component, and its position
 * and residual site when that run has them. */
export function fitSelection<T extends SiteBase>(sites: T[], sel: Selection): Selection {
  const tries: Selection[] = [
    sel,
    { ...sel, positionKey: undefined },
    { ...sel, kind: undefined },
    { ...sel, positionKey: undefined, kind: undefined },
    { ...sel, variantKey: undefined },
    { ...sel, variantKey: undefined, positionKey: undefined, kind: undefined },
  ];
  return tries.find((t) => findSite(sites, t)) ?? { ...sel, positionKey: undefined };
}

/** What a site measures, independent of its index in a run: comparable across runs. */
export function siteKey(site: { kind: SiteKind; layer: number; head?: number | null; feature?: number | null; position_key: string; variant_key?: string | null }): string {
  return `${site.kind}|${site.layer}|${site.head ?? ""}|${site.feature ?? ""}|${site.position_key}|${site.variant_key ?? ""}`;
}

export function siteAt<T extends SiteBase>(sites: T[], row: number, col: number): T | null {
  return sites.find((s) => s.row === row && s.col === col) ?? null;
}

export function componentLabel(sel: Selection): string {
  const pos = `${sel.positionKey !== undefined ? ` @ ${sel.positionKey}` : ""}${sel.variantKey ? ` ${sel.variantKey}` : ""}`;
  switch (sel.part) {
    case "head":
      return `L${sel.layer} H${sel.head}${pos}`;
    case "attn":
      return `L${sel.layer} attn${pos}`;
    case "mlp":
      return `L${sel.layer} mlp${pos}`;
    case "feature":
      return `L${sel.layer} F${sel.feature}${pos}`;
    default:
      return `L${sel.layer} ${sel.kind ? KIND_SHORT[sel.kind] : "resid"}${pos}`;
  }
}

export const KIND_NAMES: Record<SiteKind, string> = {
  resid_pre: "residual stream before the layer",
  resid_mid: "residual stream between attention and MLP",
  resid_post: "residual stream after the layer",
  attn_out: "attention output",
  mlp_out: "MLP output",
  head: "attention head output (z)",
  sae_feature: "SAE feature",
};

export const KIND_SHORT: Record<SiteKind, string> = {
  resid_pre: "resid pre",
  resid_mid: "resid mid",
  resid_post: "resid post",
  attn_out: "attn out",
  mlp_out: "mlp out",
  head: "head",
  sae_feature: "feature",
};

export function layoutTitle(layout: Layout): string {
  switch (layout.kind) {
    case "heads":
      return "Layer × head";
    case "layer_position":
      return "Layer × position";
    case "layer_components":
      return "Layer × component";
    case "steering":
      return "Layer × strength";
    default:
      return "Chosen sites";
  }
}

/** Parse "L9 H6", "9.6", "l9h6" into a head selection. */
export function parseHeadQuery(q: string): { layer: number; head: number } | null {
  const m =
    q.trim().match(/^l?\s*(\d+)\s*[.\s,h]\s*h?\s*(\d+)$/i) ?? q.trim().match(/^l(\d+)h(\d+)$/i);
  if (!m) return null;
  return { layer: Number(m[1]), head: Number(m[2]) };
}

/** Move a selection with the arrow keys: across [resid, heads…, attn, mlp], and between layers. */
export function nextSelection(
  cur: Selection | null,
  key: string,
  nLayers: number,
  nHeads: number,
): Selection | null {
  const order: { part: MapPart; head?: number }[] = [
    { part: "resid" },
    ...Array.from({ length: nHeads }, (_, h) => ({ part: "head" as const, head: h })),
    { part: "attn" },
    { part: "mlp" },
  ];
  if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(key)) return null;
  if (!cur) return { layer: 0, part: "head", head: 0 };
  const start = cur;
  let i = order.findIndex((o) => o.part === start.part && (o.part !== "head" || o.head === start.head));
  if (i < 0) i = 1;
  let layer = start.layer;
  switch (key) {
    case "ArrowLeft":
      i = Math.max(0, i - 1);
      break;
    case "ArrowRight":
      i = Math.min(order.length - 1, i + 1);
      break;
    case "ArrowUp":
      layer = Math.max(0, layer - 1);
      break;
    case "ArrowDown":
      layer = Math.min(nLayers - 1, layer + 1);
      break;
    case "Home":
      i = 0;
      break;
    case "End":
      i = order.length - 1;
      break;
    default:
      return null;
  }
  const part = order[i].part;
  return { ...start, layer, part, head: order[i].head, kind: part === "resid" ? start.kind : undefined };
}
