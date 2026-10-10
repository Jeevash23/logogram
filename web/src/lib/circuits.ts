// Sets of sites (circuits): building the sets that test a circuit, and checking a scope of sets as
// the server does (SiteSet, SiteSetsScope and check_sets in src/logogram), so the form can say what
// is wrong before a run is started.

import type { PositionSpec, SiteKind, SiteResult, SiteSetSpec, SiteSpec, UniverseKind } from "../api/types";
import { plural } from "./format";
import { siteText } from "./spec";

/** A run holds at most this many sets. */
export const MAX_SETS = 64;
/** A set's label is at most this long. */
export const MAX_LABEL = 80;
/** A set holds at most this many sites. */
export const MAX_SET_SITES = 4096;

/** Sites a set can hold: what writes into the residual stream, and the stream before or after a
 * layer. (Between attention and the MLP it isn't a single hook.) */
export const SET_SITE_KINDS: readonly SiteKind[] = ["head", "attn_out", "mlp_out", "resid_pre", "resid_post"];
export const UNIVERSE_KINDS: readonly UniverseKind[] = ["head", "attn_out", "mlp_out"];

export const UNIVERSE_NAMES: Record<UniverseKind, { one: string; many: string }> = {
  head: { one: "head", many: "heads" },
  attn_out: { one: "attention output", many: "attention outputs" },
  mlp_out: { one: "MLP output", many: "MLP outputs" },
};

/** "every head and MLP output": what a set that keeps its sites replaces. */
export function universeText(universe: readonly UniverseKind[] | null | undefined): string {
  if (!universe || universe.length === 0) return "nothing else";
  const names = universe.map((k) => UNIVERSE_NAMES[k].one);
  return `every ${names.length === 1 ? names[0] : `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`}`;
}

export function isUniverseKind(kind: string): kind is UniverseKind {
  return (UNIVERSE_KINDS as readonly string[]).includes(kind);
}

/** In words, what a set does to the model: "keeps 3 sites, replaces every other head". */
export function setWhat(complement: boolean, size: number, universe: readonly UniverseKind[] | null | undefined): string {
  if (!complement) return `replaces ${plural(size, "site")}`;
  if (!size) return `replaces ${universeText(universe)}`;
  return `keeps ${plural(size, "site")} and replaces ${universeText(universe).replace(/^every /, "every other ")}`;
}

/** The components a circuit of these sites is made of, as the rest of the model it is kept apart
 * from; an error when they can't make one universe. */
export function universeFor(sites: SiteSpec[]): { universe: UniverseKind[] } | { error: string } {
  const kinds = new Set(sites.map((x) => x.kind));
  const outside = [...kinds].filter((k) => !isUniverseKind(k));
  if (outside.length) {
    return {
      error: `A circuit is kept apart from the rest of the model's heads, attention outputs or MLP outputs, so it can't hold ${outside
        .map((k) => KIND_PLURAL[k])
        .join(" or ")}. Remove ${outside.length === 1 && sites.filter((x) => x.kind === outside[0]).length === 1 ? "that site" : "those sites"} first.`,
    };
  }
  if (kinds.has("head") && kinds.has("attn_out")) {
    return { error: "A circuit can hold heads or attention outputs, not both: a layer's attention output is the sum of its heads." };
  }
  if (kinds.size === 0) return { error: "Choose at least one site for the circuit." };
  return { universe: UNIVERSE_KINDS.filter((k) => kinds.has(k)) };
}

const KIND_PLURAL: Record<SiteKind, string> = {
  resid_pre: "residual stream sites",
  resid_mid: "residual stream sites",
  resid_post: "residual stream sites",
  attn_out: "attention outputs",
  mlp_out: "MLP outputs",
  head: "heads",
  sae_feature: "SAE features",
};

export const CIRCUIT_KEPT = "Circuit kept";
export const CIRCUIT_REMOVED = "Circuit removed";
export const EVERYTHING = "Everything replaced";

/** A site's label within a set's label, kept short enough for the label's limit. */
function short(text: string, room: number): string {
  return text.length <= room ? text : `${text.slice(0, Math.max(1, room - 1))}…`;
}

/**
 * The sets that test a circuit: the circuit kept alone (everything else replaced), the circuit
 * removed, and everything replaced, which the other two are read against. With `minimality`, one
 * more set per site keeps the circuit without that site: how much faithfulness each site adds.
 */
export function circuitSets(sites: SiteSpec[], options: { minimality: boolean }): SiteSetSpec[] {
  const sets: SiteSetSpec[] = [
    { label: CIRCUIT_KEPT, sites, complement: true },
    { label: CIRCUIT_REMOVED, sites, complement: false },
    { label: EVERYTHING, sites: [], complement: true },
  ];
  if (options.minimality && sites.length > 1) {
    for (const site of sites) {
      sets.push({
        label: `Without ${short(siteText(site), MAX_LABEL - 8)}`,
        sites: sites.filter((x) => x !== site),
        complement: true,
      });
    }
  }
  return sets;
}

/** How many sets testing a circuit of n sites takes. */
export function circuitSetCount(n: number, minimality: boolean): number {
  return 3 + (minimality && n > 1 ? n : 0);
}

/** The sizes of nested circuits from a ranking: 1, 2, 4, 8, … and k itself. */
export function nestedSizes(k: number): number[] {
  const sizes: number[] = [];
  for (let size = 1; size < k; size *= 2) sizes.push(size);
  if (k >= 1) sizes.push(k);
  return sizes;
}

/**
 * Nested circuits from a ranking, strongest first: the top 1, 2, 4, 8, … and k sites kept alone,
 * for faithfulness against size; everything replaced, which they are read against; and the top k
 * removed.
 */
export function topKSets(ranked: SiteSpec[], k: number): SiteSetSpec[] {
  const top = ranked.slice(0, Math.max(0, Math.min(k, ranked.length)));
  const sets: SiteSetSpec[] = nestedSizes(top.length).map((size) => ({
    label: `Top ${size} kept`,
    sites: top.slice(0, size),
    complement: true,
  }));
  sets.push({ label: EVERYTHING, sites: [], complement: true });
  sets.push({ label: `Top ${top.length} removed`, sites: top, complement: false });
  return sets;
}

/** A run's single heads, attention outputs and MLP outputs that have an effect, strongest first
 * (ties by index), as sites of a spec: what a circuit can be built from. */
export function rankedSites(sites: SiteResult[]): SiteSpec[] {
  return sites
    .filter((x) => isUniverseKind(x.kind) && !x.variant && x.effect.mean !== null)
    .sort((a, b) => Math.abs(b.effect.mean ?? 0) - Math.abs(a.effect.mean ?? 0) || a.index - b.index)
    .map((x) => ({ kind: x.kind as SiteKind, layer: x.layer, head: x.kind === "head" ? x.head : null, position: x.position }));
}

/** Two sites alone and together: the run reports what intervening on both does beyond the sum. */
export function interactionSets(a: SiteSpec, b: SiteSpec): SiteSetSpec[] {
  const la = short(siteText(a), 38);
  const lb = short(siteText(b), 38);
  return [
    { label: la, sites: [a], complement: false },
    { label: lb === la ? `${lb} (2)` : lb, sites: [b], complement: false },
    { label: `${la} and ${lb}`, sites: [a, b], complement: false },
  ];
}

/** A label not yet used by these sets, from a base such as "Set 3". */
export function freeLabel(sets: SiteSetSpec[], base: string): string {
  const taken = new Set(sets.map((x) => x.label));
  if (!taken.has(base)) return base;
  for (let i = 2; ; i++) {
    const label = `${base} (${i})`;
    if (!taken.has(label)) return label;
  }
}

/** Parse a site typed as on the map: "L9 H6", "L3 mlp", "L3 attn", "L2 resid pre", "L2 resid post". */
export function parseSiteQuery(query: string): { kind: SiteKind; layer: number; head: number | null } | null {
  const q = query.trim().toLowerCase().replace(/[_-]/g, " ").replace(/\s+/g, " ");
  const head = q.match(/^l?\s*(\d+)\s*[.,\s]?\s*h\s*(\d+)$/) ?? q.match(/^(\d+)\s*[.,]\s*(\d+)$/);
  if (head) return { kind: "head", layer: Number(head[1]), head: Number(head[2]) };
  const part = q.match(/^l?\s*(\d+)\s+(attn|attention|attn out|attention output|mlp|mlp out|mlp output|resid pre|resid post|residual pre|residual post)$/);
  if (!part) return null;
  const word = part[2];
  const kind: SiteKind = word.startsWith("attn") || word.startsWith("attention")
    ? "attn_out"
    : word.startsWith("mlp")
      ? "mlp_out"
      : word.endsWith("pre")
        ? "resid_pre"
        : "resid_post";
  return { kind, layer: Number(part[1]), head: null };
}

/** A site as a spec writes it. */
export function siteOf(parsed: { kind: SiteKind; layer: number; head: number | null }, position: PositionSpec): SiteSpec {
  return { kind: parsed.kind, layer: parsed.layer, head: parsed.kind === "head" ? parsed.head : null, position };
}

/** The first thing wrong with a scope of sets, as the server would refuse it; null when it is
 * valid. With the loaded model's shape, sites it doesn't have are refused too. */
export function siteSetsError(
  universe: UniverseKind[] | null,
  sets: SiteSetSpec[],
  model?: { n_layers: number; n_heads: number; site_kinds: string[] } | null,
): string | null {
  if (sets.length === 0) return "Add at least one set of sites.";
  if (sets.length > MAX_SETS) return `A run holds at most ${MAX_SETS} sets; this has ${sets.length}. Remove ${plural(sets.length - MAX_SETS, "set")}.`;
  const labels = new Set<string>();
  for (const set of sets) {
    const label = set.label.trim();
    if (!label) return "Give every set a label.";
    if (label.length > MAX_LABEL) return `The label “${short(label, 40)}” is longer than ${MAX_LABEL} characters. Shorten it.`;
    if (labels.has(label)) return `Two sets are labelled “${label}”. Give each set its own label.`;
    labels.add(label);
  }
  if (universe !== null) {
    if (universe.length === 0) return "Choose what the rest of the model is made of: heads, attention outputs or MLP outputs.";
    if (universe.includes("head") && universe.includes("attn_out")) {
      return "The rest of the model can't be both heads and attention outputs: a layer's attention output is the sum of its heads. Choose one.";
    }
  }
  for (const set of sets) {
    const name = `“${set.label.trim()}”`;
    if (!set.complement && set.sites.length === 0) return `The set ${name} replaces nothing: add a site, or make it keep its sites and replace the rest.`;
    if (set.sites.length > MAX_SET_SITES) return `The set ${name} holds more than ${MAX_SET_SITES} sites.`;
    for (const site of set.sites) {
      if (!SET_SITE_KINDS.includes(site.kind)) {
        return `The set ${name} holds ${siteText(site)}, a ${site.kind === "sae_feature" ? "feature" : "site"} a set can't hold. Sets hold heads, attention or MLP outputs, and the residual stream before or after a layer.`;
      }
      if (set.complement && !(universe ?? []).includes(site.kind as UniverseKind)) {
        return universe
          ? `The set ${name} keeps ${siteText(site)}, which isn't one of the ${universe.map((k) => UNIVERSE_NAMES[k].many).join(" or ")} it replaces the rest of. Add ${UNIVERSE_NAMES[site.kind as UniverseKind]?.many ?? "that kind of site"} to the rest of the model, or remove the site.`
          : `The set ${name} keeps its sites and replaces the rest: choose what the rest of the model is made of.`;
      }
      if (model) {
        if (site.layer >= model.n_layers) return `The set ${name} has a site in layer ${site.layer}; this model has ${plural(model.n_layers, "layer")}.`;
        if (site.kind === "head" && (site.head ?? 0) >= model.n_heads) return `The set ${name} has head ${site.head}; this model has ${plural(model.n_heads, "head")} per layer.`;
        if (model.site_kinds.length && !model.site_kinds.includes(site.kind)) return `This model has no ${site.kind} site, which the set ${name} uses.`;
      }
    }
    const attention = new Set(set.sites.filter((x) => x.kind === "attn_out").map((x) => x.layer));
    const both = set.sites.find((x) => x.kind === "head" && attention.has(x.layer));
    if (both) {
      return `The set ${name} has both heads and the attention output of layer ${both.layer}; the attention output is the sum of the heads. Use one or the other.`;
    }
  }
  if (sets.some((x) => x.complement) && universe === null) {
    return "A set that keeps its sites replaces the rest of the model: choose what the rest is made of.";
  }
  if (model && universe) {
    const missing = universe.find((k) => model.site_kinds.length && !model.site_kinds.includes(k));
    if (missing) return `This model has no ${missing} site, so the rest of the model can't include ${UNIVERSE_NAMES[missing].many}.`;
  }
  return null;
}
