// Spec builders and plain-language descriptions (mirroring src/logogram/spec.py).

import type {
  BaselineSpec,
  ExperimentKind,
  ExperimentSpec,
  Measure,
  ModelInfo,
  PositionSpec,
  ScopeSpec,
  SiteSpec,
  Spec,
} from "../api/types";
import { count } from "./format";
import type { Selection } from "./sites";

export function positionText(p: PositionSpec): string {
  switch (p.kind) {
    case "all":
      return "all positions";
    case "last":
      return "the last token";
    case "index":
      return `token index ${p.index}`;
    case "label":
      return `position ${p.label}`;
  }
}

export function baselineText(b: BaselineSpec): string {
  switch (b.kind) {
    case "zero":
      return "zero";
    case "mean":
      return `mean over ${b.reference} prompts`;
    case "resample":
      return `resample from ${b.donors} ${b.pool} donor${b.donors === 1 ? "" : "s"}, seed ${b.seed}`;
  }
}

export function experimentText(e: ExperimentSpec): string {
  if (e.kind === "activation_patching") {
    return e.direction === "clean_to_corrupt" ? "Patch clean → corrupt" : "Patch corrupt → clean";
  }
  if (e.kind === "direct_logit_attribution") return `Direct logit attribution (${e.prompts} prompts)`;
  switch (e.baseline.kind) {
    case "zero":
      return "Zero-ablate";
    case "mean":
      return `Mean-ablate (${e.baseline.reference} mean)`;
    case "resample":
      return `Resample-ablate (${e.baseline.donors} ${e.baseline.pool} donors, seed ${e.baseline.seed})`;
  }
}

export function experimentShort(e: ExperimentSpec): string {
  if (e.kind === "activation_patching") {
    return e.direction === "clean_to_corrupt" ? "Patch clean→corrupt" : "Patch corrupt→clean";
  }
  if (e.kind === "direct_logit_attribution") return `Direct attribution, ${e.prompts}`;
  return { zero: "Zero ablation", mean: "Mean ablation", resample: "Resample ablation" }[e.baseline.kind];
}

/** What a method's per-prompt values are (see Summary.measure). */
export function measureOf(e: ExperimentSpec | null | undefined): Measure {
  return e?.kind === "direct_logit_attribution" ? "attribution" : "intervention";
}

/** The names of a run's two values, and what they mean, for its method. */
export function measureWords(e: ExperimentSpec | null | undefined) {
  if (measureOf(e) === "attribution") {
    return {
      effect: "Share of logit diff",
      delta: "Direct effect",
      mean: "mean share of the logit difference",
      effectLegend:
        "Share of the mean logit difference this component writes directly. Negative values push toward the distractor.",
      deltaLegend: "What the component writes directly into the logit difference (answer − distractor), in logits.",
    };
  }
  const restores = e?.kind === "activation_patching" && e.direction === "clean_to_corrupt";
  return {
    effect: "Normalized effect",
    delta: "Δ logit diff",
    mean: "mean normalized effect",
    effectLegend: `Normalized effect: 1 means the site alone ${restores ? "restores the clean behavior" : "shifts the output as far as the corrupt prompt does"}; 0 means no change.`,
    deltaLegend: "Change in logit difference (answer − distractor) caused by the intervention.",
  };
}

export function scopeText(s: ScopeSpec): string {
  switch (s.kind) {
    case "heads":
      return `every head, ${positionText(s.position)}`;
    case "layer_position":
      return `${s.site} at every layer and ${s.positions === "each" ? "every position" : "each labelled position"}`;
    case "layer_components":
      return `${s.components.join(" and ")} per layer, ${positionText(s.position)}`;
    case "sites":
      return `${s.sites.length} chosen site${s.sites.length === 1 ? "" : "s"}`;
  }
}

export function scopeShort(s: ScopeSpec): string {
  switch (s.kind) {
    case "heads":
      return "Layer × head";
    case "layer_position":
      return "Layer × position";
    case "layer_components":
      return "Attn and MLP";
    case "sites":
      return s.sites.length === 1 ? "Single site" : "Chosen sites";
  }
}

/** Which run receives the intervention and where the replacement comes from. */
export function directionQuestion(e: ExperimentSpec): string {
  if (e.kind === "activation_patching") {
    return e.direction === "clean_to_corrupt" ? "Does this restore the behavior?" : "Does this break it?";
  }
  if (e.kind === "direct_logit_attribution") return "How much does this write directly toward the answer?";
  return "Does removing this break the behavior?";
}

export function receiverText(e: ExperimentSpec): { receiver: string; source: string } {
  if (e.kind === "activation_patching") {
    return e.direction === "clean_to_corrupt"
      ? { receiver: "corrupt prompt", source: "the clean prompt" }
      : { receiver: "clean prompt", source: "the corrupt prompt" };
  }
  if (e.kind === "direct_logit_attribution") return { receiver: `${e.prompts} prompt`, source: "its own forward pass" };
  return { receiver: "clean prompt", source: baselineText(e.baseline) };
}

export function siteFromSelection(sel: Selection, residKind: SiteSpec["kind"] = sel.kind ?? "resid_pre"): SiteSpec {
  const kind: SiteSpec["kind"] =
    sel.part === "head" ? "head" : sel.part === "attn" ? "attn_out" : sel.part === "mlp" ? "mlp_out" : residKind;
  const position: PositionSpec =
    sel.positionKey === undefined
      ? { kind: "all" }
      : sel.positionKey === "last"
        ? { kind: "last" }
        : /^-?\d+$/.test(sel.positionKey)
          ? { kind: "index", index: Number(sel.positionKey) }
          : { kind: "label", label: sel.positionKey };
  return { kind, layer: sel.layer, head: sel.part === "head" ? sel.head : null, position };
}

export function defaultSpec(opts: {
  model: ModelInfo | null;
  modelId?: string;
  dataset: string;
  name?: string;
}): Spec {
  const m = opts.model;
  return {
    logogram_spec: 1,
    name: opts.name ?? "Which heads restore the answer?",
    notes: "",
    model: {
      id: m?.id ?? opts.modelId ?? "openai-community/gpt2",
      revision: m?.revision ?? null,
      dtype: m?.dtype ?? "float32",
      device: "auto",
      process_weights: m?.process_weights ?? true,
    },
    dataset: { path: opts.dataset, sha256: null, limit: null },
    tokenization: { prepend_bos: true },
    experiment: { kind: "activation_patching", direction: "clean_to_corrupt" },
    scope: { kind: "heads", position: { kind: "all" } },
    metric: { kind: "logit_diff", normalization: "dataset_gap" },
    statistics: { bootstrap: 1000, ci: 0.95, seed: 0 },
    execution: { batch_size: 64 },
  };
}

export function suggestName(spec: Pick<Spec, "experiment" | "scope">): string {
  const what = experimentShort(spec.experiment);
  const where = spec.scope.kind === "sites" && spec.scope.sites.length === 1
    ? siteText(spec.scope.sites[0])
    : scopeShort(spec.scope).toLowerCase();
  return `${what} · ${where}`;
}

export function siteText(site: SiteSpec): string {
  const pos = site.position.kind === "all" ? "" : ` @ ${positionKey(site.position)}`;
  if (site.kind === "head") return `L${site.layer} H${site.head}${pos}`;
  const name = { resid_pre: "resid pre", resid_mid: "resid mid", resid_post: "resid post", attn_out: "attn", mlp_out: "mlp" }[site.kind];
  return `L${site.layer} ${name}${pos}`;
}

export function positionKey(p: PositionSpec): string {
  switch (p.kind) {
    case "all":
      return "all";
    case "last":
      return "last";
    case "index":
      return String(p.index);
    case "label":
      return p.label;
  }
}

/** Rows the sweep will run: sites × prompts × donors. A decomposition runs each prompt once. */
export function workload(spec: Spec, n: number, nLayers: number, nHeads: number, nPositions: number | null, nLabels: number): number | null {
  if (spec.experiment.kind === "direct_logit_attribution") return n;
  const s = spec.scope;
  let sites = 0;
  if (s.kind === "heads") sites = nLayers * nHeads;
  else if (s.kind === "layer_components") sites = nLayers * s.components.length;
  else if (s.kind === "sites") sites = s.sites.length;
  else if (s.positions === "each") {
    if (nPositions === null) return null;
    sites = nLayers * nPositions;
  } else sites = nLayers * nLabels;
  const donors = spec.experiment.kind === "ablation" && spec.experiment.baseline.kind === "resample" ? spec.experiment.baseline.donors : 1;
  return sites * n * donors;
}

/** What running the spec costs, in words. */
export function workloadText(spec: Spec, rows: number): string {
  if (spec.experiment.kind === "direct_logit_attribution") return `That is one forward and one backward pass for each of the ${count(rows)} prompts.`;
  return `That is ${count(rows)} patched forward passes.`;
}

/** Fit a sweep to a method. Direct attribution reads what heads, attention and MLP outputs write,
 * at the last token where the logit difference is measured. */
export function scopeFor(kind: ExperimentKind, scope: ScopeSpec): ScopeSpec {
  if (kind !== "direct_logit_attribution") return scope;
  const last: PositionSpec = { kind: "last" };
  switch (scope.kind) {
    case "heads":
      return { kind: "heads", position: last };
    case "layer_components": {
      const components = scope.components.filter((c) => c === "attn_out" || c === "mlp_out");
      return { kind: "layer_components", components: components.length ? components : ["attn_out", "mlp_out"], position: last };
    }
    case "layer_position":
      return { kind: "heads", position: last };
    case "sites": {
      const sites = scope.sites
        .filter((x) => x.kind === "head" || x.kind === "attn_out" || x.kind === "mlp_out")
        .map((x) => ({ ...x, position: last }));
      return sites.length ? { kind: "sites", sites } : { kind: "heads", position: last };
    }
  }
}
