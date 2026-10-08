// Spec builders and plain-language descriptions (mirroring src/logogram/spec.py).

import type {
  BaselineSpec,
  ExperimentSpec,
  ModelInfo,
  PositionSpec,
  ScopeSpec,
  SiteSpec,
  Spec,
} from "../api/types";
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
  return { zero: "Zero ablation", mean: "Mean ablation", resample: "Resample ablation" }[e.baseline.kind];
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
  return "Does removing this break the behavior?";
}

export function receiverText(e: ExperimentSpec): { receiver: string; source: string } {
  if (e.kind === "activation_patching") {
    return e.direction === "clean_to_corrupt"
      ? { receiver: "corrupt prompt", source: "the clean prompt" }
      : { receiver: "clean prompt", source: "the corrupt prompt" };
  }
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

/** Rows the sweep will run: sites × prompts × donors. */
export function workload(spec: Spec, n: number, nLayers: number, nHeads: number, nPositions: number | null, nLabels: number): number | null {
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
