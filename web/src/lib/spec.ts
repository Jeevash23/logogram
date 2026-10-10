// Spec builders and plain-language descriptions (mirroring src/logogram/spec.py).

import type {
  BaselineSpec,
  ExperimentKind,
  PathReceiverSpec,
  ExperimentSpec,
  Measure,
  ModelInfo,
  PositionSpec,
  ScopeSpec,
  Side,
  SiteSpec,
  Spec,
} from "../api/types";
import { count } from "./format";
import { metricWords } from "./metrics";
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

/** A path's receiver as the results write it, for example "L9 H9 q" or "logits". */
export function receiverLabel(r: PathReceiverSpec): string {
  return r.kind === "logits" ? "logits" : `L${r.layer} H${r.head} ${r.input}`;
}

/** A steering strength as the results write it, for example ×2 or ×−0.5. */
export function strengthText(c: number): string {
  return `×${String(c).replace("-", "−")}`;
}

export function experimentText(e: ExperimentSpec): string {
  if (e.kind === "activation_patching") {
    return e.direction === "clean_to_corrupt" ? "Patch clean → corrupt" : "Patch corrupt → clean";
  }
  if (e.kind === "direct_logit_attribution") return `Direct logit attribution (${e.prompts} prompts)`;
  if (e.kind === "attribution_patching") {
    const arrow = e.direction === "clean_to_corrupt" ? "clean → corrupt" : "corrupt → clean";
    return e.method === "integrated_gradients"
      ? `Estimate patching ${arrow} with integrated gradients (${e.steps} steps)`
      : `Estimate patching ${arrow}`;
  }
  if (e.kind === "steering") {
    const toward = e.apply_to === "clean" ? "corrupt" : "clean";
    return `Steer ${e.apply_to} prompts toward ${toward} (${e.coefficients.map(strengthText).join(", ")}${e.control ? ", random control" : ""})`;
  }
  if (e.kind === "path_patching") {
    const arrow = e.direction === "clean_to_corrupt" ? "clean → corrupt" : "corrupt → clean";
    const names = e.receivers.slice(0, 4).map(receiverLabel).join(", ") + (e.receivers.length > 4 ? ` and ${e.receivers.length - 4} more` : "");
    return `Path patching ${arrow} into ${names} (${e.freeze_mlps ? "heads and MLPs" : "other heads"} held)`;
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
  if (e.kind === "direct_logit_attribution") return `Direct attribution, ${e.prompts}`;
  if (e.kind === "attribution_patching") {
    const arrow = e.direction === "clean_to_corrupt" ? "clean→corrupt" : "corrupt→clean";
    return e.method === "integrated_gradients" ? `Estimated ${arrow}, integrated gradients` : `Estimated ${arrow}`;
  }
  if (e.kind === "steering") return `Steer ${e.apply_to}→${e.apply_to === "clean" ? "corrupt" : "clean"}`;
  if (e.kind === "path_patching") return `Paths ${e.direction === "clean_to_corrupt" ? "clean→corrupt" : "corrupt→clean"}`;
  return { zero: "Zero ablation", mean: "Mean ablation", resample: "Resample ablation" }[e.baseline.kind];
}

/** What a method's per-prompt values are (see Summary.measure). */
export function measureOf(e: ExperimentSpec | null | undefined): Measure {
  if (e?.kind === "direct_logit_attribution") return "attribution";
  if (e?.kind === "attribution_patching") return "estimate";
  return "intervention";
}

/** The names of a run's two values, and what they mean, for its method and metric. Direct logit
 * attribution always splits the logit difference. */
export function measureWords(e: ExperimentSpec | null | undefined, metric?: { kind?: string | null; target?: Side | null } | null) {
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
  const m = metricWords(metric);
  const delta = `Δ ${m.short}`;
  if (e?.kind === "steering") {
    return {
      effect: "Normalized effect",
      delta,
      mean: "mean normalized effect",
      effectLegend: `Normalized effect: 1 means steering moved the ${e.apply_to} prompts as far as switching to the ${e.apply_to === "clean" ? "corrupt" : "clean"} prompt; 0 means no change.`,
      deltaLegend: `Change in the ${m.label} (${m.formula}) caused by adding the direction.`,
    };
  }
  if (e?.kind === "path_patching") {
    return {
      effect: "Normalized effect",
      delta,
      mean: "mean normalized effect",
      effectLegend: `Normalized effect through the receivers only: 1 means the path alone ${e.direction === "clean_to_corrupt" ? "restores the clean behavior" : "shifts the output as far as the corrupt prompt does"}; 0 means nothing travels along it.`,
      deltaLegend: `Change in the ${m.label} (${m.formula}) when only the receivers' inputs are patched.`,
    };
  }
  const restores = (e?.kind === "activation_patching" || e?.kind === "attribution_patching") && e.direction === "clean_to_corrupt";
  if (measureOf(e) === "estimate") {
    return {
      effect: "Estimated effect",
      delta: `Estimated Δ ${m.short}`,
      mean: "mean estimated effect",
      effectLegend: `Estimated normalized effect, to first order: 1 would mean the site alone ${restores ? "restores the clean behavior" : "shifts the output as far as the corrupt prompt does"}. Verify the strongest by patching.`,
      deltaLegend: `First-order estimate of the change in the ${m.label} (${m.formula}) patching would cause.`,
    };
  }
  return {
    effect: "Normalized effect",
    delta,
    mean: "mean normalized effect",
    effectLegend: `Normalized effect: 1 means the site alone ${restores ? "restores the clean behavior" : "shifts the output as far as the corrupt prompt does"}; 0 means no change.`,
    deltaLegend: `Change in the ${m.label} (${m.formula}) caused by the intervention.`,
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
    case "features":
      return `every SAE feature, ${positionText(s.position)}, keeping the top ${s.top}${
        s.choose_on !== null ? ` chosen on ${Math.round(s.choose_on * 100)}% of the prompts (seed ${s.seed}) and reported on the rest` : ""
      }`;
    case "site_sets":
      return `${s.sets.length} set${s.sets.length === 1 ? "" : "s"} of sites, each at once`;
  }
}

export function scopeShort(s: ScopeSpec): string {
  switch (s.kind) {
    case "heads":
      return "Layer × head";
    case "layer_position":
      return "Layer × position";
    case "layer_components":
      if (s.components.length === 1) return `${s.components[0].replace("_", " ")} per layer`;
      return s.components.length === 2 && s.components.includes("attn_out") && s.components.includes("mlp_out") ? "Attn and MLP" : "Components per layer";
    case "sites":
      return s.sites.length === 1 ? "Single site" : "Chosen sites";
    case "features":
      return "SAE features";
    case "site_sets":
      return "Sets of sites";
  }
}

/** Which run receives the intervention and where the replacement comes from. */
export function directionQuestion(e: ExperimentSpec): string {
  if (e.kind === "activation_patching") {
    return e.direction === "clean_to_corrupt" ? "Does this restore the behavior?" : "Does this break it?";
  }
  if (e.kind === "direct_logit_attribution") return "How much does this write directly toward the answer?";
  if (e.kind === "attribution_patching") {
    return e.direction === "clean_to_corrupt" ? "Would this restore the behavior, to first order?" : "Would this break it, to first order?";
  }
  if (e.kind === "steering") return "Does adding this direction move the behavior, more than a random one?";
  if (e.kind === "path_patching") return "How much of the effect travels through these receivers?";
  return "Does removing this break the behavior?";
}

export function receiverText(e: ExperimentSpec): { receiver: string; source: string } {
  if (e.kind === "activation_patching") {
    return e.direction === "clean_to_corrupt"
      ? { receiver: "corrupt prompt", source: "the clean prompt" }
      : { receiver: "clean prompt", source: "the corrupt prompt" };
  }
  if (e.kind === "direct_logit_attribution") return { receiver: `${e.prompts} prompt`, source: "its own forward pass" };
  if (e.kind === "attribution_patching") {
    return e.direction === "clean_to_corrupt"
      ? { receiver: "corrupt prompt", source: "the clean prompt" }
      : { receiver: "clean prompt", source: "the corrupt prompt" };
  }
  if (e.kind === "steering") return { receiver: `${e.apply_to} prompt`, source: "the mean difference of the training pairs" };
  if (e.kind === "path_patching") {
    return e.direction === "clean_to_corrupt"
      ? { receiver: "corrupt prompt", source: "the clean prompt, through the receivers" }
      : { receiver: "clean prompt", source: "the corrupt prompt, through the receivers" };
  }
  return { receiver: "clean prompt", source: baselineText(e.baseline) };
}

export function siteFromSelection(sel: Selection, residKind: SiteSpec["kind"] = sel.kind ?? "resid_pre"): SiteSpec {
  const kind: SiteSpec["kind"] =
    sel.part === "head" ? "head" : sel.part === "attn" ? "attn_out" : sel.part === "mlp" ? "mlp_out" : sel.part === "feature" ? "sae_feature" : residKind;
  const position: PositionSpec =
    sel.positionKey === undefined
      ? { kind: "all" }
      : sel.positionKey === "last"
        ? { kind: "last" }
        : /^-?\d+$/.test(sel.positionKey)
          ? { kind: "index", index: Number(sel.positionKey) }
          : { kind: "label", label: sel.positionKey };
  return {
    kind,
    layer: sel.layer,
    head: sel.part === "head" ? sel.head : null,
    ...(sel.part === "feature" ? { feature: sel.feature } : {}),
    position,
  };
}

export function defaultSpec(opts: {
  model: ModelInfo | null;
  modelId?: string;
  dataset: string;
  name?: string;
}): Spec {
  const m = opts.model;
  return {
    logogram_spec: 2,
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
    statistics: { bootstrap: 1000, ci: 0.95, seed: 0, cluster: null },
    execution: { batch_size: 64 },
  };
}

export function suggestName(spec: Pick<Spec, "experiment" | "scope">): string {
  const what = experimentShort(spec.experiment);
  const where = spec.scope.kind === "sites" && spec.scope.sites.length === 1
    ? siteText(spec.scope.sites[0])
    : spec.scope.kind === "features"
      ? "SAE features"
      : scopeShort(spec.scope).toLowerCase();
  return `${what} · ${where}`;
}

export function siteText(site: SiteSpec): string {
  const pos = site.position.kind === "all" ? "" : ` @ ${positionKey(site.position)}`;
  if (site.kind === "head") return `L${site.layer} H${site.head}${pos}`;
  if (site.kind === "sae_feature") return `L${site.layer} F${site.feature}${pos}`;
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

/** Rows the sweep will run: sites × prompts × donors. A decomposition runs each prompt once, an
 * estimate once per prompt (once per step of integrated gradients). */
export function workload(spec: Spec, n: number, nLayers: number, nHeads: number, nPositions: number | null, nLabels: number): number | null {
  if (spec.experiment.kind === "direct_logit_attribution") return n;
  if (spec.experiment.kind === "attribution_patching") {
    return spec.experiment.method === "integrated_gradients" ? n * (spec.experiment.steps ?? 1) : n;
  }
  const s = spec.scope;
  if (spec.experiment.kind === "steering") {
    const e = spec.experiment;
    const sites = s.kind === "layer_components" ? nLayers : s.kind === "sites" ? s.sites.length : 0;
    const test = n - Math.round(n * e.train_fraction);
    return sites * e.coefficients.length * (e.control ? 2 : 1) * Math.max(0, test);
  }
  let sites = 0;
  if (s.kind === "heads") sites = nLayers * nHeads;
  else if (s.kind === "layer_components") sites = nLayers * s.components.length;
  else if (s.kind === "sites") sites = s.sites.length;
  else if (s.kind === "site_sets") sites = s.sets.length;
  else if (s.kind === "features") return n; // only attribution patching sweeps every feature
  else if (s.positions === "each") {
    if (nPositions === null) return null;
    sites = nLayers * nPositions;
  } else sites = nLayers * nLabels;
  const donors = spec.experiment.kind === "ablation" && spec.experiment.baseline.kind === "resample" ? spec.experiment.baseline.donors : 1;
  return sites * n * donors;
}

/** What running the spec costs, in words. */
export function workloadText(spec: Spec, rows: number): string {
  const e = spec.experiment;
  if (e.kind === "direct_logit_attribution") return `That is one forward and one backward pass for each of the ${count(rows)} prompts.`;
  if (e.kind === "attribution_patching") {
    if (e.method === "integrated_gradients") {
      const steps = e.steps ?? 1;
      return `That is ${count(rows)} forward and backward passes, ${steps} for each of the ${count(Math.round(rows / steps))} prompts, for every site at once.`;
    }
    return `That is two forward passes and one backward pass for each of the ${count(rows)} prompts, for every site at once.`;
  }
  if (e.kind === "steering") return `That is ${count(rows)} steered forward passes on the held-out prompts.`;
  if (e.kind === "path_patching") return `That is ${count(rows)} paths, three forward passes each (fewer when senders come after the receivers).`;
  if (spec.scope.kind === "site_sets") return `That is ${count(rows)} patched forward passes, every site of a set at once.`;
  return `That is ${count(rows)} patched forward passes.`;
}

/** Fit a sweep to a method. Direct attribution reads what heads, attention and MLP outputs write,
 * at the last token where the logit difference is measured. */
export function scopeFor(kind: ExperimentKind, scope: ScopeSpec): ScopeSpec {
  const last: PositionSpec = { kind: "last" };
  // Only attribution patching sweeps every SAE feature; other methods patch chosen ones.
  if (scope.kind === "features" && kind !== "attribution_patching") return { kind: "heads", position: { kind: "all" } };
  // Sets are intervened on by patching or ablation; other methods measure their sites one by one.
  if (scope.kind === "site_sets" && kind !== "activation_patching" && kind !== "ablation") {
    const sites = setMembers(scope.sets);
    return scopeFor(kind, sites.length ? { kind: "sites", sites } : { kind: "heads", position: { kind: "all" } });
  }
  if (kind === "steering") {
    // Steering adds to one residual stream site per layer, at one token.
    const single = (p: PositionSpec) => (p.kind === "all" ? last : p);
    if (scope.kind === "layer_components") {
      const resid = scope.components.find((c) => c === "resid_pre" || c === "resid_mid" || c === "resid_post") ?? "resid_pre";
      return { kind: "layer_components", components: [resid], position: single(scope.position) };
    }
    if (scope.kind === "sites") {
      const sites = scope.sites
        .filter((x) => x.kind === "resid_pre" || x.kind === "resid_mid" || x.kind === "resid_post")
        .map((x) => ({ ...x, position: single(x.position) }));
      if (sites.length) return { kind: "sites", sites };
    }
    return { kind: "layer_components", components: ["resid_pre"], position: last };
  }
  if (kind === "path_patching") {
    // Paths start at heads, attention outputs or MLP outputs.
    if (scope.kind === "layer_position") return { kind: "heads", position: { kind: "all" } };
    if (scope.kind === "layer_components") {
      const components = scope.components.filter((c) => c === "attn_out" || c === "mlp_out");
      return { ...scope, components: components.length ? components : ["attn_out", "mlp_out"] };
    }
    if (scope.kind === "sites") {
      const sites = scope.sites.filter((x) => x.kind === "head" || x.kind === "attn_out" || x.kind === "mlp_out");
      return sites.length ? { kind: "sites", sites } : { kind: "heads", position: { kind: "all" } };
    }
    return scope;
  }
  if (kind !== "direct_logit_attribution") return scope;
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
    case "features":
      return { kind: "heads", position: last };
    case "site_sets":
      return { kind: "heads", position: last };
  }
}

/** Every site of some sets, each once, in the order they first appear. */
export function setMembers(sets: { sites: SiteSpec[] }[]): SiteSpec[] {
  const seen = new Set<string>();
  const out: SiteSpec[] = [];
  for (const set of sets) {
    for (const site of set.sites) {
      const key = siteText(site);
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(site);
    }
  }
  return out;
}
