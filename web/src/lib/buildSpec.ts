// The experiment form as a version 2 spec, or what is missing. Every choice that can change a
// number is written out; nothing the form doesn't show is assumed. Restrictions a method places on
// the metric and the answers (src/logogram/direct.py, paths.py, engine.py, features.py and
// circuits.py) are checked here too, so a run doesn't fail on something the form could have said.

import type { DatasetDetail, MetricSpec, ModelStatus, PositionSpec, SAEStatus, ScopeSpec, Spec } from "../api/types";
import { siteSetsError } from "./circuits";
import { plural, visibleToken } from "./format";
import type { FormState } from "./formState";
import { splitCount, suggestName } from "./spec";

/** What the form knows about the prompts it will run on. */
export interface DatasetFacts {
  /** Prompts used: the first `limit`, or every prompt. */
  n: number;
  /** Prompts whose answer or distractor is a set of single tokens. */
  sets: number;
  /** Prompts whose answer or distractor is a continuation of several tokens for the loaded model,
   * which only metrics that read continuations can score; null when no model is loaded. */
  continuations: number | null;
  /** One such answer, to show. */
  example: string | null;
  /** Fields of the meta every used prompt has (and that can name a cluster), with how many
   * different values each takes. */
  fields: { field: string; clusters: number }[];
}

// statistics.cluster in spec.py: a field name, as a key of the prompts' meta.
const CLUSTER_KEY = /^[A-Za-z_][A-Za-z0-9_.-]{0,63}$/;

/** Facts about the first `limit` prompts of a dataset (all of them without a limit). */
export function datasetFacts(dataset: DatasetDetail | null | undefined, limit: number | null): DatasetFacts | null {
  if (!dataset) return null;
  const records = dataset.records.slice(0, limit ?? undefined);
  const n = records.length;
  const sets = records.filter((r) => Array.isArray(r.answer) || Array.isArray(r.distractor)).length;
  const continued = dataset.continuations?.filter((i) => i < n) ?? null;
  const first = continued?.length ? records[continued[0]] : undefined;
  const example = first ? [first.answer, first.distractor].find((a): a is string => typeof a === "string" && a.length > 0) ?? null : null;
  const fields: DatasetFacts["fields"] = [];
  if (n > 0) {
    const keys = Object.keys(records[0].meta ?? {}).filter((k) => CLUSTER_KEY.test(k));
    for (const key of keys) {
      if (!records.every((r) => r.meta && key in r.meta)) continue;
      const values = new Set(records.map((r) => clusterValue(r.meta?.[key])));
      fields.push({ field: key, clusters: values.size });
    }
  }
  return { n, sets, continuations: continued?.length ?? null, example, fields };
}

/** A meta value as the bootstrap groups it (the server compares their text). */
function clusterValue(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value);
}

/** Strengths as typed: numbers separated by commas or spaces (a typographic minus is fine). */
export function parseStrengths(text: string): { values: number[] } | { error: string } {
  const parts = text.replace(/−/g, "-").split(/[\s,;]+/).filter(Boolean);
  if (parts.length === 0) return { error: "Give at least one steering strength, such as 1." };
  const values = parts.map(Number);
  const bad = parts.find((_, i) => !Number.isFinite(values[i]));
  if (bad !== undefined) return { error: `“${bad}” isn't a number. Separate strengths with commas, like -1, 1, 2.` };
  if (new Set(values).size !== values.length) return { error: "Each steering strength appears once." };
  if (values.length > 16) return { error: "Use at most 16 steering strengths." };
  return { values };
}

/** The form's metric as a spec writes it, or null while the KL divergence has no target. */
export function metricOf(form: Pick<FormState, "metric" | "klTarget" | "normalization">): MetricSpec | null {
  if (form.metric === "kl") return form.klTarget ? { kind: "kl", target: form.klTarget, normalization: form.normalization } : null;
  return { kind: form.metric, normalization: form.normalization };
}

const SEED_MAX = 2 ** 32;
const isSeed = (v: number) => Number.isInteger(v) && v >= 0 && v < SEED_MAX;

/** Whether a scope measures SAE features (which then belong to the spec's SAE). */
export function usesFeatures(scope: ScopeSpec): boolean {
  return scope.kind === "features" || (scope.kind === "sites" && scope.sites.some((x) => x.kind === "sae_feature"));
}

/** Whether the scope intervenes on every position of a component somewhere. */
function everyPosition(scope: ScopeSpec): boolean {
  if (scope.kind === "heads" || scope.kind === "layer_components" || scope.kind === "features") return scope.position.kind === "all";
  if (scope.kind === "sites") return scope.sites.some((x) => x.position.kind === "all");
  if (scope.kind === "site_sets") return true;
  return false;
}

/** Build a spec from the form, or explain what's missing. */
export function buildSpec(
  form: FormState,
  ctx: {
    model: ModelStatus;
    datasetPath: string | null;
    datasetSha: string | null;
    sae?: SAEStatus;
    /** What is known about the prompts; without it, checks that need them are left to the run. */
    facts?: DatasetFacts | null;
  },
): { spec: Spec } | { error: string } {
  if (!ctx.datasetPath) return { error: "Choose prompts first (Prompts view)." };
  let experiment: Spec["experiment"];
  if (form.kind === "activation_patching") {
    experiment = { kind: "activation_patching", direction: form.direction };
  } else if (form.kind === "attribution_patching") {
    if (form.atpMethod === "integrated_gradients") {
      if (!Number.isInteger(form.atpSteps) || form.atpSteps < 2 || form.atpSteps > 64) {
        return { error: "Integrated gradients take a whole number of steps from 2 to 64." };
      }
      experiment = { kind: "attribution_patching", direction: form.direction, method: "integrated_gradients", steps: form.atpSteps };
    } else {
      experiment = { kind: "attribution_patching", direction: form.direction, method: "gradient", steps: null };
    }
  } else if (form.kind === "path_patching") {
    if (form.pathReceivers.length === 0) return { error: "Add at least one receiver: a later head's query, key or value, or the logits." };
    experiment = { kind: "path_patching", direction: form.direction, receivers: form.pathReceivers, freeze_mlps: form.pathFreezeMlps };
  } else if (form.kind === "steering") {
    if (!form.steerApplyTo) return { error: "Choose which prompts to steer." };
    const strengths = parseStrengths(form.steerStrengths);
    if ("error" in strengths) return strengths;
    experiment = {
      kind: "steering",
      apply_to: form.steerApplyTo,
      coefficients: strengths.values,
      train_fraction: form.steerTrain,
      seed: form.steerSeed,
      control: form.steerControl,
    };
  } else if (form.kind === "direct_logit_attribution") {
    if (!form.dlaPrompts) return { error: "Choose which prompts' logit difference to split." };
    experiment = { kind: "direct_logit_attribution", prompts: form.dlaPrompts };
  } else {
    if (!form.baseline) return { error: "Choose a baseline for the ablation. Logogram never assumes one." };
    experiment = { kind: "ablation", baseline: form.baseline };
  }
  const scope = form.scope;
  const positionError = (p: PositionSpec) =>
    p.kind === "label" && !p.label ? "Choose which named position to use." : null;
  if (scope.kind === "heads" || scope.kind === "layer_components" || scope.kind === "features") {
    const err = positionError(scope.position);
    if (err) return { error: err };
  }
  if (scope.kind === "layer_components" && scope.components.length === 0) {
    return { error: "Choose at least one component." };
  }
  if (scope.kind === "sites" && scope.sites.some((x) => positionError(x.position))) {
    return { error: "Choose which named position to use." };
  }
  if (scope.kind === "features") {
    if (!Number.isInteger(scope.top) || scope.top < 1 || scope.top > 500) return { error: "Keep from 1 to 500 of the strongest features." };
    if (scope.choose_on !== null) {
      if (!(scope.choose_on > 0 && scope.choose_on < 1)) return { error: "Choose the features on a share of the prompts between 0 and 100%." };
      if (scope.seed === null || !isSeed(scope.seed)) return { error: "Give the split of the prompts a seed: a whole number from 0." };
    }
  }
  if (scope.kind === "site_sets") {
    if (experiment.kind !== "activation_patching" && experiment.kind !== "ablation") {
      return { error: "Sets of sites are intervened on by activation patching or ablation. Choose one of those." };
    }
    if (scope.sets.some((set) => set.sites.some((x) => positionError(x.position)))) return { error: "Choose which named position to use." };
    const err = siteSetsError(scope.universe, scope.sets, ctx.model.info ?? null);
    if (err) return { error: err };
  }
  // SAE features (src/logogram/features.py): patched, zero-ablated or estimated from one gradient,
  // in a run of their own.
  if (scope.kind === "sites" && scope.sites.some((x) => x.kind === "sae_feature")) {
    if (scope.sites.some((x) => x.kind !== "sae_feature")) {
      return { error: "A run measures either SAE features or model components. Put the features in a run of their own." };
    }
    const allowed = experiment.kind === "activation_patching" || experiment.kind === "attribution_patching" ||
      (experiment.kind === "ablation" && experiment.baseline.kind === "zero");
    if (!allowed) return { error: "SAE features can be patched, zero-ablated, or estimated by attribution patching. Choose one of those." };
  }
  if (usesFeatures(scope) && experiment.kind === "attribution_patching" && experiment.method !== "gradient") {
    return { error: "SAE features are estimated from a single gradient. Choose one gradient, or estimate the model's components with integrated gradients." };
  }

  // The metric, and what the method and the answers allow.
  const metric = metricOf(form);
  if (!metric) return { error: "Choose which prompt's prediction the KL divergence is measured from." };
  if (experiment.kind === "direct_logit_attribution" && metric.kind !== "logit_diff") {
    return { error: "Direct logit attribution splits the logit difference, the one metric that is a sum of what each component writes. Choose the logit difference as the metric." };
  }
  const facts = ctx.facts;
  if (facts) {
    const several = facts.continuations ?? 0;
    const example = facts.example ? ` (such as “${visibleToken(facts.example)}”)` : "";
    const have = (k: number) => `${plural(k, "prompt")} ${k === 1 ? "has" : "have"}`;
    if (experiment.kind === "direct_logit_attribution" && (facts.sets > 0 || several > 0)) {
      return {
        error: `Direct logit attribution splits the logit difference between two single tokens, and ${have(facts.sets + several)} an answer set or an answer of several tokens. Use single-token answers, or activation patching.`,
      };
    }
    if (several > 0 && metric.kind === "logit_diff") {
      return {
        error: `${have(several)} an answer or distractor of several tokens${example}. The logit difference reads one token: choose the log-probability difference, or another metric that reads answers of several tokens.`,
      };
    }
    if (several > 0 && experiment.kind === "path_patching") {
      return { error: `Path patching reads the metric at the last prompt position, and ${have(several)} answers of several tokens${example}. Use single-token answers or sets, or activation patching.` };
    }
    if (several > 0 && experiment.kind === "ablation" && experiment.baseline.kind === "mean" && everyPosition(scope)) {
      return {
        error: `Mean ablation replaces each position with its mean over prompts of the same length, and the tokens appended to read answers of several tokens${example} have no such mean. Ablate at one position, use zero or resample ablation, or use single-token answers.`,
      };
    }
    const featuresEverywhere = (scope.kind === "features" && scope.position.kind === "all") ||
      (scope.kind === "sites" && scope.sites.some((x) => x.kind === "sae_feature" && x.position.kind === "all"));
    if (several > 0 && featuresEverywhere) {
      return { error: "With answers of several tokens, SAE features are patched or estimated at one position of the prompt (the last token, or a named position), not at every position." };
    }
    // Splits of the prompts need enough on each side (features.py, steering.py).
    if (scope.kind === "features" && scope.choose_on !== null && facts.n > 0) {
      const choose = splitCount(facts.n, scope.choose_on);
      if (choose < 1 || facts.n - choose < 2) {
        return { error: `Choosing features on ${Math.round(scope.choose_on * 100)}% of ${plural(facts.n, "prompt")} leaves ${choose} to choose them and ${facts.n - choose} to report them, but it needs at least one and two. Use more prompts, or another share.` };
      }
    }
    if (experiment.kind === "steering" && facts.n > 0) {
      const train = splitCount(facts.n, experiment.train_fraction);
      if (train < 1 || facts.n - train < 2) {
        return { error: `Steering needs at least one pair to compute the direction and two to measure it on, but ${plural(facts.n, "pair")} with a training share of ${Math.round(experiment.train_fraction * 100)}% leaves ${train} and ${facts.n - train}. Use more prompts or another training share.` };
      }
    }
    if (experiment.kind === "ablation" && experiment.baseline.kind === "resample" && facts.n > 0 && experiment.baseline.donors > facts.n - 1) {
      return { error: `Each prompt draws ${plural(experiment.baseline.donors, "donor")} from the other prompts, but there ${facts.n - 1 === 1 ? "is" : "are"} only ${facts.n - 1}. Lower the donor count, or use more prompts.` };
    }
  }

  // Statistics.
  if (!Number.isInteger(form.bootstrap) || form.bootstrap < 100 || form.bootstrap > 100_000) {
    return { error: "Use from 100 to 100,000 bootstrap resamples." };
  }
  if (!isSeed(form.statSeed)) return { error: "The bootstrap seed is a whole number from 0." };
  if (form.cluster !== null) {
    if (!CLUSTER_KEY.test(form.cluster)) return { error: `“${form.cluster}” can't name a field of the prompts' meta. Choose another, or resample single prompts.` };
    const field = facts?.fields.find((f) => f.field === form.cluster);
    if (facts && !field) {
      return { error: `Not every prompt has “${form.cluster}” in its meta, so they can't be grouped by it. Choose another field, or resample single prompts.` };
    }
    if (field && field.clusters < 2) {
      return { error: `Every prompt has the same ${form.cluster}, so there is one cluster and nothing to resample. Choose another field, or resample single prompts.` };
    }
  }
  if (!Number.isInteger(form.batchSize) || form.batchSize < 1 || form.batchSize > 4096) return { error: "Use a batch size from 1 to 4,096." };

  const info = ctx.model.info;
  const ref = ctx.model.ref;
  const model: Spec["model"] = info
    ? {
        id: info.id,
        revision: info.revision,
        dtype: info.dtype,
        device: ref?.device ?? info.device,
        process_weights: info.process_weights,
      }
    : (form.modelRef ?? {
        id: "openai-community/gpt2",
        revision: null,
        dtype: "float32",
        device: "auto",
        process_weights: true,
      });
  // SAE features belong to an SAE: the loaded one, or the one a saved spec names.
  const features = usesFeatures(scope);
  const loaded = ctx.sae?.state === "ready" && ctx.sae.info ? { repo: ctx.sae.info.repo, path: ctx.sae.info.path, revision: ctx.sae.info.revision } : null;
  const sae = features ? (loaded ?? form.saeRef) : null;
  if (features && !sae) return { error: "Load the SAE these features belong to (Explore → Features)." };
  if (scope.kind === "features" && experiment.kind !== "attribution_patching") {
    return { error: "Only attribution patching estimates every SAE feature. Choose it, or patch chosen features." };
  }
  const spec: Spec = {
    logogram_spec: 2,
    name: form.name.trim() || suggestName({ experiment, scope }),
    notes: form.notes,
    model,
    dataset: { path: ctx.datasetPath, sha256: ctx.datasetSha, limit: form.limit },
    tokenization: { prepend_bos: form.prependBos },
    experiment,
    scope: scope.kind === "site_sets" ? { ...scope, sets: scope.sets.map((set) => ({ ...set, label: set.label.trim() })) } : scope,
    metric,
    statistics: { bootstrap: form.bootstrap, ci: form.ci, seed: form.statSeed, cluster: form.cluster },
    execution: { batch_size: form.batchSize },
    predictions: form.predictions,
    ...(sae ? { sae } : {}),
  };
  return { spec };
}

/** How what will run differs from the saved spec the form was opened from. */
export function savedDifferences(
  form: FormState,
  model: ModelStatus,
  datasetPath: string | null,
  datasetSha: string | null,
  name: (id: string) => string = (id) => id,
): string[] {
  const out: string[] = [];
  const ref = form.modelRef;
  const info = model.info;
  const short = (rev: string | null | undefined) => (rev ? rev.slice(0, 7) : "latest");
  if (ref && info) {
    if (ref.id !== info.id) {
      out.push(`It was written for ${name(ref.id)}; the loaded model is ${name(info.id)}.`);
    } else {
      if (ref.revision && ref.revision !== info.revision) {
        out.push(`It was written for revision ${short(ref.revision)}; the loaded model is ${short(info.revision)}.`);
      }
      if (ref.dtype !== info.dtype) out.push(`It was written for ${ref.dtype}; the model is loaded in ${info.dtype}.`);
      if (ref.process_weights !== info.process_weights) out.push("Weight processing differs from the saved spec.");
      if (ref.device !== "auto" && ref.device !== info.device) out.push(`It was written for ${ref.device}; the model is loaded on ${info.device}.`);
    }
  }
  const saved = form.savedDataset;
  if (saved && datasetPath) {
    if (saved.path !== datasetPath) out.push(`It used ${saved.path}; the prompts chosen now are ${datasetPath}.`);
    else if (saved.sha256 && datasetSha && saved.sha256 !== datasetSha) out.push(`${saved.path} has changed since it was saved.`);
  }
  return out;
}
