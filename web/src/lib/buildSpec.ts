// The experiment form as a version 3 spec, or what is missing. Every choice that can change a
// number is written out; nothing the form doesn't show is assumed. Restrictions a method places on
// the metric and the answers (src/logogram/direct.py, paths.py, engine.py, features.py and
// circuits.py) are checked here too, so a run doesn't fail on something the form could have said.

import type {
  Answer,
  Continuations,
  DatasetDetail,
  FaithfulnessCurveSpec,
  MetricSpec,
  ModelStatus,
  PositionSpec,
  SAERef,
  SAEStatus,
  ScopeSpec,
  SiteKind,
  Spec,
  SteeringDirection,
} from "../api/types";
import { siteSetsError } from "./circuits";
import { count, plural, shortRevision, visibleToken } from "./format";
import { formTokenization, type FormState } from "./formState";
import { modelRefOf, saeWritesText, sameDevice, siteText, splitCount, steerSiteOf, suggestName } from "./spec";

/** The residual stream sites the mean difference is added to (src/logogram/steering.py). */
const RESIDUAL: SiteKind[] = ["resid_pre", "resid_mid", "resid_post"];

/** What the system message and the person's message of a chat template may hold (spec.py). */
const CHAT_TEXT_MAX = 4000;

/** Why the form's chat template can't be used, or null: the start of a reply answers a message,
 * and the loaded model must ship a template. */
export function chatTemplateError(form: Pick<FormState, "chatTemplate">, model: ModelStatus): string | null {
  const t = form.chatTemplate;
  if (!t) return null;
  if (t.kind === "assistant_prefix" && !t.user.trim()) {
    return "Write the person's message that the reply answers (Chat template, under Data and model), or wrap prompts as the person's message.";
  }
  if ((t.system?.length ?? 0) > CHAT_TEXT_MAX || (t.kind === "assistant_prefix" && t.user.length > CHAT_TEXT_MAX)) {
    return "Keep the chat template's messages under 4,000 characters each.";
  }
  const info = model.info;
  if (info && info.extra?.chat_template_sha256 === null) {
    return `${info.id} has no chat template, so its prompts can only be read as plain text. Choose None under Chat template, or load a model made for chat.`;
  }
  return null;
}

/** What the form knows about the prompts it will run on. */
export interface DatasetFacts {
  /** Prompts used: the first `limit`, or every prompt. */
  n: number;
  /** Prompts whose answer or distractor is a set of single tokens. */
  sets: number;
  /** Prompts whose answer or distractor is a continuation of several tokens for the loaded model,
   * which only metrics that read continuations can score; null when no model is loaded. */
  continuations: number | null;
  /** Prompts with a set or a continuation of several tokens (each counted once): those whose
   * answer and distractor aren't both single tokens. */
  notSingle: number;
  /** The first prompt with a continuation of several tokens, to show. Either its answer or its
   * distractor (or both) is the continuation. */
  example: { index: number; answer: Answer; distractor: Answer } | null;
  /** Fields of the meta every used prompt has (and that can name a cluster), with how many
   * different values each takes. */
  fields: { field: string; clusters: number }[];
  /** Prompts whose clean and corrupt prompts have different token lengths for the loaded model,
   * matched label by label; null when no model is loaded. */
  uneven: number[] | null;
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
  const first = continued?.length ? continued[0] : null;
  const example = first === null ? null : { index: first, answer: records[first].answer, distractor: records[first].distractor };
  const several = new Set(continued ?? []);
  const notSingle = records.filter((r, i) => Array.isArray(r.answer) || Array.isArray(r.distractor) || several.has(i)).length;
  const fields: DatasetFacts["fields"] = [];
  if (n > 0) {
    const keys = Object.keys(records[0].meta ?? {}).filter((k) => CLUSTER_KEY.test(k));
    for (const key of keys) {
      if (!records.every((r) => r.meta && key in r.meta)) continue;
      const values = new Set(records.map((r) => clusterValue(r.meta?.[key])));
      fields.push({ field: key, clusters: values.size });
    }
  }
  const uneven = dataset.uneven ? dataset.uneven.map((u) => u.index).filter((i) => i < n) : null;
  return { n, sets, continuations: continued?.length ?? null, notSingle, example, fields, uneven };
}

// -- pairs of different lengths ---------------------------------------------------------------

/** What carries a value from one prompt into another at a position (src/logogram/sites.py), in a
 * few words; null for a method that reads one prompt at a time. Between prompts of different
 * lengths, such a position must be the last token or a label, which are matched from one prompt
 * to the other. */
export function carriedBy(form: Pick<FormState, "kind" | "baseline" | "steerDirection">): string | null {
  switch (form.kind) {
    case "activation_patching":
      return "patching";
    case "attribution_patching":
      return "attribution patching";
    case "path_patching":
      return "path patching";
    case "ablation":
      return form.baseline?.kind === "mean" ? "mean ablation" : form.baseline?.kind === "resample" ? "resample ablation" : null;
    case "steering":
      return form.steerDirection === "sae_feature" ? null : "steering by the mean difference";
    default:
      return null;
  }
}

/** Pairs of different lengths in words, to start a sentence: "Pair 3", "Pairs 1, 2, 4, 5, 7 and
 * 3 more". */
export function unevenPairsText(indices: readonly number[]): string {
  if (indices.length === 1) return `Pair ${indices[0]}`;
  const shown = indices.slice(0, 5).join(", ");
  return indices.length > 5 ? `Pairs ${shown} and ${count(indices.length - 5)} more` : `Pairs ${shown.replace(/, (\d+)$/, " and $1")}`;
}

/** Why every position and token indices can't be chosen with this method and these prompts, as a
 * sentence; null when they can. */
export function unlinedReason(form: Pick<FormState, "kind" | "baseline" | "steerDirection">, facts: DatasetFacts | null | undefined): string | null {
  const uneven = facts?.uneven;
  const method = carriedBy(form);
  if (!uneven?.length || !method) return null;
  return `${unevenPairsText(uneven)} ${uneven.length === 1 ? "has" : "have"} prompts of different lengths: ${method} matches them at the last token or a labelled position, not at every position or a token index.`;
}

/** The scope with every position and token index replaced by the last token, for prompts of
 * different lengths. */
export function lastInstead(scope: ScopeSpec): ScopeSpec {
  const fix = (p: PositionSpec): PositionSpec => (p.kind === "all" || p.kind === "index" ? { kind: "last" } : p);
  switch (scope.kind) {
    case "heads":
    case "layer_components":
    case "features":
    case "head_inputs":
    case "neurons":
      return { ...scope, position: fix(scope.position) };
    case "sites":
      return { ...scope, sites: scope.sites.map((x) => ({ ...x, position: fix(x.position) })) };
    case "layer_position":
      return scope.positions === "each" ? { ...scope, positions: "labels" } : scope;
    case "site_sets":
      return { ...scope, sets: scope.sets.map((set) => ({ ...set, sites: set.sites.map((x) => ({ ...x, position: fix(x.position) })) })) };
    case "graph":
    case "edges":
      return scope; // edges have no position to choose
  }
}

/** The first position of a scope that doesn't line up between prompts of different lengths: every
 * position, or a token index. */
function unlinedPosition(scope: ScopeSpec): PositionSpec | null {
  const bad = (p: PositionSpec) => p.kind === "all" || p.kind === "index";
  const positions: PositionSpec[] =
    scope.kind === "sites"
      ? scope.sites.map((x) => x.position)
      : scope.kind === "site_sets"
        ? scope.sets.flatMap((set) => set.sites.map((x) => x.position))
        : scope.kind === "layer_position"
          ? scope.positions === "each" ? [{ kind: "all" }] : []
          : scope.kind === "graph" || scope.kind === "edges"
            ? [{ kind: "all" }] // an edge carries its sender's output at every position
            : [scope.position];
  return positions.find(bad) ?? null;
}

/** Why the form can't run on these prompts of different lengths, or null (sites.py, atp.py and
 * circuits.py refuse the same). */
function unevenError(form: FormState, experiment: Spec["experiment"], scope: ScopeSpec, uneven: readonly number[]): string | null {
  const pairs = `${unevenPairsText(uneven)} ${uneven.length === 1 ? "has" : "have"} clean and corrupt prompts of different token lengths`;
  const instead = "Choose the last token or a labelled position, which are matched from one prompt to the other, or use pairs of equal length.";
  if (experiment.kind === "attribution_patching" && experiment.method === "integrated_gradients") {
    return `${pairs}. Integrated gradients run on inputs between the two prompts' embeddings, position by position, which needs their positions to line up. Estimate with one gradient, or use pairs of equal length.`;
  }
  if (scope.kind === "site_sets" && experiment.kind === "ablation" && (experiment.baseline.kind === "mean" || experiment.baseline.kind === "resample")) {
    return `${pairs}. ${experiment.baseline.kind === "mean" ? "Mean" : "Resample"} ablation of a set replaces whole activations position by position with other prompts', which needs the prompts to line up. Patch the sets at labels or the last token, zero-ablate them, or use pairs of equal length.`;
  }
  if (scope.kind === "graph" || scope.kind === "edges") {
    const what = scope.kind === "edges" ? "Path patching an edge" : experiment.kind === "acdc" ? "ACDC" : "Edge attribution";
    return `${pairs}. ${what} carries each sender's output from one prompt to the other at every position, which needs their positions to line up. Use pairs of equal length.`;
  }
  const method = carriedBy(form);
  if (!method) return null;
  const kept = scope.kind === "site_sets" ? scope.sets.find((set) => set.complement) : undefined;
  if (kept) {
    return `${pairs}. The set “${kept.label}” keeps its sites and patches every other position of the model, which has no meaning across different lengths. Patch sets of sites at labels or the last token, or zero-ablate the rest of the model, or use pairs of equal length.`;
  }
  const bad = unlinedPosition(scope);
  if (!bad) return null;
  const where = bad.kind === "index" ? `token index ${bad.index}` : "every position";
  return `${pairs}, so their positions don't line up by index: ${method} at ${where} has no meaning across different lengths. ${instead}`;
}

/** A prompt with an answer or distractor of several tokens, in words, after "such as". */
export function exampleText(example: DatasetFacts["example"]): string {
  if (!example) return "";
  const quoted = (a: Answer) =>
    typeof a === "string"
      ? `“${visibleToken(a)}”`
      : `any of ${a.slice(0, 3).map((m) => `“${visibleToken(m)}”`).join(", ")}${a.length > 3 ? ", …" : ""}`;
  return `prompt ${example.index}, answer ${quoted(example.answer)}, distractor ${quoted(example.distractor)}`;
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

/** A faithfulness curve's numbers of edges as typed: whole numbers from 1, in increasing order,
 * separated by commas or spaces. */
export function parseCounts(text: string): { values: number[] } | { error: string } {
  const parts = text.split(/[\s,;]+/).filter(Boolean);
  if (parts.length === 0) return { error: "Give at least one number of edges for the curve, such as 10." };
  const values = parts.map(Number);
  const bad = parts.find((_, i) => !Number.isInteger(values[i]) || values[i] < 1);
  if (bad !== undefined) return { error: `“${bad}” isn't a whole number from 1. Separate the numbers of edges with commas, like 10, 100, 1000.` };
  if (values.length > 24) return { error: "Use at most 24 numbers of edges for the curve." };
  if (values.some((v, i) => i > 0 && v <= values[i - 1])) return { error: "List the curve's numbers of edges in increasing order, each once." };
  return { values };
}

/** ACDC's threshold as typed: a number above 0, in the metric's units. */
export function parseThreshold(text: string): { value: number } | { error: string } {
  const typed = text.trim().replace(/−/g, "-");
  if (!typed) return { error: "Give ACDC a threshold: how small a change in the metric's mean leaves an edge out, such as 0.1." };
  const value = Number(typed);
  if (!Number.isFinite(value)) return { error: `“${text.trim()}” isn't a number. Give the threshold as a number above 0, like 0.1.` };
  if (value <= 0) return { error: "ACDC's threshold is a number above 0: with 0, no edge could be removed." };
  return { value };
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
  if (scope.kind === "heads" || scope.kind === "layer_components" || scope.kind === "features" || scope.kind === "head_inputs" || scope.kind === "neurons") {
    return scope.position.kind === "all";
  }
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
    // With grouped-query attention a key or value receiver names a key/value head (paths.py).
    const info = ctx.model.info;
    const kv = info?.extra?.n_key_value_heads ?? info?.n_heads;
    const missing = info && kv ? form.pathReceivers.find((r) => r.kind === "head" && r.input !== "q" && r.head >= kv) : undefined;
    if (info && kv && missing && missing.kind === "head") {
      const group = info.n_heads / kv;
      return {
        error: `L${missing.layer} KV${missing.head} doesn't exist: ${info.id} shares each key and value among ${plural(group, "head")}, so its layers have ${plural(kv, "key/value head")} (0 to ${kv - 1}), and a key or value receiver names one of them.`,
      };
    }
    experiment = { kind: "path_patching", direction: form.direction, receivers: form.pathReceivers, freeze_mlps: form.pathFreezeMlps };
  } else if (form.kind === "steering") {
    if (!form.steerApplyTo) return { error: "Choose which prompts to steer." };
    const strengths = parseStrengths(form.steerStrengths);
    if ("error" in strengths) return strengths;
    let direction: SteeringDirection = { kind: "mean_difference" };
    if (form.steerDirection === "sae_feature") {
      const feature = form.steerFeature;
      if (feature === null || !Number.isInteger(feature) || feature < 0) {
        return { error: "Give the number of the SAE feature to steer along: a whole number from 0." };
      }
      direction = { kind: "sae_feature", feature };
    }
    let continuations: Continuations | null = null;
    if (form.steerContinue) {
      const prompts = form.steerContinuePrompts;
      const tokens = form.steerContinueTokens;
      if (!Number.isInteger(prompts) || prompts < 1 || prompts > 8) return { error: "Continue from 1 to 8 of the held-out prompts." };
      if (!Number.isInteger(tokens) || tokens < 1 || tokens > 64) return { error: "Continue each prompt by 1 to 64 tokens." };
      continuations = { prompts, tokens };
    }
    experiment = {
      kind: "steering",
      apply_to: form.steerApplyTo,
      direction,
      coefficients: strengths.values,
      train_fraction: form.steerTrain,
      seed: form.steerSeed,
      control: form.steerControl,
      continuations,
    };
  } else if (form.kind === "edge_attribution") {
    let steps: number | null = null;
    if (form.atpMethod === "integrated_gradients") {
      if (!Number.isInteger(form.atpSteps) || form.atpSteps < 2 || form.atpSteps > 64) {
        return { error: "Integrated gradients take a whole number of steps from 2 to 64." };
      }
      steps = form.atpSteps;
    }
    let curve: FaithfulnessCurveSpec | null = null;
    if (form.edgeCurve) {
      if (!form.edgePatch) return { error: "Choose which edges the faithfulness curve patches." };
      const counts = parseCounts(form.edgeCounts);
      if ("error" in counts) return counts;
      if (!isSeed(form.edgeSeed)) return { error: "The random control's seed is a whole number from 0." };
      if (!Number.isInteger(form.edgeDraws) || form.edgeDraws < 1 || form.edgeDraws > 16) {
        return { error: "Draw the random edges from 1 to 16 times." };
      }
      curve = { counts: counts.values, patch: form.edgePatch, control: { seed: form.edgeSeed, draws: form.edgeDraws } };
    }
    experiment = { kind: "edge_attribution", direction: form.direction, method: form.atpMethod, steps, curve };
  } else if (form.kind === "acdc") {
    const threshold = parseThreshold(form.acdcThreshold);
    if ("error" in threshold) return threshold;
    if (!form.acdcRule) return { error: "Choose which change in the metric ACDC compares with its threshold." };
    experiment = { kind: "acdc", direction: form.direction, threshold: threshold.value, rule: form.acdcRule };
  } else if (form.kind === "direct_logit_attribution") {
    if (!form.dlaPrompts) return { error: "Choose which prompts' logit difference to split." };
    experiment = { kind: "direct_logit_attribution", prompts: form.dlaPrompts };
  } else {
    if (!form.baseline) return { error: "Choose a baseline for the ablation. Logogram never assumes one." };
    experiment = { kind: "ablation", baseline: form.baseline };
  }
  const scope = form.scope;
  // The edges of the model's graph are edge attribution's and ACDC's, and only they measure them.
  const onGraph = experiment.kind === "edge_attribution" || experiment.kind === "acdc";
  if (onGraph !== (scope.kind === "graph")) {
    return {
      error: onGraph
        ? `${experiment.kind === "acdc" ? "ACDC removes" : "Edge attribution estimates"} the edges of the model's graph. Choose the whole graph or some of its layers under What to sweep.`
        : "Only edge attribution and ACDC measure the edges of the model's graph. Choose one of them, or another sweep.",
    };
  }
  if (scope.kind === "graph" && scope.layers !== null) {
    if (scope.layers.length === 0) return { error: "Choose at least one layer whose heads and MLPs receive edges, or every layer." };
    const layers = ctx.model.info?.n_layers;
    if (layers !== undefined && scope.layers.some((l) => l >= layers)) {
      return { error: `The loaded model has ${plural(layers, "layer")} (0 to ${layers - 1}). Choose layers it has.` };
    }
  }
  // Edges one at a time (an edge attribution's verification): path patching, every head and MLP
  // held, into the receivers the edges name.
  if (scope.kind === "edges") {
    if (experiment.kind !== "path_patching") return { error: "Edges are measured one at a time by path patching. Choose it, or another sweep." };
    if (!experiment.freeze_mlps) {
      return { error: "Edges are path-patched with every head and MLP held, so the change reaches each receiver directly. Hold the MLPs." };
    }
    const receivers = new Map(scope.edges.map((x) => [JSON.stringify(x.receiver), x.receiver]));
    experiment = { ...experiment, receivers: [...receivers.values()] };
  }
  const positionError = (p: PositionSpec) =>
    p.kind === "label" && !p.label ? "Choose which named position to use." : null;
  if (scope.kind === "heads" || scope.kind === "layer_components" || scope.kind === "features" || scope.kind === "head_inputs" || scope.kind === "neurons") {
    const err = positionError(scope.position);
    if (err) return { error: err };
  }
  // Queries, keys, values and neurons (src/logogram/sites.py, direct.py): the loaded model offers
  // them where it measured, on loading, that they are what its attention and MLPs read.
  const loadedModel = ctx.model.info;
  const finer: SiteKind[] =
    scope.kind === "head_inputs"
      ? [scope.input]
      : scope.kind === "neurons"
        ? ["neuron"]
        : scope.kind === "sites"
          ? scope.sites.map((x) => x.kind).filter((k) => k === "q" || k === "k" || k === "v" || k === "neuron")
          : [];
  if (loadedModel) {
    const missing = finer.find((k) => !loadedModel.site_kinds.includes(k));
    if (missing) return { error: loadedModel.extra?.site_notes?.[missing as "q"] ?? `${loadedModel.id} has no ${missing} sites. Choose another sweep.` };
  }
  if (scope.kind === "neurons") {
    if (scope.layers.length === 0) return { error: "Choose the layers whose neurons to sweep, such as 3, 4 (What to sweep → Layers)." };
    if (new Set(scope.layers).size !== scope.layers.length) return { error: "List each layer once." };
    if (loadedModel) {
      const beyond = scope.layers.find((layer) => layer >= loadedModel.n_layers);
      if (beyond !== undefined) return { error: `Layer ${beyond} doesn't exist; ${loadedModel.id} has layers 0 to ${loadedModel.n_layers - 1}.` };
    }
  }
  if (experiment.kind === "direct_logit_attribution") {
    const noEffect = scope.kind === "head_inputs" ? (scope.input === "v" ? null : scope.input) : null;
    const listed = scope.kind === "sites" ? scope.sites.find((x) => x.kind === "q" || x.kind === "k") : undefined;
    if (noEffect || listed) {
      return { error: "Queries and keys have no direct effect: they act on the logits only through the softmax of their head's attention. Choose values, heads' outputs or neurons, or patch the queries and keys." };
    }
    if (scope.kind === "neurons" && scope.position.kind !== "last") return { error: "Direct effects of neurons are read at the last token. Set the position to the last token." };
  }
  if ((experiment.kind === "steering" || experiment.kind === "path_patching") && finer.length > 0) {
    return {
      error:
        experiment.kind === "steering"
          ? "Steering adds a direction to the residual stream, not to a head's query, key or value or a neuron. Choose residual sites."
          : "Paths start at heads' outputs, attention outputs or MLP outputs, not at queries, keys, values or neurons. Choose those.",
    };
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
    const example = facts.example ? ` (such as ${exampleText(facts.example)})` : "";
    const have = (k: number) => `${plural(k, "prompt")} ${k === 1 ? "has" : "have"}`;
    if (experiment.kind === "direct_logit_attribution" && facts.notSingle > 0) {
      return {
        error: `Direct logit attribution splits the logit difference between two single tokens, and ${have(facts.notSingle)} a set or several tokens as an answer or distractor. Use single-token answers, or activation patching.`,
      };
    }
    if (several > 0 && metric.kind === "logit_diff") {
      return {
        error: `${have(several)} an answer or distractor of several tokens${example}. The logit difference reads one token: choose the log-probability difference, or another metric that reads answers of several tokens.`,
      };
    }
    if (several > 0 && onGraph) {
      const what = experiment.kind === "acdc" ? "ACDC" : "Edge attribution";
      return { error: `${what} reads the metric at the last prompt position, and ${have(several)} an answer or distractor of several tokens${example}. Use single-token answers or sets.` };
    }
    if (several > 0 && experiment.kind === "path_patching") {
      return { error: `Path patching reads the metric at the last prompt position, and ${have(several)} an answer or distractor of several tokens${example}. Use single-token answers or sets, or activation patching.` };
    }
    if (several > 0 && experiment.kind === "ablation" && experiment.baseline.kind === "mean" && everyPosition(scope)) {
      return {
        error: `Mean ablation replaces each position with its mean over prompts of the same length, and the tokens appended to read an answer or distractor of several tokens${example} have no such mean. Ablate at one position, use zero or resample ablation, or use single-token answers.`,
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
    // Pairs of different lengths are matched at the last token or a label only.
    const uneven = facts.uneven?.length ? unevenError(form, experiment, scope, facts.uneven) : null;
    if (uneven) return { error: uneven };
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
  const chatError = chatTemplateError(form, ctx.model);
  if (chatError) return { error: chatError };

  const info = ctx.model.info;
  const ref = ctx.model.ref;
  const model: Spec["model"] = info
    ? modelRefOf(info, ref?.device ?? info.device)
    : (form.modelRef ?? {
        id: "openai-community/gpt2",
        source: "hub",
        revision: null,
        sha256: null,
        dtype: "float32",
        device: "auto",
        process_weights: true,
      });
  // SAE features belong to an SAE: the loaded one, or the one a saved spec names.
  const features = usesFeatures(scope);
  const steersFeature = experiment.kind === "steering" && experiment.direction.kind === "sae_feature";
  const loadedInfo = ctx.sae?.state === "ready" && ctx.sae.info ? ctx.sae.info : null;
  const loaded = loadedInfo ? { repo: loadedInfo.repo, path: loadedInfo.path, revision: loadedInfo.revision } : null;
  const sae = features || steersFeature ? (loaded ?? form.saeRef) : null;
  if (features && !sae) return { error: "Load the SAE these features belong to (Explore → Features)." };
  if (experiment.kind === "steering") {
    const direction = experiment.direction;
    if (direction.kind === "sae_feature") {
      if (!sae) return { error: "Load the SAE this feature belongs to (Explore → Features), or steer along the mean difference." };
      // Added where the SAE writes, and nowhere else (src/logogram/steering.py).
      if (loadedInfo) {
        if (direction.feature >= loadedInfo.d_sae) {
          return { error: `The loaded SAE has ${count(loadedInfo.d_sae)} features (0 to ${count(loadedInfo.d_sae - 1)}), so there is no feature ${direction.feature}.` };
        }
        const where = steerSiteOf(loadedInfo);
        if (scope.kind !== "sites" || scope.sites.some((x) => x.kind !== where.kind || x.layer !== where.layer)) {
          return {
            error: `Steering along an SAE feature adds its decoder direction where the SAE writes: ${saeWritesText(loadedInfo)}. Choose that site alone (${siteText({ ...where, position: { kind: "all" } })}), at one token of each prompt.`,
          };
        }
      }
    } else if (scope.kind === "sites" && scope.sites.some((x) => !RESIDUAL.includes(x.kind))) {
      return { error: "Steering by the mean difference adds it to the residual stream. Choose residual sites, or steer along an SAE feature where its SAE writes." };
    }
  }
  if (scope.kind === "features" && experiment.kind !== "attribution_patching") {
    return { error: "Only attribution patching estimates every SAE feature. Choose it, or patch chosen features." };
  }
  const spec: Spec = {
    logogram_spec: 4,
    name: form.name.trim() || suggestName({ experiment, scope }),
    notes: form.notes,
    model,
    dataset: { path: ctx.datasetPath, sha256: ctx.datasetSha, limit: form.limit },
    tokenization: formTokenization(form),
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

/** How what will run differs from the saved spec the form was opened from. A loaded SAE takes the
 * place of the one the form's features belong to (see buildSpec), so another one is said too. */
export function savedDifferences(
  form: FormState,
  model: ModelStatus,
  datasetPath: string | null,
  datasetSha: string | null,
  name: (id: string) => string = (id) => id,
  sae?: SAEStatus,
): string[] {
  const out: string[] = [];
  const ref = form.modelRef;
  const info = model.info;
  if (ref && info) {
    const local = ref.source === "local" || info.source === "local";
    if (local && (ref.source !== info.source || ref.sha256 !== info.sha256)) {
      out.push(`It was written for ${name(ref.id)}${ref.source === "local" ? ", a local model" : ""}; the loaded model is ${name(info.id)}${info.source === "local" ? ", a local model with other files" : ""}.`);
    } else if (!local && ref.id !== info.id) {
      out.push(`It was written for ${name(ref.id)}; the loaded model is ${name(info.id)}.`);
    } else {
      if (ref.revision && ref.revision !== info.revision) {
        out.push(`It was written for revision ${shortRevision(ref.revision)}; the loaded model is ${shortRevision(info.revision)}.`);
      }
      if (ref.dtype !== info.dtype) out.push(`It was written for ${ref.dtype}; the model is loaded in ${info.dtype}.`);
      if (ref.process_weights !== info.process_weights) out.push("Weight processing differs from the saved spec.");
      if (!sameDevice(ref.device, info.device)) out.push(`It was written for ${ref.device}; the model is loaded on ${info.device}.`);
    }
  }
  const saved = form.savedDataset;
  if (saved && datasetPath) {
    if (saved.path !== datasetPath) out.push(`It used ${saved.path}; the prompts chosen now are ${datasetPath}.`);
    else if (saved.sha256 && datasetSha && saved.sha256 !== datasetSha) out.push(`${saved.path} has changed since it was saved.`);
  }
  // Its features are numbered in its own SAE: in another, the same number is another feature.
  const written = form.saeRef;
  const loaded = sae?.state === "ready" ? sae.info : undefined;
  const steersFeature = form.kind === "steering" && form.steerDirection === "sae_feature";
  if (written && loaded && (usesFeatures(form.scope) || steersFeature)) {
    const folder = (ref: SAERef) => `${ref.repo}${trimPath(ref.path) ? `/${trimPath(ref.path)}` : ""}`;
    if (written.repo !== loaded.repo || trimPath(written.path) !== trimPath(loaded.path)) {
      out.push(`It was written for the SAE ${folder(written)}; the loaded SAE is ${folder(loaded)}, where the same feature numbers are other features.`);
    } else if (written.revision && written.revision !== loaded.revision) {
      const now = loaded.revision ? `revision ${shortRevision(loaded.revision)}` : "unpinned";
      out.push(`It was written for revision ${shortRevision(written.revision)} of the SAE ${folder(written)}; the loaded one is ${now}.`);
    }
  }
  return out;
}

/** An SAE's folder in its repository, without the slashes around it. */
const trimPath = (path: string) => path.replace(/^\/+|\/+$/g, "");
