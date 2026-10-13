// The experiment form: what it holds, how it starts, and how a saved spec fills it. Kept apart from
// the store so that the form's draft storage and the spec builder can use it without a cycle.

import type {
  AcdcRule,
  AttributionMethod,
  BaselineSpec,
  ChatTemplateSpec,
  ExperimentKind,
  MetricKind,
  Normalization,
  PathReceiverSpec,
  PredictionSettings,
  SAERef,
  ScopeSpec,
  Side,
  Spec,
  SteeringDirection,
} from "../api/types";

/** Steps of integrated gradients the form offers until changed; the spec allows 2 to 64. */
export const DEFAULT_IG_STEPS = 10;

/** The numbers of edges a faithfulness curve keeps until changed: counts beyond the graph's size
 * are left out when it runs. */
export const DEFAULT_EDGE_COUNTS = "1, 3, 10, 30, 100, 300, 1000, 3000, 10000";

/** The experiment form. Choices that change a number, such as an ablation's baseline, which
 * prompts a direct attribution splits or which prompt a KL divergence is measured from, have no
 * default: they stay empty until chosen. The others start at a value the form shows. */
export interface FormState {
  predictions: PredictionSettings | null;
  name: string;
  nameEdited: boolean;
  kind: ExperimentKind;
  direction: "clean_to_corrupt" | "corrupt_to_clean";
  baseline: BaselineSpec | null;
  dlaPrompts: Side | null;
  /** Steering: which prompts receive the direction (no default), the strengths as typed, the
   * share of pairs that train the direction, its seed, and whether a random control runs. */
  steerApplyTo: Side | null;
  steerStrengths: string;
  steerTrain: number;
  steerSeed: number;
  steerControl: boolean;
  /** What steering adds: the mean difference of the pairs, or a feature of the SAE (`saeRef`, or
   * the loaded one) where the SAE writes, `steerFeature` being its number. */
  steerDirection: SteeringDirection["kind"];
  steerFeature: number | null;
  /** Whether the run records what the steered model says: greedy continuations of the first
   * `steerContinuePrompts` held-out prompts, `steerContinueTokens` new tokens each. */
  steerContinue: boolean;
  steerContinuePrompts: number;
  steerContinueTokens: number;
  /** Path patching: where the paths end (at least one), and whether MLPs are held too. */
  pathReceivers: PathReceiverSpec[];
  pathFreezeMlps: boolean;
  /** Attribution patching and edge attribution: one gradient, or integrated gradients over a
   * number of steps. */
  atpMethod: AttributionMethod;
  atpSteps: number;
  /** Edge attribution: whether the faithfulness curve runs, its numbers of edges as typed, which
   * edges it patches (no default: it decides what faithfulness 1 means), and its random control's
   * seed and number of draws. */
  edgeCurve: boolean;
  edgeCounts: string;
  edgePatch: "chosen" | "rest" | null;
  edgeSeed: number;
  edgeDraws: number;
  /** ACDC: the threshold as typed, in the metric's units, and which change in the metric it is
   * compared with. Neither has a default: they decide which edges stay. */
  acdcThreshold: string;
  acdcRule: AcdcRule | null;
  /** The SAE a saved spec's features belong to; used when no SAE is loaded. */
  saeRef: SAERef | null;
  scope: ScopeSpec;
  /** The metric, and for the KL divergence the prompt whose prediction it is measured from. */
  metric: MetricKind;
  klTarget: Side | null;
  normalization: Normalization;
  bootstrap: number;
  ci: number;
  statSeed: number;
  /** A field of the prompts' meta whose values group them for the bootstrap; null resamples single
   * prompts. */
  cluster: string | null;
  batchSize: number;
  limit: number | null;
  /** Whether prompts begin with the beginning-of-sequence token. A chat template writes the
   * beginning itself, so with one the spec says false; this keeps the choice for without. */
  prependBos: boolean;
  /** The model's own chat template prompts are wrapped in (the person's message, or the start of
   * the model's reply), or null for plain text. */
  chatTemplate: ChatTemplateSpec | null;
  notes: string;
  /** The model a saved spec asks for; used when no model is loaded. */
  modelRef: Spec["model"] | null;
  /** The prompts a saved spec was written for, to point out when they have changed. */
  savedDataset: { path: string; sha256: string | null } | null;
  /** A saved experiment that hasn't run: running the form fills its folder. */
  draftId: string | null;
}

export const DEFAULT_FORM: FormState = {
  predictions: null,
  name: "",
  nameEdited: false,
  kind: "activation_patching",
  direction: "clean_to_corrupt",
  baseline: null,
  dlaPrompts: null,
  steerApplyTo: null,
  steerStrengths: "-2, -1, 1, 2, 4",
  steerTrain: 0.5,
  steerSeed: 0,
  steerControl: true,
  steerDirection: "mean_difference",
  steerFeature: null,
  steerContinue: false,
  steerContinuePrompts: 3,
  steerContinueTokens: 20,
  pathReceivers: [],
  pathFreezeMlps: false,
  atpMethod: "gradient",
  atpSteps: DEFAULT_IG_STEPS,
  edgeCurve: true,
  edgeCounts: DEFAULT_EDGE_COUNTS,
  edgePatch: null,
  edgeSeed: 0,
  edgeDraws: 1,
  acdcThreshold: "",
  acdcRule: null,
  saeRef: null,
  scope: { kind: "heads", position: { kind: "all" } },
  metric: "logit_diff",
  klTarget: null,
  normalization: "dataset_gap",
  bootstrap: 1000,
  ci: 0.95,
  statSeed: 0,
  cluster: null,
  batchSize: 64,
  limit: null,
  prependBos: true,
  chatTemplate: null,
  notes: "",
  modelRef: null,
  savedDataset: null,
  draftId: null,
};

/** The form for a saved spec. Runs' specs arrive as version 2 (the server upgrades older ones);
 * a field missing anyway takes the form's own starting value. */
export function formFromSpec(spec: Spec): FormState {
  const e = spec.experiment;
  const metric = spec.metric;
  return {
    predictions: spec.predictions ?? null,
    name: spec.name,
    nameEdited: true,
    kind: e.kind,
    direction:
      e.kind === "activation_patching" ||
      e.kind === "attribution_patching" ||
      e.kind === "path_patching" ||
      e.kind === "edge_attribution" ||
      e.kind === "acdc"
        ? e.direction
        : "clean_to_corrupt",
    baseline: e.kind === "ablation" ? e.baseline : null,
    dlaPrompts: e.kind === "direct_logit_attribution" ? e.prompts : null,
    steerApplyTo: e.kind === "steering" ? e.apply_to : null,
    steerStrengths: e.kind === "steering" ? e.coefficients.join(", ") : DEFAULT_FORM.steerStrengths,
    steerTrain: e.kind === "steering" ? e.train_fraction : DEFAULT_FORM.steerTrain,
    steerSeed: e.kind === "steering" ? e.seed : DEFAULT_FORM.steerSeed,
    steerControl: e.kind === "steering" ? e.control : DEFAULT_FORM.steerControl,
    // Specs from before steering had a direction steered along the mean difference.
    steerDirection: e.kind === "steering" ? (e.direction?.kind ?? "mean_difference") : DEFAULT_FORM.steerDirection,
    steerFeature: e.kind === "steering" && e.direction?.kind === "sae_feature" ? e.direction.feature : null,
    steerContinue: e.kind === "steering" && !!e.continuations,
    steerContinuePrompts: e.kind === "steering" && e.continuations ? e.continuations.prompts : DEFAULT_FORM.steerContinuePrompts,
    steerContinueTokens: e.kind === "steering" && e.continuations ? e.continuations.tokens : DEFAULT_FORM.steerContinueTokens,
    pathReceivers: e.kind === "path_patching" ? e.receivers : [],
    pathFreezeMlps: e.kind === "path_patching" ? e.freeze_mlps : false,
    atpMethod: e.kind === "attribution_patching" || e.kind === "edge_attribution" ? (e.method ?? "gradient") : DEFAULT_FORM.atpMethod,
    atpSteps: (e.kind === "attribution_patching" || e.kind === "edge_attribution") && e.steps ? e.steps : DEFAULT_FORM.atpSteps,
    edgeCurve: e.kind === "edge_attribution" ? !!e.curve : DEFAULT_FORM.edgeCurve,
    edgeCounts: e.kind === "edge_attribution" && e.curve ? e.curve.counts.join(", ") : DEFAULT_FORM.edgeCounts,
    edgePatch: e.kind === "edge_attribution" && e.curve ? e.curve.patch : null,
    edgeSeed: e.kind === "edge_attribution" && e.curve ? e.curve.control.seed : DEFAULT_FORM.edgeSeed,
    edgeDraws: e.kind === "edge_attribution" && e.curve ? e.curve.control.draws : DEFAULT_FORM.edgeDraws,
    acdcThreshold: e.kind === "acdc" ? String(e.threshold) : DEFAULT_FORM.acdcThreshold,
    acdcRule: e.kind === "acdc" ? e.rule : null,
    saeRef: spec.sae ?? null,
    scope: spec.scope.kind === "features"
      ? { ...spec.scope, choose_on: spec.scope.choose_on ?? null, seed: spec.scope.seed ?? null }
      : spec.scope,
    metric: (metric.kind ?? "logit_diff") as MetricKind,
    klTarget: metric.kind === "kl" ? metric.target : null,
    normalization: metric.normalization,
    bootstrap: spec.statistics.bootstrap,
    ci: spec.statistics.ci,
    statSeed: spec.statistics.seed,
    cluster: spec.statistics.cluster ?? null,
    batchSize: spec.execution.batch_size,
    limit: spec.dataset.limit,
    // A spec with a chat template says false; without one again, prompts begin with BOS.
    prependBos: spec.tokenization.chat_template ? true : spec.tokenization.prepend_bos,
    chatTemplate: spec.tokenization.chat_template ?? null,
    notes: spec.notes,
    modelRef: spec.model,
    savedDataset: { path: spec.dataset.path, sha256: spec.dataset.sha256 },
    draftId: null,
  };
}

/** How the form's prompts are tokenized, as a spec writes it: a chat template writes the
 * beginning of the text, so with one there is no beginning-of-sequence token to prepend. */
export function formTokenization(form: Pick<FormState, "prependBos" | "chatTemplate">): Spec["tokenization"] {
  return { prepend_bos: form.chatTemplate ? false : form.prependBos, chat_template: form.chatTemplate };
}
