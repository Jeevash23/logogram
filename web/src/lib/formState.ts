// The experiment form: what it holds, how it starts, and how a saved spec fills it. Kept apart from
// the store so that the form's draft storage and the spec builder can use it without a cycle.

import type {
  AttributionMethod,
  BaselineSpec,
  ExperimentKind,
  MetricKind,
  Normalization,
  PathReceiverSpec,
  PredictionSettings,
  SAERef,
  ScopeSpec,
  Side,
  Spec,
} from "../api/types";

/** Steps of integrated gradients the form offers until changed; the spec allows 2 to 64. */
export const DEFAULT_IG_STEPS = 10;

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
  /** Path patching: where the paths end (at least one), and whether MLPs are held too. */
  pathReceivers: PathReceiverSpec[];
  pathFreezeMlps: boolean;
  /** Attribution patching: one gradient, or integrated gradients over a number of steps. */
  atpMethod: AttributionMethod;
  atpSteps: number;
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
  prependBos: boolean;
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
  pathReceivers: [],
  pathFreezeMlps: false,
  atpMethod: "gradient",
  atpSteps: DEFAULT_IG_STEPS,
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
    direction: e.kind === "activation_patching" || e.kind === "attribution_patching" || e.kind === "path_patching" ? e.direction : "clean_to_corrupt",
    baseline: e.kind === "ablation" ? e.baseline : null,
    dlaPrompts: e.kind === "direct_logit_attribution" ? e.prompts : null,
    steerApplyTo: e.kind === "steering" ? e.apply_to : null,
    steerStrengths: e.kind === "steering" ? e.coefficients.join(", ") : DEFAULT_FORM.steerStrengths,
    steerTrain: e.kind === "steering" ? e.train_fraction : DEFAULT_FORM.steerTrain,
    steerSeed: e.kind === "steering" ? e.seed : DEFAULT_FORM.steerSeed,
    steerControl: e.kind === "steering" ? e.control : DEFAULT_FORM.steerControl,
    pathReceivers: e.kind === "path_patching" ? e.receivers : [],
    pathFreezeMlps: e.kind === "path_patching" ? e.freeze_mlps : false,
    atpMethod: e.kind === "attribution_patching" ? (e.method ?? "gradient") : DEFAULT_FORM.atpMethod,
    atpSteps: e.kind === "attribution_patching" && e.steps ? e.steps : DEFAULT_FORM.atpSteps,
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
    prependBos: spec.tokenization.prepend_bos,
    notes: spec.notes,
    modelRef: spec.model,
    savedDataset: { path: spec.dataset.path, sha256: spec.dataset.sha256 },
    draftId: null,
  };
}
