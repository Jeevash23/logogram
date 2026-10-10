// An unsaved experiment form, kept in this browser's storage per project so a reload or reopening
// the project doesn't lose it. Storage can be missing, full or blocked: every access is guarded,
// and a stored form is used only when it has exactly the current form's shape. A form stored
// before a field existed takes the value the form starts with (and shows) for that field.

import { DEFAULT_FORM, type FormState } from "./formState";

const VERSION = 1;

export interface StoredForm {
  form: FormState;
  /** When it was stored, in milliseconds since 1970. */
  savedAt: number;
}

/** A short, stable name for a project folder, so storage keys don't spell out its path. */
export function projectKey(path: string): string {
  // 32-bit FNV-1a: enough to tell a person's projects apart.
  let hash = 0x811c9dc5;
  for (let i = 0; i < path.length; i++) {
    hash ^= path.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

export function formStorageKey(projectPath: string): string {
  return `logogram.experimentForm.${projectKey(projectPath)}`;
}

export function serializeForm(form: FormState, savedAt: number): string {
  return JSON.stringify({ version: VERSION, savedAt, form });
}

/** The stored form, or null when there is none or it doesn't fit the current form exactly. */
export function parseStoredForm(text: string | null | undefined): StoredForm | null {
  if (!text) return null;
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isObject(data) || data.version !== VERSION || !isNumber(data.savedAt)) return null;
  const form = parseForm(data.form);
  return form ? { form, savedAt: data.savedAt } : null;
}

/** A form, if the value has every field of one with the right kind of value; otherwise null.
 * Fields added since forms were first stored may be missing: they take the form's starting
 * value. */
export function parseForm(value: unknown): FormState | null {
  if (!isObject(value)) return null;
  const form: Record<string, unknown> = {};
  for (const [key, valid] of Object.entries(FIELDS)) {
    if (!(key in value) && ADDED.has(key as keyof FormState)) {
      form[key] = DEFAULT_FORM[key as keyof FormState];
      continue;
    }
    if (!(key in value) || !valid(value[key])) return null;
    form[key] = value[key];
  }
  form.scope = upgradeScope(form.scope as Record<string, unknown>);
  return form as unknown as FormState;
}

/** Fields of the form that a form stored by an earlier version doesn't have. */
const ADDED = new Set<keyof FormState>(["atpMethod", "atpSteps", "metric", "klTarget", "cluster"]);

/** A sweep stored before it had every field: an every-feature sweep chose and reported its
 * features on every prompt. */
function upgradeScope(scope: Record<string, unknown>): Record<string, unknown> {
  if (scope.kind === "features" && !("choose_on" in scope)) return { ...scope, choose_on: null, seed: null };
  return scope;
}

export function readStoredForm(projectPath: string): StoredForm | null {
  try {
    return parseStoredForm(globalThis.localStorage?.getItem(formStorageKey(projectPath)));
  } catch {
    return null; // storage blocked: nothing to restore
  }
}

export function writeStoredForm(projectPath: string, form: FormState, savedAt = Date.now()): void {
  try {
    globalThis.localStorage?.setItem(formStorageKey(projectPath), serializeForm(form, savedAt));
  } catch {
    // Storage full or blocked: the form still works, it just won't survive a reload.
  }
}

export function clearStoredForm(projectPath: string): void {
  try {
    globalThis.localStorage?.removeItem(formStorageKey(projectPath));
  } catch {
    // Storage blocked: there is nothing stored to clear.
  }
}

// -- the shape of a form -----------------------------------------------------------------------

type Check = (value: unknown) => boolean;

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

const isString: Check = (v) => typeof v === "string";
const isBoolean: Check = (v) => typeof v === "boolean";
function isNumber(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}
const oneOf = (...values: string[]): Check => (v) => typeof v === "string" && values.includes(v);
const orNull = (check: Check): Check => (v) => v === null || check(v);
const listOf = (check: Check): Check => (v) => Array.isArray(v) && v.every(check);
const optional = (check: Check): Check => (v) => v === undefined || v === null || check(v);

const STREAMS = ["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"];
const SIDES = oneOf("clean", "corrupt");

const isPosition: Check = (v) => {
  if (!isObject(v)) return false;
  if (v.kind === "all" || v.kind === "last") return true;
  if (v.kind === "index") return isNumber(v.index);
  if (v.kind === "label") return isString(v.label);
  return false;
};

const isSite: Check = (v) =>
  isObject(v) &&
  oneOf(...STREAMS, "head", "sae_feature")(v.kind) &&
  isNumber(v.layer) &&
  optional(isNumber)(v.head) &&
  optional(isNumber)(v.feature) &&
  isPosition(v.position);

const isSiteSet: Check = (v) => isObject(v) && isString(v.label) && listOf(isSite)(v.sites) && isBoolean(v.complement);

const isScope: Check = (v) => {
  if (!isObject(v)) return false;
  switch (v.kind) {
    case "heads":
      return isPosition(v.position);
    case "layer_position":
      return oneOf(...STREAMS)(v.site) && oneOf("each", "labels")(v.positions);
    case "layer_components":
      return listOf(oneOf(...STREAMS))(v.components) && isPosition(v.position);
    case "sites":
      return listOf(isSite)(v.sites);
    case "features":
      // Stored before the held-out choice existed, it has neither field (see upgradeScope).
      return isPosition(v.position) && isNumber(v.top) &&
        (("choose_on" in v) ? orNull(isNumber)(v.choose_on) && orNull(isNumber)(v.seed) : !("seed" in v));
    case "site_sets":
      return orNull(listOf(oneOf("head", "attn_out", "mlp_out")))(v.universe) && listOf(isSiteSet)(v.sets);
    default:
      return false;
  }
};

const isBaseline: Check = (v) => {
  if (!isObject(v)) return false;
  if (v.kind === "zero") return true;
  if (v.kind === "mean") return SIDES(v.reference);
  if (v.kind === "resample") return SIDES(v.pool) && isNumber(v.donors) && isNumber(v.seed);
  return false;
};

const isReceiver: Check = (v) =>
  isObject(v) && (v.kind === "logits" || (v.kind === "head" && isNumber(v.layer) && isNumber(v.head) && oneOf("q", "k", "v")(v.input)));

const isPredictions: Check = (v) =>
  isObject(v) &&
  v.method === "final_norm_logit_lens" &&
  isNumber(v.prompt_index) &&
  SIDES(v.which) &&
  isObject(v.position) &&
  (v.position.kind === "last" || (v.position.kind === "index" && isNumber(v.position.index))) &&
  isNumber(v.top_k);

const isSaeRef: Check = (v) => isObject(v) && isString(v.repo) && isString(v.path) && orNull(isString)(v.revision);

const isModelRef: Check = (v) =>
  isObject(v) && isString(v.id) && orNull(isString)(v.revision) && isString(v.dtype) && isString(v.device) && isBoolean(v.process_weights);

const isSavedDataset: Check = (v) => isObject(v) && isString(v.path) && orNull(isString)(v.sha256);

/** Every field of the form and what it may hold. Adding a field to FormState without adding it
 * here is a type error, so a stored form always has every field. */
const FIELDS: { [K in keyof FormState]-?: Check } = {
  predictions: orNull(isPredictions),
  name: isString,
  nameEdited: isBoolean,
  kind: oneOf("activation_patching", "ablation", "attribution_patching", "direct_logit_attribution", "path_patching", "steering"),
  direction: oneOf("clean_to_corrupt", "corrupt_to_clean"),
  baseline: orNull(isBaseline),
  dlaPrompts: orNull(SIDES),
  steerApplyTo: orNull(SIDES),
  steerStrengths: isString,
  steerTrain: isNumber,
  steerSeed: isNumber,
  steerControl: isBoolean,
  pathReceivers: listOf(isReceiver),
  pathFreezeMlps: isBoolean,
  atpMethod: oneOf("gradient", "integrated_gradients"),
  atpSteps: isNumber,
  saeRef: orNull(isSaeRef),
  scope: isScope,
  metric: oneOf("logit_diff", "logprob_diff", "logprob", "prob", "prob_diff", "kl"),
  klTarget: orNull(SIDES),
  normalization: oneOf("dataset_gap", "prompt_gap"),
  bootstrap: isNumber,
  ci: isNumber,
  statSeed: isNumber,
  cluster: orNull(isString),
  batchSize: isNumber,
  limit: orNull(isNumber),
  prependBos: isBoolean,
  notes: isString,
  modelRef: orNull(isModelRef),
  savedDataset: orNull(isSavedDataset),
  draftId: orNull(isString),
};
