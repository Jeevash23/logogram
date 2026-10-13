// Training an SAE or a tuned lens in the project (src/logogram/sae_training.py and
// src/logogram/lenses.py): the dialogs' fields, the training specs they make, and their words.

import type {
  LensTrainingSpec,
  ModelInfo,
  ModelRef,
  SAETrainingSpec,
  TrainedModel,
  TrainingPlan,
  TrainingProgress,
} from "../api/types";
import { bytes, count, duration, pct, plural } from "./format";
import { parseLayers } from "./spec";

/** A trained SAE's or lens's name, which is also its folder's (src/logogram/spec.py). */
export const TRAINED_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$/;

export type SaeSite = SAETrainingSpec["site"];
export const SAE_SITES: SaeSite[] = ["resid_pre", "resid_post", "attn_out", "mlp_out"];
export const SAE_SITE_WORDS: Record<SaeSite, string> = {
  resid_pre: "Residual stream before the layer",
  resid_post: "Residual stream after the layer",
  attn_out: "Attention output",
  mlp_out: "MLP output",
};

/** Where an SAE trained in the project reads and writes, in words. */
export function trainedSiteText(site: string | null | undefined, layer: number | null | undefined): string {
  switch (site) {
    case "resid_pre":
      return `Residual stream entering layer ${layer}`;
    case "resid_post":
      return `Residual stream leaving layer ${layer}`;
    case "attn_out":
      return `Layer ${layer}'s attention output`;
    case "mlp_out":
      return `Layer ${layer}'s MLP output`;
    default:
      return `${site} of layer ${layer}`;
  }
}

/** What the fields shared by both dialogs hold: numbers as typed, so one can be half written. */
export interface CommonFields {
  name: string;
  corpus: string;
  steps: string;
  batchTokens: string;
  lr: string;
  seed: string;
  heldOut: string;
  beta1: string;
  beta2: string;
  eps: string;
  logEvery: string;
  prependBos: boolean;
  maxTokens: string;
  modelBatch: string;
}

export interface SaeFields extends CommonFields {
  site: SaeSite;
  layer: number;
  dSae: string;
  k: string;
  normalize: SAETrainingSpec["normalize"];
  auxk: boolean;
  auxkK: string;
  auxkCoefficient: string;
  auxkDeadAfter: string;
}

export interface LensFields extends CommonFields {
  /** Every layer but the last, or the layers typed in `layers`. */
  everyLayer: boolean;
  layers: string;
  weightDecay: boolean;
  decay: string;
}

/** ``base``, or ``base-2``, ``base-3``..., whichever isn't taken. */
export function freeName(base: string, taken: string[]): string {
  if (!taken.includes(base)) return base;
  for (let i = 2; ; i++) if (!taken.includes(`${base}-${i}`)) return `${base}-${i}`;
}

/** A model's name as part of a trained SAE's or lens's: the last part of its id, in the
 * characters a name may hold. */
export function modelStem(id: string): string {
  const last = id.split("/").pop() ?? id;
  return last.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[^A-Za-z0-9]+/, "").slice(0, 40) || "model";
}

/** A power of two near ``x``. */
const nearPower = (x: number) => 2 ** Math.max(0, Math.round(Math.log2(Math.max(1, x))));

function commonFields(info: ModelInfo, name: string, corpus: string, steps: number, lr: string, eps: string, logEvery: number): CommonFields {
  return {
    name,
    corpus,
    steps: String(steps),
    batchTokens: "1024",
    lr,
    seed: "0",
    heldOut: "0.1",
    beta1: "0.9",
    beta2: "0.999",
    eps,
    logEvery: String(logEvery),
    prependBos: info.extra?.bos !== false,
    maxTokens: String(Math.min(128, info.n_ctx)),
    modelBatch: "16",
  };
}

/** Where the SAE dialog starts, for the loaded model: a TopK SAE eight times as wide as the
 * residual stream, keeping 32 features per token, with the learning rate Gao et al. (2024) scale
 * by the width and their AuxK. Every value is shown and can be changed. */
export function saeFields(info: ModelInfo, corpus: string, taken: string[]): SaeFields {
  const layer = Math.floor(info.n_layers / 2);
  const site: SaeSite = info.site_kinds.includes("resid_post") ? "resid_post" : (SAE_SITES.find((s) => info.site_kinds.includes(s)) ?? "resid_post");
  const dSae = 8 * info.d_model;
  const lr = 2e-4 / Math.sqrt(dSae / 2 ** 14);
  return {
    ...commonFields(info, freeName(`${modelStem(info.id)}-${site}-${layer}`, taken), corpus, 2000, Number(lr.toPrecision(2)).toString(), "6.25e-10", 20),
    site,
    layer,
    dSae: String(dSae),
    k: String(Math.min(32, dSae)),
    normalize: "mean_squared_norm",
    auxk: true,
    auxkK: String(Math.min(dSae, nearPower(info.d_model / 2))),
    auxkCoefficient: "0.03125",
    auxkDeadAfter: String(100 * 1024),
  };
}

/** Where the lens dialog starts: a translator for every layer but the last, trained with AdamW
 * and a little weight decay towards the logit lens. */
export function lensFields(info: ModelInfo, corpus: string, taken: string[]): LensFields {
  return {
    ...commonFields(info, freeName(`${modelStem(info.id)}-lens`, taken), corpus, 250, "0.001", "1e-8", 10),
    everyLayer: true,
    layers: "",
    weightDecay: true,
    decay: "0.001",
  };
}

type Built<T> = { spec: T; error?: undefined } | { spec?: undefined; error: string };

/** A number typed in a field: what the field says it must be, or an error that says so. */
function reader() {
  const errors: string[] = [];
  const read = (text: string, label: string, check: (x: number) => boolean, rule: string, whole = false): number => {
    const value = Number(text.trim());
    if (!text.trim() || !Number.isFinite(value) || (whole && !Number.isInteger(value)) || !check(value)) {
      errors.push(`${label} must be ${rule}.`);
    }
    return value;
  };
  return { read, errors };
}

function readCommon(f: CommonFields, model: ModelRef, read: ReturnType<typeof reader>["read"], errors: string[]) {
  if (!TRAINED_NAME.test(f.name.trim())) {
    errors.push("Name it with letters, digits, dots, dashes and underscores, starting with a letter or digit.");
  }
  if (!f.corpus) errors.push("Choose a corpus to train on.");
  return {
    name: f.name.trim(),
    model,
    corpus: { name: f.corpus, sha256: null },
    tokenization: {
      prepend_bos: f.prependBos,
      max_tokens: read(f.maxTokens, "Tokens per text", (x) => x >= 2 && x <= 4096, "a whole number from 2 to 4,096", true),
    },
    steps: read(f.steps, "Steps", (x) => x >= 1, "a whole number, 1 or more", true),
    batch_tokens: read(f.batchTokens, "Tokens per step", (x) => x >= 1, "a whole number, 1 or more", true),
    lr: read(f.lr, "The learning rate", (x) => x > 0 && x <= 1, "more than 0 and at most 1"),
    betas: [
      read(f.beta1, "Adam's first beta", (x) => x >= 0 && x < 1, "at least 0 and less than 1"),
      read(f.beta2, "Adam's second beta", (x) => x >= 0 && x < 1, "at least 0 and less than 1"),
    ] as [number, number],
    eps: read(f.eps, "Adam's epsilon", (x) => x > 0 && x <= 1, "more than 0 and at most 1"),
    seed: read(f.seed, "The seed", (x) => x >= 0 && x < 2 ** 32, "a whole number from 0", true),
    held_out: read(f.heldOut, "The held-out share", (x) => x > 0 && x < 1, "more than 0 and less than 1"),
    log_every: read(f.logEvery, "Steps per point of the curve", (x) => x >= 1, "a whole number, 1 or more", true),
    model_batch: read(f.modelBatch, "Texts per forward pass", (x) => x >= 1 && x <= 1024, "a whole number from 1 to 1,024", true),
  };
}

/** The SAE's training spec, or the first thing wrong with the fields. */
export function saeSpec(f: SaeFields, model: ModelRef): Built<SAETrainingSpec> {
  const { read, errors } = reader();
  const common = readCommon(f, model, read, errors);
  const dSae = read(f.dSae, "The number of features", (x) => x >= 1 && x <= 2 ** 21, "a whole number from 1 to 2,097,152", true);
  const k = read(f.k, "Features per token", (x) => x >= 1, "a whole number, 1 or more", true);
  if (Number.isFinite(k) && Number.isFinite(dSae) && k > dSae) errors.push("Features per token can't be more than the number of features.");
  const auxk = f.auxk
    ? {
        k: read(f.auxkK, "AuxK's latents", (x) => x >= 1 && x <= dSae, "a whole number from 1 to the number of features", true),
        coefficient: read(f.auxkCoefficient, "AuxK's coefficient", (x) => x > 0 && x <= 10, "more than 0 and at most 10"),
        dead_after: read(f.auxkDeadAfter, "Tokens before a latent is dead", (x) => x >= 1, "a whole number, 1 or more", true),
      }
    : null;
  if (errors.length) return { error: errors[0] };
  return {
    spec: {
      logogram_sae_training: 1,
      ...common,
      site: f.site,
      layer: f.layer,
      architecture: { kind: "topk", k },
      d_sae: dSae,
      normalize: f.normalize,
      auxk,
    },
  };
}

/** The lens's training spec, or the first thing wrong with the fields. */
export function lensSpec(f: LensFields, model: ModelRef, nLayers: number): Built<LensTrainingSpec> {
  const { read, errors } = reader();
  const common = readCommon(f, model, read, errors);
  let layers: number[] | null = null;
  if (!f.everyLayer) {
    const parsed = parseLayers(f.layers);
    if (!parsed || parsed.length === 0) errors.push("List the layers that get a translator, such as 0, 2, 4.");
    else if (parsed.some((l) => l >= nLayers - 1)) {
      errors.push(`Translators go on layers 0 to ${nLayers - 2}: the last layer's readout is the model's own prediction.`);
    } else layers = [...new Set(parsed)].sort((a, b) => a - b);
  }
  const decay = f.weightDecay ? read(f.decay, "The weight decay", (x) => x > 0 && x <= 1, "more than 0 and at most 1") : null;
  if (errors.length) return { error: errors[0] };
  return { spec: { logogram_lens_training: 1, ...common, layers, weight_decay: decay } };
}

/** What every other setting of an SAE's training is, in one line. */
export function saeSettingsText(f: SaeFields, width: number): string {
  const auxk = f.auxk ? `AuxK: ${f.auxkK} dead latents × ${f.auxkCoefficient}, dead after ${count(Number(f.auxkDeadAfter))} tokens` : "no AuxK";
  const norm = f.normalize === "mean_squared_norm" ? `activations scaled to a mean squared norm of ${count(width)}` : "activations as they are";
  return [`Adam β ${f.beta1}, ${f.beta2}, ε ${f.eps}`, norm, auxk, ...commonSettings(f)].join(" · ");
}

export function lensSettingsText(f: LensFields): string {
  const decay = f.weightDecay ? `weight decay ${f.decay}` : "no weight decay";
  return [`AdamW β ${f.beta1}, ${f.beta2}, ε ${f.eps}`, decay, ...commonSettings(f)].join(" · ");
}

function commonSettings(f: CommonFields): string[] {
  return [
    `a point of the curve every ${f.logEvery} steps`,
    f.prependBos ? "texts begin with BOS" : "texts begin without BOS",
    `at most ${f.maxTokens} tokens per text`,
    `${f.modelBatch} texts per forward pass`,
  ];
}

/** What a plan says training will read and hold, and how long it takes here, in sentences. */
export function planText(plan: TrainingPlan): { reads: string; memory: string; time: string | null } {
  const t = plan.tokens;
  const x = plan.texts;
  const reads =
    t && x
      ? `Trains on ${x.read !== undefined ? `${count(x.read)} of the ` : "the "}${plural(x.train, "training text")}: ${count(t.used)} tokens${
          t.epochs && t.epochs > 1.05 ? `, each read ${t.epochs.toFixed(1)} times` : ""
        }. Measures on ${plural(x.held_out, "held-out text")} (${count(t.held_out)} tokens).`
      : "";
  const m = plan.memory;
  const verdict = m ? { fits: "fits", tight: "is tight", wont_fit: "won't fit" }[m.verdict] : "";
  const memory = m ? `Holds about ${bytes(m.total)} at once, of ${bytes(m.available)} free: it ${verdict}.` : "";
  const s = plan.seconds;
  const time = s
    ? `Takes about ${duration(s.total)} here${
        s.train !== undefined
          ? ` (${[s.collect !== undefined ? `collecting ${duration(s.collect)}` : null, `training ${duration(s.train)}`, s.evaluate !== undefined ? `measuring ${duration(s.evaluate)}` : null]
              .filter(Boolean)
              .join(", ")})`
          : ""
      }, from one forward pass and one step timed now.`
    : null;
  return { reads, memory, time };
}

/** Where a training is, in words, and how far along (null when that isn't known). */
export function progressText(progress: TrainingProgress, what: "SAE" | "lens", name: string): { text: string; share: number | null } {
  const done = progress.done ?? 0;
  const total = progress.total ?? 0;
  const share = total > 0 ? Math.min(1, done / total) : null;
  switch (progress.stage) {
    case "loading":
      return { text: "Loading the model", share: null };
    case "reading":
      return { text: "Reading and tokenizing the corpus", share: null };
    case "collecting":
      return { text: `Collecting activations: ${count(done)} of ${count(total)} tokens`, share };
    case "training":
      return { text: `Step ${count(done)} of ${count(total)}`, share };
    case "evaluating":
      return { text: `Measuring on the held-out texts: ${count(done)} of ${plural(total, "text")}`, share };
    case "writing":
      return { text: `Writing ${what === "SAE" ? "saes" : "lenses"}/${name}`, share: 1 };
    case "finished":
      return { text: `Trained ${what === "SAE" ? "saes" : "lenses"}/${name}`, share: 1 };
    default:
      return { text: "Starting", share: null };
  }
}

/** The model something was trained on, in words. */
export function trainedOnText(model: TrainedModel | null | undefined): string {
  if (!model) return "an unknown model";
  if (model.source === "local") return `the local model ${model.id}`;
  return `${model.id}${model.revision ? ` at ${model.revision.slice(0, 7)}` : ""}`;
}

/** Whether the loaded model is the one something was trained on. */
export function trainedOnLoaded(model: TrainedModel | null | undefined, info: ModelInfo | null | undefined): boolean {
  if (!model || !info) return false;
  if (model.source !== (info.source ?? "hub")) return false;
  const same = model.source === "local" ? !!model.sha256 && model.sha256 === info.sha256 : model.id === info.id && model.revision === info.revision;
  return same && model.process_weights === info.process_weights;
}

/** A share of the variance explained, of features, of the loss recovered: as a percentage. */
export const share = (x: number | null | undefined) => pct(x, 1);
