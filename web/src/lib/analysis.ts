import type { AnalysisOptions } from "../api/client";
import type { DatasetDetail, MetricSpec, ModelInfo, ModelRef, ModelStatus, PredictionSettings, PromptRecord, Spec } from "../api/types";
import { metricOf } from "./buildSpec";
import { formTokenization, type FormState } from "./formState";
import { modelRefOf, sameDevice } from "./spec";

/** Whether the loaded model is the one a spec names: the same files (a Hugging Face model by id
 * and revision, a local one by the hash of its files), dtype, weight processing and device. */
export function sameModel(model: ModelRef, info: ModelInfo): boolean {
  const source = model.source ?? "hub";
  if (source !== (info.source ?? "hub")) return false;
  const files = source === "local"
    ? model.sha256 === (info.sha256 ?? null)
    : model.id === info.id && (!model.revision || model.revision === info.revision);
  return files && model.dtype === info.dtype && model.process_weights === info.process_weights && sameDevice(model.device, info.device);
}

/** What changes how prompts are tokenized: when it changes, they are read again. */
export function tokenizationKey(options: Pick<AnalysisOptions, "prepend_bos" | "chat_template">): string {
  return JSON.stringify([options.prepend_bos, options.chat_template ?? null]);
}

/** What one prompt pair's tokens depend on, and nothing else: the pair, the loaded model, the
 * dataset's hash, how prompts are tokenized and the metric's kind (whether answers of several
 * tokens can be read). A change elsewhere in the analysis context, such as the batch size or the
 * prompt limit, or a reloaded copy of the same prompts, leaves the tokens as they are. */
export function tokensKey(
  datasetPath: string,
  index: number,
  record: PromptRecord,
  loaded: ModelInfo,
  options: AnalysisOptions,
  metric: MetricSpec,
): string {
  return JSON.stringify([datasetPath, index, record, modelRefOf(loaded), options.dataset_sha256 ?? null, tokenizationKey(options), metric.kind]);
}

export function predictionSettingsKey(s: PredictionSettings): string {
  return JSON.stringify([s.method, s.prompt_index, s.which, s.position.kind, s.position.kind === "index" ? s.position.index : null, s.top_k]);
}

export type AnalysisSource = "run" | "form";

/** Whether a view shows the active run's results: the results themselves, and the model map
 * while it paints them. */
export function showsRun(view: string | undefined, mapOverlay: "data" | "structure" | undefined): boolean {
  return view === "results" || (view === "explore" && (mapOverlay ?? "data") === "data");
}

/**
 * Where prompts are read from, as the token strip, baseline, attention and predictions show
 * them. The configure view works on the experiment form. A view that shows the active run reads
 * the prompts as that run did (BOS, prompt limit, batch, model, metric). The other views keep the
 * last choice: attention opened from the results follows the run, and a baseline checked while
 * setting up an experiment uses the form.
 */
export function analysisSourceFor(state: {
  view?: string;
  mapOverlay?: "data" | "structure";
  analysisSource: AnalysisSource;
  activeRunId: string | null;
}): AnalysisSource {
  if (state.view === "experiment") return "form";
  if (state.activeRunId && showsRun(state.view, state.mapOverlay)) return "run";
  return state.analysisSource;
}

/** One analysis context for the token strip, baseline, attention and their caches. */
export function analysisContext(state: {
  view?: string;
  mapOverlay?: "data" | "structure";
  analysisSource: AnalysisSource;
  activeRunId: string | null;
  runDetails: Record<string, { spec: Spec }>;
  form: FormState;
  model: ModelStatus;
  datasetPath: string | null;
  dataset: DatasetDetail | null;
  project: { session_id: string } | null;
}) {
  const spec = analysisSourceFor(state) === "run" && state.activeRunId
    ? state.runDetails[state.activeRunId]?.spec : undefined;
  const info = state.model.info;
  const loaded = info ? modelRefOf(info) : undefined;
  const model = spec?.model ?? loaded ?? state.form.modelRef ?? undefined;
  const tokenization = spec ? spec.tokenization : formTokenization(state.form);
  const options: AnalysisOptions = {
    model,
    dataset_sha256: spec?.dataset.sha256 ?? state.dataset?.sha256,
    prepend_bos: tokenization.prepend_bos,
    chat_template: tokenization.chat_template ?? null,
    limit: spec ? spec.dataset.limit : state.form.limit,
    batch_size: spec?.execution.batch_size ?? state.form.batchSize,
  };
  // The metric the prompts are read for: the run's, or the form's (null while the form's KL
  // divergence has no target). The baseline measures it; tokenizing reads only its kind, which
  // says whether answers of several tokens can be read.
  const metric: MetricSpec | null = spec?.metric ?? metricOf(state.form);
  const tokenMetric: MetricSpec = metric ?? readingOnly(state.form);
  const mismatch = model && info && !sameModel(model, info);
  // The start of a reply answers a message: until it is written, prompts can't be wrapped.
  const unanswered = options.chat_template?.kind === "assistant_prefix" && !options.chat_template.user.trim();
  const error = mismatch
    ? "Load the model with this experiment’s revision, dtype, device and weight processing to inspect its prompts."
    : unanswered
      ? "Write the person's message that the reply answers (Configure, under Data and model) to read the prompts in the chat template."
      : null;
  const errorTitle = mismatch ? "The loaded model differs" : "The chat template isn't complete";
  const n = Math.min(state.dataset?.n ?? 0, options.limit ?? Infinity);
  const key = JSON.stringify([state.project?.session_id, state.datasetPath, options, loaded]);
  return {
    options, n, error, errorTitle, metric, tokenMetric,
    key,
    /** The baseline depends on the metric too. */
    baselineKey: JSON.stringify([key, metric]),
    metricError: metric ? null : "Choose which prompt's prediction the KL divergence is measured from (Experiment, Metric) to check the baseline with it.",
    label: `${spec ? "Selected run" : "Experiment form"} · ${options.chat_template ? (options.chat_template.kind === "user_turn" ? "chat template, person's message" : "chat template, start of the reply") : `BOS ${options.prepend_bos ? "on" : "off"}`} · ${options.limit === null ? "all prompts" : `first ${options.limit} prompts`} · batch ${options.batch_size}`,
  };
}

/** The form's metric for tokenizing, while its KL divergence has no target yet: tokenizing reads
 * only the metric's kind, so the target it stands in with changes nothing. */
function readingOnly(form: FormState): MetricSpec {
  return form.metric === "kl"
    ? { kind: "kl", target: form.klTarget ?? "clean", normalization: form.normalization }
    : { kind: form.metric, normalization: form.normalization };
}
