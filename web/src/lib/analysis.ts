import type { AnalysisOptions } from "../api/client";
import type { DatasetDetail, MetricSpec, ModelStatus, PredictionSettings, Spec } from "../api/types";
import { metricOf } from "./buildSpec";
import type { FormState } from "./formState";

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
  const loaded = info ? {
    id: info.id, revision: info.revision, dtype: info.dtype, device: info.device,
    process_weights: info.process_weights,
  } : undefined;
  const model = spec?.model ?? loaded ?? state.form.modelRef ?? undefined;
  const options: AnalysisOptions = {
    model,
    dataset_sha256: spec?.dataset.sha256 ?? state.dataset?.sha256,
    prepend_bos: spec?.tokenization.prepend_bos ?? state.form.prependBos,
    limit: spec ? spec.dataset.limit : state.form.limit,
    batch_size: spec?.execution.batch_size ?? state.form.batchSize,
  };
  // The metric the prompts are read for: the run's, or the form's (null while the form's KL
  // divergence has no target). The baseline measures it; tokenizing reads only its kind, which
  // says whether answers of several tokens can be read.
  const metric: MetricSpec | null = spec?.metric ?? metricOf(state.form);
  const tokenMetric: MetricSpec = metric ?? readingOnly(state.form);
  const mismatch = model && info && (
    model.id !== info.id || (model.revision && model.revision !== info.revision) ||
    model.dtype !== info.dtype || model.process_weights !== info.process_weights ||
    (model.device !== "auto" && model.device !== info.device)
  );
  const error = mismatch ? "Load the model with this experiment’s revision, dtype, device and weight processing to inspect its prompts." : null;
  const n = Math.min(state.dataset?.n ?? 0, options.limit ?? Infinity);
  const key = JSON.stringify([state.project?.session_id, state.datasetPath, options, loaded]);
  return {
    options, n, error, metric, tokenMetric,
    key,
    /** The baseline depends on the metric too. */
    baselineKey: JSON.stringify([key, metric]),
    metricError: metric ? null : "Choose which prompt's prediction the KL divergence is measured from (Experiment, Metric) to check the baseline with it.",
    label: `${spec ? "Selected run" : "Experiment form"} · BOS ${options.prepend_bos ? "on" : "off"} · ${options.limit === null ? "all prompts" : `first ${options.limit} prompts`} · batch ${options.batch_size}`,
  };
}

/** The form's metric for tokenizing, while its KL divergence has no target yet: tokenizing reads
 * only the metric's kind, so the target it stands in with changes nothing. */
function readingOnly(form: FormState): MetricSpec {
  return form.metric === "kl"
    ? { kind: "kl", target: form.klTarget ?? "clean", normalization: form.normalization }
    : { kind: form.metric, normalization: form.normalization };
}
