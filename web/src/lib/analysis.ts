import type { AnalysisOptions } from "../api/client";
import type { DatasetDetail, ModelStatus, PredictionSettings, Spec } from "../api/types";
import type { FormState } from "../store/app";

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
 * the prompts as that run did (BOS, prompt limit, batch, model). The other views keep the last
 * choice: attention opened from the results follows the run, and a baseline checked while
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
  const mismatch = model && info && (
    model.id !== info.id || (model.revision && model.revision !== info.revision) ||
    model.dtype !== info.dtype || model.process_weights !== info.process_weights ||
    (model.device !== "auto" && model.device !== info.device)
  );
  const error = mismatch ? "Load the model with this experiment’s revision, dtype, device and weight processing to inspect its prompts." : null;
  const n = Math.min(state.dataset?.n ?? 0, options.limit ?? Infinity);
  return {
    options, n, error,
    key: JSON.stringify([state.project?.session_id, state.datasetPath, options, loaded]),
    label: `${spec ? "Selected run" : "Experiment form"} · BOS ${options.prepend_bos ? "on" : "off"} · ${options.limit === null ? "all prompts" : `first ${options.limit} prompts`} · batch ${options.batch_size}`,
  };
}
