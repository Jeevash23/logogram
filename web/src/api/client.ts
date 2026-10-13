// A thin JSON client. The session cookie (set by the launch URL) authenticates every request.

import type {
  ActivationsExported,
  CorpusFormat,
  CorpusImported,
  CorpusListing,
  CorpusPage,
  CorpusTop,
  DirectionListing,
  HubDatasetFiles,
  ReadPosition,
  ReadSite,
  Representations,
  RowsSource,
  TopUnit,
  AttentionData,
  BaselineReport,
  ChatTemplateSpec,
  ChatWrapped,
  LocalModel,
  Comparison,
  DatasetChanged,
  DatasetCreated,
  DatasetDetail,
  DeletionPlan,
  DeriveRequest,
  DeviceCheck,
  EdgePlan,
  EdgeVerification,
  Environment,
  EstimateResponse,
  ExampleInfo,
  FolderListing,
  Job,
  MetricSpec,
  ModelRef,
  TaskDatasetCreated,
  TaskInfo,
  ModelPreset,
  ModelStatus,
  SAECatalogue,
  SAEFit,
  SAERef,
  SAEStatus,
  SAETrainingSpec,
  LensTrainingSpec,
  TrainedLens,
  TrainedSAE,
  TrainingPlan,
  TokenFeatures,
  FeatureReport,
  FeatureWrites,
  PositionSpec,
  ProjectInfo,
  PromptRecord,
  ProblemReport,
  RecentProject,
  Reproduction,
  RobustnessChange,
  RunAnnotationsChange,
  RunDetail,
  RunListing,
  ServerState,
  SiteDetail,
  Spec,
  SystemReport,
  ThemeSetting,
  TokenStripData,
  Dtype,
  Device,
  NoteInput,
  ResearchNote,
  PredictionReport,
  PredictionSettings,
  HeadKindsReport,
  LookAttention,
  LookContinuation,
  LookFeatureAlong,
  LookFeatures,
  LookReport,
  LookSteer,
  QueueAdded,
  QueuedRun,
  QueueStatus,
  RunMemory,
  ScopeSpec,
  ExperimentKind,
  Revisions,
  SweepField,
  SweepPlan,
  SweepRequest,
  SweepStarted,
  SweepTable,
  UpdateStatus,
} from "./types";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

/** What a tab whose session the server no longer accepts says (it restarted, or the link is old). */
export const SESSION_ENDED_TEXT = "This page's session has ended. Open the link printed in the terminal where Logogram is running.";

let projectSession: string | null = null;
export function setProjectSession(session: string | null) {
  projectSession = session;
}

export interface AnalysisOptions {
  model?: Spec["model"];
  dataset_sha256?: string | null;
  prepend_bos: boolean;
  /** The chat template prompts are wrapped in, as a spec's tokenization.chat_template says. */
  chat_template?: ChatTemplateSpec | null;
  limit: number | null;
  batch_size: number;
}

/** What every reading of activations states, and what it reads. */
export interface RepresentationsRequest {
  source: RowsSource;
  site: ReadSite;
  position: ReadPosition;
  prepend_bos: boolean;
  batch_size: number;
  model?: ModelRef;
}

/** Where a unit is most active in a corpus: every choice that changes the answer. */
export interface CorpusTopRequest {
  corpus: string;
  corpus_sha256: string | null;
  unit: TopUnit;
  texts: number | null;
  max_tokens: number;
  top_k: number;
  context: number;
  prepend_bos: boolean;
  batch_size: number;
  model?: ModelRef;
}

function errorMessage(data: unknown): string | null {
  if (!data || typeof data !== "object") return null;
  const record = data as { error?: unknown; detail?: unknown };
  if (record.error !== undefined) return String(record.error);
  if (record.detail !== undefined) return JSON.stringify(record.detail);
  return null;
}

/**
 * Routes that read or change the open project, as the server lists them (PROJECT_SCOPED in
 * src/logogram/server/app.py). A prefix ending in "/" covers everything under it; the others
 * cover the route itself and what is under it. A tab sends the project it shows with each of them.
 */
export const PROJECT_SCOPED = [
  "/api/project",
  "/api/research",
  "/api/directions",
  "/api/corpora",
  "/api/corpus",
  "/api/representations",
  "/api/activations",
  "/api/dataset",
  "/api/datasets/",
  "/api/tokenize",
  "/api/baseline",
  "/api/attention",
  "/api/predictions",
  "/api/sae/load",
  "/api/sae/fit",
  "/api/sae/tokens",
  "/api/sae/feature",
  "/api/sae/writes",
  "/api/saes/",
  "/api/lenses",
  "/api/runs",
  "/api/drafts",
  "/api/compare",
  "/api/queue",
  "/api/sweeps",
  // Not on the server's list, but closing must name the project the tab means to close.
  "/api/projects/close",
];

/** Whether a request to this path (with or without a query) says which project it is for. */
export function isProjectScoped(path: string): boolean {
  const route = path.split(/[?#]/)[0];
  return PROJECT_SCOPED.some((prefix) => (prefix.endsWith("/") ? route.startsWith(prefix) : route === prefix || route.startsWith(`${prefix}/`)));
}

async function request<T>(method: string, path: string, body?: unknown, format: "json" | "blob" = "json"): Promise<T> {
  let response: Response;
  const scoped = isProjectScoped(path);
  const session = projectSession;
  try {
    response = await fetch(path, {
      method,
      credentials: "same-origin",
      headers: {
        ...(body === undefined ? {} : { "content-type": "application/json" }),
        ...(scoped ? { "x-logogram-project": session ?? "none" } : {}),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(
      "Can't reach the Logogram server. Check that `logogram` is still running in your terminal.",
      0,
    );
  }
  if (scoped && projectSession !== session) throw new ApiError("The project changed while this request was running.", 409);
  if (response.status === 401) throw new ApiError(SESSION_ENDED_TEXT, 401);
  if (response.ok && format === "blob") {
    const blob = await response.blob();
    if (scoped && projectSession !== session) throw new ApiError("The project changed while this request was running.", 409);
    return blob as T;
  }
  const text = await response.text();
  let data: unknown = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { error: text };
    }
  }
  if (!response.ok) {
    throw new ApiError(errorMessage(data) ?? `Request failed (${response.status}).`, response.status);
  }
  if (scoped && projectSession !== session) throw new ApiError("The project changed while this request was running.", 409);
  return data as T;
}

const get = <T>(path: string) => request<T>("GET", path);
const post = <T>(path: string, body: unknown = {}) => request<T>("POST", path, body);

export const api = {
  state: () => get<ServerState>("/api/state"),
  update: () => get<UpdateStatus>("/api/update"),
  checkUpdate: () => post<UpdateStatus>("/api/update/check"),
  settings: (values: {
    system_check_seen?: boolean;
    theme?: ThemeSetting;
    update_check?: boolean;
    single_key_shortcuts?: boolean;
    walkthrough?: "finished" | "skipped";
  }) =>
    post<Record<string, unknown>>("/api/settings", values),
  system: () => get<SystemReport>("/api/system"),
  /** The versions and devices a rerun here would use, to compare with a run's manifest. */
  environment: () => get<Environment>("/api/environment"),
  /** A problem report, built by the server (which scrubs paths and names from the error) and
   * sent nowhere. */
  report: (body: { include_hardware: boolean; last_error: string | null }) => post<ProblemReport>("/api/report", body),

  folders: (path?: string) =>
    get<FolderListing>(`/api/fs${path ? `?path=${encodeURIComponent(path)}` : ""}`),
  recent: () => get<RecentProject[]>("/api/projects/recent"),
  /** Take a project out of the recent list (its folder is left as it is). */
  forgetRecent: (path: string) => request<RecentProject[]>("DELETE", `/api/projects/recent?path=${encodeURIComponent(path)}`),
  createProject: (name: string, parent: string | null) =>
    post<ProjectInfo>("/api/projects", { name, parent }),
  openProject: (path: string) => post<ProjectInfo>("/api/projects/open", { path }),
  /** Give the open project another name (its project.json; its folder keeps its name). */
  renameProject: (name: string) => request<ProjectInfo>("PATCH", "/api/project", { name }),
  /** What deleting the project in this folder would delete, and what stops it. */
  deletionPlan: (path: string) => get<DeletionPlan>(`/api/projects/deletion?path=${encodeURIComponent(path)}`),
  /** Delete the project in this folder for good, its name typed; the open project is closed first. */
  deleteProject: (path: string, name: string) =>
    request<RecentProject[]>("DELETE", `/api/projects?path=${encodeURIComponent(path)}&name=${encodeURIComponent(name)}`),
  examples: () => get<ExampleInfo[]>("/api/examples"),
  /** Copy a bundled example into the projects folder, once, and open it (a copy is never replaced). */
  openExample: (example: string) => post<ProjectInfo>("/api/projects/example", { example }),
  closeProject: () => post<{ ok: boolean }>("/api/projects/close"),
  project: () => get<ProjectInfo>("/api/project"),

  /** A dataset, and what the loaded model makes of its prompts: with or without the
   * beginning-of-sequence token, or wrapped in a chat template. */
  dataset: (path: string, prependBos = true, chatTemplate: ChatTemplateSpec | null = null) =>
    get<DatasetDetail>(
      `/api/dataset?path=${encodeURIComponent(path)}&prepend_bos=${chatTemplate ? false : prependBos}` +
        (chatTemplate ? `&chat_template=${encodeURIComponent(JSON.stringify(chatTemplate))}` : ""),
    ),
  tasks: () => get<TaskInfo[]>("/api/tasks"),
  /** Generate a task's prompts into a new dataset; the reply names the metric that reads them. */
  generateTask: (body: { task: string; name: string; n: number; seed: number; options: Record<string, unknown>; overwrite?: boolean }) =>
    post<TaskDatasetCreated>("/api/datasets/generate", body),
  importDataset: (name: string, text: string, overwrite = false) =>
    post<DatasetCreated>("/api/datasets/import", { name, text, overwrite }),
  savePair: (body: PromptRecord & { name: string; overwrite?: boolean }) =>
    post<DatasetCreated>("/api/datasets/pair", body),
  /** Save a dataset's prompt pairs, some edited and some left out, as a new dataset. */
  deriveDataset: (body: DeriveRequest) => post<DatasetCreated>("/api/datasets/derive", body),
  /** Give a dataset file another name; refused while a draft or a queued run names it. */
  renameDataset: (name: string, newName: string) =>
    post<DatasetChanged>(`/api/datasets/${encodeURIComponent(name)}/rename`, { name: newName }),
  /** Delete a dataset file; runs that ran on it keep their own copy of its prompts. */
  deleteDataset: (name: string) => request<DatasetChanged>("DELETE", `/api/datasets/${encodeURIComponent(name)}`),
  /** The metric decides whether answers of several tokens can be read. */
  tokenize: (body: { dataset?: string; index?: number; record?: PromptRecord; metric: MetricSpec } & AnalysisOptions) =>
    post<TokenStripData>("/api/tokenize", body),

  presets: () =>
    get<{
      presets: ModelPreset[];
      suggestions: string[];
    }>("/api/models/presets"),
  estimate: (body: { id: string; source?: "hub" | "local"; sha256?: string | null; revision?: string | null; dtype: Dtype; device: Device; process_weights: boolean }) =>
    post<EstimateResponse>("/api/models/estimate", body),
  /** Whether a model can be loaded on this device here, and if not, why. */
  checkDevice: (device: Device) => get<DeviceCheck>(`/api/models/device?device=${encodeURIComponent(device)}`),
  /** The models added from folders on this machine. */
  localModels: () => get<LocalModel[]>("/api/models/local"),
  /** Hash a model folder and add it under a name: a job, with progress in bytes. */
  addLocalModel: (path: string, name: string) => post<Job>("/api/models/local", { path, name }),
  /** Forget a local model; its folder is left as it is. */
  removeLocalModel: (name: string) => request<LocalModel[]>("DELETE", `/api/models/local/${encodeURIComponent(name)}`),
  /** A prompt as the loaded model reads it in its own chat template. */
  chatWrap: (body: { chat_template: ChatTemplateSpec; text: string; model?: ModelRef }) => post<ChatWrapped>("/api/chat/wrap", body),
  /** What a run of this batch size would need on the loaded model, against the memory free. */
  runMemory: (body: { batch_size: number; seq_len: number; method: ExperimentKind; scope?: ScopeSpec; prompts?: number; bootstrap?: number }) =>
    post<RunMemory>("/api/models/run-memory", body),
  loadModel: (body: ModelRef) => post<Job>("/api/models/load", body),
  unloadModel: () => post<ModelStatus>("/api/models/unload"),

  baseline: (dataset: string, options: AnalysisOptions, metric: MetricSpec) =>
    post<BaselineReport>("/api/baseline", { dataset, metric, ...options }),
  attention: (body: {
    dataset: string;
    index: number;
    layer: number;
    head: number;
    which: "clean" | "corrupt";
  } & AnalysisOptions) => post<AttentionData>("/api/attention", body),
  predictions: (body: { dataset: string; index: number; settings: PredictionSettings } & AnalysisOptions) => post<PredictionReport>("/api/predictions", body),
  look: (body: { text: string; top_k: number; position: number | null } & AnalysisOptions) => post<LookReport>("/api/look", body),
  lookAttention: (body: { text: string; layer: number; head: number } & AnalysisOptions) =>
    post<LookAttention>("/api/look/attention", body),
  lookFeatures: (body: { text: string; top_k: number } & AnalysisOptions) => post<LookFeatures>("/api/look/features", body),
  /** One feature of the loaded SAE along the text, and its largest activation there. */
  lookFeature: (body: { text: string; feature: number } & AnalysisOptions) => post<LookFeatureAlong>("/api/look/feature", body),
  /** What the model writes after the text, greedily; with `steer`, also with a feature added. */
  lookContinue: (body: { text: string; tokens: number; steer: LookSteer | null } & AnalysisOptions) =>
    post<LookContinuation>("/api/look/continue", body),
  headKinds: (body: { seed: number; sequences: number; length: number; bootstrap: number; ci: number } & AnalysisOptions) =>
    post<HeadKindsReport>("/api/heads/kinds", body),
  research: () => get<{ logogram_research: 1; notes: ResearchNote[] }>("/api/research"),
  directions: () => get<DirectionListing[]>("/api/directions"),
  /** Give a saved direction another name; runs that used it keep their copy. */
  renameDirection: (name: string, to: string) => post<DirectionListing>(`/api/directions/${encodeURIComponent(name)}/rename`, { name: to }),
  deleteDirection: (name: string) => request<{ ok: boolean }>("DELETE", `/api/directions/${encodeURIComponent(name)}`),
  corpora: () => get<CorpusListing[]>("/api/corpora"),
  /** A page of a corpus's texts: those containing `query`, a hundred at a time. */
  corpus: (name: string, query = "", page = 0) =>
    get<CorpusPage>(`/api/corpora/${encodeURIComponent(name)}?q=${encodeURIComponent(query)}&page=${page}`),
  /** A corpus from a file's text, read as `format` says. */
  importCorpus: (body: { name: string; text: string; filename: string; format: CorpusFormat; text_field: string; label_field: string | null; limit: number | null; overwrite?: boolean }) =>
    post<CorpusImported>("/api/corpora/import", body),
  /** A corpus from a dataset's prompts, labelled by side ("side"), a meta field ("meta:FIELD") or not. */
  corpusFromPrompts: (body: { name: string; dataset: string; sides: "clean" | "corrupt" | "both"; label: string | null; limit: number | null; overwrite?: boolean }) =>
    post<CorpusImported>("/api/corpora/import/dataset", body),
  /** A Hugging Face dataset's files (one request to Hugging Face, when asked). */
  hubDatasetFiles: (repo: string, revision: string | null) =>
    get<HubDatasetFiles>(`/api/hub/dataset-files?repo=${encodeURIComponent(repo)}${revision ? `&revision=${encodeURIComponent(revision)}` : ""}`),
  /** Download one file of a Hugging Face dataset and read it as a corpus: a job, with progress. */
  importHubCorpus: (body: { name: string; repo: string; path: string; revision: string | null; split: "lines" | "paragraphs"; text_field: string; label_field: string | null; limit: number | null; overwrite?: boolean }) =>
    post<Job>("/api/corpora/import/hub", body),
  renameCorpus: (name: string, to: string) => post<CorpusListing>(`/api/corpora/${encodeURIComponent(name)}/rename`, { name: to }),
  deleteCorpus: (name: string) => request<{ ok: boolean }>("DELETE", `/api/corpora/${encodeURIComponent(name)}`),
  /** Where a unit is most active in a corpus. */
  corpusTop: (body: CorpusTopRequest) => post<CorpusTop>("/api/corpus/top", body),
  /** The principal components of the activations at a site over prompts or texts. */
  representations: (body: RepresentationsRequest) => post<Representations>("/api/representations/pca", body),
  /** Save a principal component, or the mean difference between two classes, as a direction. */
  saveRepresentation: (
    body: RepresentationsRequest & {
      name: string;
      save: { kind: "component"; component: number } | { kind: "mean_difference"; positive: string; negative: string };
      overwrite?: boolean;
    },
  ) => post<DirectionListing>("/api/representations/direction", body),
  /** Write the activations at some sites to activations/NAME.safetensors in the project. */
  exportActivations: (body: Omit<RepresentationsRequest, "site"> & { sites: ReadSite[]; name: string; overwrite?: boolean }) =>
    post<ActivationsExported>("/api/activations/export", body),
  saveNote: (value: NoteInput, edit?: { id: string; revision: number }) => edit
    ? request<ResearchNote>("PUT", `/api/research/${encodeURIComponent(edit.id)}`, { ...value, revision: edit.revision })
    : post<ResearchNote>("/api/research", value),
  deleteNote: (note: ResearchNote) => request<{ ok: boolean }>("DELETE", `/api/research/${encodeURIComponent(note.id)}?revision=${note.revision}`),

  runs: () => get<RunListing[]>("/api/runs"),
  run: (id: string) => get<RunDetail>(`/api/runs/${encodeURIComponent(id)}`),
  exportRun: (id: string, text = false) =>
    request<Blob>("GET", `/api/runs/${encodeURIComponent(id)}/export.csv${text ? "?text=true" : ""}`, undefined, "blob"),
  exportSites: (id: string) => request<Blob>("GET", `/api/runs/${encodeURIComponent(id)}/sites.csv`, undefined, "blob"),
  runReport: (id: string) => request<Blob>("GET", `/api/runs/${encodeURIComponent(id)}/report.html`, undefined, "blob"),
  runFigure: (id: string, value: "effect" | "delta", unit: boolean) =>
    request<Blob>("GET", `/api/runs/${encodeURIComponent(id)}/figure.svg?value=${value}&unit=${unit}`, undefined, "blob"),
  methods: (id: string) => get<{ text: string }>(`/api/runs/${encodeURIComponent(id)}/methods`),
  startRun: (spec: Spec, draftId: string | null = null) =>
    post<{ run_id: string; job: Job }>("/api/runs", { spec, draft_id: draftId }),
  /** Run the spec when the job in progress ends, or now if nothing is running. */
  queueRun: (spec: Spec, draftId: string | null = null) =>
    post<QueueAdded>("/api/queue", { spec, draft_id: draftId }),
  removeQueuedRun: (id: string) => request<QueuedRun[]>("DELETE", `/api/queue/${encodeURIComponent(id)}`),
  /** The queue as the project keeps it, and how it stands (another process may have changed it). */
  queueStatus: () => get<QueueStatus>("/api/queue/state"),
  /** Run the queue here: its next run starts now if nothing is running. */
  startQueue: () => post<QueueStatus>("/api/queue/start"),
  /** Start no more of the queue's runs; the run in progress goes on. */
  pauseQueue: () => post<QueueStatus>("/api/queue/pause"),
  /** Take every run out of the queue. */
  clearQueue: () => request<QueueStatus>("DELETE", "/api/queue"),
  saveDraft: (spec: Spec, draftId: string | null = null) =>
    post<{ run_id: string; path: string }>("/api/drafts", { spec, draft_id: draftId }),
  rerun: (id: string) =>
    post<{ run_id: string; job: Job }>(`/api/runs/${encodeURIComponent(id)}/rerun`),
  /** Save what a probe or a learned subspace run found at one of its sites as a direction. */
  saveRunDirection: (id: string, body: { site: number; name: string; row: number | null }) =>
    post<DirectionListing>(`/api/runs/${encodeURIComponent(id)}/directions`, body),
  /** Whether a rerun reproduced the run it reran, value by value. */
  reproduction: (id: string) => get<Reproduction>(`/api/runs/${encodeURIComponent(id)}/reproduction`),
  robustness: (id: string, change: RobustnessChange) =>
    post<{ run_id: string; job: Job }>(`/api/runs/${encodeURIComponent(id)}/robustness`, change),
  verify: (id: string, top: number) =>
    post<{ run_id: string; job: Job }>(`/api/runs/${encodeURIComponent(id)}/verify`, { top }),
  /** An edge attribution spec on the loaded model, before it runs: the graph's size, what a run
   * costs, and why it would be refused. */
  edgePlan: (spec: Spec) => post<EdgePlan>("/api/edges/plan", { spec }),
  /** An edge attribution run's estimates against its latest verification by path patching; null
   * until one has finished. */
  edgeVerification: (id: string) => get<EdgeVerification | null>(`/api/runs/${encodeURIComponent(id)}/edge-verification`),
  derived: (id: string) => get<RunListing[]>(`/api/runs/${encodeURIComponent(id)}/derived`),
  /** Rename, star or archive a run: its annotations.json, never its spec. */
  annotateRun: (id: string, change: RunAnnotationsChange) =>
    request<RunListing>("PATCH", `/api/runs/${encodeURIComponent(id)}`, change),
  /** Delete a finished, failed or cancelled run, or a draft, folder and all. */
  deleteRun: (id: string) => request<{ ok: boolean }>("DELETE", `/api/runs/${encodeURIComponent(id)}`),

  saeSuggestions: (model: string) => get<{ repo: string; detail: string }[]>(`/api/sae/suggestions?model=${encodeURIComponent(model)}`),
  /** The published SAEs for a model from the catalogue that ships with Logogram: nothing is fetched. */
  saeCatalogue: (model: string, query: string, limit = 100) =>
    get<SAECatalogue>(`/api/sae/catalogue?model=${encodeURIComponent(model)}&q=${encodeURIComponent(query)}&limit=${limit}`),
  saeFolders: (repo: string) => get<{ repo: string; revision: string; folders: string[] }>(`/api/sae/folders?repo=${encodeURIComponent(repo)}`),
  loadSae: (ref: SAERef) => post<Job>("/api/sae/load", ref),
  /** The SAEs trained in the project, as their training recorded them. */
  projectSaes: () => get<TrainedSAE[]>("/api/saes/project"),
  /** What training this SAE on the loaded model would take; with measure, the time timed here. */
  planSae: (spec: SAETrainingSpec, measure = true) => post<TrainingPlan>("/api/saes/plan", { spec, measure }),
  /** Train an SAE in the project: a job whose progress carries the loss curve. */
  trainSae: (spec: SAETrainingSpec) => post<Job>("/api/saes/train", { spec }),
  /** The tuned lenses trained in the project, each saying whether it reads the loaded model. */
  lenses: () => get<TrainedLens[]>("/api/lenses"),
  planLens: (spec: LensTrainingSpec, measure = true) => post<TrainingPlan>("/api/lenses/plan", { spec, measure }),
  /** Train a tuned lens in the project, as a job. */
  trainLens: (spec: LensTrainingSpec) => post<Job>("/api/lenses/train", { spec }),
  unloadSae: () => post<SAEStatus>("/api/sae/unload"),
  saeFit: (dataset: string, options: Partial<AnalysisOptions>) => post<SAEFit>("/api/sae/fit", { dataset, ...options }),
  saeTokens: (dataset: string, index: number, which: "clean" | "corrupt", options: Partial<AnalysisOptions>) =>
    post<TokenFeatures>("/api/sae/tokens", { dataset, index, which, ...options }),
  saeFeature: (dataset: string, index: number, which: "clean" | "corrupt", feature: number, options: Partial<AnalysisOptions>) =>
    post<FeatureReport>("/api/sae/feature", { dataset, index, which, feature, ...options }),
  /** The tokens a feature promotes and suppresses directly, read at one token of one prompt: the
   * position given, or where the feature fires most in the prompt. */
  saeWrites: (body: { dataset: string; index: number; which: "clean" | "corrupt"; feature: number; position: PositionSpec | null; top: number } & Partial<AnalysisOptions>) =>
    post<FeatureWrites>("/api/sae/writes", body),
  siteDetail: (id: string, site: number) =>
    get<SiteDetail>(`/api/runs/${encodeURIComponent(id)}/sites/${site}`),
  compare: (a: string, b: string) =>
    get<Comparison>(`/api/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`),
  /** The fields a run's spec can be swept over, with its own value of each. */
  sweepFields: (id: string) => get<SweepField[]>(`/api/runs/${encodeURIComponent(id)}/sweep-fields`),
  /** The runs a sweep would make, without making them. */
  planSweep: (body: SweepRequest) => post<SweepPlan>("/api/sweeps/plan", body),
  /** Queue a sweep's runs: the first starts at once when nothing is running. */
  startSweep: (body: SweepRequest) => post<SweepStarted>("/api/sweeps", body),
  sweep: (id: string, choose: "first" | "all", top: number) =>
    get<SweepTable>(`/api/sweeps/${encodeURIComponent(id)}?choose=${choose}&top=${top}`),
  exportSweep: (id: string, choose: "first" | "all", top: number) =>
    request<Blob>("GET", `/api/sweeps/${encodeURIComponent(id)}/table.csv?choose=${choose}&top=${top}`, undefined, "blob"),
  /** A model's branches and tags on Hugging Face (one request to Hugging Face, when asked). */
  revisions: (id: string) => get<Revisions>(`/api/models/revisions?id=${encodeURIComponent(id)}`),
  cancelJob: () => post<{ job: Job | null }>("/api/jobs/cancel"),
};

