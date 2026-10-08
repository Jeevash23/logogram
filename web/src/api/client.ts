// A thin JSON client. The session cookie (set by the launch URL) authenticates every request.

import type {
  AttentionData,
  BaselineReport,
  Comparison,
  DatasetDetail,
  EstimateResponse,
  FolderListing,
  IOITemplate,
  Job,
  ModelPreset,
  ModelStatus,
  ProjectInfo,
  PromptRecord,
  RecentProject,
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
  ExperimentSpec,
  NoteInput,
  ResearchNote,
  PredictionReport,
  PredictionSettings,
  UpdateStatus,
} from "./types";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

let projectSession: string | null = null;
export function setProjectSession(session: string | null) {
  projectSession = session;
}

export interface AnalysisOptions {
  model?: Spec["model"];
  dataset_sha256?: string | null;
  prepend_bos: boolean;
  limit: number | null;
  batch_size: number;
}

function errorMessage(data: unknown): string | null {
  if (!data || typeof data !== "object") return null;
  const record = data as { error?: unknown; detail?: unknown };
  if (record.error !== undefined) return String(record.error);
  if (record.detail !== undefined) return JSON.stringify(record.detail);
  return null;
}

async function request<T>(method: string, path: string, body?: unknown, format: "json" | "blob" = "json"): Promise<T> {
  let response: Response;
  const scoped = /^\/api\/(project$|dataset(?:s)?(?:\/|\?)|tokenize$|baseline$|attention$|predictions$|research(?:\/|$)|runs(?:\/|$)|drafts$|compare\?|projects\/close$)/.test(path);
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
  if (response.status === 401) {
    throw new ApiError(
      "This page's session has ended. Open the link printed in the terminal where Logogram is running.",
      401,
    );
  }
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
  settings: (values: { system_check_seen?: boolean; theme?: ThemeSetting; update_check?: boolean }) =>
    post<Record<string, unknown>>("/api/settings", values),
  system: () => get<SystemReport>("/api/system"),

  folders: (path?: string) =>
    get<FolderListing>(`/api/fs${path ? `?path=${encodeURIComponent(path)}` : ""}`),
  recent: () => get<RecentProject[]>("/api/projects/recent"),
  createProject: (name: string, parent: string | null) =>
    post<ProjectInfo>("/api/projects", { name, parent }),
  openProject: (path: string) => post<ProjectInfo>("/api/projects/open", { path }),
  openExample: () => post<ProjectInfo>("/api/projects/example"),
  closeProject: () => post<{ ok: boolean }>("/api/projects/close"),
  project: () => get<ProjectInfo>("/api/project"),

  dataset: (path: string, prependBos = true) => get<DatasetDetail>(`/api/dataset?path=${encodeURIComponent(path)}&prepend_bos=${prependBos}`),
  ioiTemplates: () => get<IOITemplate[]>("/api/ioi/templates"),
  generateIOI: (body: {
    name: string;
    n: number;
    seed: number;
    templates: string[];
    patterns: ("ABBA" | "BABA")[];
    corruption: "flip" | "abc";
    overwrite?: boolean;
  }) => post<{ name: string; path: string; n: number }>("/api/datasets/ioi", body),
  importDataset: (name: string, text: string, overwrite = false) =>
    post<{ name: string; path: string; n: number }>("/api/datasets/import", { name, text, overwrite }),
  savePair: (body: PromptRecord & { name: string; overwrite?: boolean }) =>
    post<{ name: string; path: string; n: number }>("/api/datasets/pair", body),
  tokenize: (body: { dataset?: string; index?: number; record?: PromptRecord } & Partial<AnalysisOptions>) =>
    post<TokenStripData>("/api/tokenize", body),

  presets: () =>
    get<{
      presets: ModelPreset[];
      suggestions: string[];
    }>("/api/models/presets"),
  estimate: (body: { id: string; revision?: string | null; dtype: Dtype; device: Device }) =>
    post<EstimateResponse>("/api/models/estimate", body),
  loadModel: (body: {
    id: string;
    revision?: string | null;
    dtype: Dtype;
    device: Device;
    process_weights: boolean;
  }) => post<Job>("/api/models/load", body),
  unloadModel: () => post<ModelStatus>("/api/models/unload"),

  baseline: (dataset: string, options: AnalysisOptions) => post<BaselineReport>("/api/baseline", { dataset, ...options }),
  attention: (body: {
    dataset: string;
    index: number;
    layer: number;
    head: number;
    which: "clean" | "corrupt";
  } & AnalysisOptions) => post<AttentionData>("/api/attention", body),
  predictions: (body: { dataset: string; index: number; settings: PredictionSettings } & AnalysisOptions) => post<PredictionReport>("/api/predictions", body),
  research: () => get<{ logogram_research: 1; notes: ResearchNote[] }>("/api/research"),
  saveNote: (value: NoteInput, edit?: { id: string; revision: number }) => edit
    ? request<ResearchNote>("PUT", `/api/research/${encodeURIComponent(edit.id)}`, { ...value, revision: edit.revision })
    : post<ResearchNote>("/api/research", value),
  deleteNote: (note: ResearchNote) => request<{ ok: boolean }>("DELETE", `/api/research/${encodeURIComponent(note.id)}?revision=${note.revision}`),

  runs: () => get<RunListing[]>("/api/runs"),
  run: (id: string) => get<RunDetail>(`/api/runs/${encodeURIComponent(id)}`),
  exportRun: (id: string) => request<Blob>("GET", `/api/runs/${encodeURIComponent(id)}/export.csv`, undefined, "blob"),
  startRun: (spec: Spec, draftId: string | null = null) =>
    post<{ run_id: string; job: Job }>("/api/runs", { spec, draft_id: draftId }),
  saveDraft: (spec: Spec, draftId: string | null = null) =>
    post<{ run_id: string; path: string }>("/api/drafts", { spec, draft_id: draftId }),
  rerun: (id: string) =>
    post<{ run_id: string; job: Job }>(`/api/runs/${encodeURIComponent(id)}/rerun`),
  robustness: (id: string, experiment: ExperimentSpec) =>
    post<{ run_id: string; job: Job }>(`/api/runs/${encodeURIComponent(id)}/robustness`, {
      experiment,
    }),
  verify: (id: string, top: number) =>
    post<{ run_id: string; job: Job }>(`/api/runs/${encodeURIComponent(id)}/verify`, { top }),
  derived: (id: string) => get<RunListing[]>(`/api/runs/${encodeURIComponent(id)}/derived`),
  siteDetail: (id: string, site: number) =>
    get<SiteDetail>(`/api/runs/${encodeURIComponent(id)}/sites/${site}`),
  compare: (a: string, b: string) =>
    get<Comparison>(`/api/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`),
  cancelJob: () => post<{ job: Job | null }>("/api/jobs/cancel"),
};

export type Api = typeof api;
