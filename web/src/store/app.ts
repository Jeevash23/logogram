// Application state. Server state arrives over the API and the WebSocket event stream.

import { create } from "zustand";

import { api, ApiError, setProjectSession } from "../api/client";
import { analysisContext } from "../lib/analysis";
import type {
  BaselineReport,
  BaselineSpec,
  DatasetDetail,
  ExperimentKind,
  ExperimentSpec,
  Job,
  Layout,
  ModelShape,
  ModelStatus,
  ProjectInfo,
  RunDetail,
  RunListing,
  ScopeSpec,
  ServerState,
  SiteBase,
  SiteResult,
  Spec,
  ThemeSetting,
  SiteSpec,
  PredictionSettings,
  NoteInput,
  UpdateStatus,
} from "../api/types";
import type { ResolvedTheme } from "../lib/color";
import { siteFromSelection } from "../lib/spec";
import { fitSelection, isResidKind, sitesOnComponent, type Selection } from "../lib/sites";

export type Screen = "loading" | "system" | "projects" | "workbench" | "error";
export type View =
  | "explore"
  | "heads"
  | "predictions"
  | "notes"
  | "prompts"
  | "baseline"
  | "experiment"
  | "results"
  | "attention"
  | "compare"
  | "spec";

export const VIEWS: { id: View; label: string }[] = [
  { id: "prompts", label: "Prompts" },
  { id: "baseline", label: "Baseline" },
  { id: "experiment", label: "Experiment" },
  { id: "results", label: "Results" },
  { id: "attention", label: "Attention" },
  { id: "compare", label: "Compare" },
  { id: "spec", label: "Spec" },
  { id: "explore", label: "Model explorer" },
  { id: "heads", label: "Head comparison" },
  { id: "predictions", label: "Layer predictions" },
  { id: "notes", label: "Research notes" },
];

export type Workspace = "explore" | "experiment" | "evidence";
export function workspaceFor(view: View): Workspace {
  if (["explore", "attention", "heads", "predictions"].includes(view)) return "explore";
  if (["prompts", "baseline", "experiment", "spec"].includes(view)) return "experiment";
  return "evidence";
}

export interface LiveRun {
  runId: string;
  layout: Layout | null;
  sites: SiteBase[];
  cells: Record<number, SiteResult>;
  layersDone: number[];
  progress: { done: number; total: number; layer: number; memory: number | null; elapsed_s: number } | null;
  status: "running" | "finished" | "failed" | "cancelled";
  error?: string;
  nPrompts?: number;
  model?: ModelShape;
}

export interface Notice {
  id: number;
  tone: "info" | "error";
  text: string;
}

/** The experiment form. Choices that change a number, such as an ablation's baseline or which
 * prompts a direct attribution splits, have no default: they stay empty until chosen. */
export interface FormState {
  predictions: PredictionSettings | null;
  name: string;
  nameEdited: boolean;
  kind: ExperimentKind;
  direction: "clean_to_corrupt" | "corrupt_to_clean";
  baseline: BaselineSpec | null;
  dlaPrompts: "clean" | "corrupt" | null;
  /** Steering: which prompts receive the direction (no default), the strengths as typed, the
   * share of pairs that train the direction, its seed, and whether a random control runs. */
  steerApplyTo: "clean" | "corrupt" | null;
  steerStrengths: string;
  steerTrain: number;
  steerSeed: number;
  steerControl: boolean;
  scope: ScopeSpec;
  normalization: "dataset_gap" | "prompt_gap";
  bootstrap: number;
  ci: number;
  statSeed: number;
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
  scope: { kind: "heads", position: { kind: "all" } },
  normalization: "dataset_gap",
  bootstrap: 1000,
  ci: 0.95,
  statSeed: 0,
  batchSize: 64,
  limit: null,
  prependBos: true,
  notes: "",
  modelRef: null,
  savedDataset: null,
  draftId: null,
};

export function formFromSpec(spec: Spec): FormState {
  const e = spec.experiment;
  return {
    predictions: spec.predictions ?? null,
    name: spec.name,
    nameEdited: true,
    kind: e.kind,
    direction: e.kind === "activation_patching" ? e.direction : "clean_to_corrupt",
    baseline: e.kind === "ablation" ? e.baseline : null,
    dlaPrompts: e.kind === "direct_logit_attribution" ? e.prompts : null,
    steerApplyTo: e.kind === "steering" ? e.apply_to : null,
    steerStrengths: e.kind === "steering" ? e.coefficients.join(", ") : DEFAULT_FORM.steerStrengths,
    steerTrain: e.kind === "steering" ? e.train_fraction : DEFAULT_FORM.steerTrain,
    steerSeed: e.kind === "steering" ? e.seed : DEFAULT_FORM.steerSeed,
    steerControl: e.kind === "steering" ? e.control : DEFAULT_FORM.steerControl,
    scope: spec.scope,
    normalization: spec.metric.normalization,
    bootstrap: spec.statistics.bootstrap,
    ci: spec.statistics.ci,
    statSeed: spec.statistics.seed,
    batchSize: spec.execution.batch_size,
    limit: spec.dataset.limit,
    prependBos: spec.tokenization.prepend_bos,
    notes: spec.notes,
    modelRef: spec.model,
    savedDataset: { path: spec.dataset.path, sha256: spec.dataset.sha256 },
    draftId: null,
  };
}

interface Store {
  exploreMode: "atlas" | "layer";
  mapOverlay: "data" | "structure";
  tokenPosition: number | null;
  stagedSites: SiteSpec[];
  headPins: Selection[];
  researchVersion: number;
  noteEditor: { value: NoteInput; edit?: { id: string; revision: number } } | null;
  connection: "connecting" | "connected" | "reconnecting";
  connectionError: string | null;
  analysisSource: "run" | "form";
  screen: Screen;
  bootError: string | null;
  version: string;
  firstRun: boolean;
  projectsParent: string;
  themeSetting: ThemeSetting;
  theme: ResolvedTheme;

  project: ProjectInfo | null;
  model: ModelStatus;
  job: Job | null;
  runs: RunListing[];

  view: View;
  datasetPath: string | null;
  dataset: DatasetDetail | null;
  promptIndex: number;
  activeRunId: string | null;
  runDetails: Record<string, RunDetail>;
  live: Record<string, LiveRun>;
  selection: Selection | null;
  mapMetric: "effect" | "delta";
  scaleMode: "auto" | "unit";
  /** Print each heatmap cell's value. Off by default: color carries the pattern. */
  cellValues: boolean;
  /** What is known about newer versions; checked only as the user allows. */
  update: UpdateStatus | null;
  compareIds: [string | null, string | null];
  compareMode: "side" | "diff";
  flags: Record<string, { against: string; sites: number[] }>;
  form: FormState;
  baselines: Record<string, BaselineReport>;
  specSource: "run" | "draft";
  inspectorFocus: { section: "evidence" | "runs"; at: number } | null;
  pendingRunId: string | null;

  paletteOpen: boolean;
  modelDialogOpen: boolean;
  robustnessDialogOpen: boolean;
  notices: Notice[];

  // actions
  boot: () => Promise<void>;
  applyServerState: (s: ServerState) => void;
  setTheme: (t: ThemeSetting) => void;
  resolveTheme: () => void;
  goto: (screen: Screen) => void;
  setView: (v: View) => void;
  notify: (text: string, tone?: Notice["tone"]) => void;
  dismiss: (id: number) => void;
  guard: <T>(fn: () => Promise<T>) => Promise<T | undefined>;

  enterProject: (project: ProjectInfo) => Promise<void>;
  leaveProject: () => void;
  refreshProject: () => Promise<void>;
  refreshRuns: () => Promise<void>;
  selectDataset: (path: string | null, source?: "run" | "form") => Promise<void>;
  reloadDataset: () => Promise<void>;
  setPromptIndex: (i: number) => void;
  applyJob: (job: Job) => void;
  checkUpdates: () => Promise<void>;
  setUpdateConsent: (allowed: boolean) => Promise<void>;
  fitSelection: () => void;

  openRun: (id: string | null, view?: View) => Promise<void>;
  openDraft: (id: string) => Promise<void>;
  loadRun: (id: string, force?: boolean) => Promise<RunDetail | undefined>;
  select: (sel: Selection | null) => void;
  setTokenPosition: (position: number | null) => void;
  stageSelection: (selection: Selection) => void;
  configureStaged: () => void;
  pinHead: (selection: Selection) => void;
  prepareNote: (sites: SiteSpec[]) => void;
  startRun: (spec: Spec) => Promise<void>;
  startRobustness: (runId: string, experiment: ExperimentSpec) => Promise<void>;
  /** Patch, for real, the strongest sites an attribution patching run estimated. */
  startVerification: (runId: string, top: number) => Promise<void>;
  cancelJob: () => Promise<void>;
  setCompare: (a: string | null, b: string | null) => void;
  setForm: (patch: Partial<FormState>) => void;
  prefillExperiment: (kind: FormState["kind"], sel: Selection) => void;
  focusInspector: (section: "evidence" | "runs") => void;

  handleEvent: (event: Record<string, unknown> & { type: string }) => void;
}

let noticeId = 0;
// Opening runs awaits the server; only the latest request may change what is shown.
let openSeq = 0;
let projectSeq = 0;
let datasetSeq = 0;
// One request per run detail at a time, however many components ask for it.
const inflight = new Map<string, Promise<RunDetail | undefined>>();
let starting = false;

function emptyLive(runId: string): LiveRun {
  return { runId, layout: null, sites: [], cells: {}, layersDone: [], progress: null, status: "running" };
}

function systemTheme(): ResolvedTheme {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

export const useStore = create<Store>((set, get) => ({
  exploreMode: "atlas",
  mapOverlay: "data",
  tokenPosition: null,
  stagedSites: [],
  headPins: [],
  researchVersion: 0,
  noteEditor: null,
  connection: "connecting",
  connectionError: null,
  analysisSource: "form",
  screen: "loading",
  bootError: null,
  version: "",
  firstRun: false,
  projectsParent: "",
  themeSetting: "light",
  theme: "light",

  project: null,
  model: { state: "none" },
  job: null,
  runs: [],

  view: "prompts",
  datasetPath: null,
  dataset: null,
  promptIndex: 0,
  activeRunId: null,
  runDetails: {},
  live: {},
  selection: null,
  mapMetric: "effect",
  scaleMode: "auto",
  cellValues: false,
  update: null,
  compareIds: [null, null],
  compareMode: "side",
  flags: {},
  form: DEFAULT_FORM,
  baselines: {},
  specSource: "run",
  inspectorFocus: null,
  pendingRunId: null,

  paletteOpen: false,
  modelDialogOpen: false,
  robustnessDialogOpen: false,
  notices: [],

  boot: async () => {
    try {
      const state = await api.state();
      get().applyServerState(state);
      if (state.first_run) set({ screen: "system" });
      else if (state.project) await get().enterProject(state.project);
      else set({ screen: "projects" });
    } catch (err) {
      set({ screen: "error", bootError: err instanceof Error ? err.message : String(err) });
    }
  },

  applyServerState: (s) => {
    set({
      version: s.version,
      firstRun: s.first_run,
      projectsParent: s.projects_parent,
      themeSetting: s.theme,
      update: s.update ?? null,
    });
    get().handleEvent({ type: "model", ...s.model });
    if (s.job) get().applyJob(s.job);
    else set({ job: null });
    get().resolveTheme();
    if (get().screen !== "loading" && s.project?.session_id !== get().project?.session_id) {
      if (s.project) void get().enterProject(s.project);
      else get().leaveProject();
    }
  },

  checkUpdates: async () => {
    const update = await get().guard(() => api.checkUpdate());
    if (update) set({ update });
  },

  setUpdateConsent: async (allowed) => {
    const saved = await get().guard(() => api.settings({ update_check: allowed }));
    if (!saved) return;
    const update = await api.update().catch(() => null);
    if (update) set({ update });
  },

  applyJob: (job) => {
    if (job.project_session && job.project_session !== get().project?.session_id) return;
    // A response can arrive after a newer event about the same job; keep the newer one.
    const current = get().job;
    if (current && current.id === job.id && current.version > job.version) return;
    set({ job });
  },

  setTheme: (t) => {
    set({ themeSetting: t });
    get().resolveTheme();
    void api.settings({ theme: t }).catch(() => undefined);
  },

  resolveTheme: () => {
    const setting = get().themeSetting;
    const theme = setting === "system" ? systemTheme() : setting;
    document.documentElement.dataset.theme = theme;
    set({ theme });
  },

  goto: (screen) => set({ screen }),
  setView: (view) => set({ view, ...(view === "experiment" ? { analysisSource: "form" as const } : {}) }),

  notify: (text, tone = "info") => {
    const id = ++noticeId;
    set((s) => ({ notices: [...s.notices.slice(-3), { id, tone, text }] }));
    if (tone === "info") window.setTimeout(() => get().dismiss(id), 5000);
  },
  dismiss: (id) => set((s) => ({ notices: s.notices.filter((n) => n.id !== id) })),

  guard: async (fn) => {
    const seq = projectSeq;
    try {
      return await fn();
    } catch (err) {
      const message = err instanceof ApiError || err instanceof Error ? err.message : String(err);
      if (seq === projectSeq) get().notify(message, "error");
      return undefined;
    }
  },

  enterProject: async (project) => {
    if (get().project?.session_id === project.session_id) return;
    const seq = ++projectSeq;
    setProjectSession(project.session_id);
    openSeq += 1; // whatever was being opened belongs to the previous project
    inflight.clear();
    set({
      project,
      tokenPosition: null, stagedSites: [], headPins: [], noteEditor: null, researchVersion: 0,
      job: get().job?.project_session && get().job?.project_session !== project.session_id ? null : get().job,
      datasetPath: null,
      dataset: null,
      promptIndex: 0,
      analysisSource: "form",
      view: "prompts",
      inspectorFocus: null,
      robustnessDialogOpen: false,
      notices: [],
      screen: "workbench",
      runs: [],
      activeRunId: null,
      pendingRunId: null,
      runDetails: {},
      live: {},
      selection: null,
      compareIds: [null, null],
      flags: {},
      form: DEFAULT_FORM,
      baselines: {},
      specSource: "run",
    });
    const first = project.datasets.find((d) => !d.error)?.path ?? null;
    await get().selectDataset(first);
    if (seq !== projectSeq) return;
    await get().refreshRuns();
    if (seq !== projectSeq) return;
    const runs = get().runs;
    // A run in progress (after a reload, say) comes first: its map fills in as it streams.
    const running = runs.find((r) => r.status === "running");
    const finished = runs.find((r) => r.status === "finished");
    const draft = runs.find((r) => r.status === "draft");
    if (running) {
      set((s) => ({ live: { ...s.live, [running.id]: s.live[running.id] ?? emptyLive(running.id) } }));
      await get().openRun(running.id, "results");
    } else if (finished) await get().openRun(finished.id, "explore");
    else if (draft) await get().openDraft(draft.id);
    else set({ view: project.datasets.length ? "baseline" : "prompts" });
  },

  leaveProject: () => {
    projectSeq += 1;
    openSeq += 1;
    datasetSeq += 1;
    inflight.clear();
    setProjectSession(null);
    set({ project: null, screen: "projects", datasetPath: null, dataset: null, promptIndex: 0,
      tokenPosition: null, stagedSites: [], headPins: [], noteEditor: null, researchVersion: 0,
      job: get().job?.project_session ? null : get().job,
      runs: [], runDetails: {}, activeRunId: null, pendingRunId: null, live: {}, selection: null,
      compareIds: [null, null], flags: {}, form: DEFAULT_FORM, baselines: {}, specSource: "run",
      analysisSource: "form", view: "prompts", inspectorFocus: null, robustnessDialogOpen: false,
      notices: [],
    });
  },

  refreshProject: async () => {
    const seq = projectSeq;
    const project = await get().guard(() => api.project());
    if (project && seq === projectSeq) set({ project });
  },

  refreshRuns: async () => {
    if (!get().project) return;
    const seq = projectSeq;
    const runs = await get().guard(() => api.runs());
    if (!runs || seq !== projectSeq) return;
    set({ runs });
    // A run this tab still shows as running may have ended while the event stream was down.
    for (const [id, live] of Object.entries(get().live)) {
      const listed = runs.find((r) => r.id === id);
      if (live.status !== "running" || !listed || listed.status === "running" || listed.status === "draft") continue;
      const status = listed.status as LiveRun["status"];
      set((st) => ({ live: { ...st.live, [id]: { ...st.live[id], status, error: listed.error ?? undefined } } }));
      void get().loadRun(id, true).then(() => get().fitSelection());
    }
  },

  selectDataset: async (path, source = "form") => {
    const seq = ++datasetSeq;
    const project = projectSeq;
    set({ datasetPath: path, dataset: null, promptIndex: 0, tokenPosition: null, stagedSites: [], analysisSource: source });
    if (!path) return;
    const detail = await get().guard(() => api.dataset(path, analysisContext(get()).options.prepend_bos));
    if (detail && seq === datasetSeq && project === projectSeq) set({ dataset: detail });
  },

  reloadDataset: async () => {
    // Token issues and lengths depend on the loaded model's tokenizer: refetch when it changes.
    const path = get().datasetPath;
    if (!path || !get().project) return;
    const seq = ++datasetSeq;
    const project = projectSeq;
    const detail = await api.dataset(path, analysisContext(get()).options.prepend_bos).catch(() => null);
    if (detail && seq === datasetSeq && project === projectSeq) set({ dataset: detail });
  },

  setPromptIndex: (i) => {
    const n = analysisContext(get()).n;
    if (n === 0) return;
    const next = Math.max(0, Math.min(n - 1, i));
    set({ promptIndex: next, ...(next !== get().promptIndex ? { tokenPosition: null } : {}) });
  },

  openRun: async (id, view) => {
    const seq = ++openSeq;
    const previous = get().activeRunId ? get().runDetails[get().activeRunId as string]?.spec.model : get().model.info;
    set({ activeRunId: id });
    if (view) set({ view });
    if (!id) return;
    const detail = await get().loadRun(id);
    if (!detail || seq !== openSeq) return; // another run was opened meanwhile
    // Follow the run's dataset so the token strip shows the prompts it used.
    if (detail.spec.dataset.path !== get().datasetPath) await get().selectDataset(detail.spec.dataset.path, "run");
    if (seq !== openSeq) return;
    const changed = previous && (previous.id !== detail.spec.model.id || previous.revision !== detail.spec.model.revision);
    set({ analysisSource: "run", promptIndex: 0, tokenPosition: null, ...(changed ? { stagedSites: [], headPins: [], selection: null } : {}) });
    void get().reloadDataset();
    get().fitSelection();
    if (detail.summary) {
      const derived = await api.derived(id).catch(() => [] as RunListing[]);
      const latest = derived.find((r) => r.status === "finished" && r.derived_from?.kind === "robustness");
      if (latest) {
        const cmp = await api.compare(id, latest.id).catch(() => null);
        if (cmp && seq === openSeq) set((s) => ({ flags: { ...s.flags, [id]: { against: latest.id, sites: cmp.flagged } } }));
      }
    }
  },

  openDraft: async (id) => {
    // A saved spec that hasn't run: load it into the experiment form, ready to run.
    const seq = ++openSeq;
    const detail = await get().loadRun(id, true);
    if (!detail || seq !== openSeq) return;
    set({ form: { ...formFromSpec(detail.spec), draftId: id }, view: "experiment", activeRunId: id, analysisSource: "form" });
    if (detail.spec.dataset.path !== get().datasetPath) await get().selectDataset(detail.spec.dataset.path);
  },

  loadRun: async (id, force = false) => {
    const cached = get().runDetails[id];
    if (cached && !force && cached.summary) return cached;
    const pending = inflight.get(id);
    if (pending && !force) return pending;
    const project = projectSeq;
    const request = get()
      .guard(() => api.run(id))
      .then(async (detail) => {
        if (!detail || projectSeq !== project) return undefined; // the project changed
        set((s) => ({ runDetails: { ...s.runDetails, [id]: detail } }));
        if (get().activeRunId === id && get().analysisSource === "run" && detail.spec.dataset.path !== get().datasetPath) {
          await get().selectDataset(detail.spec.dataset.path, "run");
          if (projectSeq !== project) return undefined;
        }
        return detail;
      })
      .finally(() => {
        if (inflight.get(id) === request) inflight.delete(id);
      });
    inflight.set(id, request);
    return request;
  },

  select: (selection) => set({ selection }),

  setTokenPosition: (tokenPosition) => set({ tokenPosition }),
  stageSelection: (selection) => {
    const site = siteFromSelection(selection, selection.kind ?? "resid_pre");
    if (get().tokenPosition !== null) site.position = { kind: "index", index: get().tokenPosition as number };
    const staged = get().stagedSites;
    if (staged.some((s) => JSON.stringify(s) === JSON.stringify(site))) return;
    set({ stagedSites: [...staged, site] });
  },
  configureStaged: () => {
    const st = get();
    if (!st.stagedSites.length) return;
    const saved = st.analysisSource === "run" && st.activeRunId ? st.runDetails[st.activeRunId]?.spec : undefined;
    const form = saved ? formFromSpec(saved) : st.form;
    set({ view: "experiment", analysisSource: "form", stagedSites: [], form: { ...form, scope: { kind: "sites", sites: st.stagedSites }, draftId: null, nameEdited: false, name: "" } });
  },
  pinHead: (selection) => {
    if (selection.part !== "head") return;
    const pins = get().headPins;
    if (pins.some((p) => p.layer === selection.layer && p.head === selection.head)) {
      set({ headPins: pins.filter((p) => p.layer !== selection.layer || p.head !== selection.head) });
    } else if (pins.length < 2) set({ headPins: [...pins, selection] });
    else get().notify("Two heads are pinned. Remove one in Head comparison to choose another.");
  },
  prepareNote: (sites) => {
    const st = get();
    const saved = st.activeRunId ? st.runDetails[st.activeRunId]?.spec : undefined;
    const model = saved?.model ?? analysisContext(st).options.model;
    if (!model || !sites.length) return;
    set({ noteEditor: { value: { title: sites.length === 1 ? `Layer ${sites[0].layer} · ${sites[0].kind === "head" ? `Head ${sites[0].head}` : sites[0].kind}` : `${sites.length} selected sites`, body: "", model, sites, run_id: saved ? st.activeRunId : null } } });
  },

  fitSelection: () => {
    // Keep the selected component when the shown run changes, dropping a position or residual
    // site that run doesn't have.
    const { selection, activeRunId } = get();
    if (!selection || !activeRunId) return;
    const { sites } = runView(get().runDetails[activeRunId], get().live[activeRunId]);
    if (sites.length === 0) return;
    const fitted = fitSelection(sites, selection);
    if (fitted !== selection) set({ selection: fitted });
  },

  startRun: async (spec) => {
    if (starting) return; // a second click while the first request is on its way
    starting = true;
    const seq = projectSeq;
    const draftId = get().form.draftId;
    const out = await get().guard(() => api.startRun(spec, draftId));
    starting = false;
    if (!out || seq !== projectSeq) return;
    openSeq += 1;
    get().applyJob(out.job);
    set((s) => ({
      form: { ...s.form, draftId: null },
      pendingRunId: out.run_id,
      activeRunId: out.run_id,
      analysisSource: "run",
      view: "results",
      // The run's first events may already have arrived over the WebSocket; keep them.
      live: { ...s.live, [out.run_id]: s.live[out.run_id] ?? emptyLive(out.run_id) },
    }));
    get().fitSelection();
    await get().refreshRuns();
    void get().loadRun(out.run_id, true);
  },

  startRobustness: async (runId, experiment) => {
    const seq = projectSeq;
    const out = await get().guard(() => api.robustness(runId, experiment));
    if (!out || seq !== projectSeq) return;
    get().applyJob(out.job);
    set((s) => ({
      pendingRunId: out.run_id,
      compareIds: [runId, out.run_id],
      compareMode: "side",
      view: "compare",
      live: { ...s.live, [out.run_id]: s.live[out.run_id] ?? emptyLive(out.run_id) },
    }));
    await get().refreshRuns();
  },

  startVerification: async (runId, top) => {
    const seq = projectSeq;
    const out = await get().guard(() => api.verify(runId, top));
    if (!out || seq !== projectSeq) return;
    get().applyJob(out.job);
    set((s) => ({
      pendingRunId: out.run_id,
      compareIds: [runId, out.run_id],
      compareMode: "side",
      view: "compare",
      live: { ...s.live, [out.run_id]: s.live[out.run_id] ?? emptyLive(out.run_id) },
    }));
    await get().refreshRuns();
  },

  cancelJob: async () => {
    await get().guard(() => api.cancelJob());
  },

  setCompare: (a, b) => set({ compareIds: [a, b] }),
  setForm: (patch) => {
    set((st) => ({ form: { ...st.form, ...patch }, analysisSource: "form" }));
    if ("prependBos" in patch) { set({ tokenPosition: null, stagedSites: [] }); void get().reloadDataset(); }
    get().setPromptIndex(get().promptIndex);
  },

  prefillExperiment: (kind, sel) => {
    const form = get().form;
    // Use the residual site of the active run when the selection is a residual cell.
    // A residual cell names its site; otherwise use the residual site the active run measured.
    const id = get().activeRunId;
    const { sites } = id ? runView(get().runDetails[id], get().live[id]) : { sites: [] as SiteBase[] };
    const measured = sitesOnComponent(sites, sel).find((x) => isResidKind(x.kind))?.kind;
    const residKind = sel.kind ?? (measured && isResidKind(measured) ? measured : "resid_pre");
    const site = siteFromSelection(sel, residKind);
    set({
      selection: sel,
      view: "experiment",
      analysisSource: "form",
      form: {
        ...form,
        kind,
        baseline: kind === "ablation" ? form.baseline : null,
        scope: { kind: "sites", sites: [site] },
        nameEdited: false,
        draftId: null,
        modelRef: null,
        savedDataset: null,
        notes: "",
      },
    });
  },

  focusInspector: (section) => set({ inspectorFocus: { section, at: Date.now() } }),

  handleEvent: (event) => {
    const { type } = event;
    const data = event as Record<string, unknown>;
    if (data.project_session && data.project_session !== get().project?.session_id) return;
    switch (type) {
      case "research.updated":
        set((st) => ({ researchVersion: st.researchVersion + 1 }));
        break;
      case "model": {
        const { type: _type, ...rest } = event;
        const model = rest as unknown as ModelStatus;
        const before = get().model.info;
        set({ model });
        const now = model.info;
        if (now?.id !== before?.id || now?.revision !== before?.revision || now?.dtype !== before?.dtype || now?.process_weights !== before?.process_weights || now?.device !== before?.device) {
          set({ baselines: {}, tokenPosition: null, ...(!get().activeRunId ? { stagedSites: [], headPins: [], selection: null } : {}) });
          if (now?.extra?.bos === false && get().form.prependBos) {
            // The spec says so too: the checkbox shows it, and the run records it.
            set((st) => ({ form: { ...st.form, prependBos: false } }));
            get().notify(`${now.id} has no beginning-of-sequence token, so prompts now start without one.`);
          }
          void get().reloadDataset();
        }
        break;
      }
      case "job": {
        const { type: _type, ...rest } = event;
        const job = rest as unknown as Job;
        get().applyJob(job);
        if (job.status !== "running") {
          void get().refreshRuns();
          if (job.status === "failed" && job.error) get().notify(job.error, "error");
          if (job.kind === "load_model" && job.status === "finished") get().notify("Model loaded.");
        }
        break;
      }
      case "run.started": {
        const runId = String(data.run_id);
        set((s) => ({
          live: {
            ...s.live,
            [runId]: {
              runId,
              layout: data.layout as Layout,
              sites: data.sites as SiteBase[],
              cells: {},
              layersDone: [],
              progress: null,
              status: "running",
              nPrompts: data.n_prompts as number,
              model: data.model as ModelShape,
            },
          },
        }));
        if (runId === get().activeRunId) get().fitSelection();
        void get().refreshRuns();
        break;
      }
      case "run.layer": {
        const runId = String(data.run_id);
        const sites = data.sites as SiteResult[];
        set((s) => {
          const prev = s.live[runId] ?? emptyLive(runId);
          const cells = { ...prev.cells };
          for (const site of sites) cells[site.index] = site;
          return {
            live: {
              ...s.live,
              [runId]: { ...prev, cells, layersDone: [...prev.layersDone, data.layer as number] },
            },
          };
        });
        break;
      }
      case "run.progress": {
        const runId = String(data.run_id);
        set((s) => {
          const prev = s.live[runId] ?? emptyLive(runId);
          return {
            live: { ...s.live, [runId]: { ...prev, progress: data as unknown as LiveRun["progress"] } },
          };
        });
        break;
      }
      case "run.finished":
      case "run.failed":
      case "run.cancelled": {
        const runId = String(data.run_id);
        const status = type.slice(4) as LiveRun["status"];
        set((s) => {
          const prev = s.live[runId] ?? emptyLive(runId);
          return { live: { ...s.live, [runId]: { ...prev, status, error: data.error as string | undefined } } };
        });
        void get().refreshRuns();
        void get().loadRun(runId, true).then(async () => {
          if (runId === get().activeRunId) get().fitSelection();
          // A finished robustness check paints its flags onto the original run.
          const detail = get().runDetails[runId];
          const from = detail?.manifest?.derived_from;
          if (status === "finished" && (from?.kind === "robustness" || from?.kind === "verification")) {
            const cmp = await api.compare(from.run, runId).catch(() => null);
            if (cmp) set((s) => ({ flags: { ...s.flags, [from.run]: { against: runId, sites: cmp.flagged } } }));
          }
        });
        if (status === "cancelled" && !data.replay) get().notify("Run cancelled. The spec and any dataset snapshot remain; partial results were discarded.");
        break;
      }
      case "update": {
        const { type: _type, ...rest } = event;
        set({ update: rest as unknown as UpdateStatus });
        break;
      }
      case "project": {
        if (data.session_id !== get().project?.session_id) {
          // Clear synchronously; stale form actions cannot run while the new project loads.
          get().leaveProject();
          const seq = projectSeq;
          void api.state().then((s) => { if (seq === projectSeq) get().applyServerState(s); })
            .catch(() => set({ connection: "reconnecting" }));
        }
        break;
      }
      default:
        break;
    }
  },
}));

/** The sites and results to draw for a run: the summary when finished, live cells while running. */
export function runView(
  detail: RunDetail | undefined,
  live: LiveRun | undefined,
): {
  layout: Layout | null;
  sites: SiteBase[];
  results: Record<number, SiteResult>;
  running: boolean;
  model: ModelShape | null;
} {
  if (detail?.summary) {
    const results: Record<number, SiteResult> = {};
    for (const s of detail.summary.sites) results[s.index] = s;
    return {
      layout: detail.summary.layout,
      sites: detail.summary.sites,
      results,
      running: false,
      model: detail.summary.model,
    };
  }
  if (live) {
    return {
      layout: live.layout,
      sites: live.sites,
      results: live.cells,
      running: live.status === "running",
      model: live.model ?? null,
    };
  }
  return { layout: null, sites: [], results: {}, running: false, model: null };
}
