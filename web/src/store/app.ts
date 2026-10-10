// Application state. Server state arrives over the API and the WebSocket event stream.

import { create } from "zustand";

import { api, ApiError, setProjectSession } from "../api/client";
import { analysisContext, analysisSourceFor } from "../lib/analysis";
import type {
  BaselineReport,
  DatasetDetail,
  Job,
  SAEStatus,
  Layout,
  ModelShape,
  ModelStatus,
  ProjectInfo,
  RobustnessChange,
  RunDetail,
  RunListing,
  ServerState,
  SiteBase,
  SiteResult,
  Spec,
  ThemeSetting,
  SiteSpec,
  NoteInput,
  UpdateStatus,
} from "../api/types";
import type { ResolvedTheme } from "../lib/color";
import { siteFromSelection } from "../lib/spec";
import { DEFAULT_FORM, formFromSpec, type FormState } from "../lib/formState";
import { clearStoredForm, readStoredForm, writeStoredForm, type StoredForm } from "../lib/formDraft";
import {
  continuesEdit,
  editFields,
  EMPTY_HISTORY,
  forgetDraft,
  pushHistory,
  redoHistory,
  sameData,
  sameForm,
  undoHistory,
  type FormHistory,
  type LastEdit,
} from "../lib/formHistory";
import { fitSelection, isResidKind, sitesOnComponent, type Selection } from "../lib/sites";

export type Screen = "loading" | "system" | "projects" | "workbench" | "error";
export type View =
  | "explore"
  | "heads"
  | "predictions"
  | "features"
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
  { id: "features", label: "Features" },
  { id: "notes", label: "Research notes" },
];

export type Workspace = "explore" | "experiment" | "evidence";
export function workspaceFor(view: View): Workspace {
  if (["explore", "attention", "heads", "predictions", "features"].includes(view)) return "explore";
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

export interface NoticeAction {
  label: string;
  run: () => void;
}

export interface Notice {
  id: number;
  tone: "info" | "error";
  text: string;
  /** A button beside the text, such as Undo. The notice closes when it is pressed. */
  action?: NoticeAction;
  /** Kept until dismissed: it carries an action, or something the reader must not miss. Only
   * plain confirmations close by themselves. */
  persist?: boolean;
  /** A newer notice with the same key replaces this one. */
  key?: string;
}

export interface NoticeOptions {
  action?: NoticeAction;
  persist?: boolean;
  key?: string;
}

/** How long a plain confirmation stays on screen. */
export const NOTICE_MS = 5000;

/** How an action from elsewhere replaces the experiment form. */
export interface ReplaceFormOptions {
  /** The notice that offers to undo it: "Experiment form replaced" unless given; null for none. */
  notice?: string | null;
  /** Put back whatever else the action used up, such as the staged sites it moved into the form. */
  onUndo?: () => void;
}

export const FORM_REPLACED = "Experiment form replaced";
export const FORM_RESTORED = "Restored your unsaved experiment form";

export { DEFAULT_FORM, formFromSpec };
export type { FormState };

interface Store {
  exploreMode: "atlas" | "layer";
  mapOverlay: "data" | "structure";
  tokenPosition: number | null;
  stagedSites: SiteSpec[];
  headPins: Selection[];
  researchVersion: number;
  noteEditor: { value: NoteInput; edit?: { id: string; revision: number } } | null;
  /** The event stream. "ended": the server no longer accepts this tab's session, so it stopped. */
  connection: "connecting" | "connected" | "reconnecting" | "ended";
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
  /** The loaded sparse autoencoder; it belongs to the loaded model. */
  sae: SAEStatus;
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
  /** Undo and redo for the form: what each change replaced, and what was undone. */
  formHistory: FormHistory;
  baselines: Record<string, BaselineReport>;
  specSource: "run" | "draft";
  inspectorFocus: { section: "evidence" | "runs"; at: number } | null;
  pendingRunId: string | null;

  paletteOpen: boolean;
  modelDialogOpen: boolean;
  robustnessDialogOpen: boolean;
  /** Asking whether to cancel the running job, which discards its partial results. */
  cancelConfirmOpen: boolean;
  /** The sheet of keyboard shortcuts (? or the command palette). */
  shortcutsOpen: boolean;
  notices: Notice[];

  // actions
  boot: () => Promise<void>;
  applyServerState: (s: ServerState) => void;
  setTheme: (t: ThemeSetting) => void;
  resolveTheme: () => void;
  goto: (screen: Screen) => void;
  /** Show a view. Views that show the active run read its prompts as it did; see
   * analysisSourceFor. */
  setView: (v: View) => void;
  setMapOverlay: (overlay: "data" | "structure") => void;
  notify: (text: string, tone?: Notice["tone"], options?: NoticeOptions) => number;
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
  /** Load a saved, unrun experiment into the form. Quiet when opening a project, where the form
   * was empty: no undo step and no notice. */
  openDraft: (id: string, options?: { quiet?: boolean }) => Promise<void>;
  loadRun: (id: string, force?: boolean) => Promise<RunDetail | undefined>;
  select: (sel: Selection | null) => void;
  setTokenPosition: (position: number | null) => void;
  stageSelection: (selection: Selection) => void;
  configureStaged: () => void;
  pinHead: (selection: Selection) => void;
  prepareNote: (sites: SiteSpec[]) => void;
  startRun: (spec: Spec) => Promise<void>;
  startRobustness: (runId: string, change: RobustnessChange) => Promise<void>;
  /** Patch, for real, the strongest sites an attribution patching run estimated. */
  startVerification: (runId: string, top: number) => Promise<void>;
  cancelJob: () => Promise<void>;
  setCompare: (a: string | null, b: string | null) => void;
  /** A change made in the form. Rapid edits to the same fields are one undo step; with record
   * false (the suggested name following the rest) it is no step at all. */
  setForm: (patch: Partial<FormState>, options?: { record?: boolean }) => void;
  /** Replace the form from elsewhere (a map cell, a run, a diagnostic): one undo step, and a
   * notice offering to undo it. Returns whether anything changed. */
  replaceForm: (form: FormState, options?: ReplaceFormOptions) => boolean;
  undoForm: () => void;
  redoForm: () => void;
  /** The form is saved as this draft: running it fills the draft's folder, and the copy kept
   * in the browser in case of a reload is no longer needed. */
  markFormSaved: (draftId: string) => void;
  /** A finished, failed or cancelled run's spec, in the form, to change and run again. */
  editRun: (spec: Spec) => void;
  prefillExperiment: (kind: FormState["kind"], sel: Selection) => void;
  focusInspector: (section: "evidence" | "runs") => void;

  handleEvent: (event: Record<string, unknown> & { type: string }) => void;
}

let noticeId = 0;
// The last edit made in the form, so that typing makes one undo step rather than one per key.
let lastEdit: LastEdit | null = null;
const FORM_NOTICE = "form-replaced";
// The form as the open project last saved or ran it (an empty form when the project opens). A
// form that differs from it is unsaved, and is kept in the browser in case of a reload.
let savedForm: FormState = DEFAULT_FORM;
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
  // Until the saved appearance arrives, follow the system, as the page does (tokens.css).
  themeSetting: "system",
  theme: systemTheme(),

  project: null,
  model: { state: "none" },
  sae: { state: "none" },
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
  formHistory: EMPTY_HISTORY,
  baselines: {},
  specSource: "run",
  inspectorFocus: null,
  pendingRunId: null,

  paletteOpen: false,
  modelDialogOpen: false,
  robustnessDialogOpen: false,
  cancelConfirmOpen: false,
  shortcutsOpen: false,
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
      sae: s.sae ?? { state: "none" },
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
  setView: (view) => {
    const bos = analysisContext(get()).options.prepend_bos;
    set({ view });
    followAnalysisSource(bos);
  },
  setMapOverlay: (mapOverlay) => {
    const bos = analysisContext(get()).options.prepend_bos;
    set({ mapOverlay });
    followAnalysisSource(bos);
  },

  notify: (text, tone = "info", options = {}) => {
    const id = ++noticeId;
    const notice: Notice = { id, tone, text, ...options };
    set((s) => {
      const others = notice.key ? s.notices.filter((n) => n.key !== notice.key) : s.notices;
      return { notices: [...others.slice(-3), notice] };
    });
    // Errors, notices with an action and important ones stay until dismissed (WCAG 2.2.1).
    if (tone === "info" && !notice.action && !notice.persist) globalThis.setTimeout(() => get().dismiss(id), NOTICE_MS);
    return id;
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
    lastEdit = null;
    savedForm = DEFAULT_FORM;
    // Read before opening a draft or anything else can replace the form (and so the stored copy).
    const stored = readStoredForm(project.path);
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
      cancelConfirmOpen: false,
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
      formHistory: EMPTY_HISTORY,
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
    else if (draft) await get().openDraft(draft.id, { quiet: true });
    else set({ view: project.datasets.length ? "baseline" : "prompts" });
    if (stored && seq === projectSeq) restoreForm(stored);
  },

  leaveProject: () => {
    projectSeq += 1;
    lastEdit = null;
    savedForm = DEFAULT_FORM; // the stored copy stays, for when the project opens again
    openSeq += 1;
    datasetSeq += 1;
    inflight.clear();
    setProjectSession(null);
    set({ project: null, screen: "projects", datasetPath: null, dataset: null, promptIndex: 0,
      tokenPosition: null, stagedSites: [], headPins: [], noteEditor: null, researchVersion: 0,
      job: get().job?.project_session ? null : get().job,
      runs: [], runDetails: {}, activeRunId: null, pendingRunId: null, live: {}, selection: null,
      compareIds: [null, null], flags: {}, form: DEFAULT_FORM, formHistory: EMPTY_HISTORY, baselines: {}, specSource: "run",
      analysisSource: "form", view: "prompts", inspectorFocus: null, robustnessDialogOpen: false,
      cancelConfirmOpen: false, notices: [],
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

  openDraft: async (id, options = {}) => {
    // A saved spec that hasn't run: load it into the experiment form, ready to run.
    const seq = ++openSeq;
    const detail = await get().loadRun(id, true);
    if (!detail || seq !== openSeq) return;
    const bos = analysisContext(get()).options.prepend_bos;
    const form = { ...formFromSpec(detail.spec), draftId: id };
    set({ view: "experiment", activeRunId: id, analysisSource: "form" });
    savedForm = form;
    if (options.quiet) set({ form });
    else swapForm(form);
    if (detail.spec.dataset.path !== get().datasetPath) await get().selectDataset(detail.spec.dataset.path);
    else analysisChanged(bos);
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
        if (get().activeRunId === id && analysisSourceFor(get()) === "run" && detail.spec.dataset.path !== get().datasetPath) {
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
    const bos = analysisContext(st).options.prepend_bos;
    const staged = st.stagedSites;
    const saved = analysisSourceFor(st) === "run" && st.activeRunId ? st.runDetails[st.activeRunId]?.spec : undefined;
    const form = saved ? formFromSpec(saved) : st.form;
    set({ view: "experiment", analysisSource: "form", stagedSites: [] });
    swapForm({ ...form, scope: { kind: "sites", sites: staged }, draftId: null, nameEdited: false, name: "" }, {
      // Undo returns the sites to the tray, beside any staged since.
      onUndo: () => useStore.setState((now) => ({
        stagedSites: [...staged, ...now.stagedSites.filter((x) => !staged.some((y) => sameData(x, y)))],
      })),
    });
    analysisChanged(bos);
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
    const ran = get().form;
    const draftId = ran.draftId;
    const out = await get().guard(() => api.startRun(spec, draftId));
    starting = false;
    if (!out || seq !== projectSeq) return;
    openSeq += 1;
    // The run's folder keeps the spec: the form as run is saved.
    savedForm = { ...ran, draftId: null };
    get().applyJob(out.job);
    set((s) => ({
      // The draft's folder now holds this run: no form, current or in the history, fills it again.
      form: { ...s.form, draftId: null },
      formHistory: draftId ? forgetDraft(s.formHistory, draftId) : s.formHistory,
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

  startRobustness: async (runId, change) => {
    const seq = projectSeq;
    const out = await get().guard(() => api.robustness(runId, change));
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
  setForm: (patch, options = {}) => {
    const st = get();
    const before = st.form;
    const form = { ...before, ...patch };
    if (sameData(form, before)) return;
    const bos = analysisContext(st).options.prepend_bos;
    let history = st.formHistory;
    if (options.record !== false) {
      const at = Date.now();
      const fields = editFields(patch);
      history = continuesEdit(lastEdit, fields, at) && history.past.length ? { past: history.past, future: [] } : pushHistory(history, before);
      lastEdit = { fields, at };
    }
    set({ form, formHistory: history, analysisSource: "form" });
    // Staged positions count tokens as they were tokenized; BOS moves every token.
    if (form.prependBos !== before.prependBos) set({ stagedSites: [] });
    analysisChanged(bos);
  },

  replaceForm: (form, options) => {
    const bos = analysisContext(get()).options.prepend_bos;
    const changed = swapForm(form, options);
    if (changed) analysisChanged(bos);
    return changed;
  },

  undoForm: () => stepForm(undoHistory),
  redoForm: () => stepForm(redoHistory),

  markFormSaved: (draftId) => {
    savedForm = { ...get().form, draftId };
    set((st) => ({ form: { ...st.form, draftId } }));
  },

  editRun: (spec) => {
    const bos = analysisContext(get()).options.prepend_bos;
    set({ view: "experiment", analysisSource: "form" });
    swapForm(formFromSpec(spec));
    analysisChanged(bos);
  },

  prefillExperiment: (kind, sel) => {
    const bos = analysisContext(get()).options.prepend_bos;
    const form = get().form;
    // Use the residual site of the active run when the selection is a residual cell.
    // A residual cell names its site; otherwise use the residual site the active run measured.
    const id = get().activeRunId;
    const { sites } = id ? runView(get().runDetails[id], get().live[id]) : { sites: [] as SiteBase[] };
    const measured = sitesOnComponent(sites, sel).find((x) => isResidKind(x.kind))?.kind;
    const residKind = sel.kind ?? (measured && isResidKind(measured) ? measured : "resid_pre");
    const site = siteFromSelection(sel, residKind);
    set({ selection: sel, view: "experiment", analysisSource: "form" });
    // A new experiment at this site; the notes stay, since they are the user's own words.
    swapForm({
      ...form,
      kind,
      baseline: kind === "ablation" ? form.baseline : null,
      scope: { kind: "sites", sites: [site] },
      nameEdited: false,
      draftId: null,
      modelRef: null,
      savedDataset: null,
    });
    analysisChanged(bos);
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
            // The spec says so too: the checkbox shows it, and the run records it. It follows the
            // model, so a form that was saved stays saved.
            if (sameForm(get().form, savedForm)) savedForm = { ...savedForm, prependBos: false };
            set((st) => ({ form: { ...st.form, prependBos: false } }));
            get().notify(`${now.id} has no beginning-of-sequence token, so prompts now start without one.`, "info", { persist: true });
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
          if (job.kind === "load_sae" && job.status === "finished") get().notify("SAE loaded. Measure its fit on these prompts before trusting its features.", "info", { persist: true });
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
        if (status === "cancelled" && !data.replay) get().notify("Run cancelled. The spec and any dataset snapshot remain; partial results were discarded.", "info", { persist: true });
        break;
      }
      case "sae": {
        const { type: _type, ...rest } = event;
        set({ sae: rest as unknown as SAEStatus });
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

// Keep an unsaved form across reloads: in this browser only, per project folder.
useStore.subscribe((state, prev) => {
  if (state.form === prev.form || !state.project || state.project.session_id !== prev.project?.session_id) return;
  if (sameForm(state.form, savedForm) || sameForm(state.form, DEFAULT_FORM)) clearStoredForm(state.project.path);
  else writeStoredForm(state.project.path, state.form);
});

/** Put back the form left unsaved when the project was last open, offering to undo that. */
function restoreForm(stored: StoredForm): void {
  const st = useStore.getState();
  // Edits made while the project was opening are newer than the stored form: they win.
  if (!sameForm(st.form, savedForm)) return;
  let form = stored.form;
  // A draft that has run since, or was removed, has no folder for this form to fill.
  if (form.draftId && !st.runs.some((r) => r.id === form.draftId && r.status === "draft")) form = { ...form, draftId: null };
  if (sameForm(form, st.form)) return;
  st.replaceForm(form, { notice: FORM_RESTORED });
}

/**
 * Replace the form as one undo step, with a notice offering to undo it. The caller handles what
 * the change means for the analysis context (see analysisChanged).
 */
function swapForm(next: FormState, options: ReplaceFormOptions = {}): boolean {
  const st = useStore.getState();
  const before = st.form;
  if (sameData(next, before)) return false;
  lastEdit = null;
  useStore.setState({ form: next, formHistory: pushHistory(st.formHistory, before) });
  const text = options.notice === undefined ? FORM_REPLACED : options.notice;
  if (text !== null) {
    st.notify(text, "info", {
      key: FORM_NOTICE,
      action: {
        label: "Undo",
        run: () => {
          useStore.getState().replaceForm(before, { notice: null });
          options.onUndo?.();
        },
      },
    });
  }
  return true;
}

/** Undo or redo one step of the form's history. */
function stepForm(step: typeof undoHistory): void {
  const st = useStore.getState();
  const next = step(st.formHistory, st.form);
  lastEdit = null;
  if (!next) {
    // Only steps that change nothing were left: forget them, so Undo and Redo show as unavailable.
    if (step === undoHistory && st.formHistory.past.length) useStore.setState({ formHistory: { ...st.formHistory, past: [] } });
    if (step === redoHistory && st.formHistory.future.length) useStore.setState({ formHistory: { ...st.formHistory, future: [] } });
    return;
  }
  const bos = analysisContext(st).options.prepend_bos;
  useStore.setState({ form: next.form, formHistory: next.history });
  analysisChanged(bos);
}

/**
 * After the view (or the map's overlay) changes: store where prompts are now read from, and when
 * that is the active run, read its prompts, from its dataset, as it did. Changing BOS moves every
 * token, so a chosen token no longer means the same one.
 */
function followAnalysisSource(bosBefore: boolean): void {
  const st = useStore.getState();
  const source = analysisSourceFor(st);
  if (source !== st.analysisSource) useStore.setState({ analysisSource: source });
  const spec = source === "run" && st.activeRunId ? st.runDetails[st.activeRunId]?.spec : undefined;
  if (spec && spec.dataset.path !== st.datasetPath) {
    void st.selectDataset(spec.dataset.path, "run");
    return;
  }
  analysisChanged(bosBefore);
}

/** After anything that can change the analysis context: tokens move when BOS changes, so the
 * prompts are read again; the prompt index stays within the prompts used. */
function analysisChanged(bosBefore: boolean): void {
  const st = useStore.getState();
  if (analysisContext(st).options.prepend_bos !== bosBefore) {
    useStore.setState({ tokenPosition: null });
    void st.reloadDataset();
  }
  st.setPromptIndex(st.promptIndex);
}

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
