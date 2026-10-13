// Application state. Server state arrives over the API and the WebSocket event stream.

import { create } from "zustand";

import { api, ApiError, setProjectSession } from "../api/client";
import { analysisContext, analysisSourceFor, tokenizationKey } from "../lib/analysis";
import { usesFeatures } from "../lib/buildSpec";
import type {
  BaselineReport,
  DatasetDetail,
  Environment,
  Job,
  SAEStatus,
  Layout,
  ModelShape,
  ModelStatus,
  ProjectInfo,
  PartialRun,
  QueuedRun,
  QueueState,
  QueueStatus,
  RobustnessChange,
  RunAnnotationsChange,
  RunDetail,
  MetricKind,
  RunListing,
  ScopeSpec,
  ServerState,
  SiteBase,
  SiteResult,
  Spec,
  ThemeSetting,
  SiteSpec,
  NoteInput,
  SweepRequest,
  UpdateStatus,
} from "../api/types";
import type { ResolvedTheme } from "../lib/color";
import { count, plural } from "../lib/format";
import { METRICS } from "../lib/metrics";
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
import type { Route } from "../lib/route";
import { fitSelection, isResidKind, sitesOnComponent, type Selection } from "../lib/sites";
import { hasEdits, type PromptEdits } from "../lib/promptEdits";

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
  | "sweep"
  | "spec"
  | "text";

export type Workspace = "explore" | "experiment" | "evidence";

/** Each workspace's tabs, in order: the one place their names are set, so the tabs, the command
 * palette and the number keys agree. */
export const WORKSPACE_VIEWS: { id: Workspace; views: { id: View; label: string }[] }[] = [
  {
    id: "explore",
    views: [
      { id: "explore", label: "Model" },
      { id: "attention", label: "Attention" },
      { id: "heads", label: "Head comparison" },
      { id: "predictions", label: "Layer predictions" },
      { id: "features", label: "Features" },
      { id: "text", label: "Your text" },
    ],
  },
  {
    id: "experiment",
    views: [
      { id: "prompts", label: "Prompts" },
      { id: "baseline", label: "Baseline" },
      { id: "experiment", label: "Configure" },
      { id: "spec", label: "Spec" },
    ],
  },
  {
    id: "evidence",
    views: [
      { id: "results", label: "Results" },
      { id: "compare", label: "Compare runs" },
      { id: "sweep", label: "Sweeps" },
      { id: "notes", label: "Research notes" },
    ],
  },
];

export const VIEWS: { id: View; label: string }[] = WORKSPACE_VIEWS.flatMap((w) => w.views);

/** A sweep over sets of sites, each intervened on at once. */
export type SiteSetsScope = Extract<ScopeSpec, { kind: "site_sets" }>;
export function workspaceFor(view: View): Workspace {
  if (["explore", "attention", "heads", "predictions", "features", "text"].includes(view)) return "explore";
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

/** How often the history is read again while a run runs in another process, whose progress
 * isn't streamed here: often enough to see it end, rarely enough to cost nothing. */
export const ELSEWHERE_POLL_MS = 4000;

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
  /** How the walkthrough of the example ended, from the person's settings: until it has, the
   * first screens offer it. */
  walkthrough: "finished" | "skipped" | null;
  projectsParent: string;
  themeSetting: ThemeSetting;
  theme: ResolvedTheme;
  /** Whether single keys (digits, [ and ], letters, ?) act as shortcuts. */
  singleKeys: boolean;

  project: ProjectInfo | null;
  model: ModelStatus;
  /** The loaded sparse autoencoder; it belongs to the loaded model. */
  sae: SAEStatus;
  job: Job | null;
  /** Runs waiting for the job in progress to end, first in line first (the server's queue). */
  queue: QueuedRun[];
  /** How the queue stands (it is kept in the project, so it outlives the server): waiting for
   * the job in progress, running, paused (as it comes back with its project) or idle. */
  queueState: QueueState;
  /** Who runs the queue when another process does, such as `logogram queue run`. */
  queueRunner: string | null;
  /** Why the project's saved queue can't be read, and how to fix it. */
  queueError: string | null;
  runs: RunListing[];
  /** The versions and devices a rerun here would use, to compare with a run's manifest. */
  environment: Environment | null;

  view: View;
  datasetPath: string | null;
  dataset: DatasetDetail | null;
  promptIndex: number;
  /** Unsaved edits to datasets' prompt pairs, by dataset path. A dataset file never changes:
   * edits are saved as a new dataset (lib/promptEdits). */
  datasetEdits: Record<string, PromptEdits>;
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
  /** The sweep the sweep view shows: one spec run across several values of one choice. */
  sweepId: string | null;
  /** True while a project is being opened: its several steps are one place, not many. */
  entering: boolean;
  /** Where the URL the page was opened with asks to be: the project opens there. */
  pendingRoute: Route | null;
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
  /** The run whose spec the sweep dialog sweeps, while it is open. */
  sweepDialogRun: string | null;
  /** Asking whether to cancel the running job (it keeps what finished). */
  cancelConfirmOpen: boolean;
  /** The sheet of keyboard shortcuts (? or the command palette). */
  shortcutsOpen: boolean;
  /** A dialog from the Help menu or the palette: the guide, the problem report, or About. */
  helpDialog: "guide" | "report" | "about" | null;
  /** The last error the app showed (a notice, a view that stopped, or a failed start), for the
   * problem report. */
  lastError: string | null;
  notices: Notice[];

  // actions
  boot: () => Promise<void>;
  applyServerState: (s: ServerState) => void;
  setTheme: (t: ThemeSetting) => void;
  resolveTheme: () => void;
  setSingleKeys: (on: boolean) => void;
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
  /** A dataset file renamed (`to` its new path) or deleted (`to` null), here or in another tab:
   * the prompts chosen and their unsaved edits follow it. */
  followDataset: (from: string, to: string | null) => void;
  reloadDataset: () => Promise<void>;
  setPromptIndex: (i: number) => void;
  /** Keep a dataset's unsaved edits, or drop them (null, or edits that change nothing). */
  setDatasetEdits: (path: string, edits: PromptEdits | null) => void;
  applyJob: (job: Job) => void;
  /** Read the server's environment, once, to compare runs' manifests with it. */
  loadEnvironment: () => void;
  checkUpdates: () => Promise<void>;
  setUpdateConsent: (allowed: boolean) => Promise<void>;
  fitSelection: () => void;

  /** Show a run. A run of another model (or revision) drops the selection, as a site of one model
   * is no site of another; `at` selects a site once the run is open (a sweep's value at a site). */
  openRun: (id: string | null, view?: View, at?: Selection) => Promise<void>;
  /** Load a saved, unrun experiment into the form. Quiet when opening a project, where the form
   * was empty: no undo step and no notice. */
  openDraft: (id: string, options?: { quiet?: boolean }) => Promise<void>;
  loadRun: (id: string, force?: boolean) => Promise<RunDetail | undefined>;
  select: (sel: Selection | null) => void;
  setTokenPosition: (position: number | null) => void;
  stageSelection: (selection: Selection) => void;
  configureStaged: () => void;
  /** Put sets of sites into the form, intervened on by patching or ablation: built from the
   * staged sites (back in the tray on Undo) or from a finished run, whose spec the form takes. */
  configureSets: (scope: SiteSetsScope, options?: { staged?: boolean; runId?: string }) => void;
  pinHead: (selection: Selection) => void;
  prepareNote: (sites: SiteSpec[]) => void;
  startRun: (spec: Spec) => Promise<void>;
  /** Run the spec when the job in progress ends: it waits in the queue until then. */
  queueRun: (spec: Spec) => Promise<void>;
  /** Take a run out of the queue before it starts. */
  removeQueuedRun: (id: string) => Promise<void>;
  /** The queue and how it stands, from the server (or from an event). */
  applyQueue: (status: QueueStatus) => void;
  /** Read the queue again as the project keeps it: another process may have changed it. */
  refreshQueue: () => Promise<void>;
  /** Run the queue here (a paused one, say): its next run starts when nothing is running. */
  startQueue: () => Promise<void>;
  /** Start no more of the queue's runs; the run in progress goes on. */
  pauseQueue: () => Promise<void>;
  /** Take every run out of the queue. Returns whether it was cleared. */
  clearQueue: () => Promise<boolean>;
  startRobustness: (runId: string, change: RobustnessChange) => Promise<void>;
  /** Patch, for real, the strongest sites an attribution patching run estimated. */
  startVerification: (runId: string, top: number) => Promise<void>;
  /** Show a sweep in the sweep view. */
  openSweep: (id: string | null) => void;
  /** Queue a sweep's runs, then show the sweep. Returns whether it was queued. */
  startSweep: (body: SweepRequest) => Promise<boolean>;
  /** Rename, star or archive a run: what is said about it, kept beside its spec, which never
   * changes. Returns whether it was saved. */
  annotateRun: (id: string, change: RunAnnotationsChange) => Promise<boolean>;
  /** Delete a run or a draft, folder and all, and forget it wherever this tab shows it. Returns
   * whether it was deleted. */
  deleteRun: (id: string) => Promise<boolean>;
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
  /** Say that a new dataset was saved; when another metric reads its answers than the form's,
   * offer to switch to it, as a step of the form's history. */
  datasetSaved: (path: string, metric?: MetricKind) => void;

  handleEvent: (event: Record<string, unknown> & { type: string }) => void;
}

let noticeId = 0;
// The last edit made in the form, so that typing makes one undo step rather than one per key.
let lastEdit: LastEdit | null = null;
const FORM_NOTICE = "form-replaced";
// The notice that offers to start a queue that came back paused.
const QUEUE_NOTICE = "queue-paused";
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
let environmentRequest: Promise<void> | null = null;
// The project whose queue the store shows (a queue's status names its project's session).
let queueSession: string | null = null;

/** Who runs the queue, as the server names it, for the middle of a sentence: "logogram queue run
 * (process 1234)" or "the Logogram app (process 1234)". */
export function runnerName(runner: string): string {
  return runner.replace(/`/g, "");
}

/** Whether a run started now waits in the queue: a job is running here, the queue is about to
 * start runs, or another process runs the queue. A paused queue doesn't hold a new run back. */
export function runWouldQueue(st: Pick<Store, "job" | "queue" | "queueState" | "queueRunner">): boolean {
  return st.job?.status === "running" || st.queueRunner !== null || (st.queue.length > 0 && st.queueState !== "paused");
}

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
  walkthrough: null,
  projectsParent: "",
  // Until the saved appearance arrives, follow the system, as the page does (tokens.css).
  themeSetting: "system",
  theme: systemTheme(),
  singleKeys: true,

  project: null,
  model: { state: "none" },
  sae: { state: "none" },
  job: null,
  queue: [],
  queueState: "idle",
  queueRunner: null,
  queueError: null,
  runs: [],
  environment: null,

  view: "prompts",
  datasetPath: null,
  dataset: null,
  promptIndex: 0,
  datasetEdits: {},
  activeRunId: null,
  runDetails: {},
  live: {},
  selection: null,
  mapMetric: "effect",
  scaleMode: "auto",
  cellValues: false,
  update: null,
  compareIds: [null, null],
  sweepId: null,
  entering: false,
  pendingRoute: null,
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
  sweepDialogRun: null,
  cancelConfirmOpen: false,
  shortcutsOpen: false,
  helpDialog: null,
  lastError: null,
  notices: [],

  boot: async () => {
    try {
      const state = await api.state();
      get().applyServerState(state);
      // The URL's place only applies to the project open at boot.
      if (state.first_run) set({ screen: "system", pendingRoute: null });
      else if (state.project) await get().enterProject(state.project);
      else set({ screen: "projects", pendingRoute: null });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      set({ screen: "error", bootError: message, lastError: message });
    }
  },

  applyServerState: (s) => {
    set({
      version: s.version,
      firstRun: s.first_run,
      projectsParent: s.projects_parent,
      themeSetting: s.theme,
      singleKeys: s.single_key_shortcuts ?? true,
      walkthrough: s.walkthrough ?? null,
      update: s.update ?? null,
      sae: s.sae ?? { state: "none" },
      queue: s.queue ?? [],
    });
    if (s.queue_status) get().applyQueue(s.queue_status);
    get().handleEvent({ type: "model", ...s.model });
    if (s.job) get().applyJob(s.job);
    else set({ job: null });
    get().resolveTheme();
    if (get().screen !== "loading" && s.project?.session_id !== get().project?.session_id) {
      if (s.project) void get().enterProject(s.project);
      else get().leaveProject();
    }
  },

  loadEnvironment: () => {
    // Static while the server runs: read once, when a view first compares a run with it.
    if (get().environment || environmentRequest) return;
    environmentRequest = api.environment().then(
      (environment) => set({ environment }),
      () => {
        environmentRequest = null; // try again next time
      },
    );
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

  setSingleKeys: (on) => {
    set({ singleKeys: on });
    void api.settings({ single_key_shortcuts: on }).catch(() => undefined);
  },

  resolveTheme: () => {
    const setting = get().themeSetting;
    const theme = setting === "system" ? systemTheme() : setting;
    document.documentElement.dataset.theme = theme;
    set({ theme });
  },

  goto: (screen) => set({ screen }),
  setView: (view) => {
    const bos = tokenizationKey(analysisContext(get()).options);
    set({ view });
    followAnalysisSource(bos);
  },
  setMapOverlay: (mapOverlay) => {
    const bos = tokenizationKey(analysisContext(get()).options);
    set({ mapOverlay });
    followAnalysisSource(bos);
  },

  notify: (text, tone = "info", options = {}) => {
    const id = ++noticeId;
    const notice: Notice = { id, tone, text, ...options };
    set((s) => {
      const others = notice.key ? s.notices.filter((n) => n.key !== notice.key) : s.notices;
      return { notices: [...others.slice(-3), notice], ...(tone === "error" ? { lastError: text } : {}) };
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
      queue: get().queue.filter((q) => q.project_session === project.session_id),
      ...(queueSession === project.session_id ? {} : { queueState: "idle" as const, queueRunner: null, queueError: null }),
      datasetPath: null,
      dataset: null,
      promptIndex: 0,
      datasetEdits: {},
      analysisSource: "form",
      view: "prompts",
      inspectorFocus: null,
      robustnessDialogOpen: false,
      sweepDialogRun: null,
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
      sweepId: null,
      entering: true,
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
    await get().refreshQueue();
    if (seq !== projectSeq) return;
    const runs = get().runs;
    // The page's URL (after a reload, or a kept link) names the view and run to open, where they
    // still exist. Otherwise a run in progress comes first: its map fills in as it streams.
    const route = get().pendingRoute;
    const wanted = route?.screen === "workbench" ? route : null;
    const runOf = (id: string | undefined) => (id ? runs.find((r) => r.id === id) : undefined);
    const asked = wanted && runOf(wanted.compare?.[0] ?? wanted.run);
    // This server's run, which streams here; one in another process doesn't.
    const running = runs.find((r) => r.status === "running" && !r.elsewhere);
    const finished = runs.find((r) => r.status === "finished");
    const draft = runs.find((r) => r.status === "draft");
    if (running) set((s) => ({ live: { ...s.live, [running.id]: s.live[running.id] ?? emptyLive(running.id) } }));
    if (wanted?.compare && runOf(wanted.compare[0]) && runOf(wanted.compare[1])) set({ compareIds: wanted.compare });
    if (wanted?.sweep) set({ sweepId: wanted.sweep });
    // A place chosen while the project was opening (a workspace clicked as the page loads) wins
    // over the one restored here.
    const chosen = get().view !== "prompts" ? get().view : undefined;
    if (asked && asked.status !== "draft") await get().openRun(asked.id, chosen ?? wanted.view);
    else if (asked) {
      await get().openDraft(asked.id, { quiet: true });
      set({ view: chosen ?? wanted.view });
    } else if (running) await get().openRun(running.id, chosen ?? wanted?.view ?? "results");
    else if (finished) await get().openRun(finished.id, chosen ?? wanted?.view ?? "explore");
    else if (draft) {
      await get().openDraft(draft.id, { quiet: true });
      const view = chosen ?? wanted?.view;
      if (view) set({ view });
    } else set({ view: chosen ?? wanted?.view ?? (project.datasets.length ? "baseline" : "prompts") });
    if (stored && seq === projectSeq) restoreForm(stored);
    if (seq === projectSeq) set({ entering: false, pendingRoute: null, ...(route?.screen === "projects" ? { screen: "projects" as const } : {}) });
    // Runs queued before the project last closed (or the server stopped) wait, paused: nothing
    // heavy starts on its own.
    const waiting = get().queue.length;
    if (seq === projectSeq && get().queueState === "paused" && waiting > 0) {
      get().notify(
        `${waiting === 1 ? "A run waits" : `${count(waiting)} runs wait`} in the queue from before. It is paused: nothing starts until you start it. The history lists the queue, where you can clear it.`,
        "info",
        { persist: true, key: QUEUE_NOTICE, action: { label: "Start the queue", run: () => void get().startQueue() } },
      );
    }
  },

  leaveProject: () => {
    projectSeq += 1;
    lastEdit = null;
    savedForm = DEFAULT_FORM; // the stored copy stays, for when the project opens again
    openSeq += 1;
    datasetSeq += 1;
    inflight.clear();
    setProjectSession(null);
    set({ project: null, screen: "projects", datasetPath: null, dataset: null, promptIndex: 0, datasetEdits: {},
      tokenPosition: null, stagedSites: [], headPins: [], noteEditor: null, researchVersion: 0,
      job: get().job?.project_session ? null : get().job, queue: [], queueState: "idle", queueRunner: null, queueError: null,
      runs: [], runDetails: {}, activeRunId: null, pendingRunId: null, live: {}, selection: null,
      compareIds: [null, null], sweepId: null, entering: false, flags: {}, form: DEFAULT_FORM, formHistory: EMPTY_HISTORY, baselines: {}, specSource: "run",
      analysisSource: "form", view: "prompts", inspectorFocus: null, robustnessDialogOpen: false, sweepDialogRun: null,
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
    const before = get().runs;
    set({ runs });
    void get().refreshQueue();
    // A run in another process has ended: say so, and read it again where it is shown.
    for (const run of runs) {
      const was = before.find((r) => r.id === run.id);
      if (!was?.elsewhere || was.status !== "running" || run.status === "running") continue;
      const ended = run.status === "finished" ? "finished" : run.status === "cancelled" ? "was cancelled" : "failed";
      get().notify(`“${run.name}”, running in another process, ${ended}.`, "info", {
        action: { label: run.status === "finished" ? "Show results" : "Show the run", run: () => void get().openRun(run.id, "results") },
      });
      if (get().runDetails[run.id] || get().activeRunId === run.id) void get().loadRun(run.id, true).then(() => get().fitSelection());
    }
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
    const { prepend_bos, chat_template } = analysisContext(get()).options;
    const detail = await get().guard(() => api.dataset(path, prepend_bos, chat_template ?? null));
    if (detail && seq === datasetSeq && project === projectSeq) set({ dataset: detail });
  },

  followDataset: (from, to) => {
    const st = get();
    // Unsaved edits belong to the file's prompts, which a rename keeps and a deletion takes away.
    const edits = st.datasetEdits[from];
    if (edits) {
      const { [from]: _gone, ...rest } = st.datasetEdits;
      set({ datasetEdits: to ? { ...rest, [to]: edits } : rest });
    }
    if (st.datasetPath !== from) return;
    const next = to ?? st.project?.datasets.find((d) => !d.error && d.path !== from)?.path ?? null;
    void st.selectDataset(next, st.analysisSource);
  },

  reloadDataset: async () => {
    // Token issues and lengths depend on the loaded model's tokenizer: refetch when it changes.
    const path = get().datasetPath;
    if (!path || !get().project) return;
    const seq = ++datasetSeq;
    const project = projectSeq;
    const { prepend_bos, chat_template } = analysisContext(get()).options;
    const detail = await api.dataset(path, prepend_bos, chat_template ?? null).catch(() => null);
    if (detail && seq === datasetSeq && project === projectSeq) set({ dataset: detail });
  },

  setPromptIndex: (i) => {
    const n = analysisContext(get()).n;
    if (n === 0) return;
    const next = Math.max(0, Math.min(n - 1, i));
    set({ promptIndex: next, ...(next !== get().promptIndex ? { tokenPosition: null } : {}) });
  },

  setDatasetEdits: (path, edits) => {
    const { [path]: _dropped, ...rest } = get().datasetEdits;
    set({ datasetEdits: edits && hasEdits(edits) ? { ...rest, [path]: edits } : rest });
  },

  openRun: async (id, view, at) => {
    // A run whose spec can't be read has nothing to show: say why, as the history does.
    const listed = id ? get().runs.find((r) => r.id === id) : undefined;
    if (listed?.damaged) {
      get().notify(listed.error ?? "This run's spec can't be read.", "error");
      return;
    }
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
    set({ analysisSource: "run", promptIndex: 0, tokenPosition: null, ...(changed ? { stagedSites: [], headPins: [], selection: null } : {}), ...(at ? { selection: at } : {}) });
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
    const bos = tokenizationKey(analysisContext(get()).options);
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
    // A set of sites or an edge has no single place in the model to stage.
    if (selection.part === "set" || selection.part === "edge") return;
    const site = siteFromSelection(selection, selection.kind ?? "resid_pre");
    if (get().tokenPosition !== null) site.position = { kind: "index", index: get().tokenPosition as number };
    const staged = get().stagedSites;
    if (staged.some((s) => JSON.stringify(s) === JSON.stringify(site))) return;
    set({ stagedSites: [...staged, site] });
  },
  configureSets: (scope, options = {}) => {
    const base = formShown(get(), options.runId);
    // Sets are intervened on by patching or ablation; another method becomes patching, in the
    // direction the form has (an estimate's direction, from an attribution patching run).
    const kind = base.kind === "ablation" ? "ablation" : "activation_patching";
    configureForm({ ...base, kind, scope }, options.staged ?? false);
  },
  configureStaged: () => {
    const st = get();
    if (!st.stagedSites.length) return;
    configureForm({ ...formShown(st), scope: { kind: "sites", sites: st.stagedSites } }, true);
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
    const out = await get().guard(() => api.startRun(spec, ran.draftId));
    starting = false;
    if (!out || seq !== projectSeq) return;
    await showStartedRun(ran, out.run_id, out.job);
  },

  queueRun: async (spec) => {
    if (starting) return; // a second press while the first request is on its way
    starting = true;
    const seq = projectSeq;
    const ran = get().form;
    const out = await get().guard(() => api.queueRun(spec, ran.draftId));
    starting = false;
    if (!out || seq !== projectSeq) return;
    // Nothing was running or waiting any more: it started at once, as Run would have.
    if (out.run_id && out.job) return showStartedRun(ran, out.run_id, out.job);
    formSent(ran);
    // The queue event shows it waiting; the notice says what it waits for. A paused queue, or one
    // another process runs, says so.
    await get().refreshQueue();
    if (seq !== projectSeq) return;
    const { queueState, queueRunner } = get();
    if (queueState === "paused") {
      get().notify(`Queued “${spec.name}”. The queue is paused: nothing in it starts until you start it.`, "info", {
        persist: true,
        key: QUEUE_NOTICE,
        action: { label: "Start the queue", run: () => void get().startQueue() },
      });
      return;
    }
    if (queueRunner) {
      get().notify(`Queued “${spec.name}”. The queue is being run by ${runnerName(queueRunner)}: it starts when its turn comes.`);
      return;
    }
    const place = out.queue.findIndex((q) => q.id === out.queued?.id);
    const job = get().job;
    const ahead = place > 0 ? out.queue[place - 1].title : place === 0 && job?.status === "running" ? job.title : null;
    get().notify(ahead ? `Queued “${spec.name}”. It starts after “${ahead}”.` : `Queued “${spec.name}”.`);
  },

  removeQueuedRun: async (id) => {
    const seq = projectSeq;
    const run = get().queue.find((q) => q.id === id);
    const left = await get().guard(() => api.removeQueuedRun(id));
    if (!left || seq !== projectSeq) return;
    // The queue event says so too; take it out now, so it can't be removed twice.
    set((s) => ({ queue: s.queue.filter((q) => q.id !== id) }));
    get().notify(run ? `Removed “${run.title}” from the queue.` : "Removed the run from the queue.");
  },

  applyQueue: (status) => {
    queueSession = status.project_session;
    set({ queue: status.runs ?? [], queueState: status.state ?? "idle", queueRunner: status.runner ?? null, queueError: status.error ?? null });
  },

  refreshQueue: async () => {
    if (!get().project) return;
    const seq = projectSeq;
    // Quietly: the history reads it again with the runs, and the server's events say the rest.
    const status = await api.queueStatus().catch(() => null);
    if (status && seq === projectSeq && status.project_session === get().project?.session_id) get().applyQueue(status);
  },

  startQueue: async () => {
    const seq = projectSeq;
    const status = await get().guard(() => api.startQueue());
    if (!status || seq !== projectSeq) return;
    get().applyQueue(status);
    get().notify(status.state === "running" ? "Started the queue. Its runs follow one another." : "Started the queue. Its next run starts when the job in progress ends.");
  },

  pauseQueue: async () => {
    const seq = projectSeq;
    const status = await get().guard(() => api.pauseQueue());
    if (!status || seq !== projectSeq) return;
    get().applyQueue(status);
    get().notify("Paused the queue. The run in progress goes on; nothing starts after it until you start the queue.");
  },

  clearQueue: async () => {
    const seq = projectSeq;
    const n = get().queue.length;
    const status = await get().guard(() => api.clearQueue());
    if (!status || seq !== projectSeq) return false;
    get().applyQueue(status);
    get().notify(n === 1 ? "Cleared the queue: its run won't run." : `Cleared the queue: its ${count(n)} runs won't run.`);
    return true;
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

  openSweep: (id) => {
    set({ sweepId: id });
    get().setView("sweep");
  },

  startSweep: async (body) => {
    const seq = projectSeq;
    const out = await get().guard(() => api.startSweep(body));
    if (!out || seq !== projectSeq) return false;
    if (out.job) get().applyJob(out.job);
    // The queue event says so too; the sweep view counts the values still waiting from it.
    set({ queue: out.queue });
    const made = out.plan.runs.filter((r) => !r.original).length;
    const kept = out.plan.runs.find((r) => r.original);
    const stands = kept ? `; this run stands in for ${out.plan.label} ${kept.label}` : "";
    const queued = `Queued ${plural(made, "run")} of the sweep over ${out.plan.plural}${stands}.`;
    // A paused queue starts nothing: what waits in it waits until it is started (the first run
    // starts at once when nothing is running).
    await get().refreshQueue();
    if (seq !== projectSeq) return false;
    const waiting = made - (out.job ? 1 : 0);
    if (get().queueState === "paused" && waiting > 0) {
      const which = out.job ? `The first has started; the other${waiting === 1 ? " waits" : ` ${count(waiting)} wait`}` : made === 1 ? "It waits" : "They wait";
      get().notify(`${queued} ${which} in the queue, which is paused: nothing in it starts until you start it.`, "info", {
        persist: true,
        key: QUEUE_NOTICE,
        action: { label: "Start the queue", run: () => void get().startQueue() },
      });
    } else {
      get().notify(`${queued} They run one after another.`);
    }
    await get().refreshRuns();
    if (seq === projectSeq) get().openSweep(out.sweep);
    return true;
  },

  annotateRun: async (id, change) => {
    const seq = projectSeq;
    const listing = await get().guard(() => api.annotateRun(id, change));
    if (!listing || seq !== projectSeq) return false;
    applyAnnotations(listing);
    return true;
  },

  deleteRun: async (id) => {
    const seq = projectSeq;
    const run = get().runs.find((r) => r.id === id);
    const out = await get().guard(() => api.deleteRun(id));
    if (!out || seq !== projectSeq) return false;
    forgetRun(id);
    get().notify(`Deleted “${run?.title || run?.name || id}”.`);
    void get().refreshRuns();
    return true;
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
    const bos = tokenizationKey(analysisContext(st).options);
    let history = st.formHistory;
    if (options.record !== false) {
      const at = Date.now();
      const fields = editFields(patch);
      history = continuesEdit(lastEdit, fields, at) && history.past.length ? { past: history.past, future: [] } : pushHistory(history, before);
      lastEdit = { fields, at };
    }
    set({ form, formHistory: history, analysisSource: "form" });
    // Staged positions count tokens as they were tokenized; BOS or a chat template moves every
    // token.
    if (form.prependBos !== before.prependBos || !sameData(form.chatTemplate, before.chatTemplate)) set({ stagedSites: [] });
    analysisChanged(bos);
  },

  replaceForm: (form, options) => {
    const bos = tokenizationKey(analysisContext(get()).options);
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
    const bos = tokenizationKey(analysisContext(get()).options);
    set({ view: "experiment", analysisSource: "form" });
    swapForm(formFromSpec(spec));
    analysisChanged(bos);
  },

  prefillExperiment: (kind, sel) => {
    if (sel.part === "set" || sel.part === "edge") return; // patched as a whole in a run of its own
    const bos = tokenizationKey(analysisContext(get()).options);
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

  datasetSaved: (path, metric) => {
    const form = get().form;
    if (!metric || metric === form.metric || !METRICS[metric]) {
      get().notify(`Saved ${path}. Check the baseline next.`);
      return;
    }
    const words = METRICS[metric].label;
    get().notify(`Saved ${path}. Its answers are read with the ${words}; the experiment form uses the ${METRICS[form.metric].label}.`, "info", {
      key: "dataset-metric",
      action: {
        label: `Use the ${words}`,
        run: () => {
          const now = get();
          now.replaceForm({ ...now.form, metric }, { notice: `The experiment form now uses the ${words}.` });
        },
      },
    });
  },

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
          if (job.kind === "add_model" && job.status === "finished") get().notify(`${job.title.replace(/^Add /, "Added ")}. Choose it in Load a model.`);
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
          if (status === "finished" && from && (from.kind === "robustness" || from.kind === "verification")) {
            const cmp = await api.compare(from.run, runId).catch(() => null);
            if (cmp) set((s) => ({ flags: { ...s.flags, [from.run]: { against: runId, sites: cmp.flagged } } }));
          }
        });
        if (status === "cancelled" && !data.replay) {
          const partial = data.partial as PartialRun | null | undefined;
          // SAE features are reported only when the run ends: what finished before isn't kept.
          const scope = get().runs.find((r) => r.id === runId)?.scope ?? get().runDetails[runId]?.spec.scope;
          get().notify(
            partial
              ? `Run cancelled. ${partial.note} Its spec and any dataset snapshot remain.`
              : scope && usesFeatures(scope)
                ? "Run cancelled. A run of SAE features reports nothing before it ends, so it kept no results. Its spec and any dataset snapshot remain."
                : "Run cancelled before anything finished. Its spec and any dataset snapshot remain.",
            "info",
            { persist: true },
          );
        }
        break;
      }
      case "queue":
        get().applyQueue({
          project_session: (data.project_session as string | null | undefined) ?? null,
          runs: (data.runs as QueuedRun[] | undefined) ?? [],
          state: (data.state as QueueState | undefined) ?? "idle",
          runner: (data.runner as string | null | undefined) ?? null,
          error: (data.error as string | null | undefined) ?? null,
        });
        break;
      case "runs.changed":
        // A run renamed, starred, archived or deleted, here or in another tab.
        if (data.deleted) forgetRun(String(data.run_id));
        void get().refreshRuns();
        break;
      case "project.changed": {
        // The project renamed, or a dataset renamed or deleted, here or in another tab.
        const change = data.dataset as { path: string; to: string | null } | null | undefined;
        void get()
          .refreshProject()
          .then(() => {
            if (change) get().followDataset(change.path, change.to);
          });
        break;
      }
      case "queue.skipped":
        get().notify(`“${String(data.title)}” couldn't start and was taken out of the queue. ${String(data.error)}`, "error");
        break;
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
        // A tab that has already left the project (it closed or deleted it) has nothing to do.
        if ((data.session_id ?? null) !== (get().project?.session_id ?? null)) {
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

// While a run runs in another process (logogram run, a notebook), nothing streams its end here:
// read the history again on a slow timer until no such run is left.
let elsewherePoll: ReturnType<typeof setInterval> | null = null;
useStore.subscribe((state, prev) => {
  if (state.runs === prev.runs && state.queueRunner === prev.queueRunner) return;
  const watching = state.runs.some((r) => r.status === "running" && r.elsewhere) || state.queueRunner !== null;
  if (watching && elsewherePoll === null) {
    elsewherePoll = globalThis.setInterval(() => void useStore.getState().refreshRuns(), ELSEWHERE_POLL_MS);
  } else if (!watching && elsewherePoll !== null) {
    globalThis.clearInterval(elsewherePoll);
    elsewherePoll = null;
  }
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

/**
 * The form `ran` went to the server to run, now or from the queue. The run's folder or the queue
 * keeps its spec, so the form as sent is saved. A draft's folder belongs to that run now: no form,
 * current or in the history, fills it again.
 */
function formSent(ran: FormState): void {
  savedForm = { ...ran, draftId: null };
  const draftId = ran.draftId;
  useStore.setState((s) => ({
    form: { ...s.form, draftId: null },
    formHistory: draftId ? forgetDraft(s.formHistory, draftId) : s.formHistory,
  }));
}

/** What is said about a run, as the server now has it, wherever this tab keeps its listing. */
function applyAnnotations(listing: RunListing): void {
  const said = {
    title: listing.title ?? null,
    starred: !!listing.starred,
    archived: !!listing.archived,
    annotations_error: listing.annotations_error ?? null,
  };
  useStore.setState((s) => {
    const detail = s.runDetails[listing.id];
    return {
      runs: s.runs.map((r) => (r.id === listing.id ? { ...r, ...said } : r)),
      ...(detail?.listing ? { runDetails: { ...s.runDetails, [listing.id]: { ...detail, listing: { ...detail.listing, ...said } } } } : {}),
    };
  });
}

/**
 * A run deleted, in this tab or another: forget it wherever it was shown. Runs derived from it
 * stay (the history shows they came from a deleted run); a comparison with it is left half chosen;
 * a form that was saved as this draft keeps its contents, now unsaved.
 */
function forgetRun(id: string): void {
  const st = useStore.getState();
  const bos = tokenizationKey(analysisContext(st).options);
  const { [id]: _detail, ...runDetails } = st.runDetails;
  const { [id]: _live, ...live } = st.live;
  const [a, b] = st.compareIds;
  const draft = st.form.draftId === id;
  if (draft) savedForm = DEFAULT_FORM; // its folder is gone: the form is kept in the browser instead
  inflight.delete(id);
  if (st.activeRunId === id) openSeq += 1; // whatever was opening it has nothing to show
  useStore.setState({
    runs: st.runs.filter((r) => r.id !== id),
    runDetails,
    live,
    flags: Object.fromEntries(Object.entries(st.flags).filter(([run, flag]) => run !== id && flag.against !== id)),
    compareIds: a === id || b === id ? [a === id ? null : a, b === id ? null : b] : st.compareIds,
    ...(st.activeRunId === id ? { activeRunId: null, analysisSource: "form" as const } : {}),
    ...(st.pendingRunId === id ? { pendingRunId: null } : {}),
    ...(draft ? { form: { ...st.form, draftId: null } } : {}),
    formHistory: forgetDraft(st.formHistory, id),
  });
  // The analyses read the form's prompts now, if they read the run's.
  if (st.activeRunId === id) analysisChanged(bos);
}

/** Show a run of the form `ran` that has just started, its results filling in as they stream. */
async function showStartedRun(ran: FormState, runId: string, job: Job): Promise<void> {
  const st = useStore.getState();
  openSeq += 1;
  formSent(ran);
  st.applyJob(job);
  useStore.setState((s) => ({
    pendingRunId: runId,
    activeRunId: runId,
    analysisSource: "run",
    view: "results",
    // The run's first events may already have arrived over the WebSocket; keep them.
    live: { ...s.live, [runId]: s.live[runId] ?? emptyLive(runId) },
  }));
  st.fitSelection();
  await st.refreshRuns();
  void st.loadRun(runId, true);
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
  const bos = tokenizationKey(analysisContext(st).options);
  useStore.setState({ form: next.form, formHistory: next.history });
  analysisChanged(bos);
}

/**
 * After the view (or the map's overlay) changes: store where prompts are now read from, and when
 * that is the active run, read its prompts, from its dataset, as it did. Changing BOS moves every
 * token, so a chosen token no longer means the same one.
 */
function followAnalysisSource(bosBefore: string): void {
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

/** The form to build on: from a run's spec (this one, or the run the analyses show), or the
 * form itself. */
function formShown(st: Store, runId?: string): FormState {
  const run = runId ? st.runDetails[runId]?.spec : undefined;
  const saved = run ?? (analysisSourceFor(st) === "run" && st.activeRunId ? st.runDetails[st.activeRunId]?.spec : undefined);
  return saved ? formFromSpec(saved) : st.form;
}

/** Open the experiment form on `form`, as a new draft and one undo step. With `fromTray`, the
 * staged sites went into it: they leave the tray, and Undo returns them, beside any staged since. */
function configureForm(form: FormState, fromTray: boolean): void {
  const st = useStore.getState();
  const bos = tokenizationKey(analysisContext(st).options);
  const staged = fromTray ? st.stagedSites : [];
  useStore.setState({ view: "experiment", analysisSource: "form", ...(fromTray ? { stagedSites: [] } : {}) });
  swapForm({ ...form, draftId: null, nameEdited: false, name: "" }, {
    onUndo: fromTray
      ? () => useStore.setState((now) => ({
          stagedSites: [...staged, ...now.stagedSites.filter((x) => !staged.some((y) => sameData(x, y)))],
        }))
      : undefined,
  });
  analysisChanged(bos);
}

/** After anything that can change the analysis context: tokens move when BOS or the chat
 * template changes (`bosBefore` is tokenizationKey's before), so the prompts are read again; the
 * prompt index stays within the prompts used. */
function analysisChanged(bosBefore: string): void {
  const st = useStore.getState();
  if (tokenizationKey(analysisContext(st).options) !== bosBefore) {
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
