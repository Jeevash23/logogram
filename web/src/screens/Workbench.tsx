import { useEffect, useRef, useState } from "react";

import { CancelRunDialog } from "../components/CancelRunDialog";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { DatasetChip, ModelChip } from "../components/Header";
import { History } from "../components/History";
import { Inspector } from "../components/Inspector";
import { ModelDialog } from "../components/ModelDialog";
import { RobustnessDialog } from "../components/RobustnessDialog";
import { Splitter } from "../components/Splitter";
import { StatusLine } from "../components/StatusLine";
import { TokenStrip } from "../components/TokenStrip";
import { TopBar } from "../components/TopBar";
import { Button, Dialog, Icon } from "../components/ui";
import { useAnalysisContext } from "../lib/hooks";
import { useStore, workspaceFor, type View, type Workspace } from "../store/app";
import { AttentionView } from "../views/AttentionView";
import { BaselineView } from "../views/BaselineView";
import { CompareView } from "../views/CompareView";
import { ExperimentView } from "../views/ExperimentView";
import { ExploreView } from "../views/ExploreView";
import { HeadComparisonView } from "../views/HeadComparisonView";
import { PredictionsView } from "../views/PredictionsView";
import { FeaturesView } from "../views/FeaturesView";
import { PromptsView } from "../views/PromptsView";
import { ResearchView, NoteEditorDialog } from "../views/ResearchView";
import { ResultsView } from "../views/ResultsView";
import { SpecView } from "../views/SpecView";
import s from "./Workbench.module.css";

const WORKSPACES: { id: Workspace; label: string; views: { id: View; label: string }[] }[] = [
  { id: "explore", label: "Explore", views: [{ id: "explore", label: "Model" }, { id: "attention", label: "Attention" }, { id: "heads", label: "Head comparison" }, { id: "predictions", label: "Layer predictions" }, { id: "features", label: "Features" }] },
  { id: "experiment", label: "Experiment", views: [{ id: "prompts", label: "Prompts" }, { id: "baseline", label: "Baseline" }, { id: "experiment", label: "Configure" }, { id: "spec", label: "Spec" }] },
  { id: "evidence", label: "Evidence", views: [{ id: "results", label: "Results" }, { id: "compare", label: "Compare runs" }, { id: "notes", label: "Research notes" }] },
];

export function Workbench() {
  const view = useStore(st => st.view);
  const setView = useStore(st => st.setView);
  const workspace = workspaceFor(view);
  const pins = useStore(st => st.headPins.length);
  const staged = useStore(st => st.stagedSites.length);
  const runs = useStore(st => st.runs.length);
  const inspectorFocus = useStore(st => st.inspectorFocus);
  const project = useStore(st => st.project?.session_id);
  const activeRunId = useStore(st => st.activeRunId);
  const exploreMode = useStore(st => st.exploreMode);
  const viewPane = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(window.innerWidth);
  const [inspector, setInspector] = useState(true);
  const [drawer, setDrawer] = useState<"history" | "inspector" | null>(null);
  const [inspectorWidth, setInspectorWidth] = useState(() => {
    try { return Math.max(280, Math.min(480, Number(localStorage.getItem("logogram.inspectorWidth")) || 320)); }
    catch { return 320; }
  });
  const startWidth = useRef(inspectorWidth);
  const lastView = useRef<Record<Workspace, View>>({ explore: "explore", experiment: "experiment", evidence: "results" });
  const narrow = width < 1120;
  const supportsInspector = ["explore", "attention", "results"].includes(view);
  const showInspector = !narrow && inspector && supportsInspector;
  useEffect(() => { const resize = () => setWidth(window.innerWidth); window.addEventListener("resize", resize); return () => window.removeEventListener("resize", resize); }, []);
  useEffect(() => { lastView.current[workspace] = view; setDrawer(null); }, [workspace, view, project]);
  useEffect(() => { viewPane.current?.scrollTo({ top: 0, left: 0 }); }, [view, exploreMode, activeRunId]);
  useEffect(() => {
    if (!inspectorFocus) return;
    if (narrow || !supportsInspector) setDrawer("inspector");
    else setInspector(true);
  }, [inspectorFocus, narrow, supportsInspector]);

  const goWorkspace = (w: Workspace) =>
    w === "experiment" && staged > 0 ? useStore.getState().configureStaged() : setView(lastView.current[w]);

  return <div className={s.workbench} style={{ ["--ws" as string]: `var(--${workspace})` }}>
    <TopBar workspace={workspace} onWorkspace={goWorkspace} onHistory={() => setDrawer("history")} runs={runs} staged={staged} />
    <div className={s.subbar}>
      <nav className={s.tabs} aria-label={`${workspace} views`}>{WORKSPACES.find(w => w.id === workspace)?.views.map(v => <button key={v.id} type="button" className={s.tab} aria-current={view === v.id ? "page" : undefined} onClick={() => setView(v.id)}>{v.label}{v.id === "heads" && pins > 0 && <span className={s.count}>{pins}/2</span>}</button>)}</nav>
      <div className={s.context}>
        <NextStep />
        <ModelChip />
        <DatasetChip />
        <Button size="small" variant="ghost" aria-expanded={showInspector || drawer === "inspector"} onClick={() => narrow || !supportsInspector ? setDrawer("inspector") : setInspector(!inspector)}>{showInspector ? "Hide inspector" : "Inspector"}</Button>
      </div>
    </div>
    <div className={s.body}>
      <main className={s.center}>
        <div className={s.view} ref={viewPane}><ErrorBoundary key={view} name="This view"><ActiveView view={view} /></ErrorBoundary></div>
        {!["notes", "compare", "spec"].includes(view) && <ErrorBoundary name="The token strip" resetKey={view}><TokenStrip /></ErrorBoundary>}
      </main>
      {showInspector && <><Splitter orientation="vertical" label="Resize the inspector" onReset={() => setInspectorWidth(320)} onResize={(delta, phase) => {
        if (phase === "start") { startWidth.current = inspectorWidth; return; }
        const next = Math.max(280, Math.min(480, startWidth.current - delta)); setInspectorWidth(next);
        if (phase === "end") { try { localStorage.setItem("logogram.inspectorWidth", String(next)); } catch { /* use session size */ } }
      }} /><aside className={s.inspector} style={{ width: inspectorWidth }} aria-label="Inspector"><SafeInspector /></aside></>}
    </div>
    <StatusLine />
    <CancelRunDialog />
    <ModelDialog />
    <RobustnessDialog />
    <NoteEditorDialog />
    <Dialog open={drawer !== null} onOpenChange={open => { if (!open) setDrawer(null); }} title={drawer === "history" ? "Experiment history" : "Inspector"} wide><div className={s.drawer}>{drawer === "history" ? <ErrorBoundary name="The history"><History onOpen={() => setDrawer(null)} /></ErrorBoundary> : <SafeInspector />}</div></Dialog>
  </div>;
}

const STEPS = ["Load a model", "Prepare prompts", "Check the baseline", "Run the experiment"];

/** Until the first run finishes: the next of four steps, one click away. */
function NextStep() {
  const model = useStore((st) => st.model);
  const dataset = useStore((st) => st.dataset);
  const completed = useStore((st) => st.runs.some((r) => r.status === "finished"));
  const context = useAnalysisContext();
  const baseline = useStore((st) => st.baselines[context.key]);
  if (completed) return null;
  const step = model.state !== "ready" ? 0 : !dataset ? 1 : !baseline ? 2 : 3;
  const go = () =>
    step === 0 ? useStore.setState({ modelDialogOpen: true }) : useStore.getState().setView(step === 1 ? "prompts" : step === 2 ? "baseline" : "experiment");
  return <button type="button" className={s.next} onClick={go} aria-label={`Next step: ${STEPS[step]}`} title="Your first experiment, in four steps">
    <span className={s.nextDots} aria-hidden="true">{STEPS.map((label, i) => <span key={label} data-state={i < step ? "done" : i === step ? "now" : "todo"} />)}</span>
    <span className={s.nextText}><span className={s.nextCount}>Step {step + 1} of 4</span>{STEPS[step]}</span>
    <Icon name="arrowRight" size={14} />
  </button>;
}

/** The inspector, contained: an error in it leaves the rest of the workbench working. Choosing
 * another component or run tries again. */
function SafeInspector() {
  const view = useStore((st) => st.view);
  const activeRunId = useStore((st) => st.activeRunId);
  const selection = useStore((st) => st.selection);
  return (
    <ErrorBoundary name="The inspector" resetKey={`${view}|${activeRunId}|${JSON.stringify(selection)}`}>
      <Inspector />
    </ErrorBoundary>
  );
}

function ActiveView({ view }: { view: View }) {
  switch (view) {
    case "explore": return <ExploreView />;
    case "heads": return <HeadComparisonView />;
    case "predictions": return <PredictionsView />;
    case "features": return <FeaturesView />;
    case "notes": return <ResearchView />;
    case "prompts": return <PromptsView />;
    case "baseline": return <BaselineView />;
    case "experiment": return <ExperimentView />;
    case "results": return <ResultsView />;
    case "attention": return <AttentionView />;
    case "compare": return <CompareView />;
    case "spec": return <SpecView />;
  }
}
