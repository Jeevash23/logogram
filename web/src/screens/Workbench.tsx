import { useEffect, useRef, useState } from "react";

import { Header } from "../components/Header";
import { History } from "../components/History";
import { Inspector } from "../components/Inspector";
import { ModelDialog } from "../components/ModelDialog";
import { RobustnessDialog } from "../components/RobustnessDialog";
import { Splitter } from "../components/Splitter";
import { StatusLine } from "../components/StatusLine";
import { TokenStrip } from "../components/TokenStrip";
import { Button, Dialog } from "../components/ui";
import { useAnalysisContext } from "../lib/hooks";
import { useStore, workspaceFor, type View, type Workspace } from "../store/app";
import { AttentionView } from "../views/AttentionView";
import { BaselineView } from "../views/BaselineView";
import { CompareView } from "../views/CompareView";
import { ExperimentView } from "../views/ExperimentView";
import { ExploreView } from "../views/ExploreView";
import { HeadComparisonView } from "../views/HeadComparisonView";
import { PredictionsView } from "../views/PredictionsView";
import { PromptsView } from "../views/PromptsView";
import { ResearchView, NoteEditorDialog } from "../views/ResearchView";
import { ResultsView } from "../views/ResultsView";
import { SpecView } from "../views/SpecView";
import s from "./Workbench.module.css";

const WORKSPACES: { id: Workspace; label: string; views: { id: View; label: string }[] }[] = [
  { id: "explore", label: "Explore", views: [{ id: "explore", label: "Model" }, { id: "attention", label: "Attention" }, { id: "heads", label: "Head comparison" }, { id: "predictions", label: "Layer predictions" }] },
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

  return <div className={s.workbench}>
    <Header />
    <div className={s.workspaceBar}>
      <nav className={s.workspaces} aria-label="Workspaces">{WORKSPACES.map(w => <button key={w.id} type="button" aria-current={workspace === w.id ? "page" : undefined} onClick={() => w.id === "experiment" && staged > 0 ? useStore.getState().configureStaged() : setView(lastView.current[w.id])}>{w.label}{w.id === "experiment" && staged > 0 && <span className={s.count} title="Staged sites">{staged}</span>}</button>)}</nav>
      <Button size="small" variant="ghost" aria-expanded={drawer === "history"} onClick={() => setDrawer("history")}>History <span className={s.count}>{runs}</span></Button>
    </div>
    <FirstExperiment />
    <div className={s.viewBar}>
      <nav className={s.tabs} aria-label={`${workspace} views`}>{WORKSPACES.find(w => w.id === workspace)?.views.map(v => <button key={v.id} type="button" className={s.tab} aria-current={view === v.id ? "page" : undefined} onClick={() => setView(v.id)}>{v.label}{v.id === "heads" && pins > 0 && <span className={s.count}>{pins}/2</span>}</button>)}</nav>
      <Button size="small" variant="ghost" aria-expanded={showInspector || drawer === "inspector"} onClick={() => narrow || !supportsInspector ? setDrawer("inspector") : setInspector(!inspector)}>{showInspector ? "Hide inspector" : "Inspector"}</Button>
    </div>
    <div className={s.body}>
      <main className={s.center}>
        <div className={s.view} ref={viewPane}><ActiveView view={view} /></div>
        {!["notes", "compare", "spec"].includes(view) && <TokenStrip />}
      </main>
      {showInspector && <><Splitter orientation="vertical" label="Resize the inspector" onReset={() => setInspectorWidth(320)} onResize={(delta, phase) => {
        if (phase === "start") { startWidth.current = inspectorWidth; return; }
        const next = Math.max(280, Math.min(480, startWidth.current - delta)); setInspectorWidth(next);
        if (phase === "end") { try { localStorage.setItem("logogram.inspectorWidth", String(next)); } catch { /* use session size */ } }
      }} /><aside className={s.inspector} style={{ width: inspectorWidth }} aria-label="Inspector"><Inspector /></aside></>}
    </div>
    <StatusLine />
    <ModelDialog />
    <RobustnessDialog />
    <NoteEditorDialog />
    <Dialog open={drawer !== null} onOpenChange={open => { if (!open) setDrawer(null); }} title={drawer === "history" ? "Experiment history" : "Inspector"} wide><div className={s.drawer}>{drawer === "history" ? <History onOpen={() => setDrawer(null)} /> : <Inspector />}</div></Dialog>
  </div>;
}

function FirstExperiment() {
  const model = useStore((st) => st.model);
  const dataset = useStore((st) => st.dataset);
  const completed = useStore((st) => st.runs.some((r) => r.status === "finished"));
  const context = useAnalysisContext();
  const baseline = useStore((st) => st.baselines[context.key]);
  if (completed) return null;
  const step = model.state !== "ready" ? 0 : !dataset ? 1 : !baseline ? 2 : 3;
  return <nav className={s.firstRun} aria-label="First experiment">
    {["Load a model", "Prepare prompts", "Check baseline", "Run experiment"].map((label, i) => <button
      key={label} type="button" aria-current={step === i ? "step" : undefined}
      onClick={() => i === 0 ? useStore.setState({ modelDialogOpen: true }) : useStore.getState().setView(i === 1 ? "prompts" : i === 2 ? "baseline" : "experiment")}
    >{i + 1}. {label}</button>)}
  </nav>;
}

function ActiveView({ view }: { view: View }) {
  switch (view) {
    case "explore": return <ExploreView />;
    case "heads": return <HeadComparisonView />;
    case "predictions": return <PredictionsView />;
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
