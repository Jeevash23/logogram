import { useEffect, useMemo, useRef, type KeyboardEvent } from "react";

import type { SiteBase, SiteResult } from "../api/types";
import { ModelMap } from "../components/ModelMap/ModelMap";
import { Legend } from "../components/ModelMap/Legend";
import { Logogram } from "../components/Logogram";
import { LogogramDial } from "../components/LogogramDial";
import { Button, Empty, Segmented, Select } from "../components/ui";
import { fitText, font, prepareCanvas, ring, useChromeColors, useElementSize } from "../lib/canvas";
import { divergingScale, niceBound, SCALE_FLOOR, textOn } from "../lib/color";
import { count, signed } from "../lib/format";
import { modelName, siteValue, useActiveRun, useArchitecture, useRunProfile, type Architecture } from "../lib/hooks";
import { componentLabel, findSite, sameSelection, selectionOfSite, type Selection } from "../lib/sites";
import { siteFromSelection, siteText } from "../lib/spec";
import { useStore } from "../store/app";
import s from "./ExploreView.module.css";

export function componentMeasurement(sites: SiteBase[], results: Record<number, SiteResult>, selection: Selection) {
  const site = findSite(sites, selection);
  return site ? results[site.index] : undefined;
}

export function ExploreView() {
  const arch = useArchitecture();
  const run = useActiveRun();
  const selection = useStore(st => st.selection);
  const select = useStore(st => st.select);
  const mode = useStore(st => st.exploreMode);
  const overlay = useStore(st => st.mapOverlay);
  const modelState = useStore(st => st.model.state);
  const layer = Math.min(selection?.layer ?? 0, Math.max(0, (arch?.nLayers ?? 1) - 1));
  const pins = useStore(st => st.headPins);
  const tokenPosition = useStore(st => st.tokenPosition);
  const current = selection && arch && selection.layer < arch.nLayers ? selection : null;
  const profile = useRunProfile(run);
  const strongest = useMemo(
    () => Object.values(run.results).filter((x) => x.effect.mean !== null).sort((a, b) => Math.abs(b.effect.mean ?? 0) - Math.abs(a.effect.mean ?? 0)).slice(0, 3),
    [run.results],
  );

  const changeLayer = (next: number) => select({ ...(current ?? { part: "head", head: 0 }), layer: next });

  return <div className={s.explorer}>
    <div className={s.heading}>
      <div><p className={s.eyebrow}>Model explorer</p><h1>{arch ? modelName(arch.id) : "Look inside a model"}</h1>
        <p className={s.description}>{arch ? `${count(arch.nLayers)} layers · ${count(arch.nLayers * arch.nHeads)} heads${arch.dModel ? ` · ${count(arch.dModel)} residual dimensions` : ""}` : "Explore its structure, choose a component, and follow the evidence."}</p>
      </div>
      <div className={s.controls}>
        <Segmented label="Explorer view" value={mode} onChange={v => useStore.setState({ exploreMode: v })} options={[{ value: "atlas", label: "Model atlas" }, { value: "layer", label: "Layer explorer" }]} />
        <Select aria-label="Model overlay" value={overlay} onChange={e => useStore.getState().setMapOverlay(e.target.value as "data" | "structure")}><option value="data">Intervention results</option><option value="structure">Architecture</option></Select>
      </div>
    </div>
    {!arch ? <Empty title={modelState === "loading" ? "Loading the model…" : "Start with a model"} action={<Button variant="primary" onClick={() => useStore.setState({ modelDialogOpen: true })}>Load a model</Button>}>The map uses the loaded model’s actual layers, attention heads, and supported intervention sites.</Empty> : <>
      <div className={s.contextLine}>
        <span className={s.contextRun}>
          {run.id && <Logogram seed={run.id} profile={profile} size={30} />}
          <span>{run.detail?.spec.name ?? (run.running ? "Experiment running" : "No experiment selected")}</span>
        </span>
        <span>{run.running ? "Results arrive layer by layer" : strongest.length ? <>Strongest: {strongest.map((x, i) => <span key={x.index}>{i > 0 && " · "}<button type="button" className={s.reading} onClick={() => select(selectionOfSite(x))}>{x.label} <strong>{signed(x.effect.mean, 2)}</strong></button></span>)}</> : run.id ? `${count(Object.keys(run.results).length)} measured sites` : "Run an experiment to color the model"}</span>
      </div>
      {mode === "atlas" ? <div className={s.atlasRow}>
        <div className={s.atlas}><ModelMap prominent structureOnly={overlay === "structure"} /><p className={s.hint}>Double-click a component to open its layer. Arrow keys move through the map. Empty outlines have no measurement in this run.</p></div>
        {run.id && profile && overlay === "data" && <aside className={s.dialPanel} aria-label="Run logogram"><LogogramDial seed={run.id} profile={profile} sites={run.sites} results={run.results} size={150} /></aside>}
      </div> : <div className={s.layerWorkspace}>
        <nav className={s.layerNav} aria-label="Model layers"><span className={s.eyebrow}>Layers</span>
          {Array.from({ length: arch.nLayers }, (_, l) => <button key={l} type="button" aria-current={l === layer ? "true" : undefined} onClick={() => changeLayer(l)}><span>Layer {l}</span><span className={s.layerCount}>{arch.nHeads} heads</span></button>)}
        </nav>
        <div className={s.layerDetail}>
          <div className={s.layerHeading}><div><p className={s.eyebrow}>Transformer block</p><h2>Layer {layer}</h2></div><div className={s.controls}><Button size="small" variant="ghost" icon="chevronLeft" aria-label="Previous layer" disabled={layer === 0} onClick={() => changeLayer(layer - 1)} /><Button size="small" variant="ghost" icon="chevronRight" aria-label="Next layer" disabled={layer >= arch.nLayers - 1} onClick={() => changeLayer(layer + 1)} /></div></div>
          <LayerDiagram arch={arch} layer={layer} />
        </div>
      </div>}
      <div className={s.selectionBar}>
        <div><span className={s.eyebrow}>Selected component</span><strong>{current ? componentLabel(current) : "Choose a component on the model"}</strong><span className={s.hint}> {tokenPosition === null ? "Add at the component’s selected position, or all positions" : `Add at token index ${tokenPosition}`}</span></div>
        <div className={s.controls}>
          {mode === "atlas" && <Button disabled={!current} onClick={() => useStore.setState({ exploreMode: "layer" })}>Open layer</Button>}
          <Button disabled={!current} onClick={() => {
            if (!current) return;
            const site = siteFromSelection(current);
            if (tokenPosition !== null) site.position = { kind: "index", index: tokenPosition };
            useStore.getState().prepareNote([site]);
          }}>Save selection</Button>
          {current?.part === "head" && <Button aria-pressed={pins.some(p => p.layer === current.layer && p.head === current.head)} onClick={() => useStore.getState().pinHead(current)}>Pin for comparison</Button>}
          <Button variant="primary" disabled={!current} onClick={() => current && useStore.getState().stageSelection(current)}>Add to experiment</Button>
        </div>
      </div>
      <SelectionTray />
    </>}
  </div>;
}

interface Node { x: number; y: number; w: number; h: number; label: string; detail?: string; sel?: Selection }

/** How a layer's computation is drawn. The residual additions are measured when a model loads
 * (backends/transformer_lens.check_model); only GPT-2's normalization placement is also known. */
type Flow = "pre_norm" | "sequential" | "parallel" | "inventory";

export function flowOf(structure: string): Flow {
  if (structure === "sequential_pre_norm") return "pre_norm";
  if (structure === "sequential" || structure === "parallel") return structure;
  return "inventory";
}

const FLOW_TEXT: Record<Flow, string> = {
  pre_norm: "Residual connections show this model's computation order.",
  sequential: "Attention and then the MLP are each added to the residual stream, as measured when the model loaded.",
  parallel: "Attention and the MLP both read the residual stream entering the layer, and both are added to it, as measured when the model loaded.",
  inventory: "Component inventory: the order of computation couldn't be verified for this model.",
};

function geometry(arch: Architecture, layer: number, width: number) {
  const flow = flowOf(arch.structure);
  const norms = flow === "pre_norm";
  const spine = 21;
  const x = Math.max(66, width * .19), w = Math.max(80, width - x - 10);
  const columns = Math.min(12, arch.nHeads, Math.max(2, Math.floor(w / 57)));
  const pitch = w / columns;
  const rows = Math.ceil(arch.nHeads / columns);
  const nodes: Node[] = [];
  const add = (y: number, label: string, kind: "resid_pre" | "resid_mid" | "resid_post") => nodes.push({ x: 9, y, w: width - 18, h: 29, label, sel: { layer, part: "resid", kind } });
  add(8, "Residual in", "resid_pre");
  if (norms) nodes.push({ x, y: 67, w, h: 31, label: "Layer norm" });
  const headsY = norms ? 140 : 88;
  for (let h = 0; h < arch.nHeads; h++) nodes.push({ x: x + h % columns * pitch + 2, y: headsY + Math.floor(h / columns) * 51, w: pitch - 6, h: 43, label: `H${h}`, sel: { layer, part: "head", head: h } });
  const attentionY = headsY + rows * 51 + 28;
  nodes.push({ x, y: attentionY, w, h: 46, label: "Attention output", detail: "Combined heads · output projection", sel: { layer, part: "attn" } });
  const midY = attentionY + 77;
  if (arch.siteKinds.includes("resid_mid")) add(midY, "Residual after attention", "resid_mid");
  if (norms) nodes.push({ x, y: midY + 58, w, h: 31, label: "Layer norm" });
  const mlpY = midY + (norms ? 113 : 62);
  if (arch.siteKinds.includes("mlp_out")) nodes.push({ x, y: mlpY, w, h: 58, label: "MLP output", detail: arch.dMlp && arch.dModel ? `${count(arch.dModel)} → ${count(arch.dMlp)} → ${count(arch.dModel)}` : "Feed-forward component", sel: { layer, part: "mlp" } });
  const postY = mlpY + 89;
  add(postY, "Residual out", "resid_post");
  return { nodes, height: postY + 45, spine, branch: x + w / 2, right: x + w, attentionY, midY, mlpY, headsY, flow };
}

function LayerDiagram({ arch, layer }: { arch: Architecture; layer: number }) {
  const run = useActiveRun();
  const colors = useChromeColors();
  const selection = useStore(st => st.selection);
  const select = useStore(st => st.select);
  const metric = useStore(st => st.mapMetric);
  const overlay = useStore(st => st.mapOverlay);
  const scaleMode = useStore(st => st.scaleMode);
  const theme = useStore(st => st.theme);
  const host = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const { width } = useElementSize(host);
  const geo = useMemo(() => geometry(arch, layer, width), [arch, layer, width]);
  const bound = metric === "effect" && scaleMode === "unit" ? 1 : niceBound(Object.values(run.results).map(r => siteValue(r, metric)), SCALE_FLOOR[metric]);
  const color = useMemo(() => divergingScale(bound, theme), [bound, theme]);
  const selectionFor = (node: Node) => node.sel ? { ...node.sel, positionKey: selection?.positionKey } : null;
  useEffect(() => {
    if (!canvas.current || width < 1) return;
    const ctx = prepareCanvas(canvas.current, width, geo.height);
    if (!ctx) return;
    const text = (label: string, x: number, y: number, size = 12, muted = false, align: CanvasTextAlign = "center") => {
      ctx.font = font(colors, size, muted ? 450 : 560); ctx.fillStyle = muted ? colors.muted : colors.text; ctx.textAlign = align; ctx.textBaseline = "middle"; ctx.fillText(label, x, y);
    };
    const path = (points: number[][]) => { ctx.beginPath(); ctx.strokeStyle = colors.line; ctx.lineWidth = 1.3; points.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.stroke(); };
    if (geo.flow !== "inventory") {
      const { spine, branch, right, attentionY, midY, mlpY } = geo;
      path([[spine, 38], [spine, geo.height - 27]]);
      path([[spine, 47], [branch, 47], [branch, attentionY + 62], [spine, attentionY + 62]]);
      if (geo.flow === "parallel") {
        // The MLP reads the residual stream entering the layer, before attention is added: the
        // same wire as attention's, continued past it.
        path([[branch, 47], [right + 6, 47], [right + 6, mlpY + 29], [right, mlpY + 29]]);
        path([[branch, mlpY + 58], [branch, mlpY + 76], [spine, mlpY + 76]]);
      } else {
        path([[spine, midY + 40], [branch, midY + 40], [branch, mlpY + 76], [spine, mlpY + 76]]);
      }
      for (const y of [attentionY + 62, mlpY + 76]) {
        ctx.beginPath(); ctx.arc(spine, y, 7, 0, 2 * Math.PI); ctx.fillStyle = colors.surface; ctx.fill(); ctx.stroke();
        text("+", spine, y, 12, true);
      }
    }
    text(`${arch.nHeads} attention heads${arch.dHead ? ` · ${arch.dHead} dimensions each` : ""}`, (width + Math.max(66, width * .19)) / 2, geo.headsY - 16, 11, true);
    for (const node of geo.nodes) {
      const sel = node.sel ? { ...node.sel, positionKey: selection?.positionKey } : null;
      const site = sel ? componentMeasurement(run.sites, run.results, sel) : undefined;
      const value = siteValue(site, metric);
      const painted = overlay === "data" && value !== null;
      const fill = painted ? color(value) : colors.surface;
      ctx.fillStyle = fill; ctx.strokeStyle = colors.lineSoft; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.roundRect(node.x, node.y, node.w, node.h, 4); ctx.fill(); if (!painted) ctx.stroke();
      ctx.font = font(colors, node.sel?.part === "head" ? 13 : 12.5, 550); ctx.textAlign = node.sel?.part === "resid" ? "left" : "center"; ctx.textBaseline = "middle"; ctx.fillStyle = painted ? textOn(fill, theme) : colors.text;
      const tx = node.sel?.part === "resid" ? node.x + 12 : node.x + node.w / 2;
      ctx.fillText(fitText(ctx, node.label, node.w - 18), tx, node.y + (node.detail ? 17 : node.h / 2));
      if (node.detail) { ctx.font = font(colors, 11, 450); ctx.fillText(fitText(ctx, node.detail, node.w - 18), tx, node.y + 35); }
      if (sel && sameSelection(sel, selection)) ring(ctx, node.x, node.y, node.w, node.h, colors.text, colors.surface, 1.5);
    }
  }, [geo, width, colors, selection, run.sites, run.results, color, metric, theme, overlay]);

  const onKey = (e: KeyboardEvent<HTMLButtonElement>, i: number) => {
    if (!["ArrowRight", "ArrowLeft", "ArrowDown", "ArrowUp", "Home", "End"].includes(e.key)) return;
    e.preventDefault();
    const buttons = Array.from(host.current?.querySelectorAll<HTMLButtonElement>("button") ?? []);
    const next = e.key === "Home" ? 0 : e.key === "End" ? buttons.length - 1 : Math.min(buttons.length - 1, Math.max(0, i + (["ArrowLeft", "ArrowUp"].includes(e.key) ? -1 : 1)));
    buttons[next]?.focus(); buttons[next]?.click();
  };

  const interactive = geo.nodes.filter(node => node.sel && arch.siteKinds.includes(siteFromSelection(node.sel, node.sel.kind).kind));
  return <>
    <div ref={host} className={s.diagram} style={{ minHeight: geo.height }}>
      <canvas ref={canvas} role="img" aria-label={`Layer ${layer}: ${arch.nHeads} attention heads, residual sites, attention output and MLP output. ${FLOW_TEXT[geo.flow]}`} />
      {interactive.map((node, i) => { const sel = selectionFor(node) as Selection; const data = componentMeasurement(run.sites, run.results, sel); return <button key={componentLabel(sel)} type="button" className={s.hit} style={{ left: node.x, top: node.y, width: node.w, height: node.h }} aria-label={`${componentLabel(sel)}: ${data ? `${metric} ${signed(siteValue(data, metric), 3)}` : "no single measurement at this position"}`} aria-pressed={sameSelection(sel, selection)} onClick={() => select(sel)} onKeyDown={e => onKey(e, i)} />; })}
    </div>
    <p className={s.hint}>{FLOW_TEXT[geo.flow]} Select a component to inspect it. Cells show a single measured site; choose a position in Evidence when a component has several.</p>
    {overlay === "data" && <Legend bound={bound} hasFlags={false} flagCount={0} />}
  </>;
}

export function SelectionTray() {
  const sites = useStore(st => st.stagedSites);
  if (!sites.length) return null;
  return <section className={s.tray} aria-label="Experiment selection">
    <div className={s.trayHeading}><div><span className={s.eyebrow}>Experiment selection</span><strong>{sites.length} site{sites.length === 1 ? "" : "s"} · measured individually</strong></div><div className={s.controls}><Button size="small" onClick={() => useStore.getState().prepareNote(sites)}>Save selection</Button><Button variant="primary" onClick={() => useStore.getState().configureStaged()}>Configure experiment</Button></div></div>
    <div className={s.chips}>{sites.map((site, i) => <span key={`${siteText(site)}-${i}`} className={s.chip}>{siteText(site)}<span className={s.hint}>{site.position.kind === "all" ? " · all positions" : ""}</span><button type="button" aria-label={`Remove ${siteText(site)} from experiment`} onClick={() => useStore.setState({ stagedSites: sites.filter((_, j) => i !== j) })}>×</button></span>)}</div>
    <p className={s.hint}>Each site is a separate intervention in the sweep. Review direction, baseline, and execution settings before running.</p>
  </section>;
}
