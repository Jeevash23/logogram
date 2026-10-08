import { useEffect, useRef, useState } from "react";

import { api } from "../api/client";
import type { PredictionReport, PredictionSettings } from "../api/types";
import { Button, Callout, Empty, Field, Input, Segmented, Select, Spinner } from "../components/ui";
import { font, prepareCanvas, useChromeColors, useElementSize } from "../lib/canvas";
import { predictionSettingsKey } from "../lib/analysis";
import { pct, signed, visibleToken } from "../lib/format";
import { useActiveRun, useAnalysisContext } from "../lib/hooks";
import { formFromSpec, useStore } from "../store/app";
import s from "./views.module.css";
import p from "./PredictionsView.module.css";

export function PredictionsView() {
  const model = useStore(st => st.model);
  const dataset = useStore(st => st.datasetPath);
  const index = useStore(st => st.promptIndex);
  const token = useStore(st => st.tokenPosition);
  const selectedLayer = useStore(st => st.selection?.layer);
  const context = useAnalysisContext();
  const run = useActiveRun();
  const saved = useStore(st => st.analysisSource === "run" ? st.runDetails[st.activeRunId ?? ""]?.spec.predictions : st.form.predictions);
  const savedReport = useStore(st => st.analysisSource === "run" ? st.runDetails[st.activeRunId ?? ""]?.predictions : null);
  const [which, setWhich] = useState<"clean" | "corrupt">(saved?.which ?? "clean");
  const [positionMode, setPositionMode] = useState<"last" | "index">(saved?.position.kind ?? "last");
  const [positionText, setPositionText] = useState(String(saved?.position.kind === "index" ? saved.position.index : 0));
  const position = positionText.trim() ? Number(positionText) : NaN;
  const [topK, setTopK] = useState(saved?.top_k ?? 5);
  const [response, setResponse] = useState<{ key: string; report: PredictionReport } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const request = useRef(0);
  const settings: PredictionSettings = { method: "final_norm_logit_lens", which, position: positionMode === "last" ? { kind: "last" } : { kind: "index", index: position }, top_k: topK, prompt_index: index };
  const settingsKey = predictionSettingsKey(settings);
  const requestKey = `${context.key}|${settingsKey}`;
  const data = response?.key === requestKey ? response.report : null;
  useEffect(() => {
    request.current += 1; setResponse(savedReport && predictionSettingsKey(savedReport.settings) === settingsKey ? { key: requestKey, report: savedReport } : null); setError(null); setLoading(false);
    return () => { request.current += 1; };
  }, [context.key, settingsKey, savedReport]);
  const supported = model.info?.extra?.prediction_method === "final_norm_logit_lens";
  const valid = Number.isInteger(topK) && topK >= 1 && topK <= 20 && (positionMode === "last" || Number.isInteger(position));
  const ready = model.state === "ready" && !!dataset && context.n > 0 && !context.error && supported && valid;

  const measure = async () => {
    if (!dataset || !ready) return;
    const seq = ++request.current;
    setLoading(true); setError(null); setResponse(null);
    // Keep the exact context with the optional diagnostic when saving the next experiment spec.
    const st = useStore.getState();
    const source = st.analysisSource === "run" && run.detail ? formFromSpec(run.detail.spec) : st.form;
    useStore.setState({ form: { ...source, predictions: settings } });
    try {
      const report = await api.predictions({ dataset, index, settings, ...context.options });
      if (seq === request.current) setResponse({ key: requestKey, report });
    } catch (e) { if (seq === request.current) setError((e as Error).message); }
    finally { if (seq === request.current) setLoading(false); }
  };
  const selectLayer = (layer: number) => useStore.getState().select({ layer, part: "resid", kind: "resid_post", positionKey: String(data?.position ?? -1) });

  return <div className={s.view}>
    <div className={s.head}><div className={s.titleBlock}><h1 className={s.title}>Layer predictions</h1><p className={s.subtitle}>Read how the residual stream’s vocabulary projection changes across the model.</p></div><span className={p.methodTag}>Logit lens</span></div>
    <section className={p.method} aria-label="Prediction method"><h2>Final norm + vocabulary projection</h2><p>At each layer’s residual output, recompute the model’s final normalization and apply its vocabulary projection, including bias. Softmax gives the displayed probabilities. These intermediate projections are a diagnostic; they are not causal effects or calibrated early predictions.</p><p className={s.small}>{context.label} · original length-group batch composition preserved</p></section>
    <div className={p.controls}>
      <Field label="Prompt variant"><Segmented label="Prediction prompt variant" value={which} onChange={setWhich} options={[{ value: "clean", label: "Clean" }, { value: "corrupt", label: "Corrupt" }]} /></Field>
      <Field label="Read at position"><Select value={positionMode} onChange={e => setPositionMode(e.target.value as "last" | "index")}><option value="last">Last token</option><option value="index">Token index</option></Select></Field>
      {positionMode === "index" && <Field label="Token index"><Input type="number" step={1} value={positionText} onChange={e => setPositionText(e.target.value)} /></Field>}
      <Field label="Top tokens per layer"><Select value={topK} onChange={e => setTopK(Number(e.target.value))}>{Array.from({ length: 20 }, (_, i) => i + 1).map(k => <option key={k} value={k}>{k}</option>)}</Select></Field>
      {token !== null && <Button onClick={() => { setPositionMode("index"); setPositionText(String(token)); }}>Use selected token {token}</Button>}
      <Button variant="primary" disabled={!ready || loading} onClick={() => void measure()}>{loading ? "Measuring…" : "Measure predictions"}</Button>
    </div>
    <p className={s.small}>Prompt {index} · method, prompt index, position, variant, and top-token count are included in the experiment form when measured. Save the spec in Experiment to reproduce this diagnostic with a run.</p>
    {savedReport && data !== savedReport && <Button onClick={() => {
      const settings = savedReport.settings; setWhich(settings.which); setPositionMode(settings.position.kind); if (settings.position.kind === "index") setPositionText(String(settings.position.index)); setTopK(settings.top_k); useStore.getState().setPromptIndex(settings.prompt_index);
    }}>Open saved diagnostic · prompt {savedReport.index}</Button>}
    {data && data === savedReport && <p className={s.small}>Saved diagnostic from {run.detail?.spec.name} · available without loading the model</p>}
    {(context.error || error) && <Callout tone="error">{context.error ?? error}</Callout>}
    {!data && (model.state !== "ready" ? <Empty title="Load the model to inspect its layer predictions" action={<Button onClick={() => useStore.setState({ modelDialogOpen: true })}>Load a model</Button>}>This view computes a diagnostic using the model’s actual weights.</Empty> : !supported ? <Callout>Per-layer predictions need the model’s final normalization and unembedding to reproduce its output. For the loaded model they didn’t when it loaded, so this diagnostic would misdescribe it.</Callout> : !dataset ? <Empty title="Choose prompts first" action={<Button onClick={() => useStore.getState().setView("prompts")}>Set up prompts</Button>}>The diagnostic uses one prompt from the selected dataset.</Empty> : null)}
    {loading && <Spinner label="Projecting each layer’s residual output…" />}
    {data && <>
      <div className={p.reportHeading}><div><h2>Prompt {data.index} · token {data.position} <span>{visibleToken(data.tokens[data.position])}</span></h2><p className={s.small}>{data.settings.which} input · batch members {data.batch_members.join(", ")} · {data.layers.length} layers</p></div><Button disabled={selectedLayer === undefined} onClick={() => { if (selectedLayer !== undefined) selectLayer(selectedLayer); useStore.setState({ view: "explore", exploreMode: "layer" }); }}>Open selected layer</Button></div>
      {data.position !== data.tokens.length - 1 && <Callout>The answer and distractor below are the dataset’s reference tokens. At this earlier position, they are not necessarily the expected next token.</Callout>}
      <PredictionTrace data={data} selectedLayer={selectedLayer} />
      <div className={p.tableScroll}><table className={`${s.table} ${p.table}`}><caption>Per-layer vocabulary projections. Select a layer to inspect its residual output.</caption><thead><tr><th scope="col">Layer</th><th scope="col">Top projected tokens · probability</th><th scope="col">Answer · {visibleToken(data.answer)}</th><th scope="col">Distractor · {visibleToken(data.distractor)}</th><th scope="col">Logit difference</th></tr></thead><tbody>{data.layers.map(layer => <tr key={layer.layer} aria-selected={selectedLayer === layer.layer}><th scope="row"><button type="button" aria-label={`Select residual output of layer ${layer.layer}`} aria-pressed={selectedLayer === layer.layer} onClick={() => selectLayer(layer.layer)}>Layer {layer.layer}</button></th><td><div className={p.tokens}>{layer.top.map((t, i) => <span key={t.id} title={`Token id ${t.id} · probability ${t.prob.toPrecision(8)}`}><small>{i + 1}</small><strong>{visibleToken(t.token) || "∅"}</strong><span>{pct(t.prob, 2)}</span></span>)}</div></td><td>{pct(layer.answer_prob, 3)}</td><td>{pct(layer.distractor_prob, 3)}</td><td>{signed(layer.logit_diff, 3)}</td></tr>)}</tbody></table></div>
      {data.issues.map(issue => <Callout key={`${issue.index}-${issue.kind}`}>Prompt {issue.index}: {issue.message}</Callout>)}
    </>}
  </div>;
}

function PredictionTrace({ data, selectedLayer }: { data: PredictionReport; selectedLayer?: number }) {
  const host = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const { width } = useElementSize(host);
  const colors = useChromeColors();
  const max = Math.min(1, Math.max(.01, Math.ceil(Math.max(...data.layers.flatMap(l => [l.answer_prob, l.distractor_prob])) * 100) / 100));
  useEffect(() => {
    if (!canvas.current || width < 1) return;
    const height = 215, left = 52, right = 20, top = 16, bottom = 30;
    const ctx = prepareCanvas(canvas.current, width, height);
    if (!ctx) return;
    const x = (i: number) => left + i / Math.max(1, data.layers.length - 1) * (width - left - right);
    const y = (v: number) => top + (1 - v / max) * (height - top - bottom);
    ctx.font = font(colors, 11, 450); ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let i = 0; i <= 4; i++) {
      const value = max * i / 4;
      ctx.fillStyle = colors.muted; ctx.fillText(pct(value, max <= .04 ? 2 : 1), left - 10, y(value));
      ctx.strokeStyle = colors.lineSoft; ctx.beginPath(); ctx.moveTo(left, y(value)); ctx.lineTo(width - right, y(value)); ctx.stroke();
    }
    if (selectedLayer !== undefined && selectedLayer < data.layers.length) { ctx.strokeStyle = colors.muted; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(x(selectedLayer), top); ctx.lineTo(x(selectedLayer), height - bottom); ctx.stroke(); ctx.setLineDash([]); }
    for (const field of ["answer_prob", "distractor_prob"] as const) {
      ctx.strokeStyle = field === "answer_prob" ? colors.text : colors.muted;
      ctx.lineWidth = 2; ctx.setLineDash(field === "answer_prob" ? [] : [5, 4]); ctx.beginPath();
      data.layers.forEach((layer, i) => i === 0 ? ctx.moveTo(x(i), y(layer[field])) : ctx.lineTo(x(i), y(layer[field]))); ctx.stroke(); ctx.setLineDash([]);
      data.layers.forEach((layer, i) => { ctx.fillStyle = colors.surface; ctx.beginPath(); ctx.arc(x(i), y(layer[field]), 3, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); });
    }
    ctx.font = font(colors, 11, 450); ctx.fillStyle = colors.muted; ctx.textAlign = "center";
    const every = Math.max(1, Math.ceil(data.layers.length / Math.max(2, Math.floor(width / 55))));
    data.layers.forEach((layer, i) => { if (i % every === 0 || i === data.layers.length - 1) ctx.fillText(`L${layer.layer}`, x(i), height - 10); });
  }, [width, data, selectedLayer, colors, max]);
  return <div className={p.trace} ref={host}><div className={p.legend}><span><i />Answer · {visibleToken(data.answer)}</span><span><i className={p.dashed} />Distractor · {visibleToken(data.distractor)}</span><span className={s.faint}>Projected probability · 0–{pct(max, max <= .04 ? 2 : 0)}</span></div><canvas ref={canvas} role="img" aria-label="Projected answer and distractor probability by layer. Exact values and layer selection are in the table below." /></div>;
}
