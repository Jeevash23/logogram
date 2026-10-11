import { useCallback, useEffect, useMemo, useState } from "react";

import { api } from "../api/client";
import type { AttentionData } from "../api/types";
import { Heatmap, type Axis } from "../components/Heatmap/Heatmap";
import { ScaleBar } from "../components/Heatmap/ScaleBar";
import { Button, Callout, Empty, Field, Segmented, Select, Spinner } from "../components/ui";
import { inkScale, type ColorScale, type ResolvedTheme } from "../lib/color";
import { ci, pct, signed, visibleToken } from "../lib/format";
import { useActiveRun, useAnalysisContext, useArchitecture } from "../lib/hooks";
import { componentLabel, findSite, type Selection } from "../lib/sites";
import { useStore } from "../store/app";
import { SelectionTray } from "./ExploreView";
import s from "./views.module.css";
import h from "./HeadComparisonView.module.css";

export function HeadComparisonView() {
  const pins = useStore(st => st.headPins);
  const arch = useArchitecture();
  const context = useAnalysisContext();
  const model = useStore(st => st.model);
  const dataset = useStore(st => st.datasetPath);
  const index = useStore(st => st.promptIndex);
  const token = useStore(st => st.tokenPosition);
  const theme = useStore(st => st.theme);
  const run = useActiveRun();
  const [which, setWhich] = useState<"clean" | "corrupt">("clean");
  const [response, setResponse] = useState<{ key: string; values: AttentionData[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [keyPosition, setKeyPosition] = useState(0);
  const ready = model.state === "ready" && !!dataset && !context.error;
  const pinKey = JSON.stringify(pins.map(p => [p.layer, p.head]));
  const requestKey = JSON.stringify([context.key, dataset, index, which, pinKey]);
  const data = response?.key === requestKey ? response.values : [];
  const color = useMemo(() => inkScale(1, theme), [theme]);

  useEffect(() => {
    let cancelled = false;
    setResponse(null); setError(null); setLoading(false);
    if (!ready || !dataset || !pins.length) return;
    setLoading(true);
    Promise.all(pins.map(p => api.attention({ dataset, index, layer: p.layer, head: p.head as number, which, ...context.options })))
      .then(value => { if (!cancelled) setResponse({ key: requestKey, values: value }); }, (e: Error) => { if (!cancelled) setError(e.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [ready, dataset, index, which, context.key, pinKey]);

  const row = Math.min(token ?? (data[0]?.length ?? 1) - 1, (data[0]?.length ?? 1) - 1);
  const chooseCell = useCallback((r: number, c: number) => { useStore.getState().setTokenPosition(r); setKeyPosition(c); }, []);

  return <div className={s.view}>
    <div className={s.head}><div className={s.titleBlock}><h1 className={s.title}>Head comparison</h1><p className={s.subtitle}>Two heads, the same prompt, and a shared attention scale. Select a query token to follow what each head reads.</p></div><Segmented label="Prompt variant" value={which} onChange={setWhich} options={[{ value: "clean", label: "Clean" }, { value: "corrupt", label: "Corrupt" }]} /></div>
    <p className={s.small}>{context.label}</p>
    {!arch ? <Empty title="Load a model to compare its heads" action={<Button onClick={() => useStore.setState({ modelDialogOpen: true })}>Load a model</Button>}>You can also pin heads while exploring a saved experiment.</Empty> : <>
      {(context.error || error) && <Callout tone="error">{context.error ?? error}</Callout>}
      {model.state !== "ready" && <Callout>Saved intervention evidence is available below. <Button size="small" onClick={() => useStore.setState({ modelDialogOpen: true })}>Load the model for attention</Button></Callout>}
      {!dataset && <Callout>Choose a dataset in Experiment → Prompts to measure attention.</Callout>}
      {loading && <Spinner label="Computing attention for the pinned heads…" />}
      <div className={h.pair}>{[0, 1].map(slot => {
        const pin = pins[slot], d = data[slot];
        const site = pin ? findSite(run.sites, pin) : null;
        const evidence = site ? run.results[site.index] : null;
        const top = (d?.pattern[row] ?? []).map((weight, position) => ({ weight, position })).filter(p => p.position <= row).sort((a, b) => b.weight - a.weight || a.position - b.position).slice(0, 5);
        return <section className={h.column} key={slot} aria-label={`Head ${slot + 1}`}>
          {pin ? <>
            <div className={h.heading}><div><p className={s.faint}>Head {slot + 1}</p><h2>{componentLabel(pin)}</h2></div><div className={s.headActions}><Button size="small" onClick={() => { useStore.setState({ selection: pin, exploreMode: "layer" }); useStore.getState().setView("explore"); }}>Open layer</Button><Button size="small" variant="ghost" onClick={() => useStore.getState().pinHead(pin)}>Remove</Button></div></div>
            <div className={h.evidence}>
              <span className={s.small}>Intervention evidence · {run.detail?.spec.name ?? "No selected run"}</span>
              {evidence ? <><strong>{signed(evidence.effect.mean, 3)}<span> normalized effect</span></strong><span className={s.small}>{run.ciLevel}% CI {ci(evidence.effect.lo, evidence.effect.hi, 3)} · n = {evidence.n} · {site?.position_key === "all" ? "all positions" : `position ${site?.position_key}`}</span></> : <p className={s.small}>No single measured intervention at this selection. Choose a position in Evidence if this head has several measurements.</p>}
            </div>
            {d && <>
              <HeadAttention data={d} row={row} keyPosition={keyPosition} color={color} theme={theme} onSelect={chooseCell} label={`Attention comparison L${pin.layer} H${pin.head}`} />
              <div className={h.reading}><h3>Query {row} · {visibleToken(d.tokens[row])}</h3><p className={s.small}>Largest attention weights in this row</p>{top.map(t => <div key={t.position}><span>{t.position} · <strong>{visibleToken(d.tokens[t.position])}</strong></span><span>{pct(t.weight, 2)}</span></div>)}</div>
            </>}
            <div className={s.headActions}><Button size="small" onClick={() => useStore.getState().stageSelection(pin)}>Add to experiment</Button><Button size="small" onClick={() => { useStore.getState().select(pin); useStore.getState().setView("attention"); }}>Open prompt average</Button></div>
          </> : <HeadPicker slot={slot} nLayers={arch.nLayers} nHeads={arch.nHeads} pins={pins} />}
        </section>;
      })}</div>
      {!!data.length && <ScaleBar bound={1} theme={theme} kind="ink" label="Attention weight · shared scale" />}
      <p className={s.small}>Attention weights describe which positions a head reads. The intervention effect is a separate causal measurement from the selected run, with its own position and method.</p>
      <SelectionTray />
      <Button onClick={() => useStore.getState().setView("explore")}>Return to model explorer</Button>
    </>}
  </div>;
}

/** One pinned head's pattern. Its accessors keep their identity while a run streams progress
 * elsewhere, so the map is drawn again only for new attention data or a new query row. */
function HeadAttention({ data, row, keyPosition, color, theme, onSelect, label }: {
  data: AttentionData;
  row: number;
  keyPosition: number;
  color: ColorScale;
  theme: ResolvedTheme;
  onSelect: (r: number, c: number) => void;
  label: string;
}) {
  const axes: Axis[] = useMemo(() => data.tokens.map((t, i) => ({ key: String(i), label: `${i} ${visibleToken(t)}`, emphasis: i === row })), [data, row]);
  const value = useCallback((r: number, c: number) => (c > r ? undefined : data.pattern[r][c]), [data]);
  const tooltip = useCallback((r: number, c: number) => <span>Query {r} → key {c}: {pct(data.pattern[r][c], 2)}</span>, [data]);
  const column = Math.min(keyPosition, row);
  const selected = useMemo(() => ({ r: row, c: column }), [row, column]);
  return <Heatmap rows={axes} cols={axes} value={value} color={color} theme={theme} tokens cellMin={7} cellMax={27} rowTitle="Query" colTitle="Key" selected={selected} onSelect={onSelect} ariaLabel={label} tooltip={tooltip} />;
}

function HeadPicker({ slot, nLayers, nHeads, pins }: { slot: number; nLayers: number; nHeads: number; pins: Selection[] }) {
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(Math.min(slot, nHeads - 1));
  const duplicate = pins.some(p => p.layer === layer && p.head === head);
  return <div className={h.picker}><span className={s.faint}>Head {slot + 1}</span><h2>Choose a head</h2><p className={s.subtitle}>Pin one from the model explorer, or choose its layer and head here.</p><div className={s.row}>
    <Field label={`Layer for head ${slot + 1}`}><Select value={layer} onChange={e => setLayer(Number(e.target.value))}>{Array.from({ length: nLayers }, (_, l) => <option key={l} value={l}>Layer {l}</option>)}</Select></Field>
    <Field label={`Head for slot ${slot + 1}`}><Select value={head} onChange={e => setHead(Number(e.target.value))}>{Array.from({ length: nHeads }, (_, h) => <option key={h} value={h}>H{h}</option>)}</Select></Field>
  </div><Button variant="primary" disabled={duplicate} onClick={() => useStore.getState().pinHead({ layer, head, part: "head" })}>{duplicate ? "Already pinned" : "Pin head"}</Button></div>;
}
