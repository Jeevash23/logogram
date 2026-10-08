import { useEffect, useMemo, useState } from "react";

import { api } from "../api/client";
import type { AttentionData } from "../api/types";
import { Heatmap, type Axis } from "../components/Heatmap/Heatmap";
import { ScaleBar } from "../components/Heatmap/ScaleBar";
import { Button, Callout, Empty, Segmented, Spinner } from "../components/ui";
import { useAnalysisContext } from "../lib/hooks";
import { inkScale } from "../lib/color";
import { visibleToken } from "../lib/format";
import { useStore } from "../store/app";
import s from "./views.module.css";
import a from "./AttentionView.module.css";

export function AttentionView() {
  const selection = useStore((st) => st.selection);
  const model = useStore((st) => st.model);
  const datasetPath = useStore((st) => st.datasetPath);
  const index = useStore((st) => st.promptIndex);
  const setIndex = useStore((st) => st.setPromptIndex);
  const context = useAnalysisContext();
  const n = context.n;
  const theme = useStore((st) => st.theme);
  const token = useStore(st => st.tokenPosition);
  const [keyPosition, setKeyPosition] = useState(0);
  const [which, setWhich] = useState<"clean" | "corrupt">("clean");
  const [data, setData] = useState<AttentionData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const head = selection?.part === "head" ? { layer: selection.layer, head: selection.head as number } : null;
  const ready = model.state === "ready" && !!datasetPath && !!head && !context.error;

  useEffect(() => {
    setError(null);
    setData(null);
    setLoading(false);
    if (!ready || !head || !datasetPath) return;
    let cancelled = false;
    setLoading(true);
    api.attention({ dataset: datasetPath, index, layer: head.layer, head: head.head, which, ...context.options }).then(
      (d) => {
        if (!cancelled) {
          setData(d);
          setLoading(false);
        }
      },
      (e: Error) => {
        if (!cancelled) {
          setError(e.message);
          setLoading(false);
        }
      },
    );
    return () => {
      cancelled = true;
    };
  }, [ready, head?.layer, head?.head, datasetPath, index, which, context.key]);

  const color = useMemo(() => inkScale(1, theme), [theme]);

  if (!head) {
    return (
      <div className={s.view}>
        <Empty title="Select an attention head" action={<Button onClick={() => useStore.getState().setView("explore")}>Explore the model</Button>}>
          Click a head on the model map, or move to one with the arrow keys, to see where it attends.
        </Empty>
      </div>
    );
  }
  if (model.state !== "ready") {
    return (
      <div className={s.view}>
        <Empty
          title="Load a model to see attention"
          action={<Button onClick={() => useStore.setState({ modelDialogOpen: true })}>Load a model</Button>}
        >
          Attention patterns are computed live from the loaded model.
        </Empty>
      </div>
    );
  }

  const labelAt = new Map<number, string>();
  for (const [label, pos] of Object.entries(data?.labels ?? {})) labelAt.set(pos, label);
  const axis: Axis[] = (data?.tokens ?? []).map((t, i) => ({
    key: String(i),
    label: labelAt.has(i) ? `${visibleToken(t)} ${labelAt.get(i)}` : visibleToken(t),
    emphasis: labelAt.has(i),
  }));
  // The average mixes prompts: name a position only by what is true of all of them.
  const avgLabelAt = new Map<number, string>();
  for (const [label, pos] of Object.entries(data?.average_labels ?? {})) avgLabelAt.set(pos, label);
  const avgAxis: Axis[] = (data?.average_tokens ?? []).map((t, i) => {
    const base = t === null ? `#${i}` : visibleToken(t);
    return {
      key: String(i),
      label: avgLabelAt.has(i) ? `${base} ${avgLabelAt.get(i)}` : base,
      emphasis: avgLabelAt.has(i),
    };
  });

  const last = data ? Math.min(token ?? data.length - 1, data.length - 1) : 0;
  const chooseCell = (r: number, c: number) => { useStore.getState().setTokenPosition(r); setKeyPosition(c); };
  const topKey = (m: number[][] | undefined) => {
    if (!m) return null;
    const row = m[last];
    let best = 0;
    for (let k = 1; k < row.length; k++) if (row[k] > row[best]) best = k;
    return { pos: best, weight: row[best] };
  };
  const one = topKey(data?.pattern);
  const avg = topKey(data?.average);

  return (
    <div className={s.view}>
      <div className={s.head}>
        <div className={s.titleBlock}>
          <h2 className={s.title}>
            Attention · L{head.layer} H{head.head}
          </h2>
          <p className={s.subtitle}>
            Each row is a query position and each column a key position: how much the head at that row reads from
            that column. Rows sum to 1. Later positions are masked.
          </p>
        </div>
        <div className={s.headActions}>
          <Segmented label="Prompt" value={which} onChange={setWhich} options={[{ value: "clean", label: "Clean" }, { value: "corrupt", label: "Corrupt" }]} />
          <Button size="small" variant="ghost" icon="chevronLeft" aria-label="Previous prompt" disabled={index === 0} onClick={() => setIndex(index - 1)} />
          <span className={s.small}>
            prompt {index} of {n}
          </span>
          <Button size="small" variant="ghost" icon="chevronRight" aria-label="Next prompt" disabled={index >= n - 1} onClick={() => setIndex(index + 1)} />
          {loading && <Spinner />}
        </div>
      </div>
      <p className={s.small}>{context.label}</p>
      {context.error && <Callout tone="error" title="The loaded model differs">{context.error}</Callout>}
      {error && <p className={s.small}>{error}</p>}
      {data && (
        <>
          <p className={s.sentence}>
            At query token {last}, this head attends most to position {one?.pos}{" "}
            <strong>{visibleToken(data.tokens[one?.pos ?? 0])}</strong>
            {one && labelAt.has(one.pos) ? ` (${labelAt.get(one.pos)})` : ""} with weight{" "}
            <strong>{one?.weight.toFixed(2)}</strong> in this prompt
            {avg && (
              <>
                , and to position {avg.pos}
                {avgLabelAt.has(avg.pos) ? ` (${avgLabelAt.get(avg.pos)} in every prompt)` : ""} with weight{" "}
                <strong>{avg.weight.toFixed(2)}</strong> on average
              </>
            )}
            .
          </p>
          <div className={a.pair}>
            <div className={a.panel}>
              <div className={a.panelHead}>
                <span className={s.panelTitle}>
                  Prompt {data.index} · {data.which}
                </span>
              </div>
              <Heatmap
                rows={axis}
                cols={axis}
                value={(r, c) => (c > r ? undefined : data.pattern[r][c])}
                color={color}
                theme={theme}
                selected={token === null ? null : { r: last, c: Math.min(keyPosition, last) }}
                onSelect={chooseCell}
                tokens
                cellMin={8}
                cellMax={26}
                rowTitle="Query"
                colTitle="Key"
                tooltip={(r, c) => (
                  <>
                    <div>
                      <strong>{visibleToken(data.tokens[r])}</strong> ({r}) reads <strong>{visibleToken(data.tokens[c])}</strong> ({c})
                    </div>
                    <div>weight {data.pattern[r][c].toFixed(3)}</div>
                  </>
                )}
                ariaLabel={`Attention pattern of L${head.layer} H${head.head} on prompt ${data.index}`}
              />
            </div>
            <div className={a.panel}>
              <div className={a.panelHead}>
                <span className={s.panelTitle}>Average over {data.n_average} prompts</span>
                <span className={s.faint}>
                  {data.n_average === data.n_total
                    ? "the whole dataset"
                    : `of ${data.n_total}: those with the same token length (${data.length})`}
                </span>
              </div>
              <Heatmap
                rows={avgAxis}
                cols={avgAxis}
                value={(r, c) => (c > r ? undefined : data.average[r][c])}
                color={color}
                theme={theme}
                selected={token === null ? null : { r: last, c: Math.min(keyPosition, last) }}
                onSelect={chooseCell}
                tokens
                cellMin={8}
                cellMax={26}
                rowTitle="Query"
                colTitle="Key"
                tooltip={(r, c) => (
                  <>
                    <div>
                      position {r} reads position {c}
                    </div>
                    <div>mean weight {data.average[r][c].toFixed(3)}</div>
                  </>
                )}
                ariaLabel={`Average attention pattern of L${head.layer} H${head.head}`}
              />
            </div>
          </div>
          <ScaleBar bound={1} theme={theme} kind="ink" label="Attention weight" />
          <p className={s.small}>Attention shows which positions this head reads. Use an intervention to measure its causal contribution.</p>
          <div className={s.headActions}><Button onClick={() => selection && useStore.getState().pinHead(selection)}>Pin for comparison</Button><Button onClick={() => useStore.getState().setView("heads")}>Compare heads</Button></div>
          <p className={s.faint}>
            In the average, a position shows its token and label only when they are the same in every averaged prompt;
            other positions show their index (#2).
          </p>
        </>
      )}
    </div>
  );
}
