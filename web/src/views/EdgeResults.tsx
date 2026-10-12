import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent, type MouseEvent } from "react";

import { api } from "../api/client";
import type { EdgeCurve as CurveData, EdgeRun, EdgeVerification, SiteBase, SiteResult, Stat } from "../api/types";
import { ScaleBar } from "../components/Heatmap/ScaleBar";
import { Button, Segmented } from "../components/ui";
import { font, inkRing, prepareCanvas, useChromeColors, useElementSize } from "../lib/canvas";
import { divergingScale, niceBound, SCALE_FLOOR, type ResolvedTheme } from "../lib/color";
import {
  curveHandles,
  curvePoint,
  graphColumns,
  inputOffset,
  logTicks,
  nearestEdge,
  rankEdges,
  receiverPlace,
  senderPlace,
  type EdgeCurve,
} from "../lib/edges";
import { ci, count, num, plural, qText, signed } from "../lib/format";
import { useStore } from "../store/app";
import s from "./views.module.css";
import x from "./EdgeResults.module.css";

/** How many of the strongest edges the drawing shows, as chosen. */
const DRAWN = [25, 50, 100, 200] as const;
/** Rows of the ranked list shown at first, and added by each "Show more". */
const LIST_STEP = 25;

/**
 * An edge attribution run's results: the strongest edges drawn between the model's components
 * (layers left to right, in the effect colors), the faithfulness curve, the estimates against
 * their verification by path patching, and every listed edge ranked. Also a verification run's
 * edges (path patching, one edge at a time), which have no curve.
 */
export function EdgeResults({
  runId,
  finished,
  edges,
  sites,
  results,
  nLayers,
  nHeads,
  theme,
  effectLabel,
  ciLevel,
  selectedIndex,
  onSelect,
  receiver,
  source,
}: {
  runId: string;
  finished: boolean;
  /** The summary's account of the run (null for a verification run). */
  edges: EdgeRun | null;
  sites: SiteBase[];
  results: Record<number, SiteResult>;
  nLayers: number;
  nHeads: number;
  theme: ResolvedTheme;
  effectLabel: string;
  ciLevel: number;
  selectedIndex: number | null;
  onSelect: (site: SiteResult) => void;
  receiver: string;
  source: string;
}) {
  const ranked = useMemo(
    () => rankEdges(sites.flatMap((site) => (site.kind === "edge" && site.edge && results[site.index] ? [results[site.index]] : []))),
    [sites, results],
  );
  const [drawn, setDrawn] = useState<(typeof DRAWN)[number]>(50);
  // The drawing shows the strongest, and the selected edge wherever it ranks.
  const shown = useMemo(() => {
    const top = ranked.slice(0, drawn);
    const selected = ranked.find((e) => e.index === selectedIndex);
    return selected && !top.includes(selected) ? [...top, selected] : top;
  }, [ranked, drawn, selectedIndex]);
  const bound = useMemo(() => niceBound(shown.map((e) => e.effect.mean), SCALE_FLOOR.effect), [shown]);
  const color = useMemo(() => divergingScale(bound, theme), [bound, theme]);
  const verification = useVerification(runId, finished && !!edges);
  const measured = useMemo(() => new Map((verification?.rows ?? []).map((r) => [r.estimate_index, r])), [verification]);

  if (ranked.length === 0) return null;
  return (
    <div className={x.results}>
      {edges && <EdgeOverview edges={edges} listed={ranked.length} />}
      <section className={x.figure} aria-labelledby={`${runId}-graph`}>
        <div className={x.figureHead}>
          <h2 id={`${runId}-graph`} className={s.panelTitle}>
            The strongest edges, layers left to right
          </h2>
          <Segmented
            label="Edges drawn"
            value={String(drawn)}
            onChange={(v) => setDrawn(Number(v) as (typeof DRAWN)[number])}
            options={DRAWN.map((n) => ({ value: String(n), label: count(n), disabled: n > ranked.length && n !== DRAWN[0] }))}
          />
          <ScaleBar bound={bound} theme={theme} label={effectLabel} />
        </div>
        <EdgeGraph
          edges={shown}
          nLayers={nLayers}
          nHeads={nHeads}
          color={color}
          selectedIndex={selectedIndex}
          onSelect={onSelect}
          effectLabel={effectLabel}
          ciLevel={ciLevel}
        />
        <p className={s.small}>
          Heads are dots, head 0 at the top, and MLPs squares; the embeddings are on the left and the logits on the right. An
          edge ends just above a head at its query, at it for its key, and just below it for its value. Wider and deeper in color
          is stronger; where edges cross, the stronger is on top.
        </p>
      </section>
      {(edges?.curve || verification) && (
        <div className={x.pair}>
          {edges?.curve && <CurveChart curve={edges.curve} nEdges={edges.n_edges} ciLevel={ciLevel} receiver={receiver} source={source} />}
          {verification && <VerificationTable data={verification} ciLevel={ciLevel} selectedIndex={selectedIndex} onSelect={(i) => {
            const site = ranked.find((e) => e.index === i);
            if (site) onSelect(site);
          }} />}
        </div>
      )}
      <EdgeList ranked={ranked} measured={measured} effectLabel={effectLabel} ciLevel={ciLevel} selectedIndex={selectedIndex} onSelect={onSelect} total={edges?.n_edges ?? ranked.length} />
    </div>
  );
}

/** How many edges were estimated, how, and what was measured about the model first. */
function EdgeOverview({ edges, listed }: { edges: EdgeRun; listed: number }) {
  const checks = edges.checks;
  return (
    <div className={x.overview}>
      <p className={x.overviewText}>
        {count(edges.n_edges)} edges estimated, from {count(edges.n_senders)} senders into {count(edges.n_receivers)} inputs
        {edges.method === "integrated_gradients" ? `, with integrated gradients over ${edges.steps} steps` : ", from one gradient each"}.{" "}
        {listed < edges.n_edges
          ? `The strongest ${count(listed)} are listed here; Export has every edge as a CSV, and edges.parquet in the run's folder holds them all.`
          : "Every edge is listed here."}
      </p>
      <details className={x.more}>
        <summary>How the estimates were taken</summary>
        <p className={s.small}>
          Each edge&rsquo;s estimate is the sender&rsquo;s output on the source prompt minus on the receiver prompt, dotted with the
          gradient of the metric {edges.gradient}, summed over the positions. It is a first-order estimate of patching that edge
          alone; verify the strongest by path patching.
        </p>
        <p className={s.small}>
          Measured on the model first, as the edges rely on it: giving each receiver its own copy of the residual stream
          reproduces the model&rsquo;s output{checks.split !== undefined ? ` (to within ${sci(checks.split)}, relative)` : ""};
          changing one head&rsquo;s copy changes no other head&rsquo;s query, key or value
          {checks.separate !== undefined ? ` (to within ${sci(checks.separate)})` : ""}; and each copy is the residual stream
          itself{checks.stream !== undefined ? ` (to within ${sci(checks.stream)})` : ""}, which each receiver normalizes on its
          own, recomputing the scale from its own input.{" "}
          {edges.structure === "parallel"
            ? "In this model a layer's MLP reads the residual stream its heads read, so a layer's heads have no edge into its own MLP."
            : "A layer's MLP reads what its heads write, so a layer's heads have an edge into its own MLP."}
        </p>
      </details>
    </div>
  );
}

/** The faithfulness axis's labels: 0 and 1, then the ends and 0.5 where they don't crowd them. */
function yTicks(y0: number, y1: number, yAt: (v: number) => number): number[] {
  const out: number[] = [];
  for (const v of [0, 1, y0, y1, 0.5]) {
    if (v < y0 || v > y1 || out.some((u) => Math.abs(yAt(u) - yAt(v)) < 14)) continue;
    out.push(v);
  }
  return out;
}

const sci = (v: number) => (v === 0 ? "0" : Math.abs(v) >= 0.01 ? num(v, 3) : v.toExponential(1).replace("-", "−"));

/** The latest finished verification of an edge attribution run by path patching, fetched again
 * when one finishes. */
function useVerification(runId: string, enabled: boolean): EdgeVerification | null {
  // Changes when a verification of this run starts or ends.
  const key = useStore((st) =>
    st.runs
      .filter((r) => r.derived_from?.run === runId && r.derived_from.kind === "verification")
      .map((r) => `${r.id}:${r.status}`)
      .join(","),
  );
  const [data, setData] = useState<EdgeVerification | null>(null);
  useEffect(() => {
    setData(null);
    if (!enabled) return;
    let cancelled = false;
    api.edgeVerification(runId).then(
      (v) => !cancelled && setData(v),
      () => !cancelled && setData(null),
    );
    return () => {
      cancelled = true;
    };
  }, [runId, enabled, key]);
  return data;
}

/**
 * The strongest edges drawn on canvas between the model's components: the embeddings, each
 * layer's heads (dots, head 0 at the top) and MLP (a square), and the logits, left to right.
 * Edges are drawn in the effect colors, wider when stronger, the strongest first: each weaker
 * edge goes beneath those already drawn. The selected edge is circled in ink. Arrow keys move
 * through the edges, strongest first.
 */
function EdgeGraph({
  edges,
  nLayers,
  nHeads,
  color,
  selectedIndex,
  onSelect,
  effectLabel,
  ciLevel,
}: {
  edges: SiteResult[];
  nLayers: number;
  nHeads: number;
  color: ReturnType<typeof divergingScale>;
  selectedIndex: number | null;
  onSelect: (site: SiteResult) => void;
  effectLabel: string;
  ciLevel: number;
}) {
  const colors = useChromeColors();
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const { width } = useElementSize(wrapRef);
  const [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(null);
  const [focused, setFocused] = useState(false);
  const helpId = useId();
  const selected = edges.findIndex((e) => e.index === selectedIndex);

  const geometry = useMemo(() => {
    if (width === 0) return null;
    const columns = graphColumns(nLayers);
    const left = 30;
    const right = 34;
    const top = 12;
    const bottom = 30;
    const w = Math.max(width, left + right + (columns - 1) * 14);
    const rowH = Math.max(9, Math.min(18, 280 / Math.max(1, nHeads)));
    const rowsH = Math.max(150, nHeads * rowH);
    const step = (w - left - right) / Math.max(1, columns - 1);
    const xAt = (column: number) => left + column * step;
    const yAt = (head: number | null) => (head === null ? top + rowsH / 2 : top + (head + 0.5) * (rowsH / Math.max(1, nHeads)));
    const offset = Math.min(4, rowsH / Math.max(1, nHeads) / 4);
    const curves: EdgeCurve[] = edges.map((site) => {
      const e = site.edge!;
      const a = senderPlace(e.sender);
      const b = receiverPlace(e.receiver, nLayers);
      return { x1: xAt(a.column), y1: yAt(a.head), x2: xAt(b.column), y2: yAt(b.head) + inputOffset(e.receiver) * offset };
    });
    return { w, h: top + rowsH + bottom, top, rowsH, step, xAt, yAt, curves, columns };
  }, [width, nLayers, nHeads, edges]);

  const widthOf = (v: number | null) => 1 + 3 * Math.min(1, Math.abs(v ?? 0) / color.max);

  // The edges, then the nodes over them, drawn again only when they or the theme change.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !geometry) return;
    const ctx = prepareCanvas(canvas, geometry.w, geometry.h);
    if (!ctx) return;
    ctx.lineCap = "round";
    // The strongest first: with destination-over, each weaker edge goes beneath the stronger
    // ones already drawn, so they stay on top where edges cross.
    ctx.globalCompositeOperation = "destination-over";
    edges.forEach((site, i) => {
      const c = geometry.curves[i];
      const [c1x, c1y, c2x, c2y] = curveHandles(c);
      ctx.beginPath();
      ctx.moveTo(c.x1, c.y1);
      ctx.bezierCurveTo(c1x, c1y, c2x, c2y, c.x2, c.y2);
      ctx.strokeStyle = color(site.effect.mean);
      ctx.lineWidth = widthOf(site.effect.mean);
      ctx.stroke();
    });
    ctx.globalCompositeOperation = "source-over";
    // Which nodes the drawn edges touch: drawn in ink, the rest faintly.
    const touched = new Set<string>();
    for (const site of edges) {
      const e = site.edge!;
      const a = senderPlace(e.sender);
      const b = receiverPlace(e.receiver, nLayers);
      touched.add(`${a.column}:${a.head}`);
      touched.add(`${b.column}:${b.head}`);
    }
    const dot = (column: number, head: number | null, r: number) => {
      ctx.beginPath();
      ctx.arc(geometry.xAt(column), geometry.yAt(head), r, 0, Math.PI * 2);
      ctx.fillStyle = touched.has(`${column}:${head}`) ? colors.text : colors.faint;
      ctx.fill();
    };
    dot(0, null, 4);
    dot(geometry.columns - 1, null, 4);
    for (let layer = 0; layer < nLayers; layer++) {
      for (let h = 0; h < nHeads; h++) dot(1 + 2 * layer, h, 2);
      const mx = geometry.xAt(2 + 2 * layer);
      const my = geometry.yAt(null);
      ctx.fillStyle = touched.has(`${2 + 2 * layer}:null`) ? colors.text : colors.faint;
      ctx.fillRect(mx - 3, my - 3, 6, 6);
    }
    // Labels under the columns: every layer when there is room, otherwise every few.
    ctx.font = font(colors, 10.5, 500);
    ctx.fillStyle = colors.muted;
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    const y = geometry.top + geometry.rowsH + 20;
    const every = Math.max(1, Math.ceil(26 / (2 * geometry.step)));
    for (let layer = 0; layer < nLayers; layer += every) {
      ctx.fillText(`L${layer}`, (geometry.xAt(1 + 2 * layer) + geometry.xAt(2 + 2 * layer)) / 2, y);
    }
    ctx.textAlign = "left";
    ctx.fillText("emb", Math.max(0, geometry.xAt(0) - 10), y);
    ctx.textAlign = "right";
    ctx.fillText("logits", geometry.w - 2, y);
  }, [geometry, edges, color, colors, nLayers, nHeads]);

  // Hover and selection, above: the edge cased in ink, and the selected one circled.
  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas || !geometry) return;
    const ctx = prepareCanvas(canvas, geometry.w, geometry.h);
    if (!ctx) return;
    ctx.lineCap = "round";
    const trace = (i: number) => {
      const c = geometry.curves[i];
      const [c1x, c1y, c2x, c2y] = curveHandles(c);
      ctx.beginPath();
      ctx.moveTo(c.x1, c.y1);
      ctx.bezierCurveTo(c1x, c1y, c2x, c2y, c.x2, c.y2);
    };
    const emphasize = (i: number, casing: string) => {
      const value = edges[i]?.effect.mean ?? null;
      trace(i);
      ctx.strokeStyle = casing;
      ctx.lineWidth = widthOf(value) + 3;
      ctx.stroke();
      trace(i);
      ctx.strokeStyle = color(value);
      ctx.lineWidth = widthOf(value);
      ctx.stroke();
    };
    if (hover && hover.i !== selected && hover.i < edges.length) emphasize(hover.i, colors.muted);
    if (selected >= 0) {
      emphasize(selected, colors.text);
      const mid = curvePoint(geometry.curves[selected], 0.5);
      inkRing(ctx, mid.x - 6, mid.y - 6, 12, 12, colors.text, colors.bg);
    }
  }, [geometry, hover, selected, edges, color, colors]);

  const hit = (event: MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas || !geometry) return null;
    const rect = canvas.getBoundingClientRect();
    return nearestEdge(geometry.curves, event.clientX - rect.left, event.clientY - rect.top, 6);
  };

  const onKeyDown = (event: KeyboardEvent) => {
    const keys = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"];
    if (!keys.includes(event.key) || edges.length === 0) return;
    event.preventDefault();
    const at = selected < 0 ? -1 : selected;
    const next =
      event.key === "Home"
        ? 0
        : event.key === "End"
          ? edges.length - 1
          : event.key === "ArrowUp" || event.key === "ArrowLeft"
            ? Math.max(0, at - 1)
            : Math.min(edges.length - 1, at + 1);
    onSelect(edges[next]);
  };

  const active = selected >= 0 ? edges[selected] : null;
  const hovered = hover ? edges[hover.i] : null;
  return (
    <div className={x.graph} ref={wrapRef}>
      <div className={x.scroll}>
        <div className={x.stage}>
          <canvas
            ref={canvasRef}
            className={x.canvas}
            tabIndex={0}
            role="group"
            aria-roledescription="interactive graph"
            aria-label={`The ${plural(edges.length, "strongest edge", "strongest edges")}, layers left to right`}
            aria-describedby={helpId}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            onMouseMove={(event) => {
              const i = hit(event);
              setHover(i === null ? null : { i, x: event.clientX, y: event.clientY });
            }}
            onMouseLeave={() => setHover(null)}
            onClick={(event) => {
              const i = hit(event);
              if (i !== null) onSelect(edges[i]);
            }}
            onKeyDown={onKeyDown}
            style={{ cursor: hover ? "pointer" : "default" }}
          />
          <canvas ref={overlayRef} className={x.overlay} aria-hidden="true" />
        </div>
      </div>
      <p id={helpId} className={x.help}>
        Arrow keys move through the edges, strongest first; Home and End go to the strongest and the weakest drawn.
      </p>
      <div className={x.readout} aria-live="polite" aria-atomic="true">
        {focused && active && (
          <>
            Edge {selected + 1} of {edges.length}: {active.label}. {effectLabel} {signed(active.effect.mean, 3)}, {ciLevel}% CI{" "}
            {ci(active.effect.lo, active.effect.hi, 3)}.
          </>
        )}
      </div>
      {hover && hovered && (
        <div className={x.tooltip} style={{ left: hover.x + 14, top: hover.y + 14 }}>
          <div className={x.tooltipTitle}>{hovered.label}</div>
          <div>
            {effectLabel} <strong>{signed(hovered.effect.mean, 3)}</strong>
          </div>
          <div className={x.tooltipFaint}>
            {ciLevel}% CI {ci(hovered.effect.lo, hovered.effect.hi, 3)}
            {hovered.q !== null && hovered.q !== undefined && <> · q {qText(hovered.q)}</>}
          </div>
        </div>
      )}
    </div>
  );
}

/** Faithfulness against the number of edges kept, on a log scale: the strongest edges in ink,
 * edges chosen at random dashed, each with its interval as a band. */
function CurveChart({ curve, nEdges, ciLevel, receiver, source }: { curve: CurveData; nEdges: number; ciLevel: number; receiver: string; source: string }) {
  const host = useRef<HTMLDivElement>(null);
  const { width } = useElementSize(host);
  const titleId = useId();
  const points = curve.strongest.map((p, i) => ({ k: p.k, strong: p.faithfulness, random: curve.random[i]?.faithfulness ?? null }));
  if (points.length === 0) {
    return (
      <figure className={x.chart}>
        <figcaption className={x.chartTitle}>Faithfulness against the number of edges</figcaption>
        <p className={s.small}>Every number of edges asked for was larger than the graph, so the curve has no points.</p>
      </figure>
    );
  }
  const ks = points.map((p) => p.k);
  const kLo = Math.min(...ks);
  const kHi = Math.max(...ks);
  const lo = kLo === kHi ? kLo / 3 : kLo;
  const hi = kLo === kHi ? kHi * 3 : kHi;
  const values = [0, 1];
  for (const p of points) {
    for (const f of [p.strong, p.random]) {
      for (const v of [f?.lo, f?.hi, f?.mean]) if (v !== null && v !== undefined && Number.isFinite(v)) values.push(v);
    }
  }
  const y0 = Math.min(...values);
  const y1 = Math.max(...values);
  const W = Math.max(280, Math.min(width, 560));
  const H = 220;
  const L = 44;
  const R = 14;
  const T = 12;
  const B = 40;
  const xAt = (k: number) => L + ((Math.log10(k) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo) || 1)) * (W - L - R);
  const yAt = (v: number) => T + (1 - (v - y0) / (y1 - y0 || 1)) * (H - T - B);
  const ticks = logTicks(Math.max(1, Math.ceil(lo)), Math.floor(hi));
  const line = (pick: (p: (typeof points)[number]) => Stat | null) =>
    points
      .flatMap((p) => {
        const m = pick(p)?.mean;
        return m === null || m === undefined || !Number.isFinite(m) ? [] : [`${xAt(p.k)},${yAt(m)}`];
      })
      .join(" ");
  const band = (pick: (p: (typeof points)[number]) => Stat | null) => {
    const usable = points.filter((p) => pick(p)?.lo != null && pick(p)?.hi != null);
    if (usable.length < 2) return null;
    const upper = usable.map((p) => `${xAt(p.k)},${yAt(pick(p)!.hi as number)}`);
    const lower = usable.reverse().map((p) => `${xAt(p.k)},${yAt(pick(p)!.lo as number)}`);
    return [...upper, ...lower].join(" ");
  };
  const strongBand = band((p) => p.strong);
  const randomBand = band((p) => p.random);
  const description = points
    .map((p) => `${count(p.k)} edges: ${num(p.strong.mean, 2)}, random ${num(p.random?.mean, 2)}`)
    .join("; ");
  const every = curve.every.faithfulness.mean;
  return (
    <figure className={x.chart} ref={host} aria-labelledby={titleId}>
      <figcaption id={titleId} className={x.chartTitle}>
        Faithfulness against the number of edges
      </figcaption>
      {width > 0 && (
        <svg width={W} height={H} role="img" aria-label={`Faithfulness of the strongest edges and of edges chosen at random, with ${ciLevel}% intervals. ${description}.`}>
          <line className={x.axis} x1={L} x2={W - R} y1={H - B} y2={H - B} />
          {[0, 1].map((v) => (
            <line key={v} className={x.reference} x1={L} x2={W - R} y1={yAt(v)} y2={yAt(v)} />
          ))}
          {yTicks(y0, y1, yAt).map((v) => (
              <text key={v} className={x.tick} x={L - 8} y={yAt(v)} textAnchor="end" dominantBaseline="middle">
                {num(v, v === 0 || v === 1 ? 0 : 1)}
              </text>
            ))}
          {ticks.map((k) => (
            <text key={k} className={x.tick} x={xAt(k)} y={H - B + 16} textAnchor="middle">
              {count(k)}
            </text>
          ))}
          <text className={x.axisLabel} x={(L + W - R) / 2} y={H - 6} textAnchor="middle">
            Edges kept (log scale)
          </text>
          <text className={x.axisLabel} x={-((T + H - B) / 2)} y={11} textAnchor="middle" transform="rotate(-90)">
            Faithfulness
          </text>
          {randomBand && <polygon className={x.randomBand} points={randomBand} />}
          {strongBand && <polygon className={x.strongBand} points={strongBand} />}
          <polyline className={x.randomLine} points={line((p) => p.random)} />
          <polyline className={x.strongLine} points={line((p) => p.strong)} />
          {points.map((p) =>
            p.random?.mean != null ? <circle key={`r${p.k}`} className={x.randomDot} cx={xAt(p.k)} cy={yAt(p.random.mean)} r={3} /> : null,
          )}
          {points.map((p) =>
            p.strong.mean != null ? (
              <circle
                key={`s${p.k}`}
                className={x.strongDot}
                cx={xAt(p.k)}
                cy={yAt(p.strong.mean)}
                r={3.5}
                tabIndex={0}
                aria-label={`The ${plural(p.k, "strongest edge", "strongest edges")}: faithfulness ${num(p.strong.mean, 2)}, ${ci(p.strong.lo, p.strong.hi)}; edges chosen at random: ${num(p.random?.mean, 2)}`}
              >
                <title>{`${count(p.k)} strongest: ${num(p.strong.mean, 3)} (${ci(p.strong.lo, p.strong.hi, 3)}); random ${num(p.random?.mean, 3)} (${ci(p.random?.lo, p.random?.hi, 3)})`}</title>
              </circle>
            ) : null,
          )}
        </svg>
      )}
      <div className={x.key} aria-hidden="true">
        <span>
          <i className={x.keyStrong} /> The strongest edges
        </span>
        <span>
          <i className={x.keyRandom} /> Edges chosen at random ({plural(curve.draws, "draw")}, seed {curve.seed})
        </span>
      </div>
      <p className={s.small}>
        {curve.patch === "chosen"
          ? `The ${receiver} prompts ran with only the chosen edges carrying the ${source} prompts' values.`
          : `The ${receiver} prompts ran with every edge but the chosen carrying the ${source} prompts' values.`}{" "}
        Faithfulness is the share of the gap from no edge chosen (0) to every edge chosen (1) the chosen edges close, with{" "}
        {ciLevel}% intervals as bands.{" "}
        {every !== null && every !== undefined
          ? `Patching every edge of the graph gave ${num(every, 3)}${curve.patch === "chosen" ? " (it should be 1)" : " (it should be 0)"}${curve.source_difference != null ? `, the ${source} run's metric to within ${sci(curve.source_difference)}` : ""}.`
          : ""}
        {curve.dropped.length > 0 ? ` Left out, being more than the graph's ${count(nEdges)} edges: ${curve.dropped.map(count).join(", ")}.` : ""}
      </p>
    </figure>
  );
}

/** An edge attribution run's estimates against its latest verification by path patching, for
 * the comparison of the two runs; nothing until the verification has finished. */
export function EdgeVerificationPanel({
  runId,
  ciLevel,
  selectedIndex,
  onSelect,
}: {
  runId: string;
  ciLevel: number;
  selectedIndex: number | null;
  onSelect: (estimateIndex: number) => void;
}) {
  const data = useVerification(runId, true);
  return data ? <VerificationTable data={data} ciLevel={ciLevel} selectedIndex={selectedIndex} onSelect={onSelect} /> : null;
}

/** Each verified edge's estimate beside its measurement by path patching, ranked both ways, and
 * how well they agree. */
function VerificationTable({
  data,
  ciLevel,
  selectedIndex,
  onSelect,
}: {
  data: EdgeVerification;
  ciLevel: number;
  selectedIndex: number | null;
  onSelect: (estimateIndex: number) => void;
}) {
  const titleId = useId();
  return (
    <section className={x.verification} aria-labelledby={titleId}>
      <h2 id={titleId} className={s.panelTitle}>
        Estimates against path patching
      </h2>
      <p className={s.small}>
        The {plural(data.n, "strongest estimated edge", "strongest estimated edges")}, each path-patched alone with every other
        head and MLP held.{" "}
        {data.spearman !== null ? `Spearman's rank correlation of the two: ${num(data.spearman, 2)}. ` : ""}
        {count(data.same_sign)} of {count(data.n)} have the same sign.
      </p>
      <div className={x.tableScroll}>
        <table className={`${s.table} ${x.table}`}>
          <caption className="visually-hidden">Each verified edge: its estimate and its effect by path patching, with {ciLevel}% intervals and ranks</caption>
          <thead>
            <tr>
              <th scope="col">Edge</th>
              <th scope="col" className={s.num}>Estimate</th>
              <th scope="col" className={s.num}>Rank</th>
              <th scope="col" className={s.num}>Path patching</th>
              <th scope="col" className={s.num}>Rank</th>
            </tr>
          </thead>
          <tbody>
            {data.rows.map((row) => {
              const differs = (row.estimate.mean ?? 0) * (row.measured.mean ?? 0) < 0;
              return (
                <tr key={row.label} aria-selected={row.estimate_index === selectedIndex}>
                  <th scope="row">
                    <button type="button" className={x.edgeButton} onClick={() => onSelect(row.estimate_index)} aria-pressed={row.estimate_index === selectedIndex}>
                      {row.label}
                    </button>
                  </th>
                  <td className={s.num} title={ci(row.estimate.lo, row.estimate.hi, 3)}>
                    {signed(row.estimate.mean, 3)}
                  </td>
                  <td className={s.num}>{row.estimate_rank ?? "—"}</td>
                  <td className={s.num} title={ci(row.measured.lo, row.measured.hi, 3)}>
                    {signed(row.measured.mean, 3)}
                    {differs && <span className={x.flag}> other sign</span>}
                  </td>
                  <td className={s.num}>{row.measured_rank ?? "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/** Every listed edge, strongest first, a page at a time: its estimate with its interval, its
 * change in the metric, its q-value, and its measurement by path patching where it was
 * verified. */
function EdgeList({
  ranked,
  measured,
  effectLabel,
  ciLevel,
  selectedIndex,
  onSelect,
  total,
}: {
  ranked: SiteResult[];
  measured: Map<number, EdgeVerification["rows"][number]>;
  effectLabel: string;
  ciLevel: number;
  selectedIndex: number | null;
  onSelect: (site: SiteResult) => void;
  total: number;
}) {
  const [shown, setShown] = useState(LIST_STEP);
  // The selected edge is shown wherever it ranks.
  const at = ranked.findIndex((e) => e.index === selectedIndex);
  useEffect(() => {
    if (at >= shown) setShown(Math.ceil((at + 1) / LIST_STEP) * LIST_STEP);
  }, [at, shown]);
  const rows = ranked.slice(0, shown);
  const ext = rows.flatMap((r) => [r.effect.lo ?? r.effect.mean ?? 0, r.effect.hi ?? r.effect.mean ?? 0, 0]);
  const lo = Math.min(...ext);
  const hi = Math.max(...ext);
  const pos = (v: number) => ((v - lo) / (hi - lo || 1)) * 100;
  const titleId = useId();
  return (
    <section className={x.list} aria-labelledby={titleId}>
      <h2 id={titleId} className={s.panelTitle}>
        Edges, strongest first
      </h2>
      <div className={x.tableScroll}>
        <table className={`${s.table} ${x.table}`}>
          <caption className="visually-hidden">The listed edges, strongest first, with {ciLevel}% intervals</caption>
          <thead>
            <tr>
              <th scope="col" className={s.num}>#</th>
              <th scope="col">Edge</th>
              <th scope="col">
                {effectLabel} · {ciLevel}% CI
              </th>
              <th scope="col" className={s.num}>q</th>
              {measured.size > 0 && (
                <th scope="col" className={s.num}>
                  Path patching
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {rows.map((site, i) => {
              const m = measured.get(site.index);
              return (
                <tr key={site.index} aria-selected={site.index === selectedIndex}>
                  <td className={s.num}>{i + 1}</td>
                  <th scope="row">
                    <button type="button" className={x.edgeButton} onClick={() => onSelect(site)} aria-pressed={site.index === selectedIndex}>
                      {site.label}
                    </button>
                  </th>
                  <td>
                    <span className={x.interval}>
                      <span className={x.value}>{signed(site.effect.mean, 3)}</span>
                      <span className={x.track} title={`${signed(site.effect.mean, 3)} (${ci(site.effect.lo, site.effect.hi, 3)})`} aria-label={`${ciLevel}% CI ${ci(site.effect.lo, site.effect.hi, 3)}`} role="img">
                        <span className={x.zero} style={{ left: `${pos(0)}%` }} />
                        {site.effect.lo !== null && site.effect.hi !== null && (
                          <span className={x.ci} style={{ left: `${pos(site.effect.lo)}%`, width: `${Math.max(0.5, pos(site.effect.hi) - pos(site.effect.lo))}%` }} />
                        )}
                        <span className={x.point} style={{ left: `${pos(site.effect.mean ?? 0)}%` }} />
                      </span>
                    </span>
                  </td>
                  <td className={s.num}>{site.q !== null && site.q !== undefined ? qText(site.q) : "—"}</td>
                  {measured.size > 0 && <td className={s.num}>{m ? signed(m.measured.mean, 3) : ""}</td>}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className={x.listFooter}>
        <span className={s.small}>
          {count(rows.length)} of {count(ranked.length)} listed{ranked.length < total ? `, the strongest of ${count(total)} edges` : ""}.
        </span>
        {shown < ranked.length && (
          <Button size="small" variant="ghost" onClick={() => setShown(shown + LIST_STEP)}>
            Show {count(Math.min(LIST_STEP, ranked.length - shown))} more
          </Button>
        )}
      </div>
    </section>
  );
}
