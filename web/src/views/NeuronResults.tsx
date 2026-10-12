import { useEffect, useMemo, useRef, useState } from "react";

import type { Layout, SiteBase, SiteResult } from "../api/types";
import { Button } from "../components/ui";
import { font, prepareCanvas, useChromeColors, useElementSize } from "../lib/canvas";
import type { ColorScale } from "../lib/color";
import { count, num, plural, signed } from "../lib/format";
import { siteValue } from "../lib/hooks";
import { Forest } from "./Forest";
import n from "./NeuronResults.module.css";

/** Bins of a layer's distribution of neuron effects, over the scale's −bound…+bound. */
const BINS = 48;
const HIST_HEIGHT = 76;
/** The strongest neurons a layer lists, and how many "Show more" lists. */
const LISTED = 10;
const LISTED_MORE = 50;

/**
 * A run over every neuron of some layers. Thousands of neurons don't fit a map, so each layer
 * shows the distribution of all its neurons' effects (bars in the effect colors, on the same
 * scale as the legend above, their heights on a square-root scale so a few strong neurons stay
 * visible beside thousands near zero) and its strongest neurons, ranked, with their intervals.
 */
export function NeuronResults({
  layout,
  sites,
  results,
  metric,
  color,
  bound,
  ciLevel,
  selectedIndex,
  onSelect,
  flagged,
  running,
  partial,
}: {
  layout: Layout;
  sites: SiteBase[];
  results: Record<number, SiteResult>;
  metric: "effect" | "delta";
  color: ColorScale;
  bound: number;
  ciLevel: number;
  selectedIndex: number | null;
  onSelect: (site: SiteResult) => void;
  flagged: Set<number>;
  running: boolean;
  partial: boolean;
}) {
  // A run's sites by row (layer), built once per run.
  const byRow = useMemo(() => {
    const rows = new Map<number, SiteBase[]>();
    for (const site of sites) {
      const list = rows.get(site.row);
      if (list) list.push(site);
      else rows.set(site.row, [site]);
    }
    return rows;
  }, [sites]);
  return (
    <div className={n.layers}>
      {layout.rows.map((row, r) => (
        <NeuronLayer
          key={row.key}
          layer={row.label}
          total={row.neurons ?? byRow.get(r)?.length ?? 0}
          sites={byRow.get(r) ?? []}
          results={results}
          metric={metric}
          color={color}
          bound={bound}
          ciLevel={ciLevel}
          selectedIndex={selectedIndex}
          onSelect={onSelect}
          flagged={flagged}
          waiting={running ? "Waiting for this layer" : partial ? "Not measured: the run stopped before this layer" : "No results"}
        />
      ))}
    </div>
  );
}

function NeuronLayer({
  layer,
  total,
  sites,
  results,
  metric,
  color,
  bound,
  ciLevel,
  selectedIndex,
  onSelect,
  flagged,
  waiting,
}: {
  layer: string;
  total: number;
  sites: SiteBase[];
  results: Record<number, SiteResult>;
  metric: "effect" | "delta";
  color: ColorScale;
  bound: number;
  ciLevel: number;
  selectedIndex: number | null;
  onSelect: (site: SiteResult) => void;
  flagged: Set<number>;
  waiting: string;
}) {
  const [more, setMore] = useState(false);
  // This layer's results; the same array while nothing of this layer changes.
  const measured = useMemo(
    () => sites.map((s) => results[s.index]).filter((x): x is SiteResult => !!x && x.effect.mean !== null),
    [sites, results],
  );
  const values = useMemo(
    () => measured.map((x) => siteValue(x, metric)).filter((v): v is number => v !== null && Number.isFinite(v)),
    [measured, metric],
  );
  const above = measured.filter((x) => (x.effect.lo ?? 0) > 0).length;
  const below = measured.filter((x) => (x.effect.hi ?? 0) < 0).length;
  const selected = selectedIndex === null ? undefined : measured.find((x) => x.index === selectedIndex);
  const selectedValue = selected ? siteValue(selected, metric) : null;
  const titleId = `neurons-layer-${layer}`;
  return (
    <section className={n.layer} aria-labelledby={titleId}>
      <div className={n.head}>
        <h2 id={titleId} className={n.title}>
          Layer {layer}
        </h2>
        <span className={n.counts}>
          {measured.length === 0
            ? `${count(total)} neurons`
            : `${count(measured.length)} of ${plural(total, "neuron")} · ${count(above)} with intervals above zero · ${count(below)} below`}
        </span>
      </div>
      {measured.length === 0 ? (
        <p className={n.waiting}>{waiting}</p>
      ) : (
        <div className={n.body}>
          <div className={n.distribution}>
            <div className={n.caption}>Every neuron's {metric === "effect" ? "mean effect" : "mean change"}</div>
            <Histogram values={values} color={color} bound={bound} marked={selectedValue} label={`Layer ${layer}: distribution of ${plural(values.length, "neuron")}`} />
            <p className={n.note}>Bar heights grow with the square root of the number of neurons, so a few strong ones stay visible.</p>
          </div>
          <div className={n.ranked}>
            <Forest
              results={measured}
              selectedIndex={selectedIndex}
              onSelect={onSelect}
              flagged={flagged}
              limit={more ? LISTED_MORE : LISTED}
              ciLevel={ciLevel}
            />
            {measured.length > LISTED && (
              <Button size="small" variant="ghost" className={n.more} onClick={() => setMore(!more)} aria-expanded={more}>
                {more ? `Show the strongest ${LISTED}` : `Show the strongest ${Math.min(LISTED_MORE, measured.length)}`}
              </Button>
            )}
          </div>
        </div>
      )}
    </section>
  );
}

/** A distribution drawn on canvas: bins over −bound…+bound in the effect colors, with zero and the
 * scale's ends marked, and the selected neuron's value marked in ink. */
function Histogram({
  values,
  color,
  bound,
  marked,
  label,
}: {
  values: number[];
  color: ColorScale;
  bound: number;
  marked: number | null;
  label: string;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const { width } = useElementSize(wrap);
  const colors = useChromeColors();
  const counts = useMemo(() => {
    const bins = new Array<number>(BINS).fill(0);
    const m = bound > 0 ? bound : 1;
    for (const v of values) {
      const t = (Math.max(-m, Math.min(m, v)) + m) / (2 * m);
      bins[Math.min(BINS - 1, Math.floor(t * BINS))] += 1;
    }
    return bins;
  }, [values, bound]);
  useEffect(() => {
    const el = canvas.current;
    if (!el || width === 0) return;
    const height = HIST_HEIGHT + 18;
    const ctx = prepareCanvas(el, width, height);
    if (!ctx) return;
    const m = bound > 0 ? bound : 1;
    const tallest = Math.sqrt(Math.max(1, ...counts));
    const step = width / BINS;
    for (let b = 0; b < BINS; b++) {
      if (!counts[b]) continue;
      const h = Math.max(1, (Math.sqrt(counts[b]) / tallest) * (HIST_HEIGHT - 6));
      const x = b * step + 0.5;
      const w = Math.max(1, step - 1);
      ctx.fillStyle = color(-m + ((b + 0.5) * 2 * m) / BINS);
      ctx.fillRect(x, HIST_HEIGHT - h, w, h);
      // The scale's neutral is close to the page in both themes: a faint outline keeps the many
      // neurons near zero visible.
      ctx.globalAlpha = 0.55;
      ctx.strokeStyle = colors.faint;
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, HIST_HEIGHT - h + 0.5, Math.max(0, w - 1), Math.max(0, h - 1));
      ctx.globalAlpha = 1;
    }
    // The baseline, zero, and the scale's ends.
    ctx.fillStyle = colors.line;
    ctx.fillRect(0, HIST_HEIGHT, width, 1);
    ctx.fillStyle = colors.text;
    ctx.fillRect(Math.round(width / 2), 2, 1, HIST_HEIGHT - 2);
    ctx.font = font(colors, 10, 500);
    ctx.fillStyle = colors.faint;
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left";
    ctx.fillText(num(-m, m < 1 ? 2 : 1), 0, height - 3);
    ctx.textAlign = "center";
    ctx.fillText("0", width / 2, height - 3);
    ctx.textAlign = "right";
    ctx.fillText(num(m, m < 1 ? 2 : 1), width, height - 3);
    if (marked !== null && Number.isFinite(marked)) {
      // The selected neuron, circled in ink above its bar.
      const x = ((Math.max(-m, Math.min(m, marked)) + m) / (2 * m)) * width;
      ctx.strokeStyle = colors.text;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(Math.min(width - 4, Math.max(4, x)), 5, 3.5, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.fillStyle = colors.text;
      ctx.fillRect(Math.round(Math.min(width - 4, Math.max(4, x))), 9, 1, HIST_HEIGHT - 9);
    }
  }, [width, counts, color, bound, colors, marked]);
  const nonzero = counts.reduce((a, c) => a + (c > 0 ? 1 : 0), 0);
  const spread = values.length ? `from ${signed(Math.min(...values), 2)} to ${signed(Math.max(...values), 2)}` : "";
  return (
    <div ref={wrap} className={n.histogram}>
      <canvas
        ref={canvas}
        role="img"
        aria-label={`${label}, ${spread}, in ${plural(nonzero, "bin")} of ${BINS}${marked !== null ? `; the selected neuron at ${signed(marked, 3)}` : ""}`}
      />
    </div>
  );
}
