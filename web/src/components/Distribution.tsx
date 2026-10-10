import { scaleLinear } from "d3-scale";
import { useMemo, useRef } from "react";

import { useElementSize } from "../lib/canvas";
import { divergingScale, markColor, niceBound } from "../lib/color";
import { useStore } from "../store/app";
import { num, signed } from "../lib/format";
import s from "./Distribution.module.css";

interface Point {
  index: number;
  value: number | null;
}

const R = 3;

/** Per-prompt values as a beeswarm (or a histogram for large n), with the mean and its CI. Pass
 * values that keep their identity between renders (useMemo), or the layout is computed again. */
export function Distribution({
  values,
  mean,
  lo,
  hi,
  onPick,
  label = "Per-prompt normalized effect",
  pointName = "Prompt",
  showKey = true,
}: {
  values: Point[];
  mean: number | null;
  lo: number | null;
  hi: number | null;
  onPick?: (index: number) => void;
  label?: string;
  pointName?: string;
  showKey?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const { width } = useElementSize(ref);
  // Kept while values keep their identity: the beeswarm below is quadratic in their number.
  const finite = useMemo(
    () => values.filter((v): v is { index: number; value: number } => v.value !== null && Number.isFinite(v.value)),
    [values],
  );
  const theme = useStore((st) => st.theme);
  const signScale = useMemo(
    () => divergingScale(niceBound(finite.map((v) => v.value), 0.1), theme),
    [finite, theme],
  );

  const plot = useMemo(() => {
    if (width === 0 || finite.length === 0) return null;
    const ext = [0, ...finite.map((v) => v.value), lo ?? 0, hi ?? 0];
    let min = Math.min(...ext);
    let max = Math.max(...ext);
    if (max - min < 1e-6) {
      min -= 0.5;
      max += 0.5;
    }
    const pad = (max - min) * 0.06;
    const x = scaleLinear().domain([min - pad, max + pad]).range([R + 2, width - R - 2]).nice();
    const histogram = () => {
      const bins = 32;
      const [d0, d1] = x.domain();
      const step = (d1 - d0) / bins;
      const counts = new Array(bins).fill(0);
      for (const v of finite) counts[Math.min(bins - 1, Math.floor((v.value - d0) / step))] += 1;
      const top = Math.max(...counts);
      return { x, kind: "hist" as const, counts, step, d0, top, height: 72 };
    };
    if (finite.length > 400) return histogram();
    // Beeswarm: place each dot at the smallest offset from the center line that doesn't overlap.
    const sorted = [...finite].sort((a, b) => a.value - b.value);
    const placed: { index: number; value: number; cx: number; dy: number }[] = [];
    for (const v of sorted) {
      const cx = x(v.value);
      const near = placed.filter((p) => Math.abs(p.cx - cx) < 2 * R + 0.5);
      let dy = 0;
      for (let k = 0; k < 60; k++) {
        const candidate = (k % 2 === 0 ? 1 : -1) * Math.ceil(k / 2) * (2 * R - 0.5);
        if (near.every((p) => Math.hypot(p.cx - cx, p.dy - candidate) >= 2 * R - 0.2)) {
          dy = candidate;
          break;
        }
      }
      placed.push({ ...v, cx, dy });
    }
    const spread = Math.max(...placed.map((p) => Math.abs(p.dy)), 0);
    const MAX_SPREAD = 46;
    // Dense piles are squeezed to fit; dots then overlap, and their transparency shows density.
    const squeeze = spread > MAX_SPREAD ? MAX_SPREAD / spread : 1;
    for (const p of placed) p.dy *= squeeze;
    const height = Math.max(44, Math.min(spread, MAX_SPREAD) * 2 + 2 * R + 26);
    return { x, kind: "swarm" as const, placed, height, dense: squeeze < 1 };
  }, [width, finite, lo, hi]);

  return (
    <div className={s.wrap} ref={ref}>
      <div className={s.caption}>{label}</div>
      {plot && (
        <svg width={width} height={plot.height} role="img" aria-label={`${label}: mean ${num(mean, 2)}, ${finite.length} prompts`}>
          {(() => {
            const base = plot.height - 18;
            const mid = plot.kind === "swarm" ? (base - 4) / 2 + 2 : 0;
            const ticks = plot.x.ticks(4);
            const tickText = plot.x.tickFormat(4); // as many decimals as the tick step needs
            return (
              <>
                {lo !== null && hi !== null && (
                  <rect className={s.band} x={plot.x(lo)} y={2} width={Math.max(1, plot.x(hi) - plot.x(lo))} height={base - 2} />
                )}
                <line className={s.zero} x1={plot.x(0)} x2={plot.x(0)} y1={2} y2={base} />
                {plot.kind === "swarm"
                  ? plot.placed.map((p) => (
                      <circle
                        key={p.index}
                        className={plot.dense ? s.dotDense : s.dot}
                        style={{ fill: markColor(signScale, p.value) }}
                        cx={p.cx}
                        cy={mid + p.dy}
                        r={R}
                        onClick={() => onPick?.(p.index)}
                      >
                        <title>{`${pointName} ${p.index}: ${signed(p.value, 3)}`}</title>
                      </circle>
                    ))
                  : plot.counts.map((c, i) => {
                      const x0 = plot.x(plot.d0 + i * plot.step);
                      const x1 = plot.x(plot.d0 + (i + 1) * plot.step);
                      const h = (c / plot.top) * (base - 6);
                      return <rect key={i} className={s.bar} x={x0 + 0.5} width={Math.max(0, x1 - x0 - 1)} y={base - h} height={h} />;
                    })}
                {mean !== null && <line className={s.mean} x1={plot.x(mean)} x2={plot.x(mean)} y1={0} y2={base + 2} />}
                <line className={s.axis} x1={0} x2={width} y1={base + 0.5} y2={base + 0.5} />
                {ticks.map((t) => (
                  <text key={t} className={s.tick} x={plot.x(t)} y={plot.height - 4} textAnchor="middle">
                    {tickText(t)}
                  </text>
                ))}
              </>
            );
          })()}
        </svg>
      )}
      {showKey && (
        <div className={s.key}>
          <span className={s.keyMean} /> mean <span className={s.keyBand} /> confidence interval{" "}
          <span className={s.keyZero} /> zero
        </div>
      )}
    </div>
  );
}
