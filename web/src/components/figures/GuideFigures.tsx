import { useEffect, useMemo, useRef, useState } from "react";

import { divergingScale, markColor } from "../../lib/color";
import { IOI_HEADS } from "../../lib/figures";
import { GUIDE_FIGURES } from "../../lib/guide";
import { random } from "../../lib/logogram";
import { usePrefersReducedMotion } from "../../lib/motion";
import { useStore } from "../../store/app";
import { Logogram } from "../Logogram";
import f from "./Figures.module.css";
import g from "./GuideFigures.module.css";
import { vars } from "./PatchingFigures";
import { StagedFigure } from "./StagedFigure";

// The guide's figures: the workbench part by part, how to read the model map, how a run's results
// write its logogram, and a bootstrap interval built from each prompt's effect. Their words are in
// lib/guide.ts.

// -- The workbench ---------------------------------------------------------------------------------

/** The parts of the workbench, in the order of the stages, as boxes of the drawing. */
const PARTS = [
  { x: 132, y: 5, w: 136, h: 18 }, // workspaces
  { x: 6, y: 29, w: 150, h: 14 }, // views
  { x: 196, y: 29, w: 150, h: 14 }, // model and prompts
  { x: 2, y: 47, w: 292, h: 160 }, // the view
  { x: 298, y: 47, w: 100, h: 197 }, // inspector
  { x: 2, y: 211, w: 292, h: 33 }, // tokens
  { x: 300, y: 5, w: 94, h: 18 }, // history and commands
];

export function WorkbenchFigure() {
  const { label, stages } = GUIDE_FIGURES.workbench;
  const theme = useStore((st) => st.theme);
  const color = divergingScale(1, theme);
  return (
    <StagedFigure label={label} stages={stages} layout="list" still={-1} ms={2600}>
      {(stage) => {
        const off = (part: number) => (stage >= 0 && stage !== part ? "" : undefined);
        return (
          <svg viewBox="0 0 400 262" width={400} height={262}>
            <rect className={g.frame} x={0.5} y={0.5} width={399} height={261} rx={8} />
            {/* The top bar: the brand, the project, the workspaces, and the tools. */}
            <line className={f.hairline} x1={0} x2={400} y1={26} y2={26} />
            <circle className={g.ink} cx={14} cy={13} r={5} />
            <rect className={g.text} x={26} y={11} width={44} height={4} rx={2} />
            <g className={f.dim} data-off={off(0)}>
              <rect x={136} y={7} width={40} height={14} rx={7} fill="var(--explore)" />
              <rect className={g.pill} x={180} y={7} width={44} height={14} rx={7} />
              <rect className={g.pill} x={228} y={7} width={38} height={14} rx={7} />
            </g>
            <g className={f.dim} data-off={off(6)}>
              {[304, 324, 344].map((x) => (
                <rect key={x} className={g.text} x={x} y={11} width={14} height={4} rx={2} />
              ))}
              <circle className={g.tool} cx={372} cy={13} r={4.5} />
              <circle className={g.tool} cx={388} cy={13} r={4.5} />
            </g>
            {/* The views' tabs, the model and the prompts. */}
            <line className={f.hairline} x1={0} x2={400} y1={45} y2={45} />
            <g className={f.dim} data-off={off(1)}>
              {[10, 40, 70, 100, 126].map((x, i) => (
                <rect key={x} className={i === 0 ? g.ink : g.text} x={x} y={34} width={22} height={4} rx={2} />
              ))}
              <rect x={10} y={42} width={22} height={2} fill="var(--explore)" />
            </g>
            <g className={f.dim} data-off={off(2)}>
              <rect className={g.chip} x={200} y={31} width={70} height={11} rx={5.5} />
              <circle cx={208} cy={36.5} r={2.5} fill="var(--live)" />
              <rect className={g.chip} x={276} y={31} width={66} height={11} rx={5.5} />
            </g>
            {/* The view: a map of a model, with its scale. */}
            <g className={f.dim} data-off={off(3)}>
              {Array.from({ length: 6 }, (_, r) =>
                Array.from({ length: 10 }, (_, c) => {
                  const v = MAP_VALUES[(r * 10 + c) % MAP_VALUES.length];
                  return <rect key={`${r}.${c}`} x={46 + c * 19} y={62 + r * 19} width={16} height={16} rx={2} fill={color(v)} />;
                }),
              )}
              <rect x={46} y={184} width={186} height={5} rx={2} fill={`url(#workbench-scale)`} />
            </g>
            <line className={f.hairline} x1={296} x2={296} y1={45} y2={246} />
            {/* The inspector: a component's evidence. */}
            <g className={f.dim} data-off={off(4)}>
              <rect className={g.ink} x={306} y={58} width={34} height={6} rx={2} />
              <rect className={g.text} x={306} y={72} width={70} height={4} rx={2} />
              <rect className={g.ink} x={306} y={90} width={46} height={10} rx={2} />
              {[110, 120, 130, 140].map((y) => (
                <rect key={y} className={g.text} x={306} y={y} width={78} height={3} rx={1.5} />
              ))}
              {[0.1, 0.25, 0.3, 0.35, 0.42, 0.5, 0.55, 0.7].map((t, i) => (
                <circle key={i} cx={312 + t * 74} cy={168 + (i % 3) * 4} r={2.2} fill={markColor(color, t)} />
              ))}
            </g>
            {/* The tokens of the prompt pair shown. */}
            <line className={f.hairline} x1={0} x2={296} y1={209} y2={209} />
            <g className={f.dim} data-off={off(5)}>
              {[0, 1].map((row) =>
                [12, 34, 54, 78, 100, 124, 146, 170, 190, 214, 238].map((x, i) => (
                  <rect key={`${row}.${x}`} className={i === 7 ? g.marked : g.token} x={x} y={215 + row * 12} width={i % 3 === 0 ? 18 : 20} height={9} rx={2} />
                )),
              )}
            </g>
            {/* The status line. */}
            <line className={f.hairline} x1={0} x2={400} y1={248} y2={248} />
            <rect className={g.text} x={8} y={253} width={24} height={4} rx={2} />
            <defs>
              <linearGradient id="workbench-scale">
                {[-1, 0, 1].map((t) => (
                  <stop key={t} offset={`${(t + 1) * 50}%`} stopColor={color(t)} />
                ))}
              </linearGradient>
            </defs>
            {stage >= 0 && <rect key={stage} className={g.focus} x={PARTS[stage].x} y={PARTS[stage].y} width={PARTS[stage].w} height={PARTS[stage].h} rx={5} />}
          </svg>
        );
      }}
    </StagedFigure>
  );
}

/** Effects for the cells of the drawn maps: mostly near zero, a few strong either way. */
const MAP_VALUES = [0, 0.05, -0.04, 0.1, 0, 0.02, 0.62, -0.08, 0, 0.04, 0.03, 0, -0.55, 0.06, 0.12, 0, 0.02, 0.35, -0.03, 0, 0.08, -0.12, 0, 0.9, 0.04, 0, -0.06, 0.02, 0.18, 0, -0.3, 0.05, 0, 0.03, 0.46, -0.02, 0, 0.07, 0, -0.15];

// -- The map ---------------------------------------------------------------------------------------

const COLUMNS = ["res", "0", "1", "2", "3", "4", "5", "6", "7", "attn", "mlp"];
const M = { left: 52, top: 34, pitch: 22, cell: 18, rows: 6 };
/** The chosen cell: head 7 of layer 3, a strong one. */
const CHOSEN = { row: 3, column: 8 };

export function MapFigure() {
  const { label, stages } = GUIDE_FIGURES.map;
  const theme = useStore((st) => st.theme);
  const color = divergingScale(1, theme);
  // The residual stream, the heads, the attention output and the MLP, a little apart.
  const x = (c: number) => M.left + c * M.pitch + (c >= 1 ? 4 : 0) + (c >= 9 ? 8 : 0) + (c >= 10 ? 8 : 0);
  const y = (r: number) => M.top + r * M.pitch;
  const value = (r: number, c: number) => MAP_VALUES[(r * 11 + c * 3) % MAP_VALUES.length];
  const chosen = value(CHOSEN.row, CHOSEN.column);
  // The inspector's sketch of the chosen cell: its interval and each prompt's effect, from −0.2 to 1.
  const sketch = (v: number) => 364 + 172 * ((v + 0.2) / 1.2);
  return (
    <StagedFigure label={label} stages={stages} ms={[2600, 3000, 3000, 3600]}>
      {(stage, cycle) => (
        <svg viewBox="0 0 560 196" width={560} height={196}>
          {COLUMNS.map((name, c) => (
            <text
              key={name}
              className={`${f.tick} ${stage === 1 ? g.strong : ""}`}
              x={x(c) + M.cell / 2}
              y={M.top - 8}
              textAnchor="middle"
            >
              {name}
            </text>
          ))}
          {Array.from({ length: M.rows }, (_, r) => (
            <g key={r}>
              <text className={`${f.tick} ${stage === 0 ? g.strong : ""}`} x={M.left - 10} y={y(r) + 13} textAnchor="end">
                {r}
              </text>
              {COLUMNS.map((_, c) => (
                <rect
                  key={`${c}-${stage >= 2 ? cycle : "plain"}`}
                  className={stage >= 2 ? f.enter : undefined}
                  style={vars({ "--delay": `${stage === 2 ? r * 110 : 0}ms` })}
                  x={x(c)}
                  y={y(r)}
                  width={M.cell}
                  height={M.cell}
                  rx={2}
                  fill={stage >= 2 ? color(value(r, c)) : "var(--line)"}
                />
              ))}
            </g>
          ))}
          <text className={f.note} x={M.left - 10} y={M.top - 8} textAnchor="end">
            layer
          </text>
          {/* A row: one layer. */}
          {stage === 0 && <rect key={`row-${cycle}`} className={`${f.ring} ${f.enter}`} x={M.left - 4} y={y(2) - 4} width={x(10) + M.cell - M.left + 8} height={M.cell + 8} rx={5} />}
          {/* One component of each kind. */}
          {stage === 1 &&
            [0, 5, 9, 10].map((c, i) => (
              <rect
                key={`kind-${c}-${cycle}`}
                className={`${f.ring} ${f.enter}`}
                style={vars({ "--delay": `${i * 350}ms` })}
                x={x(c) - 3}
                y={y(1) - 3}
                width={M.cell + 6}
                height={M.cell + 6}
                rx={4}
              />
            ))}
          {/* The scale, symmetric around zero, always beside the map. */}
          {stage >= 2 && (
            <g key={`scale-${cycle}`} className={f.enter} style={vars({ "--delay": stage === 2 ? "700ms" : "0ms" })}>
              <defs>
                <linearGradient id="map-scale">
                  {[-1, -0.5, 0, 0.5, 1].map((t) => (
                    <stop key={t} offset={`${(t + 1) * 50}%`} stopColor={color(t)} />
                  ))}
                </linearGradient>
              </defs>
              <rect x={M.left} y={y(M.rows) + 6} width={x(10) + M.cell - M.left} height={6} rx={2} fill="url(#map-scale)" />
              <text className={f.tick} x={M.left} y={y(M.rows) + 26}>
                −1
              </text>
              <text className={f.tick} x={(M.left + x(10) + M.cell) / 2} y={y(M.rows) + 26} textAnchor="middle">
                0
              </text>
              <text className={f.tick} x={x(10) + M.cell} y={y(M.rows) + 26} textAnchor="end">
                +1
              </text>
            </g>
          )}
          {/* A chosen cell, circled in ink, and the inspector's sketch of its evidence. */}
          {stage >= 3 && (
            <g key={`chosen-${cycle}`}>
              <circle className={`${f.ring} ${f.enter}`} cx={x(CHOSEN.column) + M.cell / 2} cy={y(CHOSEN.row) + M.cell / 2} r={14} />
              <g className={f.enter} style={vars({ "--delay": "300ms" })}>
                <rect className={g.card} x={350} y={14} width={200} height={168} rx={8} />
                <text className={f.label} x={364} y={38}>
                  L{CHOSEN.row} H{CHOSEN.column - 1}
                </text>
                <text className={f.note} x={364} y={54}>
                  Attention head {CHOSEN.column - 1} in layer {CHOSEN.row}
                </text>
                <rect x={364} y={68} width={14} height={14} rx={2} fill={color(chosen)} />
                <text className={g.big} x={384} y={84}>
                  {signedValue(chosen)}
                </text>
                <text className={f.note} x={364} y={104}>
                  mean effect, with its 95% interval
                </text>
                <line className={f.axis} x1={364} x2={536} y1={140} y2={140} />
                <line className={g.interval} x1={sketch(chosen - 0.11)} x2={sketch(chosen + 0.12)} y1={140} y2={140} />
                {[-0.3, -0.18, -0.1, -0.06, -0.02, 0.01, 0.04, 0.08, 0.13, 0.2, 0.31].map((d, i) => (
                  <circle key={i} cx={sketch(chosen + d)} cy={156 + (i % 2) * 6} r={2.4} fill={markColor(color, chosen + d)} />
                ))}
                <text className={f.tick} x={364} y={176}>
                  each prompt's effect
                </text>
              </g>
            </g>
          )}
        </svg>
      )}
    </StagedFigure>
  );
}

function signedValue(v: number): string {
  return `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`;
}

// -- The logogram ----------------------------------------------------------------------------------

/** The IOI example's logogram: each layer's strongest head effect, as the run's glyph takes it. */
function ioiProfile(): number[] {
  const out = new Array<number>(12).fill(0);
  for (const h of IOI_HEADS) if (Math.abs(h.effect) > Math.abs(out[h.layer])) out[h.layer] = h.effect;
  return out;
}

const PROFILES: number[][] = [
  new Array(12).fill(0),
  Array.from({ length: 12 }, (_, i) => (i === 2 ? 0.8 : 0)),
  Array.from({ length: 12 }, (_, i) => (i === 8 ? -0.8 : 0)),
  ioiProfile(),
];

/** A profile moving to its next values over a moment, or at once under reduced motion. */
function useEased(target: number[]): number[] {
  const reduced = usePrefersReducedMotion();
  const [shown, setShown] = useState(target);
  const from = useRef(target);
  const current = useRef(target);
  useEffect(() => {
    if (reduced) {
      current.current = target;
      setShown(target);
      return;
    }
    from.current = current.current;
    let frame = 0;
    const start = performance.now();
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / 700);
      const e = 1 - (1 - t) ** 3;
      const next = target.map((v, i) => from.current[i] + (v - from.current[i]) * e);
      current.current = next;
      setShown(next);
      if (t < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [target, reduced]);
  return shown;
}

export function LogogramFigure() {
  const { label, stages } = GUIDE_FIGURES.logogram;
  return (
    <StagedFigure label={label} stages={stages} ms={[2600, 3000, 3000, 4000]}>
      {(stage) => <Glyph profile={PROFILES[stage] ?? PROFILES[PROFILES.length - 1]} />}
    </StagedFigure>
  );
}

function Glyph({ profile }: { profile: number[] }) {
  const shown = useEased(profile);
  return (
    <div className={g.glyph} style={{ flex: "none" }}>
      <Logogram seed="guide" profile={shown} size={176} detail={0.6} haze={false} />
      {/* The layers' places around the ring, clockwise from the top. */}
      <svg className={g.dial} viewBox="0 0 240 240" width={240} height={240}>
        {Array.from({ length: 12 }, (_, i) => {
          const a = -Math.PI / 2 + (Math.PI * 2 * (i + 0.5)) / 12;
          const [r0, r1] = [104, 110];
          return (
            <g key={i}>
              <line className={f.axis} x1={120 + r0 * Math.cos(a)} y1={120 + r0 * Math.sin(a)} x2={120 + r1 * Math.cos(a)} y2={120 + r1 * Math.sin(a)} />
              {i % 3 === 0 && (
                <text className={f.tick} x={120 + 118 * Math.cos(a)} y={120 + 118 * Math.sin(a) + 4} textAnchor="middle">
                  {i}
                </text>
              )}
            </g>
          );
        })}
        <path className={f.arrow} d="M 150 14 A 108 108 0 0 1 196 36" />
        <path className={f.arrow} d="M 190 30 L 196 36 L 188 38" />
      </svg>
    </div>
  );
}

// -- An interval -----------------------------------------------------------------------------------

/** Per-prompt effects for the sketch: drawn from a seed, so the figure is the same every time. */
function sample(): { effects: number[]; means: number[]; mean: number; low: number; high: number } {
  const rand = random(7);
  const normal = () => {
    const u = Math.max(1e-9, rand());
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  };
  const effects = Array.from({ length: 24 }, () => 0.32 + 0.22 * normal());
  const mean = effects.reduce((a, b) => a + b, 0) / effects.length;
  const means = Array.from({ length: 160 }, () => {
    let sum = 0;
    for (let i = 0; i < effects.length; i++) sum += effects[Math.floor(rand() * effects.length)];
    return sum / effects.length;
  });
  const sorted = [...means].sort((a, b) => a - b);
  return { effects, means, mean, low: sorted[Math.floor(0.025 * sorted.length)], high: sorted[Math.ceil(0.975 * sorted.length) - 1] };
}

export function IntervalFigure() {
  const { label, stages } = GUIDE_FIGURES.interval;
  const theme = useStore((st) => st.theme);
  const color = divergingScale(1, theme);
  const data = useMemo(sample, []);
  const x = (v: number) => 40 + ((v + 0.3) / 1.3) * 480;
  return (
    <StagedFigure label={label} stages={stages} ms={[2600, 2400, 3600, 4000]}>
      {(stage, cycle) => (
        <svg viewBox="0 0 560 176" width={560} height={176}>
          <line className={f.axis} x1={x(-0.3)} x2={x(1)} y1={70} y2={70} />
          {[-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1].map((t) => (
            <text key={t} className={f.tick} x={x(t)} y={88} textAnchor="middle">
              {t === 0 ? "0" : t.toFixed(1).replace("-", "−")}
            </text>
          ))}
          <text className={f.note} x={x(-0.3)} y={14}>
            effect, one dot per prompt
          </text>
          {data.effects.map((v, i) => (
            <circle
              key={`${i}-${cycle}`}
              className={f.enter}
              style={vars({ "--delay": `${stage === 0 ? i * 60 : 0}ms` })}
              cx={x(v)}
              cy={46 + (i % 4) * 6}
              r={3.6}
              fill={markColor(color, v)}
            />
          ))}
          {stage >= 1 && (
            <g key={`mean-${cycle}`} className={f.enter}>
              <line className={g.mean} x1={x(data.mean)} x2={x(data.mean)} y1={28} y2={76} />
              <text className={f.label} x={x(data.mean) + 6} y={30}>
                mean {data.mean.toFixed(2)}
              </text>
            </g>
          )}
          {stage >= 2 && (
            <g key={`means-${cycle}`}>
              <text className={f.note} x={x(-0.3)} y={112}>
                the mean of each draw
              </text>
              {data.means.map((v, i) => (
                <line
                  key={i}
                  className={`${g.draw} ${f.enter}`}
                  style={vars({ "--delay": `${stage === 2 ? i * 16 : 0}ms` })}
                  x1={x(v)}
                  x2={x(v)}
                  y1={122}
                  y2={142}
                />
              ))}
            </g>
          )}
          {stage >= 3 && (
            <g key={`interval-${cycle}`} className={f.enter}>
              <path className={f.arrow} d={`M ${x(data.low)} 148 V 154 H ${x(data.high)} V 148`} strokeWidth={1.6} />
              <text className={f.label} x={(x(data.low) + x(data.high)) / 2} y={170} textAnchor="middle">
                95% interval {data.low.toFixed(2)} to {data.high.toFixed(2)}
              </text>
              <line className={g.interval} x1={x(data.low)} x2={x(data.high)} y1={70} y2={70} />
            </g>
          )}
        </svg>
      )}
    </StagedFigure>
  );
}
