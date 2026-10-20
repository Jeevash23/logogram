import { useEffect, useId, useState, type CSSProperties } from "react";

import { divergingScale } from "../../lib/color";
import { METER, PAIR, PATCHED_HEAD, patchedAt, patchedEffect } from "../../lib/figures";
import { usePrefersReducedMotion } from "../../lib/motion";
import { useStore } from "../../store/app";
import f from "./Figures.module.css";
import p from "./Patching.module.css";
import { StagedFigure, type FigureStage } from "./StagedFigure";

// The figures of Start here that follow the IOI example's question: its two prompts and the
// logit difference each gives, then activation patching, as three runs of GPT-2 small and where
// each puts the answer on one scale. What they draw is in lib/figures.ts.

/** Inline custom properties, such as an animation's --delay. */
export function vars(values: Record<`--${string}`, string | number>): CSSProperties {
  return values as CSSProperties;
}

/** The logit difference's colors, as the baseline draws them: cobalt where the model prefers the
 * answer, amber where it prefers the distractor. */
function usePreference(): (value: number) => string {
  const theme = useStore((st) => st.theme);
  const scale = divergingScale(0.75, theme);
  return (value) => scale(value);
}

/** A share of the scale's half width (from lib/figures.ts) as a position along it. */
interface Scale {
  x: (value: number) => number;
  y: number;
  left: number;
  right: number;
}

/** The scale the answer is read on: the distractor at the left, the answer at the right, zero in
 * the middle. */
function ScaleAxis({ scale }: { scale: Scale }) {
  const zero = scale.x(0);
  return (
    <g>
      <line className={f.axis} x1={scale.left} x2={scale.right} y1={scale.y} y2={scale.y} />
      <line className={f.axis} x1={zero} x2={zero} y1={scale.y - 6} y2={scale.y + 6} />
      <text className={f.tick} x={zero} y={scale.y - 12} textAnchor="middle">
        0
      </text>
      <text className={f.label} x={scale.left - 10} y={scale.y + 4} textAnchor="end">
        {PAIR.distractor}
      </text>
      <text className={f.label} x={scale.right + 10} y={scale.y + 4}>
        {PAIR.answer}
      </text>
    </g>
  );
}

/** Where a run puts the answer: a dot on the scale, named, sliding in from `from`. */
function Mark({ scale, value, from = 0, name, delay, color, below }: { scale: Scale; value: number; from?: number; name: string; delay: number; color: string; below?: boolean }) {
  const x = scale.x(value);
  return (
    <g transform={`translate(${x} ${scale.y})`}>
      <g className={f.slide} style={vars({ "--delay": `${delay}ms`, "--from": `${scale.x(from) - x}px` })}>
        <circle r={5.5} fill={color} stroke="var(--surface)" strokeWidth={1.5} />
        <text className={f.note} y={below ? 22 : -12} textAnchor="middle">
          {name}
        </text>
      </g>
    </g>
  );
}

/** A bracket under the scale from one value to another, named under its middle. */
function Bracket({ scale, from, to, y, name, delay, strong }: { scale: Scale; from: number; to: number; y: number; name: string; delay: number; strong?: boolean }) {
  const [a, b] = [scale.x(from), scale.x(to)].sort((m, n) => m - n);
  return (
    <g className={f.enter} style={vars({ "--delay": `${delay}ms` })}>
      <path className={f.arrow} d={`M ${a} ${y - 5} V ${y} H ${b} V ${y - 5}`} strokeWidth={strong ? 1.6 : 1} opacity={strong ? 1 : 0.7} />
      <text className={strong ? f.label : f.note} x={(a + b) / 2} y={y + 15} textAnchor="middle">
        {name}
      </text>
    </g>
  );
}

// -- The prompt pair -------------------------------------------------------------------------------

/** A prompt, token by token as they come in, and the name the model prefers after it. */
function PromptRow({ which, tokens, changed, answer, hidden }: { which: string; tokens: readonly string[]; changed?: number; answer: string; hidden?: boolean }) {
  return (
    <div className={p.row} style={hidden ? { visibility: "hidden" } : undefined}>
      <span className={p.rowLabel}>{which}</span>
      <span className={p.tokens}>
        {tokens.map((token, i) => (
          <span key={i} className={`${p.token} ${i === changed ? p.changed : ""}`} style={vars({ "--i": i })}>
            {token}
          </span>
        ))}
        <span className={p.answer} style={vars({ "--i": tokens.length })}>
          → {answer}
        </span>
      </span>
    </div>
  );
}

/** When the last token of a row has come in, and its answer after it (as Patching.module.css
 * times them). */
const ROW_MS = PAIR.clean.length * 55 + 400;

const PAIR_SCALE: Scale = { x: (v) => 460 + v * 260, y: 40, left: 200, right: 720 };

export function PromptPairFigure({ label, stages }: { label: string; stages: FigureStage[] }) {
  const color = usePreference();
  return (
    <StagedFigure label={label} stages={stages} ms={[3600, 3600, 3400]}>
      {(stage, cycle) => (
        <div className={p.pair}>
          <PromptRow key={`clean-${cycle}`} which="clean" tokens={PAIR.clean} answer={PAIR.answer} />
          {stage >= 1 ? (
            <PromptRow key={`corrupt-${cycle}`} which="corrupt" tokens={PAIR.corrupt} changed={PAIR.changed} answer={PAIR.distractor} />
          ) : (
            <PromptRow which="corrupt" tokens={PAIR.corrupt} answer={PAIR.distractor} hidden />
          )}
          <svg viewBox="0 0 900 84" width={900} height={84} className={p.scale}>
            <text className={f.note} x={PAIR_SCALE.left} y={12}>
              logit difference
            </text>
            <ScaleAxis scale={PAIR_SCALE} />
            <Mark key={`clean-${cycle}`} scale={PAIR_SCALE} value={METER.clean} name="clean" delay={ROW_MS} color={color(METER.clean)} />
            {stage >= 1 && <Mark key={`corrupt-${cycle}`} scale={PAIR_SCALE} value={METER.corrupt} name="corrupt" delay={ROW_MS} color={color(METER.corrupt)} />}
            {stage >= 2 && <Bracket key={`gap-${cycle}`} scale={PAIR_SCALE} from={METER.corrupt} to={METER.clean} y={64} name="the gap" delay={150} strong />}
          </svg>
        </div>
      )}
    </StagedFigure>
  );
}

// -- Patching ----------------------------------------------------------------------------------

// Each run as GPT-2 small's 144 heads: a row per layer from layer 0 at the top, a cell per head, as
// on the map. Every grid starts on the cells' pattern, so one pattern draws all three.
const PITCH = 12;
const CELL = 10;
const SIZE = 12 * PITCH - (PITCH - CELL);
const TOP = 40;
const RUNS = [
  { x: 79, name: "Clean", says: "prefers Mary" },
  { x: 379, name: "Corrupt", says: "prefers John" },
  { x: 679, name: "Patched", says: "moves toward Mary" },
];
/** How long a run takes to fill its rows, and when the copied head arrives in the patched run. */
const ROW_GAP = 70;
const TRAVEL = { delay: 250, ms: 900 };
const ARRIVES = TRAVEL.delay + TRAVEL.ms;
const PATCH_SCALE: Scale = { x: (v) => 450 + v * 260, y: 302, left: 190, right: 710 };

function center(run: number) {
  return { x: RUNS[run].x + PATCHED_HEAD.head * PITCH + CELL / 2, y: TOP + PATCHED_HEAD.layer * PITCH + CELL / 2 };
}

/** The copy's path: from the head in the clean run, under the corrupt run, into the patched run. */
function arc(): string {
  const [a, b] = [center(0), center(2)];
  return `M ${a.x} ${a.y + 8} C ${a.x} ${a.y + 118}, ${b.x} ${b.y + 118}, ${b.x} ${b.y + 8}`;
}

/** A point of the copy's path, from 0 at the clean run to 1 at the patched run. */
function along(t: number): { x: number; y: number } {
  const [a, b] = [center(0), center(2)];
  const p0 = { x: a.x, y: a.y + 8 };
  const p1 = { x: a.x, y: a.y + 118 };
  const p2 = { x: b.x, y: b.y + 118 };
  const p3 = { x: b.x, y: b.y + 8 };
  const u = 1 - t;
  const w = [u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t];
  return { x: w[0] * p0.x + w[1] * p1.x + w[2] * p2.x + w[3] * p3.x, y: w[0] * p0.y + w[1] * p1.y + w[2] * p2.y + w[3] * p3.y };
}

/** The head's output on its way from the clean run to the patched run. It isn't drawn under
 * reduced motion: the path shows where it goes. */
function Travelling() {
  const reduced = usePrefersReducedMotion();
  const [t, setT] = useState(0);
  useEffect(() => {
    if (reduced) return;
    let frame = 0;
    const start = performance.now() + TRAVEL.delay;
    const step = (now: number) => {
      const v = Math.min(1, Math.max(0, (now - start) / TRAVEL.ms));
      setT(v);
      if (v < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [reduced]);
  if (reduced || t <= 0 || t >= 1) return null;
  const eased = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
  const at = along(eased);
  return <circle className={f.ink} cx={at.x} cy={at.y} r={3.5} />;
}

/** A run's rows, filling one after another from `delay`, from row `from` up to (not including)
 * row `to`. */
function Rows({ run, from = 0, to = 12, delay = 0, fill }: { run: number; from?: number; to?: number; delay?: number; fill: string }) {
  return (
    <>
      {Array.from({ length: to - from }, (_, k) => from + k).map((row) => (
        <rect
          key={row}
          className={f.enter}
          style={vars({ "--delay": `${delay + (row - from) * ROW_GAP}ms` })}
          x={RUNS[run].x}
          y={TOP + row * PITCH}
          width={SIZE}
          height={CELL}
          fill={fill}
        />
      ))}
    </>
  );
}

export function PatchingFigure({ label, stages, caption }: { label: string; stages: FigureStage[]; caption: string }) {
  const color = usePreference();
  const raw = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const effect = patchedEffect();
  const patched = patchedAt(effect);
  const fill = `url(#${raw}-ink)`;
  const head = center(2);
  return (
    <StagedFigure label={label} stages={stages} caption={caption} ms={[2800, 2800, 3400, 4600]}>
      {(stage, cycle) => (
        <svg viewBox="0 0 900 374" width={900} height={374}>
          <defs>
            <pattern id={`${raw}-cells`} x={RUNS[0].x} y={TOP} width={PITCH} height={PITCH} patternUnits="userSpaceOnUse">
              <rect className={f.cell} width={CELL} height={CELL} />
            </pattern>
            <pattern id={`${raw}-ink`} x={RUNS[0].x} y={TOP} width={PITCH} height={PITCH} patternUnits="userSpaceOnUse">
              <rect className={f.cellInk} width={CELL} height={CELL} />
            </pattern>
          </defs>
          {RUNS.map((run) => (
            <g key={run.name}>
              <text className={f.label} x={run.x} y={16}>
                {run.name}
              </text>
              <rect x={run.x} y={TOP} width={SIZE} height={SIZE} fill={`url(#${raw}-cells)`} />
            </g>
          ))}
          {/* Each run, computed layer by layer, and what it says. */}
          <g key={`clean-${cycle}`}>
            <Rows run={0} fill={fill} />
            <text className={`${f.note} ${f.enter}`} style={vars({ "--delay": `${12 * ROW_GAP}ms` })} x={RUNS[0].x} y={TOP - 10}>
              {RUNS[0].says}
            </text>
          </g>
          {stage >= 1 && (
            <g key={`corrupt-${cycle}`}>
              <Rows run={1} fill={fill} />
              <text className={`${f.note} ${f.enter}`} style={vars({ "--delay": `${12 * ROW_GAP}ms` })} x={RUNS[1].x} y={TOP - 10}>
                {RUNS[1].says}
              </text>
            </g>
          )}
          {stage >= 2 && (
            <g key={`patched-${cycle}`}>
              <circle className={`${f.ring} ${f.enter}`} cx={center(0).x} cy={center(0).y} r={8.5} />
              <path className={`${f.arrow} ${f.draw}`} style={vars({ "--length": 1, "--delay": `${TRAVEL.delay}ms` })} pathLength={1} d={arc()} opacity={0.75} />
              <path
                className={`${f.arrow} ${f.enter}`}
                style={vars({ "--delay": `${ARRIVES}ms` })}
                d={`M ${head.x - 3.5} ${head.y + 13.5} L ${head.x} ${head.y + 8} L ${head.x + 3.5} ${head.y + 13.5}`}
              />
              <Travelling />
              <Rows run={2} to={PATCHED_HEAD.layer} fill={fill} />
              <rect
                className={`${f.ink} ${f.enter}`}
                style={vars({ "--delay": `${ARRIVES}ms` })}
                x={head.x - CELL / 2}
                y={head.y - CELL / 2}
                width={CELL}
                height={CELL}
              />
              <Rows run={2} from={PATCHED_HEAD.layer} delay={ARRIVES + 120} fill={fill} />
              <text
                className={`${f.note} ${f.enter}`}
                style={vars({ "--delay": `${ARRIVES + 120 + 3 * ROW_GAP}ms` })}
                x={RUNS[2].x}
                y={TOP - 10}
              >
                {RUNS[2].says}
              </text>
            </g>
          )}
          {/* Where each run puts the answer, on one scale. */}
          <ScaleAxis scale={PATCH_SCALE} />
          <Mark key={`m-clean-${cycle}`} scale={PATCH_SCALE} value={METER.clean} name="clean" delay={12 * ROW_GAP} color={color(METER.clean)} />
          {stage >= 1 && (
            <Mark key={`m-corrupt-${cycle}`} scale={PATCH_SCALE} value={METER.corrupt} name="corrupt" delay={12 * ROW_GAP} color={color(METER.corrupt)} />
          )}
          {stage >= 3 && (
            <g key={`measure-${cycle}`}>
              <Mark scale={PATCH_SCALE} value={patched} from={METER.corrupt} name="patched" delay={150} color={color(patched)} />
              <Bracket scale={PATCH_SCALE} from={METER.corrupt} to={patched} y={326} name="effect" delay={1150} strong />
              <Bracket scale={PATCH_SCALE} from={METER.corrupt} to={METER.clean} y={354} name="the gap" delay={1500} />
            </g>
          )}
        </svg>
      )}
    </StagedFigure>
  );
}
