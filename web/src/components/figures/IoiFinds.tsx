import { useId } from "react";

import { divergingScale } from "../../lib/color";
import { IOI_HEADS, type IoiRole } from "../../lib/figures";
import { useStore } from "../../store/app";
import f from "./Figures.module.css";
import { vars } from "./PatchingFigures";
import { StagedFigure, type FigureStage } from "./StagedFigure";

// What the IOI example finds: GPT-2 small's 144 heads, a row per layer as on the map, painting in
// layer by layer as a run's results do, each colored by its effect on the effect scale; then the
// heads of each role, ringed, with their effects. The effects are the guide's (lib/figures.ts).

/** The roles, in the order of the stages after the first. */
const ROLES: IoiRole[] = ["induction", "s-inhibition", "name-mover", "negative-name-mover"];

const PITCH = 20;
const CELL = 17;
const LEFT = 44;
const TOP = 34;
const SIZE = 12 * PITCH - (PITCH - CELL);
/** The effect scale's reach, either side of zero. */
const REACH = 0.5;

export function IoiFindsFigure({ label, stages }: { label: string; stages: FigureStage[] }) {
  return (
    <StagedFigure label={label} stages={stages} layout="list" still={-1} ms={[3400, 2600, 2600, 2600, 2600]}>
      {(stage, cycle) => <HeadMap stage={stage} cycle={cycle} />}
    </StagedFigure>
  );
}

function HeadMap({ stage, cycle }: { stage: number; cycle: number }) {
  const theme = useStore((st) => st.theme);
  const color = divergingScale(REACH, theme);
  const gradient = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  // Which heads are ringed: one role's, or every role's when the figure is still.
  const role = stage >= 1 ? ROLES[stage - 1] : null;
  const ringed = IOI_HEADS.filter((h) => (stage === -1 ? true : h.role === role));
  const effect = new Map<string, number>(IOI_HEADS.map((h) => [`${h.layer}.${h.head}`, h.effect]));
  const x = (head: number) => LEFT + head * PITCH;
  const y = (layer: number) => TOP + layer * PITCH;
  return (
    <svg viewBox="0 0 360 358" width={360} height={358}>
      <defs>
        <linearGradient id={gradient}>
          {[-1, -0.5, 0, 0.5, 1].map((t) => (
            <stop key={t} offset={`${(t + 1) * 50}%`} stopColor={color(t * REACH)} />
          ))}
        </linearGradient>
      </defs>
      <text className={f.tick} x={LEFT} y={TOP - 18}>
        head
      </text>
      {[0, 5, 11].map((h) => (
        <text key={h} className={f.tick} x={x(h) + CELL / 2} y={TOP - 4} textAnchor="middle">
          {h}
        </text>
      ))}
      <text className={f.tick} x={LEFT - 10} y={TOP - 18} textAnchor="end">
        layer
      </text>
      {Array.from({ length: 12 }, (_, layer) => (
        <g key={`${layer}-${cycle}`} className={f.enter} style={vars({ "--delay": `${stage === 0 ? layer * 110 : 0}ms` })}>
          <text className={f.tick} x={LEFT - 8} y={y(layer) + CELL - 3} textAnchor="end">
            {layer}
          </text>
          {Array.from({ length: 12 }, (_, head) => {
            const value = effect.get(`${layer}.${head}`) ?? 0;
            const named = ringed.some((h) => h.layer === layer && h.head === head);
            return (
              <rect
                key={head}
                className={f.dim}
                data-off={role && !named ? "" : undefined}
                x={x(head)}
                y={y(layer)}
                width={CELL}
                height={CELL}
                rx={2}
                fill={color(value)}
              />
            );
          })}
        </g>
      ))}
      {ringed.map((h) => (
        <rect
          key={`${h.layer}.${h.head}-${stage}-${cycle}`}
          className={`${f.ring} ${f.enter}`}
          style={vars({ "--delay": stage === 0 ? "0ms" : "120ms" })}
          x={x(h.head) - 2.5}
          y={y(h.layer) - 2.5}
          width={CELL + 5}
          height={CELL + 5}
          rx={3.5}
        />
      ))}
      {/* The ringed heads' effects, as the guide lists them. */}
      {role && (
        <text key={`${role}-${cycle}`} className={`${f.label} ${f.enter}`} style={vars({ "--delay": "120ms" })} x={LEFT} y={TOP + SIZE + 78}>
          {ringed.map((h) => `${h.layer}.${h.head} ${h.effect > 0 ? "+" : "−"}${Math.abs(h.effect).toFixed(2)}`).join("  ·  ")}
        </text>
      )}
      {/* The effect scale, as the map shows it beside its cells. */}
      <g transform={`translate(${LEFT} ${TOP + SIZE + 24})`}>
        <rect width={SIZE} height={8} rx={2} fill={`url(#${gradient})`} />
        <text className={f.tick} x={0} y={22}>
          −{REACH}
        </text>
        <text className={f.tick} x={SIZE / 2} y={22} textAnchor="middle">
          0
        </text>
        <text className={f.tick} x={SIZE} y={22} textAnchor="end">
          +{REACH}
        </text>
        <text className={f.note} x={SIZE + 10} y={8}>
          effect
        </text>
      </g>
    </svg>
  );
}
