import { Fragment } from "react";

import { PAIR } from "../../lib/figures";
import f from "./Figures.module.css";
import m from "./ModelParts.module.css";
import { StagedFigure, type FigureStage } from "./StagedFigure";

// The parts of a model, drawn: the prompt's tokens at the bottom, each with its residual stream
// rising through the layers; in each layer an attention head moving information from John to the
// last token, and an MLP on every stream; the prediction read from the last stream at the top.
// Under it, one layer as a row of the map. The terms of Start here are the stages: each shows its
// part, and the others step back.

/** The stages, in the order of Start here's terms. */
const PARTS = ["token", "layer", "stream", "head", "mlp", "site"] as const;
type Part = (typeof PARTS)[number];

const TOKENS = [
  { x: 84, text: PAIR.clean[0] },
  { x: 132, text: PAIR.clean[1] },
  { x: 180, text: PAIR.clean[2] },
  { x: 228, text: PAIR.clean[3] },
  { x: 276, text: "…", gap: true },
  { x: 324, text: PAIR.clean[PAIR.clean.length - 1] },
];
const STREAMS = TOKENS.filter((t) => !t.gap).map((t) => t.x);
const JOHN = TOKENS[3].x;
const LAST = TOKENS[TOKENS.length - 1].x;
/** The layers drawn, from the bottom: their top edge and name. */
const LAYERS = [
  { y: 226, name: "layer 0" },
  { y: 168, name: "layer 1" },
  { y: 84, name: "layer 11" },
];
const BAND = 40;
const TOKEN_Y = 284;
/** The layer whose head is the site. */
const SITE_LAYER = 1;

// One layer as a row of the map: the residual stream, the heads, the attention output and the MLP.
const ROW_Y = 344;
const MAP = { res: 48, heads: 72, pitch: 12, attn: 226, mlp: 256, size: 10 };
const SITE_HEAD = 9;

export function ModelPartsFigure({ label, stages }: { label: string; stages: FigureStage[] }) {
  return (
    <StagedFigure label={label} stages={stages} layout="list" still={-1} ms={3000}>
      {(stage, cycle) => <Parts part={stage >= 0 ? PARTS[stage] : null} cycle={cycle} />}
    </StagedFigure>
  );
}

function Parts({ part, cycle }: { part: Part | null; cycle: number }) {
  // Everything is in focus until a part is chosen; then the others step back.
  const off = (...parts: Part[]) => (part === null || parts.includes(part) ? undefined : "");
  const headY = (layer: (typeof LAYERS)[number]) => layer.y + 12;
  const mlpY = (layer: (typeof LAYERS)[number]) => layer.y + BAND - 15;
  return (
    <svg viewBox="0 0 360 362" width={360} height={362}>
      {/* The prediction, read from the last token's stream after the last layer. */}
      <g className={f.dim} data-off={off("stream")}>
        <text className={f.label} x={LAST} y={16} textAnchor="middle">
          {PAIR.answer}
        </text>
        <path className={f.arrow} d={`M ${LAST} 54 V 26 M ${LAST - 3.5} 31.5 L ${LAST} 26 L ${LAST + 3.5} 31.5`} />
      </g>

      {/* The layers, as bands the streams pass through. */}
      {LAYERS.map((layer, i) => (
        <g key={layer.name} className={f.dim} data-off={off("layer", "site") ?? (part === "site" && i !== SITE_LAYER ? "" : undefined)}>
          <rect className={part === "layer" && i === SITE_LAYER ? m.bandOn : m.band} x={44} y={layer.y} width={306} height={BAND} rx={4} />
          <text className={f.note} x={0} y={layer.y + BAND / 2 + 4}>
            {layer.name}
          </text>
        </g>
      ))}
      <text className={f.note} x={0} y={152}>
        …
      </text>

      {/* The residual streams: a running sum at each token, rising through the layers. */}
      <g className={f.dim} data-off={off("stream")}>
        {STREAMS.map((x) => (
          <line key={x} className={x === LAST ? m.lastStream : m.stream} x1={x} x2={x} y1={TOKEN_Y} y2={x === LAST ? 56 : 66} />
        ))}
        {part === "stream" && <circle key={`pulse-${cycle}`} className={m.pulse} cx={LAST} cy={TOKEN_Y} r={4} />}
      </g>

      {/* Attention: in each layer, a head moving information from John to the last token. */}
      <g className={f.dim} data-off={off("head", "layer")}>
        {LAYERS.map((layer) => (
          <Fragment key={layer.name}>
            <path className={`${m.attend} ${part === "head" ? m.flow : ""}`} d={`M ${JOHN} ${headY(layer) + 4} C ${JOHN + 30} ${headY(layer) - 12}, ${LAST - 40} ${headY(layer) - 12}, ${LAST - 7} ${headY(layer)}`} />
            <rect className={f.ink} x={LAST - 5} y={headY(layer) - 5} width={10} height={10} rx={1.5} />
          </Fragment>
        ))}
      </g>

      {/* The MLPs: one on each stream, working on each token by itself. */}
      <g className={f.dim} data-off={off("mlp", "layer")}>
        {LAYERS.map((layer) =>
          STREAMS.map((x) => <rect key={`${layer.name}-${x}`} className={m.mlp} x={x - 6} y={mlpY(layer) - 6} width={12} height={12} rx={2} />),
        )}
      </g>

      {/* A site: where an experiment intervenes, here one head of layer 1. */}
      {(part === "site" || part === null) && (
        <g key={`site-${cycle}`} className={f.enter}>
          <circle className={f.ring} cx={LAST} cy={headY(LAYERS[SITE_LAYER])} r={10} />
          {part === "site" && (
            <text className={f.label} x={LAST - 16} y={headY(LAYERS[SITE_LAYER]) + 4} textAnchor="end">
              a site
            </text>
          )}
        </g>
      )}

      {/* The tokens. */}
      <g className={f.dim} data-off={off("token")}>
        {TOKENS.map((t) =>
          t.gap ? (
            <text key={t.x} className={f.note} x={t.x} y={TOKEN_Y + 16} textAnchor="middle">
              …
            </text>
          ) : (
            <g key={t.x}>
              <rect className={part === "token" ? m.tokenOn : m.token} x={t.x - 22} y={TOKEN_Y} width={44} height={22} rx={4} />
              <text className={m.tokenText} x={t.x} y={TOKEN_Y + 15} textAnchor="middle">
                {t.text}
              </text>
            </g>
          ),
        )}
      </g>

      {/* The same layer, as a row of the map. */}
      <g className={f.dim} data-off={part === "token" ? "" : undefined}>
        <text className={f.note} x={0} y={ROW_Y - 22}>
          One layer, as a row of the map
        </text>
        <MapCell x={MAP.res} on={part === null || part === "layer" || part === "stream"} label="res" />
        {Array.from({ length: 12 }, (_, h) => (
          <MapCell
            key={h}
            x={MAP.heads + h * MAP.pitch}
            on={part === null || part === "layer" || (part === "head" && h === SITE_HEAD) || (part === "site" && h === SITE_HEAD)}
            label={h === 0 ? "heads" : undefined}
            ring={part === "site" && h === SITE_HEAD}
          />
        ))}
        <MapCell x={MAP.attn} on={part === null || part === "layer" || part === "head"} label="attn" />
        <MapCell x={MAP.mlp} on={part === null || part === "layer" || part === "mlp"} label="mlp" />
      </g>
    </svg>
  );
}

function MapCell({ x, on, label, ring }: { x: number; on: boolean; label?: string; ring?: boolean }) {
  return (
    <g>
      <rect className={on ? m.mapOn : m.mapOff} x={x} y={ROW_Y} width={MAP.size} height={MAP.size} rx={1.5} />
      {ring && <circle className={f.ring} cx={x + MAP.size / 2} cy={ROW_Y + MAP.size / 2} r={9} />}
      {label && (
        <text className={f.tick} x={x} y={ROW_Y - 5}>
          {label}
        </text>
      )}
    </g>
  );
}
