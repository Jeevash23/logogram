import { useId } from "react";

import { FIGURE_CAPTION, START, type StartAction } from "../lib/start";
import { emphasis } from "../lib/walkthrough";
import type { View } from "../store/app";
import { walkThroughExample } from "../store/help";
import { Button } from "./ui";
import g from "./GuideDialog.module.css";
import s from "./StartHere.module.css";

/**
 * The guide's first page, Start here: its words (lib/start.ts) and a figure of the three runs
 * patching compares. The walkthrough's button works on every screen; the buttons that open a view
 * or the examples show only in the workbench, like the other pages' buttons, and close the guide.
 */
export function StartHere({ inWorkbench, go, toExamples }: { inWorkbench: boolean; go: (view: View) => void; toExamples: () => void }) {
  const id = useId();
  const shown = (a: StartAction) => a.to === "tour" || inWorkbench;
  const button = (a: StartAction) => (
    <Button
      key={a.label}
      size="small"
      onClick={() => (a.to === "tour" ? void walkThroughExample() : a.to === "view" ? go(a.view) : toExamples())}
    >
      {a.label}
    </Button>
  );

  return (
    <>
      {START.map((section) => {
        const titleId = `${id}-${section.id}`;
        const actions = (section.actions ?? []).filter(shown);
        return (
          <div key={section.id} className={s.section}>
            {section.title && (
              <h3 id={titleId} className={s.title}>
                {section.title}
              </h3>
            )}
            {section.paragraphs.map((p) => (
              <p key={p}>
                <Words text={p} />
              </p>
            ))}
            {section.terms && (
              <dl className={g.rows} aria-labelledby={section.title ? titleId : undefined}>
                {section.terms.map((t) => (
                  <div key={t.term} className={g.row}>
                    <dt>{t.term}</dt>
                    <dd>
                      <Words text={t.text} />
                    </dd>
                  </div>
                ))}
              </dl>
            )}
            {section.steps && (
              <ol className={s.steps} aria-labelledby={section.title ? titleId : undefined}>
                {section.steps.map((step) => (
                  <li key={step.text}>
                    <span>
                      <Words text={step.text} />
                    </span>
                    {step.action && shown(step.action) && <span className={s.stepAction}>{button(step.action)}</span>}
                  </li>
                ))}
              </ol>
            )}
            {actions.length > 0 && <div className={g.actions}>{actions.map(button)}</div>}
            {section.id === "patch" && <PatchingFigure />}
          </div>
        );
      })}
    </>
  );
}

/** A paragraph, with the names of controls in bold. */
function Words({ text }: { text: string }) {
  return <>{emphasis(text).map((part, i) => (part.strong ? <strong key={i}>{part.text}</strong> : part.text))}</>;
}

// The figure: each run as GPT-2 small's 144 heads, a row per layer from layer 0 at the top and a
// cell per head, as on the map. Ink and hairlines only: the effect colors are for data.
const PITCH = 8;
const CELL = 6;
const SIZE = 12 * PITCH - (PITCH - CELL);
const TOP = 24;
const RUNS = [
  { x: 8, name: "Clean", says: "answers Mary" },
  { x: 160, name: "Corrupt", says: "answers John" },
  { x: 312, name: "Patched", says: "moves toward Mary" },
];
/** Head 9.9, a name mover: ringed where it is read, in the clean run, and filled where it is
 * written, in the patched run. */
const HEAD = { layer: 9, head: 9 };

function PatchingFigure() {
  const captionId = useId();
  const cx = (run: number) => RUNS[run].x + HEAD.head * PITCH + CELL / 2;
  const cy = TOP + HEAD.layer * PITCH + CELL / 2;
  const [from, to] = [cx(0), cx(2)];
  return (
    <figure className={s.figure} aria-labelledby={captionId}>
      <svg className={s.drawing} viewBox="0 0 432 166" aria-hidden="true" focusable="false">
        <defs>
          {/* Cells on a grid of PITCH from the origin: every run starts on that grid. */}
          <pattern id="start-here-cells" width={PITCH} height={PITCH} patternUnits="userSpaceOnUse">
            <rect className={s.cell} width={CELL} height={CELL} />
          </pattern>
        </defs>
        {RUNS.map((run) => (
          <g key={run.name}>
            <text className={s.run} x={run.x} y={14}>
              {run.name}
            </text>
            <rect x={run.x} y={TOP} width={SIZE} height={SIZE} fill="url(#start-here-cells)" />
            <text className={s.says} x={run.x} y={160}>
              {run.says}
            </text>
          </g>
        ))}
        <circle className={s.ring} cx={from} cy={cy} r={6.5} />
        <rect className={s.mark} x={to - CELL / 2} y={cy - CELL / 2} width={CELL} height={CELL} />
        <path className={s.arrow} d={`M ${from} ${cy + 7} C ${from} ${cy + 45}, ${to} ${cy + 45}, ${to} ${cy + 7}`} />
        <path className={s.arrow} d={`M ${to - 3.5} ${cy + 12.5} L ${to} ${cy + 7} L ${to + 3.5} ${cy + 12.5}`} />
      </svg>
      <figcaption id={captionId} className={s.caption}>
        {FIGURE_CAPTION}
      </figcaption>
    </figure>
  );
}
