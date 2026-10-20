import { useId, type CSSProperties } from "react";

import { figureStages, START, type StartAction, type StartSection } from "../lib/start";
import { emphasis } from "../lib/walkthrough";
import type { View } from "../store/app";
import { walkThroughExample } from "../store/help";
import { IoiFindsFigure } from "./figures/IoiFinds";
import { ModelPartsFigure } from "./figures/ModelParts";
import { PatchingFigure, PromptPairFigure } from "./figures/PatchingFigures";
import { QuestionList } from "./Questions";
import { Button } from "./ui";
import g from "./GuideDialog.module.css";
import s from "./StartHere.module.css";

/**
 * The guide's first page, Start here: its words (lib/start.ts), the four questions Logogram
 * answers, and figures that move through what the words say (components/figures): the two prompts
 * and the logit difference, patching a head, the parts of a model, and what the IOI example finds.
 * The walkthrough's button works on every screen; the buttons that open a view or the examples
 * show only in the workbench, like the other pages' buttons, and close the guide.
 */
export function StartHere({ inWorkbench, go, toExamples }: { inWorkbench: boolean; go: (view: View) => void; toExamples: () => void }) {
  const id = useId();
  const shown = (a: StartAction) => a.to === "tour" || inWorkbench;
  const button = (a: StartAction, primary = false) => (
    <Button
      key={a.label}
      size="small"
      variant={primary ? "primary" : "secondary"}
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
            {section.id === "ask" ? (
              <QuestionList compact />
            ) : (
              section.terms &&
              section.figure?.id !== "parts" && (
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
              )
            )}
            {section.steps && (
              <ol className={s.steps} aria-labelledby={section.title ? titleId : undefined}>
                {section.steps.map((step, i) => (
                  <li key={step.text} style={{ "--step": `var(--${STEP_COLORS[i] ?? "text"})` } as CSSProperties}>
                    <span className={s.stepNumber} aria-hidden="true">
                      {i + 1}
                    </span>
                    <span className={s.stepBody}>
                      <span>
                        <Words text={step.text} />
                      </span>
                      {step.action && shown(step.action) && <span className={s.stepAction}>{button(step.action, true)}</span>}
                    </span>
                  </li>
                ))}
              </ol>
            )}
            {actions.length > 0 && <div className={g.actions}>{actions.map((a) => button(a))}</div>}
            <Figure section={section} />
          </div>
        );
      })}
    </>
  );
}

/** Where each step of the first hour happens, by the color with that job: a loaded model is green,
 * and the steps in a workspace take its color. The first step opens the example. */
const STEP_COLORS = ["text", "live", "experiment", "experiment", "evidence", "evidence"];

function Figure({ section }: { section: StartSection }) {
  const figure = section.figure;
  if (!figure) return null;
  const stages = figureStages(section);
  switch (figure.id) {
    case "pair":
      return <PromptPairFigure label={figure.label} stages={stages} />;
    case "patching":
      return <PatchingFigure label={figure.label} stages={stages} caption={figure.caption ?? ""} />;
    case "parts":
      return <ModelPartsFigure label={figure.label} stages={stages} />;
    case "finds":
      return <IoiFindsFigure label={figure.label} stages={stages} />;
  }
}

/** A paragraph, with the names of controls in bold. */
function Words({ text }: { text: string }) {
  return <>{emphasis(text).map((part, i) => (part.strong ? <strong key={i}>{part.text}</strong> : part.text))}</>;
}
