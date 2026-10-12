import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { ExampleInfo } from "../api/types";
import { count } from "../lib/format";
import { modelName } from "../lib/hooks";
import { metricWords } from "../lib/metrics";
import { experimentText, scopeText } from "../lib/spec";
import { TOUR_EXAMPLE } from "../lib/walkthrough";
import { useStore } from "../store/app";
import { startTour } from "../store/help";
import s from "./ExampleGallery.module.css";
import { Logogram } from "./Logogram";
import { Button, Callout, Spinner } from "./ui";

/**
 * The bundled examples, each with what it asks, its model and method, and the result to expect,
 * read against a publication. Opening one copies it into the projects folder, once: a copy made
 * before opens as it was left, never replaced.
 */
export function ExampleGallery({ onOpen, disabled }: { onOpen?: () => void; disabled?: boolean }) {
  const [examples, setExamples] = useState<ExampleInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [opening, setOpening] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    api.examples().then(
      (list) => live && setExamples(list),
      (e: Error) => live && setError(e.message),
    );
    return () => {
      live = false;
    };
  }, []);

  // Until the walkthrough has been finished or skipped, its example offers it.
  const offerWalkthrough = useStore((st) => st.walkthrough === null);

  const open = async (example: ExampleInfo, walkThrough = false) => {
    setOpening(example.id);
    onOpen?.();
    const st = useStore.getState();
    const project = await st.guard(() => api.openExample(example.id));
    setOpening(null);
    if (!project) return;
    await st.enterProject(project);
    if (walkThrough) startTour();
  };

  const lead = (
    <p className={s.lead}>
      Experiments on GPT-2 small whose results can be read against the papers that found them. Each opens as a project
      of its own, a copy you can change and run again.
    </p>
  );
  if (error) return <Callout tone="error" title="The examples can't be listed">{error}</Callout>;
  if (!examples) {
    return (
      <>
        {lead}
        <div className={s.loading}>
          <Spinner /> Listing the examples…
        </div>
      </>
    );
  }
  return (
    <>
      {lead}
      <ul className={s.list} aria-label="Examples">
        {examples.map((example) => {
          const { spec } = example;
          const action = example.copied ? "Open your copy" : "Open";
          return (
            <li key={example.id} className={s.item}>
              <Logogram seed={example.path} size={34} className={s.glyph} />
              <div className={s.body}>
                <h3 className={s.name}>{example.name}</h3>
                <p className={s.question}>{example.question}</p>
                <p className={s.method}>
                  {modelName(spec.model.id)} · {experimentText(spec.experiment)}, {scopeText(spec.scope)} ·{" "}
                  {metricWords(spec.metric).label} · {count(example.n_prompts)} prompt pairs
                </p>
                <p className={s.expect}>
                  <span className={s.label}>Expect</span> {example.expect} <span className={s.source}>{example.source}.</span>
                </p>
                <p className={s.path} title={example.path}>
                  <span className={s.pathLabel}>{example.copied ? "Your copy:" : "Copies to"}</span>
                  {/* The left-to-right mark keeps the leading slash in place when truncating from the left. */}
                  <span className={s.pathText}>{"\u200e" + example.path}</span>
                </p>
              </div>
              <div className={s.actions}>
                <Button onClick={() => void open(example)} disabled={disabled || opening !== null}>
                  {opening === example.id ? "Opening…" : action}
                  <span className="visually-hidden">{example.copied ? ` of ${example.name}` : ` ${example.name}`}</span>
                </Button>
                {offerWalkthrough && example.id === TOUR_EXAMPLE && (
                  <Button
                    variant="ghost"
                    onClick={() => void open(example, true)}
                    disabled={disabled || opening !== null}
                    tip="Open the example, and show its prompts, baseline, experiment and results step by step"
                  >
                    Walk me through it
                  </Button>
                )}
              </div>
            </li>
          );
        })}
      </ul>
    </>
  );
}
