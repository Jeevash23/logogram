import { useEffect, useId, useRef, type CSSProperties, type ReactNode } from "react";

import { BRAND_SEED, Logogram } from "../components/Logogram";
import { Button } from "../components/ui";
import { PURPOSE, START_NOTE } from "../lib/start";
import { useStore } from "../store/app";
import { openGuide, useHelp } from "../store/help";
import s from "./Screens.module.css";

/**
 * The first screens: the logogram over a bloom of the three workspace colors, what each workspace
 * is for, in its color, and what Logogram finds, plainly. Under it, Start here opens the guide at
 * its first page: the first thing the keyboard reaches, it stays in view while the work beside it
 * scrolls, and it is the solid ink button until the walkthrough of the example has been finished
 * or skipped.
 */
export function Welcome({ children }: { children: ReactNode }) {
  const version = useStore((st) => st.version);
  const fresh = useStore((st) => st.walkthrough === null);
  const noteId = useId();
  const start = useRef<HTMLButtonElement>(null);
  const opened = useRef(false);

  // Radix gives the focus back only to a dialog's own trigger. When the guide this button opened
  // closes (or the list of shortcuts it can hand over to), the keyboard comes back here, unless
  // the guide went somewhere else.
  useEffect(
    () =>
      useStore.subscribe((st, prev) => {
        const isOpen = (state: typeof st) => state.helpDialog === "guide" || state.shortcutsOpen;
        if (!opened.current || !isOpen(prev) || isOpen(st)) return;
        opened.current = false;
        if (!useHelp.getState().toView) window.setTimeout(() => start.current?.focus(), 0);
      }),
    [],
  );

  return (
    <div className={s.welcome}>
      <aside className={s.hero} aria-label="Logogram">
        <div className={s.brand}>
          <span className={s.wordmark}>Logogram</span>
          <span className={s.version}>{version}</span>
        </div>
        <div className={s.stage} aria-hidden="true">
          <span className={s.bloomExplore} />
          <span className={s.bloomExperiment} />
          <span className={s.bloomEvidence} />
          <Logogram seed={BRAND_SEED} size={280} detail={1} weight={1.05} haze={false} className={s.heroGlyph} />
        </div>
        <p className={s.tagline}>
          <span className={s.word} style={word("--explore")}>Explore</span> what a model computes.{" "}
          <span className={s.word} style={word("--experiment")}>Experiment</span> on it, one component at a time.{" "}
          <span className={s.word} style={word("--evidence")}>Read the evidence</span>, prompt by prompt.
        </p>
        <p className={s.purpose}>{PURPOSE}</p>
        <div className={s.start}>
          <Button
            ref={start}
            variant={fresh ? "primary" : "secondary"}
            onClick={() => {
              opened.current = true;
              openGuide("start");
            }}
            aria-describedby={noteId}
          >
            Start here
          </Button>
          <p id={noteId} className={s.startNote}>
            {START_NOTE}
          </p>
        </div>
      </aside>
      <main className={s.paper}>{children}</main>
    </div>
  );
}

/** A workspace's word, underlined in its color: text on the workspace colors is ink. */
function word(color: string): CSSProperties {
  return { "--word": `var(${color})` } as CSSProperties;
}
