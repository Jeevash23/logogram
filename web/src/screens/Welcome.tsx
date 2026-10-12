import type { CSSProperties, ReactNode } from "react";

import { BRAND_SEED, Logogram } from "../components/Logogram";
import { useStore } from "../store/app";
import s from "./Screens.module.css";

/**
 * The first screens: the logogram over a bloom of the three workspace colors, and what each
 * workspace is for, in its color. The work sits beside it.
 */
export function Welcome({ children }: { children: ReactNode }) {
  const version = useStore((st) => st.version);
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
      </aside>
      <main className={s.paper}>{children}</main>
    </div>
  );
}

/** A workspace's word, underlined in its color: text on the workspace colors is ink. */
function word(color: string): CSSProperties {
  return { "--word": `var(${color})` } as CSSProperties;
}
