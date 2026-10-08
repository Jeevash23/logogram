import type { ReactNode } from "react";

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
          <span style={{ color: "var(--explore)" }}>Explore</span> what a model computes.{" "}
          <span style={{ color: "var(--experiment)" }}>Experiment</span> on it, one component at a time.{" "}
          <span style={{ color: "var(--evidence)" }}>Read the evidence</span>, prompt by prompt.
        </p>
      </aside>
      <div className={s.paper}>{children}</div>
    </div>
  );
}
