import type { ReactNode } from "react";

import { BRAND_SEED, Logogram } from "../components/Logogram";
import { useStore } from "../store/app";
import s from "./Screens.module.css";

/**
 * The first screens: a dark chamber with a lit screen and a logogram on it, after the film's
 * first contact. The work sits beside it, on paper.
 */
export function Welcome({ children }: { children: ReactNode }) {
  const version = useStore((st) => st.version);
  return (
    <div className={s.welcome}>
      <aside className={s.chamber} aria-label="Logogram">
        <div className={s.brand}>
          <span className={s.wordmark}>Logogram</span>
          <span className={s.version}>{version}</span>
        </div>
        <div className={s.screen}>
          <Logogram seed={BRAND_SEED} size={300} detail={1} weight={1.05} className={s.heroGlyph} />
        </div>
        <div className={s.chamberText}>
          <p className={s.tagline}>Causal experiments inside language models.</p>
          <p className={s.chamberNote}>Patch and ablate what a model computes, and read the evidence prompt by prompt. Everything runs on this computer.</p>
        </div>
      </aside>
      <div className={s.paper}>{children}</div>
    </div>
  );
}
