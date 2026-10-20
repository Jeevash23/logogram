import { useEffect, useId, useRef, useState, type CSSProperties, type ReactNode } from "react";

import { usePrefersReducedMotion } from "../../lib/motion";
import { Icon, Tip, useRadioGroup } from "../ui";
import s from "./Figures.module.css";

/** One stage of a figure: what its button says, and what it shows, in a sentence. */
export interface FigureStage {
  title: string;
  text: string;
}

/** How long a stage stays before the next, unless the figure says otherwise. */
const STAGE_MS = 3000;

/**
 * An explanatory figure that moves through stages, as Start here and the guide draw them: a
 * drawing, and under it the stages, each a button with a line that fills while it plays. It plays
 * through them and starts again while it is in view and the page is shown; choosing a stage shows
 * it and stops there, and Play goes on from it. When less motion is asked for, it doesn't play: it
 * shows its last stage, which holds the whole picture, and the stages can still be chosen.
 *
 * The drawing is `children(stage, cycle)`: it draws a stage, and moves into it with CSS animations
 * that play as its marks appear, which Figures.module.css turns off under reduced motion.
 *
 * In the "list" layout the stages are a list beside the drawing, each with its sentence, all in
 * view: a list of terms, say, whose drawing shows the one chosen.
 *
 * The stages are a radio group: one Tab stop, the arrow keys move from stage to stage and show it.
 */
export function StagedFigure({
  label,
  stages,
  children,
  caption,
  ms = STAGE_MS,
  layout = "row",
  still,
  className,
}: {
  /** What the figure is, for its play button and for assistive technology. */
  label: string;
  stages: FigureStage[];
  /** Draws a stage; `cycle` changes each time the figure starts over. */
  children: (stage: number, cycle: number) => ReactNode;
  /** A line about the whole figure, under its stages. */
  caption?: string;
  /** How long each stage stays: one for all, or one each. */
  ms?: number | number[];
  layout?: "row" | "list";
  /** The stage shown under reduced motion: the last by default, or -1 for none, when the drawing
   * shows everything at once until a stage is chosen. */
  still?: number;
  className?: string;
}) {
  const reduced = usePrefersReducedMotion();
  const last = stages.length - 1;
  // `run` counts the stages shown, so that a stage shown again starts its line again.
  // `cycle` counts the times it has started over, so a drawing can draw its first stage afresh.
  const rest = still ?? last;
  const [shown, setShown] = useState(() => ({ stage: reduced ? rest : 0, run: 0, cycle: 0 }));
  const [playing, setPlaying] = useState(!reduced);
  const [inView, setInView] = useState(false);
  const [pageShown, setPageShown] = useState(() => typeof document === "undefined" || !document.hidden);
  const root = useRef<HTMLElement>(null);
  const spent = useRef({ run: 0, ms: 0 });
  const id = useId();
  const textId = `${id}-text`;

  // It moves only while enough of it is in view, and the page is shown.
  useEffect(() => {
    const el = root.current;
    if (!el || typeof IntersectionObserver === "undefined") {
      setInView(true);
      return;
    }
    const watch = new IntersectionObserver(([entry]) => setInView(entry.isIntersecting), { threshold: 0.35 });
    watch.observe(el);
    return () => watch.disconnect();
  }, []);
  useEffect(() => {
    const change = () => setPageShown(!document.hidden);
    document.addEventListener("visibilitychange", change);
    return () => document.removeEventListener("visibilitychange", change);
  }, []);
  // Less motion asked for, now or later: the whole picture, still.
  useEffect(() => {
    if (!reduced) return;
    setPlaying(false);
    setShown((st) => ({ ...st, stage: rest, run: st.run + 1 }));
  }, [reduced, rest]);

  const duration = (stage: number) => (Array.isArray(ms) ? (ms[stage] ?? STAGE_MS) : ms);
  const moving = playing && inView && pageShown && !reduced;
  // The next stage once this one has had its time, counting the time it was shown before a pause.
  useEffect(() => {
    if (!moving) return;
    const { stage, run } = shown;
    if (spent.current.run !== run) spent.current = { run, ms: 0 };
    const begun = performance.now();
    const timer = window.setTimeout(
      () =>
        setShown((st) => {
          const next = (st.stage + 1) % stages.length;
          return { stage: next, run: st.run + 1, cycle: next === 0 ? st.cycle + 1 : st.cycle };
        }),
      Math.max(0, duration(stage) - spent.current.ms),
    );
    return () => {
      window.clearTimeout(timer);
      if (spent.current.run === run) spent.current.ms += performance.now() - begun;
    };
    // `duration` follows `ms`, which callers write as a literal.
  }, [moving, shown, stages.length]);

  const choose = (stage: number) => {
    setPlaying(false);
    setShown((st) => ({ ...st, stage, run: st.run + 1 }));
  };
  const toggle = () => {
    if (playing) {
      setPlaying(false);
      return;
    }
    // Played again from the end, it starts over.
    if (shown.stage === last) setShown((st) => ({ stage: 0, run: st.run + 1, cycle: st.cycle + 1 }));
    setPlaying(true);
  };

  const stage = shown.stage;
  const radio = useRadioGroup(
    stages.map((_, i) => ({ value: i })),
    stage,
    (i) => choose(i),
  );
  const line = (i: number) => {
    const state = i < stage ? "done" : i > stage ? "todo" : moving ? "playing" : "now";
    const style =
      state === "playing"
        ? ({ animationDuration: `${duration(i)}ms`, animationDelay: `-${Math.round(spent.current.run === shown.run ? spent.current.ms : 0)}ms` } as CSSProperties)
        : undefined;
    return (
      <span className={s.line} data-state={state} aria-hidden="true">
        <span key={`${shown.run}`} style={style} />
      </span>
    );
  };
  const play = reduced ? null : (
    <Tip text={playing ? "Pause" : "Play"}>
      <button type="button" className={s.play} onClick={toggle} aria-label={`${playing ? "Pause" : "Play"} the figure: ${label}`}>
        <Icon name={playing ? "pause" : "play"} size={14} />
      </button>
    </Tip>
  );

  if (layout === "list") {
    return (
      <figure ref={root} className={`${s.figure} ${s.split} ${className ?? ""}`} aria-label={label}>
        <div className={s.drawing} aria-hidden="true">
          {children(stage, shown.cycle)}
        </div>
        <div className={s.side}>
          <div className={s.items} role="radiogroup" aria-label={label}>
            {stages.map((st, i) => (
              <button
                key={st.title}
                type="button"
                role="radio"
                aria-checked={i === stage}
                aria-labelledby={`${id}-${i}-title`}
                aria-describedby={`${id}-${i}-text`}
                className={s.item}
                data-current={i === stage || undefined}
                {...radio(i)}
                onClick={() => choose(i)}
              >
                <span id={`${id}-${i}-title`} className={s.itemTitle}>
                  {st.title}
                </span>
                <span id={`${id}-${i}-text`} className={s.itemText}>
                  {st.text}
                </span>
              </button>
            ))}
          </div>
          {(play || caption) && (
            <div className={s.listFoot}>
              {play}
              {caption && <figcaption className={s.caption}>{caption}</figcaption>}
            </div>
          )}
        </div>
      </figure>
    );
  }

  return (
    <figure ref={root} className={`${s.figure} ${className ?? ""}`} aria-labelledby={textId}>
      <div className={s.drawing} aria-hidden="true">
        {children(stage, shown.cycle)}
      </div>
      <div className={s.controls}>
        {play}
        <div className={s.stages} role="radiogroup" aria-label={`Stages of the figure: ${label}`}>
          {stages.map((st, i) => (
            <button
              key={st.title}
              type="button"
              role="radio"
              aria-checked={i === stage}
              className={s.stage}
              {...radio(i)}
              onClick={() => choose(i)}
            >
              {line(i)}
              <span className={s.stageTitle}>
                <span className={s.stageNumber}>{i + 1}</span> {st.title}
              </span>
            </button>
          ))}
        </div>
      </div>
      <figcaption id={textId} className={s.caption}>
        {/* Read out when someone chooses a stage, not while it plays by itself. */}
        <span className={s.stageText} aria-live={moving ? "off" : "polite"}>
          {stages[stage]?.text}
        </span>
        {caption && <> {caption}</>}
      </figcaption>
    </figure>
  );
}
