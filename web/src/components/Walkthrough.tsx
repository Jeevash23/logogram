import * as Popover from "@radix-ui/react-popover";
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { createPortal } from "react-dom";

import { usePrefersReducedMotion } from "../lib/motion";
import { emphasis, STEPS, type TourContext, type TourStep } from "../lib/walkthrough";
import { useStore } from "../store/app";
import { endTour, goToStep, useHelp } from "../store/help";
import { Button } from "./ui";
import s from "./Walkthrough.module.css";

/** How long a step waits for its control to appear before pointing at its view's tab instead. */
const PATIENCE_MS = 700;
/** The space between a control and its ink outline. */
const PAD = 4;
/** Below this width the inspector opens as a panel (as in screens/Workbench.tsx). */
const WIDE = 1120;

/**
 * The walkthrough of the example: one step at a time, a popover beside a real control, which an
 * ink outline marks. Nothing dims or blocks the page: it can be used while a step is shown. Next,
 * Back, Skip and Escape move through it or end it; the keyboard goes to each step as it opens,
 * Tab from its last button goes on to the control it points at, and when it ends the keyboard goes
 * back to the page.
 */
export function Walkthrough() {
  const step = useHelp((st) => st.tourStep);
  const screen = useStore((st) => st.screen);
  // Leaving the workbench ends it (store/help.ts).
  if (step === null || screen !== "workbench") return null;
  return <TourStepView key={step} index={step} />;
}

function useWidth(): number {
  const [width, setWidth] = useState(() => window.innerWidth);
  useEffect(() => {
    const resize = () => setWidth(window.innerWidth);
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);
  return width;
}

/** What the step's view and words depend on, from the app's state. */
function useTourContext(wide: boolean): TourContext {
  const hasRun = useStore((st) => {
    const id = st.activeRunId;
    if (!id) return false;
    const status = st.runs.find((r) => r.id === id)?.status;
    return !!st.live[id] || !!st.runDetails[id]?.summary || (status !== undefined && status !== "draft");
  });
  const finished = useStore((st) => !!st.activeRunId && st.runs.find((r) => r.id === st.activeRunId)?.status === "finished");
  const modelLoaded = useStore((st) => st.model.state === "ready");
  return useMemo(() => ({ hasRun, finished, modelLoaded, wide }), [hasRun, finished, modelLoaded, wide]);
}

function shown(name: string): HTMLElement | null {
  for (const el of document.querySelectorAll<HTMLElement>(`[data-tour="${name}"]`)) {
    const box = el.getBoundingClientRect();
    if (box.width > 0 && box.height > 0) return el;
  }
  return null;
}

/** The step's controls on the page now: its anchors, else an alternative, else (once it has waited
 * for them) the tab of its view, else the view itself. */
function locate(step: TourStep, view: string, waited: boolean): HTMLElement[] {
  const anchors = step.anchors.map(shown).filter((el): el is HTMLElement => el !== null);
  if (anchors.length) return anchors;
  for (const name of step.alternatives ?? []) {
    const el = shown(name);
    if (el) return [el];
  }
  if (!waited) return [];
  const tab = shown(`tab-${view}`) ?? document.getElementById("main");
  return tab ? [tab] : [];
}

const FOCUSABLE = "button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex='-1'])";

/** The control in an element the keyboard can go to: the element itself, or the first inside it. */
function focusable(el: HTMLElement): HTMLElement | null {
  if (el.matches(FOCUSABLE)) return el;
  return el.querySelector<HTMLElement>(FOCUSABLE);
}

/**
 * The part of an element that can be seen: its box, cut by the window and by every ancestor that
 * clips what overflows it (a table that scrolls, a panel). Null when none of it shows.
 */
function visibleBox(el: HTMLElement): DOMRect | null {
  const box = el.getBoundingClientRect();
  let { left, top, right, bottom } = box;
  for (let node = el.parentElement; node && node !== document.body; node = node.parentElement) {
    const style = getComputedStyle(node);
    if (style.overflowX === "visible" && style.overflowY === "visible") continue;
    const clip = node.getBoundingClientRect();
    if (style.overflowX !== "visible") {
      left = Math.max(left, clip.left);
      right = Math.min(right, clip.right);
    }
    if (style.overflowY !== "visible") {
      top = Math.max(top, clip.top);
      bottom = Math.min(bottom, clip.bottom);
    }
  }
  left = Math.max(left, 0);
  top = Math.max(top, 0);
  right = Math.min(right, window.innerWidth);
  bottom = Math.min(bottom, window.innerHeight);
  return right > left && bottom > top ? new DOMRect(left, top, right - left, bottom - top) : null;
}

/** The part of the window a control scrolls in: its nearest ancestor that scrolls up and down, or
 * the window. */
function scrollArea(el: HTMLElement): { top: number; height: number } {
  for (let node = el.parentElement; node && node !== document.body; node = node.parentElement) {
    const overflow = getComputedStyle(node).overflowY;
    if ((overflow === "auto" || overflow === "scroll") && node.scrollHeight > node.clientHeight) {
      const box = node.getBoundingClientRect();
      const top = Math.max(box.top, 0);
      return { top, height: Math.min(box.bottom, window.innerHeight) - top };
    }
  }
  return { top: 0, height: window.innerHeight };
}

/** Scroll a control into view when too little of it shows: all of one that fits, else its top and
 * a good part of the rest. */
function bringIntoView(el: HTMLElement, smooth: boolean): void {
  const box = el.getBoundingClientRect();
  const seen = visibleBox(el);
  const area = scrollArea(el);
  const fits = box.height <= area.height;
  const shown = seen !== null && (fits ? seen.height >= box.height - 1 : seen.top <= box.top + 1 && seen.height >= area.height * 0.4);
  if (!shown) el.scrollIntoView({ block: fits ? "center" : "start", behavior: smooth ? "smooth" : "auto" });
}

/** How far a step stands from its control, and from the window's edges. */
const OFFSET = 14;
const MARGIN = 16;

/**
 * Where the popover points: the part of the control in view, so that the step stands beside it,
 * above or below. When neither side has room for the step without covering the view's tabs and
 * headings (a table or a map that fills the view), the top edge of that part instead: a step
 * placed below it sits inside the control's top, over its first rows.
 */
function pointAt(el: HTMLElement, stepHeight: () => number) {
  return {
    contextElement: el,
    getBoundingClientRect(): DOMRect {
      const box = visibleBox(el) ?? el.getBoundingClientRect();
      const needed = stepHeight() + OFFSET + MARGIN;
      const above = box.top - scrollArea(el).top;
      const below = window.innerHeight - box.bottom;
      if (above >= needed || below >= needed) return box;
      return new DOMRect(box.left, box.top, box.width, 0);
    },
  };
}

function same(a: HTMLElement[], b: HTMLElement[]): boolean {
  return a.length === b.length && a.every((el, i) => el === b[i]);
}

function TourStepView({ index }: { index: number }) {
  const step = STEPS[index];
  const last = index === STEPS.length - 1;
  const reduced = usePrefersReducedMotion();
  const wide = useWidth() >= WIDE;
  const ctx = useTourContext(wide);
  const view = step.view(ctx);
  const [targets, setTargets] = useState<HTMLElement[]>([]);
  const content = useRef<HTMLDivElement>(null);
  const titleId = `tour-${step.id}-title`;

  // Show the step's view (again, if the run it depends on starts or ends meanwhile).
  useEffect(() => {
    const st = useStore.getState();
    if (st.view !== view) st.setView(view);
  }, [view]);

  // Follow the step's controls as the view draws them, and as they come and go.
  useEffect(() => {
    const started = performance.now();
    let frame = 0;
    const find = () => {
      const found = locate(step, view, performance.now() - started > PATIENCE_MS);
      setTargets((prev) => (same(prev, found) ? prev : found));
      frame = requestAnimationFrame(find);
    };
    find();
    return () => cancelAnimationFrame(frame);
  }, [step, view]);

  // Bring the control into view once it is found.
  const first = targets[0] ?? null;
  useEffect(() => {
    if (first) bringIntoView(first, !reduced);
  }, [first, reduced]);

  const anchor = useRef<ReturnType<typeof pointAt> | null>(null);
  anchor.current = useMemo(() => (first ? pointAt(first, () => content.current?.offsetHeight ?? 260) : null), [first]);

  const finish = (how: "finished" | "skipped") => {
    // Back to the page: the control the step pointed at, or else the view.
    const control = targets.map(focusable).find((el) => el !== null) ?? null;
    endTour(how);
    window.setTimeout(() => (control ?? document.getElementById("main"))?.focus(), 0);
  };

  // Tab from the step's last button goes on to its control, so the keyboard can use it too.
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "Tab" || e.shiftKey || e.altKey || e.ctrlKey || e.metaKey) return;
    const buttons = content.current?.querySelectorAll<HTMLElement>("button:not([disabled])");
    if (!buttons?.length || document.activeElement !== buttons[buttons.length - 1]) return;
    const control = targets.map(focusable).find((el) => el !== null);
    if (!control) return;
    e.preventDefault();
    control.focus();
  };

  if (!anchor.current) return null;
  return (
    <>
      {targets.map((el, i) => (
        <Outline key={i} el={el} />
      ))}
      <Popover.Root open onOpenChange={(open) => !open && finish("skipped")}>
        <Popover.Anchor virtualRef={anchor} />
        <Popover.Portal>
          <Popover.Content
            className={s.step}
            data-tour-step={step.id}
            aria-labelledby={titleId}
            side={step.side}
            align={step.align}
            sideOffset={OFFSET}
            collisionPadding={MARGIN}
            updatePositionStrategy="always"
            onOpenAutoFocus={(e) => {
              e.preventDefault();
              // After whatever closed as the step opened (a menu, the guide) has put the focus
              // back where it was.
              window.setTimeout(() => content.current?.focus({ preventScroll: true }), 0);
            }}
            // The keyboard's way back to the page is decided in finish().
            onCloseAutoFocus={(e) => e.preventDefault()}
            // The page stays usable: clicking or tabbing into it leaves the step where it is.
            onInteractOutside={(e) => e.preventDefault()}
          >
            <div ref={content} className={s.inner} tabIndex={-1} onKeyDown={onKeyDown}>
              <p className={s.count}>
                Walkthrough · {index + 1} of {STEPS.length}
              </p>
              <Popover.Title id={titleId} className={s.title}>
                {step.title}
              </Popover.Title>
              <Popover.Description asChild>
                <div className={s.body}>
                  {step.text(ctx).map((paragraph, i) => (
                    <p key={i}>
                      {emphasis(paragraph).map((part, j) => (part.strong ? <strong key={j}>{part.text}</strong> : part.text))}
                    </p>
                  ))}
                </div>
              </Popover.Description>
              <div className={s.actions}>
                <Button size="small" variant="ghost" onClick={() => finish("skipped")}>
                  Skip
                </Button>
                <span className={s.spacer} />
                {index > 0 && (
                  <Button size="small" onClick={() => goToStep(index - 1)}>
                    Back
                  </Button>
                )}
                <Button size="small" variant="primary" onClick={() => (last ? finish("finished") : goToStep(index + 1))}>
                  {last ? "Finish" : "Next"}
                </Button>
              </div>
            </div>
          </Popover.Content>
        </Popover.Portal>
      </Popover.Root>
    </>
  );
}

/** An ink outline around the part of a control in view, following it as the page scrolls and
 * changes. It takes no clicks: the control under it works as before. */
function Outline({ el }: { el: HTMLElement }) {
  const [box, setBox] = useState<DOMRect | null>(null);
  useLayoutEffect(() => {
    let frame = 0;
    const follow = () => {
      const next = visibleBox(el);
      setBox((prev) =>
        prev && next && prev.x === next.x && prev.y === next.y && prev.width === next.width && prev.height === next.height ? prev : next,
      );
      frame = requestAnimationFrame(follow);
    };
    follow();
    return () => cancelAnimationFrame(frame);
  }, [el]);
  if (!box) return null;
  // Inside the window, so that a control at its edge (the inspector) is outlined on every side.
  const left = Math.max(box.left - PAD, 0);
  const top = Math.max(box.top - PAD, 0);
  const right = Math.min(box.right + PAD, window.innerWidth);
  const bottom = Math.min(box.bottom + PAD, window.innerHeight);
  return createPortal(
    <div
      className={s.outline}
      data-tour-outline=""
      aria-hidden="true"
      style={{ top, left, width: right - left, height: bottom - top }}
    />,
    document.body,
  );
}
