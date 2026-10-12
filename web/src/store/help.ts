// The in-app help: the guide's open page, and the walkthrough of the example, step by step. How a
// walkthrough ended is the person's own setting (on the server, in their settings folder), never
// the project's.

import { create } from "zustand";

import { api } from "../api/client";
import type { GuidePageId } from "../lib/guide";
import { STEPS, TOUR_EXAMPLE } from "../lib/walkthrough";
import { useStore } from "./app";

interface HelpState {
  /** The guide's page, kept while it is closed, so it opens where it was left. */
  guidePage: GuidePageId;
  /** The walkthrough's step, from 0, or null while it isn't running. */
  tourStep: number | null;
  /** Set when a help dialog closes to show a view: the keyboard goes to that view, not to Help. */
  toView: boolean;
}

export const useHelp = create<HelpState>(() => ({ guidePage: "workspaces", tourStep: null, toView: false }));

// Leaving the workbench, or the project the walkthrough runs in, ends it without a word: it was
// neither finished nor skipped, so the first screens keep offering it.
useStore.subscribe((st, prev) => {
  if (useHelp.getState().tourStep === null) return;
  if (st.screen !== "workbench" || st.project?.session_id !== prev.project?.session_id) useHelp.setState({ tourStep: null });
});

/** Open the guide, at a page if given. */
export function openGuide(page?: GuidePageId): void {
  if (page) useHelp.setState({ guidePage: page });
  useStore.setState({ helpDialog: "guide" });
}

/** Start the walkthrough at its first step, in the project that is open. */
export function startTour(): void {
  useStore.setState({ helpDialog: null, shortcutsOpen: false, paletteOpen: false });
  useHelp.setState({ tourStep: 0 });
}

/** Go to a step of the walkthrough. */
export function goToStep(step: number): void {
  useHelp.setState({ tourStep: Math.max(0, Math.min(STEPS.length - 1, step)) });
}

/** End the walkthrough, and remember how, so the first screens stop offering it. */
export function endTour(how: "finished" | "skipped"): void {
  useHelp.setState({ tourStep: null });
  useStore.setState({ walkthrough: how });
  void api.settings({ walkthrough: how }).catch(() => undefined);
}

/**
 * Walk through the bundled example: open it (copied into the projects folder once, as the
 * gallery opens it) unless it is the project already open, then start at the first step.
 * Returns whether it started.
 */
export async function walkThroughExample(): Promise<boolean> {
  const st = useStore.getState();
  const examples = await st.guard(() => api.examples());
  const example = examples?.find((e) => e.id === TOUR_EXAMPLE);
  if (!example) return false;
  if (st.project?.path !== example.path) {
    const project = await st.guard(() => api.openExample(TOUR_EXAMPLE));
    if (!project) return false;
    await useStore.getState().enterProject(project);
  } else if (st.screen !== "workbench") {
    st.goto("workbench");
  }
  startTour();
  return true;
}
