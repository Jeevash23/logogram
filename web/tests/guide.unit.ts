import { test, expect } from "@playwright/test";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { GUIDE_PAGES } from "../src/lib/guide";
import { guideUrl, REPOSITORY } from "../src/lib/help";
import { emphasis, STEPS, TOUR_EXAMPLE, type TourContext } from "../src/lib/walkthrough";
import { useStore, VIEWS } from "../src/store/app";
import { endTour, goToStep, openGuide, startTour, useHelp } from "../src/store/help";
import { anchorsOf, webSources } from "./docs";

// The in-app guide and the walkthrough of the example: the documentation they point to exists,
// with its headings; every control the walkthrough points at carries its data-tour attribute in
// the code (a control renamed or moved by a change elsewhere fails here, not in front of someone
// new); and how the walkthrough ended is remembered in the person's settings.

const CONTEXTS: TourContext[] = [true, false].flatMap((hasRun) =>
  [true, false].flatMap((modelLoaded) =>
    [true, false].map((wide) => ({ hasRun, finished: hasRun, modelLoaded, wide })),
  ),
);

test("the guide's pages point to documentation the repository has", () => {
  expect(GUIDE_PAGES.map((p) => p.id)).toEqual(["workspaces", "map", "numbers", "keys"]);
  const views = new Set(VIEWS.map((v) => v.id));
  for (const page of GUIDE_PAGES) {
    expect(page.docs.length).toBeGreaterThan(0);
    for (const doc of page.docs) {
      expect(existsSync(fileURLToPath(new URL(`../../${doc.path}`, import.meta.url))), doc.path).toBe(true);
      if (doc.anchor) expect(anchorsOf(doc.path), `${doc.path}#${doc.anchor}`).toContain(doc.anchor);
      expect(guideUrl(doc).startsWith(`${REPOSITORY}/blob/main/docs/`)).toBe(true);
    }
    for (const link of page.views) expect(views.has(link.view)).toBe(true);
  }
});

test("every control the walkthrough points at is marked for it in the code", () => {
  const sources = webSources();
  expect(STEPS.length).toBeGreaterThanOrEqual(5);
  expect(STEPS.length).toBeLessThanOrEqual(6);
  for (const step of STEPS) {
    for (const name of [...step.anchors, ...(step.alternatives ?? [])]) {
      expect(sources, `no control carries data-tour="${name}"`).toContain(`data-tour="${name}"`);
    }
  }
  // The tabs a step falls back to.
  expect(sources).toContain("data-tour={`tab-${v.id}`}");
  // The bundled example the walkthrough is written for.
  expect(existsSync(fileURLToPath(new URL(`../../src/logogram/examples/${TOUR_EXAMPLE}/project.json`, import.meta.url)))).toBe(true);
});

test("each step shows a view, and says something in every state the app can be in", () => {
  const views = new Set(VIEWS.map((v) => v.id));
  expect(new Set(STEPS.map((s) => s.id)).size).toBe(STEPS.length);
  for (const step of STEPS) {
    expect(step.title).toMatch(/^[A-Z][a-z]/);
    for (const ctx of CONTEXTS) {
      expect(views.has(step.view(ctx))).toBe(true);
      const text = step.text(ctx);
      expect(text.length).toBeGreaterThan(0);
      for (const paragraph of text) {
        expect(paragraph.trim()).toMatch(/[.!?]$/);
        // Names of controls are paired: none is left open.
        expect(paragraph.split("**").length % 2).toBe(1);
      }
    }
  }
  // The map is the run's once there is one, else the loaded model's.
  const map = STEPS.find((s) => s.id === "map")!;
  expect(map.view({ hasRun: true, finished: true, modelLoaded: true, wide: true })).toBe("results");
  expect(map.view({ hasRun: false, finished: false, modelLoaded: true, wide: true })).toBe("explore");
  expect(map.view({ hasRun: false, finished: false, modelLoaded: false, wide: true })).toBe("results");
});

test("leaving the workbench or the project ends the walkthrough, without saying it was skipped", () => {
  const original = globalThis.fetch;
  const sent: string[] = [];
  globalThis.fetch = (async (input: string) => {
    sent.push(input);
    return new Response("{}");
  }) as typeof fetch;
  const project = (session_id: string) => ({ session_id }) as unknown as NonNullable<ReturnType<typeof useStore.getState>["project"]>;
  const before = useStore.getState();
  try {
    useStore.setState({ screen: "workbench", project: project("a"), walkthrough: null });
    startTour();
    goToStep(2);
    // A view changes, or a run opens: it goes on.
    useStore.setState({ view: "results" });
    expect(useHelp.getState().tourStep).toBe(2);
    // All projects: it ends.
    useStore.setState({ screen: "projects" });
    expect(useHelp.getState().tourStep).toBeNull();
    // Another project: it ends too.
    useStore.setState({ screen: "workbench" });
    startTour();
    useStore.setState({ project: project("b") });
    expect(useHelp.getState().tourStep).toBeNull();
    // Neither is how it ended: the first screens keep offering it.
    expect(useStore.getState().walkthrough).toBeNull();
    expect(sent).toEqual([]);
  } finally {
    globalThis.fetch = original;
    useStore.setState({ screen: before.screen, project: before.project, view: before.view, walkthrough: null });
    useHelp.setState({ tourStep: null });
  }
});

test("names of controls are set apart from the words around them", () => {
  expect(emphasis("Press **Run**: it starts.")).toEqual([
    { text: "Press ", strong: false },
    { text: "Run", strong: true },
    { text: ": it starts.", strong: false },
  ]);
  expect(emphasis("**Check robustness**")).toEqual([{ text: "Check robustness", strong: true }]);
});

test("the walkthrough moves step by step, and how it ended is kept in the settings", async () => {
  const sent: unknown[] = [];
  const original = globalThis.fetch;
  globalThis.fetch = (async (input: string, init?: RequestInit) => {
    if (input === "/api/settings") sent.push(JSON.parse(String(init?.body)));
    return new Response("{}");
  }) as typeof fetch;
  try {
    useStore.setState({ walkthrough: null, helpDialog: "guide" });
    startTour();
    expect(useHelp.getState().tourStep).toBe(0);
    // Starting it closes the guide it was started from.
    expect(useStore.getState().helpDialog).toBeNull();
    goToStep(1);
    goToStep(99);
    expect(useHelp.getState().tourStep).toBe(STEPS.length - 1);
    goToStep(-3);
    expect(useHelp.getState().tourStep).toBe(0);
    endTour("skipped");
    expect(useHelp.getState().tourStep).toBeNull();
    expect(useStore.getState().walkthrough).toBe("skipped");
    startTour();
    endTour("finished");
    expect(useStore.getState().walkthrough).toBe("finished");
    await new Promise((resolve) => globalThis.setTimeout(resolve, 0));
    expect(sent).toEqual([{ walkthrough: "skipped" }, { walkthrough: "finished" }]);
    // The guide opens where it is asked to.
    openGuide("numbers");
    expect(useStore.getState().helpDialog).toBe("guide");
    expect(useHelp.getState().guidePage).toBe("numbers");
  } finally {
    globalThis.fetch = original;
    useStore.setState({ walkthrough: null, helpDialog: null });
    useHelp.setState({ tourStep: null, guidePage: "workspaces" });
  }
});
