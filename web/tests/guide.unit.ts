import { test, expect } from "@playwright/test";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { GUIDE_FIGURES, GUIDE_PAGES } from "../src/lib/guide";
import { guideUrl, REPOSITORY } from "../src/lib/help";
import { componentLabel, type Selection } from "../src/lib/sites";
import { IOI_HEADS, METER, PAIR, PATCHED_HEAD, patchedAt, patchedEffect, promptText } from "../src/lib/figures";
import { METHODS, QUESTIONS } from "../src/lib/questions";
import { FIGURE_CAPTION, START, START_NOTE, figureStages, startTexts, startWords, type StartAction } from "../src/lib/start";
import { emphasis, FIRST_RUN, STEPS, TOUR_EXAMPLE, type TourContext } from "../src/lib/walkthrough";
import { useStore, VIEWS } from "../src/store/app";
import { endTour, goToStep, openGuide, startTour, useHelp } from "../src/store/help";
import { anchorsOf, readRepo, slug, webSources } from "./docs";

// The in-app guide and the walkthrough of the example: the documentation they point to exists,
// with its headings; every control the walkthrough points at carries its data-tour attribute in
// the code (a control renamed or moved by a change elsewhere fails here, not in front of someone
// new); the guide's first page stays short and says what the bundled example says; and how the
// walkthrough ended is remembered in the person's settings.

/** The words a title in sentence case may still capitalize: names of models, parts and tasks. */
const NAMES = new Set(["GPT-2", "IOI", "MLP", "MLPs", "S-inhibition"]);

/** Whether a title is in sentence case: a capital first, then lower case but for names. */
function sentenceCase(title: string): boolean {
  const [first, ...rest] = title.split(" ");
  return (/^[A-Z][a-z]/.test(first) || NAMES.has(first)) && rest.every((word) => !/^[A-Z]/.test(word) || NAMES.has(word.replace(/[^\w-]/g, "")));
}

const CONTEXTS: TourContext[] = [true, false].flatMap((hasRun) =>
  [true, false].flatMap((modelLoaded) =>
    [true, false].flatMap((wide) =>
      [true, false].map((checked) => ({
        hasRun,
        finished: hasRun,
        modelLoaded,
        wide,
        baseline: checked ? { prefer: 29, n: 32 } : null,
        chosen: checked,
      })),
    ),
  ),
);

test("the guide's pages point to documentation the repository has", () => {
  expect(GUIDE_PAGES.map((p) => p.id)).toEqual(["start", "workspaces", "map", "numbers", "keys"]);
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
    expect(sentenceCase(step.title), step.title).toBe(true);
    for (const ctx of CONTEXTS) {
      expect(views.has(step.view(ctx))).toBe(true);
      const text = step.text(ctx);
      expect(text.length).toBeGreaterThan(0);
      for (const paragraph of text) {
        expect(paragraph.trim()).toMatch(/[.!?]$/);
        // Names of controls are paired: none is left open.
        expect(paragraph.split("**").length % 2).toBe(1);
      }
      const done = step.done?.(ctx);
      if (done) expect(done.trim()).toMatch(/[.!?]$/);
    }
  }
  // A step says when what it asks for is done; the experiment's moves on by itself once its run
  // starts, as pressing Run opens the run's results.
  const done = (id: string, ctx: TourContext) => STEPS.find((st) => st.id === id)!.done?.(ctx) ?? null;
  const blank: TourContext = { hasRun: false, finished: false, modelLoaded: true, wide: true, baseline: null, chosen: false };
  expect(done("baseline", blank)).toBeNull();
  expect(done("baseline", { ...blank, baseline: { prefer: 29, n: 32 } })).toContain("29 of 32");
  expect(done("site", { ...blank, chosen: true })).not.toBeNull();
  expect(STEPS.filter((st) => st.next).map((st) => st.id)).toEqual(["experiment"]);
  expect(STEPS.find((st) => st.id === "experiment")!.next!({ ...blank, hasRun: true })).toBe(true);
  // The map is the run's once there is one, else the loaded model's.
  const map = STEPS.find((s) => s.id === "map")!;
  const at = (ctx: Pick<TourContext, "hasRun" | "finished" | "modelLoaded">): TourContext => ({ ...ctx, wide: true, baseline: null, chosen: false });
  expect(map.view(at({ hasRun: true, finished: true, modelLoaded: true }))).toBe("results");
  expect(map.view(at({ hasRun: false, finished: false, modelLoaded: true }))).toBe("explore");
  expect(map.view(at({ hasRun: false, finished: false, modelLoaded: false }))).toBe("results");
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
    useHelp.setState({ tourStep: null, guidePage: "start" });
  }
});

/** A bundled example's field, as src/logogram/gallery.py writes it: string literals, joined. */
function galleryField(example: string, field: string): string {
  const gallery = readRepo("src/logogram/gallery.py");
  const start = gallery.indexOf(`id="${example}"`);
  expect(start, `gallery.py has no example ${example}`).toBeGreaterThanOrEqual(0);
  const end = gallery.indexOf("Example(", start);
  const block = gallery.slice(start, end === -1 ? undefined : end);
  const value = block.match(new RegExp(`${field}=((?:\\s*"[^"]*")+)`));
  expect(value, `${example} has no ${field}`).not.toBeNull();
  return [...value![1].matchAll(/"([^"]*)"/g)].map((m) => m[1]).join("");
}

/** How many words some text has, as startWords counts them. */
function words(texts: string[]): number {
  return texts
    .join(" ")
    .replace(/\*\*/g, "")
    .split(/\s+/)
    .filter((word) => /[\p{L}\p{N}]/u.test(word)).length;
}

test("Start here stays short and plain, and names only what the app has", () => {
  // Its figures carry most of it: the words to read stay few, and so does each figure's line.
  const prose = START.flatMap((section) => [
    ...(section.title ? [section.title] : []),
    ...section.paragraphs,
    ...(section.terms ?? []).flatMap((t) => [t.term, t.text]),
    ...(section.steps ?? []).flatMap((st) => [st.text, ...(st.action ? [st.action.label] : [])]),
    ...(section.actions ?? []).map((a) => a.label),
  ]);
  expect(words(prose)).toBeLessThanOrEqual(720);
  expect(startWords()).toBeLessThanOrEqual(900);
  // The parts of a model draw their terms, counted above; the other figures have lines of their own.
  const stages = START.flatMap((section) => section.figure?.stages ?? []);
  for (const stage of stages) {
    expect(words([stage.title]), stage.title).toBeLessThanOrEqual(3);
    expect(words([stage.text]), stage.text).toBeLessThanOrEqual(20);
    expect(sentenceCase(stage.title), stage.title).toBe(true);
  }
  // Every figure has stages to show, two or more.
  for (const section of START) if (section.figure) expect(figureStages(section).length, section.id).toBeGreaterThanOrEqual(2);
  // Every paragraph, definition, step, figure's line and caption is a sentence, with the names of
  // controls paired.
  const sentences = [
    ...START.flatMap((section) => [
      ...section.paragraphs,
      ...(section.terms ?? []).map((t) => t.text),
      ...(section.steps ?? []).map((st) => st.text),
      ...(section.figure?.stages ?? []).map((st) => st.text),
    ]),
    FIGURE_CAPTION,
    START_NOTE,
  ];
  for (const text of sentences) {
    expect(text.trim(), text).toMatch(/[.!?]$/);
    expect(text.split("**").length % 2, text).toBe(1);
  }
  // Titles in sentence case, none repeating the tab's name; only the lead has none.
  expect(new Set(START.map((section) => section.id)).size).toBe(START.length);
  for (const section of START) {
    if (section.id === "lead") expect(section.title).toBeNull();
    else expect(sentenceCase(section.title ?? ""), section.title ?? "").toBe(true);
    expect(section.title ?? "").not.toContain("Start here");
  }
  expect(START.find((section) => section.id === "hour")?.steps).toHaveLength(6);
  // Every view a button opens exists, and the walkthrough is offered from the first hour.
  const views = new Set(VIEWS.map((v) => v.id));
  const actions: StartAction[] = START.flatMap((section) => [...(section.actions ?? []), ...(section.steps ?? []).flatMap((st) => (st.action ? [st.action] : []))]);
  for (const action of actions) if (action.to === "view") expect(views.has(action.view), action.view).toBe(true);
  expect(actions.filter((a) => a.to === "tour")).toHaveLength(1);
  // Every control named in bold is labelled so in the file that draws it: renaming the control
  // there, or naming a new one here without saying where it is, fails here.
  const CONTROLS: Record<string, string> = {
    "Load a model": "web/src/components/Header.tsx",
    "Check the baseline": "web/src/views/BaselineView.tsx",
    Run: "web/src/components/RunButton.tsx",
    "Check robustness": "web/src/views/ResultsView.tsx",
    "Your text": "web/src/store/app.ts",
    "Create project": "web/src/screens/Projects.tsx",
    "Generate a task": "web/src/views/PromptsView.tsx",
  };
  const named = new Set(startTexts().flatMap((text) => emphasis(text).filter((part) => part.strong).map((part) => part.text)));
  expect([...named].sort()).toEqual(Object.keys(CONTROLS).sort());
  for (const name of named) {
    const label = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    expect(readRepo(CONTROLS[name]), `${CONTROLS[name]} has no control labelled ${name}`).toMatch(new RegExp(`(>\\s*${label}\\s*<|["\`]${label}["\`])`));
  }
  // A head is named as the page writes it and as the app's own views label it.
  const head = START.find((section) => section.id === "parts")!.terms!.find((t) => t.term === "Attention head")!.text;
  expect(head).toContain("9.9");
  expect(head).toContain(componentLabel({ part: "head", layer: 9, head: 9 } as Selection));
  // The download and the time are said as the walkthrough says them.
  const hour = START.find((section) => section.id === "hour")!.steps!.map((st) => st.text).join(" ");
  expect(hour).toContain(FIRST_RUN.download);
  expect(hour).toContain(FIRST_RUN.cpu);
  expect(STEPS.find((st) => st.id === "experiment")!.text(CONTEXTS[0]).join(" ")).toContain(`${FIRST_RUN.download}, and the sweep takes ${FIRST_RUN.cpu}`);
});

test("Start here says what the bundled IOI example says: its prompt and its heads", () => {
  const question = galleryField(TOUR_EXAMPLE, "question");
  const expected = galleryField(TOUR_EXAMPLE, "expect");
  const text = startTexts().join(" ");
  // The prompt the page asks about is the example's own.
  const quoted = [...text.matchAll(/“([^”]+)”/g)].map((m) => m[1]);
  expect(quoted.length).toBeGreaterThan(0);
  for (const prompt of quoted) expect(question, prompt).toContain(`“${prompt}”`);
  // Every head the page names (a layer and a head, such as 8.10) is one the example expects, and
  // the part of the gap they bring back is the example's words. A head may end a sentence; "0.5
  // GB" is not a head.
  const heads = [...text.matchAll(/(?<![\d.])(\d{1,2}\.\d{1,2})(?!\d|\.\d| GB)/g)].map((m) => m[1]);
  expect(new Set(heads).size).toBeGreaterThanOrEqual(8);
  for (const head of heads) expect(expected, head).toMatch(new RegExp(`(?<![\\d.])${head.replace(".", "\\.")}(?![\\d])`));
  expect(expected).toContain("a fifth to a third of the gap");
  const finds = START.find((section) => section.id === "finds")!.paragraphs.join(" ");
  expect(finds).toContain("a fifth to a third of the gap");
  // And each head keeps the role the example gives it.
  const roles = [...finds.matchAll(/(?:negative name movers|name movers|S-inhibition heads|induction head) \d[\d., and]*\d/g)].map((m) => m[0]);
  expect(roles).toHaveLength(4);
  for (const role of roles) expect(expected, role).toContain(role);
  // So is what it says of the greater-than example.
  const greater = START.find((section) => section.id === "then")!.paragraphs.join(" ").match(/MLPs 9 and 10 [^.]+/);
  expect(greater).not.toBeNull();
  expect(galleryField("greater-than-gpt2", "expect")).toContain(greater![0]);
});

test("the figures draw the example's own prompt pair, and the heads the guide to IOI lists", () => {
  // The pair is the example's question, a token each, and the corrupt prompt changes only the
  // repeated name, to the answer.
  const question = galleryField(TOUR_EXAMPLE, "question");
  expect(question).toContain(`“${promptText(PAIR.clean)}”`);
  expect(startTexts().join(" ")).toContain(`“${promptText(PAIR.clean)}”`);
  expect(PAIR.corrupt.length).toBe(PAIR.clean.length);
  const changed = PAIR.clean.flatMap((token, i) => (token === PAIR.corrupt[i] ? [] : [i]));
  expect(changed).toEqual([PAIR.changed]);
  expect(PAIR.clean[PAIR.changed]).toBe(PAIR.distractor);
  expect(PAIR.corrupt[PAIR.changed]).toBe(PAIR.answer);
  expect(question).toContain(`${PAIR.answer}?`);

  // The heads and their effects are the replication guide's table, row for row.
  const table = [...readRepo("docs/guides/replicate-ioi.md").matchAll(/^\| (\d+)\.(\d+) \| (−?[\d.]+) \| ([^|]+) \|$/gm)].map((m) => ({
    layer: Number(m[1]),
    head: Number(m[2]),
    effect: Number(m[3].replace("−", "-")),
    role: m[4].trim(),
  }));
  expect(table.length).toBe(IOI_HEADS.length);
  const ROLE_NAMES = { induction: "induction head", "s-inhibition": "S-inhibition head", "name-mover": "name mover", "negative-name-mover": "negative name mover" };
  expect(IOI_HEADS.map((h) => ({ layer: h.layer, head: h.head, effect: h.effect, role: ROLE_NAMES[h.role] }))).toEqual(table);
  // Every head drawn is one the example expects.
  const expected = galleryField(TOUR_EXAMPLE, "expect");
  for (const h of IOI_HEADS) expect(expected).toContain(`${h.layer}.${h.head}`);

  // The patched head is the one Start here names, and "about a sixth of the gap" is its effect.
  const name = `${PATCHED_HEAD.layer}.${PATCHED_HEAD.head}`;
  expect(START.find((section) => section.id === "patch")!.figure!.stages.map((st) => st.text).join(" ")).toContain(`head ${name}'s`);
  expect(Math.abs(patchedEffect() - 1 / 6)).toBeLessThan(0.02);
  // On the figures' scale the clean prompt prefers the answer and the corrupt one the distractor,
  // and the patched run sits its effect of the way from one to the other.
  expect(METER.clean).toBeGreaterThan(0);
  expect(METER.corrupt).toBeLessThan(0);
  expect((patchedAt(patchedEffect()) - METER.corrupt) / (METER.clean - METER.corrupt)).toBeCloseTo(patchedEffect(), 10);
});

test("the questions Logogram answers are the documentation's, and every method answers one", () => {
  const headings = anchorsOf("docs/concepts/choosing-a-method.md");
  expect(QUESTIONS.map((q) => q.id)).toEqual(["where", "flow", "represent", "steer"]);
  for (const q of QUESTIONS) {
    // Its title is the page on choosing a method's heading for it, and its text a sentence.
    expect(headings).toContain(slug(q.title));
    expect(q.text.trim()).toMatch(/[.!?]$/);
    expect(existsSync(fileURLToPath(new URL(`../../${q.guide.path}`, import.meta.url))), q.guide.path).toBe(true);
    if (q.guide.anchor) expect(anchorsOf(q.guide.path)).toContain(q.guide.anchor);
    expect(METHODS.some((m) => m.question === q.id), q.id).toBe(true);
    if (q.example) expect(existsSync(fileURLToPath(new URL(`../../src/logogram/examples/${q.example}/project.json`, import.meta.url)))).toBe(true);
  }
  // Start here asks them as they are, and the documentation's index lists them the same way.
  const ask = START.find((section) => section.id === "ask")!;
  expect(ask.terms!.map((t) => t.term)).toEqual(QUESTIONS.map((q) => q.title));
  const index = readRepo("docs/README.md");
  for (const q of QUESTIONS) {
    const row = index.split("\n").find((line) => line.startsWith(`| [${q.title}](concepts/choosing-a-method.md#${slug(q.title)})`));
    expect(row, q.title).toBeDefined();
    expect(row).toContain(`| ${q.text} |`);
    expect(row).toContain(`[${q.guide.title}](${q.guide.path.replace(/^docs\//, "")})`);
  }
});

test("the guide's figures say each stage in a short sentence", () => {
  for (const [name, figure] of Object.entries(GUIDE_FIGURES)) {
    expect(figure.stages.length, name).toBeGreaterThanOrEqual(2);
    expect(new Set(figure.stages.map((stage) => stage.title)).size, name).toBe(figure.stages.length);
    for (const stage of figure.stages) {
      expect(sentenceCase(stage.title), stage.title).toBe(true);
      expect(words([stage.text]), stage.text).toBeLessThanOrEqual(22);
      expect(stage.text.trim(), stage.text).toMatch(/[.!?]$/);
    }
  }
});
