import { test, expect, type Page } from "@playwright/test";

import { expectAccessible } from "./axe";

// Edge attribution on the fixture's tiny model (two layers of four heads: 110 edges), from the
// form, which says the graph's size and what a run costs before it runs, to the strongest edges
// drawn and listed, the faithfulness curve, and their verification by path patching. The run
// works in a project of its own and reopens the one that was open, so the other checks see the
// fixture as they expect. Each view is scanned by axe.

test.describe.configure({ mode: "serial" });
test.use({ bypassCSP: true, reducedMotion: "reduce" });

const ORIGIN = { origin: `http://127.0.0.1:${process.env.LOGOGRAM_BROWSER_PORT ?? "8877"}` };
let previous: string | null = null;

test.beforeEach(async ({ page }) => {
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
  previous = (await (await page.request.get("/api/state")).json()).project?.path ?? null;
});

test.afterEach(async ({ page }) => {
  if (previous) await page.request.post("/api/projects/open", { data: { path: previous }, headers: ORIGIN });
});

/** A project of its own, open in the page, with IOI prompts generated as someone would. */
async function projectWithPrompts(page: Page, n: number) {
  const state = await (await page.request.get("/api/state")).json();
  const created = await page.request.post("/api/projects", { data: { name: `Edges ${Date.now()}`, parent: state.projects_parent }, headers: ORIGIN });
  expect(created.ok()).toBe(true);
  await expect(page.getByText("No prompts yet.", { exact: false })).toBeVisible();
  await page.getByRole("button", { name: "Experiment", exact: true }).click();
  await page.getByRole("button", { name: "Prompts", exact: true }).click();
  const generator = page.getByRole("radiogroup", { name: "Add prompts" }).getByRole("radio", { name: "Generate a task" });
  if ((await generator.getAttribute("aria-checked")) !== "true") await generator.click();
  await page.getByRole("spinbutton", { name: "Prompts", exact: true }).fill(String(n));
  await page.getByRole("button", { name: `Generate ${n} prompts`, exact: true }).click();
  await expect(page.getByRole("button", { name: "Edit prompt 0", exact: true })).toBeVisible();
}

test("edges are estimated, drawn, tested as a circuit and verified by path patching", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await projectWithPrompts(page, 8);
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  await page.getByRole("radiogroup", { name: "Method" }).getByRole("radio", { name: /^Edge attribution/ }).click();

  // Before it runs: the graph's size on the loaded model, and what a run costs.
  const main = page.getByRole("main");
  await expect(main.getByText("Every edge of the graph: 110 edges from 11 senders into 27 inputs on the loaded model.", { exact: true })).toBeVisible();
  const summary = main.getByRole("region", { name: /ready to run/i });
  await expect(summary.getByText("110 edges, from 11 senders into 27 inputs", { exact: true })).toBeVisible();
  await expect(summary.getByText(/^\d+ batch(es)? of 3 forward and 1 backward pass each, then 11 edge-patched forward passes each for the curve\.$/)).toBeVisible();
  // Which edges the curve patches decides what faithfulness 1 means: no default.
  await expect(main.getByText("Required: it decides what faithfulness 1 means, so there’s no default.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Run", exact: true })).toBeDisabled();
  await page.getByRole("radiogroup", { name: "The curve patches" }).getByRole("radio", { name: /^Every edge but the chosen/ }).click();
  await page.getByRole("textbox", { name: "Numbers of edges" }).fill("1, 3, 10, 30, 300");
  await page.getByRole("spinbutton", { name: "Random draws" }).fill("2");

  // Chosen layers: the edges into layer 1 and the logits only.
  await page.getByRole("radiogroup", { name: "Edges into" }).getByRole("radio", { name: "Chosen layers" }).click();
  await page.getByRole("group", { name: "Layers whose heads and MLPs receive edges" }).getByRole("checkbox", { name: "L0" }).uncheck();
  await expect(main.getByText("The edges into layer 1 and the logits: 93 edges from 11 senders into 14 inputs on the loaded model.", { exact: true })).toBeVisible();
  await page.getByRole("radiogroup", { name: "Edges into" }).getByRole("radio", { name: "Every layer" }).click();
  await expect(main.getByText(/^Every edge of the graph: 110 edges/)).toBeVisible();
  await expect(summary.getByText(/then 13 edge-patched forward passes each for the curve\.$/)).toBeVisible();
  await expectAccessible(page, "Experiment, edge attribution");

  await page.getByRole("button", { name: "Preview spec", exact: true }).click();
  const spec = JSON.parse(await page.getByLabel("Spec JSON").innerText());
  expect(spec.experiment).toEqual({
    kind: "edge_attribution",
    direction: "clean_to_corrupt",
    method: "gradient",
    steps: null,
    curve: { counts: [1, 3, 10, 30, 300], patch: "rest", control: { seed: 0, draws: 2 } },
  });
  expect(spec.scope).toEqual({ kind: "graph", layers: null, logits: true });
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  await page.getByRole("button", { name: "Run", exact: true }).click();

  // The strongest edges drawn, the curve and the ranked list.
  await expect(main.getByRole("heading", { name: "The strongest edges, layers left to right" })).toBeVisible({ timeout: 60_000 });
  await expect(main.getByText(/^110 edges estimated, from 11 senders into 27 inputs, from one gradient each\. Every edge is listed here\.$/)).toBeVisible();
  const graph = main.getByRole("group", { name: "The 50 strongest edges, layers left to right" });
  await expect(graph).toBeVisible();
  const list = main.getByRole("region", { name: "Edges, strongest first" });
  const first = (await list.getByRole("row").nth(1).getByRole("button").innerText()).trim();
  const second = (await list.getByRole("row").nth(2).getByRole("button").innerText()).trim();
  // The keyboard moves through the edges, strongest first, and the inspector follows.
  await graph.focus();
  await page.keyboard.press("Home");
  await expect(main.getByText(new RegExp(`^Edge 1 of 50: ${escape(first)}\\. Estimated effect `))).toBeVisible();
  await page.keyboard.press("ArrowDown");
  await expect(main.getByText(new RegExp(`^Edge 2 of 50: ${escape(second)}\\. Estimated effect `))).toBeVisible();
  await expect(page.getByRole("heading", { level: 2, name: second, exact: true })).toBeVisible();
  await expect(list.getByRole("button", { name: second, exact: true })).toHaveAttribute("aria-pressed", "true");
  // Fewer edges drawn, from the list's own button as well.
  await page.getByRole("radiogroup", { name: "Edges drawn" }).getByRole("radio", { name: "25" }).click();
  await expect(main.getByRole("group", { name: "The 25 strongest edges, layers left to right" })).toBeVisible();
  await list.getByRole("button", { name: first, exact: true }).click();
  await expect(page.getByRole("heading", { level: 2, name: first, exact: true })).toBeVisible();
  await expect(list.getByRole("row")).toHaveCount(26);
  await list.getByRole("button", { name: "Show 25 more" }).click();
  await expect(list.getByRole("row")).toHaveCount(51);

  const curve = main.getByRole("figure", { name: "Faithfulness against the number of edges" });
  await expect(curve.getByRole("img")).toBeVisible();
  await expect(curve).toContainText("The corrupt prompts ran with every edge but the chosen carrying the clean prompts' values.");
  await expect(curve).toContainText("Patching every edge of the graph gave 0.000 (it should be 0)");
  await expect(curve).toContainText("Left out, being more than the graph's 110 edges: 300.");
  await expect(main.getByText(/^Corrected for 110 edges: /)).toBeVisible();
  await expectAccessible(page, "Results, edge attribution");

  // Verified by path patching: each of the strongest edges measured alone, beside its estimate.
  await main.getByRole("button", { name: "Verify top 20 by path patching" }).click();
  await expect(main.getByRole("heading", { level: 1, name: "Estimates checked by patching" })).toBeVisible();
  const table = main.getByRole("region", { name: "Estimates against path patching" });
  await expect(table).toBeVisible({ timeout: 60_000 });
  await expect(table.getByRole("row")).toHaveCount(21);
  await expect(table).toContainText(/Spearman's rank correlation of the two: −?\d\.\d\d\. \d+ of 20 have the same sign\./);
  await expectAccessible(page, "Compare, edges verified by path patching");
  await table.getByRole("button", { name: first, exact: true }).click();
  await expect(table.getByRole("button", { name: first, exact: true })).toHaveAttribute("aria-pressed", "true");

  // The estimate's own results show the verification too, in the table and in the list.
  await page.getByRole("button", { name: /^History/ }).click();
  const history = page.getByRole("dialog", { name: "Experiment history" });
  await history.getByRole("button", { name: /^Edges clean→corrupt · the whole graph(?! · verified)/ }).first().click();
  await expect(main.getByRole("region", { name: "Estimates against path patching" })).toBeVisible();
  await expect(main.getByRole("region", { name: "Edges, strongest first" }).getByRole("columnheader", { name: "Path patching" })).toBeVisible();
  await expectAccessible(page, "Results, edge attribution with its verification");
  expect(errors).toEqual([]);
});

function escape(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
