import { test, expect, type Page } from "@playwright/test";

import { expectAccessible } from "./axe";

// ACDC on the fixture's tiny model (two layers of four heads: 110 edges), from the form, which
// says how many passes a run takes and how long they may take before it runs, to the circuit
// drawn, its faithfulness and every edge listed with whether it was kept. The run works in a
// project of its own and reopens the one that was open. Each view is scanned by axe.

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
  const created = await page.request.post("/api/projects", { data: { name: `ACDC ${Date.now()}`, parent: state.projects_parent }, headers: ORIGIN });
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

test("ACDC says what a run takes, then finds a circuit and draws it", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await projectWithPrompts(page, 8);
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  await page.getByRole("radiogroup", { name: "Method" }).getByRole("radio", { name: /^Circuit discovery \(ACDC\)/ }).click();
  await page.getByRole("radiogroup", { name: "Direction" }).getByRole("radio", { name: /^Corrupt → clean/ }).click();

  // Before it runs: the graph, the passes (one per edge and batch) and how long they may take.
  const main = page.getByRole("main");
  const summary = main.getByRole("region", { name: /ready to run/i });
  await expect(summary.getByText("110 edges to test, from 11 senders into 27 inputs", { exact: true })).toBeVisible();
  await expect(summary.getByText(/^\d+ forward passes: \d+ batch(es)? of 111 edge-patched passes each, one per edge and one with every edge, about \d+ (seconds?|minutes|hours) here\.$/)).toBeVisible();
  // The threshold and the rule decide which edges stay: neither has a default.
  await expect(main.getByText("Required: it decides how small a change leaves an edge out, so there’s no default.")).toBeVisible();
  await expect(main.getByText("Required: it decides which changes count, so there’s no default.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Run", exact: true })).toBeDisabled();
  await page.getByRole("textbox", { name: "Threshold" }).fill("0.05");
  await page.getByRole("radiogroup", { name: "Compare with the threshold" }).getByRole("radio", { name: /^The change toward corrupt/ }).click();
  await expectAccessible(page, "Experiment, ACDC");

  await page.getByRole("button", { name: "Preview spec", exact: true }).click();
  const spec = JSON.parse(await page.getByLabel("Spec JSON").innerText());
  expect(spec.experiment).toEqual({ kind: "acdc", direction: "corrupt_to_clean", threshold: 0.05, rule: "toward_source" });
  expect(spec.scope).toEqual({ kind: "graph", layers: null, logits: true });
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  await page.getByRole("button", { name: "Run", exact: true }).click();

  // The circuit drawn, how many edges it kept and how faithful it is, and every edge listed.
  await expect(main.getByRole("heading", { name: "The circuit, layers left to right" })).toBeVisible({ timeout: 60_000 });
  const overview = main.getByText(/^ACDC kept \d+ of 110 edges\. The clean prompts ran with every edge in the circuit/);
  await expect(overview).toBeVisible();
  const kept = Number((await overview.innerText()).match(/^ACDC kept (\d+)/)?.[1]);
  const figures = main.getByRole("definition").filter({ hasText: /of 110$/ });
  await expect(figures).toHaveText(`${kept} of 110`);
  await expect(main.getByText("Faithfulness of the circuit", { exact: true })).toBeVisible();
  const list = main.getByRole("region", { name: "Edges, strongest first" });
  await expect(list.getByRole("columnheader", { name: "Circuit" })).toBeVisible();
  await list.getByRole("button", { name: /Show \d+ more/ }).click();
  await list.getByRole("button", { name: /Show \d+ more/ }).click();
  await list.getByRole("button", { name: /Show \d+ more/ }).click();
  await list.getByRole("button", { name: /Show \d+ more/ }).click();
  await expect(list.getByRole("cell", { name: "kept", exact: true })).toHaveCount(kept);
  await expect(list.getByRole("cell", { name: "removed", exact: true })).toHaveCount(110 - kept);
  // The inspector says how the edge chosen was tested, and what came of it.
  const first = list.getByRole("row").nth(1).getByRole("button");
  const label = (await first.innerText()).trim();
  await first.click();
  await expect(page.getByRole("heading", { level: 2, name: label, exact: true })).toBeVisible();
  await expect(page.getByText(/^ACDC tested the edges from the logits back toward the embeddings, this one after every edge downstream of /)).toBeVisible();
  await expect(main.getByText(/^Corrected for 110 edges: /)).toBeVisible();
  await expectAccessible(page, "Results, ACDC");
  expect(errors).toEqual([]);
});
