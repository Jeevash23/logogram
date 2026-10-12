import { test, expect, type Page } from "@playwright/test";

import { expectAccessible } from "./axe";

// Finer sites than a head's output or a whole MLP: every neuron of a layer, ranked with its
// interval beside the distribution of all of them, and every key on the layer × head map. Each
// check works in a project of its own and reopens the one that was open, so the other checks see
// the fixture as they expect. Each view is scanned by axe.

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
  const created = await page.request.post("/api/projects", { data: { name: `Finer sites ${Date.now()}`, parent: state.projects_parent }, headers: ORIGIN });
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

test("every neuron of a layer is ranked with its interval beside the distribution of them all", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await projectWithPrompts(page, 6);
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  await page.getByRole("radiogroup", { name: "Method" }).getByRole("radio", { name: /^Attribution patching/ }).click();
  await page.getByRole("radiogroup", { name: "Sweep" }).getByRole("radio", { name: "Neurons", exact: true }).click();
  // No layers chosen yet: the run waits for them, and says where to give them.
  await expect(page.getByText("Choose the layers whose neurons to sweep, such as 3, 4 (What to sweep → Layers).")).toBeVisible();
  await expect(page.getByText("Every neuron of each layer: 128 per layer, layers 0 to 1.")).toBeVisible();
  const layers = page.getByRole("textbox", { name: "Layers whose neurons to sweep" });
  await layers.fill("2");
  await expect(page.getByText("Layer 2 doesn't exist; tiny-gpt2 has layers 0 to 1.")).toBeVisible();
  await layers.fill("1");
  await page.getByRole("radiogroup", { name: "Position" }).getByRole("radio", { name: "Last token" }).click();
  await expect(page.getByRole("heading", { name: "Ready to run" })).toBeVisible();
  await expect(page.getByText(/^About .+ free: it fits\.$/)).toBeVisible();
  await expectAccessible(page, "Experiment, every neuron of a layer");
  await page.getByRole("button", { name: "Preview spec", exact: true }).click();
  const spec = JSON.parse(await page.getByLabel("Spec JSON").innerText());
  expect(spec.scope).toEqual({ kind: "neurons", layers: [1], position: { kind: "last" } });
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  await page.getByRole("button", { name: "Run", exact: true }).click();

  // Layer by layer: the distribution of every neuron, and the strongest with their intervals.
  const main = page.getByRole("main");
  const layer = main.getByRole("region", { name: "Layer 1" });
  await expect(layer).toContainText("128 of 128 neurons", { timeout: 60_000 });
  await expect(main.getByText("Every neuron, layer by layer", { exact: true })).toBeVisible();
  await expect(layer.getByRole("img", { name: /^Layer 1: distribution of 128 neurons, from / })).toBeVisible();
  const ranked = layer.getByRole("button", { name: /^L1 N\d+ @ last/ });
  await expect(ranked).toHaveCount(10);
  await layer.getByRole("button", { name: "Show the strongest 50", exact: true }).click();
  await expect(ranked).toHaveCount(50);
  await layer.getByRole("button", { name: "Show the strongest 10", exact: true }).click();
  await expect(ranked).toHaveCount(10);

  // The strongest neuron, in the inspector and marked on the distribution.
  await ranked.first().click();
  await expect(page.getByText(/^Neuron \d+ of the MLP in layer 1: its activation after the nonlinearity/)).toBeVisible();
  await expect(layer.getByRole("img", { name: /the selected neuron at / })).toBeVisible();
  await expect(ranked.first()).toHaveAttribute("aria-current", "true");
  // The keyboard reaches the ranked neurons.
  await ranked.nth(1).focus();
  await page.keyboard.press("Enter");
  await expect(ranked.nth(1)).toHaveAttribute("aria-current", "true");
  await expectAccessible(page, "Evidence, every neuron of a layer");
  expect(errors).toEqual([]);
});

test("every key is patched by key/value head, on the layer × head map", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await projectWithPrompts(page, 6);
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  const sweep = page.getByRole("radiogroup", { name: "Sweep" });
  await sweep.getByRole("radio", { name: "Queries, keys, values", exact: true }).click();
  const inputs = page.getByRole("radiogroup", { name: "Each head's query, key or value" });
  await expect(inputs.getByRole("radio", { name: "Query", exact: true })).toHaveAttribute("aria-checked", "true");

  // Direct effects: queries and keys have none, values do.
  await page.getByRole("radiogroup", { name: "Method" }).getByRole("radio", { name: /^Direct logit attribution/ }).click();
  await expect(inputs.getByRole("radio", { name: "Value", exact: true })).toHaveAttribute("aria-checked", "true");
  await expect(inputs.getByRole("radio", { name: "Query", exact: true })).toBeDisabled();
  await expect(inputs.getByRole("radio", { name: "Key", exact: true })).toBeDisabled();
  await expect(page.getByText(/^Queries and keys have no direct effect: they act on the logits only through the softmax/)).toBeVisible();

  await page.getByRole("radiogroup", { name: "Method" }).getByRole("radio", { name: /^Activation patching/ }).click();
  await inputs.getByRole("radio", { name: "Key", exact: true }).click();
  await expect(page.getByText("Each of the 4 heads of a layer has its own query, key and value.", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Ready to run" })).toBeVisible();
  await expectAccessible(page, "Experiment, every key");
  await page.getByRole("button", { name: "Run", exact: true }).click();

  const main = page.getByRole("main");
  await expect(main.getByText("Layer × key/value head, keys", { exact: true })).toBeVisible({ timeout: 60_000 });
  await expect(main.getByRole("group", { name: "Layer × key/value head, keys results" })).toBeVisible();
  await expect(main.getByText("Across all 8 keys", { exact: true })).toBeVisible();
  // A key from the strongest effects, in the inspector.
  const strongest = main.getByRole("button", { name: /^L\d KV\d k/ }).first();
  await expect(strongest).toBeVisible({ timeout: 60_000 });
  await strongest.click();
  await expect(page.getByText(/^The key of key\/value head \d in layer \d: what each query compares with to choose where to attend\.$/)).toBeVisible();
  await expectAccessible(page, "Evidence, every key");
  expect(errors).toEqual([]);
});
