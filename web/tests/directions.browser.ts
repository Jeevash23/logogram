import { test, expect, type Page } from "@playwright/test";

import { expectAccessible } from "./axe";

// A probe on the fixture's tiny model, from the form (what it reads has no default, and a probe
// has no metric) to its results, a site's direction saved from them, and that direction removed
// by directional ablation and patched along; then a learned subspace beside the whole site. The
// runs work in a project of their own and reopen the one that was open, so the other checks see
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
  const created = await page.request.post("/api/projects", { data: { name: `Directions ${Date.now()}`, parent: state.projects_parent }, headers: ORIGIN });
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

async function previewSpec(page: Page): Promise<Record<string, unknown>> {
  await page.getByRole("button", { name: "Preview spec", exact: true }).click();
  const spec = JSON.parse(await page.getByLabel("Spec JSON").innerText());
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  return spec;
}

test("a probe's direction is saved from its results, then removed and patched along", async ({ page }) => {
  test.setTimeout(120_000);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await projectWithPrompts(page, 12);
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  const main = page.getByRole("main");
  await page.getByRole("radiogroup", { name: "Method" }).getByRole("radio", { name: /^Linear probe/ }).click();

  // What the probe reads is what the run measures: no default, and no metric.
  await expect(main.getByText("Required: what the probe reads is what the run measures, so there's no default.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Run", exact: true })).toBeDisabled();
  await expect(main.getByText(/^A probe reads activations, not the model's output, so there is no metric/)).toBeVisible();
  await page.getByRole("radiogroup", { name: "The probe reads" }).getByRole("radio", { name: /^Which prompt it is/ }).click();
  // The residual stream after each layer, at the last token: a direction there can be used.
  await page.getByRole("radiogroup", { name: "Sweep" }).getByRole("radio", { name: "Attention and MLP per layer" }).click();
  for (const name of ["attn out", "mlp out"]) await main.getByRole("checkbox", { name, exact: true }).uncheck();
  await main.getByRole("checkbox", { name: "resid post", exact: true }).check();
  await main.getByRole("radiogroup", { name: "Position" }).getByRole("radio", { name: "Last token" }).click();
  // A probe reads one token of each prompt, one site at a time.
  const sweep = page.getByRole("radiogroup", { name: "Sweep" });
  await expect(sweep.getByRole("radio", { name: "Neurons" })).toBeDisabled();
  await expect(sweep.getByRole("radio", { name: "Sets of sites" })).toBeDisabled();
  await expect(main.getByRole("radiogroup", { name: "Position" }).first().getByRole("radio", { name: "All positions" })).toBeDisabled();
  await expectAccessible(page, "Experiment, linear probe");

  const spec = await previewSpec(page);
  expect(spec.experiment).toEqual({ kind: "probe", labels: { kind: "side" }, train_fraction: 0.5, seed: 0, regularization: 0.01, iterations: 100, control: true });
  expect(spec.scope).toEqual({ kind: "layer_components", components: ["resid_post"], position: { kind: "last" } });
  expect(spec.metric).toBeNull();
  await page.getByRole("button", { name: "Run", exact: true }).click();

  // Each site's probe beside its control, with Save as direction once the run has finished.
  const probes = main.getByRole("table", { name: "Probes by site" });
  await expect(probes).toBeVisible({ timeout: 60_000 });
  await expect(main.getByText(/^The probes read which prompt it is \(clean or corrupt\): clean, corrupt\./)).toBeVisible();
  const save = probes.getByRole("button", { name: /^Save .+ as a direction$/ }).first();
  await expect(save).toBeVisible({ timeout: 60_000 });
  await expectAccessible(page, "Results, linear probe");
  const label = (await save.getAttribute("aria-label"))!.replace(/^Save | as a direction$/g, "");
  await save.click();
  await page.getByRole("textbox", { name: `Name of the direction from ${label}` }).fill("which-prompt");
  await page.keyboard.press("Enter");
  await expect(page.getByText("Saved the direction which-prompt. Steering, directional ablation, subspace patching and readouts can use it.")).toBeVisible();
  await expect(probes.getByRole("button", { name: `${label}: saved as which-prompt` })).toBeVisible();
  await page.screenshot({ path: `${process.env.LOGOGRAM_TEST_OUTPUT ?? "test-results"}/probe-results.png`, fullPage: true });

  // The direction removed by directional ablation: offered by the form once saved.
  await page.getByRole("button", { name: "Experiment", exact: true }).click();
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  await page.getByRole("radiogroup", { name: "Method" }).getByRole("radio", { name: /^Ablation/ }).click();
  await page.getByRole("radiogroup", { name: "Baseline" }).getByRole("radio", { name: /^Remove a direction/ }).click();
  await expect(sweep.getByRole("radio", { name: "Layer × head" })).toBeDisabled();
  await page.getByRole("combobox", { name: "Direction" }).selectOption("which-prompt");
  await expect(main.getByText(/^1 dimension · L\d .+ · from a probe$/)).toBeVisible();
  const ablation = await previewSpec(page);
  expect(ablation.experiment).toEqual({ kind: "ablation", baseline: { kind: "projection", direction: { name: "which-prompt", sha256: null }, reference: null } });
  await expectAccessible(page, "Experiment, directional ablation");
  await page.getByRole("button", { name: "Run", exact: true }).click();
  await expect(main.getByRole("heading", { level: 1, name: "Directional ablation · resid post per layer" })).toBeVisible({ timeout: 60_000 });
  await expect(page.getByRole("complementary", { name: "Inspector" }).getByText("Ablate the direction which-prompt (removed)")).toBeVisible();
  await expect(page.getByRole("status").filter({ hasText: "finished" })).toBeVisible({ timeout: 60_000 });
  await page.screenshot({ path: `${process.env.LOGOGRAM_TEST_OUTPUT ?? "test-results"}/directional-ablation.png`, fullPage: true });

  // Patched along, from the other prompt of each pair.
  await page.getByRole("button", { name: "Experiment", exact: true }).click();
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  await page.getByRole("radiogroup", { name: "Method" }).getByRole("radio", { name: /^Subspace patching/ }).click();
  await page.getByRole("combobox", { name: "Patch along" }).selectOption("which-prompt");
  const along = await previewSpec(page);
  expect(along.experiment).toEqual({ kind: "subspace_patching", direction: "clean_to_corrupt", subspace: { name: "which-prompt", sha256: null } });
  expect(errors).toEqual([]);
});

test("a learned subspace is measured beside the whole site and a random subspace", async ({ page }) => {
  test.setTimeout(120_000);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await projectWithPrompts(page, 12);
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  const main = page.getByRole("main");
  await page.getByRole("radiogroup", { name: "Method" }).getByRole("radio", { name: /^Learned subspace \(DAS\)/ }).click();
  await page.getByRole("spinbutton", { name: "Steps" }).fill("5");
  const spec = await previewSpec(page);
  expect(spec.experiment).toMatchObject({ kind: "das", dimensions: 1, steps: 5, control: true });
  await expectAccessible(page, "Experiment, distributed alignment search");
  await page.getByRole("button", { name: "Run", exact: true }).click();

  const subspaces = main.getByRole("table", { name: "Learned subspaces by site" });
  await expect(subspaces).toBeVisible({ timeout: 60_000 });
  await expect(subspaces.getByRole("columnheader", { name: "Random subspace" })).toBeVisible();
  await expect(subspaces.getByRole("button", { name: /^Save the subspace at .+ as a direction$/ }).first()).toBeVisible({ timeout: 60_000 });
  await expectAccessible(page, "Results, distributed alignment search");
  await page.screenshot({ path: `${process.env.LOGOGRAM_TEST_OUTPUT ?? "test-results"}/das-results.png`, fullPage: true });
  expect(errors).toEqual([]);
});
