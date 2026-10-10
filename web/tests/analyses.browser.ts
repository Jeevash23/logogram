import { test, expect, type Page } from "@playwright/test";

// Each test works in a project of its own, generated from a task, and reopens the project that
// was open before it, so the other browser checks see the fixture as they expect.

test.describe.configure({ mode: "serial" });

const ORIGIN = { origin: "http://127.0.0.1:8877" };
let previous: string | null = null;

test.beforeEach(async ({ page }) => {
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
  previous = (await (await page.request.get("/api/state")).json()).project?.path ?? null;
});

test.afterEach(async ({ page }) => {
  if (previous) await page.request.post("/api/projects/open", { data: { path: previous }, headers: ORIGIN });
});

/** A new, empty project, open in the page. */
async function freshProject(page: Page, name: string) {
  const state = await (await page.request.get("/api/state")).json();
  const created = await page.request.post("/api/projects", { data: { name: `${name} ${Date.now()}`, parent: state.projects_parent }, headers: ORIGIN });
  expect(created.ok()).toBe(true);
  await expect(page.getByText("No prompts yet.", { exact: false })).toBeVisible();
}

/** Generate IOI prompts with the task generator, as someone would. */
async function generateIOI(page: Page, n: number) {
  await page.getByRole("button", { name: "Experiment", exact: true }).click();
  await page.getByRole("button", { name: "Prompts", exact: true }).click();
  const generator = page.getByRole("radiogroup", { name: "Add prompts" }).getByRole("radio", { name: "Generate a task" });
  if ((await generator.getAttribute("aria-checked")) !== "true") await generator.click();
  await expect(page.getByRole("radiogroup", { name: "Task" }).getByRole("radio", { name: /^Indirect object identification/ })).toHaveAttribute("aria-checked", "true");
  await page.getByRole("spinbutton", { name: "Prompts", exact: true }).fill(String(n));
  const request = page.waitForRequest((r) => r.url().endsWith("/api/datasets/generate"));
  await page.getByRole("button", { name: `Generate ${n} prompts`, exact: true }).click();
  return (await request).postDataJSON();
}

test("generating a task dataset offers the metric that reads its answers", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await freshProject(page, "Task generation");
  // The form reads answers with the log-probability difference; IOI's are read with the logit difference.
  await page.getByRole("button", { name: "Experiment", exact: true }).click();
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  const metric = page.getByRole("radiogroup", { name: "Metric", exact: true });
  await metric.getByRole("radio", { name: /^Log-probability difference/ }).click();
  const body = await generateIOI(page, 6);
  expect(body).toMatchObject({ task: "ioi", name: "ioi", n: 6, seed: 0, options: { patterns: ["ABBA", "BABA"], corruption: "flip" } });
  expect(body.options.templates.length).toBeGreaterThan(0);
  const offer = page.getByRole("status").filter({ hasText: "Its answers are read with the logit difference; the experiment form uses the log-probability difference." });
  await expect(offer).toBeVisible();
  await expect(page.getByRole("table")).toContainText("When");
  await offer.getByRole("button", { name: "Use the logit difference" }).click();
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  await expect(metric.getByRole("radio", { name: /^Logit difference/ })).toHaveAttribute("aria-checked", "true");
  // The switch is a step of the form's history: its notice undoes it.
  await page.getByRole("status").filter({ hasText: "The experiment form now uses the logit difference." }).getByRole("button", { name: "Undo" }).click();
  await expect(metric.getByRole("radio", { name: /^Log-probability difference/ })).toHaveAttribute("aria-checked", "true");
  expect(errors).toEqual([]);
});

test("choosing a metric and running names the run's values by it", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await freshProject(page, "Metric choice");
  await generateIOI(page, 8);
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  const metric = page.getByRole("radiogroup", { name: "Metric", exact: true });
  // The KL divergence has no default target: the form says so and won't run.
  await metric.getByRole("radio", { name: /^KL divergence/ }).click();
  await expect(page.getByText("Required: the target changes every value, so there's no default.")).toBeVisible();
  await expect(page.getByRole("alert").filter({ hasText: "Choose which prompt's prediction the KL divergence is measured from." })).toBeVisible();
  await expect(page.getByRole("button", { name: "Run", exact: true })).toBeDisabled();
  await page.getByRole("radiogroup", { name: "KL divergence measured from" }).getByRole("radio", { name: /^The clean prompt's prediction/ }).click();
  await expect(page.getByRole("button", { name: "Run", exact: true })).toBeEnabled();
  await metric.getByRole("radio", { name: /^Log-probability difference/ }).click();
  await page.getByRole("button", { name: "Preview spec", exact: true }).click();
  const spec = JSON.parse(await page.getByLabel("Spec JSON").innerText());
  expect(spec).toMatchObject({ logogram_spec: 2, metric: { kind: "logprob_diff", normalization: "dataset_gap" }, statistics: { cluster: null } });
  // The baseline is measured with the form's metric.
  await page.getByRole("button", { name: "Baseline", exact: true }).click();
  const request = page.waitForRequest((r) => r.url().endsWith("/api/baseline"));
  await page.getByRole("button", { name: "Check the baseline", exact: true }).click();
  expect((await request).postDataJSON().metric).toEqual({ kind: "logprob_diff", normalization: "dataset_gap" });
  await expect(page.getByText("Clean log-prob diff", { exact: true })).toBeVisible();
  // The gap in the metric is judged as a run will judge it.
  await expect(page.locator("p").filter({ hasText: "Effects are normalized by this gap." })).toContainText("a gap of −0.620.");
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  await page.getByRole("button", { name: "Run", exact: true }).click();
  await expect(page.getByRole("radiogroup", { name: "Result values" }).getByRole("radio", { name: "Δ log-prob diff" })).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText("log-probability difference, normalized by the dataset gap")).toBeVisible();
  await expect(page.getByText("Clean log-prob diff", { exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

test("a circuit built from staged sites runs, and its faithfulness is read", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await freshProject(page, "Circuit");
  await generateIOI(page, 8);
  await page.getByRole("button", { name: "Explore", exact: true }).click();
  await page.getByRole("radio", { name: "Layer explorer", exact: true }).click();
  for (const name of [/^L0 H0:/, /^L0 H2:/]) {
    await page.getByRole("button", { name }).click();
    await page.getByRole("button", { name: "Add to experiment", exact: true }).click();
  }
  const tray = page.getByRole("region", { name: "Experiment selection" });
  await expect(tray).toContainText("L0 H0");
  await tray.getByRole("button", { name: "Test as a circuit", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "Test 2 sites as a circuit" });
  await expect(dialog).toContainText("Circuit kept");
  await expect(dialog.getByRole("checkbox", { name: "Heads" })).toBeChecked();
  await dialog.getByRole("button", { name: "Set up the test", exact: true }).click();
  // One step of the form's history, with the sets in the sweep editor.
  await expect(page.getByRole("status").filter({ hasText: "Experiment form replaced" })).toBeVisible();
  await expect(page.getByRole("radiogroup", { name: "Sweep" }).getByRole("radio", { name: "Sets of sites" })).toHaveAttribute("aria-checked", "true");
  for (const label of ["Circuit kept", "Circuit removed", "Everything replaced", "Without L0 H0", "Without L0 H2"]) {
    await expect(page.getByRole("group", { name: new RegExp(`^Set \\d+: ${label}$`) })).toBeVisible();
  }
  await expect(tray).not.toBeVisible();
  await page.getByRole("button", { name: "Preview spec", exact: true }).click();
  const spec = JSON.parse(await page.getByLabel("Spec JSON").innerText());
  expect(spec.scope).toMatchObject({ kind: "site_sets", universe: ["head"] });
  expect(spec.scope.sets).toHaveLength(5);
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  await page.getByRole("button", { name: "Run", exact: true }).click();
  await expect(page.getByText("Faithfulness against the number of sites kept")).toBeVisible({ timeout: 60_000 });
  const kept = page.getByRole("row").filter({ has: page.getByRole("button", { name: "Circuit kept", exact: true }) });
  await expect(kept).toContainText("keeps 2 sites and replaces every other head");
  // Faithfulness, with its interval, beside the share of replacing everything.
  await expect(kept.getByRole("cell").nth(3)).toHaveText(/^−?\d+\.\d\d \[/);
  await expect(page.getByRole("region", { name: "What each site adds" })).toContainText("Leaving L0 H0 out of Circuit kept lowers its faithfulness");
  await kept.getByRole("button", { name: "Circuit kept", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Against replacing everything" })).toBeVisible();
  await expect(page.getByText("Faithfulness", { exact: true }).last()).toBeVisible();
  expect(errors).toEqual([]);
});
