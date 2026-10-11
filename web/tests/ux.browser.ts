import { test, expect, type Page } from "@playwright/test";

// What the interface shows rather than hides: the experiment form's summary beside Run, a site
// chosen on Compare in both runs, the switch for single-key shortcuts, and the baseline's dot plot
// as one tab stop. Tests that run experiments work in a project of their own and reopen the one
// that was open, so the other checks see the fixture as they expect.

test.describe.configure({ mode: "serial" });

const ORIGIN = { origin: `http://127.0.0.1:${process.env.LOGOGRAM_BROWSER_PORT ?? "8877"}` };
let previous: string | null = null;

test.beforeEach(async ({ page }) => {
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
  previous = (await (await page.request.get("/api/state")).json()).project?.path ?? null;
});

test.afterEach(async ({ page }) => {
  await page.request.post("/api/settings", { data: { single_key_shortcuts: true }, headers: ORIGIN });
  if (previous) await page.request.post("/api/projects/open", { data: { path: previous }, headers: ORIGIN });
});

/** A new project with IOI prompts, open in the page. */
async function projectWithPrompts(page: Page, name: string, n: number) {
  const state = await (await page.request.get("/api/state")).json();
  const created = await page.request.post("/api/projects", { data: { name: `${name} ${Date.now()}`, parent: state.projects_parent }, headers: ORIGIN });
  expect(created.ok()).toBe(true);
  await expect(page.getByText("No prompts yet.", { exact: false })).toBeVisible();
  await page.getByRole("spinbutton", { name: "Prompts", exact: true }).fill(String(n));
  await page.getByRole("button", { name: `Generate ${n} prompts`, exact: true }).click();
  await expect(page.getByRole("table")).toContainText("When");
}

test("the form says what will run beside Run, and a site chosen on Compare shows in both runs", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await projectWithPrompts(page, "Compare", 4);
  await page.getByRole("button", { name: "Configure", exact: true }).click();

  // An ablation has no default baseline: the summary says so beside the disabled Run.
  await page.getByRole("radiogroup", { name: "Method", exact: true }).getByRole("radio", { name: /^Ablation/ }).click();
  const missing = page.getByRole("region", { name: "Not ready to run" });
  await expect(missing).toBeVisible();
  await expect(missing.getByRole("button", { name: "Run", exact: true })).toHaveAttribute("aria-disabled", "true");
  await page.getByRole("radiogroup", { name: "Method", exact: true }).getByRole("radio", { name: /^Activation patching/ }).click();

  const summary = page.getByRole("region", { name: "Ready to run" });
  await expect(summary).toContainText("over 4 prompts");
  await expect(summary).toContainText(/About .+ free: it (fits|'s tight)/);
  await expect(summary).toContainText("Not checked for these prompts yet.");
  await summary.getByRole("button", { name: "Run", exact: true }).click();

  // A robustness check of the finished run opens on Compare.
  const check = page.getByRole("button", { name: "Check robustness", exact: true });
  await expect(check).toBeVisible({ timeout: 60_000 });
  await check.click();
  await page.getByRole("dialog", { name: "Check robustness" }).getByRole("button", { name: "Run the check", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Robustness check" })).toBeVisible();
  await expect(page.getByText(/^Spearman rank correlation over/)).toBeVisible({ timeout: 60_000 });

  // Choosing a cell shows that site in both runs, with a way into each.
  await page.getByRole("group", { name: "Run A", exact: true }).focus();
  await page.keyboard.press("ArrowRight");
  const site = page.getByRole("region", { name: "The selected site in both runs" });
  await expect(site).toContainText(/Run A: normalized effect, CI/);
  await expect(site).toContainText(/Run B: normalized effect, CI/);
  await expect(site).toContainText("B − A");
  const label = await site.getByRole("heading", { level: 2 }).innerText();
  await site.getByRole("button", { name: "Open run B at this site", exact: true }).click();
  await expect(page.getByRole("button", { name: "Results", exact: true })).toHaveAttribute("aria-current", "page");
  await expect(page.getByRole("complementary", { name: "Inspector" }).getByRole("heading", { level: 2 })).toContainText(label.split(" @")[0]);
  expect(errors).toEqual([]);
});

test("single-key shortcuts can be turned off, and on again from the command palette", async ({ page }) => {
  const model = page.getByRole("button", { name: "Model", exact: true });
  const attention = page.getByRole("button", { name: "Attention", exact: true });
  await model.click();
  await page.keyboard.press("?");
  const dialog = page.getByRole("dialog", { name: "Keyboard shortcuts" });
  await dialog.getByRole("checkbox", { name: "Use single-key shortcuts" }).uncheck();
  await page.keyboard.press("Escape");
  // Keys pressed while a dialog is still closing go to it, not to the workbench.
  await expect(page.locator("[role=dialog]")).toHaveCount(0);

  // Off: digits and ? do nothing.
  await page.keyboard.press("2");
  await page.keyboard.press("?");
  await expect(dialog).toBeHidden();
  await expect(model).toHaveAttribute("aria-current", "page");
  await expect(model).not.toHaveAttribute("aria-keyshortcuts");

  // On again: the palette has the switch.
  await page.keyboard.press("ControlOrMeta+k");
  await page.getByLabel("Command", { exact: true }).fill("single-key");
  await expect(page.getByRole("option", { name: /^Turn on single-key shortcuts/ })).toHaveAttribute("aria-selected", "true");
  await page.keyboard.press("Enter");
  await expect(page.locator("[role=dialog]")).toHaveCount(0);
  // The palette runs its command once it has closed: the hints come back with the keys.
  await expect(attention).toHaveAttribute("aria-keyshortcuts", "2");
  await page.keyboard.press("2");
  await expect(attention).toHaveAttribute("aria-current", "page");
});

test("the baseline's dot plot is one tab stop, and the arrow keys go from prompt to prompt", async ({ page }) => {
  await page.getByRole("button", { name: "Experiment", exact: true }).click();
  await page.getByRole("button", { name: "Baseline", exact: true }).click();
  const check = page.getByRole("button", { name: /^(Check the baseline|Run again)$/ });
  await check.click();
  const plot = page.getByRole("group", { name: /per prompt\. The arrow keys choose a prompt\.$/ });
  await expect(plot).toBeVisible();
  await expect(plot.locator("[tabindex]")).toHaveCount(0);
  await page.keyboard.press("[");
  await page.keyboard.press("[");
  await plot.focus();
  await page.keyboard.press("Home");
  await page.keyboard.press("ArrowRight");
  await expect(page.locator('[aria-live="polite"]').filter({ hasText: /^Prompt 1: clean / })).toHaveCount(1);
  await expect(page.getByRole("button", { name: /^Previous prompt/ })).toBeEnabled();
});
