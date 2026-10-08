import { test, expect } from "@playwright/test";

test.describe.configure({ mode: "serial" });

test.beforeEach(async ({ page }) => {
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
});

test("saved analysis context, keyboard attention and exports", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.getByRole("button", { name: "Experiment", exact: true }).click();
  await page.getByRole("button", { name: "Baseline", exact: true }).click();
  await expect(page.getByText("Selected run · BOS off · first 5 prompts · batch 2")).toBeVisible();
  const request = page.waitForRequest((r) => r.url().endsWith("/api/baseline"));
  await page.getByRole("button", { name: "Check the baseline" }).click();
  expect((await request).postDataJSON()).toMatchObject({ prepend_bos: false, limit: 5, batch_size: 2 });
  await expect(page.getByText("Clean logit diff", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Explore", exact: true }).click();
  const map = page.locator('canvas[aria-label*="Model map"]');
  await map.focus();
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("a");
  const attention = page.getByRole("group", { name: /Attention pattern of/ });
  await expect(attention).toBeVisible();
  await attention.focus();
  await page.keyboard.press("ArrowDown");
  await expect(page.getByText(/Query 1:.*Key 0:.*Value/)).toBeVisible();
  await page.getByRole("button", { name: "Evidence", exact: true }).click();
  await page.getByRole("button", { name: "Results", exact: true }).click();
  for (const name of ["Export per-prompt CSV", "Export figure PNG", "Download methods"]) {
    await page.getByRole("button", { name: "Export", exact: true }).click();
    const download = page.waitForEvent("download");
    await page.getByRole("menuitem", { name, exact: true }).click();
    expect(await (await download).failure()).toBeNull();
  }
  expect(errors).toEqual([]);
});

test("narrow workbench preserves the content and opens keyboard dismissible panels", async ({ page }) => {
  await page.setViewportSize({ width: 800, height: 700 });
  const center = await page.getByRole("main").boundingBox();
  expect(center?.width).toBeGreaterThanOrEqual(780);
  await page.getByRole("button", { name: /^History/ }).click();
  await expect(page.getByRole("dialog", { name: "Experiment history" })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await page.getByRole("button", { name: "Inspector", exact: true }).click();
  await expect(page.getByRole("dialog", { name: "Inspector" })).toBeVisible();
  await page.keyboard.press("Escape");
});

test("layer selection stages exact sites and persists research notes", async ({ page }) => {
  await page.getByRole("radio", { name: "Layer explorer", exact: true }).click();
  await page.getByRole("button", { name: /^L0 H0: effect/ }).click();
  await page.getByRole("button", { name: /^clean token 9:/ }).click();
  await page.getByRole("button", { name: "Save selection", exact: true }).click();
  await page.getByRole("textbox", { name: "Title", exact: true }).fill("Selected-token study");
  await page.getByRole("textbox", { name: "Observation or question" }).fill("Check a candidate at the repeated name.");
  await page.getByRole("button", { name: "Save note", exact: true }).click();
  await page.getByRole("button", { name: "Add to experiment", exact: true }).click();
  await expect(page.getByRole("region", { name: "Experiment selection" })).toContainText("L0 H0 @ 9");
  await page.getByRole("button", { name: "Configure experiment", exact: true }).click();
  await expect(page.getByRole("spinbutton", { name: "Batch size", exact: true })).toHaveValue("2");
  await expect(page.getByRole("checkbox", { name: "Prepend the beginning-of-sequence token" })).not.toBeChecked();
  await page.getByRole("button", { name: "Preview spec", exact: true }).click();
  const spec = JSON.parse(await page.getByLabel("Spec JSON").innerText());
  expect(spec.scope).toEqual({ kind: "sites", sites: [{ kind: "head", layer: 0, head: 0, position: { kind: "index", index: 9 } }] });
  await page.reload();
  await page.getByRole("button", { name: "Evidence", exact: true }).click();
  await page.getByRole("button", { name: "Research notes", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Selected-token study" })).toBeVisible();
  await page.getByRole("button", { name: "Open selection", exact: true }).click();
  await expect(page.getByRole("region", { name: "Experiment selection" })).toContainText("L0 H0 @ 9");
});

test("two heads share a query and prediction reports survive a completed run", async ({ page }) => {
  await page.getByRole("button", { name: "Head comparison", exact: true }).click();
  await page.getByRole("region", { name: "Head 1", exact: true }).getByRole("button", { name: "Pin head" }).click();
  await page.getByRole("region", { name: "Head 2", exact: true }).getByRole("button", { name: "Pin head" }).click();
  await expect(page.getByRole("group", { name: "Attention comparison L0 H0" })).toBeVisible();
  await expect(page.getByRole("group", { name: "Attention comparison L0 H1" })).toBeVisible();
  await page.getByRole("button", { name: /^clean token 9:/ }).click();
  await expect(page.getByRole("heading", { name: /^Query 9/ })).toHaveCount(2);
  await page.getByRole("button", { name: "Layer predictions", exact: true }).click();
  await page.getByRole("button", { name: "Measure predictions", exact: true }).click();
  const table = page.getByRole("table", { name: /Per-layer vocabulary projections/ });
  await expect(table).toBeVisible();
  const projection = await table.innerText();
  await page.getByRole("button", { name: "Experiment", exact: true }).click();
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  await expect(page.getByRole("region", { name: "Included prediction diagnostic" })).toBeVisible();
  await page.getByRole("textbox", { name: "Name", exact: true }).fill("Prediction regression");
  await page.getByRole("button", { name: "Run", exact: true }).click();
  await expect(page.getByRole("button", { name: "Open saved predictions" })).toBeVisible();
  await page.reload();
  await page.getByRole("button", { name: "Layer predictions", exact: true }).click();
  await expect(page.getByText(/Saved diagnostic from Prediction regression/)).toBeVisible();
  await expect(table).toHaveText(projection, { useInnerText: true });
});

test("project changes clear the other tab and reject its stale writes", async ({ page, context }) => {
  const state = await (await page.request.get("/api/state")).json();
  const second = await context.newPage();
  await second.goto("/");
  await expect(second.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
  const created = await second.request.post("/api/projects", { data: { name: "Second project", parent: state.projects_parent }, headers: { origin: "http://127.0.0.1:8877" } });
  expect(created.ok()).toBe(true);
  await expect(page.getByText("No prompts yet.", { exact: false })).toBeVisible();
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).not.toBeVisible();
  const response = await page.request.post("/api/drafts", { data: { spec: {} }, headers: { origin: "http://127.0.0.1:8877", "x-logogram-project": state.project.session_id } });
  expect(response.status()).toBe(409);
  await page.getByRole("spinbutton", { name: "Prompts", exact: true }).fill("4");
  await page.getByRole("button", { name: "Generate 4 prompts", exact: true }).click();
  await page.getByRole("button", { name: "Next step: Check the baseline", exact: true }).click();
  await page.getByRole("button", { name: "Check the baseline", exact: true }).click();
  await expect(page.getByText("Clean logit diff", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Next step: Run the experiment", exact: true }).click();
  await page.getByRole("button", { name: "Run", exact: true }).click();
  await expect(page.getByRole("button", { name: "Export", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: /Saved prompts of this run/ })).toBeVisible();
  await second.close();
});
