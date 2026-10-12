import { test, expect, type Page } from "@playwright/test";

import { expectAccessible } from "./axe";

// A run that stops early keeps what finished, and the queue is kept in the project: opened again,
// a project's queue comes back paused, to start or clear. The fixture server cancels runs named
// "Partial run" once their first layer has finished, and holds runs named "Held run" until they
// are cancelled. Each test works in a project of its own and reopens the one that was open, so
// the other checks see the fixture.

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

/** A project of the test's own with four prompts; returns its folder. */
async function newProject(page: Page, name: string): Promise<string> {
  const state = await (await page.request.get("/api/state")).json();
  const created = await page.request.post("/api/projects", { data: { name: `${name} ${Date.now()}`, parent: state.projects_parent }, headers: ORIGIN });
  expect(created.ok()).toBe(true);
  await expect(page.getByText("No prompts yet.", { exact: false })).toBeVisible();
  await page.getByRole("spinbutton", { name: "Prompts", exact: true }).fill("4");
  await page.getByRole("button", { name: "Generate 4 prompts", exact: true }).click();
  await expect(page.getByRole("table")).toContainText("When");
  return (await created.json()).path as string;
}

/** Set the experiment's name in the form and press its button (Run, or Queue). */
async function start(page: Page, name: string, button: "Run" | "Queue") {
  await page.getByRole("button", { name: "Experiment", exact: true }).click();
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  await page.getByRole("textbox", { name: "Name", exact: true }).fill(name);
  await page.getByRole("button", { name: button, exact: true }).click();
}

test("a run that stops early shows the layers that finished, and nothing for the rest", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await newProject(page, "Partial");
  await start(page, "Partial run", "Run");
  const kept = "Stopped after 1 of 2 layers: these are the layers that finished.";
  await expect(page.getByRole("status").filter({ hasText: `Run cancelled. ${kept}` })).toBeVisible({ timeout: 60_000 });

  // Its results open like a finished run's, with a note that says how far it got.
  const main = page.getByRole("main");
  await expect(main.getByRole("heading", { level: 1, name: "Partial run" })).toBeVisible();
  await expect(main.getByText("This run was cancelled before it finished")).toBeVisible();
  await expect(main.getByText(kept, { exact: true })).toBeVisible();
  await expect(main.getByRole("list", { name: "What this run doesn't keep" })).toContainText("q-values are left out");
  await expect(main.getByRole("button", { name: "Export" })).toBeVisible();
  await expect(main.getByRole("button", { name: "Check robustness" })).toHaveCount(0);
  // Layer 0's heads have values; layer 1's weren't measured, and nothing is drawn for them.
  await main.getByText("View values as a table").first().click();
  const table = main.getByRole("table").first();
  await expect(table.getByRole("row", { name: /^0: 0 / })).not.toContainText("Not measured");
  await expect(table.getByRole("row", { name: /^1: 1 / })).toContainText("Not measured");
  await expectAccessible(page, "Results, a run that stopped early");

  // The history says it is partial, and how far it got.
  await page.getByRole("button", { name: /^History/ }).click();
  const history = page.getByRole("dialog", { name: "Experiment history" });
  const row = history.getByRole("button", { name: /^Partial run/ });
  await expect(row).toContainText("partial");
  await expect(row).toContainText("cancelled after 1 of 2 layers, kept what finished");
  await expect(history.getByRole("checkbox", { name: "Select Partial run to compare" })).toBeEnabled();
  await expectAccessible(page, "History, a run that stopped early");
  await page.keyboard.press("Escape");
  expect(errors).toEqual([]);
});

test("a queue comes back paused when its project opens again, to start or clear", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const folder = await newProject(page, "Paused queue");
  await start(page, "Held run", "Run");
  const footer = page.locator("footer");
  await expect(footer).toContainText("“Held run”");
  await start(page, "First queued", "Queue");
  await expect(page.getByRole("status").filter({ hasText: "Queued “First queued”. It starts after “Held run”." })).toBeVisible();
  await start(page, "Second queued", "Queue");
  await expect(footer).toContainText("2 queued · waiting");

  // Paused, the queue starts nothing once the job in progress ends.
  await page.getByRole("button", { name: /^History/ }).click();
  const history = page.getByRole("dialog", { name: "Experiment history" });
  await expect(history.getByText("Waiting for “Held run” to end")).toBeVisible();
  await history.getByRole("button", { name: "Pause the queue", exact: true }).click();
  await expect(history.getByText("Paused: nothing starts until you start it")).toBeVisible();
  await page.keyboard.press("Escape");
  await footer.getByRole("button", { name: "Cancel", exact: true }).click();
  const cancel = page.getByRole("alertdialog", { name: "Cancel this run?" });
  await expect(cancel).toContainText("What it has finished is kept and marked as partial: the layers that finished. The layer in progress isn't kept.");
  await expect(cancel).toContainText("The queue is paused, so nothing starts after it.");
  await cancel.getByRole("button", { name: "Cancel the run" }).click();
  await expect(footer).toContainText("“Held run” was cancelled", { timeout: 60_000 });
  await expect(footer).toContainText("2 queued · paused");

  // Opened again (as after a restart of the server), the project's queue is back, paused, and
  // says so. Run runs at once beside it.
  const reopened = await page.request.post("/api/projects/open", { data: { path: folder }, headers: ORIGIN });
  expect(reopened.ok()).toBe(true);
  await expect(page.getByRole("status").filter({ hasText: "2 runs wait in the queue from before. It is paused" })).toBeVisible();
  await expect(footer).toContainText("2 queued · paused");
  await page.getByRole("button", { name: "Experiment", exact: true }).click();
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  await expect(page.getByRole("button", { name: "Run", exact: true })).toBeVisible();

  await page.getByRole("button", { name: /^History/ }).click();
  const queued = history.getByRole("list", { name: "Queued" });
  await expect(queued.getByRole("listitem")).toHaveCount(2);
  await expect(history.getByRole("button", { name: "Start the queue", exact: true })).toBeVisible();
  await expectAccessible(page, "History, a paused queue");
  // Clearing asks first.
  await history.getByRole("button", { name: "Clear the queue", exact: true }).click();
  const clear = page.getByRole("alertdialog", { name: "Clear the queue?" });
  await expect(clear).toContainText("The 2 runs waiting in the queue won't run.");
  await expectAccessible(page, "Clearing the queue");
  await clear.getByRole("button", { name: "Keep it", exact: true }).click();
  await expect(queued.getByRole("listitem")).toHaveCount(2);

  // Started, it runs to the end, in order, and empties.
  await history.getByRole("button", { name: "Start the queue", exact: true }).click();
  await expect(history.getByRole("list", { name: "Queued" })).toHaveCount(0, { timeout: 60_000 });
  await page.keyboard.press("Escape");
  await expect(footer).toContainText("Ready", { timeout: 60_000 });
  await expect(footer).not.toContainText("queued ·");
  await page.getByRole("button", { name: /^History/ }).click();
  await expect(history.getByRole("button", { name: /^First queued/ })).toContainText("just now");
  await expect(history.getByRole("button", { name: /^Second queued/ })).toContainText("just now");
  await page.keyboard.press("Escape");
  expect(errors).toEqual([]);
});
