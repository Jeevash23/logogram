import { test, expect } from "@playwright/test";

// Runs queued while another runs start when it ends. The fixture server holds runs named "Held
// run" until they are cancelled, so there is always one to queue behind. The test works in a
// project of its own and reopens the one that was open, so the other checks see the fixture.

test.describe.configure({ mode: "serial" });

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

test("a run queued while another runs starts when that one ends", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const state = await (await page.request.get("/api/state")).json();
  const created = await page.request.post("/api/projects", { data: { name: `Queue ${Date.now()}`, parent: state.projects_parent }, headers: ORIGIN });
  expect(created.ok()).toBe(true);
  await expect(page.getByText("No prompts yet.", { exact: false })).toBeVisible();
  await page.getByRole("spinbutton", { name: "Prompts", exact: true }).fill("4");
  await page.getByRole("button", { name: "Generate 4 prompts", exact: true }).click();
  await expect(page.getByRole("table")).toContainText("When");

  await page.getByRole("button", { name: "Configure", exact: true }).click();
  const name = page.getByRole("textbox", { name: "Name", exact: true });
  await name.fill("Held run");
  await page.getByRole("button", { name: "Run", exact: true }).click();
  const footer = page.locator("footer");
  await expect(footer).toContainText("“Held run”");

  // While it runs, Run reads Queue, and the experiment waits its turn.
  await page.getByRole("button", { name: "Experiment", exact: true }).click();
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  await name.fill("Second run");
  await page.getByRole("button", { name: "Queue", exact: true }).click();
  await expect(page.getByRole("status").filter({ hasText: "Queued “Second run”. It starts after “Held run”." })).toBeVisible();
  // Ctrl/⌘ Enter queues too.
  await name.fill("Removed run");
  await page.keyboard.press("ControlOrMeta+Enter");
  await expect(page.getByRole("status").filter({ hasText: "Queued “Removed run”. It starts after “Second run”." })).toBeVisible();
  await expect(footer).toContainText("2 queued");

  // The history lists the queue above the runs; a queued run can be taken out before it starts.
  await page.getByRole("button", { name: /^History/ }).click();
  const history = page.getByRole("dialog", { name: "Experiment history" });
  const queued = history.getByRole("list", { name: "Queued" });
  await expect(queued.getByRole("listitem")).toHaveCount(2);
  await expect(queued.getByRole("listitem").first()).toContainText("Second run");
  await queued.getByRole("button", { name: "Remove “Removed run” from the queue" }).click();
  await expect(queued.getByRole("listitem")).toHaveCount(1);
  // The keyboard stays in the list.
  await expect(queued.getByRole("button", { name: "Remove “Second run” from the queue" })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("status").filter({ hasText: "Removed “Removed run” from the queue." })).toBeVisible();
  await expect(footer).toContainText("1 queued");

  // Cancelling the held run leaves the queue: the queued run starts, and runs to the end.
  await footer.getByRole("button", { name: "Cancel", exact: true }).click();
  const cancel = page.getByRole("alertdialog", { name: "Cancel this run?" });
  await expect(cancel).toContainText("The run waiting in the queue starts next.");
  await cancel.getByRole("button", { name: "Cancel the run" }).click();
  await expect(footer).toContainText("Ready", { timeout: 60_000 });
  await expect(page.getByRole("button", { name: "Run", exact: true })).toBeEnabled();
  await page.getByRole("button", { name: /^History/ }).click();
  await expect(history.getByRole("list", { name: "Queued" })).toHaveCount(0);
  await expect(history.getByRole("button", { name: /^Second run/ })).toContainText("just now");
  await expect(history.getByRole("button", { name: /^Held run/ })).toContainText("cancelled");
  await expect(history.getByRole("button", { name: /^Removed run/ })).toHaveCount(0);
  await page.keyboard.press("Escape");
  expect(errors).toEqual([]);
});
