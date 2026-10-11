import { test, expect, type Page } from "@playwright/test";

import { comparedAcrossMachines, fakeRuns, otherVersions, TRACEBACK, type FakeRun } from "./fakeRuns";

// What a run that didn't finish here says: why it failed (with its traceback), that it runs in
// another process (until it ends), and where a finished run ran when that differs from here.

test.describe.configure({ mode: "serial" });

async function open(page: Page) {
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
}

function history(page: Page) {
  return page.getByRole("dialog", { name: "Experiment history" });
}

test("a failed run says why, with its traceback behind Show details", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const error = "Can't load the configuration of <huggingface cache>/models--example/snapshots/abc.";
  await fakeRuns(page, () => [{ id: "fake-failed", name: "A sweep that failed", status: "failed", error, traceback: TRACEBACK }]);
  await open(page);
  await page.getByRole("button", { name: /^History/ }).click();
  await history(page).getByRole("button", { name: /A sweep that failed/ }).click();

  const main = page.getByRole("main");
  await expect(main.getByRole("heading", { level: 1, name: "A sweep that failed" })).toBeVisible();
  const callout = main.getByRole("alert").filter({ hasText: "This run failed" });
  await expect(callout).toContainText(error);
  const traceback = callout.getByLabel("Traceback", { exact: true });
  await expect(traceback).toBeHidden();
  // The disclosure opens from the keyboard.
  await callout.getByText("Show details", { exact: true }).focus();
  await page.keyboard.press("Enter");
  await expect(traceback).toBeVisible();
  await expect(callout.getByText("Hide details", { exact: true })).toBeVisible();
  for (const line of TRACEBACK.split("\n")) await expect(traceback).toContainText(line.trim());
  // Long tracebacks scroll inside their box, which the keyboard can reach.
  await traceback.focus();
  await expect(traceback).toBeFocused();

  await callout.getByRole("button", { name: "Copy the traceback" }).click();
  await expect(page.getByRole("status").filter({ hasText: "Traceback copied." })).toBeVisible();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(TRACEBACK);
  await callout.getByText("Hide details", { exact: true }).click();
  await expect(traceback).toBeHidden();
  expect(errors).toEqual([]);
});

test("a run in another process says so, and its results appear when it ends", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  let status: FakeRun["status"] = "running";
  await fakeRuns(page, () => [{ id: "fake-elsewhere", name: "A sweep from a notebook", status, elsewhere: status === "running" }]);
  await open(page);
  await page.getByRole("button", { name: /^History/ }).click();
  const row = history(page).getByRole("button", { name: /A sweep from a notebook/ });
  await expect(row).toContainText("running in another process");
  await expect(row.getByRole("status", { name: "Running in another process" })).toBeVisible();
  await row.click();

  const main = page.getByRole("main");
  await expect(main.getByRole("heading", { level: 1, name: "A sweep from a notebook" })).toBeVisible();
  await expect(main.getByText(/^Running in another process( · on .+)?$/)).toBeVisible();
  const elsewhere = main.getByRole("note").filter({ hasText: "This run is running in another process" });
  await expect(elsewhere).toContainText("Its results appear here when it finishes.");
  // Nothing streams from another process: there is no progress bar to wait on.
  await expect(main.getByRole("progressbar")).toHaveCount(0);

  status = "finished";
  // The history is read again on a slow timer, so the run's end shows without a reload.
  const notice = page.getByRole("status").filter({ hasText: "“A sweep from a notebook”, running in another process, finished." });
  await expect(notice).toBeVisible({ timeout: 20_000 });
  await expect(elsewhere).toHaveCount(0);
  await expect(main.getByText("Strongest effect", { exact: true })).toBeVisible();
  await expect(main.getByText(/^Finished /)).toBeVisible();
  expect(errors).toEqual([]);
});

test("results say when a run ran with other versions, and Compare says which", async ({ page }) => {
  await otherVersions(page, { torch: "0.0.1" });
  await comparedAcrossMachines(page, [
    { key: "torch", label: "PyTorch", a: "2.5.1", b: "2.6.0" },
    { key: "device.type", label: "Device", a: "cpu", b: "cuda" },
  ]);
  await open(page);
  await page.getByRole("button", { name: "Evidence", exact: true }).click();
  await page.getByRole("button", { name: "Results", exact: true }).click();
  const main = page.getByRole("main");
  await expect(main.getByRole("heading", { level: 1, name: "Without BOS" })).toBeVisible();
  const note = main.getByRole("note").filter({ hasText: "A rerun here may differ in the last digits." });
  await expect(note).toHaveText(/^This run ran with PyTorch \S+ \(here 0\.0\.1\)\. A rerun here may differ in the last digits\.$/);
  const runId = decodeURIComponent(new URL(page.url()).hash.split("/").at(-1) ?? "");

  await page.getByRole("button", { name: "Compare runs", exact: true }).click();
  await page.getByRole("combobox", { name: "Run A" }).selectOption(runId);
  await page.getByRole("combobox", { name: "Run B" }).selectOption(runId);
  const where = main.getByRole("region", { name: "Where the runs ran" });
  await expect(where).toContainText("They ran with other versions or on other devices, which alone can change results in the last digits");
  for (const text of ["PyTorch", "2.5.1", "2.6.0", "Device", "CPU", "CUDA"]) await expect(where).toContainText(text);
});
