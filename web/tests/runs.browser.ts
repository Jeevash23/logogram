import { test, expect, type Page } from "@playwright/test";

import type { ContinuationsFile } from "../src/api/types";
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
  await history(page).getByRole("button", { name: /^A sweep that failed/ }).click();

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
  const row = history(page).getByRole("button", { name: /^A sweep from a notebook/ });
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

test("a run's methods line names the chat template its prompts were read in and its local model's files", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await fakeRuns(page, () => [
    {
      id: "fake-chat",
      name: "Heads in a chat",
      status: "finished",
      spec: {
        model: { id: "tiny-gpt2", source: "local", revision: null, sha256: "a".repeat(64), device: "cpu", dtype: "float32", process_weights: true },
        tokenization: { prepend_bos: false, chat_template: { kind: "user_turn", system: null } },
      },
    },
  ]);
  await open(page);
  await page.getByRole("button", { name: /^History/ }).click();
  await history(page).getByRole("button", { name: /^Heads in a chat/ }).click();
  const main = page.getByRole("main");
  await expect(main.getByRole("heading", { level: 1, name: "Heads in a chat" })).toBeVisible();
  const methods = main.locator("p").filter({ hasText: "resamples, seed" }).first();
  await expect(methods).toContainText("prompts in the chat template, as the person's message");
  await expect(methods).toContainText("tiny-gpt2, local files aaaaaaaaaaaa, float32");
  expect(errors).toEqual([]);
});

test("a cancelled run of SAE features says why it kept nothing", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await fakeRuns(page, () => [
    {
      id: "fake-features",
      name: "Chosen features",
      status: "cancelled",
      spec: { scope: { kind: "sites", sites: [{ kind: "sae_feature", layer: 1, head: null, feature: 3, position: { kind: "last" } }] } },
    },
  ]);
  await open(page);
  await page.getByRole("button", { name: /^History/ }).click();
  await history(page).getByRole("button", { name: /^Chosen features/ }).click();
  const main = page.getByRole("main");
  await expect(main.getByRole("heading", { level: 1, name: "Chosen features" })).toBeVisible();
  const callout = main.getByRole("note").filter({ hasText: "This run was cancelled" });
  await expect(callout).toContainText("It has no results: a run of SAE features reports nothing before it ends, so it keeps none.");
  expect(errors).toEqual([]);
});

test("what the steered model says follows the run shown, not the site chosen in another", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const said = (tokens: string[]) => ({ ids: tokens.map((_, i) => i), tokens });
  const continuations = (run: string, rows: ContinuationsFile["rows"]): ContinuationsFile => ({
    logogram_continuations: 1, run_id: run, receiver: "corrupt", tokens: 2, decoding: "greedy", rows,
    prompts: [{
      index: 3, text: "When Mary and John went to the store, John gave a drink to", prompt: ["When"], unsteered: said([" Mary", "."]),
      steered: rows.map((r) => ({ ...said([` ${r.label}`, "!"]), row: r.row, coefficient: 1, control: false, key: `×1 at ${r.label}` })),
    }],
  });
  await fakeRuns(page, () => [
    { id: "fake-layers", name: "Steered in every layer", status: "finished", continuations: continuations("fake-layers", [{ row: 0, label: "L0" }, { row: 1, label: "L1" }]) },
    { id: "fake-site", name: "Steered at one site", status: "finished", continuations: continuations("fake-site", [{ row: 0, label: "L1 resid pre" }]) },
  ]);
  await open(page);
  const main = page.getByRole("main");
  const lines = main.getByRole("region", { name: "What the steered model says" });
  const show = async (name: string) => {
    await page.getByRole("button", { name: /^History/ }).click();
    await history(page).getByRole("button", { name: new RegExp(`^${name}`) }).click();
    await expect(main.getByRole("heading", { level: 1, name })).toBeVisible();
  };
  // Both runs read once already, as after running them.
  await show("Steered at one site");
  await expect(lines.getByRole("rowheader")).toHaveText(["Without steering", "×1 at L1 resid pre"]);
  await show("Steered in every layer");
  await lines.getByRole("combobox", { name: "Steered site" }).selectOption("1");
  await expect(lines.getByRole("rowheader")).toHaveText(["Without steering", "×1 at L1"]);
  // A run of one steered site shows its lines, whichever layer was chosen in the other.
  await show("Steered at one site");
  await expect(lines.getByRole("rowheader")).toHaveText(["Without steering", "×1 at L1 resid pre"]);
  expect(errors).toEqual([]);
});
