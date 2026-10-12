import { test, expect, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import type { ProjectInfo, RunDetail, RunListing, Spec, SweepStarted } from "../src/api/types";

// A sweep: a run swept over its donor seed from its results, then read in the sweep view, which
// is reached from the run, the history, a comparison and the command palette, and works with the
// keyboard. The test works in a project of its own, so the runs it makes don't change what the
// other checks see, and reopens the fixture's project afterwards.

test.describe.configure({ mode: "serial" });
// Axe runs inside the page, so this context sets the app's content security policy aside.
test.use({ bypassCSP: true, reducedMotion: "reduce" });

const ORIGIN = { origin: `http://127.0.0.1:${process.env.LOGOGRAM_BROWSER_PORT ?? "8877"}` };
const AXE = readFileSync(fileURLToPath(new URL("../node_modules/axe-core/axe.min.js", import.meta.url)), "utf8");
let previous: string | null = null;

test.beforeEach(async ({ page }) => {
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
  previous = (await (await page.request.get("/api/state")).json()).project?.path ?? null;
});

test.afterEach(async ({ page }) => {
  if (previous) await page.request.post("/api/projects/open", { data: { path: previous }, headers: ORIGIN });
});

async function violations(page: Page): Promise<string[]> {
  await page.evaluate(AXE);
  return page.evaluate(async () => {
    // @ts-expect-error axe is defined by the script above
    const result = await window.axe.run(document, { resultTypes: ["violations"] });
    return result.violations.map((v: { id: string; nodes: { target: string[] }[] }) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).join(", ")}`);
  });
}

/** A project of the test's own, with a finished resample ablation of the fixture's model in it. */
async function projectWithRun(page: Page): Promise<{ project: ProjectInfo; runId: string }> {
  const state = await (await page.request.get("/api/state")).json();
  const fixture = { ...ORIGIN, "x-logogram-project": state.project.session_id };
  const runs = (await (await page.request.get("/api/runs", { headers: fixture })).json()) as RunListing[];
  const base = runs.find((r) => r.status === "finished");
  expect(base).toBeTruthy();
  const detail = (await (await page.request.get(`/api/runs/${encodeURIComponent(base!.id)}`, { headers: fixture })).json()) as RunDetail;
  const created = await page.request.post("/api/projects", { data: { name: `Sweep ${Date.now()}`, parent: state.projects_parent }, headers: ORIGIN });
  expect(created.ok()).toBe(true);
  const project = (await created.json()) as ProjectInfo;
  const scoped = { ...ORIGIN, "x-logogram-project": project.session_id };
  expect((await page.request.post("/api/datasets/ioi", { data: { name: "ioi", n: 6, seed: 0 }, headers: scoped })).ok()).toBe(true);
  const spec: Spec = {
    ...detail.spec,
    name: "Donors",
    notes: "",
    dataset: { path: "datasets/ioi.jsonl", sha256: null, limit: null },
    experiment: { kind: "ablation", baseline: { kind: "resample", pool: "corrupt", donors: 2, seed: 0 } },
  };
  const started = await page.request.post("/api/runs", { data: { spec }, headers: scoped });
  expect(started.ok()).toBe(true);
  const runId = ((await started.json()) as { run_id: string }).run_id;
  await expect
    .poll(
      async () => ((await (await page.request.get("/api/runs", { headers: scoped })).json()) as RunListing[]).find((r) => r.id === runId)?.status,
      { timeout: 60_000 },
    )
    .toBe("finished");
  return { project, runId };
}

test("a sweep over a seed: started from a run, read across its values, reached from everywhere", async ({ page }) => {
  // Three runs of a tiny model, one after another, on a machine that may be busy.
  test.setTimeout(240_000);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const { project, runId } = await projectWithRun(page);
  const main = page.getByRole("main");
  const tab = (name: string) => page.getByRole("button", { name, exact: true });
  // The tab follows the project it was switched to.
  await expect(page.getByRole("button", { name: new RegExp(project.name) }).first()).toBeVisible();
  // The run started and ended outside this tab, which may have opened the project before it
  // existed: the run's link opens it.
  await page.goto(`/?token=browser-test-token#/results/${encodeURIComponent(runId)}`);
  await expect(main.getByRole("heading", { level: 1, name: "Donors" })).toBeVisible();

  // The dialog: the run's own donor seed comes first, and stands in for itself.
  await main.getByRole("button", { name: "Sweep…", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "Sweep this run" });
  await expect(dialog.getByRole("radio", { name: /^Donor seed/ })).toHaveAttribute("aria-checked", "true");
  const seeds = dialog.getByRole("textbox", { name: "Donor seeds, in order" });
  await expect(seeds).toHaveValue("0, 1, 2, 3, 4");
  await seeds.fill("0, 1, 1");
  await expect(dialog.getByRole("alert")).toContainText("Donor seed 1 is listed twice. List each donor seed once.");
  await expect(dialog.getByRole("button", { name: /^Queue/ })).toBeDisabled();
  await seeds.fill("0, 1, 2");
  await expect(dialog.getByRole("status")).toHaveText(
    "Queues 2 runs, one after another: donor seeds 1 and 2. Donor seed 0 is this run, which stands in for it.",
  );
  await dialog.getByRole("button", { name: "Queue 2 runs", exact: true }).click();
  await expect(dialog).toBeHidden();
  await expect(page.getByRole("status").filter({ hasText: "Queued 2 runs of the sweep over donor seeds; this run stands in for donor seed 0." })).toBeVisible();

  // The sweep view: which choice varied, over which values, and each value's run.
  await expect(tab("Sweeps")).toHaveAttribute("aria-current", "page");
  await expect(page).toHaveURL(/#\/sweep\/[^/]+$/);
  const title = main.getByRole("heading", { level: 1 });
  await expect(title).toHaveText(/^\d+ of \d+ sites? (keeps its|keep their) sign and size across 3 donor seeds$/, { timeout: 90_000 });
  await expect(main.getByText("Donors, run once for each of 3 donor seeds: 0, 1 and 2.")).toBeVisible();
  const values = main.getByRole("region", { name: "The values and their runs" });
  await expect(values.getByRole("row")).toHaveCount(4);
  await expect(values.getByRole("row").nth(1)).toContainText("the run the sweep was made from");
  await expect(main.getByText(/^Chosen on the first value, donor seed 0:/)).toBeVisible();
  await expect(main.getByText("Of the 8 sites shown,", { exact: false })).toBeVisible();
  await expect(main.getByText(/^Each bar is one run's 95% interval/)).toBeVisible();
  expect(await violations(page)).toEqual([]);

  // The chart, with the keyboard: up and down between sites, left and right between values.
  const chart = main.getByRole("group", { name: "Normalized effect of the 8 sites at each donor seed" });
  const readout = main.locator('[aria-live="polite"]').filter({ hasText: / at donor seed / });
  await chart.focus();
  await expect(readout).toHaveText(/^\S.* at donor seed 0: [+−]\d\.\d{3}, 95% interval .+\.$/);
  await page.keyboard.press("ArrowRight");
  await expect(readout).toHaveText(/ at donor seed 1: /);
  await page.keyboard.press("End");
  await page.keyboard.press("ArrowDown");
  await expect(readout).toHaveText(/ at donor seed 2: /);
  const site = (await readout.textContent())?.split(" at donor seed ")[0] ?? "";
  // The selected site at that value, its spread across values beside a run's own interval.
  const selected = main.getByRole("region", { name: `${site} at donor seed 2` });
  await expect(selected).toContainText("Across the donor seeds");
  await expect(selected).toContainText("Within a run");
  // Enter opens that value's run, at the site.
  await chart.focus();
  await page.keyboard.press("Enter");
  await expect(main.getByRole("heading", { level: 1, name: "Donors · donor seed 2" })).toBeVisible();
  await expect(page).toHaveURL(/#\/results\/[^/]+$/);
  await expect(page.getByRole("complementary", { name: "Inspector" }).getByRole("heading", { level: 2, name: site })).toBeVisible();

  // The run says it belongs to the sweep, and leads back to it.
  const note = main.getByText("This run is donor seed 2 in a sweep over 3 donor seeds.");
  await expect(note).toBeVisible();
  await main.getByRole("button", { name: "Show the sweep", exact: true }).click();
  await expect(title).toHaveText(/across 3 donor seeds$/);

  // Chosen on all values instead, and said so.
  await main.getByRole("radiogroup", { name: "Choose the sites on" }).getByRole("radio", { name: "All values" }).click();
  await expect(main.getByText(/^Chosen on all values:/)).toBeVisible();

  // The table, as CSV: one row per shown site and value.
  const download = page.waitForEvent("download");
  await main.getByRole("button", { name: "Export CSV", exact: true }).click();
  const file = await download;
  expect(file.suggestedFilename()).toMatch(/^sweep-[A-Za-z0-9._-]+\.csv$/);
  const lines = readFileSync((await file.path()) as string, "utf8").trim().split("\n");
  expect(lines[0]).toBe(
    "sweep,field,value_index,value,run,status,site_rank,site,kind,layer,head,feature,neuron,position,variant,effect,effect_lo,effect_hi,effect_sd,n,verdict,spread_mean,spread_sd",
  );
  expect(lines).toHaveLength(1 + 8 * 3);

  // From the history: the sweep's runs nest under the run it was made from, and lead to it.
  await page.getByRole("button", { name: /^History/ }).click();
  const history = page.getByRole("dialog", { name: "Experiment history" });
  const member = history.locator("li[data-run]").filter({ hasText: "Sweep · donor seed 1" });
  await expect(member).toContainText("in a sweep over 3 donor seeds");
  const memberId = (await member.getAttribute("data-run")) as string;
  await expect(history.locator(`li[data-run="${runId}"]`)).toContainText("in a sweep over 3 donor seeds");
  await member.getByRole("button", { name: "Actions for “Sweep · donor seed 1”" }).click();
  await page.getByRole("menuitem", { name: "Show the sweep" }).click();
  await expect(history).toBeHidden();
  await expect(title).toHaveText(/across 3 donor seeds$/);

  // From a comparison of two of its values.
  await tab("Compare runs").click();
  await main.getByRole("combobox", { name: "Run A" }).selectOption(runId);
  await main.getByRole("combobox", { name: "Run B" }).selectOption(memberId);
  await expect(main.getByText("A and B are donor seed 0 and donor seed 1 of one sweep over 3 donor seeds.")).toBeVisible();
  await main.getByRole("button", { name: "Show the sweep", exact: true }).click();
  await expect(tab("Sweeps")).toHaveAttribute("aria-current", "page");

  // From the command palette.
  await tab("Results").click();
  await page.keyboard.press("ControlOrMeta+k");
  await page.getByLabel("Command", { exact: true }).fill("open sweep");
  await expect(page.getByRole("option", { name: /^Open sweep: Donors over 3 donor seeds/ })).toHaveAttribute("aria-selected", "true");
  await page.keyboard.press("Enter");
  await expect(tab("Sweeps")).toHaveAttribute("aria-current", "page");
  await expect(title).toHaveText(/across 3 donor seeds$/);

  // A value whose run is gone is missing, never zero.
  const scoped = { ...ORIGIN, "x-logogram-project": project.session_id };
  expect((await page.request.delete(`/api/runs/${encodeURIComponent(memberId)}`, { headers: scoped })).ok()).toBe(true);
  await expect(values.getByRole("row").nth(2)).toContainText("None: not run");
  await expect(title).toHaveText(/across 2 donor seeds$/);
  await expect(main.getByText("1 of 3 values have no results", { exact: false })).toBeVisible();
  expect(await violations(page)).toEqual([]);
  expect(errors).toEqual([]);
});

test("a sweep whose runs all wait in the queue says so", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const state = await (await page.request.get("/api/state")).json();
  const fixture = { ...ORIGIN, "x-logogram-project": state.project.session_id };
  const runs = (await (await page.request.get("/api/runs", { headers: fixture })).json()) as RunListing[];
  const base = runs.find((r) => r.status === "finished");
  expect(base).toBeTruthy();
  const detail = (await (await page.request.get(`/api/runs/${encodeURIComponent(base!.id)}`, { headers: fixture })).json()) as RunDetail;
  const created = await page.request.post("/api/projects", { data: { name: `Waiting sweep ${Date.now()}`, parent: state.projects_parent }, headers: ORIGIN });
  expect(created.ok()).toBe(true);
  const project = (await created.json()) as ProjectInfo;
  const scoped = { ...ORIGIN, "x-logogram-project": project.session_id };
  expect((await page.request.post("/api/datasets/ioi", { data: { name: "ioi", n: 4, seed: 0 }, headers: scoped })).ok()).toBe(true);
  const spec: Spec = { ...detail.spec, name: "Seeds", notes: "", dataset: { path: "datasets/ioi.jsonl", sha256: null, limit: null } };
  // The fixture server holds a run named "Held run" until it is cancelled: the sweep waits behind it.
  expect((await page.request.post("/api/runs", { data: { spec: { ...spec, name: "Held run" } }, headers: scoped })).ok()).toBe(true);
  const started = await page.request.post("/api/sweeps", { data: { spec, field: "statistics.seed", values: [1, 2] }, headers: scoped });
  expect(started.ok()).toBe(true);
  const sweep = (await started.json()) as SweepStarted;
  expect(sweep.job).toBeNull();
  expect(sweep.queue).toHaveLength(2);
  try {
    await page.goto(`/?token=browser-test-token#/sweep/${encodeURIComponent(sweep.sweep)}`);
    const main = page.getByRole("main");
    await expect(main.getByRole("heading", { level: 1, name: "This sweep's runs are waiting in the queue" })).toBeVisible();
    await expect(main.getByText("Its 2 runs wait in the queue. The sweep is shown here once its first run starts.")).toBeVisible();
    await expect(main.getByText("This sweep can't be shown")).toHaveCount(0);
    expect(await violations(page)).toEqual([]);
  } finally {
    // Nothing of it runs: the queue is cleared, then the held run cancelled.
    expect((await page.request.delete("/api/queue", { headers: scoped })).ok()).toBe(true);
    expect((await page.request.post("/api/jobs/cancel", { headers: ORIGIN })).ok()).toBe(true);
    await expect.poll(async () => (await (await page.request.get("/api/state")).json()).job?.status, { timeout: 60_000 }).not.toBe("running");
  }
  expect(errors).toEqual([]);
});
