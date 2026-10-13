import { test, expect, type Page } from "@playwright/test";
import { existsSync, renameSync } from "node:fs";

// Managing projects and datasets in the app: renaming and deleting them, each deletion saying what
// goes and what stays, a recent project whose folder is missing, and the model dialog waiting for
// a device that is there. Each check works in a project of its own and reopens the one that was
// open, so the other checks see the fixture as they expect.

test.describe.configure({ mode: "serial" });

const BASE = `http://127.0.0.1:${process.env.LOGOGRAM_BROWSER_PORT ?? "8877"}`;
const ORIGIN = { origin: BASE };
// Set-up requests go straight to the server, as a script would, with the session token.
const API = { authorization: "Bearer browser-test-token", origin: BASE };
let previous: string | null = null;

test.beforeEach(async ({ page }) => {
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
  previous = (await (await page.request.get("/api/state")).json()).project?.path ?? null;
});

test.afterEach(async ({ page }) => {
  if (previous) await page.request.post("/api/projects/open", { data: { path: previous }, headers: ORIGIN });
});

/** A project of its own, open in the page. */
async function scratch(page: Page, name: string): Promise<{ path: string; name: string }> {
  const state = await (await page.request.get("/api/state")).json();
  const created = await page.request.post("/api/projects", { data: { name, parent: state.projects_parent }, headers: ORIGIN });
  expect(created.ok()).toBe(true);
  await expect(page.getByRole("button", { name: new RegExp(`^${name}`) })).toBeVisible();
  return created.json();
}

/** The header's chip for the dataset chosen, by its file's name without .jsonl. */
const chip = (page: Page, name: string) => page.getByRole("button", { name: new RegExp(`^${name}\\.jsonl · `) });

/** The spec of the fixture's finished run, to save drafts of. */
async function fixtureSpec(page: Page): Promise<Record<string, unknown>> {
  const runs: { id: string; status: string }[] = await (await page.request.get("/api/runs", { headers: API })).json();
  const finished = runs.find((r) => r.status === "finished");
  expect(finished).toBeTruthy();
  return (await (await page.request.get(`/api/runs/${finished?.id}`, { headers: API })).json()).spec;
}

test("a dataset is renamed and deleted where it is chosen, and a draft that names one keeps it", async ({ page }) => {
  const spec = await fixtureSpec(page);
  await scratch(page, `Datasets ${Date.now()}`);
  for (const name of ["first", "second"]) {
    expect((await page.request.post("/api/datasets/ioi", { data: { name, n: 4 }, headers: API })).ok()).toBe(true);
  }
  await page.reload(); // lists the datasets made from outside the page
  await page.getByRole("button", { name: "Experiment", exact: true }).click();
  await page.getByRole("button", { name: "Prompts", exact: true }).click();
  await expect(chip(page, "first")).toBeVisible();

  // In the Prompts view: the file gets the new name, and stays chosen.
  await page.getByRole("button", { name: "Rename…", exact: true }).click();
  const rename = page.getByRole("dialog", { name: "Rename first.jsonl" });
  await rename.getByRole("textbox", { name: "Name" }).fill("second");
  await expect(rename.getByText("second.jsonl already exists. Choose another name.")).toBeVisible();
  await expect(rename.getByRole("button", { name: "Rename", exact: true })).toBeDisabled();
  await rename.getByRole("textbox", { name: "Name" }).fill("renamed");
  await rename.getByRole("button", { name: "Rename", exact: true }).click();
  await expect(rename).toBeHidden();
  await expect(page.getByText("Renamed first.jsonl to renamed.jsonl.")).toBeVisible();
  await expect(chip(page, "renamed")).toBeVisible();

  // From the dataset chip: what goes and what stays is said first.
  await chip(page, "renamed").click();
  await page.getByRole("menuitem", { name: "Delete renamed.jsonl…" }).click();
  const question = page.getByRole("alertdialog", { name: "Delete renamed.jsonl?" });
  await expect(question).toContainText("This deletes datasets/renamed.jsonl and its 4 prompt pairs, for good.");
  await expect(question).toContainText("Runs that ran on it keep their own copy of its prompts");
  await question.getByRole("button", { name: "Delete the dataset" }).click();
  await expect(page.getByText("Deleted datasets/renamed.jsonl.", { exact: false })).toBeVisible();
  // The other dataset is chosen now.
  await expect(chip(page, "second")).toBeVisible();

  // A draft that names it keeps it, and the refusal names the draft.
  const waiting = { ...spec, name: "Waiting idea", dataset: { path: "datasets/second.jsonl", sha256: null, limit: null } };
  const draft = await page.request.post("/api/drafts", { data: { spec: waiting }, headers: API });
  expect(draft.ok()).toBe(true);
  await page.getByRole("button", { name: "Delete…", exact: true }).click();
  await page.getByRole("alertdialog", { name: "Delete second.jsonl?" }).getByRole("button", { name: "Delete the dataset" }).click();
  await expect(page.getByRole("alert").filter({ hasText: "the draft “Waiting idea”" })).toBeVisible();
  await expect(chip(page, "second")).toBeVisible();
});

test("a project is renamed, and deleted once its name is typed", async ({ page }) => {
  const project = await scratch(page, `Scratch ${Date.now()}`);
  await page.getByRole("button", { name: new RegExp(`^${project.name}`) }).click();
  await page.getByRole("menuitem", { name: "Rename project…" }).click();
  const rename = page.getByRole("dialog", { name: "Rename the project" });
  await expect(rename.getByRole("textbox", { name: "Name" })).toHaveValue(project.name);
  const renamed = `${project.name} renamed`;
  await rename.getByRole("textbox", { name: "Name" }).fill(renamed);
  await rename.getByRole("textbox", { name: "Name" }).press("Enter");
  await expect(rename).toBeHidden();
  await expect(page.getByRole("button", { name: new RegExp(`^${renamed}`) })).toBeVisible();

  await page.getByRole("button", { name: new RegExp(`^${renamed}`) }).click();
  await page.getByRole("menuitem", { name: "Delete project…" }).click();
  const question = page.getByRole("dialog", { name: `Delete “${renamed}”?` });
  await expect(question).toContainText(`This deletes the folder ${project.path} and everything in it, for good: 0 runs, 0 drafts and 0 datasets.`);
  const remove = question.getByRole("button", { name: "Delete the project" });
  await expect(remove).toBeDisabled();
  const typed = question.getByRole("textbox", { name: `Type the project's name, “${renamed}”, to delete it` });
  await typed.fill(project.name);
  await expect(remove).toBeDisabled();
  await typed.fill(renamed);
  await remove.click();
  await expect(page.getByRole("heading", { name: "Projects", level: 1 })).toBeVisible();
  await expect(page.getByText(`Deleted “${renamed}” and its folder.`)).toBeVisible();
  expect(existsSync(project.path)).toBe(false);
  await expect(page.getByRole("button", { name: new RegExp(`^${renamed}`) })).toHaveCount(0);
});

test("a recent project whose folder is missing stays listed until it is removed", async ({ page }) => {
  const project = await scratch(page, `Wanderer ${Date.now()}`);
  await page.getByRole("button", { name: new RegExp(`^${project.name}`) }).click();
  await page.getByRole("menuitem", { name: "Close project" }).click();
  await expect(page.getByRole("heading", { name: "Projects", level: 1 })).toBeVisible();
  renameSync(project.path, `${project.path}-moved`); // a drive unplugged, say
  await page.reload();
  const item = page.getByRole("button", { name: new RegExp(`^${project.name}`) });
  await expect(item).toContainText("Folder not found");
  await item.click();
  const problem = page.getByRole("alert").filter({ hasText: "isn't there any more" });
  await expect(problem).toBeVisible();
  await problem.getByRole("button", { name: "Remove from the list" }).click();
  await expect(item).toHaveCount(0);
  expect(existsSync(`${project.path}-moved`)).toBe(true); // its folder is left as it is
});

/** Wait until no job runs on the server. */
async function idle(page: Page) {
  await expect
    .poll(async () => (await (await page.request.get("/api/state", { headers: API })).json()).job?.status ?? "none", { timeout: 60_000 })
    .not.toBe("running");
}

test("a run is rerun from its results and from the history, and says whether it reproduced exactly", async ({ page }) => {
  const spec = await fixtureSpec(page);
  await scratch(page, `Reruns ${Date.now()}`);
  expect((await page.request.post("/api/datasets/ioi", { data: { name: "ioi", n: 6 }, headers: API })).ok()).toBe(true);
  const heads = { ...spec, name: "Heads sweep", dataset: { path: "datasets/ioi.jsonl", sha256: null, limit: 4 } };
  const { run_id: id } = await (await page.request.post("/api/runs", { data: { spec: heads }, headers: API })).json();
  await idle(page);
  await page.reload();
  await page.getByRole("button", { name: "Evidence", exact: true }).click();
  await page.getByRole("button", { name: "Results", exact: true }).click();
  await expect(page.getByRole("main").getByRole("heading", { level: 1, name: "Heads sweep" })).toBeVisible();
  const said = page.getByRole("status").filter({ hasText: new RegExp(`^Rerun finished\\. Identical to ${id}: all [\\d,]+ per-prompt values match exactly\\.$`) });

  // From the run's results: the rerun opens as it streams, and says so when it ends.
  await page.getByRole("button", { name: "Rerun", exact: true }).click();
  await expect(said).toBeVisible({ timeout: 60_000 });
  await said.getByRole("button", { name: "Dismiss" }).click();
  await idle(page);

  // From the history's menu, for the same run.
  await page.getByRole("button", { name: /^History/ }).click();
  const history = page.getByRole("dialog", { name: "Experiment history" });
  await history.locator(`li[data-run="${id}"]`).getByRole("button", { name: /^Actions for/ }).click();
  await page.getByRole("menuitem", { name: "Rerun" }).click();
  await expect(history).toBeHidden();
  await expect(said).toBeVisible({ timeout: 60_000 });
  await idle(page);
  const listed: { derived_from: { run: string; kind: string } | null }[] = await (await page.request.get("/api/runs", { headers: API })).json();
  expect(listed.filter((r) => r.derived_from?.kind === "rerun" && r.derived_from.run === id)).toHaveLength(2);
});

test("the model dialog waits for a device that is there", async ({ page }) => {
  await page.getByRole("button", { name: /^tiny-gpt2 · / }).click();
  const dialog = page.getByRole("dialog", { name: "Load a model" });
  const device = dialog.getByRole("radiogroup", { name: "Device" });
  await device.getByRole("radio", { name: "CUDA" }).click();
  const problem = dialog.getByRole("alert").filter({ hasText: "The device chosen isn't available here" });
  await expect(problem).toContainText("CUDA was chosen, but PyTorch can't see a CUDA GPU here.");
  await expect(dialog.getByRole("button", { name: "Load", exact: true })).toBeDisabled();
  await device.getByRole("radio", { name: "CPU" }).click();
  await expect(problem).toBeHidden();
  await page.keyboard.press("Escape");
  // The loaded model is still loaded.
  await expect(page.getByRole("button", { name: /^tiny-gpt2 · cpu/ })).toBeVisible();
});
