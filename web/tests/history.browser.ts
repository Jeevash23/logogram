import { test, expect, type Page } from "@playwright/test";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

// Managing runs in the history: titles, stars and the archive, search, and deleting a run after
// asking. The test works in a project of its own and reopens the one that was open, so the other
// checks see the fixture as they expect.

test.describe.configure({ mode: "serial" });

const BASE = `http://127.0.0.1:${process.env.LOGOGRAM_BROWSER_PORT ?? "8877"}`;
const ORIGIN = { origin: BASE };
// Set-up requests go straight to the server, as a script would, with the session token.
const API = { authorization: "Bearer browser-test-token", origin: BASE };
let previous: string | null = null;

interface Listing {
  id: string;
  name: string;
  status: string;
  title?: string | null;
}

test.beforeEach(async ({ page }) => {
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
  previous = (await (await page.request.get("/api/state")).json()).project?.path ?? null;
});

test.afterEach(async ({ page }) => {
  if (previous) await page.request.post("/api/projects/open", { data: { path: previous }, headers: ORIGIN });
});

async function runs(page: Page): Promise<Listing[]> {
  return (await page.request.get("/api/runs", { headers: API })).json();
}

/** Run a spec and wait until it has finished and let go of the server. */
async function run(page: Page, request: Promise<{ json: () => Promise<{ run_id: string }> }>): Promise<string> {
  const { run_id: id } = await (await request).json();
  await expect.poll(async () => (await runs(page)).find((r) => r.id === id)?.status, { timeout: 60_000 }).toBe("finished");
  await expect.poll(async () => (await (await page.request.get("/api/state", { headers: API })).json()).job?.status ?? "none", { timeout: 60_000 }).not.toBe("running");
  return id;
}

/**
 * A project of its own, open in the page: two runs of the fixture's spec (the first rerun once),
 * a draft, and a run whose spec can't be read. Returns the project's folder and the runs' ids.
 */
async function projectWithRuns(page: Page) {
  const fixture = (await runs(page)).find((r) => r.status === "finished");
  expect(fixture).toBeTruthy();
  const { spec } = await (await page.request.get(`/api/runs/${fixture?.id}`, { headers: API })).json();
  const state = await (await page.request.get("/api/state")).json();
  const created = await page.request.post("/api/projects", { data: { name: `Managing runs ${Date.now()}`, parent: state.projects_parent }, headers: ORIGIN });
  expect(created.ok()).toBe(true);
  const project = await created.json();
  await expect(page.getByText("No prompts yet.", { exact: false })).toBeVisible();
  expect((await page.request.post("/api/datasets/ioi", { data: { name: "ioi", n: 6 }, headers: API })).ok()).toBe(true);
  const named = (name: string) => ({ ...spec, name, dataset: { ...spec.dataset, path: "datasets/ioi.jsonl", sha256: null, limit: 4 } });
  const heads = await run(page, page.request.post("/api/runs", { data: { spec: named("Heads sweep") }, headers: API }));
  const rerun = await run(page, page.request.post(`/api/runs/${heads}/rerun`, { headers: API }));
  const other = await run(page, page.request.post("/api/runs", { data: { spec: named("Second sweep") }, headers: API }));
  const draft = (await (await page.request.post("/api/drafts", { data: { spec: named("Next idea") }, headers: API })).json()).run_id;
  const broken = "20000101-000000-broken";
  mkdirSync(join(project.path, "experiments", broken), { recursive: true });
  writeFileSync(join(project.path, "experiments", broken, "spec.json"), "{");
  return { path: project.path as string, heads, rerun, other, draft, broken };
}

async function openHistory(page: Page) {
  await page.getByRole("button", { name: /^History/ }).click();
  const history = page.getByRole("dialog", { name: "Experiment history" });
  await expect(history).toBeVisible();
  return history;
}

const row = (history: ReturnType<Page["getByRole"]>, id: string) => history.locator(`li[data-run="${id}"]`);

test("runs are renamed, starred, archived and found, with the keyboard", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const ids = await projectWithRuns(page);
  await page.reload();
  const history = await openHistory(page);
  const list = history.getByRole("list", { name: "Runs" });
  await expect(list.locator(":scope > li")).toHaveCount(5);
  // The rerun sits under the run it came from.
  const order = await list.locator(":scope > li").evaluateAll((items) => items.map((li) => li.getAttribute("data-run")));
  expect(order.indexOf(ids.rerun)).toBe(order.indexOf(ids.heads) + 1);

  // F2 renames a run in place; Enter keeps the title, and the keyboard stays on the run.
  const heads = row(history, ids.heads);
  await heads.getByRole("button", { name: /^Heads sweep/ }).focus();
  await page.keyboard.press("F2");
  const field = heads.getByRole("textbox", { name: "Title of “Heads sweep”" });
  await expect(field).toBeFocused();
  await expect(field).toHaveValue("Heads sweep");
  await page.keyboard.type("Name movers, clean");
  await page.keyboard.press("Enter");
  await expect(heads.getByRole("button", { name: /^Name movers, clean/ })).toBeFocused();
  // Escape leaves a title as it was, and keeps the history open.
  const second = row(history, ids.other);
  await second.getByRole("button", { name: /^Second sweep/ }).focus();
  await page.keyboard.press("F2");
  await page.keyboard.type("Not this");
  await page.keyboard.press("Escape");
  await expect(history).toBeVisible();
  await expect(second.getByRole("button", { name: /^Second sweep/ })).toBeFocused();

  // A star, from its button; only starred runs, from the filter.
  await heads.getByRole("button", { name: "Star “Name movers, clean”" }).click();
  await expect(heads.getByRole("button", { name: "Star “Name movers, clean”" })).toHaveAttribute("aria-pressed", "true");
  const show = history.getByRole("radiogroup", { name: "Show" });
  await show.getByRole("radio", { name: /^Starred/ }).click();
  await expect(list.locator(":scope > li")).toHaveCount(1);
  await expect(list).toContainText("Name movers, clean");
  await show.getByRole("radio", { name: /^All/ }).click();

  // Archive, from the run's menu with the keyboard: it leaves the list, and the keyboard stays in it.
  await second.getByRole("button", { name: "Actions for “Second sweep”" }).focus();
  await page.keyboard.press("Enter");
  await page.getByRole("menuitem", { name: "Archive", exact: true }).focus();
  await page.keyboard.press("Enter");
  await expect(row(history, ids.other)).toHaveCount(0);
  await expect(list.locator(":scope > li")).toHaveCount(4);
  await expect.poll(() => page.evaluate(() => document.activeElement?.closest("li")?.getAttribute("data-run") ?? null)).not.toBeNull();
  await show.getByRole("radio", { name: /^Archived/ }).click();
  await expect(list.locator(":scope > li")).toHaveCount(1);
  await show.getByRole("radio", { name: /^All/ }).click();

  // Search by title, id, method, model or dataset; archived runs it finds are pointed out.
  const search = history.getByRole("searchbox", { name: "Search runs" });
  await search.fill("name movers");
  await expect(list.locator(":scope > li")).toHaveCount(1);
  await expect(history.getByText("1 run found.")).toBeVisible();
  await search.fill(ids.draft);
  await expect(list).toContainText("Next idea");
  // The model, the dataset as it was named when the run read it, and the method: the first run
  // and the draft (the second is archived; the rerun read the first run's snapshot).
  await search.fill("tiny-gpt2 ioi.jsonl patch");
  await expect(list.locator(":scope > li")).toHaveCount(2);
  await search.fill("second sweep");
  await expect(history.getByText("No runs match “second sweep”.")).toBeVisible();
  await expect(history.getByText("1 archived run matches too.")).toBeVisible();
  await history.getByRole("button", { name: "Show archived runs" }).click();
  await expect(list.locator(":scope > li")).toHaveCount(1);
  await row(history, ids.other).getByRole("button", { name: "Actions for “Second sweep”" }).click();
  await page.getByRole("menuitem", { name: "Move out of the archive" }).click();
  await expect(history.getByText("No archived runs match “second sweep”.")).toBeVisible();
  // Escape clears the search first; then it closes the history.
  await search.focus();
  await page.keyboard.press("Escape");
  await expect(search).toHaveValue("");
  await expect(history).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(history).not.toBeVisible();

  // The title is what the run is called elsewhere: on Compare, and in the command palette, which
  // searches every run (here, a draft by its id).
  await page.getByRole("button", { name: "Evidence", exact: true }).click();
  await page.getByRole("button", { name: "Compare runs", exact: true }).click();
  const runA = page.getByRole("combobox", { name: "Run A" });
  await expect(runA.locator("optgroup[label=Starred] option")).toHaveText([`Name movers, clean · ${ids.heads}`]);
  await page.keyboard.press("ControlOrMeta+k");
  await page.getByLabel("Command", { exact: true }).fill(ids.draft);
  await expect(page.getByRole("option", { name: "Open draft: Next idea" })).toBeVisible();
  await page.getByLabel("Command", { exact: true }).fill("name movers");
  await page.keyboard.press("Enter");
  await expect(page.getByRole("main").getByRole("heading", { level: 1, name: "Name movers, clean" })).toBeVisible();
  await expect(page.getByText("Spec name: Heads sweep")).toBeVisible();
  expect(errors).toEqual([]);
});

test("a run is deleted after asking, and a run whose spec can't be read says why", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const ids = await projectWithRuns(page);
  await page.reload();
  const history = await openHistory(page);

  // A run whose spec can't be read is listed, saying what is wrong and how to fix it.
  const broken = row(history, ids.broken);
  await expect(broken).toContainText("unreadable");
  await expect(broken).toContainText("Its spec.json can't be read: Invalid JSON");
  await expect(broken).toContainText("or delete the run.");

  // Deleting asks first, saying what goes and what stays, with the keyboard on keeping it.
  const heads = row(history, ids.heads);
  const actions = heads.getByRole("button", { name: "Actions for “Heads sweep”" });
  await actions.click();
  await page.getByRole("menuitem", { name: "Delete…" }).click();
  const question = page.getByRole("alertdialog", { name: "Delete “Heads sweep”?" });
  await expect(question).toContainText(`experiments/${ids.heads}`);
  await expect(question).toContainText("The run derived from it (a rerun) stays, but no longer links back to it.");
  await expect(question.getByRole("button", { name: "Keep it" })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(question).not.toBeVisible();
  await expect(history).toBeVisible();
  await expect(actions).toBeFocused();
  await actions.click();
  await page.getByRole("menuitem", { name: "Delete…" }).click();
  await question.getByRole("button", { name: "Delete the run" }).click();
  await expect(page.getByRole("status").filter({ hasText: "Deleted “Heads sweep”." })).toBeVisible();
  await expect(heads).toHaveCount(0);
  expect(existsSync(join(ids.path, "experiments", ids.heads))).toBe(false);
  // The rerun stays, and says where it came from.
  await expect(row(history, ids.rerun)).toContainText("From a deleted run");
  // The keyboard moved on to the run that took its place.
  await expect.poll(() => page.evaluate(() => document.activeElement?.closest("li")?.getAttribute("data-run") ?? null)).not.toBeNull();

  // The damaged run can be deleted too, and a draft.
  await broken.getByRole("button", { name: `Actions for “${ids.broken}”` }).click();
  await page.getByRole("menuitem", { name: "Delete…" }).click();
  await expect(page.getByRole("alertdialog")).toContainText("with everything in it");
  await page.getByRole("alertdialog").getByRole("button", { name: "Delete the run" }).click();
  await expect(broken).toHaveCount(0);
  await row(history, ids.draft).getByRole("button", { name: "Actions for “Next idea”" }).click();
  await page.getByRole("menuitem", { name: "Delete…" }).click();
  await page.getByRole("alertdialog").getByRole("button", { name: "Delete the draft" }).click();
  await expect(row(history, ids.draft)).toHaveCount(0);
  expect((await runs(page)).map((r) => r.id).sort()).toEqual([ids.other, ids.rerun].sort());
  expect(errors).toEqual([]);
});
