import { test, expect, type Page } from "@playwright/test";

import type { DatasetDetail, PromptRecord } from "../src/api/types";
import { matchRows } from "../src/lib/promptRows";

// The prompts table: searched, paged, chosen from the keyboard, and edited into a new dataset.
// The test works in a project of its own, generated from a task, and reopens the project that
// was open before it, so the other browser checks see the fixture as they expect.

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

/** A dataset as the server reads it, asked for as the page asks: for the project it shows. */
async function readDataset(page: Page, path: string): Promise<DatasetDetail> {
  const state = await (await page.request.get("/api/state")).json();
  const response = await page.request.get(`/api/dataset?path=${encodeURIComponent(path)}`, { headers: { "x-logogram-project": state.project.session_id } });
  expect(response.ok()).toBe(true);
  return response.json();
}

test("prompt pairs are searched, paged, chosen, edited and removed, then saved as a new dataset", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const state = await (await page.request.get("/api/state")).json();
  const created = await page.request.post("/api/projects", { data: { name: `Prompt edits ${Date.now()}`, parent: state.projects_parent }, headers: ORIGIN });
  expect(created.ok()).toBe(true);
  await expect(page.getByText("No prompts yet.", { exact: false })).toBeVisible();
  await page.getByRole("button", { name: "Experiment", exact: true }).click();
  await page.getByRole("button", { name: "Prompts", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "The prompt pairs a run measures" })).toBeVisible();
  const generator = page.getByRole("radiogroup", { name: "Add prompts" }).getByRole("radio", { name: "Generate a task" });
  if ((await generator.getAttribute("aria-checked")) !== "true") await generator.click();
  await page.getByRole("spinbutton", { name: "Prompts", exact: true }).fill("150");
  await page.getByRole("button", { name: "Generate 150 prompts", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "150 prompt pairs in ioi.jsonl" })).toBeVisible();
  const original = await readDataset(page, "datasets/ioi.jsonl");
  const row = (i: number) => page.getByRole("button", { name: `Select prompt ${i}`, exact: true });
  const strip = page.getByText(/^Prompt \d+ of 150$/);

  // A hundred rows a page, and every row reachable.
  await expect(page.getByText("Rows 1–100 of 150").first()).toBeVisible();
  await expect(row(99)).toBeVisible();
  await expect(row(100)).toHaveCount(0);
  await page.getByRole("button", { name: "Next page" }).last().click();
  await expect(page.getByText("Rows 101–150 of 150").first()).toBeVisible();
  await expect(row(149)).toBeVisible();
  await expect(row(0)).toHaveCount(0);
  await page.getByRole("button", { name: "Previous page" }).first().click();
  await expect(row(0)).toBeVisible();

  // The keyboard chooses a row as a click does: the arrows, Page Up and Down, Home and End move
  // and choose; Enter and Space choose the row in focus.
  await row(0).focus();
  await page.keyboard.press("ArrowDown");
  await expect(row(1)).toBeFocused();
  await expect(row(1)).toHaveAttribute("aria-pressed", "true");
  await expect(strip).toHaveText("Prompt 1 of 150");
  await page.keyboard.press("End");
  await expect(row(149)).toBeFocused();
  await expect(page.getByText("Rows 101–150 of 150").first()).toBeVisible();
  await page.keyboard.press("Home");
  await expect(row(0)).toBeFocused();
  await page.keyboard.press("PageDown");
  await expect(row(100)).toBeFocused();
  await page.keyboard.press("ArrowUp");
  await expect(row(99)).toBeFocused();
  await expect(strip).toHaveText("Prompt 99 of 150");
  await row(5).focus();
  await page.keyboard.press("Enter");
  await expect(row(5)).toHaveAttribute("aria-pressed", "true");
  await expect(row(99)).toHaveAttribute("aria-pressed", "false");
  await row(6).focus();
  await page.keyboard.press("Space");
  await expect(strip).toHaveText("Prompt 6 of 150");
  // A row's number is the table's one stop for the Tab key: the chosen row's.
  await expect(row(6)).toHaveAttribute("tabindex", "0");
  await expect(row(5)).toHaveAttribute("tabindex", "-1");
  // A prompt chosen elsewhere brings its page with it.
  await row(99).click();
  await page.getByRole("button", { name: "Next prompt (])" }).click();
  await expect(strip).toHaveText("Prompt 100 of 150");
  await expect(row(100)).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByText("Rows 101–150 of 150").first()).toBeVisible();
  await page.getByRole("button", { name: "Previous page" }).first().click();
  await expect(row(0)).toBeVisible();

  // A search looks through every row, not just the page shown.
  const name = (original.records[130].answer as string).trim();
  const found = matchRows(original.records, name);
  expect(found.length).toBeGreaterThan(0);
  expect(found.length).toBeLessThan(150);
  const search = page.getByRole("searchbox", { name: "Search the prompt pairs" });
  await search.fill(name.toUpperCase());
  await expect(page.getByText(`${found.length} of 150 rows match`)).toBeVisible();
  await expect(page.getByText(`Matches 1–${Math.min(100, found.length)} of ${found.length}`).first()).toBeVisible();
  await expect(row(found[0])).toBeVisible();
  await expect(row(found[Math.min(100, found.length) - 1])).toBeVisible();
  const missed = original.records.findIndex((_, i) => !found.includes(i));
  await expect(row(missed)).toHaveCount(0);
  await search.fill("no prompt says this");
  await expect(page.getByText("No rows match")).toBeVisible();
  await expect(page.getByText("No prompt pair contains “no prompt says this”.")).toBeVisible();
  await page.getByRole("button", { name: "Show every row", exact: true }).click();
  await expect(search).toHaveValue("");
  await expect(search).toBeFocused();
  await expect(page.getByText("Rows 1–100 of 150").first()).toBeVisible();
  await search.fill(name);
  await page.getByRole("button", { name: "Clear the search", exact: true }).click();
  await expect(search).toHaveValue("");
  await search.fill(name);
  await expect(page.getByText(`${found.length} of 150 rows match`)).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(search).toHaveValue("");
  await expect(page.getByText("Rows 1–100 of 150").first()).toBeVisible();

  // Edit a row on the second page: its answer becomes a set of two tokens.
  await page.getByRole("button", { name: "Next page" }).first().click();
  const before: PromptRecord = original.records[120];
  const other = original.records.map((r) => r.answer as string).find((a) => a !== before.answer && a !== before.distractor) as string;
  await page.getByRole("button", { name: "Edit prompt 120", exact: true }).click();
  const editor = page.getByRole("dialog", { name: "Edit prompt 120" });
  await expect(editor.getByRole("textbox", { name: "Clean prompt" })).toBeFocused();
  await expect(editor.getByRole("textbox", { name: "Clean prompt" })).toHaveValue(before.clean);
  await editor.getByRole("radiogroup", { name: "Answer: one answer or a set" }).getByRole("radio", { name: "Set of tokens" }).click();
  const answer = editor.getByRole("textbox", { name: "Answer", exact: true });
  await expect(answer).toHaveValue(before.answer as string);
  await answer.fill(`${before.answer}\n${before.answer}`);
  await expect(editor.getByRole("alert").filter({ hasText: "twice" })).toBeVisible();
  await expect(editor.getByRole("button", { name: "Apply", exact: true })).toBeDisabled();
  await answer.fill(`${before.answer}\n${other}`);
  await editor.getByRole("textbox", { name: "ID", exact: true }).fill("checked-by-hand");
  await editor.getByRole("button", { name: "Apply", exact: true }).click();
  await expect(editor).not.toBeVisible();
  const edited = page.getByRole("row").filter({ has: row(120) });
  await expect(edited).toContainText("edited");
  await expect(edited).toContainText("any of 2");
  const bar = page.getByRole("region", { name: "Unsaved edits" });
  await expect(bar).toContainText("1 prompt pair edited. ioi.jsonl stays as it is");
  const datasets = page.getByRole("group", { name: "Datasets" });
  await expect(datasets.getByRole("button", { name: /^ioi\.jsonl/ })).toContainText("150 prompts · unsaved edits");

  // Remove the next row; it stays in sight, to restore.
  await page.getByRole("button", { name: "Remove prompt 121", exact: true }).click();
  await expect(page.getByRole("row").filter({ has: row(121) })).toContainText("removed");
  await expect(page.getByRole("button", { name: "Restore prompt 121", exact: true })).toBeVisible();
  await expect(bar).toContainText("1 prompt pair edited and 1 removed.");

  // Edits stay while another view is open.
  await page.getByRole("button", { name: "Baseline", exact: true }).click();
  await page.getByRole("button", { name: "Prompts", exact: true }).click();
  await expect(bar).toContainText("1 prompt pair edited and 1 removed.");

  // Save them as a new dataset, which is chosen; the file edited is as it was.
  await expect(bar.getByRole("textbox", { name: "New dataset name" })).toHaveValue("ioi-edited");
  await bar.getByRole("button", { name: "Save as a new dataset", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "149 prompt pairs in ioi-edited.jsonl" })).toBeVisible();
  await expect(datasets.getByRole("button", { name: /^ioi-edited\.jsonl/ })).toHaveAttribute("aria-pressed", "true");
  await expect(datasets.getByRole("button", { name: /^ioi\.jsonl/ })).toHaveText("ioi.jsonl150 prompts");
  await expect(bar).toHaveCount(0);
  await expect(page.getByRole("status").filter({ hasText: "Saved datasets/ioi-edited.jsonl." })).toBeVisible();
  const saved = await readDataset(page, "datasets/ioi-edited.jsonl");
  expect(saved.n).toBe(149);
  expect(saved.records[120]).toEqual({ ...before, answer: [before.answer, other], id: "checked-by-hand" });
  expect(saved.records[121]).toEqual(original.records[122]);
  expect(saved.records.slice(0, 120)).toEqual(original.records.slice(0, 120));
  expect((await readDataset(page, "datasets/ioi.jsonl")).sha256).toBe(original.sha256);

  // Unsaved edits can be discarded, after a question whose safe answer keeps them.
  await page.getByRole("button", { name: "Remove prompt 0", exact: true }).click();
  await page.getByRole("button", { name: "Discard edits", exact: true }).click();
  const question = page.getByRole("alertdialog", { name: "Discard the unsaved edits?" });
  await expect(question).toContainText("1 prompt pair removed: the edits will be lost. ioi-edited.jsonl stays as it is.");
  await expect(question.getByRole("button", { name: "Keep the edits" })).toBeFocused();
  await question.getByRole("button", { name: "Discard edits", exact: true }).click();
  await expect(page.getByRole("region", { name: "Unsaved edits" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Remove prompt 0", exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

test("prompts the model can't use are removed in one step, and the rest saved as a new dataset", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const state = await (await page.request.get("/api/state")).json();
  const created = await page.request.post("/api/projects", { data: { name: `Unusable prompts ${Date.now()}`, parent: state.projects_parent }, headers: ORIGIN });
  expect(created.ok()).toBe(true);
  await expect(page.getByText("No prompts yet.", { exact: false })).toBeVisible();
  await page.getByRole("button", { name: "Experiment", exact: true }).click();
  await page.getByRole("button", { name: "Prompts", exact: true }).click();
  await page.getByRole("radiogroup", { name: "Add prompts" }).getByRole("radio", { name: "Import JSONL" }).click();
  const pair = (a: string, b: string, extra = "") => ({
    clean: `When ${a} and ${b} went to the store, ${b} gave a drink to`,
    corrupt: `When ${a} and ${b} went to the store${extra}, ${a} gave a drink to`,
    answer: ` ${a}`,
    distractor: ` ${b}`,
  });
  // The last corrupt prompt is a token longer than its clean prompt.
  const lines = [pair("Mary", "John"), pair("John", "Mary"), pair("Mary", "John", " store")];
  await page.getByRole("textbox", { name: "Paste lines, or choose a file" }).fill(lines.map((l) => JSON.stringify(l)).join("\n"));
  await page.getByRole("textbox", { name: "File name" }).fill("mixed");
  await page.getByRole("button", { name: "Import", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "3 prompt pairs in mixed.jsonl" })).toBeVisible();
  const unusable = page.getByRole("alert").filter({ hasText: "1 prompt can't be used with this model" });
  await expect(unusable).toBeVisible();
  await unusable.getByRole("button", { name: "Remove this prompt", exact: true }).click();
  await expect(page.getByRole("button", { name: "Restore prompt 2", exact: true })).toBeVisible();
  await expect(unusable.getByRole("button", { name: "Remove this prompt", exact: true })).toBeDisabled();
  const bar = page.getByRole("region", { name: "Unsaved edits" });
  await expect(bar).toContainText("1 prompt pair removed. mixed.jsonl stays as it is");
  await bar.getByRole("button", { name: "Save as a new dataset", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "2 prompt pairs in mixed-edited.jsonl" })).toBeVisible();
  await expect(page.getByText("can't be used with this model")).toHaveCount(0);
  expect(errors).toEqual([]);
});
