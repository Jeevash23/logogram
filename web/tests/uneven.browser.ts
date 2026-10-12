import { test, expect } from "@playwright/test";

import { expectAccessible } from "./axe";

// Prompt pairs of different token lengths: imported, shown with both prompts' labels in the table
// and the token strip, edited with the corrupt prompt's positions, and offered only the positions
// that are matched between them. The test works in a project of its own and reopens the project
// that was open before it, so the other browser checks see the fixture as they expect.

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

const TEMPLATE = "When {A} and {B} went to the store, {C} gave a drink to";

/** A template filled in, with each slot's span and the last word's. */
function fill(values: Record<string, string>): { text: string; spans: Record<string, [number, number]> } {
  let text = "";
  const spans: Record<string, [number, number]> = {};
  for (const part of TEMPLATE.split(/(\{[A-Z]\})/)) {
    const slot = /^\{([A-Z])\}$/.exec(part)?.[1];
    const value = slot ? values[slot] : part;
    if (slot) spans[slot] = [text.length, text.length + value.length];
    text += value;
  }
  return { text, spans };
}

/** An IOI pair whose second subject is `clean` and `corrupt` (one name, or two), labelled in both. */
function pair(clean: string, corrupt: string) {
  const labels = (filled: ReturnType<typeof fill>) => ({
    IO: filled.spans.A,
    S1: filled.spans.B,
    S2: filled.spans.C,
    end: [filled.text.length - 2, filled.text.length] as [number, number],
  });
  const a = fill({ A: "Mary", B: "John", C: clean });
  const b = fill({ A: "Mary", B: "John", C: corrupt });
  return { clean: a.text, corrupt: b.text, answer: " Mary", distractor: " John", positions: labels(a), corrupt_positions: labels(b) };
}

const PAIRS = [pair("John", "Mary"), pair("John", "Mary Bob"), pair("John Bob", "Mary"), pair("John", "Mary")];

test("pairs of different lengths show both prompts' labels, and are patched at labels or the last token", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const state = await (await page.request.get("/api/state")).json();
  const created = await page.request.post("/api/projects", { data: { name: `Uneven pairs ${Date.now()}`, parent: state.projects_parent }, headers: ORIGIN });
  expect(created.ok()).toBe(true);
  await page.getByRole("button", { name: "Experiment", exact: true }).click();
  await page.getByRole("button", { name: "Prompts", exact: true }).click();
  await page.getByRole("radiogroup", { name: "Add prompts" }).getByRole("radio", { name: "Import JSONL" }).click();
  await page.getByRole("textbox", { name: "Paste lines, or choose a file" }).fill(PAIRS.map((p) => JSON.stringify(p)).join("\n"));
  await page.getByRole("textbox", { name: "File name" }).fill("uneven");
  await page.getByRole("button", { name: "Import", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "4 prompt pairs in uneven.jsonl" })).toBeVisible();

  // The table: both prompts of a pair of different lengths, with their lengths and labels.
  const table = page.getByRole("table");
  const row = (i: number) => table.getByRole("row").filter({ has: page.getByRole("button", { name: `Select prompt ${i}`, exact: true }) });
  await expect(row(1)).toContainText("15 tokens");
  await expect(row(1)).toContainText("16 tokens");
  await expect(row(1)).toContainText("S2 “Mary Bob”");
  await expect(row(2)).toContainText("S2 “John Bob”");
  // Pairs of the same length read as they always did.
  await expect(row(0)).not.toContainText("tokens");
  await expectAccessible(page, "Prompts with pairs of different lengths");

  // The token strip lines the prompts up by their labels, each with its own positions.
  await page.getByRole("button", { name: "Select prompt 1", exact: true }).click();
  await expect(page.getByText("15 and 16 tokens, lined up by their labels and last tokens.", { exact: false })).toBeVisible();
  const grid = page.getByRole("group", { name: /lined up by their labels/ }).first();
  await expect(grid.getByRole("button", { name: "corrupt token 11: ·Bob", exact: true })).toBeVisible();
  await expect(grid.getByRole("button", { name: "clean: no token here", exact: true })).toBeDisabled();
  // The arrows move along each prompt's own tokens, skipping where it has none.
  await grid.getByRole("button", { name: "clean token 9: ,", exact: true }).focus();
  await page.keyboard.press("ArrowRight");
  await expect(grid.getByRole("button", { name: "clean token 10: ·John", exact: true })).toBeFocused();
  await page.keyboard.press("ArrowDown");
  await expect(grid.getByRole("button", { name: "corrupt token 11: ·Bob", exact: true })).toBeFocused();

  // The editor keeps the corrupt prompt's positions.
  await page.getByRole("button", { name: "Edit prompt 1", exact: true }).click();
  const editor = page.getByRole("dialog", { name: "Edit prompt 1" });
  await expect(editor.getByRole("checkbox", { name: /Label them in the corrupt prompt too/ })).toBeChecked();
  await expect(editor.getByRole("textbox", { name: "Position S2: start in the corrupt prompt" })).toHaveValue("38");
  await expect(editor.getByText("“Mary Bob”")).toBeVisible();
  await editor.getByRole("button", { name: "Cancel", exact: true }).click();

  // The form offers only the positions matched between the prompts of each pair.
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  const position = page.getByRole("radiogroup", { name: "Position" });
  await expect(position.getByRole("radio", { name: "All positions" })).toBeDisabled();
  await expect(position.getByRole("radio", { name: "Token index" })).toBeDisabled();
  await expect(page.getByText("Pairs 1 and 2 have prompts of different lengths: patching matches them at the last token or a labelled position").first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Not ready to run" })).toBeVisible();
  await expect(page.getByText("Pairs 1 and 2 have clean and corrupt prompts of different token lengths", { exact: false })).toBeVisible();
  await position.getByRole("radio", { name: "Last token" }).click();
  await expect(page.getByRole("heading", { name: "Ready to run" })).toBeVisible();
  await position.getByRole("radio", { name: "Named position" }).click();
  await expect(page.getByRole("heading", { name: "Ready to run" })).toBeVisible();
  await expectAccessible(page, "The experiment form with pairs of different lengths");
  // A set of sites starts empty, to fill with sites at labels or the last token; it can't keep its
  // sites and patch every other position.
  await page.getByRole("radiogroup", { name: "Sweep" }).getByRole("radio", { name: "Sets of sites" }).click();
  const set = page.getByRole("group", { name: "Set 1: Set 1" });
  await expect(set.getByRole("radiogroup", { name: "What set 1 does" }).getByRole("radio", { name: "Keep these, replace the rest" })).toBeDisabled();
  await set.getByRole("textbox", { name: /^Site to add to Set 1/ }).fill("L1 H2");
  const where = set.getByRole("combobox", { name: "Position of the site to add to Set 1" });
  await expect(where.getByRole("option", { name: "All positions" })).toHaveCount(0);
  await where.selectOption({ label: "Position S2" });
  await set.getByRole("button", { name: "Add site", exact: true }).click();
  await expect(set.getByText("L1 H2 @ S2")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Ready to run" })).toBeVisible();
  await page.getByRole("radiogroup", { name: "Sweep" }).getByRole("radio", { name: "Layer × head" }).click();
  // Zero ablation carries nothing between prompts: every position is offered again.
  await page.getByRole("radiogroup", { name: "Method" }).getByRole("radio", { name: /^Ablation/ }).click();
  await page.getByRole("radiogroup", { name: "Baseline" }).getByRole("radio", { name: /^Zero/ }).click();
  await expect(position.getByRole("radio", { name: "All positions" })).toBeEnabled();
  expect(errors).toEqual([]);
});
