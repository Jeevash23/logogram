import { test, expect, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";

// The examples open as projects of their own, and a run's notes are read beside its result. Each
// test reopens the project that was open before it, so the other browser checks see the fixture
// as they expect.

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

interface Example {
  id: string;
  name: string;
  question: string;
  expect: string;
  source: string;
  copied: boolean;
  spec: { name: string; notes: string };
}

const examplesOf = async (page: Page): Promise<Example[]> => (await page.request.get("/api/examples")).json();

test("the gallery offers each example, and opens it as a copy of its own, once", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const examples = await examplesOf(page);
  expect(examples.length).toBeGreaterThanOrEqual(4);
  expect(examples.filter((e) => e.copied)).toEqual([]);
  await page.getByRole("button", { name: "All projects" }).click();
  const gallery = page.getByRole("list", { name: "Examples" });
  await expect(gallery.getByRole("listitem")).toHaveCount(examples.length);
  for (const example of examples) {
    const item = gallery.getByRole("listitem").filter({ has: page.getByRole("heading", { level: 3, name: example.name, exact: true }) });
    // What it asks, on which model and by which method, and the result to expect.
    await expect(item).toContainText(example.question);
    await expect(item).toContainText("GPT-2 small");
    await expect(item).toContainText(`Expect ${example.expect}`);
    await expect(item).toContainText(example.source);
    await item.getByRole("button", { name: `Open ${example.name}`, exact: true }).click();
    await expect.poll(async () => (await (await page.request.get("/api/state")).json()).project?.name).toBe(example.name);
    // The copy holds the example's experiment, ready to run, with its notes.
    await page.getByRole("button", { name: "Experiment", exact: true }).click();
    await page.getByRole("button", { name: "Configure", exact: true }).click();
    await expect(page.getByRole("textbox", { name: "Name" })).toHaveValue(example.spec.name);
    await expect(page.getByRole("textbox", { name: "Notes" })).toHaveValue(example.spec.notes);
    await page.getByRole("button", { name: "All projects" }).click();
    // Opened again, it is the copy as it was left: never replaced.
    await expect(item.getByRole("button", { name: `Open your copy of ${example.name}`, exact: true })).toBeVisible();
  }
  const copies = await examplesOf(page);
  expect(copies.every((e) => e.copied)).toBe(true);
  expect(new Set(copies.map((e) => e.id)).size).toBe(copies.length);
  // The first screen offers the same examples.
  await page.getByRole("button", { name: "System check" }).click();
  await expect(page.getByRole("list", { name: "Examples" }).getByRole("listitem")).toHaveCount(examples.length);
  expect(errors).toEqual([]);
});

test("a run's notes are read beside its result, and in its methods", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  // The fixture's run has no notes: nothing is shown for them.
  await page.getByRole("button", { name: "Evidence", exact: true }).click();
  await page.getByRole("button", { name: "Results", exact: true }).click();
  await expect(page.getByRole("main").getByRole("heading", { level: 1, name: "Without BOS" })).toBeVisible();
  await expect(page.getByRole("main").getByText(/^Notes\b/)).toHaveCount(0);

  const state = await (await page.request.get("/api/state")).json();
  const created = await page.request.post("/api/projects", { data: { name: `Notes ${Date.now()}`, parent: state.projects_parent }, headers: ORIGIN });
  expect(created.ok()).toBe(true);
  await expect(page.getByText("No prompts yet.", { exact: false })).toBeVisible();
  await page.getByRole("button", { name: "Experiment", exact: true }).click();
  await page.getByRole("button", { name: "Prompts", exact: true }).click();
  await page.getByRole("spinbutton", { name: "Prompts", exact: true }).fill("6");
  await page.getByRole("button", { name: "Generate 6 prompts", exact: true }).click();
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  const expectation = [
    "Expect the heads of the last layer to restore most of the answer, as name movers do in GPT-2 small.",
    "If the first layer matters more, the tiny model reads the names differently.",
    "Either way, the strongest head should restore less than all of it, and the interval should exclude zero.",
    "This note is long enough to be folded.",
  ].join("\n");
  await page.getByRole("textbox", { name: "Notes" }).fill(expectation);
  await page.getByRole("button", { name: "Run", exact: true }).click();
  const notes = page.getByRole("main").locator("p").filter({ hasText: /^Notes Expect the heads/ });
  await expect(notes).toBeVisible({ timeout: 60_000 });
  await expect(page.getByRole("button", { name: "Export", exact: true })).toBeVisible({ timeout: 60_000 });
  // Folded to a few lines, then shown in full.
  const toggle = page.getByRole("button", { name: "Show all notes", exact: true });
  await expect(toggle).toHaveAttribute("aria-expanded", "false");
  const folded = (await notes.boundingBox())?.height ?? 0;
  await toggle.click();
  await expect(page.getByRole("button", { name: "Show less", exact: true })).toHaveAttribute("aria-expanded", "true");
  expect((await notes.boundingBox())?.height ?? 0).toBeGreaterThan(folded);
  await expect(notes).toHaveText(`Notes ${expectation}`, { useInnerText: true });
  // The methods name them too.
  await page.getByRole("button", { name: "Export", exact: true }).click();
  const download = page.waitForEvent("download");
  await page.getByRole("menuitem", { name: "Download methods", exact: true }).click();
  const path = await (await download).path();
  expect(readFileSync(path, "utf8")).toContain(`Notes saved with the spec: ${expectation}`);
  expect(errors).toEqual([]);
});
