import { test, expect, type Page } from "@playwright/test";
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { fakeRuns, otherVersions, TRACEBACK } from "./fakeRuns";

// Axe checks every view of the workbench, the inspector and a feature's details for
// accessibility violations, from those that block people using assistive technology to headings
// out of order. It runs inside the page; the app's content security policy forbids injected
// scripts, so this test's browser context sets the policy aside (the app itself is unchanged).
// Reduced motion makes transitions instant, so axe reads each page as it settles rather than
// halfway through one (a button fading from disabled to enabled has, for a moment, white text on
// grey).

test.use({ bypassCSP: true, reducedMotion: "reduce" });

const AXE = readFileSync(fileURLToPath(new URL("../node_modules/axe-core/axe.min.js", import.meta.url)), "utf8");

interface Violation {
  id: string;
  impact: string | null;
  help: string;
  targets: string[];
}

interface AxeNode {
  target: string[];
  any: { data?: { fgColor?: string; bgColor?: string; contrastRatio?: number } | null }[];
}

/** The violations on the page, or only in the part of it that ``within`` (a CSS selector) picks. */
async function violations(page: Page, within?: string): Promise<Violation[]> {
  await page.evaluate(AXE);
  return page.evaluate(async (part) => {
    // @ts-expect-error axe is defined by the script above
    const result = await window.axe.run(part ? { include: [[part]] } : document, { resultTypes: ["violations"] });
    return result.violations.map((v: { id: string; impact: string | null; help: string; nodes: AxeNode[] }) => ({
      id: v.id,
      impact: v.impact,
      help: v.help,
      // A contrast failure names its colors, so a report says what to fix.
      targets: v.nodes.slice(0, 3).map((n) => {
        const colors = n.any.map((check) => check.data).find((data) => data?.fgColor);
        return colors ? `${n.target.join(" ")}: ${colors.fgColor} on ${colors.bgColor}, ${colors.contrastRatio}:1` : n.target.join(" ");
      }),
    }));
  }, within);
}

/** No violations once the view settles: a moment between two states (a reply arriving, a panel
 * filling in) can look different, so a view gets three seconds to come clean. */
async function expectAccessible(page: Page, view: string, within?: string) {
  await expect.poll(() => violations(page, within), { message: `${view} has accessibility violations`, timeout: 3_000 }).toEqual([]);
}

async function open(page: Page, name: string) {
  await page.getByRole("button", { name, exact: true }).click();
  await expect(page.getByRole("button", { name, exact: true })).toHaveAttribute("aria-current", "page");
}

test("every view has no accessibility violations", async ({ page }) => {
  // Every view in turn, each scanned: about 25 seconds on a busy machine, more than the default
  // 30 leaves room for.
  test.setTimeout(90_000);
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
  await expectAccessible(page, "Explore, model");
  await page.locator('canvas[aria-label*="Model map"]').focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("complementary", { name: "Inspector" }).getByRole("heading", { level: 2, name: "L0 H0" })).toBeVisible();
  await expectAccessible(page, "Explore, a component in the inspector");
  for (const view of ["Attention", "Head comparison", "Layer predictions", "Features"]) {
    await open(page, view);
    await expectAccessible(page, `Explore, ${view}`);
    if (view === "Attention") {
      const finder = page.getByRole("region", { name: "Find heads by what they attend to" });
      await finder.getByLabel("Tokens in each copy").fill("8");
      await finder.getByLabel("Sequences").fill("4");
      await finder.getByRole("button", { name: "Find heads", exact: true }).click();
      await expect(finder.getByRole("group", { name: "Induction scores of every head" })).toBeVisible();
      await expectAccessible(page, "Explore, heads found by what they attend to");
    }
  }
  await page.getByLabel("Features on each token").locator("button[aria-pressed]").first().click();
  await expect(page.getByText("Where it fires most")).toBeVisible();
  await expectAccessible(page, "Explore, a feature");
  await open(page, "Your text");
  await page.getByRole("button", { name: "Read", exact: true }).click();
  await expect(page.getByRole("group", { name: /^Tokens of the text/ })).toBeVisible();
  await expect(page.getByRole("group", { name: /^Attention of L0 H0 in your text/ })).toBeVisible();
  await expectAccessible(page, "Explore, your text");
  await page.getByRole("button", { name: "Experiment", exact: true }).click();
  for (const view of ["Prompts", "Baseline", "Configure", "Spec"]) {
    await open(page, view);
    await expectAccessible(page, `Experiment, ${view}`);
  }
  // The prompts wrapped in the model's chat template, the first shown as the model reads it.
  await open(page, "Configure");
  const template = page.getByRole("radiogroup", { name: "Chat template" });
  await template.getByRole("radio", { name: "The start of the reply" }).click();
  await page.getByRole("textbox", { name: "The person's message the reply answers" }).fill("Hello");
  await expect(page.getByRole("figure", { name: /reads for the first prompt/ }).locator("mark")).toBeVisible();
  await expectAccessible(page, "Experiment, Configure, prompts in the chat template");
  await template.getByRole("radio", { name: "None" }).click();
  // The model dialog, with a model folder about to be added.
  await page.getByRole("button", { name: /^tiny-gpt2 · / }).click();
  const models = page.getByRole("dialog", { name: "Load a model" });
  await expect(models.getByRole("radiogroup", { name: "Model" })).toBeVisible();
  await expectAccessible(page, "The model dialog");
  await models.getByRole("button", { name: "Add a model folder" }).click();
  await page.getByRole("dialog", { name: "Choose a model folder" }).getByRole("button", { name: "Choose this folder" }).click();
  await expect(models.getByRole("region", { name: "Add a model folder" })).toBeVisible();
  await expectAccessible(page, "The model dialog, adding a model folder");
  await models.getByRole("region", { name: "Add a model folder" }).getByRole("button", { name: "Cancel" }).click();
  await page.keyboard.press("Escape");
  await expect(models).toBeHidden();
  // A prompt pair being edited, and edits waiting to be saved as a new dataset.
  await open(page, "Prompts");
  await page.getByRole("button", { name: "Edit prompt 0", exact: true }).click();
  const editor = page.getByRole("dialog", { name: "Edit prompt 0" });
  await editor.getByRole("radiogroup", { name: "Answer: one answer or a set" }).getByRole("radio", { name: "Set of tokens" }).click();
  await expect(editor.getByText("How the loaded model tokenizes it")).toBeVisible();
  await expectAccessible(page, "Experiment, a prompt pair being edited");
  await editor.getByRole("button", { name: "Add a position", exact: true }).click();
  await expect(editor.getByRole("alert").filter({ hasText: "Name each position, or remove it." })).toBeVisible();
  await expectAccessible(page, "Experiment, a prompt pair that can't be applied");
  await editor.getByRole("button", { name: "Cancel", exact: true }).click();
  await page.getByRole("button", { name: "Remove prompt 1", exact: true }).click();
  await page.getByRole("searchbox", { name: "Search the prompt pairs" }).fill("went");
  await expect(page.getByRole("region", { name: "Unsaved edits" })).toBeVisible();
  await expectAccessible(page, "Experiment, unsaved edits and a search");
  await page.getByRole("button", { name: "Discard edits", exact: true }).click();
  const question = page.getByRole("alertdialog", { name: "Discard the unsaved edits?" });
  await expectAccessible(page, "Experiment, discarding edits");
  await question.getByRole("button", { name: "Discard edits", exact: true }).click();
  await expect(page.getByRole("region", { name: "Unsaved edits" })).toHaveCount(0);
  await page.getByRole("button", { name: "Evidence", exact: true }).click();
  for (const view of ["Results", "Compare runs", "Sweeps", "Research notes"]) {
    await open(page, view);
    await expectAccessible(page, `Evidence, ${view}`);
  }
  // Help: its menu, the problem report (with and without the hardware) and About.
  const help = page.getByRole("banner").getByRole("button", { name: "Help", exact: true });
  await help.click();
  await expect(page.getByRole("menu")).toBeVisible();
  await expectAccessible(page, "The Help menu");
  await page.getByRole("menuitem", { name: "Report a problem…" }).click();
  const report = page.getByRole("dialog", { name: "Report a problem" });
  await expect(report.getByRole("region", { name: "The report" })).toBeVisible();
  await expectAccessible(page, "Report a problem");
  await report.getByRole("checkbox", { name: "Include hardware details" }).check();
  await expect(report.getByText("Processor", { exact: true })).toBeVisible();
  await expectAccessible(page, "Report a problem, with hardware details");
  await page.keyboard.press("Escape");
  await expect(report).toBeHidden();
  await help.click();
  await page.getByRole("menuitem", { name: "About Logogram" }).click();
  await expect(page.getByRole("dialog", { name: "About Logogram" }).getByRole("heading", { name: "Updates" })).toBeVisible();
  await expectAccessible(page, "About Logogram");
  await page.keyboard.press("Escape");
  // The screens before a project, with the examples they offer.
  await page.getByRole("button", { name: "All projects" }).click();
  await expect(page.getByRole("list", { name: "Examples" })).toBeVisible();
  await expectAccessible(page, "Projects");
  await page.getByRole("button", { name: "System check" }).click();
  await expect(page.getByRole("list", { name: "Examples" })).toBeVisible();
  await expectAccessible(page, "System check");
});

test("runs that failed, run elsewhere or ran with other versions have no accessibility violations", async ({ page }) => {
  await fakeRuns(page, () => [
    { id: "fake-failed", name: "A sweep that failed", status: "failed", error: "Can't load the model.", traceback: TRACEBACK },
    { id: "fake-elsewhere", name: "A sweep from a notebook", status: "running", elsewhere: true },
  ]);
  await otherVersions(page, { torch: "0.0.1", numpy: "0.0.2" });
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Evidence", exact: true }).click();
  await open(page, "Results");
  await expect(page.getByRole("note").filter({ hasText: "A rerun here may differ in the last digits." })).toBeVisible();
  await expectAccessible(page, "Results, a run from other versions");
  const history = page.getByRole("dialog", { name: "Experiment history" });
  for (const [name, view] of [
    ["A sweep from a notebook", "a run in another process"],
    ["A sweep that failed", "a failed run"],
  ]) {
    await page.getByRole("button", { name: /^History/ }).click();
    await expect(history.getByRole("button", { name: new RegExp(`^${name}`) })).toBeVisible();
    await expectAccessible(page, `History, ${view}`);
    await history.getByRole("button", { name: new RegExp(`^${name}`) }).click();
    await expect(page.getByRole("main").getByRole("heading", { level: 1, name })).toBeVisible();
    await expectAccessible(page, `Results, ${view}`);
  }
  await page.getByText("Show details", { exact: true }).click();
  await expect(page.getByLabel("Traceback", { exact: true })).toBeVisible();
  await expectAccessible(page, "Results, a failed run's traceback");
});

test("the history's states have no accessibility violations", async ({ page }) => {
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
  // A run whose spec can't be read, beside the fixture's run while the test lasts.
  const state = await (await page.request.get("/api/state")).json();
  const broken = join(state.project.path, "experiments", "20000101-000000-broken");
  mkdirSync(broken, { recursive: true });
  writeFileSync(join(broken, "spec.json"), "{");
  try {
    await page.reload();
    // Opening the project settles on a view, which closes any drawer: open the history after it.
    await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
    await page.getByRole("button", { name: /^History/ }).click();
    const history = page.getByRole("dialog", { name: "Experiment history" });
    await expect(history.getByText("unreadable", { exact: true })).toBeVisible();
    await expectAccessible(page, "History");
    const search = history.getByRole("searchbox", { name: "Search runs" });
    await search.fill("without");
    await expect(history.getByText("1 run found.")).toBeVisible();
    await expectAccessible(page, "History, a search");
    await search.fill("");
    const run = history.locator("li[data-run]").filter({ hasText: "Without BOS" });
    // (A menu, like every menu in the app, floats outside the page's landmarks while it is open.)
    await run.getByRole("button", { name: "Actions for “Without BOS”" }).click();
    await page.getByRole("menuitem", { name: "Rename…" }).click();
    await expect(run.getByRole("textbox", { name: "Title of “Without BOS”" })).toBeFocused();
    await expectAccessible(page, "History, renaming a run");
    await page.keyboard.press("Escape");
    // Starred, and shown alone; then the star comes off again, as the fixture had it.
    await run.getByRole("button", { name: "Star “Without BOS”" }).click();
    const show = history.getByRole("radiogroup", { name: "Show" });
    await show.getByRole("radio", { name: /^Starred/ }).click();
    await expect(run.getByRole("button", { name: "Star “Without BOS”" })).toHaveAttribute("aria-pressed", "true");
    await expectAccessible(page, "History, starred runs");
    await run.getByRole("button", { name: "Star “Without BOS”" }).click();
    await expect(history.getByText("No starred runs.", { exact: false })).toBeVisible();
    await show.getByRole("radio", { name: /^All/ }).click();
    await run.getByRole("button", { name: "Actions for “Without BOS”" }).click();
    await page.getByRole("menuitem", { name: "Delete…" }).click();
    const question = page.getByRole("alertdialog", { name: "Delete “Without BOS”?" });
    await expect(question.getByRole("button", { name: "Keep it" })).toBeFocused();
    await expectAccessible(page, "History, deleting a run");
    await question.getByRole("button", { name: "Keep it" }).click();
    await expect(run).toBeVisible();
  } finally {
    rmSync(broken, { recursive: true, force: true });
  }
});
