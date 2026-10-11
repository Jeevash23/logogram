import { test, expect, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

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

async function violations(page: Page): Promise<Violation[]> {
  await page.evaluate(AXE);
  return page.evaluate(async () => {
    // @ts-expect-error axe is defined by the script above
    const result = await window.axe.run(document, { resultTypes: ["violations"] });
    return result.violations.map((v: { id: string; impact: string | null; help: string; nodes: { target: string[] }[] }) => ({
      id: v.id,
      impact: v.impact,
      help: v.help,
      targets: v.nodes.slice(0, 3).map((n) => n.target.join(" ")),
    }));
  });
}

async function expectAccessible(page: Page, view: string) {
  const found = await violations(page);
  expect(found, `${view}: ${JSON.stringify(found, null, 2)}`).toEqual([]);
}

async function open(page: Page, name: string) {
  await page.getByRole("button", { name, exact: true }).click();
  await expect(page.getByRole("button", { name, exact: true })).toHaveAttribute("aria-current", "page");
}

test("every view has no accessibility violations", async ({ page }) => {
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
  }
  await page.getByLabel("Features on each token").locator("button[aria-pressed]").first().click();
  await expect(page.getByText("Where it fires most")).toBeVisible();
  await expectAccessible(page, "Explore, a feature");
  await page.getByRole("button", { name: "Experiment", exact: true }).click();
  for (const view of ["Prompts", "Baseline", "Configure", "Spec"]) {
    await open(page, view);
    await expectAccessible(page, `Experiment, ${view}`);
  }
  await page.getByRole("button", { name: "Evidence", exact: true }).click();
  for (const view of ["Results", "Compare runs", "Research notes"]) {
    await open(page, view);
    await expectAccessible(page, `Evidence, ${view}`);
  }
});
