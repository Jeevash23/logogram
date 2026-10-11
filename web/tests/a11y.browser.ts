import { test, expect, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

// Axe checks each workspace's main views for accessibility violations that block or seriously
// hinder people using assistive technology. It runs inside the page; the app's content security
// policy forbids injected scripts, so this test's browser context sets the policy aside (the app
// itself is unchanged).

test.use({ bypassCSP: true });

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

/** Serious and critical violations: what keeps someone from using the view. */
async function expectAccessible(page: Page, view: string) {
  const found = (await violations(page)).filter((v) => v.impact === "serious" || v.impact === "critical");
  expect(found, `${view}: ${JSON.stringify(found, null, 2)}`).toEqual([]);
}

test("the main views have no serious accessibility violations", async ({ page }) => {
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
  await expectAccessible(page, "Explore, model");
  await page.getByRole("button", { name: "Experiment", exact: true }).click();
  for (const view of ["Prompts", "Baseline", "Configure", "Spec"]) {
    await page.getByRole("button", { name: view, exact: true }).click();
    await expect(page.getByRole("button", { name: view, exact: true })).toBeVisible();
    await expectAccessible(page, `Experiment, ${view}`);
  }
  await page.getByRole("button", { name: "Evidence", exact: true }).click();
  await expectAccessible(page, "Evidence, results");
});
