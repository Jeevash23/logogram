import { expect, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

// Axe, run inside the page as a11y.browser.ts runs it, for checks that scan views of their own.
// The app's content security policy forbids injected scripts, so a check that uses this sets the
// policy aside for its browser context: test.use({ bypassCSP: true, reducedMotion: "reduce" }).

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

async function violations(page: Page): Promise<Violation[]> {
  await page.evaluate(AXE);
  return page.evaluate(async () => {
    // @ts-expect-error axe is defined by the script above
    const result = await window.axe.run(document, { resultTypes: ["violations"] });
    return result.violations.map((v: { id: string; impact: string | null; help: string; nodes: AxeNode[] }) => ({
      id: v.id,
      impact: v.impact,
      help: v.help,
      targets: v.nodes.slice(0, 3).map((n) => {
        const colors = n.any.map((check) => check.data).find((data) => data?.fgColor);
        return colors ? `${n.target.join(" ")}: ${colors.fgColor} on ${colors.bgColor}, ${colors.contrastRatio}:1` : n.target.join(" ");
      }),
    }));
  });
}

/** No violations once the view settles (three seconds to come clean, as in a11y.browser.ts). */
export async function expectAccessible(page: Page, view: string) {
  await expect.poll(() => violations(page), { message: `${view} has accessibility violations`, timeout: 3_000 }).toEqual([]);
}
