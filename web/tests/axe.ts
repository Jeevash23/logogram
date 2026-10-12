// Axe, run inside the page, for checks that scan the states they reach (a11y.browser.ts scans
// every view the same way). The test using it sets `bypassCSP` (the app's content security policy
// forbids injected scripts) and reduced motion, so axe reads each state as it settles.

import { expect, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

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

/** No violations once the page settles (it gets three seconds to come clean). */
export async function expectAccessible(page: Page, what: string): Promise<void> {
  await expect.poll(() => violations(page), { message: `${what} has accessibility violations`, timeout: 3_000 }).toEqual([]);
}
