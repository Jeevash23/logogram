import { test, expect, type Locator, type Page } from "@playwright/test";

import { GUIDE_PAGES } from "../src/lib/guide";
import { guideUrl } from "../src/lib/help";
import { expectAccessible } from "./axe";

// Help → Guide, from the keyboard alone: its pages are tabs the arrow keys move between, each
// with buttons that open the views it talks about (closing the guide and putting the keyboard in
// the view) and links to the documentation on GitHub, which open in a new tab and are never
// followed here. Axe checks every page, in the theme the run is in, and in an 800-pixel window the
// guide fits. The walkthrough it can start has checks of its own (walkthrough.browser.ts).

test.use({ bypassCSP: true, reducedMotion: "reduce" });

test.beforeEach(async ({ page }) => {
  await page.route(/^https?:\/\/(?!127\.0\.0\.1[:/])/, (route) => route.abort());
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
});

function helpButton(page: Page): Locator {
  return page.getByRole("banner").getByRole("button", { name: "Help", exact: true });
}

function guide(page: Page): Locator {
  return page.getByRole("dialog", { name: "Guide", exact: true });
}

/** Tab, from wherever the keyboard is, until ``target`` has it. */
async function tabTo(page: Page, target: Locator, max = 16): Promise<void> {
  for (let i = 0; i < max; i++) {
    if (await target.evaluate((el) => el === document.activeElement)) return;
    await page.keyboard.press("Tab");
  }
  await expect(target).toBeFocused();
}

/** Open the guide from the Help menu, with the keyboard. */
async function openFromHelp(page: Page): Promise<void> {
  await helpButton(page).focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("menuitem", { name: "Guide", exact: true })).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(guide(page)).toBeVisible();
}

test("the guide opens from Help, and its pages are tabs that work from the keyboard", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openFromHelp(page);
  const dialog = guide(page);
  await expect(dialog).toContainText("How Logogram works, in short.");
  const tabs = dialog.getByRole("tablist", { name: "Pages of the guide" });
  await expect(tabs.getByRole("tab")).toHaveText(GUIDE_PAGES.map((p) => p.title));
  const tab = (title: string) => tabs.getByRole("tab", { name: title, exact: true });
  const panel = dialog.getByRole("tabpanel");

  // The open page's tab is the one the keyboard reaches; the arrow keys move between pages.
  await tabTo(page, tab("Workspaces"));
  await expect(tab("Workspaces")).toHaveAttribute("aria-selected", "true");
  await expect(panel).toHaveAccessibleName("Workspaces");
  for (const word of ["Explore", "Experiment", "Evidence"]) await expect(panel.getByRole("term").filter({ hasText: word })).toHaveCount(1);
  for (const [i, shown] of GUIDE_PAGES.entries()) {
    if (i > 0) await page.keyboard.press("ArrowRight");
    await expect(tab(shown.title)).toBeFocused();
    await expect(tab(shown.title)).toHaveAttribute("aria-selected", "true");
    await expect(panel).toHaveAccessibleName(shown.title);
    // Only the open page's tab is in the tab order.
    await expect(tabs.locator("[role=tab][tabindex='0']")).toHaveCount(1);
    // Its links to the documentation open GitHub in a new tab.
    const links = panel.getByRole("list", { name: "In the documentation" }).getByRole("link");
    await expect(links).toHaveCount(shown.docs.length);
    for (const doc of shown.docs) {
      const link = links.filter({ hasText: doc.title });
      await expect(link).toHaveAccessibleName(`${doc.title} (opens in a new tab)`);
      await expect(link).toHaveAttribute("href", guideUrl(doc));
      await expect(link).toHaveAttribute("target", "_blank");
      await expect(link).toHaveAttribute("rel", "noopener noreferrer");
    }
    await expectAccessible(page, `Guide, ${shown.title}`);
  }
  await expect(panel).toContainText("Show every keyboard shortcut.");
  // The arrows wrap around; Home and End go to the first and last pages.
  await page.keyboard.press("ArrowRight");
  await expect(tab("Workspaces")).toBeFocused();
  await page.keyboard.press("ArrowLeft");
  await expect(tab("Keyboard")).toHaveAttribute("aria-selected", "true");
  await page.keyboard.press("Home");
  await expect(tab("Workspaces")).toHaveAttribute("aria-selected", "true");
  await page.keyboard.press("End");
  await expect(tab("Keyboard")).toHaveAttribute("aria-selected", "true");

  // Show every shortcut swaps the guide for the list of shortcuts, which gives the keyboard back to
  // Help when it closes.
  await tabTo(page, panel.getByRole("button", { name: "Show every shortcut" }));
  await page.keyboard.press("Enter");
  await expect(dialog).toBeHidden();
  const shortcuts = page.getByRole("dialog", { name: "Keyboard shortcuts" });
  await expect(shortcuts).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(shortcuts).toBeHidden();
  await expect(helpButton(page)).toBeFocused();

  // Escape closes the guide, and the keyboard goes back to Help. It opens again where it was left.
  await openFromHelp(page);
  await expect(tab("Keyboard")).toHaveAttribute("aria-selected", "true");
  await page.keyboard.press("Escape");
  await expect(guide(page)).toBeHidden();
  await expect(helpButton(page)).toBeFocused();
  expect(errors).toEqual([]);
});

test("a view the guide names opens from it, with the keyboard on the view", async ({ page }) => {
  // From the command palette, at the page about the workspaces.
  await page.keyboard.press("ControlOrMeta+k");
  await page.getByLabel("Command", { exact: true }).fill("open the guide");
  await page.keyboard.press("Enter");
  const dialog = guide(page);
  await expect(dialog).toBeVisible();
  const tab = (title: string) => dialog.getByRole("tab", { name: title, exact: true });
  await tabTo(page, tab("Workspaces"));
  await page.keyboard.press("Home");
  await tabTo(page, dialog.getByRole("button", { name: "Open Evidence", exact: true }));
  await page.keyboard.press("Enter");
  await expect(dialog).toBeHidden();
  await expect(page.getByRole("button", { name: "Results", exact: true })).toHaveAttribute("aria-current", "page");
  await expect(page.locator("#main")).toBeFocused();

  // Numbers and intervals opens the experiment form.
  await openFromHelp(page);
  await tabTo(page, tab("Workspaces"));
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("ArrowRight");
  await expect(tab("Numbers and intervals")).toHaveAttribute("aria-selected", "true");
  await expect(dialog.getByRole("tabpanel")).toContainText("A percentile bootstrap interval over the prompts");
  await tabTo(page, dialog.getByRole("button", { name: "Open the experiment form", exact: true }));
  await page.keyboard.press("Enter");
  await expect(dialog).toBeHidden();
  await expect(page.getByRole("button", { name: "Configure", exact: true })).toHaveAttribute("aria-current", "page");
  await expect(page.locator("#main")).toBeFocused();

  // Help still gives the focus back to itself when a dialog it opened closes as usual.
  await openFromHelp(page);
  await page.keyboard.press("Escape");
  await expect(helpButton(page)).toBeFocused();
});

test("in an 800-pixel window the guide fits, and every page can be read", async ({ page }) => {
  await page.setViewportSize({ width: 800, height: 700 });
  await openFromHelp(page);
  const dialog = guide(page);
  const box = await dialog.boundingBox();
  expect(box && box.x >= 0 && box.x + box.width <= 800 && box.y >= 0 && box.y + box.height <= 700).toBe(true);
  for (const p of GUIDE_PAGES) {
    const tab = dialog.getByRole("tab", { name: p.title, exact: true });
    await expect(tab).toBeInViewport({ ratio: 1 });
    await tab.click();
    // The page's last link can be scrolled to inside the guide, and the tabs stay in view.
    const last = dialog.getByRole("list", { name: "In the documentation" }).getByRole("link").last();
    await last.scrollIntoViewIfNeeded();
    await expect(last).toBeInViewport({ ratio: 1 });
    await expect(dialog.getByRole("tablist")).toBeInViewport({ ratio: 1 });
  }
  // Nothing runs off the side.
  expect(await dialog.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(true);
  await expectAccessible(page, "Guide, in an 800-pixel window");
});
