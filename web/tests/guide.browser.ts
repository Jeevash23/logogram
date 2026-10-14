import { test, expect, type Locator, type Page } from "@playwright/test";

import { GUIDE_PAGES } from "../src/lib/guide";
import { guideUrl } from "../src/lib/help";
import { FIGURE_CAPTION, START } from "../src/lib/start";
import { expectAccessible } from "./axe";

// Help → Guide, from the keyboard alone: its pages are tabs the arrow keys move between, each
// with buttons that open the views it talks about (closing the guide and putting the keyboard in
// the view) and links to the documentation on GitHub, which open in a new tab and are never
// followed here. It opens on Start here, for someone new, the first time; Learn the basics in the
// command palette opens that page from any screen. Axe checks every page, in the theme the run is
// in, and in an 800-pixel window the guide fits. The walkthrough it can start, and Start here on
// the first screens, have checks of their own (walkthrough.browser.ts).

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
  await expect(dialog).toContainText("How Logogram works, in short. These pages are part of the app and work without a network");
  const tabs = dialog.getByRole("tablist", { name: "Pages of the guide" });
  await expect(tabs.getByRole("tab")).toHaveText(GUIDE_PAGES.map((p) => p.title));
  const tab = (title: string) => tabs.getByRole("tab", { name: title, exact: true });
  const panel = dialog.getByRole("tabpanel");

  // The first time, it opens on Start here. The open page's tab is the one the keyboard reaches;
  // the arrow keys move between pages.
  await tabTo(page, tab("Start here"));
  await expect(tab("Start here")).toHaveAttribute("aria-selected", "true");
  await expect(panel).toHaveAccessibleName("Start here");
  for (const [i, shown] of GUIDE_PAGES.entries()) {
    if (i > 0) await page.keyboard.press("ArrowRight");
    await expect(tab(shown.title)).toBeFocused();
    await expect(tab(shown.title)).toHaveAttribute("aria-selected", "true");
    await expect(panel).toHaveAccessibleName(shown.title);
    if (shown.id === "workspaces") {
      for (const word of ["Explore", "Experiment", "Evidence"]) await expect(panel.getByRole("term").filter({ hasText: word })).toHaveCount(1);
    }
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
  await expect(tab("Start here")).toBeFocused();
  await page.keyboard.press("ArrowLeft");
  await expect(tab("Keyboard")).toHaveAttribute("aria-selected", "true");
  await page.keyboard.press("Home");
  await expect(tab("Start here")).toHaveAttribute("aria-selected", "true");
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
  // From the command palette, at the page about the workspaces, next to Start here.
  await page.keyboard.press("ControlOrMeta+k");
  await page.getByLabel("Command", { exact: true }).fill("open the guide");
  await page.keyboard.press("Enter");
  const dialog = guide(page);
  await expect(dialog).toBeVisible();
  const tab = (title: string) => dialog.getByRole("tab", { name: title, exact: true });
  await tabTo(page, tab("Start here"));
  await page.keyboard.press("ArrowRight");
  await expect(tab("Workspaces")).toHaveAttribute("aria-selected", "true");
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

test("Start here explains the basics, and its buttons open Your text and the examples, with the keyboard there", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openFromHelp(page);
  const dialog = guide(page);
  const panel = dialog.getByRole("tabpanel", { name: "Start here" });
  await expect(panel).toBeVisible();
  // Its sections, the figure of the three runs, named by its caption, the parts of a model and the
  // first hour, step by step.
  for (const section of START) if (section.title) await expect(panel.getByRole("heading", { level: 3, name: section.title, exact: true })).toBeVisible();
  await expect(panel.getByRole("figure", { name: FIGURE_CAPTION })).toBeVisible();
  for (const term of ["Token", "Layer", "Residual stream", "Attention head", "MLP", "Site"]) await expect(panel.getByRole("term").filter({ hasText: term }).first()).toBeVisible();
  const hour = panel.getByRole("list", { name: "Your first hour" });
  await expect(hour.getByRole("listitem")).toHaveCount(6);
  await expect(hour.getByRole("listitem").first().getByRole("button", { name: "Walk me through the example", exact: true })).toBeVisible();
  await expect(panel).toContainText("When Mary and John went to the store, John gave a drink to");
  await expectAccessible(page, "Guide, Start here");

  // Open Your text closes the guide, shows the view and puts the keyboard on it.
  await tabTo(page, panel.getByRole("button", { name: "Open Your text", exact: true }), 24);
  await page.keyboard.press("Enter");
  await expect(dialog).toBeHidden();
  await expect(page.getByRole("button", { name: "Your text", exact: true })).toHaveAttribute("aria-current", "page");
  await expect(page.locator("#main")).toBeFocused();

  // Go to the examples leaves the workbench for All projects, with the keyboard on the examples.
  await openFromHelp(page);
  await expect(guide(page).getByRole("tab", { name: "Start here", exact: true })).toHaveAttribute("aria-selected", "true");
  await tabTo(page, guide(page).getByRole("button", { name: "Go to the examples", exact: true }), 24);
  await page.keyboard.press("Enter");
  await expect(guide(page)).toBeHidden();
  await expect(page.getByRole("list", { name: "Examples" })).toBeVisible();
  await expect(page.getByRole("heading", { level: 2, name: "Examples", exact: true })).toBeFocused();
  expect(errors).toEqual([]);
});

test("Learn the basics, in the command palette, opens the guide at Start here from any screen", async ({ page }) => {
  // The guide opens where it was left, unless the command says where.
  await openFromHelp(page);
  const dialog = guide(page);
  const tab = (title: string) => dialog.getByRole("tab", { name: title, exact: true });
  await tabTo(page, tab("Start here"));
  await page.keyboard.press("End");
  await expect(tab("Keyboard")).toHaveAttribute("aria-selected", "true");
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  const learn = async () => {
    await page.keyboard.press("ControlOrMeta+k");
    await page.getByLabel("Command", { exact: true }).fill("learn the basics");
    await expect(page.getByRole("option", { name: "Learn the basics", exact: true })).toHaveAttribute("aria-selected", "true");
    await page.keyboard.press("Enter");
    await expect(dialog).toBeVisible();
    await expect(tab("Start here")).toHaveAttribute("aria-selected", "true");
    await expect(dialog.getByRole("tabpanel")).toHaveAccessibleName("Start here");
  };
  await learn();
  await tabTo(page, tab("Start here"));
  await page.keyboard.press("End");
  await expect(tab("Keyboard")).toHaveAttribute("aria-selected", "true");
  await page.keyboard.press("Escape");
  // Outside the workbench too, where only the walkthrough's button is offered.
  await page.getByRole("button", { name: "All projects" }).focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("list", { name: "Examples" })).toBeVisible();
  await learn();
  const panel = dialog.getByRole("tabpanel");
  await expect(panel.getByRole("button", { name: "Walk me through the example", exact: true })).toBeVisible();
  for (const name of ["Open Your text", "Go to the examples", "Open Prompts"]) await expect(panel.getByRole("button", { name, exact: true })).toHaveCount(0);
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
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
