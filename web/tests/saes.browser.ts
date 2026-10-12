import { test, expect } from "@playwright/test";

import { expectAccessible } from "./axe";

// Published SAEs by name: the catalogue of SAELens's registry in the Features view (the fixture
// server's small one for the tiny model), searched by release, hook, layer and width, with what
// Logogram reads of each, chosen with the keyboard and loaded by its repository and folder. The SAE
// it loads is the fixture's own (local/exact), served from a folder in place of Hugging Face, so
// the other checks see the SAE they expect. Scanned by axe in the theme the run uses.

test.use({ bypassCSP: true, reducedMotion: "reduce" });

test.beforeEach(async ({ page }) => {
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
});

test("published SAEs are found by name, chosen with the keyboard and loaded from their folder", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.getByRole("button", { name: "Explore", exact: true }).click();
  await page.getByRole("button", { name: "Features", exact: true }).click();
  await page.getByLabel("The loaded SAE").getByRole("button", { name: "Load another SAE", exact: true }).click();

  const list = page.getByRole("listbox", { name: "Published SAEs for tiny-gpt2" });
  const options = list.getByRole("option");
  await expect(page.getByText("5 in SAELens 0.0.test's registry", { exact: false })).toBeVisible();
  await expect(page.getByText("Logogram reads 3 of them exactly", { exact: false })).toBeVisible();
  // Those read exactly first, then the one checked when it loads, then the one not read.
  await expect(options).toHaveCount(5);
  await expect(options.nth(0)).toContainText("blocks.1.hook_resid_pre");
  await expect(options.nth(0)).toContainText("Reads exactly");
  await expect(options.nth(3)).toContainText("Checked on loading");
  await expect(options.nth(4)).toContainText("blocks.1.hook_z");
  await expect(options.nth(4)).toContainText("Not read");
  await expectAccessible(page, "Features, the published SAEs");

  // Searched by layer, width, hook and release.
  const search = page.getByRole("combobox", { name: "Search" });
  await search.fill("layer 1");
  await expect(options).toHaveCount(3);
  await search.fill("64k");
  await expect(options).toHaveCount(2);
  await expect(options.nth(0)).toContainText("Layer 0's MLP input → output, with a skip connection");
  await search.fill("hook_z");
  await expect(options).toHaveCount(1);
  // One Logogram doesn't read says why, and can't be loaded.
  const pickled = page.getByRole("group", { name: "The chosen SAE: tiny-attn-pickled, blocks.1.hook_z" });
  await expect(pickled.getByText("Logogram doesn't read this SAE", { exact: true })).toBeVisible();
  await expect(pickled.getByText(/PyTorch pickle/)).toBeVisible();
  await expect(pickled.getByRole("button", { name: "Load this SAE" })).toBeDisabled();
  await search.fill("nothing like this");
  await expect(page.getByText("No SAE for tiny-gpt2 matches “nothing like this”.", { exact: true })).toBeVisible();
  await expectAccessible(page, "Features, a search that matches nothing");
  await search.fill("tiny-res");
  await expect(options).toHaveCount(2);

  // The keyboard: the arrows move the choice, and Enter goes to its Load button.
  await expect(options.nth(0)).toHaveAttribute("aria-selected", "true");
  await search.press("ArrowDown");
  await expect(options.nth(1)).toHaveAttribute("aria-selected", "true");
  await expect(options.nth(0)).toHaveAttribute("aria-selected", "false");
  await expect(search).toHaveAttribute("aria-activedescendant", (await options.nth(1).getAttribute("id")) ?? "");
  await expect(page.getByRole("group", { name: "The chosen SAE: tiny-res, blocks.0.hook_resid_pre" })).toBeVisible();
  await search.press("ArrowUp");
  const chosen = page.getByRole("group", { name: "The chosen SAE: tiny-res, blocks.1.hook_resid_pre" });
  await expect(chosen.getByText("local/exact", { exact: false })).toBeVisible();
  await expect(chosen.getByText(/its files at 0123456 were read whole/)).toBeVisible();
  await expectAccessible(page, "Features, a published SAE chosen");

  // Down to an 800-pixel window, nothing runs off the side.
  await page.setViewportSize({ width: 800, height: 900 });
  await expect.poll(() => list.evaluate((el) => el.scrollWidth - el.clientWidth)).toBeLessThanOrEqual(0);
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);

  await search.focus();
  await search.press("Enter");
  await expect(chosen.getByRole("button", { name: "Load this SAE" })).toBeFocused();
  await page.keyboard.press("Enter");
  // Loaded by its repository and folder.
  const card = page.getByLabel("The loaded SAE");
  await expect(card.getByRole("heading", { name: "exact", exact: true })).toBeVisible();
  await expect(card.getByText("local/exact", { exact: true })).toBeVisible();
  await expect(card.getByText(/the residual stream entering layer 1/)).toBeVisible();
  expect(errors).toEqual([]);
});
