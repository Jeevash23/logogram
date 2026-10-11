import { test, expect } from "@playwright/test";

// Where you are lives in the URL: Back and Forward move between views, a reload comes back to the
// same place, and the Projects screen leads back to the open project as you left it.

test("Back, Forward and reload keep your place", async ({ page }) => {
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
  // The fixture opens its finished run on the model map.
  await expect(page).toHaveURL(/#\/explore\/[^/]+$/);
  const tab = (name: string) => page.getByRole("button", { name, exact: true });

  // The Experiment workspace opens on the tab used there last: Prompts, as the project opened.
  await tab("Experiment").click();
  await expect(tab("Prompts")).toHaveAttribute("aria-current", "page");
  await tab("Configure").click();
  await tab("Spec").click();
  await expect(page).toHaveURL(/#\/spec\//);
  await page.goBack();
  await expect(tab("Configure")).toHaveAttribute("aria-current", "page");
  await page.goBack();
  await expect(tab("Prompts")).toHaveAttribute("aria-current", "page");
  await page.goBack();
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
  await page.goForward();
  await expect(tab("Prompts")).toHaveAttribute("aria-current", "page");

  await tab("Evidence").click();
  await expect(page).toHaveURL(/#\/results\//);
  await page.reload();
  await expect(tab("Results")).toHaveAttribute("aria-current", "page");
  await expect(page.getByRole("main").getByRole("heading", { level: 1, name: "Without BOS" })).toBeVisible();
});

test("the Projects screen leads back to the open project as you left it", async ({ page }) => {
  await page.goto("/?token=browser-test-token");
  const tab = (name: string) => page.getByRole("button", { name, exact: true });
  await tab("Evidence").click();
  await tab("Research notes").click();
  await page.getByRole("button", { name: "All projects" }).click();
  await expect(page).toHaveURL(/#\/projects$/);
  await page.getByRole("button", { name: "Back to Browser fixture" }).click();
  await expect(tab("Research notes")).toHaveAttribute("aria-current", "page");
  // Choosing the open project in Recent goes back to it too, rather than opening it afresh.
  await page.goBack();
  await expect(page.getByRole("button", { name: "Back to Browser fixture" })).toBeVisible();
  await page.getByRole("button", { name: /^Browser fixture/ }).click();
  await expect(tab("Research notes")).toHaveAttribute("aria-current", "page");
});
