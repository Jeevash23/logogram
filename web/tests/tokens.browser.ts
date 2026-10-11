import { test, expect } from "@playwright/test";

test("the tokens are one tab stop with arrow keys between them, and the strip stays as you left it", async ({ page }) => {
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
  const token = (row: "clean" | "corrupt", i: number) => page.getByRole("button", { name: new RegExp(`^${row} token ${i}:`) });
  await expect(token("clean", 0)).toHaveAttribute("tabindex", "0");
  await expect(token("clean", 1)).toHaveAttribute("tabindex", "-1");
  await token("clean", 0).focus();
  await page.keyboard.press("ArrowRight");
  await expect(token("clean", 1)).toBeFocused();
  await page.keyboard.press("ArrowDown");
  await expect(token("corrupt", 1)).toBeFocused();
  await page.keyboard.press("End");
  await expect(token("corrupt", 13)).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(token("corrupt", 13)).toHaveAttribute("aria-pressed", "true");
  // One Tab leaves all of them.
  await page.keyboard.press("Tab");
  await expect(page.locator("[data-col]:focus")).toHaveCount(0);

  await page.getByRole("button", { name: "Hide tokens", exact: true }).click();
  await page.reload();
  await expect(page.getByRole("button", { name: "Show tokens", exact: true })).toBeVisible();
  await expect(token("clean", 0)).toHaveCount(0);
});
