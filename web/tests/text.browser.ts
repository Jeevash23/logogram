import { test, expect } from "@playwright/test";

// Any text, read without a dataset: tokens with predictions, a token layer by layer, attention.

test("any text is read: predictions after each token, a token layer by layer, and a head's attention", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Your text", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1, name: "What the model makes of your text" })).toBeVisible();
  await page.getByLabel("Text", { exact: true }).fill("When Mary and John went to the store");
  // The fixture's run has no BOS, which the view starts from; read with one, stated.
  await page.getByRole("checkbox", { name: "Begin with the beginning-of-sequence token" }).check();
  await page.getByRole("button", { name: "Read", exact: true }).click();

  const tokens = page.getByRole("group", { name: /^Tokens of the text/ });
  await expect(tokens.getByRole("button")).toHaveCount(9); // BOS and eight words
  await expect(page.getByRole("list", { name: /^The most likely next tokens after token 8/ }).getByRole("listitem")).toHaveCount(5);
  await expect(page.getByRole("heading", { name: "What each layer would predict at token 8" })).toBeVisible();

  // One tab stop: the arrow keys move between tokens, Enter reads one layer by layer.
  await tokens.getByRole("button").last().focus();
  await page.keyboard.press("Home");
  await expect(tokens.getByRole("button").first()).toBeFocused();
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("Enter");
  await expect(page.getByRole("heading", { name: "What each layer would predict at token 1" })).toBeVisible();
  await expect(tokens.getByRole("button").nth(1)).toHaveAttribute("aria-pressed", "true");

  // A head's attention in the text.
  await page.getByLabel("Layer", { exact: true }).fill("1");
  await page.getByLabel("Head", { exact: true }).fill("2");
  await expect(page.getByRole("group", { name: "Attention of L1 H2 in your text" })).toBeVisible();

  // Editing marks the reading as old until it is read again.
  await page.getByLabel("Text", { exact: true }).fill("When Mary and John went to the park");
  await expect(page.getByText("The text or the choices changed since it was read")).toBeVisible();
  await page.getByLabel("Text", { exact: true }).press("ControlOrMeta+Enter");
  await expect(page.getByText("The text or the choices changed since it was read")).toBeHidden();
  expect(errors).toEqual([]);
});
