import { test, expect } from "@playwright/test";

// The Attention view finds heads by what they attend to, on random tokens repeated twice, and
// choosing one shows its pattern.

test("heads are found by what they attend to, and choosing one shows its attention", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Attention", exact: true }).click();
  const finder = page.getByRole("region", { name: "Find heads by what they attend to" });
  await expect(finder).toBeVisible();

  // Settings that can't work say why, and the button waits for good ones.
  await finder.getByLabel("Tokens in each copy").fill("40"); // the fixture model reads 64 tokens
  await expect(finder).toContainText(/Use sequences of 2 to 3[12] tokens: two copies must fit in the model.s context of 64 tokens/);
  await finder.getByLabel("Tokens in each copy").fill("12");
  await finder.getByLabel("Sequences").fill("8");
  const request = page.waitForRequest((r) => r.url().endsWith("/api/heads/kinds"));
  await finder.getByRole("button", { name: "Find heads", exact: true }).click();
  expect((await request).postDataJSON()).toMatchObject({ seed: 0, sequences: 8, length: 12, bootstrap: 1000, ci: 0.95 });
  await expect(finder.getByRole("group", { name: "Induction scores of every head" })).toBeVisible();
  await expect(finder.getByRole("listitem")).toHaveCount(5);
  await expect(finder).toContainText("8 sequences of 12 tokens repeated twice, seed 0");

  // Another kind of head, then a head chosen from the list.
  await finder.getByRole("radiogroup", { name: "Kind of head" }).getByRole("radio", { name: "Previous token" }).click();
  await expect(finder.getByRole("group", { name: "Previous token scores of every head" })).toBeVisible();
  const first = finder.getByRole("listitem").first().getByRole("button");
  const name = (await first.locator("span").first().innerText()).trim();
  await first.click();
  await expect(page.getByRole("heading", { level: 1, name: `Attention · ${name}` })).toBeVisible();
  await expect(first).toHaveAttribute("aria-pressed", "true");

  // A changed setting marks the scores as for other settings until they are found again.
  await finder.getByLabel("Seed").fill("1");
  await expect(finder).toContainText("These scores are for other settings");
  expect(errors).toEqual([]);
});
