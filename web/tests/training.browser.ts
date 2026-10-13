import { test, expect } from "@playwright/test";

import { expectAccessible } from "./axe";

// Training in the project: an SAE from Features, with what it would take on this machine, its
// progress and what it measured, then loaded; a tuned lens from Layer predictions, then read. The
// fixture project has a corpus of plain texts the tiny model reads word by word, and the settings
// are made small enough for it. Scanned by axe in the theme the run uses.

test.describe.configure({ mode: "serial" });
test.use({ bypassCSP: true, reducedMotion: "reduce" });

test.beforeEach(async ({ page }) => {
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
});

test("an SAE is trained in the project, said what it takes, and loaded", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.getByRole("button", { name: "Explore", exact: true }).click();
  await page.getByRole("button", { name: "Features", exact: true }).click();
  await page.getByLabel("The loaded SAE").getByRole("button", { name: "Load another SAE", exact: true }).click();
  const trained = page.getByRole("region", { name: "Trained in this project" });
  await expect(trained.getByText("None yet.", { exact: true })).toBeVisible();
  await trained.getByRole("button", { name: "Train an SAE…" }).click();

  const dialog = page.getByRole("dialog", { name: "Train an SAE" });
  // The fixture's corpus (other checks may have added corpora to the fixture project before).
  await dialog.getByLabel("Corpus", { exact: true }).selectOption("ioi-texts");
  await expect(dialog.getByLabel("Corpus", { exact: true })).toHaveValue("ioi-texts");
  await dialog.getByLabel("Name", { exact: true }).fill("browser-sae");
  await dialog.getByLabel("Features", { exact: true }).fill("64");
  await dialog.getByLabel("Features per token", { exact: true }).fill("4");
  await dialog.getByLabel("Steps", { exact: true }).fill("30");
  await dialog.getByLabel("Tokens per step", { exact: true }).fill("64");
  await dialog.getByLabel("Learning rate", { exact: true }).fill("0.01");
  // A setting that can't train is said at once, before anything is asked of the server.
  await dialog.getByLabel("Features per token", { exact: true }).fill("65");
  await expect(dialog.getByText("Features per token can't be more than the number of features.")).toBeVisible();
  await expect(dialog.getByRole("button", { name: "Train", exact: true })).toBeDisabled();
  await dialog.getByLabel("Features per token", { exact: true }).fill("4");
  // What it would take, measured on this machine.
  await expect(dialog.getByText(/^Trains on .* training texts/)).toBeVisible({ timeout: 30_000 });
  await expect(dialog.getByText(/it fits\./)).toBeVisible();
  await expect(dialog.getByText(/Takes about .* here/)).toBeVisible();
  await dialog.getByText("Every other setting").click();
  await expect(dialog.getByLabel("Adam's first beta")).toHaveValue("0.9");
  await expectAccessible(page, "Train an SAE, its settings");

  await dialog.getByRole("button", { name: "Train", exact: true }).click();
  await expect(dialog.getByText("Trained saes/browser-sae.", { exact: true })).toBeVisible({ timeout: 90_000 });
  await expect(dialog.getByRole("img", { name: /The loss curve over/ })).toBeVisible();
  const measured = dialog.getByRole("region", { name: "Measured on the held-out texts" });
  await expect(measured.getByText("Variance explained", { exact: true })).toBeVisible();
  await expect(measured.getByText("Loss recovered", { exact: true })).toBeVisible();
  await expectAccessible(page, "Train an SAE, trained");
  await dialog.getByRole("button", { name: "Load this SAE", exact: true }).click();

  const card = page.getByLabel("The loaded SAE");
  await expect(card.getByRole("heading", { name: "browser-sae", exact: true })).toBeVisible({ timeout: 30_000 });
  await expect(card.getByText("Trained in this project", { exact: true })).toBeVisible();
  await expect(card.getByText(/Its files hash to [0-9a-f]{12}, which specs pin\./)).toBeVisible();
  // The list beside the published ones has it too, and a feature patched from it names it in the spec.
  await card.getByRole("button", { name: "Load another SAE", exact: true }).click();
  await expect(page.getByRole("list", { name: "SAEs trained in this project" })).toContainText("browser-sae");
  // The checks after this one read the fixture's SAE: load the published SAE that is the same one.
  const search = page.getByRole("combobox", { name: "Search" });
  await search.fill("tiny-res");
  const chosen = page.getByRole("group", { name: "The chosen SAE: tiny-res, blocks.1.hook_resid_pre" });
  await chosen.getByRole("button", { name: "Load this SAE" }).click();
  await expect(card.getByRole("heading", { name: "exact", exact: true })).toBeVisible({ timeout: 30_000 });
  expect(errors).toEqual([]);
});

test("a tuned lens is trained and the layer predictions read through it", async ({ page }) => {
  await page.getByRole("button", { name: "Explore", exact: true }).click();
  await page.getByRole("button", { name: "Layer predictions", exact: true }).click();
  await page.getByRole("radio", { name: "Tuned lens", exact: true }).click();
  await page.getByRole("button", { name: "Train a tuned lens…", exact: true }).click();

  const dialog = page.getByRole("dialog", { name: "Train a tuned lens" });
  await dialog.getByLabel("Name", { exact: true }).fill("browser-lens");
  await dialog.getByLabel("Steps", { exact: true }).fill("20");
  await dialog.getByLabel("Tokens per step", { exact: true }).fill("64");
  await expect(dialog.getByText(/^Trains on .* training texts/)).toBeVisible({ timeout: 30_000 });
  await expectAccessible(page, "Train a tuned lens, its settings");
  await dialog.getByRole("button", { name: "Train", exact: true }).click();
  await expect(dialog.getByText("Trained lenses/browser-lens.", { exact: true })).toBeVisible({ timeout: 90_000 });
  const table = dialog.getByRole("table");
  await expect(table).toContainText("Layer 0");
  await expectAccessible(page, "Train a tuned lens, trained");
  await dialog.getByRole("button", { name: "Read the layer predictions with it", exact: true }).click();

  await expect(page.getByLabel("Tuned lens", { exact: true })).toHaveValue("browser-lens");
  await page.getByRole("button", { name: "Measure predictions", exact: true }).click();
  const projections = page.getByRole("table", { name: /Per-layer vocabulary projections/ });
  await expect(projections).toContainText("tuned lens");
  await expect(projections).toContainText("logit lens");
  await expect(page.getByText(/tuned lens browser-lens \([0-9a-f]{12}\), translators for layers 0/)).toBeVisible();
  await expectAccessible(page, "Layer predictions through a tuned lens");
});
