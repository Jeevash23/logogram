import { test, expect } from "@playwright/test";

// Prompts wrapped in the model's own chat template (the fixture's model ships a small one), and a
// model folder added, offered and removed.

test.beforeEach(async ({ page }) => {
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
});

test("the experiment's prompts are wrapped in the model's chat template, as the form shows", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.getByRole("button", { name: "Experiment", exact: true }).click();
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  const choice = page.getByRole("radiogroup", { name: "Chat template" });
  await expect(choice.getByRole("radio", { name: "None" })).toHaveAttribute("aria-checked", "true");
  await choice.getByRole("radio", { name: "The person's message" }).click();
  // The template writes the beginning of the text, so no beginning-of-sequence token is added.
  const bos = page.getByRole("checkbox", { name: /Prepend the beginning-of-sequence token/ });
  await expect(bos).toBeDisabled();
  await expect(bos).not.toBeChecked();
  // The first prompt as the model reads it: the template's text around it, the prompt underlined.
  const shown = page.getByRole("figure", { name: /What tiny-gpt2 reads for the first prompt/ });
  await expect(shown.locator("mark")).toHaveText(/^When .+ to$/);
  await expect(shown).toContainText("<|endoftext|> Hello When");
  await expect(shown).toContainText(". world");
  await page.getByRole("textbox", { name: "System message" }).fill("world");
  await expect(shown).toContainText("<|endoftext|>world, Hello When");

  // The start of the model's reply answers a message, which has no default.
  await choice.getByRole("radio", { name: "The start of the reply" }).click();
  const summary = page.getByRole("region", { name: "Not ready to run" });
  await expect(summary).toContainText("Write the person's message");
  await page.getByRole("textbox", { name: "The person's message the reply answers" }).fill("Hello");
  await expect(shown).toContainText("<|endoftext|>world, Hello Hello. world When");
  await page.getByRole("button", { name: "Preview spec", exact: true }).click();
  const spec = JSON.parse(await page.getByLabel("Spec JSON").innerText());
  expect(spec.tokenization).toEqual({
    prepend_bos: false,
    chat_template: { kind: "assistant_prefix", system: "world", user: "Hello" },
  });

  // Back to plain text: the beginning-of-sequence choice is the form's again.
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  await page.getByRole("radiogroup", { name: "Chat template" }).getByRole("radio", { name: "None" }).click();
  await expect(page.getByRole("checkbox", { name: /Prepend the beginning-of-sequence token/ })).toBeEnabled();
  await expect(page.getByRole("figure", { name: /reads for the first prompt/ })).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("your text can be read in the chat template too", async ({ page }) => {
  await page.getByRole("button", { name: "Your text", exact: true }).click();
  await page.getByLabel("Text", { exact: true }).fill("When Mary and John went to the store");
  await page.getByRole("radiogroup", { name: "Chat template" }).getByRole("radio", { name: "The person's message" }).click();
  await page.getByRole("button", { name: "Read", exact: true }).click();
  const tokens = page.getByRole("group", { name: /^Tokens of the text/ });
  // The template's beginning and " Hello" before the eight words, "." and " world" after them.
  await expect(tokens.getByRole("button")).toHaveCount(12);
  await expect(tokens.getByRole("button").first()).toHaveAccessibleName(/^Token 0, .*the chat template's/);
  await expect(tokens.getByRole("button").nth(2)).toHaveAccessibleName(/^Token 2, ·When: next/);
  await expect(page.getByRole("heading", { name: "What each layer would predict at token 11" })).toBeVisible();
  // A feature is added from the text's own first token on, not from the template's.
  const writes = page.getByRole("region", { name: "What the model writes next" });
  await writes.getByRole("checkbox", { name: "Add an SAE feature while it continues" }).check();
  await expect(writes.getByRole("spinbutton", { name: "From token" })).toHaveValue("2");
  // As it was: plain text.
  await page.getByRole("radiogroup", { name: "Chat template" }).getByRole("radio", { name: "None" }).click();
});

test("a model folder on this machine is added under a name, offered, and removed", async ({ page }) => {
  const state = await (await page.request.get("/api/state")).json();
  // The fixture's own model folder, beside its projects folder.
  const folder = `${state.projects_parent}/../model`;
  await page.getByRole("button", { name: /^tiny-gpt2 · / }).click();
  const dialog = page.getByRole("dialog", { name: "Load a model" });
  await dialog.getByRole("button", { name: "Add a model folder" }).click();
  const picker = page.getByRole("dialog", { name: "Choose a model folder" });
  await picker.getByLabel("Folder path").fill(folder);
  await picker.getByRole("button", { name: "Go", exact: true }).click();
  // Listed: the box then holds the folder's own path, without "..", and no error is shown.
  await expect(picker.getByLabel("Folder path")).toHaveValue(/^(?!.*\/\.\.\/).*\/model$/);
  await expect(picker.getByRole("alert")).toHaveCount(0);
  await picker.getByRole("button", { name: "Choose this folder" }).click();
  await expect(picker).toBeHidden();
  const adding = dialog.getByRole("region", { name: "Add a model folder" });
  await expect(adding.getByRole("textbox", { name: "Name" })).toHaveValue("model");
  await adding.getByRole("textbox", { name: "Name" }).fill("fixture-model");
  await adding.getByRole("button", { name: "Add", exact: true }).click();
  // Hashed, listed among the models, and chosen.
  const option = dialog.getByRole("radiogroup", { name: "Model" }).getByRole("radio", { name: /^fixture-model/ });
  await expect(option).toHaveAttribute("aria-checked", "true", { timeout: 30_000 });
  await expect(option).toContainText("Local");
  await expect(dialog.getByText(/fixture-model: \d+ files, SHA-256 [0-9a-f]{12}…/)).toBeVisible();
  // Read from its folder: no download, no revision, and whether it fits.
  await expect(dialog.getByText("none: a local folder")).toBeVisible();
  await expect(dialog.getByText(/free in RAM|free on (the GPU|GPU \d+)/)).toBeVisible();
  await expect(dialog.getByLabel("Model revision")).toHaveCount(0);
  // Forgotten here; its folder stays.
  await dialog.getByRole("button", { name: "Remove from the list" }).click();
  await expect(option).toHaveCount(0);
  await expect(dialog.getByRole("radio", { name: /^GPT-2 small/ })).toHaveAttribute("aria-checked", "true");
});

test("a folder typed while the first listing is on its way isn't replaced by it", async ({ page }) => {
  const state = await (await page.request.get("/api/state")).json();
  const folder = `${state.projects_parent}/../model`;
  // The picker's first listing arrives after the folder is typed, before Go is pressed.
  let release = () => {};
  const held = new Promise<void>((resolve) => (release = resolve));
  await page.route(/\/api\/fs$/, async (route) => {
    await held;
    await route.continue();
  });
  await page.getByRole("button", { name: /^tiny-gpt2 · / }).click();
  const dialog = page.getByRole("dialog", { name: "Load a model" });
  await dialog.getByRole("button", { name: "Add a model folder" }).click();
  const picker = page.getByRole("dialog", { name: "Choose a model folder" });
  await picker.getByLabel("Folder path").fill(folder);
  const late = page.waitForResponse(/\/api\/fs$/);
  release();
  await late;
  await page.evaluate(() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done))));
  await expect(picker.getByLabel("Folder path")).toHaveValue(folder);
  await page.unroute(/\/api\/fs$/);
  await picker.getByRole("button", { name: "Go", exact: true }).click();
  await expect(picker.getByLabel("Folder path")).toHaveValue(/model$/);
  await picker.getByRole("button", { name: "Cancel" }).click();
});

test("a folder typed over the box's selected text as the first listing arrives isn't added to the listed folder's path", async ({ page }) => {
  const state = await (await page.request.get("/api/state")).json();
  const folder = `${state.projects_parent}/../model`;
  // The picker's first listing arrives after the box's text is selected and before the folder is
  // typed over it, as when fill() selects and then types, or a paste follows select-all.
  let release = () => {};
  const held = new Promise<void>((resolve) => (release = resolve));
  await page.route(/\/api\/fs$/, async (route) => {
    await held;
    await route.continue();
  });
  await page.getByRole("button", { name: /^tiny-gpt2 · / }).click();
  const dialog = page.getByRole("dialog", { name: "Load a model" });
  await dialog.getByRole("button", { name: "Add a model folder" }).click();
  const picker = page.getByRole("dialog", { name: "Choose a model folder" });
  await picker.getByLabel("Folder path").selectText();
  const late = page.waitForResponse(/\/api\/fs$/);
  release();
  await late;
  // Two frames: time for the late listing to land in the box, were it written there.
  await page.evaluate(() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done))));
  await page.keyboard.insertText(folder);
  await expect(picker.getByLabel("Folder path")).toHaveValue(folder);
  await page.unroute(/\/api\/fs$/);
  await picker.getByRole("button", { name: "Go", exact: true }).click();
  await expect(picker.getByLabel("Folder path")).toHaveValue(/^(?!.*\/\.\.\/).*\/model$/);
  await expect(picker.getByRole("alert")).toHaveCount(0);
  await picker.getByRole("button", { name: "Cancel" }).click();
});

for (const go of [false, true]) {
  test(`Choose this folder chooses the folder in the box, though it isn't listed yet${go ? ", with Go's listing on its way" : ""}`, async ({ page }) => {
    const state = await (await page.request.get("/api/state")).json();
    const folder = `${state.projects_parent}/../model`;
    await page.getByRole("button", { name: /^tiny-gpt2 · / }).click();
    const dialog = page.getByRole("dialog", { name: "Load a model" });
    await dialog.getByRole("button", { name: "Add a model folder" }).click();
    const picker = page.getByRole("dialog", { name: "Choose a model folder" });
    // The folder the picker opens in is listed; the typed folder's listings are held back.
    await expect(picker.getByLabel("Folder path")).not.toHaveValue("");
    let release = () => {};
    const held = new Promise<void>((resolve) => (release = resolve));
    await page.route(/\/api\/fs\?/, async (route) => {
      await held;
      await route.continue();
    });
    await picker.getByLabel("Folder path").fill(folder);
    if (go) await picker.getByRole("button", { name: "Go", exact: true }).click();
    await picker.getByRole("button", { name: "Choose this folder" }).click();
    release();
    const adding = dialog.getByRole("region", { name: "Add a model folder" });
    await expect(adding.getByRole("textbox", { name: "Name" })).toHaveValue("model");
    await expect(picker).toBeHidden();
    await page.unroute(/\/api\/fs\?/);
  });
}

test("a folder typed before the first listing arrives is the one the picker shows", async ({ page }) => {
  const state = await (await page.request.get("/api/state")).json();
  const folder = `${state.projects_parent}/../model`;
  // The picker's first listing (of the folder it starts in) is held back until the typed folder
  // has been listed.
  let release = () => {};
  const held = new Promise<void>((resolve) => (release = resolve));
  await page.route(/\/api\/fs$/, async (route) => {
    await held;
    await route.continue();
  });
  await page.getByRole("button", { name: /^tiny-gpt2 · / }).click();
  const dialog = page.getByRole("dialog", { name: "Load a model" });
  await dialog.getByRole("button", { name: "Add a model folder" }).click();
  const picker = page.getByRole("dialog", { name: "Choose a model folder" });
  await picker.getByLabel("Folder path").fill(folder);
  await picker.getByRole("button", { name: "Go", exact: true }).click();
  await expect(picker.getByLabel("Folder path")).toHaveValue(/model$/);
  const late = page.waitForResponse(/\/api\/fs$/);
  release();
  await late;
  // Two frames: time for a late listing to land, were it shown.
  await page.evaluate(() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done))));
  await expect(picker.getByLabel("Folder path")).toHaveValue(/model$/);
  await page.unroute(/\/api\/fs$/);
  await picker.getByRole("button", { name: "Cancel" }).click();
});
