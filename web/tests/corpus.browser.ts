import { test, expect, type Page } from "@playwright/test";

import { expectAccessible } from "./axe";

// Corpora and representations in the Explore workspace: a file uploaded as a corpus, where a
// neuron is most active in it, an example opened in Your text; a dataset's prompts on their
// principal components, a component and a mean difference saved as directions, renamed and
// deleted, and the activations exported. Each view is scanned for accessibility.

test.use({ bypassCSP: true, reducedMotion: "reduce" });

const STORIES = [
  { text: "When Mary and John went to the store, John gave a drink to Mary.", label: "store" },
  { text: "When Tom and Kelly went to the park, Kelly gave a ball to Tom.", label: "park" },
  { text: "Then, Sarah and Chris had a lot of fun at the school.", label: "school" },
  { text: "Afterwards, Jessica and Robert went to the restaurant.", label: "store" },
];

async function openView(page: Page, name: string) {
  await page.getByRole("button", { name, exact: true }).click();
  await expect(page.getByRole("button", { name, exact: true })).toHaveAttribute("aria-current", "page");
}

test("a file becomes a corpus, and a neuron's most active tokens open in Your text", async ({ page }) => {
  test.setTimeout(90_000);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
  await openView(page, "Corpus");

  // A Hugging Face dataset's files are listed only when asked: nothing is fetched to show the form.
  const hub = page.waitForRequest((r) => r.url().includes("/api/hub/"), { timeout: 1_000 }).catch(() => null);
  await page.getByRole("radio", { name: "Hugging Face" }).click();
  await expect(page.getByRole("button", { name: "List its files" })).toBeDisabled();
  expect(await hub).toBeNull();
  await expectAccessible(page, "Explore, Corpus, a Hugging Face dataset's file");

  await page.getByRole("radio", { name: "Upload a file" }).click();
  const file = { name: "tales.jsonl", mimeType: "application/json", buffer: Buffer.from(STORIES.map((s) => JSON.stringify(s)).join("\n")) };
  await page.locator('input[type="file"]').setInputFiles(file);
  // The fields are read from the file: the text and the label are found by their names.
  await expect(page.getByLabel("Text field")).toHaveValue("text");
  await expect(page.getByLabel("Label field")).toHaveValue("label");
  await expect(page.getByLabel("Corpus name")).toHaveValue("tales");
  await expectAccessible(page, "Explore, Corpus, a file to upload");
  await page.getByRole("button", { name: "Import", exact: true }).click();
  const chosen = page.getByRole("group", { name: "Corpora" }).getByRole("button", { name: /^tales/ });
  await expect(chosen).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("heading", { level: 1, name: "Where a unit is most active in tales" })).toBeVisible();
  await expect(page.getByText("labels store, park, school")).toBeVisible();

  // Its texts, found by a search through them.
  await page.getByRole("button", { name: "Show the texts" }).click();
  const texts = page.getByRole("list", { name: "Texts of tales" });
  await expect(texts.getByRole("listitem")).toHaveCount(4);
  await page.getByRole("searchbox", { name: "Search the texts" }).fill("park");
  await expect(texts.getByRole("listitem")).toHaveCount(1);
  await expect(texts).toContainText("When Tom and Kelly went to the park");

  // Where a neuron is most active, with every choice stated in the request.
  await page.getByLabel("Layer", { exact: true }).fill("1");
  await page.getByLabel("Neuron", { exact: true }).fill("3");
  await page.getByLabel("Examples", { exact: true }).fill("5");
  const top = page.waitForRequest((r) => r.url().endsWith("/api/corpus/top"));
  await page.getByRole("button", { name: "Find where it is most active" }).click();
  const body = (await top).postDataJSON();
  expect(body).toMatchObject({ corpus: "tales", unit: { kind: "neuron", layer: 1, neuron: 3 }, top_k: 5, context: 8, prepend_bos: true, texts: 1000 });
  expect(body.corpus_sha256).toMatch(/^[0-9a-f]{64}$/);
  await expect(page.getByRole("heading", { name: "The 5 tokens where it is most active" })).toBeVisible();
  await expect(page.getByRole("img", { name: /^Histogram of its values over \d+ tokens/ })).toBeVisible();
  await expect(page.getByText(/^MLP neuron 3 of layer 1: its activation after the nonlinearity\./)).toBeVisible();
  await expectAccessible(page, "Explore, Corpus, with the most active tokens");

  // A neuron has no sign to speak of: no most negative tokens. The first example opens in Your text.
  await expect(page.getByRole("heading", { name: /most negative/ })).toHaveCount(0);
  const first = page.getByRole("button", { name: /^Open text \d+ in Your text, at token \d+$/ }).first();
  const label = await first.getAttribute("aria-label");
  const at = Number(/at token (\d+)$/.exec(label ?? "")?.[1]);
  await first.click();
  await expect(page.getByRole("heading", { level: 1, name: "What the model makes of your text" })).toBeVisible();
  await expect(page.getByRole("heading", { name: `What each layer would predict at token ${at}` })).toBeVisible();
  // The text is read as the corpus was: after the beginning-of-sequence token.
  await expect(page.getByRole("checkbox", { name: "Begin with the beginning-of-sequence token" })).toBeChecked();
  expect(errors).toEqual([]);
});

test("a dataset's prompts as a corpus, and an SAE feature's tokens", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
  await openView(page, "Corpus");
  await page.getByRole("radio", { name: "From prompts" }).click();
  await page.getByRole("radiogroup", { name: "Prompts" }).getByRole("radio", { name: "Clean" }).click();
  await page.getByLabel("Label each text by").selectOption("meta:pattern");
  await page.getByLabel("Corpus name").fill("ioi-clean-patterns");
  await expectAccessible(page, "Explore, Corpus, a dataset's prompts as texts");
  await page.getByRole("button", { name: "Import", exact: true }).click();
  await expect(page.getByRole("group", { name: "Corpora" }).getByRole("button", { name: /^ioi-clean-patterns/ })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByText("labels ABBA, BABA")).toBeVisible();

  // The fixture's SAE is loaded: its features can be read.
  await page.getByRole("radiogroup", { name: "Unit" }).getByRole("radio", { name: "SAE feature" }).click();
  await page.getByLabel("Feature", { exact: true }).fill("2");
  await page.getByRole("button", { name: "Find where it is most active" }).click();
  await expect(page.getByText(/^feature 2 of the SAE local\/exact: its activation\./)).toBeVisible();
  expect(errors).toEqual([]);
});

test("prompts on their principal components, saved as directions, and the activations exported", async ({ page }) => {
  test.setTimeout(90_000);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
  await openView(page, "Representations");
  await expect(page.getByRole("heading", { name: "Directions in this project" })).toBeVisible();

  await page.getByLabel("Layer", { exact: true }).fill("1");
  await page.getByLabel("Token", { exact: true }).selectOption("label");
  await page.getByLabel("Label", { exact: true }).selectOption("S2");
  const pca = page.waitForRequest((r) => r.url().endsWith("/api/representations/pca"));
  await page.getByRole("button", { name: "Find the principal components" }).click();
  expect((await pca).postDataJSON()).toMatchObject({
    source: { kind: "dataset", path: "datasets/ioi.jsonl", sides: "both", mark: "side" },
    site: { kind: "resid_post", layer: 1 },
    position: { kind: "label", label: "S2" },
  });
  const plot = page.getByRole("group", { name: "16 rows on components 1 and 2" });
  await expect(plot).toBeVisible();
  await expect(page.getByRole("list", { name: "Classes" })).toContainText("clean");
  await expect(page.getByRole("list", { name: "The share of the variance each component explains" }).getByRole("listitem")).toHaveCount(10);

  // The keyboard moves across the plot; Enter shows the row.
  await plot.focus();
  await page.keyboard.press("Home");
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("Enter");
  await expect(page.getByText(/^Prompt \d+ \((clean|corrupt)\) · component 1 /)).toBeVisible();
  await page.getByLabel("Up", { exact: true }).selectOption("2");
  await expect(page.getByRole("group", { name: "16 rows on components 1 and 3" })).toBeVisible();
  await expectAccessible(page, "Explore, Representations, with components");

  // A component, and the difference between the classes' means, saved as directions.
  await page.getByRole("button", { name: "Save the component" }).click();
  await expect(page.getByRole("table").getByText("pc1-resid_post-1", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Save the difference" }).click();
  const table = page.getByRole("table");
  await expect(table.getByText("clean-minus-corrupt-resid_post-1", { exact: true })).toBeVisible();
  await expect(table).toContainText("The mean difference between two sets");
  await expect(table).toContainText("resid_post:1");

  // Renamed, then deleted after asking.
  await page.getByRole("button", { name: "Rename pc1-resid_post-1" }).click();
  await page.getByRole("textbox", { name: "New name for pc1-resid_post-1" }).fill("first-component");
  await page.getByRole("button", { name: "Rename", exact: true }).click();
  await expect(table.getByText("first-component", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Delete first-component" }).click();
  await page.getByRole("alertdialog").getByRole("button", { name: "Delete the direction" }).click();
  await expect(table.getByText("first-component", { exact: true })).toHaveCount(0);

  // The activations, exported to the project.
  await page.getByLabel("Export name").fill("ioi-s2");
  await page.getByRole("button", { name: "Export", exact: true }).click();
  await expect(page.getByText("Wrote 16 rows of resid_post:1 to activations/ioi-s2.safetensors")).toBeVisible();

  // The saved direction is a unit of the Corpus view now, read signed.
  await openView(page, "Corpus");
  const units = page.getByRole("radiogroup", { name: "Unit" });
  await expect(units.getByRole("radio", { name: "Direction" })).toBeEnabled();
  expect(errors).toEqual([]);
});
