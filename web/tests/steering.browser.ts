import { test, expect, type Page } from "@playwright/test";

import { expectAccessible } from "./axe";

// Steering along an SAE feature, from the Features view to what the steered model says, and any
// text continued with a feature added. The run works in a project of its own and reopens the one
// that was open, so the other checks see the fixture as they expect. Each view is scanned by axe.

test.describe.configure({ mode: "serial" });
test.use({ bypassCSP: true, reducedMotion: "reduce" });

const ORIGIN = { origin: `http://127.0.0.1:${process.env.LOGOGRAM_BROWSER_PORT ?? "8877"}` };
const TEXT = "When Mary and John went to the store, John gave a drink to";
let previous: string | null = null;

test.beforeEach(async ({ page }) => {
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
  previous = (await (await page.request.get("/api/state")).json()).project?.path ?? null;
});

test.afterEach(async ({ page }) => {
  if (previous) await page.request.post("/api/projects/open", { data: { path: previous }, headers: ORIGIN });
});

/** A project of its own, open in the page, with IOI prompts generated as someone would. */
async function projectWithPrompts(page: Page, n: number) {
  const state = await (await page.request.get("/api/state")).json();
  const created = await page.request.post("/api/projects", { data: { name: `Steering ${Date.now()}`, parent: state.projects_parent }, headers: ORIGIN });
  expect(created.ok()).toBe(true);
  await expect(page.getByText("No prompts yet.", { exact: false })).toBeVisible();
  await page.getByRole("button", { name: "Experiment", exact: true }).click();
  await page.getByRole("button", { name: "Prompts", exact: true }).click();
  const generator = page.getByRole("radiogroup", { name: "Add prompts" }).getByRole("radio", { name: "Generate a task" });
  if ((await generator.getAttribute("aria-checked")) !== "true") await generator.click();
  await page.getByRole("spinbutton", { name: "Prompts", exact: true }).fill(String(n));
  await page.getByRole("button", { name: `Generate ${n} prompts`, exact: true }).click();
  await expect(page.getByRole("button", { name: "Edit prompt 0", exact: true })).toBeVisible();
}

test("a feature is steered along where its SAE writes, and the run says what the steered model says", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await projectWithPrompts(page, 8);
  await page.getByRole("button", { name: "Explore", exact: true }).click();
  await page.getByRole("button", { name: "Features", exact: true }).click();
  // The strongest feature at the last token of the first prompt.
  const chip = page.getByLabel("Features on each token").getByRole("listitem").last().locator("button[aria-pressed]").first();
  await expect(chip).toBeVisible();
  const feature = Number(/F(\d+)/.exec((await chip.textContent()) ?? "")?.[1]);
  expect(Number.isInteger(feature)).toBe(true);
  await chip.click();
  await page.getByRole("button", { name: "Steer along this feature", exact: true }).click();

  // The form: steering along that feature, where the SAE writes, at the last token.
  await expect(page.getByRole("radiogroup", { name: "Method" }).getByRole("radio", { name: /^Steering/ })).toHaveAttribute("aria-checked", "true");
  await expect(page.getByRole("radiogroup", { name: "Direction" }).getByRole("radio", { name: /^An SAE feature/ })).toHaveAttribute("aria-checked", "true");
  await expect(page.getByRole("spinbutton", { name: "Feature", exact: true })).toHaveValue(String(feature));
  await expect(page.getByText("Where the SAE writes: the residual stream entering layer 1.", { exact: true })).toBeVisible();
  // A feature belongs to one site of one layer: no site in every layer.
  await expect(page.getByRole("radiogroup", { name: "Sweep" }).getByRole("radio", { name: "One site in every layer" })).toBeDisabled();
  await expect(page.getByText("Required: it decides what 1 means, so there's no default.")).toBeVisible();
  await page.getByRole("radiogroup", { name: "Prompts to steer" }).getByRole("radio", { name: /^Steer corrupt prompts/ }).click();
  await page.getByRole("checkbox", { name: "Record what the steered model says" }).check();
  await page.getByRole("spinbutton", { name: "Held-out prompts to continue" }).fill("2");
  await page.getByRole("spinbutton", { name: "New tokens" }).fill("4");
  await expect(page.getByText(/the tokens the model generates aren't steered themselves/)).toBeVisible();
  await expectAccessible(page, "Experiment, steering along an SAE feature");
  await page.getByRole("button", { name: "Preview spec", exact: true }).click();
  const spec = JSON.parse(await page.getByLabel("Spec JSON").innerText());
  expect(spec.experiment).toMatchObject({ kind: "steering", apply_to: "corrupt", direction: { kind: "sae_feature", feature }, continuations: { prompts: 2, tokens: 4 } });
  expect(spec.scope).toEqual({ kind: "sites", sites: [{ kind: "resid_pre", layer: 1, head: null, position: { kind: "last" } }] });
  expect(spec.sae).toMatchObject({ repo: "local/exact" });
  await page.getByRole("button", { name: "Configure", exact: true }).click();
  await page.getByRole("button", { name: "Run", exact: true }).click();

  // What strength 1 is, and what the steered model says.
  const said = page.getByRole("region", { name: "What the steered model says" });
  await expect(said).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText(`Strength 1: F${feature}'s typical activation`)).toBeVisible();
  await expect(page.getByText(/Strength 1 adds what the feature writes at its typical activation, \d/)).toBeVisible();
  await expect(said.getByRole("table")).toHaveCount(2);
  const first = said.getByRole("table").first();
  await expect(first.getByRole("rowheader")).toHaveText([
    "Without steering",
    "×−2",
    "×−1",
    "×1",
    "×2",
    "×4",
    "random ×−2",
    "random ×−1",
    "random ×1",
    "random ×2",
    "random ×4",
  ]);
  // Tokens as the token strip shows them, four new ones on each line.
  await expect(first.getByRole("row").first().locator('[aria-hidden="true"] > span')).toHaveCount(4);
  await expectAccessible(page, "Evidence, what the steered model says");
  expect(errors).toEqual([]);
});

test("any text is continued, plainly and with an SAE feature added from a token on", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.getByRole("button", { name: "Explore", exact: true }).click();
  await page.getByRole("button", { name: "Your text", exact: true }).click();
  await page.getByLabel("Text", { exact: true }).fill(TEXT);
  // The fixture's run has no BOS, which the view starts from; read with one, stated.
  await page.getByRole("checkbox", { name: "Begin with the beginning-of-sequence token" }).check();
  await page.getByRole("button", { name: "Read", exact: true }).click();
  const section = page.getByRole("region", { name: "What the model writes next" });
  await section.getByRole("spinbutton", { name: "New tokens" }).fill("5");
  await section.getByRole("button", { name: "Continue the text", exact: true }).click();
  await expect(section.getByRole("rowheader")).toHaveText(["Without steering"]);
  await expect(section.getByRole("row").first().locator('[aria-hidden="true"] > span')).toHaveCount(5);

  // A feature of the loaded SAE, added from the first real token on, the new tokens included.
  await section.getByRole("checkbox", { name: "Add an SAE feature while it continues" }).check();
  await section.getByRole("spinbutton", { name: "Feature", exact: true }).fill("3");
  await expect(section.getByText(/Feature 3 is at most \d+\.\d\d on this text, at token \d+/)).toBeVisible();
  await expect(section.getByRole("spinbutton", { name: "From token" })).toHaveValue("1");
  await expect(section.getByText("Give the activation to add it at: a number, in the feature's own units.")).toBeVisible();
  await expect(section.getByRole("button", { name: "Continue the text", exact: true })).toBeDisabled();
  await section.getByRole("spinbutton", { name: "Activation" }).fill("6");
  await section.getByRole("button", { name: "Continue the text", exact: true }).click();
  await expect(section.getByRole("rowheader")).toHaveText(["Without steering", "F3 at 6.00 from token 1"]);
  await expectAccessible(page, "Explore, your text continued with a feature");
  // A token outside the text is said before anything runs.
  await section.getByRole("spinbutton", { name: "From token" }).fill("99");
  await expect(section.getByText(/^Choose a token from 0 to \d+ to add it from\.$/)).toBeVisible();
  await expect(section.getByRole("button", { name: "Continue the text", exact: true })).toBeDisabled();
  expect(errors).toEqual([]);
});
