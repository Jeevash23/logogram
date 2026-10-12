import { test, expect, type WebSocketRoute } from "@playwright/test";

import type { LookAttention, RunDetail } from "../src/api/types";

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
  // The keyboard stays on the token that was read, to go on from there.
  await expect(tokens.getByRole("button").nth(1)).toBeFocused();
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("Enter");
  await expect(page.getByRole("heading", { name: "What each layer would predict at token 2" })).toBeVisible();
  await expect(tokens.getByRole("button").nth(2)).toBeFocused();

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

test("your text and the head finder are read by the loaded model, whichever model the open run names", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  // The run the project opens on, as if it had run the loaded model in another dtype.
  await page.route(/\/api\/runs\/[^/?]+$/, async (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    const response = await route.fetch();
    const detail = (await response.json()) as RunDetail;
    await route.fulfill({ response, json: { ...detail, spec: { ...detail.spec, model: { ...detail.spec.model, dtype: "float16" } } } });
  });
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
  const loaded = (await (await page.request.get("/api/state")).json()).model.info;

  await page.getByRole("button", { name: "Your text", exact: true }).click();
  await page.getByLabel("Text", { exact: true }).fill("When Mary and John went to the store");
  const look = page.waitForRequest((r) => r.url().endsWith("/api/look"));
  await page.getByRole("button", { name: "Read", exact: true }).click();
  expect((await look).postDataJSON().model).toMatchObject({ id: loaded.id, revision: loaded.revision, dtype: loaded.dtype });
  await expect(page.getByRole("group", { name: /^Tokens of the text/ }).getByRole("button")).toHaveCount(8);

  // With no head chosen, the Attention view finds heads on random tokens, read by the loaded model.
  await page.getByRole("button", { name: "Attention", exact: true }).click();
  const finder = page.getByRole("region", { name: "Find heads by what they attend to" });
  await finder.getByLabel("Tokens in each copy").fill("12");
  await finder.getByLabel("Sequences").fill("8");
  const kinds = page.waitForRequest((r) => r.url().endsWith("/api/heads/kinds"));
  await finder.getByRole("button", { name: "Find heads", exact: true }).click();
  expect((await kinds).postDataJSON().model).toMatchObject({ id: loaded.id, dtype: loaded.dtype });
  await expect(finder.getByRole("group", { name: "Induction scores of every head" })).toBeVisible();
  await page.unrouteAll({ behavior: "wait" });
  expect(errors).toEqual([]);
});

test("a reading belongs to the model and the SAE that made it, and a head's attention to the tokens it read", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  // The server's events, into which a model or an SAE loaded meanwhile can be added.
  let events: WebSocketRoute | null = null;
  await page.routeWebSocket(/\/ws$/, (ws) => {
    events = ws;
    ws.connectToServer();
  });
  const send = (event: Record<string, unknown>) => (events as WebSocketRoute | null)?.send(JSON.stringify(event));
  // A head's pattern comes with the tokens it was read on; here they are told apart from the
  // reading's by their case.
  await page.route("**/api/look/attention", async (route) => {
    const response = await route.fetch();
    const data = (await response.json()) as LookAttention;
    await route.fulfill({ response, json: { ...data, tokens: data.tokens.map((x) => x.toUpperCase()) } });
  });
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
  const state = await (await page.request.get("/api/state")).json();
  await page.getByRole("button", { name: "Your text", exact: true }).click();
  await page.getByLabel("Text", { exact: true }).fill("When Mary and John went to the store");
  await page.getByRole("button", { name: "Read", exact: true }).click();
  const tokens = page.getByRole("group", { name: /^Tokens of the text/ });
  await expect(tokens.getByRole("button")).toHaveCount(8);
  const attention = page.getByRole("region", { name: "Where a head attends in this text" });
  await expect(attention.getByRole("group", { name: "Attention of L0 H0 in your text" })).toBeVisible();
  await attention.getByText("View values as a table").click();
  await expect(attention.getByRole("columnheader", { name: "1: ·MARY", exact: true })).toBeVisible();

  // Another model loaded: the reading was the other model's, so it goes, and says why.
  send({ type: "model", ...state.model, info: { ...state.model.info, dtype: "float16" } });
  const modelChanged = page.getByText("The model changed since the text was read: press Read to read it again.");
  await expect(modelChanged).toBeVisible();
  await expect(tokens).toHaveCount(0);
  await expect(attention).toHaveCount(0);
  await expect(page.getByRole("region", { name: "What the model writes next" })).toHaveCount(0);
  // The model that read it, loaded again: the reading is its own once more.
  send({ type: "model", ...state.model });
  await expect(modelChanged).toHaveCount(0);
  await expect(tokens.getByRole("button")).toHaveCount(8);
  await expect(attention).toBeVisible();

  // The SAE loaded when it was read, unloaded: the features it found go with the reading.
  expect(state.sae.state).toBe("ready");
  const features = page.getByRole("region", { name: /^The SAE's strongest features/ });
  send({ type: "sae", state: "none" });
  await expect(page.getByText("The SAE was unloaded since the text was read: press Read to read it again.")).toBeVisible();
  await expect(tokens).toHaveCount(0);
  await expect(features).toHaveCount(0);
  // Read without an SAE, the reading has nothing of one: the first loaded since reads its features.
  await page.getByRole("button", { name: "Read", exact: true }).click();
  await expect(tokens.getByRole("button")).toHaveCount(8);
  await expect(page.getByText("Load an SAE in Features to see which of its features fire on each token.")).toBeVisible();
  send({ type: "sae", ...state.sae });
  await expect(features).toBeVisible();
  await expect(tokens.getByRole("button")).toHaveCount(8);
  // That SAE is the reading's now: another one loaded makes it old.
  send({ type: "sae", ...state.sae, info: { ...state.sae.info, repo: "local/other" } });
  await expect(page.getByText("The SAE changed since the text was read: press Read to read it again.")).toBeVisible();
  await expect(features).toHaveCount(0);
  send({ type: "sae", ...state.sae });
  await expect(tokens.getByRole("button")).toHaveCount(8);
  await expect(features).toBeVisible();
  // The pattern read again for the reading shown again may still be on its way.
  await page.unrouteAll({ behavior: "wait" });
  expect(errors).toEqual([]);
});
