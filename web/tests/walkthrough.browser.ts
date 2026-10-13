import { test, expect, type Locator, type Page } from "@playwright/test";

import { expectAccessible } from "./axe";

// The walkthrough of the bundled IOI example, from the keyboard alone: offered on the first
// screens until it is finished or skipped, and started from Help, the guide and the command
// palette too; a step at a time beside the real controls, each with an ink outline and nothing
// dimming the page; Next, Back, Skip and Escape; the keyboard on the step, and back on the page
// when it ends; and how it ended kept in the person's settings. Axe checks every step, in the
// theme the run is in. Each test reopens the project that was open before it, so the other checks
// see the fixture as they expect.

test.use({ bypassCSP: true, reducedMotion: "reduce" });
test.describe.configure({ mode: "serial" });

const BASE = `http://127.0.0.1:${process.env.LOGOGRAM_BROWSER_PORT ?? "8877"}`;
const ORIGIN = { origin: BASE };
const API = { authorization: "Bearer browser-test-token", origin: BASE };
const STEPS = ["The prompt pairs", "Check the baseline", "The experiment", "Read the map", "Open a site", "Check that it holds"];
let previous: string | null = null;

test.beforeEach(async ({ page }) => {
  await page.route(/^https?:\/\/(?!127\.0\.0\.1[:/])/, (route) => route.abort());
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
  previous = (await (await page.request.get("/api/state")).json()).project?.path ?? null;
});

test.afterEach(async ({ page }) => {
  if (previous) await page.request.post("/api/projects/open", { data: { path: previous }, headers: ORIGIN });
});

/** The step on screen: its dialog, named by its title. */
function step(page: Page, title: string): Locator {
  return page.getByRole("dialog", { name: title });
}

/** Whether the keyboard is on the step shown. */
async function focusOnStep(page: Page): Promise<boolean> {
  return page.evaluate(() => document.activeElement?.closest("[data-tour-step]") !== null);
}

/** Tab, from wherever the keyboard is, until ``target`` has it. */
async function tabTo(page: Page, target: Locator, max = 12): Promise<void> {
  for (let i = 0; i < max; i++) {
    if (await target.evaluate((el) => el === document.activeElement)) return;
    await page.keyboard.press("Tab");
  }
  await expect(target).toBeFocused();
}

/** Press one of the step's buttons from the keyboard. */
async function press(page: Page, title: string, button: string): Promise<void> {
  await tabTo(page, step(page, title).getByRole("button", { name: button, exact: true }));
  await page.keyboard.press("Enter");
}

/** The ink outlines on the page, as boxes. */
async function outlines(page: Page) {
  return page.locator("[data-tour-outline]").evaluateAll((els) =>
    els.map((el) => {
      const box = el.getBoundingClientRect();
      return { x: box.x, y: box.y, width: box.width, height: box.height, events: getComputedStyle(el).pointerEvents };
    }),
  );
}

/** Whether a click on the middle of ``target`` reaches it: nothing covers it. */
async function reachable(target: Locator): Promise<boolean> {
  return target.evaluate((el) => {
    const box = el.getBoundingClientRect();
    const hit = document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2);
    return hit !== null && el.contains(hit);
  });
}

/** The part of ``target`` in view: its box, cut by the window and by the ancestors that clip it. */
async function inView(target: Locator) {
  return target.evaluate((el) => {
    const box = el.getBoundingClientRect();
    let { left, top, right, bottom } = box;
    for (let node = el.parentElement; node && node !== document.body; node = node.parentElement) {
      const style = getComputedStyle(node);
      if (style.overflowX === "visible" && style.overflowY === "visible") continue;
      const clip = node.getBoundingClientRect();
      if (style.overflowX !== "visible") [left, right] = [Math.max(left, clip.left), Math.min(right, clip.right)];
      if (style.overflowY !== "visible") [top, bottom] = [Math.max(top, clip.top), Math.min(bottom, clip.bottom)];
    }
    [left, top] = [Math.max(left, 0), Math.max(top, 0)];
    [right, bottom] = [Math.min(right, window.innerWidth), Math.min(bottom, window.innerHeight)];
    return { left, top, right, bottom };
  });
}

/** Whether ``box`` (an outline) surrounds the part of ``target`` in view, closely: no more than a
 * few pixels beyond it, so that it doesn't run over what is around it. */
async function surrounds(target: Locator, box: { x: number; y: number; width: number; height: number }): Promise<boolean> {
  const t = await inView(target);
  const [right, bottom] = [box.x + box.width, box.y + box.height];
  const around = box.x <= t.left && box.y <= t.top && right >= t.right - 0.5 && bottom >= t.bottom - 0.5;
  const close = box.x >= t.left - 8 && box.y >= t.top - 8 && right <= t.right + 8 && bottom <= t.bottom + 8;
  return t.right > t.left && t.bottom > t.top && around && close;
}

/** Wait until an ink outline surrounds the part of ``target`` in view. */
async function expectOutlined(page: Page, target: Locator): Promise<void> {
  const around = async () => {
    for (const box of await outlines(page)) if (await surrounds(target, box)) return true;
    return false;
  };
  await expect.poll(around, { message: "no ink outline surrounds the control" }).toBe(true);
}

/** How the walkthrough ended, as the person's settings keep it. */
async function ended(page: Page): Promise<string | null> {
  return (await (await page.request.get("/api/state")).json()).walkthrough;
}

/** Go to the first screens, from the keyboard. */
async function allProjects(page: Page): Promise<void> {
  await page.getByRole("button", { name: "All projects" }).focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("list", { name: "Examples" })).toBeVisible();
}

test("the first screen offers the walkthrough with the example, and it goes step by step from the keyboard", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  expect(await ended(page)).toBeNull();
  await allProjects(page);
  const example = page
    .getByRole("list", { name: "Examples" })
    .getByRole("listitem")
    .filter({ has: page.getByRole("heading", { level: 3, name: "IOI example", exact: true }) });
  const offer = example.getByRole("button", { name: "Walk me through it", exact: true });
  // Only the example it is written for offers it.
  await expect(page.getByRole("button", { name: "Walk me through it", exact: true })).toHaveCount(1);
  await offer.focus();
  await page.keyboard.press("Enter");

  // 1. The prompts, in the example's project: the keyboard is on the step, the table outlined.
  await expect(step(page, STEPS[0])).toBeVisible({ timeout: 30_000 });
  await expect(step(page, STEPS[0])).toContainText("Walkthrough · 1 of 6");
  await expect.poll(() => focusOnStep(page)).toBe(true);
  expect((await (await page.request.get("/api/state")).json()).project.name).toBe("IOI example");
  await expect(page.getByRole("button", { name: "Prompts", exact: true })).toHaveAttribute("aria-current", "page");
  const table = page.locator("[data-tour='prompts']");
  await expect.poll(async () => (await outlines(page)).length).toBe(1);
  const [ring] = await outlines(page);
  expect(ring.events).toBe("none");
  await expectOutlined(page, table);
  // Nothing dims or covers the page: clicks reach the top bar and the tabs while the step is shown.
  expect(await reachable(page.getByRole("banner").getByRole("button", { name: "Help", exact: true }))).toBe(true);
  expect(await reachable(page.getByRole("button", { name: "Baseline", exact: true }))).toBe(true);
  await expectAccessible(page, "Walkthrough, the prompt pairs");

  // 2. The baseline: Next moves on, to the view the step is about.
  await press(page, STEPS[0], "Next");
  await expect(step(page, STEPS[1])).toBeVisible();
  await expect(page.getByRole("button", { name: "Baseline", exact: true })).toHaveAttribute("aria-current", "page");
  await expect.poll(() => focusOnStep(page)).toBe(true);
  await expectAccessible(page, "Walkthrough, the baseline");

  // 3. The experiment: the form's heading and Run, both outlined. Back goes back.
  await press(page, STEPS[1], "Next");
  await expect(step(page, STEPS[2])).toBeVisible();
  await expect(page.getByRole("button", { name: "Configure", exact: true })).toHaveAttribute("aria-current", "page");
  await expect.poll(async () => (await outlines(page)).length).toBe(2);
  await expectAccessible(page, "Walkthrough, the experiment");
  // Tab from the step's last button goes on to the control it points at.
  await tabTo(page, step(page, STEPS[2]).getByRole("button", { name: "Next", exact: true }));
  await page.keyboard.press("Tab");
  await expect(page.locator("[data-tour='run']")).toBeFocused();
  await expect(step(page, STEPS[2])).toBeVisible();
  await page.keyboard.press("Shift+Tab");
  await press(page, STEPS[2], "Back");
  await expect(step(page, STEPS[1])).toBeVisible();
  await press(page, STEPS[1], "Next");
  await press(page, STEPS[2], "Next");

  // 4. The map: with no run yet, the loaded model's map, and what the run will paint there.
  await expect(step(page, STEPS[3])).toBeVisible();
  await expect(step(page, STEPS[3])).toContainText("When the run finishes");
  await expect(page.getByRole("button", { name: "Model", exact: true })).toHaveAttribute("aria-current", "page");
  await expect.poll(async () => (await outlines(page)).length).toBe(1);
  await expectOutlined(page, page.locator("[data-tour='map']"));
  await expectAccessible(page, "Walkthrough, the map");

  // 5. A site: the inspector beside the map.
  await press(page, STEPS[3], "Next");
  await expect(step(page, STEPS[4])).toBeVisible();
  await expectOutlined(page, page.getByRole("complementary", { name: "Inspector" }));
  await expectAccessible(page, "Walkthrough, a site");

  // 6. Robustness: with no finished run, the step points at the results' tab and says what comes.
  await press(page, STEPS[4], "Next");
  await expect(step(page, STEPS[5])).toBeVisible();
  await expect(step(page, STEPS[5])).toContainText("Once the run finishes");
  await expect(step(page, STEPS[5]).getByRole("button", { name: "Finish", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Results", exact: true })).toHaveAttribute("aria-current", "page");
  await expectAccessible(page, "Walkthrough, robustness");

  // Escape skips it: the step and its outline go, and the keyboard is back on the page.
  await page.keyboard.press("Escape");
  await expect(step(page, STEPS[5])).toBeHidden();
  await expect.poll(async () => (await outlines(page)).length).toBe(0);
  await expect(page.getByRole("button", { name: "Results", exact: true })).toBeFocused();
  await expect.poll(() => ended(page)).toBe("skipped");

  // Once skipped, the first screens stop offering it.
  await allProjects(page);
  await expect(page.getByRole("button", { name: "Walk me through it", exact: true })).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("from Help, beside a finished run, it points at the run's map, the inspector and Check robustness, and Finish is kept", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  // A finished run in the example's project, made with the fixture's tiny model on the example's
  // own prompts.
  const fixture = ((await (await page.request.get("/api/runs", { headers: API })).json()) as { id: string; status: string }[]).find(
    (r) => r.status === "finished",
  );
  const { spec } = await (await page.request.get(`/api/runs/${fixture?.id}`, { headers: API })).json();
  const opened = await page.request.post("/api/projects/example", { data: { example: "ioi-gpt2" }, headers: ORIGIN });
  expect(opened.ok()).toBe(true);
  await expect(page.getByRole("button", { name: "Prompts", exact: true })).toBeVisible();
  const name = `Heads of the tiny model ${Date.now()}`;
  const started = await page.request.post("/api/runs", {
    data: { spec: { ...spec, name, dataset: { path: "datasets/ioi.jsonl", sha256: null, limit: 6 } } },
    headers: API,
  });
  expect(started.ok()).toBe(true);
  const { run_id: id } = await started.json();
  await expect
    .poll(async () => ((await (await page.request.get("/api/runs", { headers: API })).json()) as { id: string; status: string }[]).find((r) => r.id === id)?.status, {
      timeout: 60_000,
    })
    .toBe("finished");
  // Open it from the command palette, once the app has heard of the run: it was started through
  // the API, so the palette lists it only when the app's own list of runs catches up.
  await page.keyboard.press("ControlOrMeta+k");
  await page.getByLabel("Command", { exact: true }).fill(name);
  await expect(page.getByRole("option", { name: `Open run: ${name}`, exact: true })).toHaveAttribute("aria-selected", "true", {
    timeout: 15_000,
  });
  await page.keyboard.press("Enter");
  await expect(page.getByRole("main").getByRole("heading", { level: 1, name })).toBeVisible();

  // Help → Walk me through the example, in the example that is open: it starts at once.
  const help = page.getByRole("banner").getByRole("button", { name: "Help", exact: true });
  await help.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("menuitem", { name: "Guide", exact: true })).toBeFocused();
  await page.keyboard.press("ArrowDown");
  await expect(page.getByRole("menuitem", { name: "Walk me through the example" })).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(step(page, STEPS[0])).toBeVisible();
  await expect.poll(() => focusOnStep(page)).toBe(true);
  for (const title of STEPS.slice(0, 3)) await press(page, title, "Next");

  // The map is the run's.
  await expect(step(page, STEPS[3])).toBeVisible();
  await expect(step(page, STEPS[3])).toContainText("Show values");
  await expect(page.getByRole("button", { name: "Results", exact: true })).toHaveAttribute("aria-current", "page");
  await expectOutlined(page, page.locator("[data-tour='map']"));
  await expectAccessible(page, "Walkthrough, the run's map");
  await press(page, STEPS[3], "Next");
  await expect(step(page, STEPS[4])).toBeVisible();
  await expectAccessible(page, "Walkthrough, the inspector beside the results");
  await press(page, STEPS[4], "Next");

  // Check robustness, outlined; Finish ends the walkthrough on it.
  await expect(step(page, STEPS[5])).toBeVisible();
  const robustness = page.getByRole("button", { name: "Check robustness", exact: true });
  await expectOutlined(page, robustness);
  await expectAccessible(page, "Walkthrough, Check robustness");
  await press(page, STEPS[5], "Finish");
  await expect(step(page, STEPS[5])).toBeHidden();
  await expect(robustness).toBeFocused();
  await expect.poll(() => ended(page)).toBe("finished");
  expect(errors).toEqual([]);
});

test("from the guide, in an 800-pixel window, every step stays in view, and a site opens from the Inspector button", async ({ page }) => {
  await page.setViewportSize({ width: 800, height: 700 });
  await page.getByRole("banner").getByRole("button", { name: "Help", exact: true }).focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("menuitem", { name: "Guide", exact: true })).toBeFocused();
  await page.keyboard.press("Enter");
  const guide = page.getByRole("dialog", { name: "Guide", exact: true });
  await tabTo(page, guide.getByRole("button", { name: "Walk me through the example", exact: true }));
  await page.keyboard.press("Enter");
  for (const [i, title] of STEPS.entries()) {
    const shown = step(page, title);
    await expect(shown).toBeVisible({ timeout: i === 0 ? 30_000 : 5_000 });
    await expect(shown).toBeInViewport({ ratio: 1 });
    const box = await shown.boundingBox();
    expect(box && box.x >= 0 && box.x + box.width <= 800).toBe(true);
    if (title === "Open a site") {
      await expectOutlined(page, page.locator("[data-tour='inspector-button']"));
      await expect(shown).toContainText("then Inspector to open it in a panel");
    }
    if (i < STEPS.length - 1) await press(page, title, "Next");
  }
  await expect(guide).toBeHidden();
  await press(page, STEPS[5], "Skip");
  await expect(step(page, STEPS[5])).toBeHidden();
});

test("the command palette starts it too", async ({ page }) => {
  await page.keyboard.press("ControlOrMeta+k");
  await page.getByLabel("Command", { exact: true }).fill("walk me through");
  await page.keyboard.press("Enter");
  await expect(step(page, STEPS[0])).toBeVisible({ timeout: 30_000 });
  await expect.poll(() => focusOnStep(page)).toBe(true);
  await page.keyboard.press("Escape");
  await expect(step(page, STEPS[0])).toBeHidden();
  await expect.poll(() => ended(page)).toBe("skipped");
});
