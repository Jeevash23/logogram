import { test, expect, type Locator, type Page } from "@playwright/test";

// Help, in the top bar: a menu that works from the keyboard, links to the documentation that open
// the repository's pages in a new tab, a problem report that quotes the last error without its
// paths and copies it, and About. The guide and the walkthrough it opens have checks of their own
// (guide.browser.ts, walkthrough.browser.ts). Nothing in these checks leaves the machine: links are
// read, never followed, any request for another host is refused, and the update check is answered
// here.

const REPOSITORY = "https://github.com/Jeevash23/logogram";
const DOCS: [string, string][] = [
  ["Getting started", `${REPOSITORY}/blob/main/docs/getting-started.md`],
  ["Concepts", `${REPOSITORY}/blob/main/docs/README.md#concepts`],
  ["How-to guides", `${REPOSITORY}/blob/main/docs/README.md#how-to-guides`],
  ["Reference", `${REPOSITORY}/blob/main/docs/README.md#reference`],
  ["Running on a remote GPU", `${REPOSITORY}#running-on-a-remote-gpu`],
  ["Troubleshooting", `${REPOSITORY}#troubleshooting`],
];

test.beforeEach(async ({ page }) => {
  await page.route(/^https?:\/\/(?!127\.0\.0\.1[:/])/, (route) => route.abort());
  await page.goto("/?token=browser-test-token");
  await expect(page.getByRole("main").getByRole("heading", { name: "tiny-gpt2", exact: true })).toBeVisible();
});

function helpButton(page: Page): Locator {
  return page.getByRole("banner").getByRole("button", { name: "Help", exact: true });
}

/** A row of the report or of About: the value beside a label. */
function row(scope: Locator, label: string): Locator {
  return scope.locator("dl > div").filter({ has: scope.page().locator("dt", { hasText: new RegExp(`^${label}$`) }) }).locator("dd");
}

test("the Help menu works from the keyboard, and its documentation opens on GitHub in a new tab", async ({ page }) => {
  const help = helpButton(page);
  await help.focus();
  await page.keyboard.press("Enter");
  const menu = page.getByRole("menu");
  await expect(menu).toBeVisible();
  await expect(menu.getByRole("menuitem", { name: "Guide", exact: true })).toBeFocused();
  await expect(menu.getByRole("menuitem")).toHaveText([
    "Guide",
    "Walk me through the example",
    /^Keyboard shortcuts/,
    ...DOCS.map(([title]) => new RegExp(`^${title}`)),
    "Report a problem…",
    "System check",
    "About Logogram",
  ]);
  await page.keyboard.press("ArrowDown");
  await expect(menu.getByRole("menuitem", { name: "Walk me through the example" })).toBeFocused();

  const docs = menu.getByRole("group", { name: "The documentation, on GitHub" });
  await expect(docs.getByRole("menuitem")).toHaveCount(DOCS.length);
  for (const [title, href] of DOCS) {
    const link = docs.getByRole("menuitem", { name: `${title} (opens in a new tab)`, exact: true });
    await expect(link).toHaveAttribute("href", href);
    await expect(link).toHaveAttribute("target", "_blank");
    await expect(link).toHaveAttribute("rel", "noopener noreferrer");
  }

  // Escape closes the menu and gives the focus back to Help.
  await page.keyboard.press("Escape");
  await expect(menu).toBeHidden();
  await expect(help).toBeFocused();

  // Keyboard shortcuts opens the sheet; closing it returns the focus to Help, too.
  await page.keyboard.press("Enter");
  await expect(menu.getByRole("menuitem", { name: "Guide", exact: true })).toBeFocused();
  // Radix moves the focus after the key, so each step is awaited before the next.
  await page.keyboard.press("ArrowDown");
  await expect(menu.getByRole("menuitem", { name: "Walk me through the example" })).toBeFocused();
  await page.keyboard.press("ArrowDown");
  await expect(menu.getByRole("menuitem", { name: "Keyboard shortcuts" })).toBeFocused();
  await page.keyboard.press("Enter");
  const shortcuts = page.getByRole("dialog", { name: "Keyboard shortcuts" });
  await expect(shortcuts).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(shortcuts).toBeHidden();
  await expect(help).toBeFocused();

  // System check opens the system check, and leads back to the project.
  await help.click();
  await menu.getByRole("menuitem", { name: "System check" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "System check" })).toBeVisible();
  await page.getByRole("button", { name: "Back to the project" }).click();
  await expect(helpButton(page)).toBeVisible();
});

test("a problem report quotes the last error without its paths, and is copied, not sent", async ({ page }) => {
  await page.context().grantPermissions(["clipboard-read", "clipboard-write"]);
  // An error the app shows, naming a file by its full path, as errors from libraries do.
  const file = ["", "srv", "lab", "someone", "logogram", "updates.json"].join("/");
  await page.route("**/api/update/check", (route) =>
    route.fulfill({ status: 500, contentType: "application/json", body: JSON.stringify({ error: `Couldn't write ${file}: permission denied.` }) }),
  );
  await page.keyboard.press("ControlOrMeta+k");
  await page.getByLabel("Command", { exact: true }).fill("check for updates");
  await page.keyboard.press("Enter");
  await expect(page.getByRole("alert").filter({ hasText: "Couldn't write" })).toBeVisible();

  const reports: unknown[] = [];
  page.on("request", (request) => {
    if (request.url().endsWith("/api/report")) reports.push(request.postDataJSON());
  });
  await helpButton(page).click();
  await page.getByRole("menuitem", { name: "Report a problem…" }).click();
  const dialog = page.getByRole("dialog", { name: "Report a problem" });
  const report = dialog.getByRole("region", { name: "The report" });
  await expect(report).toBeVisible();
  await expect(row(report, "Logogram")).toHaveText(/^\d+\.\d+/);
  await expect(row(report, "Model")).toHaveText("tiny-gpt2, revision test, float32 on cpu, weights processed");
  await expect(row(report, "Last error")).toHaveText("Couldn't write <path>: permission denied.");
  await expect(dialog).not.toContainText("someone");
  await expect(row(report, "Processor")).toHaveCount(0);
  expect(reports).toEqual([{ include_hardware: false, last_error: `Couldn't write ${file}: permission denied.` }]);

  // Hardware details only when asked for.
  await dialog.getByRole("checkbox", { name: "Include hardware details" }).check();
  await expect(row(report, "Processor")).toHaveCount(1);
  await expect(row(report, "Memory")).toHaveCount(1);

  const issue = dialog.getByRole("link", { name: "Open a new issue (opens in a new tab)" });
  await expect(issue).toHaveAttribute("href", `${REPOSITORY}/issues/new?template=bug_report.yml`);
  await expect(issue).toHaveAttribute("target", "_blank");
  await expect(issue).toHaveAttribute("rel", "noopener noreferrer");

  await dialog.getByRole("button", { name: "Copy the report" }).click();
  await expect(dialog.getByRole("button", { name: "Copied" })).toBeVisible();
  await expect(dialog.getByRole("status")).toHaveText("Copied. Paste it into the new issue's Problem report field.");
  const copied = await page.evaluate(() => navigator.clipboard.readText());
  expect(copied).toMatch(/^Logogram: \d/);
  expect(copied).toContain("\nModel: tiny-gpt2, revision test, float32 on cpu, weights processed\n");
  expect(copied).toContain("\nLast error: Couldn't write <path>: permission denied.\n");
  expect(copied).toContain("\nProcessor: ");
  expect(copied).not.toContain("someone");

  // Closing it returns the focus to Help; opened again, it starts without the hardware.
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(helpButton(page)).toBeFocused();
  await page.keyboard.press("Enter");
  await page.getByRole("menuitem", { name: "Report a problem…" }).click();
  await expect(dialog.getByRole("checkbox", { name: "Include hardware details" })).not.toBeChecked();
  await expect(row(report, "Processor")).toHaveCount(0);
});

test("a report that can't be built says why, and how to get one without the app", async ({ page }) => {
  await page.route("**/api/report", (route) =>
    route.fulfill({ status: 500, contentType: "application/json", body: JSON.stringify({ error: "The server is busy." }) }),
  );
  await helpButton(page).click();
  await page.getByRole("menuitem", { name: "Report a problem…" }).click();
  const dialog = page.getByRole("dialog", { name: "Report a problem" });
  const alert = dialog.getByRole("alert");
  await expect(alert).toContainText("The report couldn't be built");
  await expect(alert).toContainText("The server is busy.");
  await expect(alert).toContainText("logogram doctor --report");
  await expect(dialog.getByRole("button", { name: "Copy the report" })).toBeDisabled();
  // The issue form is still a click away: the problem may be the report itself.
  await expect(dialog.getByRole("link", { name: "Open a new issue (opens in a new tab)" })).toHaveAttribute(
    "href",
    `${REPOSITORY}/issues/new?template=bug_report.yml`,
  );
});

test("the command palette opens the report and About too", async ({ page }) => {
  for (const [command, title] of [
    ["report a problem", "Report a problem"],
    ["about logogram", "About Logogram"],
  ]) {
    await page.keyboard.press("ControlOrMeta+k");
    await page.getByLabel("Command", { exact: true }).fill(command);
    await page.keyboard.press("Enter");
    const dialog = page.getByRole("dialog", { name: title });
    await expect(dialog).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(dialog).toBeHidden();
  }
});

test("About shows the version and license, and asks PyPI only when told to", async ({ page }) => {
  const state = await (await page.request.get("/api/state")).json();
  let checks = 0;
  await page.route("**/api/update/check", (route) => {
    checks += 1;
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ ...state.update, latest: "99.0.0", available: true, old: false, checked_at: new Date().toISOString(), error: null }),
    });
  });
  await helpButton(page).click();
  await page.getByRole("menuitem", { name: "About Logogram" }).click();
  const dialog = page.getByRole("dialog", { name: "About Logogram" });
  await expect(row(dialog, "Version")).toHaveText(`${state.version}, released ${state.update.released}`);
  const license = dialog.getByRole("link", { name: "MIT (opens in a new tab)" });
  await expect(license).toHaveAttribute("href", `${REPOSITORY}/blob/main/LICENSE`);
  await expect(license).toHaveAttribute("target", "_blank");
  await expect(dialog.getByRole("link", { name: "github.com/Jeevash23/logogram (opens in a new tab)" })).toHaveAttribute("href", REPOSITORY);
  await expect(dialog.getByRole("link", { name: /^What changed in each version/ })).toHaveAttribute("href", `${REPOSITORY}/blob/main/CHANGELOG.md`);
  await expect(dialog.getByRole("heading", { name: "Updates" })).toBeVisible();
  expect(checks).toBe(0);
  await dialog.getByRole("button", { name: "Check now" }).click();
  await expect(dialog.getByText("Logogram 99.0.0 is out")).toBeVisible();
  expect(checks).toBe(1);
  await page.keyboard.press("Escape");
  await expect(helpButton(page)).toBeFocused();
});

test("the top bar keeps Help within reach in an 800-pixel window", async ({ page }) => {
  await page.setViewportSize({ width: 800, height: 700 });
  const help = helpButton(page);
  await expect(help).toBeInViewport({ ratio: 1 });
  await help.click();
  const menu = page.getByRole("menu");
  await expect(menu).toBeInViewport({ ratio: 1 });
  await menu.getByRole("menuitem", { name: "Report a problem…" }).click();
  const dialog = page.getByRole("dialog", { name: "Report a problem" });
  await expect(dialog.getByRole("region", { name: "The report" })).toBeVisible();
  const box = await dialog.boundingBox();
  expect(box && box.x >= 0 && box.x + box.width <= 800).toBe(true);
  await expect(dialog.getByRole("button", { name: "Copy the report" })).toBeInViewport();
});
