import { defineConfig } from "@playwright/test";

// LOGOGRAM_BROWSER_PORT moves the fixture server off 8877, so two checkouts can run the checks at once.
const BASE = `http://127.0.0.1:${process.env.LOGOGRAM_BROWSER_PORT ?? "8877"}`;

export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: "list",
  outputDir: process.env.LOGOGRAM_TEST_OUTPUT ?? "test-results",
  // LOGOGRAM_COLOR_SCHEME=dark runs every browser check in the dark theme (CI does both).
  use: {
    baseURL: BASE,
    viewport: { width: 1440, height: 900 },
    colorScheme: process.env.LOGOGRAM_COLOR_SCHEME === "dark" ? "dark" : "light",
  },
  projects: [
    { name: "logic", testMatch: "**/*.unit.ts" },
    { name: "chromium", testMatch: "**/*.browser.ts", use: { browserName: "chromium" } },
  ],
  webServer: process.env.LOGOGRAM_BROWSER_SERVER === "external" || process.argv.includes("--project=logic") ? undefined : {
    command: "uv run --no-sync python ../tests/browser_server.py",
    url: `${BASE}/api/state`,
    timeout: 120_000,
    reuseExistingServer: false,
    env: { HF_HUB_OFFLINE: "1", HF_HUB_DISABLE_TELEMETRY: "1", OMP_NUM_THREADS: "1", MKL_NUM_THREADS: "1", LOGOGRAM_FIXTURE_SAE: "1" },
  },
});
