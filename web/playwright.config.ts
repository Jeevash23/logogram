import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: "list",
  outputDir: process.env.LOGOGRAM_TEST_OUTPUT ?? "test-results",
  use: { baseURL: "http://127.0.0.1:8877", viewport: { width: 1440, height: 900 } },
  projects: [
    { name: "logic", testMatch: "**/*.unit.ts" },
    { name: "chromium", testMatch: "**/*.browser.ts", use: { browserName: "chromium" } },
  ],
  webServer: process.env.LOGOGRAM_BROWSER_SERVER === "external" || process.argv.includes("--project=logic") ? undefined : {
    command: "uv run --no-sync python ../tests/browser_server.py",
    url: "http://127.0.0.1:8877/api/state",
    timeout: 120_000,
    reuseExistingServer: false,
    env: { HF_HUB_OFFLINE: "1", HF_HUB_DISABLE_TELEMETRY: "1", OMP_NUM_THREADS: "1", MKL_NUM_THREADS: "1", LOGOGRAM_FIXTURE_SAE: "1" },
  },
});
