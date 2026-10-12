import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { CHANGELOG_URL, ERROR_SENT, errorForReport, GUIDE, guideUrl, LICENSE, LICENSE_URL, NEW_ISSUE_URL, REPOSITORY } from "../src/lib/help";
import { useStore } from "../src/store/app";
import { anchorsOf } from "./docs";

const read = (path: string) => readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf8");

test("the Help menu's documentation links name files and sections the repository has", () => {
  expect(GUIDE.length).toBeGreaterThan(0);
  for (const link of GUIDE) {
    const anchors = anchorsOf(link.path);
    if (link.anchor) expect(anchors, `${link.path} has no section #${link.anchor}`).toContain(link.anchor);
    const hash = link.anchor ? `#${link.anchor}` : "";
    expect(guideUrl(link)).toBe(link.path === "README.md" ? `${REPOSITORY}${hash}` : `${REPOSITORY}/blob/main/${link.path}${hash}`);
  }
  // The guide 0.4 added for remote GPUs, and troubleshooting, stay in the README.
  expect(GUIDE.filter((g) => g.path === "README.md").map((g) => g.anchor)).toEqual(["running-on-a-remote-gpu", "troubleshooting"]);
  expect(GUIDE.map((g) => g.path)).toContain("docs/getting-started.md");
});

test("the repository, license, changelog and issue form are the ones the repository has", () => {
  const pyproject = read("../../pyproject.toml");
  expect(pyproject).toContain(`Source = "${REPOSITORY}"`);
  expect(pyproject).toContain(`Changelog = "${CHANGELOG_URL}"`);
  expect(pyproject).toContain(`Issues = "${REPOSITORY}/issues"`);
  expect(pyproject).toMatch(new RegExp(`^license = "${LICENSE}"$`, "m"));
  expect(LICENSE_URL).toBe(`${REPOSITORY}/blob/main/LICENSE`);
  // The problem form exists, and the address names it and nothing else.
  expect(NEW_ISSUE_URL).toBe(`${REPOSITORY}/issues/new?template=bug_report.yml`);
  expect(read("../../.github/ISSUE_TEMPLATE/bug_report.yml")).toContain("id: report");
  // Every link leaves for the repository's own pages, over HTTPS.
  for (const url of [REPOSITORY, LICENSE_URL, CHANGELOG_URL, NEW_ISSUE_URL, ...GUIDE.map((g) => guideUrl(g))]) {
    expect(url.startsWith(`${REPOSITORY}`) && url.startsWith("https://github.com/")).toBe(true);
  }
});

test("errors the app shows are kept for a problem report, and nothing else is", () => {
  const store = useStore.getState();
  const original = globalThis.setTimeout;
  globalThis.setTimeout = (() => 0) as unknown as typeof setTimeout;
  try {
    store.leaveProject();
    useStore.setState({ lastError: null });
    store.notify("Saved datasets/ioi.jsonl. Check the baseline next.");
    expect(useStore.getState().lastError).toBeNull();
    store.notify("Couldn't reach Hugging Face for openai-community/gpt2.", "error");
    store.notify("Model loaded.");
    expect(useStore.getState().lastError).toBe("Couldn't reach Hugging Face for openai-community/gpt2.");
    // Dismissing the notice doesn't forget what it said; a newer error replaces it.
    for (const notice of useStore.getState().notices) store.dismiss(notice.id);
    expect(useStore.getState().lastError).toBe("Couldn't reach Hugging Face for openai-community/gpt2.");
    store.notify("The project changed in another tab.", "error");
    expect(useStore.getState().lastError).toBe("The project changed in another tab.");
    // Opening another project keeps it: the report is about this session.
    store.leaveProject();
    expect(useStore.getState().lastError).toBe("The project changed in another tab.");
  } finally {
    globalThis.setTimeout = original;
    useStore.setState({ lastError: null });
    store.leaveProject();
  }
});

test("a problem report asks for at most a few thousand characters of an error", () => {
  expect(errorForReport(null)).toBeNull();
  expect(errorForReport("  \n ")).toBeNull();
  expect(errorForReport("  Model failed to load.\n")).toBe("Model failed to load.");
  expect(errorForReport("x".repeat(10 * ERROR_SENT))).toHaveLength(ERROR_SENT);
});
