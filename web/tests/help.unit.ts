import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { CHANGELOG_URL, ERROR_SENT, errorForReport, GUIDE, guideUrl, LICENSE, LICENSE_URL, NEW_ISSUE_URL, REPOSITORY } from "../src/lib/help";
import { useStore } from "../src/store/app";

const read = (path: string) => readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf8");

/** The anchor GitHub gives a heading: lower case, punctuation dropped, spaces as hyphens. */
function slug(heading: string): string {
  return heading
    .trim()
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\- _]/gu, "")
    .replace(/ /g, "-");
}

test("the Help menu's guide links name sections the README has", () => {
  const anchors = [...read("../../README.md").matchAll(/^#{1,6} (.+)$/gm)].map((m) => slug(m[1]));
  expect(GUIDE.length).toBeGreaterThan(0);
  for (const section of GUIDE) {
    expect(anchors, `README has no section #${section.anchor}`).toContain(section.anchor);
    expect(slug(section.title)).toBe(section.anchor);
    expect(guideUrl(section.anchor)).toBe(`${REPOSITORY}#${section.anchor}`);
  }
  expect(GUIDE.map((g) => g.anchor)).toEqual(expect.arrayContaining(["running-on-a-remote-gpu", "troubleshooting"]));
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
  for (const url of [REPOSITORY, LICENSE_URL, CHANGELOG_URL, NEW_ISSUE_URL, ...GUIDE.map((g) => guideUrl(g.anchor))]) {
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
