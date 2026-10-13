import { test, expect } from "@playwright/test";

import type { DatasetListing, ProjectInfo } from "../src/api/types";
import { removeQueuedText } from "../src/lib/runs";
import { reproductionNotice, useStore } from "../src/store/app";

// A dataset renamed or deleted, in this tab or another: the prompts chosen, and their unsaved
// edits, follow it. A rerun says whether it reproduced the run it reran.

const dataset = (name: string): DatasetListing => ({ name, path: `datasets/${name}`, modified: null, n: 4 });
const project = (datasets: string[]): ProjectInfo => ({
  session_id: "one",
  name: "One",
  path: "projects/one",
  description: "",
  datasets: datasets.map(dataset),
});

/** The API: the project as `current()` says, and any dataset's detail. */
function stubApi(current: () => ProjectInfo) {
  const original = globalThis.fetch;
  const seen: string[] = [];
  globalThis.fetch = (async (input: string, init?: RequestInit) => {
    seen.push(`${init?.method ?? "GET"} ${input}`);
    if (input === "/api/project") return new Response(JSON.stringify(current()));
    if (input.startsWith("/api/dataset?")) {
      const path = decodeURIComponent(/path=([^&]+)/.exec(input)?.[1] ?? "");
      return new Response(JSON.stringify({ name: path.split("/").pop(), path, sha256: "0", n: 4, records: [] }));
    }
    if (input === "/api/runs") return new Response("[]");
    return new Response(JSON.stringify({ error: "Not part of this test." }), { status: 404 });
  }) as typeof fetch;
  return { seen, restore: () => void (globalThis.fetch = original) };
}

const edits = { sha256: "0", removed: [1], replaced: {} } as never;

test("the prompts chosen follow a dataset that is renamed, and their edits go with it", async () => {
  let info = project(["first.jsonl", "second.jsonl"]);
  const api = stubApi(() => info);
  const store = useStore.getState();
  try {
    store.leaveProject();
    await store.enterProject(info);
    expect(useStore.getState().datasetPath).toBe("datasets/first.jsonl");
    useStore.setState({ datasetEdits: { "datasets/first.jsonl": edits } });

    // Renamed in another tab: the event says from what to what.
    info = project(["renamed.jsonl", "second.jsonl"]);
    store.handleEvent({ type: "project.changed", project_session: "one", dataset: { path: "datasets/first.jsonl", to: "datasets/renamed.jsonl" } });
    await expect.poll(() => useStore.getState().datasetPath).toBe("datasets/renamed.jsonl");
    let st = useStore.getState();
    expect(st.project?.datasets.map((d) => d.name)).toEqual(["renamed.jsonl", "second.jsonl"]);
    expect(st.datasetEdits).toEqual({ "datasets/renamed.jsonl": edits });

    // Heard twice (this tab renamed it, and the event came too): nothing more happens.
    store.followDataset("datasets/first.jsonl", "datasets/renamed.jsonl");
    expect(useStore.getState().datasetPath).toBe("datasets/renamed.jsonl");

    // Deleted: the next dataset is chosen, and its edits go with it.
    info = project(["second.jsonl"]);
    store.handleEvent({ type: "project.changed", project_session: "one", dataset: { path: "datasets/renamed.jsonl", to: null } });
    await expect.poll(() => useStore.getState().datasetPath).toBe("datasets/second.jsonl");
    st = useStore.getState();
    expect(st.datasetEdits).toEqual({});
    expect(st.project?.datasets.map((d) => d.name)).toEqual(["second.jsonl"]);

    // Another dataset changed: the one chosen stays. The project renamed: its name changes.
    info = { ...project(["second.jsonl", "third.jsonl"]), name: "Renamed" };
    store.handleEvent({ type: "project.changed", project_session: "one", dataset: { path: "datasets/fourth.jsonl", to: null } });
    await expect.poll(() => useStore.getState().project?.name).toBe("Renamed");
    expect(useStore.getState().datasetPath).toBe("datasets/second.jsonl");

    // The last one deleted: nothing is chosen.
    info = project([]);
    store.handleEvent({ type: "project.changed", project_session: "one", dataset: { path: "datasets/second.jsonl", to: null } });
    await expect.poll(() => useStore.getState().datasetPath).toBeNull();
    // Another project's change isn't this tab's.
    const before = api.seen.length;
    store.handleEvent({ type: "project.changed", project_session: "two", dataset: null });
    expect(api.seen.length).toBe(before);
  } finally {
    api.restore();
    store.leaveProject();
  }
});

test("a finished rerun says whether it reproduced its run, and a difference offers the comparison", async () => {
  const same = {
    run_id: "b",
    original: "a",
    identical: true,
    rows: 160,
    original_rows: 160,
    largest: 0,
    text: "Identical to a: all 160 per-prompt values match exactly.",
  };
  expect(reproductionNotice(same, "a")).toEqual(["Rerun finished. Identical to a: all 160 per-prompt values match exactly.", "info", {}]);
  const differs = { ...same, identical: false, largest: 0.5, text: "Differs from a: largest change in a per-prompt value is 0.5." };
  const [text, tone, options] = reproductionNotice(differs, "a");
  expect([text, tone, options.persist]).toEqual(["Rerun finished. Differs from a: largest change in a per-prompt value is 0.5.", "info", true]);
  options.action?.run();
  expect(useStore.getState()).toMatchObject({ compareIds: ["a", "b"], view: "compare" });

  // When the rerun's run ends, the tab asks the server and says so.
  const info = project([]);
  const original = globalThis.fetch;
  globalThis.fetch = (async (input: string) => {
    if (input === "/api/project") return new Response(JSON.stringify(info));
    if (input === "/api/runs") return new Response("[]");
    if (input === "/api/runs/b") {
      return new Response(JSON.stringify({ id: "b", spec: null, summary: null, manifest: { derived_from: { run: "a", kind: "rerun", change: null } }, listing: null }));
    }
    if (input === "/api/runs/b/reproduction") return new Response(JSON.stringify(same));
    return new Response(JSON.stringify({ error: "Not part of this test." }), { status: 404 });
  }) as typeof fetch;
  const store = useStore.getState();
  try {
    store.leaveProject();
    await store.enterProject(info);
    store.handleEvent({ type: "run.finished", run_id: "b", project_session: "one" });
    await expect.poll(() => useStore.getState().notices.map((n) => n.text)).toContain("Rerun finished. Identical to a: all 160 per-prompt values match exactly.");
    // A replayed end (a tab that connects later) says nothing again.
    useStore.setState({ notices: [] });
    store.handleEvent({ type: "run.finished", run_id: "b", project_session: "one", replay: true });
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(useStore.getState().notices).toEqual([]);
  } finally {
    globalThis.fetch = original;
    store.leaveProject();
  }
});

test("taking a run out of the queue says what is lost first", () => {
  expect(removeQueuedText({ draft_id: null })).toBe(
    "It won't run, and its spec isn't kept anywhere else: queue it again from the form to run it later.",
  );
  expect(removeQueuedText({ draft_id: "20260101-000000-draft" })).toBe(
    "It won't run from the queue. It is a saved draft, so it stays in the history as it is, to run or change later.",
  );
});

test("a tab that has left its project ignores the news that it closed", () => {
  const store = useStore.getState();
  store.leaveProject();
  store.notify("Deleted “One” and its folder.");
  store.handleEvent({ type: "project", session_id: null });
  // Leaving again would clear the notice that says so.
  expect(useStore.getState().notices.map((n) => n.text)).toContain("Deleted “One” and its folder.");
});
