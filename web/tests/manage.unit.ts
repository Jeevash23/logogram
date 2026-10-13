import { test, expect } from "@playwright/test";

import type { DatasetListing, ProjectInfo } from "../src/api/types";
import { useStore } from "../src/store/app";

// A dataset renamed or deleted, in this tab or another: the prompts chosen, and their unsaved
// edits, follow it.

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

test("a tab that has left its project ignores the news that it closed", () => {
  const store = useStore.getState();
  store.leaveProject();
  store.notify("Deleted “One” and its folder.");
  store.handleEvent({ type: "project", session_id: null });
  // Leaving again would clear the notice that says so.
  expect(useStore.getState().notices.map((n) => n.text)).toContain("Deleted “One” and its folder.");
});
