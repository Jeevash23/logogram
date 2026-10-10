import { test, expect } from "@playwright/test";
import type { Job } from "../src/api/types";
import { milestone } from "../src/lib/milestones";
import { NOTICE_MS, useStore } from "../src/store/app";

test("the status line speaks at milestones, not at every progress tick", () => {
  const job: Job = { id: "j1", kind: "run", title: "Heads sweep", status: "running", run_id: "r1", progress: {}, error: null, started: 0, version: 1 };
  const said = new Set<string>();
  const keys: string[] = [];
  for (let done = 0; done <= 1000; done += 7) {
    const m = milestone(job, { done, total: 1000 }, "connected");
    if (!keys.includes(m.key)) { keys.push(m.key); said.add(m.text); }
  }
  expect([...said]).toEqual(["Started “Heads sweep”.", "“Heads sweep” is 25% done.", "“Heads sweep” is 50% done.", "“Heads sweep” is 75% done."]);
  expect(milestone({ ...job, status: "finished" }, null, "connected").text).toBe("“Heads sweep” finished.");
  expect(milestone({ ...job, status: "cancelled" }, null, "connected").text).toBe("“Heads sweep” was cancelled.");
  expect(milestone({ ...job, cancelling: true }, { done: 600, total: 1000 }, "connected").text).toBe("Cancelling “Heads sweep”.");
  expect(milestone(job, { done: 600, total: 1000 }, "reconnecting").key).toBe("reconnecting");
  expect(milestone(null, null, "connected").text).toBe("");
});

test("notices that carry an action or important news stay until dismissed", () => {
  const store = useStore.getState();
  store.leaveProject();
  const timers: { run: () => void; ms: number }[] = [];
  const original = globalThis.setTimeout;
  globalThis.setTimeout = ((run: () => void, ms: number) => {
    timers.push({ run, ms });
    return 0;
  }) as unknown as typeof setTimeout;
  try {
    store.notify("Methods copied.");
    store.notify("Run cancelled. Partial results were discarded.", "info", { persist: true });
    store.notify("Couldn't reach the server.", "error");
    let undone = 0;
    store.notify("Experiment form replaced", "info", { key: "form", action: { label: "Undo", run: () => (undone += 1) } });
    // Only the plain confirmation closes by itself.
    expect(timers.map((t) => t.ms)).toEqual([NOTICE_MS]);
    timers[0].run();
    expect(useStore.getState().notices.map((n) => n.text)).toEqual([
      "Run cancelled. Partial results were discarded.",
      "Couldn't reach the server.",
      "Experiment form replaced",
    ]);
    // A newer notice with the same key replaces the older one rather than piling up.
    store.notify("Experiment form replaced", "info", { key: "form", action: { label: "Undo", run: () => (undone += 10) } });
    const keyed = useStore.getState().notices.filter((n) => n.key === "form");
    expect(keyed).toHaveLength(1);
    keyed[0].action?.run();
    expect(undone).toBe(10);
  } finally {
    globalThis.setTimeout = original;
    store.leaveProject();
  }
});
