import { test, expect } from "@playwright/test";
import { NOTICE_MS, useStore } from "../src/store/app";

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
