import { test, expect } from "@playwright/test";
import { SESSION_ENDED_TEXT } from "../src/api/client";
import { connectEvents, SESSION_ENDED } from "../src/api/events";
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
  expect(milestone(job, { done: 600, total: 1000 }, "ended").text).toBe(SESSION_ENDED_TEXT);
  expect(milestone(null, null, "connected").text).toBe("");
});

test("a stream closed because the session ended says so and stops reconnecting", () => {
  const sockets: FakeSocket[] = [];
  class FakeSocket {
    onopen: (() => void) | null = null;
    onmessage: ((m: { data: string }) => void) | null = null;
    onclose: ((e: { code: number }) => void) | null = null;
    constructor(public url: string) { sockets.push(this); }
    close() {}
  }
  const timers: number[] = [];
  const saved = { WebSocket: globalThis.WebSocket, location: (globalThis as { location?: unknown }).location, window: (globalThis as { window?: unknown }).window };
  Object.assign(globalThis, {
    WebSocket: FakeSocket,
    location: { protocol: "http:", host: "127.0.0.1:8765" },
    window: { setTimeout: (_: () => void, ms: number) => { timers.push(ms); return 1; }, clearTimeout: () => {} },
  });
  const store = useStore.getState();
  store.leaveProject();
  try {
    const stop = connectEvents();
    expect(sockets).toHaveLength(1);
    // An ordinary drop is retried...
    sockets[0].onclose?.({ code: 1006 });
    expect(useStore.getState().connection).toBe("reconnecting");
    expect(timers).toHaveLength(1);
    // ...an ended session isn't: the page says what happened and how to go on.
    sockets[0].onclose?.({ code: SESSION_ENDED });
    expect(timers).toHaveLength(1);
    expect(useStore.getState().connection).toBe("ended");
    expect(useStore.getState().connectionError).toBe(SESSION_ENDED_TEXT);
    expect(useStore.getState().notices.some((n) => n.tone === "error" && n.text === SESSION_ENDED_TEXT)).toBe(true);
    stop();
  } finally {
    Object.assign(globalThis, saved);
    useStore.setState({ connection: "connecting", connectionError: null });
    store.leaveProject();
  }
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
