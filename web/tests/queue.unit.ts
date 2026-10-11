import { test, expect } from "@playwright/test";

import type { Job, ProjectInfo, QueuedRun, Spec } from "../src/api/types";
import { useStore } from "../src/store/app";

const project = (id: string): ProjectInfo => ({ session_id: id, name: id, path: `projects/${id}`, datasets: [], description: "" });

/** Only the spec's name matters to the queue in the app. */
const spec = (name: string) => ({ name }) as unknown as Spec;

function queued(id: string, title: string, session: string): QueuedRun {
  return { id, title, spec: spec(title), draft_id: null, derived_from: null, project_session: session, queued_at: 0 };
}

const running = (title: string, runId: string, session: string): Job => ({
  id: `job-${runId}`, kind: "run", title, status: "running", run_id: runId, progress: {}, error: null, started: 0, version: 1, project_session: session,
});

const titles = () => useStore.getState().queue.map((q) => q.title);
const lastNotice = () => useStore.getState().notices.at(-1);

/** The API as an empty project; `routes` answer the requests named in them (method and path),
 * with JSON or a whole response. */
function stubApi(routes: Record<string, (body: unknown) => unknown> = {}) {
  const original = globalThis.fetch;
  globalThis.fetch = (async (input: string, init?: RequestInit) => {
    const answer = routes[`${init?.method ?? "GET"} ${input}`];
    if (answer) {
      const out = answer(init?.body ? JSON.parse(String(init.body)) : undefined);
      return out instanceof Response ? out : new Response(JSON.stringify(out));
    }
    if (input === "/api/runs") return new Response("[]");
    return new Response(JSON.stringify({ error: "Not part of this test." }), { status: 404 });
  }) as typeof fetch;
  return () => { globalThis.fetch = original; };
}

const conflict = (error: string) => new Response(JSON.stringify({ error }), { status: 409 });

test("the queue follows the server's events, for the open project only", async () => {
  const restore = stubApi();
  const store = useStore.getState();
  try {
    store.leaveProject();
    // The server's state, read before the project opened in this tab: only its runs stay.
    useStore.setState({ queue: [queued("a", "Heads sweep", "one"), queued("b", "Elsewhere", "two")] });
    await store.enterProject(project("one"));
    expect(titles()).toEqual(["Heads sweep"]);
    // Every change arrives with the whole queue...
    store.handleEvent({ type: "queue", project_session: "one", runs: [queued("a", "Heads sweep", "one"), queued("c", "Ablation", "one")] });
    expect(titles()).toEqual(["Heads sweep", "Ablation"]);
    // ...and another project's queue isn't this tab's.
    store.handleEvent({ type: "queue", project_session: "two", runs: [] });
    expect(titles()).toEqual(["Heads sweep", "Ablation"]);
    // A run that couldn't start when its turn came says why, until dismissed.
    store.handleEvent({ type: "queue.skipped", project_session: "one", id: "a", title: "Heads sweep", error: "gone.jsonl isn't a regular file inside the project." });
    expect(lastNotice()).toMatchObject({
      tone: "error",
      text: "“Heads sweep” couldn't start and was taken out of the queue. gone.jsonl isn't a regular file inside the project.",
    });
    store.handleEvent({ type: "queue.skipped", project_session: "two", id: "b", title: "Elsewhere", error: "Not here." });
    expect(lastNotice()?.text).not.toContain("Elsewhere");
    store.leaveProject();
    expect(useStore.getState().queue).toEqual([]);
  } finally {
    restore();
    store.leaveProject();
  }
});

test("queueing the form says what it waits behind, and leaves a draft's folder to the run", async () => {
  const sent: unknown[] = [];
  const replies: unknown[] = [];
  const restore = stubApi({
    "POST /api/queue": (body) => {
      sent.push(body);
      return replies.shift();
    },
  });
  const store = useStore.getState();
  try {
    store.leaveProject();
    await store.enterProject(project("one"));
    useStore.setState({ job: running("Heads sweep", "r1", "one") });
    store.markFormSaved("draft-1");

    replies.push({ run_id: null, job: null, queued: queued("q1", "Ablation", "one"), queue: [queued("q1", "Ablation", "one")] });
    await store.queueRun(spec("Ablation"));
    expect(sent.at(-1)).toEqual({ spec: spec("Ablation"), draft_id: "draft-1" });
    expect(lastNotice()?.text).toBe("Queued “Ablation”. It starts after “Heads sweep”.");
    // The draft's folder is the queued run's: the form no longer fills it, and stays open.
    expect(useStore.getState().form.draftId).toBeNull();
    expect(useStore.getState().view).not.toBe("results");

    const both = [queued("q1", "Ablation", "one"), queued("q2", "Second look", "one")];
    replies.push({ run_id: null, job: null, queued: both[1], queue: both });
    await store.queueRun(spec("Second look"));
    expect(sent.at(-1)).toEqual({ spec: spec("Second look"), draft_id: null });
    expect(lastNotice()?.text).toBe("Queued “Second look”. It starts after “Ablation”.");

    // When everything ahead had ended meanwhile, it started at once, and shows as Run would.
    replies.push({ run_id: "r2", job: running("Third", "r2", "one"), queued: null, queue: [] });
    await store.queueRun(spec("Third"));
    expect(useStore.getState()).toMatchObject({ activeRunId: "r2", view: "results", job: { run_id: "r2" } });
  } finally {
    restore();
    store.leaveProject();
  }
});

test("a removed run leaves the queue at once, and the notice names it", async () => {
  const started = "That run isn't queued any more: it has started or was removed.";
  const restore = stubApi({
    "DELETE /api/queue/q1": () => [queued("q2", "Second look", "one")],
    "DELETE /api/queue/q2": () => conflict(started),
  });
  const store = useStore.getState();
  try {
    store.leaveProject();
    await store.enterProject(project("one"));
    useStore.setState({ queue: [queued("q1", "Ablation", "one"), queued("q2", "Second look", "one")] });
    await store.removeQueuedRun("q1");
    expect(titles()).toEqual(["Second look"]);
    expect(lastNotice()?.text).toBe("Removed “Ablation” from the queue.");
    // A run that started meanwhile can't be removed: the server says so, and the queue event
    // that took it out is on its way.
    await store.removeQueuedRun("q2");
    expect(titles()).toEqual(["Second look"]);
    expect(lastNotice()).toMatchObject({ tone: "error", text: started });
  } finally {
    restore();
    store.leaveProject();
  }
});
