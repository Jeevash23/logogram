import { test, expect } from "@playwright/test";
import { formStorageKey, parseForm, parseStoredForm, projectKey, readStoredForm, serializeForm } from "../src/lib/formDraft";
import { DEFAULT_FORM, FORM_RESTORED, useStore, type FormState } from "../src/store/app";
import type { ProjectInfo, RunListing } from "../src/api/types";

/** Browser storage for the logic tests, which run outside a browser. */
class MemoryStorage {
  items = new Map<string, string>();
  get length() { return this.items.size; }
  key(i: number) { return [...this.items.keys()][i] ?? null; }
  getItem(key: string) { return this.items.get(key) ?? null; }
  setItem(key: string, value: string) { this.items.set(key, String(value)); }
  removeItem(key: string) { this.items.delete(key); }
  clear() { this.items.clear(); }
}

const project = (id: string): ProjectInfo => ({ session_id: id, name: id, path: `projects/${id}`, datasets: [], description: "" });

const DETAILED: FormState = {
  ...DEFAULT_FORM,
  name: "Which heads move names?",
  nameEdited: true,
  kind: "ablation",
  baseline: { kind: "resample", pool: "corrupt", donors: 4, seed: 3 },
  scope: { kind: "sites", sites: [{ kind: "head", layer: 9, head: 6, position: { kind: "index", index: -1 } }, { kind: "resid_post", layer: 3, head: null, position: { kind: "label", label: "S2" } }] },
  pathReceivers: [{ kind: "head", layer: 10, head: 7, input: "q" }, { kind: "logits" }],
  predictions: { method: "final_norm_logit_lens", prompt_index: 2, which: "clean", position: { kind: "index", index: 4 }, top_k: 5 },
  saeRef: { repo: "owner/sae", path: "layer_9", revision: null },
  limit: 12,
  prependBos: false,
  notes: "Expect the name movers to carry most of it.",
  modelRef: { id: "gpt2", revision: "abc", dtype: "float32", device: "auto", process_weights: true },
  savedDataset: { path: "datasets/ioi.jsonl", sha256: null },
  draftId: "2026-10-10-draft",
};

function withStorage(run: (storage: MemoryStorage) => Promise<void> | void) {
  return async () => {
    const storage = new MemoryStorage();
    Object.defineProperty(globalThis, "localStorage", { value: storage, configurable: true, writable: true });
    try {
      await run(storage);
    } finally {
      delete (globalThis as { localStorage?: unknown }).localStorage;
      useStore.getState().leaveProject();
    }
  };
}

/** The API as an empty project with the given runs. */
function stubApi(runs: Partial<RunListing>[] = []) {
  const original = globalThis.fetch;
  globalThis.fetch = (async (input: string) => new Response(JSON.stringify(input === "/api/runs" ? runs : []))) as typeof fetch;
  return () => { globalThis.fetch = original; };
}

test("a stored form reads back exactly", () => {
  const stored = parseStoredForm(serializeForm(DETAILED, 1234));
  expect(stored).toEqual({ form: DETAILED, savedAt: 1234 });
  expect(parseStoredForm(serializeForm(DEFAULT_FORM, 1))?.form).toEqual(DEFAULT_FORM);
  // Storage keys name the project folder without spelling out its path.
  expect(formStorageKey("projects/one")).toBe(`logogram.experimentForm.${projectKey("projects/one")}`);
  expect(formStorageKey("projects/one")).not.toContain("projects");
  expect(projectKey("projects/one")).not.toBe(projectKey("projects/two"));
});

test("anything that isn't exactly a form is ignored", () => {
  const good = JSON.parse(serializeForm(DETAILED, 1));
  const variant = (change: (data: Record<string, any>) => void) => {
    const data = structuredClone(good);
    change(data);
    return JSON.stringify(data);
  };
  for (const text of [
    null,
    "",
    "not json",
    "[]",
    "null",
    JSON.stringify({ ...good, version: 2 }),
    JSON.stringify({ ...good, savedAt: "yesterday" }),
    variant((d) => { delete d.form.notes; }),
    variant((d) => { d.form.kind = "telepathy"; }),
    variant((d) => { d.form.bootstrap = "1000"; }),
    variant((d) => { d.form.limit = "12"; }),
    variant((d) => { d.form.scope = { kind: "sites", sites: [{ kind: "head", layer: "9", position: { kind: "all" } }] }; }),
    variant((d) => { d.form.scope = { kind: "heads", position: { kind: "middle" } }; }),
    variant((d) => { d.form.baseline = { kind: "resample", pool: "corrupt" }; }),
    variant((d) => { d.form.pathReceivers = [{ kind: "head", layer: 1, head: 2, input: "z" }]; }),
    variant((d) => { d.form.modelRef = { id: 7 }; }),
    variant((d) => { d.form = "a form"; }),
  ]) {
    expect(parseStoredForm(text), String(text).slice(0, 80)).toBeNull();
  }
  expect(parseForm({ ...DETAILED, extra: "ignored" })).toEqual(DETAILED);
});

test("the unsaved form is kept per project and comes back, with Undo, when the project opens", withStorage(async (storage) => {
  const restore = stubApi();
  try {
    const store = useStore.getState();
    store.leaveProject();
    await store.enterProject(project("one"));
    // An empty form is nothing to keep.
    expect(storage.length).toBe(0);
    store.setForm({ notes: "Check the duplicate token heads." });
    store.setForm({ kind: "ablation", baseline: { kind: "zero" } });
    expect(readStoredForm("projects/one")?.form.notes).toBe("Check the duplicate token heads.");
    expect(readStoredForm("projects/two")).toBeNull();

    // Leaving keeps it; opening the project again (or reloading) puts it back.
    store.leaveProject();
    expect(readStoredForm("projects/one")).not.toBeNull();
    await store.enterProject(project("one"));
    const state = useStore.getState();
    expect(state.form.notes).toBe("Check the duplicate token heads.");
    expect(state.form.baseline).toEqual({ kind: "zero" });
    const notice = state.notices.find((n) => n.text === FORM_RESTORED);
    expect(notice?.action?.label).toBe("Undo");

    // Undo returns to the empty form and forgets the stored copy.
    notice?.action?.run();
    expect(useStore.getState().form).toEqual(DEFAULT_FORM);
    expect(readStoredForm("projects/one")).toBeNull();
  } finally { restore(); }
}));

test("saving the form as a draft or running it clears the stored copy", withStorage(async () => {
  const restore = stubApi();
  try {
    const store = useStore.getState();
    store.leaveProject();
    await store.enterProject(project("two"));
    store.setForm({ statSeed: 11 });
    expect(readStoredForm("projects/two")?.form.statSeed).toBe(11);
    store.markFormSaved("draft-1");
    expect(useStore.getState().form.draftId).toBe("draft-1");
    expect(readStoredForm("projects/two")).toBeNull();
    // A change after saving is unsaved again.
    store.setForm({ statSeed: 12 });
    expect(readStoredForm("projects/two")?.form).toMatchObject({ statSeed: 12, draftId: "draft-1" });
  } finally { restore(); }
}));

test("a stored form that doesn't parse is ignored, and a draft that has run is not refilled", withStorage(async (storage) => {
  // The draft ran (and failed) since: its folder holds a run now.
  const restore = stubApi([{ id: "2026-10-10-draft", status: "failed", name: "Ran", created: "", kind: "ablation" } as Partial<RunListing>]);
  try {
    const store = useStore.getState();
    store.leaveProject();
    storage.setItem(formStorageKey("projects/broken"), "{\"version\":1,\"savedAt\":1,\"form\":{\"kind\":\"ablation\"}}");
    await store.enterProject(project("broken"));
    expect(useStore.getState().form).toEqual(DEFAULT_FORM);
    expect(useStore.getState().notices).toEqual([]);

    store.leaveProject();
    storage.setItem(formStorageKey("projects/ran"), serializeForm(DETAILED, 1));
    await store.enterProject(project("ran"));
    expect(useStore.getState().form).toEqual({ ...DETAILED, draftId: null });
  } finally { restore(); }
}));
