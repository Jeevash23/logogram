import { test, expect } from "@playwright/test";
import { analysisContext } from "../src/lib/analysis";
import { moveCell } from "../src/lib/heatmapNavigation";
import { DEFAULT_FORM, useStore } from "../src/store/app";
import type { ProjectInfo, Spec } from "../src/api/types";

const project = (id: string): ProjectInfo => ({ session_id: id, name: id, path: id, datasets: [], description: "" });
const spec: Spec = {
  logogram_spec: 2, name: "Recorded run", notes: "", model: { id: "tiny", revision: "fixed", device: "cpu", dtype: "float32", process_weights: false },
  dataset: { path: "datasets/snapshots/fixed.jsonl", sha256: "fixed", limit: 3 }, tokenization: { prepend_bos: false },
  experiment: { kind: "activation_patching", direction: "clean_to_corrupt" }, scope: { kind: "heads", position: { kind: "all" } },
  metric: { kind: "logit_diff", normalization: "dataset_gap" }, statistics: { bootstrap: 200, seed: 7, ci: 0.95, cluster: null }, execution: { batch_size: 2 },
};

test("saved analyses retain BOS, prompt limit, batch and pinned model settings", () => {
  const state = { ...useStore.getInitialState(), project: project("one"), activeRunId: "run", analysisSource: "run" as const,
    runDetails: { run: { spec } }, datasetPath: spec.dataset.path };
  const context = analysisContext(state);
  expect(context.options).toEqual({ model: spec.model, dataset_sha256: "fixed", prepend_bos: false, limit: 3, batch_size: 2 });
  const edited = analysisContext({ ...state, analysisSource: "form", form: { ...DEFAULT_FORM, prependBos: true, limit: 7, batchSize: 8 } });
  expect(edited.key).not.toBe(context.key);
  expect(edited.options.prepend_bos).toBe(true);
  expect(analysisContext({ ...state, project: project("two") }).key).not.toBe(context.key);
});

test("a view that shows a run reads prompts as that run did, after the form was configured", async () => {
  const original = globalThis.fetch;
  const requested: string[] = [];
  globalThis.fetch = (async (input: string) => {
    requested.push(input);
    const data = input.startsWith("/api/dataset?") ? { name: "fixed.jsonl", path: spec.dataset.path, n: 8, sha256: "fixed", records: [] } : [];
    return new Response(JSON.stringify(data));
  }) as typeof fetch;
  try {
    const store = useStore.getState();
    store.leaveProject();
    const detail = { id: "run", spec, summary: null, manifest: null, listing: null, folder: "experiments/run" };
    useStore.setState({ project: project("context"), screen: "workbench", activeRunId: "run", analysisSource: "run", runDetails: { run: detail },
      datasetPath: spec.dataset.path, dataset: { name: "fixed.jsonl", path: spec.dataset.path, n: 8, sha256: "fixed", records: [] } as never });
    const state = () => useStore.getState();
    store.setView("results");
    expect(analysisContext(state()).options.prepend_bos).toBe(false);
    expect(analysisContext(state()).n).toBe(3);

    // Configure works on the form (BOS on, every prompt)...
    store.setView("experiment");
    expect(analysisContext(state()).options.prepend_bos).toBe(true);
    expect(analysisContext(state()).n).toBe(8);
    store.setPromptIndex(6);
    // ...and going back to the run's results reads its prompts again, as the page shows it.
    store.setView("results");
    expect(state().analysisSource).toBe("run");
    expect(analysisContext(state()).options.prepend_bos).toBe(false);
    expect(analysisContext(state()).label).toBe("Selected run · BOS off · first 3 prompts · batch 2");
    expect(state().promptIndex).toBe(2);

    // Attention opened from the results follows the run; the baseline after configuring uses the form.
    store.setView("attention");
    expect(analysisContext(state()).options.prepend_bos).toBe(false);
    store.setView("experiment");
    store.setView("baseline");
    expect(analysisContext(state()).options.prepend_bos).toBe(true);
    // The model map follows the run only while it paints the run's results.
    store.setMapOverlay("structure");
    store.setView("explore");
    expect(analysisContext(state()).options.prepend_bos).toBe(true);
    store.setMapOverlay("data");
    expect(analysisContext(state()).options.prepend_bos).toBe(false);

    // A run on other prompts brings its own dataset back with it.
    store.setView("experiment");
    useStore.setState({ datasetPath: "datasets/other.jsonl" });
    store.setView("results");
    expect(state().datasetPath).toBe(spec.dataset.path);
    expect(requested.some((url) => url.startsWith("/api/dataset?") && url.includes("prepend_bos=false"))).toBe(true);
  } finally { globalThis.fetch = original; useStore.getState().leaveProject(); }
});

test("heatmap keyboard movement skips masked cells and respects boundaries", () => {
  const exists = (r: number, c: number) => c <= r && c !== 1;
  expect(moveCell({ r: 3, c: 0 }, "ArrowRight", 4, 4, exists)).toEqual({ r: 3, c: 2 });
  expect(moveCell({ r: 2, c: 2 }, "ArrowUp", 4, 4, exists)).toEqual({ r: 2, c: 2 });
  expect(moveCell({ r: 3, c: 2 }, "Home", 4, 4, exists)).toEqual({ r: 3, c: 0 });
  expect(moveCell({ r: 3, c: 0 }, "End", 4, 4, exists)).toEqual({ r: 3, c: 3 });
});

test("a late dataset response cannot repopulate a different project", async () => {
  const original = globalThis.fetch;
  let release: (value: Response) => void = () => {};
  globalThis.fetch = (async (input: string) => input.startsWith("/api/dataset?")
    ? await new Promise<Response>((resolve) => { release = resolve; })
    : new Response("[]", { headers: { "content-type": "application/json" } })) as typeof fetch;
  try {
    const store = useStore.getState();
    store.leaveProject();
    await store.enterProject(project("one"));
    const loading = store.selectDataset("datasets/ioi.jsonl");
    await store.enterProject(project("two"));
    release(new Response(JSON.stringify({ name: "ioi.jsonl", path: "datasets/ioi.jsonl", n: 1, sha256: "one", records: [] })));
    await loading;
    expect(useStore.getState().project?.session_id).toBe("two");
    expect(useStore.getState().dataset).toBeNull();
    expect(useStore.getState().datasetPath).toBeNull();
    expect(useStore.getState().form).toEqual(DEFAULT_FORM);
  } finally { globalThis.fetch = original; useStore.getState().leaveProject(); }
});

test("a newly finished run follows its immutable snapshot without reopening history", async () => {
  const original = globalThis.fetch;
  globalThis.fetch = (async (input: string) => {
    const data = input === "/api/runs/run" ? { id: "run", spec, summary: null, manifest: null, listing: null, folder: "experiments/run" }
      : input.startsWith("/api/dataset?") ? { name: "fixed.jsonl", path: spec.dataset.path, n: 5, sha256: "fixed", records: [] } : [];
    return new Response(JSON.stringify(data));
  }) as typeof fetch;
  try {
    const store = useStore.getState();
    store.leaveProject();
    await store.enterProject(project("snapshot"));
    useStore.setState({ activeRunId: "run", analysisSource: "run", datasetPath: "datasets/ioi.jsonl" });
    await store.loadRun("run", true);
    expect(useStore.getState().datasetPath).toBe(spec.dataset.path);
    expect(analysisContext(useStore.getState()).options.prepend_bos).toBe(false);
    expect(analysisContext(useStore.getState()).n).toBe(3);
  } finally { globalThis.fetch = original; useStore.getState().leaveProject(); }
});

test("staging preserves the selected component, token position, and scientific settings", () => {
  const store = useStore.getState();
  store.leaveProject();
  useStore.setState({ project: project("staging"), activeRunId: "run", analysisSource: "run", tokenPosition: 9,
    runDetails: { run: { id: "run", spec, summary: null, manifest: null, listing: null, folder: "experiments/run" } } });
  const component = { layer: 1, part: "resid" as const, kind: "resid_post" as const };
  store.stageSelection(component);
  store.stageSelection(component);
  expect(useStore.getState().stagedSites).toEqual([{ kind: "resid_post", layer: 1, head: null, position: { kind: "index", index: 9 } }]);
  store.configureStaged();
  const state = useStore.getState();
  expect(state.form.scope).toEqual({ kind: "sites", sites: [{ kind: "resid_post", layer: 1, head: null, position: { kind: "index", index: 9 } }] });
  expect(state.form.prependBos).toBe(false);
  expect(state.form.batchSize).toBe(2);
  expect(state.form.limit).toBe(3);
  expect(state.form.statSeed).toBe(7);
  expect(state.form.modelRef).toEqual(spec.model);
  expect(state.stagedSites).toEqual([]);
  expect(state.view).toBe("experiment");
  store.leaveProject();
});

test("project boundaries clear saved selection drafts and pinned heads", () => {
  const store = useStore.getState();
  useStore.setState({ project: project("notes"), activeRunId: "run", analysisSource: "run",
    runDetails: { run: { id: "run", spec, summary: null, manifest: null, listing: null, folder: "experiments/run" } } });
  store.pinHead({ layer: 0, part: "head", head: 1 });
  store.prepareNote([{ kind: "head", layer: 0, head: 1, position: { kind: "last" } }]);
  expect(useStore.getState().noteEditor?.value.run_id).toBe("run");
  expect(useStore.getState().headPins).toHaveLength(1);
  store.leaveProject();
  expect(useStore.getState().noteEditor).toBeNull();
  expect(useStore.getState().headPins).toEqual([]);
  expect(useStore.getState().tokenPosition).toBeNull();
});
