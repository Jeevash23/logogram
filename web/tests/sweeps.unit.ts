import { test, expect } from "@playwright/test";

import type { ProjectInfo, QueuedRun, RunDetail, RunListing, Spec, SweepMember } from "../src/api/types";
import { deleteWarning, historyLabel, historyRows } from "../src/lib/runs";
import { selectionOfSite } from "../src/lib/sites";
import { membershipText, parseValues, queuedValues, statusText, sweepsOf, sweepsOfRun, valueLabel, valueName } from "../src/lib/sweeps";
import { useStore } from "../src/store/app";

/** A finished run of GPT-2 small: activation patching over heads. */
function listing(id: string, overrides: Partial<RunListing> = {}): RunListing {
  return {
    id,
    name: `Donors · donor seed ${id}`,
    description: "",
    status: "finished",
    created: `2026-10-12T10:00:0${id.length}Z`,
    finished: null,
    wall_time_s: null,
    n_prompts: 8,
    layout_kind: "heads",
    model_id: "openai-community/gpt2",
    dataset: "datasets/snapshots/0a1b.jsonl",
    dataset_source: "datasets/ioi.jsonl",
    experiment: { kind: "ablation", baseline: { kind: "resample", pool: "corrupt", donors: 2, seed: 0 } },
    scope: { kind: "heads", position: { kind: "all" } },
    derived_from: null,
    error: null,
    ...overrides,
  };
}

/** A run of the sweep "s1" over donor seeds 0 to 3, made from "orig", which stands in for seed 0. */
function member(id: string, index: number, overrides: Partial<SweepMember> = {}, listed: Partial<RunListing> = {}): RunListing {
  const record: SweepMember = {
    kind: "sweep",
    sweep: "s1",
    field: "experiment.baseline.seed",
    value: index,
    index,
    of: 4,
    values: [0, 1, 2, 3],
    run: "orig",
    original: 0,
    change: `donor seed ${index}`,
    ...overrides,
  };
  return listing(id, { name: `Donors · donor seed ${index}`, derived_from: record, ...listed });
}

test("runs that record a sweep are grouped, with the run it was made from in its place", () => {
  const original = listing("orig", { name: "Donors" });
  const runs = [member("m3", 3, {}, { status: "failed" }), member("m1", 1), original, member("m2", 2), listing("other")];
  const [sweep] = sweepsOf(runs);
  expect(sweepsOf(runs)).toHaveLength(1);
  expect(sweep.id).toBe("s1");
  expect(sweep.values).toEqual([0, 1, 2, 3]);
  expect(Object.fromEntries(Object.entries(sweep.runs).map(([i, r]) => [i, r.id]))).toEqual({ 0: "orig", 1: "m1", 2: "m2", 3: "m3" });
  expect(sweep.name).toBe("Donors");
  // Each run knows its place: the original stands in for its value.
  expect(sweepsOfRun([sweep], "orig")).toEqual([{ sweep, index: 0 }]);
  expect(membershipText(sweep, 0)).toBe("This run stands in for donor seed 0 in a sweep over 4 donor seeds.");
  expect(membershipText(sweep, 2, "B")).toBe("B is donor seed 2 in a sweep over 4 donor seeds.");
  expect(sweepsOfRun([sweep], "other")).toEqual([]);
});

test("a deleted original leaves its value empty, a finished run wins a value claimed twice, and a run that disagrees is left out", () => {
  const runs = [
    member("a", 1, {}, { status: "failed" }),
    member("b", 1),
    member("c", 2, { values: [0, 1, 2, 9] }),
    member("d", 3),
  ];
  const [sweep] = sweepsOf(runs);
  expect(sweep.runs[0]).toBeUndefined();
  expect(sweep.runs[1].id).toBe("b");
  expect(sweep.runs[2]).toBeUndefined();
  expect(sweep.runs[3].id).toBe("d");
  // Without the run it was made from, the sweep is named after its runs, without their values.
  expect(sweep.name).toBe("Donors");
});

test("the history names a sweep's runs by their value, nests them, and says what deleting one does", () => {
  const original = listing("orig", { name: "Donors" });
  const one = member("m1", 1);
  const fromSpec = member("free", 1, { run: null, original: null });
  expect(historyLabel(one)).toBe("Sweep · donor seed 1");
  // A sweep of a spec has no run to nest under: its runs keep their names.
  expect(historyLabel(fromSpec)).toBe("Donors · donor seed 1");
  const rows = historyRows([one, original], "all", "");
  expect(rows.map((r) => [r.run.id, r.depth, r.orphan])).toEqual([["orig", 0, false], ["m1", 1, false]]);
  expect(historyRows([fromSpec], "all", "")[0].orphan).toBe(false);
  expect(deleteWarning(original, [original, one])).toContain(
    "It stands in for donor seed 0 in a sweep over 4 donor seeds. The sweep will have no results for that donor seed.",
  );
  expect(deleteWarning(original, [original, one])).toContain("(a run of a sweep)");
});

test("values are named as the sweep shows them", () => {
  expect(valueLabel("dataset.path", "datasets/ioi-seed1.jsonl")).toBe("ioi-seed1.jsonl");
  expect(valueLabel("model.revision", "step1000")).toBe("step1000");
  expect(valueLabel("model.revision", "0123456789abcdef0123456789abcdef01234567")).toBe("0123456");
  expect(valueName("statistics.seed", 3)).toBe("bootstrap seed 3");
  expect(valueName("scope.seed", 1)).toBe("feature split seed 1");
  // A field from a newer version is named by its path.
  expect(valueName("tokenization.something", "x")).toBe("tokenization.something x");
});

test("typed values are read in order, and seeds must be whole numbers in range", () => {
  expect(parseValues("seed", "0, 1,2  3\n4")).toEqual({ values: [0, 1, 2, 3, 4], error: null });
  expect(parseValues("seed", "1, x")).toEqual({ values: [], error: "Seeds are whole numbers from 0 to 4,294,967,295; “x” isn't one." });
  expect(parseValues("seed", "4294967296").error).toContain("isn't one");
  expect(parseValues("seed", "-1").error).toContain("isn't one");
  expect(parseValues("revision", "step1000, step2000 main")).toEqual({ values: ["step1000", "step2000", "main"], error: null });
});

test("values still waiting in the queue, and values without results, say so", () => {
  const waiting = { derived_from: member("q", 2).derived_from } as QueuedRun;
  const other = { derived_from: { run: "x", kind: "rerun" } } as QueuedRun;
  expect([...queuedValues([waiting, other], "s1")]).toEqual([2]);
  expect([...queuedValues([waiting], "s2")]).toEqual([]);
  expect(statusText("missing", true)).toBe("queued");
  expect(statusText("missing", false)).toBe("not run");
  expect(statusText("failed", false)).toBe("failed");
  expect(statusText("deleted", false)).toBe("deleted");
  expect(statusText("finished", false)).toBeNull();
});

test("a revision's run opened at a site keeps the site, though the run shown before was another revision's", async () => {
  const spec = (revision: string): Spec => ({
    logogram_spec: 4, name: `Pythia heads · revision ${revision}`, notes: "",
    model: { id: "EleutherAI/pythia-70m", source: "hub", revision, sha256: null, device: "auto", dtype: "float32", process_weights: true },
    dataset: { path: "datasets/ioi.jsonl", sha256: "abc", limit: null }, tokenization: { prepend_bos: true, chat_template: null },
    experiment: { kind: "activation_patching", direction: "clean_to_corrupt" }, scope: { kind: "heads", position: { kind: "all" } },
    metric: { kind: "logit_diff", normalization: "dataset_gap" }, statistics: { bootstrap: 1000, seed: 0, ci: 0.95, cluster: null }, execution: { batch_size: 64 },
  });
  const site = { index: 5, row: 3, col: 2, layer: 3, head: 2, kind: "head", position_key: "all", variant_key: null, feature: null, label: "L3 H2",
    effect: { mean: 0.5, lo: 0.4, hi: 0.6 }, delta: { mean: 1, lo: 0.9, hi: 1.1 } };
  const detail = (revision: string) => ({ id: revision, spec: spec(revision), manifest: null, listing: null, folder: `experiments/${revision}`,
    summary: { layout: { kind: "heads", rows: [], cols: [] }, sites: [site], model: null, statistics: { ci: 0.95 } } }) as unknown as RunDetail;
  const project: ProjectInfo = { session_id: "p", name: "p", path: "p", datasets: [], description: "" };
  const original = globalThis.fetch;
  globalThis.fetch = (async () => new Response("[]")) as typeof fetch;
  const store = useStore.getState();
  try {
    store.leaveProject();
    useStore.setState({ project, screen: "workbench", view: "sweep", datasetPath: "datasets/ioi.jsonl", activeRunId: "step1000",
      runDetails: { step1000: detail("step1000"), step2000: detail("step2000") } });
    // The sweep's chart: Enter on L3 H2 at revision step2000 opens that revision's run there.
    const at = selectionOfSite(site as never);
    await store.openRun("step2000", "results", at);
    expect(useStore.getState()).toMatchObject({ activeRunId: "step2000", view: "results", selection: at });
    // Opened without a site, a run of another revision drops the selection.
    await store.openRun("step1000", "results");
    expect(useStore.getState().selection).toBeNull();
  } finally {
    globalThis.fetch = original;
    store.leaveProject();
  }
});
