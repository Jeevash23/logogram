import { test, expect } from "@playwright/test";

import type { ProjectInfo, RunDetail, RunListing } from "../src/api/types";
import {
  deleteWarning,
  filterCounts,
  historyLabel,
  historyRows,
  keptText,
  matchesRun,
  paletteRuns,
  pickerGroups,
  runTitle,
} from "../src/lib/runs";
import { DEFAULT_FORM, useStore } from "../src/store/app";

/** A finished run of GPT-2 small: activation patching over heads, from datasets/ioi.jsonl. */
function listing(id: string, overrides: Partial<RunListing> = {}): RunListing {
  return {
    id,
    name: `Run ${id}`,
    description: "",
    status: "finished",
    created: null,
    finished: null,
    wall_time_s: null,
    n_prompts: 8,
    layout_kind: "heads",
    model_id: "openai-community/gpt2",
    dataset: "datasets/snapshots/0a1b.jsonl",
    dataset_source: "datasets/ioi.jsonl",
    experiment: { kind: "activation_patching", direction: "clean_to_corrupt" },
    scope: { kind: "heads", position: { kind: "all" } },
    derived_from: null,
    error: null,
    ...overrides,
  };
}

const damaged = (id: string, overrides: Partial<RunListing> = {}) =>
  listing(id, {
    name: id,
    status: "failed",
    damaged: true,
    model_id: "",
    dataset: "",
    dataset_source: null,
    experiment: {} as RunListing["experiment"],
    scope: {} as RunListing["scope"],
    error: "Its spec.json is missing, so Logogram can't tell what it ran. Restore the file from a backup or version control, or delete the run.",
    ...overrides,
  });

const ids = (rows: { run: RunListing }[]) => rows.map((r) => r.run.id);

test("a run is called by its title, or its spec's name", () => {
  expect(runTitle(listing("a", { title: "Name movers" }))).toBe("Name movers");
  expect(runTitle(listing("a"))).toBe("Run a");
  expect(runTitle(null, "fallback")).toBe("fallback");
  const check = listing("b", { derived_from: { run: "a", kind: "robustness", change: "Zero-ablate" } });
  expect(historyLabel(check)).toBe("Robustness · Zero-ablate");
  expect(historyLabel({ ...check, title: "Zero baseline" })).toBe("Zero baseline");
});

test("a search finds runs by title, name, id, method, model and dataset, every word", () => {
  const run = listing("20261011-120000-heads", { name: "Heads sweep", title: "Name movers" });
  for (const query of ["name movers", "HEADS sweep", "20261011-120000", "activation patching", "activation_patching", "patch clean→corrupt", "layer × head", "gpt2", "GPT-2 small", "ioi.jsonl", "snapshots", "  movers   gpt2  "]) {
    expect(matchesRun(run, query), query).toBe(true);
  }
  for (const query of ["ablation", "movers pythia", "greater_than.jsonl"]) expect(matchesRun(run, query), query).toBe(false);
  expect(matchesRun(run, "")).toBe(true);
  const ablation = listing("c", { experiment: { kind: "ablation", baseline: { kind: "zero" } } });
  expect(matchesRun(ablation, "zero ablation")).toBe(true);
  // A run whose spec can't be read is found by what is known of it: its id and title.
  const broken = damaged("20000101-broken", { title: "Keep for the paper" });
  expect(matchesRun(broken, "broken")).toBe(true);
  expect(matchesRun(broken, "paper")).toBe(true);
  expect(matchesRun(broken, "gpt2")).toBe(false);
  // An experiment this version doesn't know is still searched by the rest.
  const unknown = listing("d", { experiment: { kind: "from_the_future" } as unknown as RunListing["experiment"] });
  expect(matchesRun(unknown, "from the future")).toBe(true);
});

test("filters show starred runs, keep archived ones apart, and nest runs under the run they came from", () => {
  const runs = [
    listing("rerun", { derived_from: { run: "first", kind: "rerun" } }),
    listing("archived", { archived: true, starred: true }),
    listing("first", { starred: true }),
    listing("orphan", { derived_from: { run: "deleted", kind: "robustness", change: "Zero-ablate" } }),
    listing("child-of-archived", { derived_from: { run: "archived", kind: "rerun" } }),
  ];
  expect(filterCounts(runs)).toEqual({ all: 4, starred: 1, archived: 1 });
  const all = historyRows(runs, "all", "");
  expect(ids(all)).toEqual(["first", "rerun", "orphan", "child-of-archived"]);
  expect(all.map((r) => r.depth)).toEqual([0, 1, 0, 0]);
  // A run from a deleted run says so; one whose parent is only hidden doesn't.
  expect(all.find((r) => r.run.id === "orphan")?.orphan).toBe(true);
  expect(all.find((r) => r.run.id === "child-of-archived")?.orphan).toBe(false);
  expect(ids(historyRows(runs, "starred", ""))).toEqual(["first"]);
  expect(ids(historyRows(runs, "archived", ""))).toEqual(["archived"]);
  // A search keeps the nesting of what it finds.
  expect(ids(historyRows(runs, "all", "rerun"))).toEqual(["rerun"]);
  // A shared project's provenance can contain a cycle: every run stays reachable.
  const cycle = [listing("x", { derived_from: { run: "y", kind: "rerun" } }), listing("y", { derived_from: { run: "x", kind: "rerun" } })];
  expect(ids(historyRows(cycle, "all", "")).sort()).toEqual(["x", "y"]);
});

test("the palette offers starred and recent runs, and finds any run once something is typed", () => {
  const runs = Array.from({ length: 20 }, (_, i) => listing(`r${String(i).padStart(2, "0")}`));
  runs[15] = { ...runs[15], starred: true };
  runs[18] = { ...runs[18], archived: true, title: "Old but gold" };
  runs[19] = damaged("r19");
  const idle = paletteRuns(runs, "");
  expect(idle[0].id).toBe("r15");
  expect(idle.map((r) => r.id).slice(1)).toEqual(runs.slice(0, 12).map((r) => r.id));
  const typed = paletteRuns(runs, "gold");
  expect(typed.map((r) => r.id)).toContain("r18"); // archived, but found
  expect(typed.map((r) => r.id)).not.toContain("r19"); // its spec can't be read: nothing to open
  expect(typed).toHaveLength(19);
});

test("pickers list starred runs first and archived runs last", () => {
  expect(pickerGroups([listing("a"), listing("b")])).toEqual([{ label: null, runs: [listing("a"), listing("b")] }]);
  const groups = pickerGroups([listing("a"), listing("b", { starred: true }), listing("c", { archived: true })]);
  expect(groups.map((g) => [g.label, g.runs.map((r) => r.id)])).toEqual([
    ["Starred", ["b"]],
    ["Other runs", ["a"]],
    ["Archived", ["c"]],
  ]);
  expect(pickerGroups([listing("a"), listing("c", { archived: true })]).map((g) => g.label)).toEqual(["Runs", "Archived"]);
});

test("deleting says what goes and what stays", () => {
  const first = listing("20261011-first", { title: "Heads" });
  const runs = [
    first,
    listing("check-1", { derived_from: { run: first.id, kind: "robustness", change: "Zero-ablate" } }),
    listing("check-2", { derived_from: { run: first.id, kind: "robustness", change: "float16" } }),
    listing("again", { derived_from: { run: first.id, kind: "rerun" } }),
  ];
  expect(deleteWarning(first, runs)).toBe(
    "This deletes its folder, experiments/20261011-first, with the spec it ran, its per-prompt results, its summary and its manifest. It can't be undone. " +
      "The 3 runs derived from it (2 robustness checks and a rerun) stay, but no longer link back to it. To keep it out of the way instead, archive it.",
  );
  expect(deleteWarning(listing("d", { status: "draft" }), runs)).toBe("This deletes its folder, experiments/d, with its saved spec. It can't be undone.");
  expect(deleteWarning(listing("f", { status: "failed" }), [])).toBe("This deletes its folder, experiments/f, with the spec it ran and its manifest. It can't be undone.");
  expect(deleteWarning(damaged("b"), [])).toBe("This deletes its folder, experiments/b, with everything in it. It can't be undone.");
  expect(deleteWarning(listing("one"), [listing("v", { derived_from: { run: "one", kind: "verification" } })])).toContain(
    "The run derived from it (a verification by patching) stays, but no longer links back to it.",
  );
});

/** Chosen features of an SAE on layer 6, at the last token. */
const features: RunListing["scope"] = { kind: "sites", sites: [{ kind: "sae_feature", layer: 6, head: null, feature: 123, position: { kind: "last" } }] };
const estimate: RunListing["experiment"] = { kind: "attribution_patching", direction: "clean_to_corrupt", method: "gradient", steps: null };

test("cancelling says what a run keeps: SAE features, every one or chosen ones, keep nothing", () => {
  const nothing = "A run of SAE features reports nothing before it ends, so it keeps no results.";
  const every: RunListing["scope"] = { kind: "features", position: { kind: "last" }, top: 50, choose_on: null, seed: null };
  expect(keptText(listing("every", { experiment: estimate, scope: every }))).toBe(nothing);
  // Patched, zero-ablated or estimated, chosen features are reported only when the run ends too.
  expect(keptText(listing("patched", { scope: features }))).toBe(nothing);
  expect(keptText(listing("ablated", { experiment: { kind: "ablation", baseline: { kind: "zero" } }, scope: features }))).toBe(nothing);
  expect(keptText(listing("estimated", { experiment: estimate, scope: features }))).toBe(nothing);
  // The model's own components keep what finished.
  expect(keptText(listing("heads"))).toBe("What it has finished is kept and marked as partial: the layers that finished. The layer in progress isn't kept.");
  expect(keptText(listing("head", { scope: { kind: "sites", sites: [{ kind: "head", layer: 1, head: 2, position: { kind: "last" } }] } }))).toBe(
    "What it has finished is kept and marked as partial: the sites that finished. The site in progress isn't kept.",
  );
  expect(keptText(listing("estimate", { experiment: estimate }))).toBe(
    "What it has finished is kept and marked as partial: every site, over the prompts that finished. The prompts in progress aren't kept.",
  );
});

// -- the store -------------------------------------------------------------------------------

const project = (id: string): ProjectInfo => ({ session_id: id, name: id, path: `projects/${id}`, datasets: [], description: "" });

/** The API as a project with `runs`; `routes` answer the requests named in them. */
function stubApi(runs: () => RunListing[], routes: Record<string, (body: unknown) => unknown> = {}) {
  const original = globalThis.fetch;
  const seen: string[] = [];
  globalThis.fetch = (async (input: string, init?: RequestInit) => {
    const key = `${init?.method ?? "GET"} ${input}`;
    seen.push(key);
    const answer = routes[key];
    if (answer) {
      const out = answer(init?.body ? JSON.parse(String(init.body)) : undefined);
      return out instanceof Response ? out : new Response(JSON.stringify(out));
    }
    if (input === "/api/runs") return new Response(JSON.stringify(runs()));
    return new Response(JSON.stringify({ error: "Not part of this test." }), { status: 404 });
  }) as typeof fetch;
  return { seen, restore: () => { globalThis.fetch = original; } };
}

/** A run's detail with as much of its spec as the analyses read. */
const detail = (run: RunListing): RunDetail => ({
  id: run.id,
  spec: {
    name: run.name,
    model: { id: run.model_id, revision: "main", dtype: "float32", device: "cpu", process_weights: true },
    dataset: { path: run.dataset, sha256: null, limit: null },
    tokenization: { prepend_bos: false },
    execution: { batch_size: 8 },
    metric: { kind: "logit_diff", normalization: "dataset_gap" },
  } as unknown as RunDetail["spec"],
  summary: null,
  manifest: null,
  listing: run,
  folder: `experiments/${run.id}`,
});

test("a renamed, starred or archived run changes wherever this tab shows it", async () => {
  let listed = [listing("a"), listing("b")];
  const api = stubApi(() => listed, {
    "PATCH /api/runs/a": (body) => ({ ...listing("a"), ...(body as object), annotations_error: null }),
  });
  const store = useStore.getState();
  try {
    store.leaveProject();
    await store.enterProject(project("one"));
    useStore.setState({ runDetails: { a: detail(listing("a")) } });
    expect(await store.annotateRun("a", { title: "Name movers", starred: true })).toBe(true);
    const st = useStore.getState();
    expect(st.runs.find((r) => r.id === "a")).toMatchObject({ title: "Name movers", starred: true, archived: false });
    expect(st.runDetails.a.listing).toMatchObject({ title: "Name movers", starred: true });
    expect(api.seen).toContain("PATCH /api/runs/a");
  } finally {
    api.restore();
    store.leaveProject();
  }
});

test("a deleted run is forgotten everywhere: compared, shown, flagged or filling the form", async () => {
  const first = listing("first", { title: "Heads" });
  const draft = listing("draft", { status: "draft" });
  let listed = [first, listing("check", { derived_from: { run: "first", kind: "robustness", change: "Zero-ablate" } }), draft];
  const api = stubApi(() => listed, {
    "DELETE /api/runs/first": () => {
      listed = listed.filter((r) => r.id !== "first");
      return { ok: true };
    },
    "DELETE /api/runs/draft": () => {
      listed = listed.filter((r) => r.id !== "draft");
      return { ok: true };
    },
    "DELETE /api/runs/check": () => new Response(JSON.stringify({ error: "“check” is running. Wait for it to end, or cancel it, then delete it." }), { status: 409 }),
  });
  const store = useStore.getState();
  try {
    store.leaveProject();
    await store.enterProject(project("one"));
    useStore.setState({
      activeRunId: "first",
      compareIds: ["first", "check"],
      runDetails: { first: detail(first) },
      flags: { first: { against: "check", sites: [1] }, check: { against: "first", sites: [2] } },
    });
    expect(await store.deleteRun("first")).toBe(true);
    let st = useStore.getState();
    expect(st.runs.map((r) => r.id)).toEqual(["check", "draft"]);
    expect(st.activeRunId).toBeNull();
    expect(st.compareIds).toEqual([null, "check"]);
    expect(st.runDetails.first).toBeUndefined();
    expect(st.flags).toEqual({});
    expect(st.notices.at(-1)?.text).toBe("Deleted “Heads”.");

    // A draft the form was saved as: the form keeps what it says, no longer saved anywhere.
    useStore.setState({ form: { ...DEFAULT_FORM, name: "Next", nameEdited: true, draftId: "draft" } });
    expect(await store.deleteRun("draft")).toBe(true);
    st = useStore.getState();
    expect(st.form).toMatchObject({ name: "Next", draftId: null });

    // The server refuses a run in progress, and says why: nothing is forgotten.
    expect(await store.deleteRun("check")).toBe(false);
    st = useStore.getState();
    expect(st.runs.map((r) => r.id)).toEqual(["check"]);
    expect(st.notices.at(-1)).toMatchObject({ tone: "error", text: "“check” is running. Wait for it to end, or cancel it, then delete it." });

    // Another tab deleted a run: the event says so.
    useStore.setState({ compareIds: ["check", null], activeRunId: "check" });
    listed = [];
    store.handleEvent({ type: "runs.changed", project_session: "one", run_id: "check", deleted: true });
    st = useStore.getState();
    expect(st.runs).toEqual([]);
    expect(st.compareIds).toEqual([null, null]);
    expect(st.activeRunId).toBeNull();
    // Another project's event isn't this tab's.
    useStore.setState({ runs: [listing("kept")] });
    store.handleEvent({ type: "runs.changed", project_session: "two", run_id: "kept", deleted: true });
    expect(useStore.getState().runs.map((r) => r.id)).toEqual(["kept"]);
  } finally {
    api.restore();
    store.leaveProject();
  }
});

test("a run whose spec can't be read isn't opened: the tab says why", async () => {
  const broken = damaged("20000101-broken");
  const api = stubApi(() => [broken]);
  const store = useStore.getState();
  try {
    store.leaveProject();
    await store.enterProject(project("one"));
    await store.openRun(broken.id, "results");
    const st = useStore.getState();
    expect(st.activeRunId).not.toBe(broken.id);
    expect(st.notices.at(-1)).toMatchObject({ tone: "error", text: broken.error });
    expect(api.seen).not.toContain(`GET /api/runs/${broken.id}`);
  } finally {
    api.restore();
    store.leaveProject();
  }
});

test("a cancelled run of SAE features says it kept nothing, though some of it had finished", async () => {
  const chosen = listing("chosen", { status: "running", scope: features });
  const heads = listing("heads", { status: "running" });
  const api = stubApi(() => [chosen, heads], {
    "GET /api/runs/chosen": () => detail({ ...chosen, status: "cancelled" }),
    "GET /api/runs/heads": () => detail({ ...heads, status: "cancelled" }),
  });
  const store = useStore.getState();
  const said = () => useStore.getState().notices.map((n) => n.text);
  try {
    store.leaveProject();
    await store.enterProject(project("one"));
    store.handleEvent({ type: "run.cancelled", project_session: "one", run_id: "chosen", partial: null });
    expect(said()).toContain(
      "Run cancelled. A run of SAE features reports nothing before it ends, so it kept no results. Its spec and any dataset snapshot remain.",
    );
    // A run of the model's components that kept nothing had nothing finished.
    store.handleEvent({ type: "run.cancelled", project_session: "one", run_id: "heads", partial: null });
    expect(said()).toContain("Run cancelled before anything finished. Its spec and any dataset snapshot remain.");
  } finally {
    api.restore();
    store.leaveProject();
  }
});
