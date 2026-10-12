import { test, expect } from "@playwright/test";

import type { Environment, Manifest, ProjectInfo, RunDetail, RunListing, Spec } from "../src/api/types";
import { differenceText, environmentChanges, environmentNote } from "../src/lib/environment";
import { ELSEWHERE_POLL_MS, useStore } from "../src/store/app";

const here: Environment = {
  logogram: "0.4.0",
  python: "3.12.3",
  implementation: "CPython",
  os: "linux",
  architecture: "x86_64",
  versions: { torch: "2.6.0+cpu", transformer_lens: "4.0.0", transformers: "4.50.0", numpy: "2.1.0", pyarrow: "19.0.0" },
  torch_cuda: null,
  devices: ["cpu"],
  default_device: "cpu",
};

function manifest(versions: Record<string, string | null>, device = "cpu"): Manifest {
  return {
    run_id: "r",
    status: "finished",
    versions,
    device: { type: device, name: "Example CPU", cuda: null, threads: 4, deterministic_algorithms: true },
  };
}

const spec = (device: string) => ({ model: { device } }) as unknown as Spec;
const same = { logogram: "0.4.0", python: "3.11.9", torch: "2.6.0+cpu", transformer_lens: "4.0.0", transformers: "4.50.0", numpy: "2.1.0", pyarrow: "18.0.0" };

test("a run made here, with these versions, needs no note", () => {
  // Python's and pyarrow's versions don't change a run's numbers, so they aren't compared.
  expect(environmentChanges(manifest(same), spec("auto"), here)).toEqual([]);
  expect(environmentNote([])).toBeNull();
  // Without the server's environment, or a manifest, nothing can be said.
  expect(environmentChanges(manifest({ ...same, torch: "2.5.1" }), spec("auto"), null)).toEqual([]);
  expect(environmentChanges(null, spec("auto"), here)).toEqual([]);
});

test("the note names each version that differs, and the device a rerun here would use", () => {
  const changes = environmentChanges(manifest({ ...same, logogram: "0.3.0", torch: "2.5.1+cu124", numpy: "1.26.4" }, "cuda"), spec("auto"), here);
  expect(changes.map((c) => c.label)).toEqual(["Logogram", "PyTorch", "NumPy", "Device"]);
  expect(environmentNote(changes)).toBe(
    "This run ran with Logogram 0.3.0 (here 0.4.0), PyTorch 2.5.1+cu124 (here 2.6.0+cpu) and NumPy 1.26.4 (here 2.1.0), " +
      "on CUDA (here the CPU). A rerun here may differ in the last digits.",
  );
  // A version the run didn't record isn't compared.
  expect(environmentChanges(manifest({ torch: "2.6.0+cpu", numpy: null }), spec("cpu"), here)).toEqual([]);
});

test("the device is the one a rerun of the spec would use here", () => {
  const withCuda: Environment = { ...here, devices: ["cpu", "cuda"], default_device: "cuda" };
  // "auto" picks CUDA where there is one; a spec that says cpu stays on the CPU.
  expect(environmentNote(environmentChanges(manifest(same, "cpu"), spec("auto"), withCuda))).toBe(
    "This run ran on the CPU (here CUDA). A rerun here may differ in the last digits.",
  );
  expect(environmentChanges(manifest(same, "cpu"), spec("cpu"), withCuda)).toEqual([]);
  // A device this machine doesn't have can't be used: a rerun takes the default.
  expect(environmentChanges(manifest(same, "mps"), spec("mps"), here)).toMatchObject([{ label: "Device", recorded: "Apple's MPS", here: "the CPU" }]);
});

test("compare names device types", () => {
  expect(differenceText({ key: "device.type", label: "Device", a: "cpu", b: "cuda" })).toEqual({ a: "CPU", b: "CUDA" });
  expect(differenceText({ key: "torch", label: "PyTorch", a: "2.5.1", b: "2.6.0" })).toEqual({ a: "2.5.1", b: "2.6.0" });
});

// -- runs in another process --------------------------------------------------------------------

const project: ProjectInfo = { session_id: "one", name: "one", path: "projects/one", datasets: [], description: "" };

function listing(id: string, status: RunListing["status"], elsewhere: boolean): RunListing {
  return {
    id, name: `Run ${id}`, description: "", status, created: null, finished: null, wall_time_s: null, n_prompts: null,
    layout_kind: null, model_id: "m", dataset: "datasets/a.jsonl", experiment: { kind: "activation_patching", direction: "clean_to_corrupt" },
    scope: { kind: "heads", position: { kind: "all" } }, derived_from: null, error: null, elsewhere,
  } as RunListing;
}

test("a run in another process is read again on a slow timer until it ends", async () => {
  let runs: RunListing[] = [listing("mine", "finished", false), listing("there", "running", true)];
  const fetched: string[] = [];
  const original = { fetch: globalThis.fetch, setInterval: globalThis.setInterval, clearInterval: globalThis.clearInterval };
  const timers = new Map<number, { fn: () => void; ms: number }>();
  let nextTimer = 1;
  globalThis.fetch = (async (input: string) => {
    fetched.push(input);
    if (input === "/api/runs") return new Response(JSON.stringify(runs));
    if (input.startsWith("/api/runs/") && !input.includes("derived")) {
      const id = decodeURIComponent(input.split("/")[3]);
      const runSpec = {
        name: `Run ${id}`,
        model: { id: "m", revision: null, dtype: "float32", device: "cpu", process_weights: true },
        dataset: { path: "datasets/a.jsonl", sha256: null, limit: null },
        tokenization: { prepend_bos: true },
        execution: { batch_size: 8 },
        statistics: { bootstrap: 200, ci: 0.95, seed: 0, cluster: null },
      };
      const detail = { id, spec: runSpec, summary: null, manifest: null, listing: null, folder: `experiments/${id}` };
      return new Response(JSON.stringify(detail as unknown as RunDetail));
    }
    if (input.startsWith("/api/dataset")) return new Response(JSON.stringify({ name: "a.jsonl", path: "datasets/a.jsonl", sha256: "0", n: 0, records: [] }));
    return new Response(JSON.stringify({ error: "Not part of this test." }), { status: 404 });
  }) as typeof fetch;
  globalThis.setInterval = ((fn: () => void, ms: number) => {
    timers.set(nextTimer, { fn, ms });
    return nextTimer++;
  }) as unknown as typeof setInterval;
  globalThis.clearInterval = ((id: number) => void timers.delete(id)) as typeof clearInterval;
  const store = useStore.getState();
  try {
    store.leaveProject();
    await store.enterProject(project);
    const state = useStore.getState();
    // The other process's run streams nothing here: it isn't opened as this server's run.
    expect(state.live.there).toBeUndefined();
    expect(state.activeRunId).toBe("mine");
    expect([...timers.values()].map((t) => t.ms)).toEqual([ELSEWHERE_POLL_MS]);

    await store.openRun("there", "results");
    runs = [listing("mine", "finished", false), listing("there", "finished", false)];
    const reads = fetched.filter((f) => f === "/api/runs/there").length;
    const [timer] = timers.values();
    timer.fn();
    await expect.poll(() => useStore.getState().runs.find((r) => r.id === "there")?.status).toBe("finished");
    // It says so, offers the results, and reads the run again; the timer stops.
    expect(useStore.getState().notices.at(-1)).toMatchObject({
      text: "“Run there”, running in another process, finished.",
      action: { label: "Show results" },
    });
    await expect.poll(() => fetched.filter((f) => f === "/api/runs/there").length).toBe(reads + 1);
    expect(timers.size).toBe(0);
  } finally {
    Object.assign(globalThis, original);
    store.leaveProject();
  }
});
