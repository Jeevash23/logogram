import { test, expect } from "@playwright/test";

import type { DatasetDetail, EdgeSpec, ModelStatus, PromptRecord, Spec } from "../src/api/types";
import { buildSpec, datasetFacts, parseCounts } from "../src/lib/buildSpec";
import { curvePoint, edgeKey, graphColumns, inputOffset, logTicks, nearestEdge, rankEdges, receiverPlace, senderPlace } from "../src/lib/edges";
import { parseForm } from "../src/lib/formDraft";
import { DEFAULT_EDGE_COUNTS, DEFAULT_FORM, formFromSpec, type FormState } from "../src/lib/formState";
import { selectionOfSite, siteKey } from "../src/lib/sites";
import { edgeLabel, experimentText, graphText, scopeFor, scopeText, suggestName, workload, workloadText } from "../src/lib/spec";

const model: ModelStatus = {
  state: "ready",
  info: {
    id: "tiny-gpt2", revision: "test", architecture: "GPT2LMHeadModel", n_layers: 2, n_heads: 4, d_model: 32, d_head: 8,
    d_mlp: 128, d_vocab: 1000, n_ctx: 64, n_params: 1000, dtype: "float32", device: "cpu", device_name: "CPU",
    process_weights: true, site_kinds: ["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out", "head"],
    backend: "transformer_lens", backend_version: "test",
  },
  ref: { id: "tiny-gpt2", source: "hub", revision: "test", sha256: null, dtype: "float32", device: "cpu", process_weights: true },
};
const ctx = (extra: Partial<Parameters<typeof buildSpec>[1]> = {}) => ({ model, datasetPath: "datasets/ioi.jsonl", datasetSha: "abc", ...extra });
const built = (f: FormState, extra: Partial<Parameters<typeof buildSpec>[1]> = {}): Spec => {
  const out = buildSpec(f, ctx(extra));
  if ("error" in out) throw new Error(out.error);
  return out.spec;
};
const refused = (f: FormState, extra: Partial<Parameters<typeof buildSpec>[1]> = {}): string => {
  const out = buildSpec(f, ctx(extra));
  if (!("error" in out)) throw new Error("expected the form to be refused");
  return out.error;
};
/** The form switched to edge attribution, as choosing the method does. */
const edges = (patch: Partial<FormState> = {}): FormState => ({
  ...DEFAULT_FORM,
  kind: "edge_attribution",
  scope: scopeFor("edge_attribution", DEFAULT_FORM.scope),
  ...patch,
});

const edge = (sender: EdgeSpec["sender"], receiver: EdgeSpec["receiver"]): EdgeSpec => ({ sender, receiver });

test("edge attribution states its direction, its estimate and its curve, and which edges the curve patches has no default", () => {
  expect(edges().scope).toEqual({ kind: "graph", layers: null, logits: true });
  expect(refused(edges())).toBe("Choose which edges the faithfulness curve patches.");
  const spec = built(edges({ edgePatch: "rest" }));
  expect(spec.experiment).toEqual({
    kind: "edge_attribution",
    direction: "clean_to_corrupt",
    method: "gradient",
    steps: null,
    curve: { counts: [1, 3, 10, 30, 100, 300, 1000, 3000, 10000], patch: "rest", control: { seed: 0, draws: 1 } },
  });
  expect(spec.scope).toEqual({ kind: "graph", layers: null, logits: true });
  expect(spec.name).toBe("Edges clean→corrupt · the whole graph");
  // A saved spec fills the form, and the form builds the same spec again.
  expect(built(formFromSpec(spec))).toEqual(spec);

  // Integrated gradients over 2 to 64 steps, and no curve at all.
  const ig = built(edges({ atpMethod: "integrated_gradients", atpSteps: 8, edgeCurve: false, direction: "corrupt_to_clean" }));
  expect(ig.experiment).toEqual({ kind: "edge_attribution", direction: "corrupt_to_clean", method: "integrated_gradients", steps: 8, curve: null });
  expect(formFromSpec(ig)).toMatchObject({ kind: "edge_attribution", atpMethod: "integrated_gradients", atpSteps: 8, edgeCurve: false, edgePatch: null });
  expect(refused(edges({ atpMethod: "integrated_gradients", atpSteps: 1, edgeCurve: false }))).toContain("from 2 to 64");
  expect(experimentText(ig.experiment)).toBe("Estimate each edge corrupt → clean with integrated gradients (8 steps)");
  expect(experimentText(spec.experiment)).toBe("Estimate each edge clean → corrupt, then patch the strongest for a faithfulness curve");
});

test("the curve's numbers of edges increase, its random control is seeded, and its layers are the model's", () => {
  expect(parseCounts(DEFAULT_EDGE_COUNTS)).toEqual({ values: [1, 3, 10, 30, 100, 300, 1000, 3000, 10000] });
  expect(parseCounts("10 100;1000")).toEqual({ values: [10, 100, 1000] });
  expect(parseCounts("")).toEqual({ error: "Give at least one number of edges for the curve, such as 10." });
  expect(parseCounts("10, 5")).toEqual({ error: "List the curve's numbers of edges in increasing order, each once." });
  expect(parseCounts("10, 10")).toEqual({ error: "List the curve's numbers of edges in increasing order, each once." });
  expect("error" in parseCounts("1,000")).toBe(true);
  expect("error" in parseCounts("2.5")).toBe(true);
  expect(parseCounts(Array.from({ length: 25 }, (_, i) => i + 1).join(", "))).toEqual({ error: "Use at most 24 numbers of edges for the curve." });

  const curve = edges({ edgePatch: "chosen" });
  expect(refused({ ...curve, edgeCounts: "30, 10" })).toContain("increasing order");
  expect(refused({ ...curve, edgeDraws: 0 })).toBe("Draw the random edges from 1 to 16 times.");
  expect(refused({ ...curve, edgeDraws: 17 })).toBe("Draw the random edges from 1 to 16 times.");
  expect(refused({ ...curve, edgeSeed: -1 })).toBe("The random control's seed is a whole number from 0.");
  expect(built({ ...curve, edgeDraws: 3, edgeSeed: 7, edgeCounts: "5, 50" }).experiment).toMatchObject({
    curve: { counts: [5, 50], patch: "chosen", control: { seed: 7, draws: 3 } },
  });

  // Chosen layers: at least one, and only those the loaded model has.
  const layers = { kind: "graph" as const, layers: [1], logits: false };
  expect(built({ ...curve, scope: layers }).scope).toEqual(layers);
  expect(refused({ ...curve, scope: { ...layers, layers: [] } })).toBe("Choose at least one layer whose heads and MLPs receive edges, or every layer.");
  expect(refused({ ...curve, scope: { ...layers, layers: [2] } })).toBe("The loaded model has 2 layers (0 to 1). Choose layers it has.");
  expect(graphText({ layers: [0, 1, 2, 5], logits: true })).toBe("the edges into layers 0–2, 5 and the logits");
  expect(graphText({ layers: [3], logits: false })).toBe("the edges into layer 3");
  expect(scopeText({ kind: "graph", layers: null, logits: false })).toBe("every edge but those into the logits");
});

test("only edge attribution and ACDC measure the graph, and answers of several tokens are refused", () => {
  // Another method leaves the graph for a sweep of its own; edge attribution always has one.
  expect(scopeFor("activation_patching", { kind: "graph", layers: [1], logits: true })).toEqual({ kind: "heads", position: { kind: "all" } });
  expect(refused({ ...DEFAULT_FORM, scope: { kind: "graph", layers: null, logits: true } })).toContain("Only edge attribution and ACDC measure the edges");
  expect(refused(edges({ edgePatch: "chosen", scope: { kind: "heads", position: { kind: "all" } } }))).toContain("Edge attribution estimates the edges of the model's graph");

  const record = (patch: Partial<PromptRecord> = {}): PromptRecord => ({ clean: "When A and B", corrupt: "When A and C", answer: " B", distractor: " C", ...patch });
  const dataset: DatasetDetail = {
    name: "ioi.jsonl", path: "datasets/ioi.jsonl", sha256: "abc", n: 2,
    records: [record(), record({ answer: " Bob Smith" })], continuations: [1],
  };
  const facts = datasetFacts(dataset, null);
  expect(refused(edges({ edgePatch: "chosen", metric: "logprob_diff" }), { facts })).toContain("Edge attribution reads the metric at the last prompt position");
});

test("a verification measures each edge by path patching, into the receivers its edges name", () => {
  const verified: Spec = {
    ...built(edges({ edgePatch: "chosen" })),
    experiment: { kind: "path_patching", direction: "clean_to_corrupt", receivers: [{ kind: "logits" }, { kind: "mlp", layer: 1 }], freeze_mlps: true },
    scope: {
      kind: "edges",
      edges: [
        edge({ kind: "head", layer: 0, head: 1 }, { kind: "logits" }),
        edge({ kind: "embed" }, { kind: "mlp", layer: 1 }),
        edge({ kind: "mlp", layer: 0 }, { kind: "logits" }),
      ],
    },
  };
  const form = formFromSpec(verified);
  expect(form).toMatchObject({ kind: "path_patching", pathFreezeMlps: true });
  expect(built(form)).toEqual(verified);
  // The receivers are those of the edges, whatever the form held.
  expect(built({ ...form, pathReceivers: [{ kind: "logits" }] }).experiment).toEqual(verified.experiment);
  expect(refused({ ...form, pathFreezeMlps: false })).toContain("Hold the MLPs");
  expect(scopeText(verified.scope)).toBe("3 edges, one at a time");
  expect(workloadText(verified, workload(verified, 8, 2, 4, null, 0) as number)).toBe("That is 24 paths, each edge on each prompt alone, four forward passes each.");
  expect(edgeLabel((verified.scope as { edges: EdgeSpec[] }).edges[1])).toBe("embeddings → L1 MLP");
});

test("an edge run's cost is said before it runs", () => {
  const spec = built(edges({ edgePatch: "chosen", edgeCounts: "10, 100", edgeDraws: 2 }));
  expect(workload(spec, 8, 2, 4, null, 0)).toBe(8);
  expect(workloadText(spec, 8)).toBe(
    "That is three forward passes and one backward pass for each of the 8 prompts, for every edge at once. The faithfulness curve then patches the strongest edges for real, beside edges chosen at random.",
  );
  const ig = built(edges({ atpMethod: "integrated_gradients", atpSteps: 4, edgeCurve: false }));
  expect(workloadText(ig, workload(ig, 8, 2, 4, null, 0) as number)).toBe(
    "That is two forward passes, then 4 forward and backward passes, for each of the 8 prompts, for every edge at once.",
  );
  expect(suggestName(ig)).toBe("Edges clean→corrupt, integrated gradients · the whole graph");
});

test("the graph is drawn with layers left to right, a head's inputs above, at and below it", () => {
  expect(graphColumns(12)).toBe(26);
  expect(senderPlace({ kind: "embed" })).toEqual({ column: 0, head: null });
  expect(senderPlace({ kind: "head", layer: 3, head: 5 })).toEqual({ column: 7, head: 5 });
  expect(senderPlace({ kind: "mlp", layer: 3 })).toEqual({ column: 8, head: null });
  expect(receiverPlace({ kind: "head", layer: 4, head: 2, input: "q" }, 12)).toEqual({ column: 9, head: 2 });
  expect(receiverPlace({ kind: "mlp", layer: 4 }, 12)).toEqual({ column: 10, head: null });
  expect(receiverPlace({ kind: "logits" }, 12)).toEqual({ column: 25, head: null });
  expect([inputOffset({ kind: "head", layer: 1, head: 0, input: "q" }), inputOffset({ kind: "head", layer: 1, head: 0, input: "k" }), inputOffset({ kind: "head", layer: 1, head: 0, input: "v" }), inputOffset({ kind: "logits" })]).toEqual([-1, 0, 1, 0]);

  // A curve leaves and arrives horizontally, and a point near it finds it; a point far away none.
  const a = { x1: 0, y1: 0, x2: 100, y2: 50 };
  const b = { x1: 0, y1: 50, x2: 100, y2: 0 };
  expect(curvePoint(a, 0)).toEqual({ x: 0, y: 0 });
  expect(curvePoint(a, 1)).toEqual({ x: 100, y: 50 });
  expect(curvePoint(a, 0.5)).toEqual({ x: 50, y: 25 });
  expect(nearestEdge([a, b], 2, 1)).toBe(0);
  expect(nearestEdge([a, b], 98, 1)).toBe(1);
  expect(nearestEdge([a, b], 50, 25)).toBe(0); // they cross here: the first, the stronger, wins
  expect(nearestEdge([a, b], 50, 45)).toBeNull();
});

test("edges are ranked by the size of their effect, and an edge is the same edge in any run", () => {
  const site = (index: number, mean: number | null) => ({ index, effect: { mean } });
  expect(rankEdges([site(0, 0.1), site(1, -0.5), site(2, null), site(3, 0.5), site(4, Number.NaN)]).map((x) => x.index)).toEqual([1, 3, 0]);
  expect(logTicks(1, 10000)).toEqual([1, 3, 10, 30, 100, 300, 1000, 3000, 10000]);
  expect(logTicks(5, 400)).toEqual([10, 30, 100, 300]);
  expect(logTicks(1, 1e6)).toEqual([1, 10, 100, 1000, 10000, 100000, 1000000]);

  const e1 = edge({ kind: "head", layer: 0, head: 1 }, { kind: "head", layer: 1, head: 2, input: "q" });
  const e2 = edge({ kind: "head", layer: 0, head: 1 }, { kind: "head", layer: 1, head: 2, input: "k" });
  expect(edgeKey(e1)).not.toBe(edgeKey(e2));
  // Key order doesn't matter: an edge read from either run names the same edge.
  expect(edgeKey({ receiver: { input: "q", head: 2, layer: 1, kind: "head" }, sender: { head: 1, layer: 0, kind: "head" } })).toBe(edgeKey(e1));
  const base = { kind: "edge" as const, layer: 1, head: null, position_key: "all", variant_key: null };
  expect(siteKey({ ...base, edge: e1 })).not.toBe(siteKey({ ...base, edge: e2 }));
  const selection = selectionOfSite({ ...base, position: { kind: "all" }, label: "L0 H1 → L1 H2 q", edge: e1 });
  expect(selection).toEqual({ layer: 1, part: "edge", variantKey: "L0 H1 → L1 H2 q" });
});

test("a form with edges is kept in the browser, and one kept before them takes their defaults", () => {
  const form = edges({
    edgePatch: "rest",
    edgeCounts: "5, 50",
    edgeDraws: 4,
    edgeSeed: 9,
    scope: { kind: "graph", layers: [0, 1], logits: false },
    pathReceivers: [{ kind: "mlp", layer: 1 }, { kind: "head", layer: 1, head: 0, input: "v" }],
  });
  expect(parseForm(JSON.parse(JSON.stringify(form)))).toEqual(form);
  const verification = {
    ...DEFAULT_FORM,
    kind: "path_patching" as const,
    scope: { kind: "edges" as const, edges: [edge({ kind: "embed" }, { kind: "logits" }), edge({ kind: "mlp", layer: 0 }, { kind: "mlp", layer: 1 })] },
  };
  expect(parseForm(JSON.parse(JSON.stringify(verification)))).toEqual(verification);
  // An edge whose sender is the embeddings has no layer or head.
  const bad = { ...verification, scope: { kind: "edges", edges: [{ sender: { kind: "embed", layer: 0 }, receiver: { kind: "logits" } }] } };
  expect(parseForm(bad)).toBeNull();
  const before: Record<string, unknown> = { ...DEFAULT_FORM };
  for (const key of ["edgeCurve", "edgeCounts", "edgePatch", "edgeSeed", "edgeDraws"]) delete before[key];
  expect(parseForm(before)).toEqual(DEFAULT_FORM);
});
