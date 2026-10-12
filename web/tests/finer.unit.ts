import { test, expect } from "@playwright/test";

import type { ModelStatus, SiteBase, SiteSpec } from "../src/api/types";
import { buildSpec } from "../src/lib/buildSpec";
import { overlaps, parseSiteQuery, rankedSites, siteOf, siteSetsError, universeFor, universeText } from "../src/lib/circuits";
import { DEFAULT_FORM, type FormState } from "../src/lib/formState";
import { componentLabel, findSite, queryHeadsOf, queryHeadsText, selectionOfSite, siteKey } from "../src/lib/sites";
import { layersText, parseLayers, scopeFor, scopeText, siteFromSelection, siteText, workload } from "../src/lib/spec";

const ALL = { kind: "all" } as const;
const LAST = { kind: "last" } as const;
const KINDS = ["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out", "head", "q", "k", "v", "neuron"];

/** A tiny model with grouped-query attention: 4 query heads read 2 key/value heads. */
const model = (patch: { site_kinds?: string[]; site_notes?: Record<string, string> } = {}): ModelStatus => ({
  state: "ready",
  info: {
    id: "tiny-llama", revision: "test", architecture: "LlamaForCausalLM", n_layers: 2, n_heads: 4, d_model: 48, d_head: 12,
    d_mlp: 96, d_vocab: 1000, n_ctx: 64, n_params: 1000, dtype: "float32", device: "cpu", device_name: "CPU",
    process_weights: true, site_kinds: patch.site_kinds ?? KINDS, backend: "transformer_lens", backend_version: "test",
    extra: { n_key_value_heads: 2, site_notes: patch.site_notes ?? {} },
  },
});
const form = (patch: Partial<FormState> = {}): FormState => ({ ...DEFAULT_FORM, ...patch });
const build = (f: FormState, m: ModelStatus = model()) => buildSpec(f, { model: m, datasetPath: "datasets/ioi.jsonl", datasetSha: "abc" });
const errorOf = (out: ReturnType<typeof build>) => ("error" in out ? out.error : null);

test("queries, keys, values and neurons are written as the results write them", () => {
  const q: SiteSpec = { kind: "q", layer: 9, head: 6, position: ALL };
  const k: SiteSpec = { kind: "k", layer: 9, head: 2, position: LAST };
  const v: SiteSpec = { kind: "v", layer: 0, head: 3, position: { kind: "label", label: "IO" } };
  const n: SiteSpec = { kind: "neuron", layer: 3, head: null, neuron: 1201, position: ALL };
  expect([q, k, v, n].map(siteText)).toEqual(["L9 H6 q", "L9 KV2 k @ last", "L0 KV3 v @ IO", "L3 N1201"]);

  // Typed into a set as they are written, or in words.
  expect(parseSiteQuery("L9 H6 q")).toEqual({ kind: "q", layer: 9, head: 6 });
  expect(parseSiteQuery("l9 kv2 k")).toEqual({ kind: "k", layer: 9, head: 2 });
  expect(parseSiteQuery("L9 H2 value")).toEqual({ kind: "v", layer: 9, head: 2 });
  expect(parseSiteQuery("L3 N1201")).toEqual({ kind: "neuron", layer: 3, head: null, neuron: 1201 });
  expect(parseSiteQuery("L3 neuron 7")).toEqual({ kind: "neuron", layer: 3, head: null, neuron: 7 });
  expect(parseSiteQuery("L9 H6")).toEqual({ kind: "head", layer: 9, head: 6 });
  expect(parseSiteQuery("L3 mlp")).toEqual({ kind: "mlp_out", layer: 3, head: null });
  expect(parseSiteQuery("L9 H6 z")).toBeNull();
  expect(siteOf({ kind: "neuron", layer: 3, head: null, neuron: 1201 }, LAST)).toEqual({ kind: "neuron", layer: 3, head: null, neuron: 1201, position: LAST });
  expect(siteOf({ kind: "k", layer: 1, head: 1 }, ALL)).toEqual({ kind: "k", layer: 1, head: 1, position: ALL });
});

test("a key/value head is read by its group of query heads", () => {
  expect(queryHeadsOf(0, 32, 8)).toEqual([0, 1, 2, 3]);
  expect(queryHeadsOf(7, 32, 8)).toEqual([28, 29, 30, 31]);
  expect(queryHeadsOf(5, 12, 12)).toEqual([5]);
  expect(queryHeadsText([4, 5, 6, 7])).toBe("query heads 4–7");
  expect(queryHeadsText([3])).toBe("query head 3");
});

test("a selection finds one neuron, query, key or value among many", () => {
  const site = (index: number, kind: SiteBase["kind"], extra: Partial<SiteBase>): SiteBase => ({
    index, kind, layer: 1, head: null, position: ALL, position_key: "all", row: 0, col: index, label: "", ...extra,
  });
  const sites = [
    site(0, "neuron", { neuron: 0 }),
    site(1, "neuron", { neuron: 1 }),
    site(2, "q", { head: 1 }),
    site(3, "k", { head: 1 }),
    site(4, "head", { head: 1 }),
    site(5, "mlp_out", {}),
  ];
  for (const s of sites) expect(findSite(sites, selectionOfSite(s))).toBe(s);
  expect(componentLabel(selectionOfSite(sites[1]))).toBe("L1 N1");
  expect(componentLabel(selectionOfSite(sites[2]))).toBe("L1 H1 q");
  expect(componentLabel(selectionOfSite(sites[3]))).toBe("L1 KV1 k");
  // Two neurons of a layer are different sites across runs too.
  expect(siteKey(sites[0])).not.toBe(siteKey(sites[1]));
  // A site's key is unchanged for sites that have no neuron.
  expect(siteKey(sites[4])).toBe("head|1|1||all|");
  // Patching or ablating from a selection makes the same site again.
  expect(siteFromSelection(selectionOfSite(sites[1]))).toEqual({ kind: "neuron", layer: 1, head: null, neuron: 1, position: ALL });
  expect(siteFromSelection(selectionOfSite(sites[3]))).toEqual({ kind: "k", layer: 1, head: 1, position: ALL });
});

test("sets hold queries, keys, values and neurons, never with what they make", () => {
  const n = (layer: number, neuron: number): SiteSpec => ({ kind: "neuron", layer, head: null, neuron, position: ALL });
  const mlp: SiteSpec = { kind: "mlp_out", layer: 1, head: null, position: ALL };
  const info = model().info;
  expect(siteSetsError(["neuron"], [{ label: "keep", sites: [n(1, 3)], complement: true }], info)).toBeNull();
  expect(siteSetsError(["neuron", "mlp_out"], [{ label: "keep", sites: [], complement: true }], info)).toContain("made from neurons");
  expect(siteSetsError(["q", "head"], [{ label: "keep", sites: [], complement: true }], info)).toContain("made from queries");
  expect(siteSetsError(null, [{ label: "both", sites: [n(1, 2), mlp], complement: false }], info)).toContain("both neurons and MLP outputs of layer 1");
  expect(siteSetsError(null, [{ label: "kv", sites: [{ kind: "k", layer: 1, head: 2, position: ALL }], complement: false }], info)).toContain("key/value head 2; this model has 2 key/value heads");
  expect(siteSetsError(null, [{ label: "n", sites: [n(0, 96)], complement: false }], info)).toContain("neuron 96; this model's MLPs have 96 neurons");
  // A model that doesn't offer a kind says why, as measured when it loaded.
  const olmo = model({ site_kinds: KINDS.filter((k) => k !== "q"), site_notes: { q: "Its queries aren't separate per head." } }).info;
  expect(siteSetsError(null, [{ label: "q", sites: [{ kind: "q", layer: 0, head: 0, position: ALL }], complement: false }], olmo)).toBe("Its queries aren't separate per head.");
  expect(overlaps("attn_out", "v")).toBe(true);
  expect(overlaps("neuron", "q")).toBe(false);
  expect(universeFor([n(1, 3), n(0, 2)])).toEqual({ universe: ["neuron"] });
  expect(universeFor([n(1, 3), mlp])).toMatchObject({ error: expect.stringContaining("made from neurons") });
  expect(universeText(["q", "neuron"])).toBe("every query and neuron");
  // A run's strongest neurons and values make a circuit.
  const ranked = rankedSites([
    { index: 0, kind: "neuron", layer: 1, head: null, neuron: 5, position: ALL, position_key: "all", row: 0, col: 5, label: "L1 N5", n: 4, effect: { mean: 0.2, sd: 0, lo: 0, hi: 0 }, delta: { mean: 0, sd: 0, lo: 0, hi: 0 }, patched_logit_diff: null, answer_prob: null, answer_prob_delta: null, sign_flips: 0, opposite_sign: 0 },
    { index: 1, kind: "v", layer: 0, head: 1, position: LAST, position_key: "last", row: 0, col: 1, label: "L0 KV1 v @ last", n: 4, effect: { mean: -0.5, sd: 0, lo: 0, hi: 0 }, delta: { mean: 0, sd: 0, lo: 0, hi: 0 }, patched_logit_diff: null, answer_prob: null, answer_prob_delta: null, sign_flips: 0, opposite_sign: 0 },
  ]);
  expect(ranked).toEqual([
    { kind: "v", layer: 0, head: 1, position: LAST },
    { kind: "neuron", layer: 1, head: null, neuron: 5, position: ALL },
  ]);
});

test("the form sweeps every query, key or value, or every neuron of chosen layers", () => {
  const neurons = build(form({ scope: { kind: "neurons", layers: [1, 0], position: LAST } }));
  expect("spec" in neurons && neurons.spec.scope).toEqual({ kind: "neurons", layers: [1, 0], position: LAST });
  expect(errorOf(build(form({ scope: { kind: "neurons", layers: [], position: LAST } })))).toContain("Choose the layers");
  expect(errorOf(build(form({ scope: { kind: "neurons", layers: [2], position: LAST } })))).toContain("Layer 2 doesn't exist");
  const keys = build(form({ scope: { kind: "head_inputs", input: "k", position: ALL } }));
  expect("spec" in keys && keys.spec.scope).toEqual({ kind: "head_inputs", input: "k", position: ALL });
  // A model that doesn't offer them says why.
  const olmo = model({ site_kinds: KINDS.filter((k) => k !== "q" && k !== "k"), site_notes: { q: "Its queries aren't separate per head.", k: "Its keys aren't separate per head." } });
  expect(errorOf(build(form({ scope: { kind: "head_inputs", input: "q", position: ALL } }), olmo))).toBe("Its queries aren't separate per head.");
  expect(errorOf(build(form({ scope: { kind: "head_inputs", input: "v", position: ALL } }), olmo))).toBeNull();
  // Queries and keys have no direct effect; values and neurons do.
  const dla = { kind: "direct_logit_attribution" as const, dlaPrompts: "clean" as const };
  expect(errorOf(build(form({ ...dla, scope: { kind: "head_inputs", input: "q", position: ALL } })))).toContain("no direct effect");
  expect(errorOf(build(form({ ...dla, scope: { kind: "head_inputs", input: "v", position: { kind: "label", label: "IO" } } })))).toBeNull();
  expect(errorOf(build(form({ ...dla, scope: { kind: "neurons", layers: [1], position: ALL } })))).toContain("read at the last token");
  expect(errorOf(build(form({ ...dla, scope: { kind: "neurons", layers: [1], position: LAST } })))).toBeNull();
  // Steering and path patching don't take them.
  const sites = { kind: "sites" as const, sites: [{ kind: "neuron" as const, layer: 0, head: null, neuron: 3, position: LAST }] };
  expect(errorOf(build(form({ kind: "path_patching", pathReceivers: [{ kind: "logits" }], scope: sites })))).toContain("Paths start at");
});

test("switching methods fits the new sweeps to them", () => {
  const q = { kind: "head_inputs" as const, input: "q" as const, position: ALL };
  const neurons = { kind: "neurons" as const, layers: [3], position: ALL };
  expect(scopeFor("direct_logit_attribution", q)).toEqual({ kind: "head_inputs", input: "v", position: ALL });
  expect(scopeFor("direct_logit_attribution", neurons)).toEqual({ kind: "neurons", layers: [3], position: LAST });
  expect(scopeFor("attribution_patching", neurons)).toBe(neurons);
  expect(scopeFor("ablation", q)).toBe(q);
  expect(scopeFor("path_patching", q)).toEqual({ kind: "heads", position: ALL });
  expect(scopeFor("steering", neurons)).toEqual({ kind: "layer_components", components: ["resid_pre"], position: LAST });
  const listed = { kind: "sites" as const, sites: [
    { kind: "q" as const, layer: 1, head: 0, position: ALL },
    { kind: "v" as const, layer: 1, head: 0, position: { kind: "label" as const, label: "IO" } },
    { kind: "neuron" as const, layer: 1, head: null, neuron: 4, position: ALL },
  ] };
  expect(scopeFor("direct_logit_attribution", listed)).toEqual({ kind: "sites", sites: [
    { kind: "v", layer: 1, head: 0, position: { kind: "label", label: "IO" } },
    { kind: "neuron", layer: 1, head: null, neuron: 4, position: LAST },
  ] });
});

test("the work a sweep of keys or neurons takes counts key/value heads and neurons", () => {
  const spec = (scope: unknown) => {
    const out = build(form({ scope: scope as FormState["scope"] }));
    if (!("spec" in out)) throw new Error(out.error);
    return out.spec;
  };
  // 2 layers × 2 key/value heads × 10 prompts; 2 layers × 96 neurons × 10 prompts.
  expect(workload(spec({ kind: "head_inputs", input: "k", position: ALL }), 10, 2, 4, null, 0, 2, 96)).toBe(40);
  expect(workload(spec({ kind: "head_inputs", input: "q", position: ALL }), 10, 2, 4, null, 0, 2, 96)).toBe(80);
  expect(workload(spec({ kind: "neurons", layers: [0, 1], position: LAST }), 10, 2, 4, null, 0, 2, 96)).toBe(1920);
  expect(scopeText({ kind: "neurons", layers: [3, 4, 9], position: LAST })).toBe("every MLP neuron of layers 3, 4 and 9, the last token");
  expect(scopeText({ kind: "head_inputs", input: "v", position: ALL })).toBe("every value, all positions");
  expect(layersText([2])).toBe("layer 2");
  expect(parseLayers("3, 4")).toEqual([3, 4]);
  expect(parseLayers("0-2 7")).toEqual([0, 1, 2, 7]);
  expect(parseLayers("three")).toBeNull();
  expect(parseLayers("")).toBeNull();
});
