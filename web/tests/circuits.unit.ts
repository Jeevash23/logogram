import { test, expect } from "@playwright/test";

import type { ModelStatus, ProjectInfo, SiteResult, SiteSetSpec, SiteSpec } from "../src/api/types";
import { buildSpec } from "../src/lib/buildSpec";
import {
  circuitSetCount,
  circuitSets,
  interactionSets,
  MAX_SETS,
  nestedSizes,
  parseSiteQuery,
  rankedSites,
  setWhat,
  siteSetsError,
  topKSets,
  universeFor,
  universeText,
} from "../src/lib/circuits";
import { DEFAULT_FORM, type FormState } from "../src/lib/formState";
import { scopeFor } from "../src/lib/spec";
import { FORM_REPLACED, useStore } from "../src/store/app";

const head = (layer: number, h: number, position: SiteSpec["position"] = { kind: "all" }): SiteSpec => ({ kind: "head", layer, head: h, position });
const mlp = (layer: number): SiteSpec => ({ kind: "mlp_out", layer, head: null, position: { kind: "all" } });
const model: ModelStatus = {
  state: "ready",
  info: {
    id: "tiny-gpt2", revision: "test", architecture: "GPT2LMHeadModel", n_layers: 2, n_heads: 4, d_model: 32, d_head: 8,
    d_mlp: 128, d_vocab: 1000, n_ctx: 64, n_params: 1000, dtype: "float32", device: "cpu", device_name: "CPU",
    process_weights: true, site_kinds: ["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out", "head"],
    backend: "transformer_lens", backend_version: "test",
  },
};
const form = (patch: Partial<FormState> = {}): FormState => ({ ...DEFAULT_FORM, ...patch });
const build = (f: FormState) => buildSpec(f, { model, datasetPath: "datasets/ioi.jsonl", datasetSha: "abc" });
const labels = (sets: SiteSetSpec[]) => sets.map((x) => x.label);

test("a circuit is tested kept alone, removed, against everything replaced, and without each site", () => {
  const circuit = [head(1, 0), head(1, 2, { kind: "last" }), head(0, 3)];
  const sets = circuitSets(circuit, { minimality: true });
  expect(labels(sets)).toEqual(["Circuit kept", "Circuit removed", "Everything replaced", "Without L1 H0", "Without L1 H2 @ last", "Without L0 H3"]);
  expect(sets.map((x) => [x.complement, x.sites.length])).toEqual([[true, 3], [false, 3], [true, 0], [true, 2], [true, 2], [true, 2]]);
  expect(sets[3].sites).toEqual([head(1, 2, { kind: "last" }), head(0, 3)]);
  // A circuit of one site has no part of itself to leave out.
  expect(labels(circuitSets([head(1, 0)], { minimality: true }))).toEqual(["Circuit kept", "Circuit removed", "Everything replaced"]);
  expect(circuitSetCount(3, true)).toBe(6);
  expect(circuitSetCount(3, false)).toBe(3);

  // The rest of the model is made of the kinds the circuit holds.
  expect(universeFor(circuit)).toEqual({ universe: ["head"] });
  expect(universeFor([head(1, 0), mlp(0)])).toEqual({ universe: ["head", "mlp_out"] });
  expect(universeFor([head(1, 0), { kind: "attn_out", layer: 0, head: null, position: { kind: "all" } }])).toMatchObject({ error: expect.stringContaining("heads or attention outputs, not both") });
  expect(universeFor([head(1, 0), { kind: "resid_pre", layer: 0, head: null, position: { kind: "all" } }])).toMatchObject({ error: expect.stringContaining("residual stream sites") });
  expect(universeText(["head", "mlp_out"])).toBe("every head and MLP output");
  expect(setWhat(true, 3, ["head"])).toBe("keeps 3 sites and replaces every other head");
  expect(setWhat(true, 0, ["head", "mlp_out"])).toBe("replaces every head and MLP output");

  // The form writes the sets as the server reads them.
  const out = build(form({ scope: { kind: "site_sets", universe: ["head"], sets } }));
  if ("error" in out) throw new Error(out.error);
  expect(out.spec.scope).toEqual({ kind: "site_sets", universe: ["head"], sets });
  expect(out.spec.experiment).toEqual({ kind: "activation_patching", direction: "clean_to_corrupt" });
});

test("nested circuits from a run's ranking chart faithfulness against size", () => {
  expect(nestedSizes(8)).toEqual([1, 2, 4, 8]);
  expect(nestedSizes(5)).toEqual([1, 2, 4, 5]);
  expect(nestedSizes(1)).toEqual([1]);
  const result = (index: number, site: Partial<SiteResult>, mean: number | null): SiteResult => ({
    index, kind: "head", layer: 0, head: index, position: { kind: "all" }, position_key: "all", row: 0, col: index, label: `L0 H${index}`,
    n: 8, effect: { mean, sd: 0.1, lo: null, hi: null }, delta: { mean, sd: 0.1, lo: null, hi: null }, patched_logit_diff: null,
    answer_prob: null, answer_prob_delta: null, sign_flips: 0, opposite_sign: 0, ...site,
  });
  const ranked = rankedSites([
    result(0, {}, 0.1),
    result(1, {}, -0.6),
    result(2, { kind: "resid_pre", head: null }, 0.9),
    result(3, { variant: { coefficient: 2, control: false } }, 0.8),
    result(4, { kind: "mlp_out", head: null, layer: 1 }, 0.3),
    result(5, {}, null),
    result(6, {}, 0.6),
  ]);
  // Strongest first by magnitude, ties by index; no residual stream sites, steering variants or gaps.
  expect(ranked).toEqual([head(0, 1), head(0, 6), mlp(1), head(0, 0)]);
  const sets = topKSets(ranked, 3);
  expect(labels(sets)).toEqual(["Top 1 kept", "Top 2 kept", "Top 3 kept", "Everything replaced", "Top 3 removed"]);
  expect(sets[2].sites).toEqual([head(0, 1), head(0, 6), mlp(1)]);
  expect(sets.slice(0, 3).every((x) => x.complement)).toBe(true);
  expect(siteSetsError(["head", "mlp_out"], sets)).toBeNull();
  // Without MLP outputs in the rest of the model, the third circuit can't keep one.
  expect(siteSetsError(["head"], sets)).toContain("keeps L1 mlp, which isn't one of the heads it replaces the rest of");
});

test("two sites are checked alone and together", () => {
  const sets = interactionSets(head(1, 0), mlp(0));
  expect(sets).toEqual([
    { label: "L1 H0", sites: [head(1, 0)], complement: false },
    { label: "L0 mlp", sites: [mlp(0)], complement: false },
    { label: "L1 H0 and L0 mlp", sites: [head(1, 0), mlp(0)], complement: false },
  ]);
  const out = build(form({ kind: "ablation", baseline: { kind: "zero" }, scope: { kind: "site_sets", universe: null, sets } }));
  if ("error" in out) throw new Error(out.error);
  expect(out.spec.scope).toEqual({ kind: "site_sets", universe: null, sets });
});

test("sets are refused as the server refuses them, with what to change", () => {
  const ok: SiteSetSpec = { label: "A", sites: [head(1, 0)], complement: false };
  expect(siteSetsError(null, [])).toContain("Add at least one set");
  const many = Array.from({ length: MAX_SETS + 1 }, (_, i) => ({ ...ok, label: `Set ${i}` }));
  expect(siteSetsError(null, many)).toBe("A run holds at most 64 sets; this has 65. Remove 1 set.");
  expect(siteSetsError(null, many.slice(0, MAX_SETS))).toBeNull();
  expect(siteSetsError(null, [ok, { ...ok }])).toContain("Two sets are labelled “A”");
  expect(siteSetsError(null, [{ ...ok, label: "  " }])).toContain("Give every set a label");
  expect(siteSetsError(null, [{ ...ok, sites: [] }])).toContain("replaces nothing");
  expect(siteSetsError(null, [{ ...ok, complement: true }])).toContain("choose what the rest of the model is made of");
  expect(siteSetsError(["head", "attn_out"], [ok])).toContain("can't be both heads and attention outputs");
  expect(siteSetsError(["head"], [ok, { label: "Kept", sites: [mlp(0)], complement: true }])).toContain("keeps L0 mlp");
  expect(siteSetsError(null, [{ ...ok, sites: [head(0, 1), { kind: "attn_out", layer: 0, head: null, position: { kind: "all" } }] }])).toContain("both heads and the attention output of layer 0");
  expect(siteSetsError(null, [{ ...ok, sites: [{ kind: "resid_mid", layer: 0, head: null, position: { kind: "all" } }] }])).toContain("a site a set can't hold");
  expect(siteSetsError(null, [{ ...ok, sites: [head(5, 0)] }], model.info)).toContain("this model has 2 layers");
  expect(siteSetsError(null, [{ ...ok, sites: [head(1, 9)] }], model.info)).toContain("this model has 4 heads per layer");
  // Only patching and ablation intervene on sets.
  const atp = build(form({ kind: "attribution_patching", scope: { kind: "site_sets", universe: null, sets: [ok] } }));
  expect("error" in atp && atp.error).toContain("activation patching or ablation");
  // Labels are written trimmed.
  const trimmed = build(form({ scope: { kind: "site_sets", universe: null, sets: [{ ...ok, label: " A " }] } }));
  expect("spec" in trimmed && trimmed.spec.scope).toEqual({ kind: "site_sets", universe: null, sets: [ok] });
});

test("sites are typed as the map writes them", () => {
  expect(parseSiteQuery("L9 H6")).toEqual({ kind: "head", layer: 9, head: 6 });
  expect(parseSiteQuery("9.6")).toEqual({ kind: "head", layer: 9, head: 6 });
  expect(parseSiteQuery("l3 mlp")).toEqual({ kind: "mlp_out", layer: 3, head: null });
  expect(parseSiteQuery("L3 attn")).toEqual({ kind: "attn_out", layer: 3, head: null });
  expect(parseSiteQuery("L2 resid_post")).toEqual({ kind: "resid_post", layer: 2, head: null });
  expect(parseSiteQuery("L2 resid pre")).toEqual({ kind: "resid_pre", layer: 2, head: null });
  expect(parseSiteQuery("layer nine")).toBeNull();
  // Methods that measure sites one by one take a sweep of sets' sites as chosen sites.
  const scope = scopeFor("attribution_patching", { kind: "site_sets", universe: ["head"], sets: circuitSets([head(1, 0), head(0, 3)], { minimality: false }) });
  expect(scope).toEqual({ kind: "sites", sites: [head(1, 0), head(0, 3)] });
});

test("a circuit from staged sites replaces the form as one step, and Undo puts the sites back in the tray", () => {
  const project: ProjectInfo = { session_id: "circuits", name: "circuits", path: "projects/circuits", datasets: [], description: "" };
  const store = useStore.getState();
  store.leaveProject();
  useStore.setState({ project, screen: "workbench", view: "explore" });
  try {
    store.setForm({ kind: "attribution_patching", direction: "corrupt_to_clean", notes: "Name movers carry it." });
    const before = useStore.getState().form;
    store.stageSelection({ layer: 1, part: "head", head: 0 });
    store.stageSelection({ layer: 0, part: "head", head: 3 });
    const staged = useStore.getState().stagedSites;
    const scope = { kind: "site_sets" as const, universe: ["head" as const], sets: circuitSets(staged, { minimality: true }) };
    store.configureSets(scope, { staged: true });
    const after = useStore.getState();
    expect(after.view).toBe("experiment");
    expect(after.stagedSites).toEqual([]);
    // Sets are patched: an estimate's direction carries over, and so do the notes.
    expect(after.form).toMatchObject({ kind: "activation_patching", direction: "corrupt_to_clean", notes: "Name movers carry it.", scope });
    const notice = after.notices.find((n) => n.text === FORM_REPLACED);
    expect(notice?.action?.label).toBe("Undo");
    notice?.action?.run();
    expect(useStore.getState().form).toEqual(before);
    expect(useStore.getState().stagedSites).toEqual(staged);
    // And the replaced form is one step of the history: undoing the undo brings the sets back.
    store.undoForm();
    expect(useStore.getState().form.scope).toEqual(scope);
  } finally {
    store.leaveProject();
  }
});
