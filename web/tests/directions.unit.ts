import { test, expect } from "@playwright/test";

import type { DatasetDetail, ModelStatus, PromptRecord, SAEStatus, Spec } from "../src/api/types";
import { buildSpec, datasetFacts, metricOf } from "../src/lib/buildSpec";
import { DEFAULT_FORM, formFromSpec, type FormState } from "../src/lib/formState";
import { baselineText, experimentText, labelsText, measureOf, readoutText } from "../src/lib/spec";

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
const form = (patch: Partial<FormState> = {}): FormState => ({ ...DEFAULT_FORM, ...patch });
type Context = Parameters<typeof buildSpec>[1];
const ctx = (extra: Partial<Context> = {}) => ({ model, datasetPath: "datasets/ioi.jsonl", datasetSha: "abc", ...extra });
const built = (f: FormState, extra: Partial<Context> = {}): Spec => {
  const out = buildSpec(f, ctx(extra));
  if ("error" in out) throw new Error(out.error);
  return out.spec;
};
const refused = (f: FormState, extra: Partial<Context> = {}): string => {
  const out = buildSpec(f, ctx(extra));
  if (!("error" in out)) throw new Error("expected the form to be refused");
  return out.error;
};

const record = (meta: Record<string, unknown>): PromptRecord => ({ clean: "When A and B", corrupt: "When A and C", answer: " B", distractor: " C", meta });
const dataset = (records: PromptRecord[]): DatasetDetail => ({ name: "ioi.jsonl", path: "datasets/ioi.jsonl", sha256: "abc", n: records.length, records });

const atLast = { kind: "layer_components" as const, components: ["resid_post" as const], position: { kind: "last" as const } };
const everywhere = { kind: "layer_components" as const, components: ["resid_post" as const], position: { kind: "all" as const } };
const toward = { name: "toward-io", sha256: null };

test("a probe states what it reads, has no metric, and reads one position of each prompt", () => {
  const probe = form({ kind: "probe", scope: atLast });
  expect(refused(probe)).toContain("Choose what the probe reads");
  const spec = built({ ...probe, probeLabels: "side" });
  expect(spec.experiment).toEqual({ kind: "probe", labels: { kind: "side" }, train_fraction: 0.5, seed: 0, regularization: 0.01, iterations: 100, control: true });
  expect(spec.metric).toBeNull();
  // A saved spec fills the form, and the form builds the same spec again.
  expect(built(formFromSpec(spec))).toEqual(spec);
  expect(measureOf(spec.experiment)).toBe("probe");
  expect(labelsText(spec.experiment.kind === "probe" ? spec.experiment.labels : { kind: "side" })).toContain("which prompt");

  // Refused before a run, as the server refuses it.
  expect(refused({ ...probe, probeLabels: "side", scope: everywhere })).toContain("A probe reads one position of each prompt");
  expect(refused({ ...probe, probeLabels: "meta", probeField: null })).toContain("field of the prompts' meta");
  expect(refused({ ...probe, probeLabels: "side", probeTrain: 1 })).toContain("between 0 and 100%");
  expect(refused({ ...probe, probeLabels: "side", probeIterations: 0 })).toContain("1 to 1,000 iterations");

  // A field of the prompts' meta: every prompt has it, and it takes two values or more.
  const templates = dataset([record({ template: "a" }), record({ template: "b" }), record({ template: "a" }), record({ template: "b" })]);
  const byTemplate = { ...probe, probeLabels: "meta" as const, probeField: "template", probePrompts: "clean" as const };
  const facts = datasetFacts(templates, null);
  const labelled = built(byTemplate, { facts });
  expect(labelled.experiment).toMatchObject({ labels: { kind: "meta", field: "template", prompts: "clean" } });
  expect(built(formFromSpec(labelled), { facts })).toEqual(labelled);
  expect(refused({ ...byTemplate, probeField: "gender" }, { facts })).toContain("Not every prompt has “gender” in its meta");
  const same = datasetFacts(dataset([record({ template: "a" }), record({ template: "a" }), record({ template: "a" }), record({ template: "a" })]), null);
  expect(refused(byTemplate, { facts: same })).toContain("nothing for a probe to tell apart");
  // Enough pairs on each side of the split.
  const two = datasetFacts(dataset([record({ template: "a" }), record({ template: "b" })]), null);
  expect(refused({ ...probe, probeLabels: "side" }, { facts: two })).toContain("at least one pair to train on and two to measure on");
});

test("distributed alignment search learns fewer dimensions than the site has, and round-trips", () => {
  const das = form({ kind: "das", scope: atLast, dasDimensions: 2, dasSteps: 50 });
  const spec = built(das);
  expect(spec.experiment).toEqual({
    kind: "das", direction: "clean_to_corrupt", dimensions: 2, train_fraction: 0.5, seed: 0, steps: 50, learning_rate: 0.01, control: true,
  });
  expect(spec.metric).toEqual({ kind: "logit_diff", normalization: "dataset_gap" });
  expect(built(formFromSpec(spec))).toEqual(spec);
  expect(experimentText(spec.experiment)).toContain("2");

  expect(refused({ ...das, dasDimensions: 32 })).toContain("whole 32-wide activation");
  expect(refused({ ...das, dasDimensions: 65 })).toContain("1 to 64 dimensions");
  expect(refused({ ...das, dasRate: 0 })).toContain("learning rate");
  expect(refused({ ...das, scope: everywhere })).toContain("Distributed alignment search reads one position");
});

test("subspace patching, scaling and directional ablation name the saved direction they use", () => {
  const subspace = form({ kind: "subspace_patching", scope: atLast });
  expect(refused(subspace)).toContain("Choose the saved direction to patch along");
  const patched = built({ ...subspace, subspace: toward });
  expect(patched.experiment).toEqual({ kind: "subspace_patching", direction: "clean_to_corrupt", subspace: toward });
  expect(built(formFromSpec(patched))).toEqual(patched);

  const scaled = built(form({ kind: "ablation", baseline: { kind: "scale", factor: 0.5 }, scope: atLast }));
  expect(scaled.experiment).toEqual({ kind: "ablation", baseline: { kind: "scale", factor: 0.5 } });
  expect(built(formFromSpec(scaled))).toEqual(scaled);
  expect(baselineText({ kind: "scale", factor: 0.5 })).toContain("0.5");
  expect(refused(form({ kind: "ablation", baseline: { kind: "scale", factor: Number.NaN }, scope: atLast }))).toContain("scale factor");

  const removed = built(form({ kind: "ablation", baseline: { kind: "projection", direction: toward, reference: null }, scope: atLast }));
  expect(removed.experiment).toEqual({ kind: "ablation", baseline: { kind: "projection", direction: toward, reference: null } });
  expect(built(formFromSpec(removed))).toEqual(removed);
  expect(baselineText({ kind: "projection", direction: toward, reference: "clean" })).toContain("toward-io");
  expect(refused(form({ kind: "ablation", baseline: { kind: "projection", direction: { name: "", sha256: null }, reference: null }, scope: atLast })))
    .toContain("Choose the saved direction to remove");

  const steer = form({
    kind: "steering", steerApplyTo: "corrupt", steerDirection: "direction",
    scope: { kind: "sites", sites: [{ kind: "resid_pre", layer: 1, head: null, position: { kind: "last" } }] },
  });
  expect(refused(steer)).toContain("Choose the saved direction to steer along");
  const steered = built({ ...steer, steerSaved: toward });
  expect(steered.experiment).toMatchObject({ kind: "steering", direction: { kind: "direction", direction: toward } });
  expect(built(formFromSpec(steered))).toEqual(steered);
});

test("a readout reads a value inside the model, for the methods that run it forward", () => {
  const neuron = { kind: "neuron" as const, layer: 1, neuron: 7 };
  const reading = form({ kind: "ablation", baseline: { kind: "zero" }, scope: atLast, metric: "readout" });
  expect(metricOf(reading)).toBeNull();
  expect(refused(reading)).toContain("Choose what the readout reads");
  const spec = built({ ...reading, readout: neuron });
  expect(spec.metric).toEqual({ kind: "readout", unit: neuron, position: { kind: "last" }, normalization: "dataset_gap" });
  expect(built(formFromSpec(spec))).toEqual(spec);
  expect(readoutText({ kind: "readout", unit: neuron, position: { kind: "last" }, normalization: "dataset_gap" })).toBe("neuron 7 of L1's MLP at the last token");

  // Gradient methods read the model's output.
  expect(refused({ ...reading, readout: neuron, kind: "attribution_patching" })).toContain("A readout is read by activation patching, ablation and subspace patching");
  // Single sites, not sets of sites or features.
  const sets = {
    kind: "site_sets" as const,
    universe: null,
    sets: [{ label: "a", complement: false, sites: [{ kind: "resid_post" as const, layer: 0, head: null, position: { kind: "last" as const } }] }],
  };
  expect(refused({ ...reading, readout: neuron, scope: sets })).toContain("read site by site");
  // A feature readout needs its SAE.
  const feature = { kind: "sae_feature" as const, feature: 3 };
  expect(refused({ ...reading, readout: feature })).toContain("Load the SAE whose feature the readout reads");
  const sae: SAEStatus = {
    state: "ready",
    info: { repo: "owner/sae", path: "layer_1", revision: "abc", site: "resid_pre", layer: 1, d_in: 32, d_sae: 64, activation: "relu", k: null, normalize: "none", format: "test", note: "", fit: null },
  };
  const read = built({ ...reading, readout: feature }, { sae });
  expect(read.sae).toEqual({ repo: "owner/sae", path: "layer_1", revision: "abc" });
  // Reading along a saved direction, at a site and layer.
  const along = { kind: "direction" as const, direction: toward, site: "resid_post" as const, layer: 1 };
  expect(built({ ...reading, readout: along }).metric).toMatchObject({ unit: along });
});
