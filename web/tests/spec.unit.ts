import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { isProjectScoped, PROJECT_SCOPED } from "../src/api/client";
import type { DatasetDetail, ModelStatus, PromptRecord, Spec } from "../src/api/types";
import { analysisContext } from "../src/lib/analysis";
import { buildSpec, datasetFacts, metricOf } from "../src/lib/buildSpec";
import { DEFAULT_FORM, formFromSpec, type FormState } from "../src/lib/formState";
import { METRIC_KINDS, metricWords } from "../src/lib/metrics";
import { measureWords, splitCount, workload, workloadText } from "../src/lib/spec";
import { useStore } from "../src/store/app";

const model: ModelStatus = {
  state: "ready",
  info: {
    id: "tiny-gpt2", revision: "test", architecture: "GPT2LMHeadModel", n_layers: 2, n_heads: 4, d_model: 32, d_head: 8,
    d_mlp: 128, d_vocab: 1000, n_ctx: 64, n_params: 1000, dtype: "float32", device: "cpu", device_name: "CPU",
    process_weights: true, site_kinds: ["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out", "head"],
    backend: "transformer_lens", backend_version: "test",
  },
  ref: { id: "tiny-gpt2", revision: "test", dtype: "float32", device: "cpu", process_weights: true },
};
const form = (patch: Partial<FormState> = {}): FormState => ({ ...DEFAULT_FORM, ...patch });
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

const record = (patch: Partial<PromptRecord> = {}): PromptRecord => ({ clean: "When A and B", corrupt: "When A and C", answer: " B", distractor: " C", ...patch });
const dataset = (records: PromptRecord[], continuations?: number[]): DatasetDetail => ({
  name: "ioi.jsonl", path: "datasets/ioi.jsonl", sha256: "abc", n: records.length, records, ...(continuations ? { continuations } : {}),
});

test("the default form builds a version 2 spec that states every choice", () => {
  const spec = built(form());
  expect(spec).toEqual({
    logogram_spec: 2,
    name: "Patch clean→corrupt · layer × head",
    notes: "",
    model: { id: "tiny-gpt2", revision: "test", dtype: "float32", device: "cpu", process_weights: true },
    dataset: { path: "datasets/ioi.jsonl", sha256: "abc", limit: null },
    tokenization: { prepend_bos: true },
    experiment: { kind: "activation_patching", direction: "clean_to_corrupt" },
    scope: { kind: "heads", position: { kind: "all" } },
    metric: { kind: "logit_diff", normalization: "dataset_gap" },
    statistics: { bootstrap: 1000, ci: 0.95, seed: 0, cluster: null },
    execution: { batch_size: 64 },
    predictions: null,
  });
  // A saved spec fills the form, and the form builds the same spec again.
  expect(built(formFromSpec(spec))).toEqual(spec);
});

test("each metric is written as the server reads it, and the KL divergence needs its target", () => {
  for (const kind of METRIC_KINDS.filter((k) => k !== "kl")) {
    expect(built(form({ metric: kind, normalization: "prompt_gap" })).metric).toEqual({ kind, normalization: "prompt_gap" });
  }
  expect(refused(form({ metric: "kl" }))).toContain("KL divergence is measured from");
  expect(metricOf(form({ metric: "kl" }))).toBeNull();
  const spec = built(form({ metric: "kl", klTarget: "corrupt" }));
  expect(spec.metric).toEqual({ kind: "kl", target: "corrupt", normalization: "dataset_gap" });
  // A target chosen for the KL divergence isn't written for another metric.
  expect(built(form({ metric: "prob", klTarget: "corrupt" })).metric).toEqual({ kind: "prob", normalization: "dataset_gap" });
  expect(formFromSpec(spec)).toMatchObject({ metric: "kl", klTarget: "corrupt" });
});

test("a method's restrictions on the metric and the answers are said before running", () => {
  const dla = form({ kind: "direct_logit_attribution", dlaPrompts: "clean", scope: { kind: "heads", position: { kind: "last" } } });
  expect(built(dla).metric.kind).toBe("logit_diff");
  expect(refused({ ...dla, metric: "logprob_diff" })).toContain("Direct logit attribution splits the logit difference");

  // Answers of several tokens: known once the loaded model has read the dataset.
  const records = [record(), record({ answer: " Buenos Aires", distractor: " Paris" }), record()];
  const facts = datasetFacts(dataset(records, [1]), null);
  expect(facts).toMatchObject({ n: 3, sets: 0, continuations: 1, example: " Buenos Aires" });
  expect(refused(form(), { facts })).toMatch(/^1 prompt has an answer or distractor of several tokens \(such as “·Buenos·Aires”\)\. The logit difference reads one token/);
  expect(built(form({ metric: "logprob_diff" }), { facts }).metric.kind).toBe("logprob_diff");
  expect(refused(form({ metric: "logprob_diff", kind: "path_patching", pathReceivers: [{ kind: "logits" }] }), { facts })).toContain("Path patching reads the metric at the last prompt position");
  expect(refused(form({ metric: "prob", kind: "ablation", baseline: { kind: "mean", reference: "corrupt" } }), { facts })).toContain("Mean ablation replaces each position");
  // At one position, mean ablation reads them.
  expect(built(form({ metric: "prob", kind: "ablation", baseline: { kind: "mean", reference: "corrupt" }, scope: { kind: "heads", position: { kind: "last" } } }), { facts }).metric.kind).toBe("prob");
  // A limit that leaves the long answer out leaves nothing to say.
  expect(built(form({ limit: 1 }), { facts: datasetFacts(dataset(records, [1]), 1) }).metric.kind).toBe("logit_diff");
  // Sets of single tokens: the logit difference reads them, direct attribution doesn't.
  const sets = datasetFacts(dataset([record({ answer: ["33", "34"], distractor: ["31", "32"] })], []), null);
  expect(built(form(), { facts: sets }).metric.kind).toBe("logit_diff");
  expect(refused(dla, { facts: sets })).toContain("an answer set or an answer of several tokens");
});

test("the bootstrap resamples clusters only of a field every prompt has", () => {
  const records = [
    record({ meta: { template: "went", pattern: "ABBA", only: 1 } }),
    record({ meta: { template: "went", pattern: "BABA" } }),
    record({ meta: { template: "got", pattern: "ABBA" } }),
  ];
  const facts = datasetFacts(dataset(records), null);
  expect(facts?.fields).toEqual([{ field: "template", clusters: 2 }, { field: "pattern", clusters: 2 }]);
  expect(built(form({ cluster: "template" }), { facts }).statistics).toEqual({ bootstrap: 1000, ci: 0.95, seed: 0, cluster: "template" });
  expect(refused(form({ cluster: "only" }), { facts })).toContain("Not every prompt has “only”");
  // The first two prompts share one template: one cluster, nothing to resample.
  expect(refused(form({ cluster: "template", limit: 2 }), { facts: datasetFacts(dataset(records), 2) })).toContain("there is one cluster");
  expect(formFromSpec(built(form({ cluster: "pattern" }), { facts })).cluster).toBe("pattern");
});

test("attribution patching estimates with one gradient or with integrated gradients over 2 to 64 steps", () => {
  const atp = form({ kind: "attribution_patching" });
  expect(built(atp).experiment).toEqual({ kind: "attribution_patching", direction: "clean_to_corrupt", method: "gradient", steps: null });
  const ig = built({ ...atp, atpMethod: "integrated_gradients", atpSteps: 16 });
  expect(ig.experiment).toEqual({ kind: "attribution_patching", direction: "clean_to_corrupt", method: "integrated_gradients", steps: 16 });
  for (const steps of [1, 65, 2.5]) expect(refused({ ...atp, atpMethod: "integrated_gradients", atpSteps: steps })).toContain("from 2 to 64");
  expect(formFromSpec(ig)).toMatchObject({ atpMethod: "integrated_gradients", atpSteps: 16 });
  // Integrated gradients cost one forward and backward pass per step and prompt.
  expect(workload(ig, 8, 2, 4, null, 0)).toBe(128);
  expect(workloadText(ig, 128)).toBe("That is 128 forward and backward passes, 16 for each of the 8 prompts, for every site at once.");
  expect(workloadText(built(atp), 8)).toContain("two forward passes and one backward pass for each of the 8 prompts");
});

test("the every-feature sweep chooses its features on a seeded share of the prompts, or on all of them", () => {
  const sae = { state: "ready" as const, info: { repo: "owner/sae", path: "layer_1", revision: "abc", site: "resid_pre" as const, layer: 1, d_in: 32, d_sae: 64, activation: "relu", k: null, normalize: "none", format: "test", note: "", fit: null } };
  const features = form({ kind: "attribution_patching", scope: { kind: "features", position: { kind: "last" }, top: 20, choose_on: null, seed: null } });
  expect(built(features, { sae }).scope).toEqual({ kind: "features", position: { kind: "last" }, top: 20, choose_on: null, seed: null });
  const held = { ...features, scope: { kind: "features" as const, position: { kind: "last" as const }, top: 20, choose_on: 0.25, seed: 3 } };
  expect(built(held, { sae }).scope).toEqual({ kind: "features", position: { kind: "last" }, top: 20, choose_on: 0.25, seed: 3 });
  expect(refused({ ...held, scope: { ...held.scope, choose_on: 1 } }, { sae })).toContain("between 0 and 100%");
  expect(refused({ ...held, scope: { ...held.scope, seed: null } }, { sae })).toContain("Give the split of the prompts a seed");
});

test("SAE features, splits of the prompts and donors are refused before a run as the server refuses them", () => {
  const sae = { state: "ready" as const, info: { repo: "owner/sae", path: "layer_1", revision: "abc", site: "resid_pre" as const, layer: 1, d_in: 32, d_sae: 64, activation: "relu", k: null, normalize: "none", format: "test", note: "", fit: null } };
  const feature = { kind: "sae_feature" as const, layer: 1, feature: 3, position: { kind: "last" as const } };
  const every = form({ kind: "attribution_patching", scope: { kind: "features", position: { kind: "last" }, top: 20, choose_on: null, seed: null } });
  // Features are estimated from one gradient only.
  expect(refused({ ...every, atpMethod: "integrated_gradients" }, { sae })).toContain("SAE features are estimated from a single gradient");
  expect(refused(form({ kind: "attribution_patching", atpMethod: "integrated_gradients", scope: { kind: "sites", sites: [feature] } }), { sae })).toContain("single gradient");
  // Patched or zero-ablated, in a run of their own.
  expect(refused(form({ kind: "ablation", baseline: { kind: "mean", reference: "corrupt" }, scope: { kind: "sites", sites: [feature] } }), { sae })).toContain("can be patched, zero-ablated");
  expect(built(form({ kind: "ablation", baseline: { kind: "zero" }, scope: { kind: "sites", sites: [feature] } }), { sae }).scope.kind).toBe("sites");
  expect(refused(form({ scope: { kind: "sites", sites: [feature, { kind: "head", layer: 1, head: 0, position: { kind: "last" } }] } }), { sae })).toContain("either SAE features or model components");

  // A split rounds as Python does, halves to the even neighbor.
  expect([splitCount(5, 0.5), splitCount(3, 0.5), splitCount(7, 0.5), splitCount(10, 0.25), splitCount(8, 0.75)]).toEqual([2, 2, 4, 2, 6]);
  const three = datasetFacts(dataset([record(), record(), record()]), null);
  expect(refused({ ...every, scope: { ...every.scope, choose_on: 0.5, seed: 0 } as FormState["scope"] }, { sae, facts: three })).toContain("leaves 2 to choose them and 1 to report them");
  expect(refused(form({ kind: "steering", steerApplyTo: "corrupt", scope: { kind: "layer_components", components: ["resid_pre"], position: { kind: "last" } } }), { facts: datasetFacts(dataset([record(), record()]), null) }))
    .toContain("leaves 1 and 1");
  expect(refused(form({ kind: "ablation", baseline: { kind: "resample", pool: "corrupt", donors: 3, seed: 0 } }), { facts: three })).toContain("draws 3 donors from the other prompts, but there are only 2");
});

test("the metric's own words name a run's values; direct attribution stays with the logit difference", () => {
  const patch = { kind: "activation_patching" as const, direction: "clean_to_corrupt" as const };
  expect(measureWords(patch).delta).toBe("Δ logit diff");
  expect(measureWords(patch, { kind: "prob_diff" }).delta).toBe("Δ prob diff");
  expect(measureWords(patch, { kind: "prob_diff" }).deltaLegend).toBe("Change in the probability difference (P(answer) − P(distractor)) caused by the intervention.");
  expect(measureWords({ kind: "attribution_patching", direction: "clean_to_corrupt", method: "gradient", steps: null }, { kind: "logprob" }).delta).toBe("Estimated Δ log P(answer)");
  expect(measureWords({ kind: "direct_logit_attribution", prompts: "clean" }, { kind: "kl", target: "clean" }).effect).toBe("Share of logit diff");
  expect(metricWords({ kind: "kl", target: "clean" }).label).toBe("KL divergence from the clean prompt's prediction");
  // Runs from before metrics could be chosen measured the logit difference.
  expect(metricWords(null).label).toBe("logit difference");
  expect(metricWords({ kind: "something new" }).short).toBe("logit diff");
});

test("analyses read the prompts for the run's metric, or the form's", () => {
  const spec = built(form({ metric: "prob_diff", limit: 3 }));
  const base = { ...useStore.getInitialState(), project: { session_id: "one" }, datasetPath: spec.dataset.path, runDetails: { run: { spec } } };
  const run = analysisContext({ ...base, activeRunId: "run", analysisSource: "run" });
  expect(run.metric).toEqual({ kind: "prob_diff", normalization: "dataset_gap" });
  expect(run.tokenMetric).toEqual(run.metric);
  const kl = analysisContext({ ...base, activeRunId: null, analysisSource: "form", form: form({ metric: "kl" }) });
  // Without a target there is nothing to measure the baseline with, but prompts still tokenize.
  expect(kl.metric).toBeNull();
  expect(kl.metricError).toContain("KL divergence");
  expect(kl.tokenMetric.kind).toBe("kl");
  const chosen = analysisContext({ ...base, activeRunId: null, analysisSource: "form", form: form({ metric: "kl", klTarget: "clean" }) });
  // The other analyses share a cache key; the baseline's follows the metric too.
  expect(chosen.key).toBe(kl.key);
  expect(chosen.baselineKey).not.toBe(kl.baselineKey);
});

test("every project-scoped route on the server says which project a tab means", () => {
  const app = readFileSync(fileURLToPath(new URL("../../src/logogram/server/app.py", import.meta.url)), "utf8");
  const block = app.match(/PROJECT_SCOPED = \(([\s\S]*?)\)/);
  const server = [...(block?.[1] ?? "").matchAll(/"([^"]+)"/g)].map((m) => m[1]);
  expect(server.length).toBeGreaterThan(10);
  for (const prefix of server) {
    expect(PROJECT_SCOPED).toContain(prefix);
    if (prefix.endsWith("/")) {
      expect(isProjectScoped(`${prefix}generate`), prefix).toBe(true);
    } else {
      for (const path of [prefix, `${prefix}/x`, `${prefix}?a=1`]) expect(isProjectScoped(path), path).toBe(true);
    }
  }
  for (const path of ["/api/state", "/api/tasks", "/api/sae/load", "/api/sae/suggestions?model=x", "/api/models/load", "/api/projectsx", "/api/runsx"]) {
    expect(isProjectScoped(path), path).toBe(false);
  }
});
