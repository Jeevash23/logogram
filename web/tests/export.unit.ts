import { test, expect } from "@playwright/test";

import type { ModelStatus, RunDetail, Spec } from "../src/api/types";
import { buildSpec } from "../src/lib/buildSpec";
import { methodsText } from "../src/lib/export";
import { DEFAULT_FORM, type FormState } from "../src/lib/formState";

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

function run(patch: Partial<FormState>): RunDetail {
  const out = buildSpec({ ...DEFAULT_FORM, ...patch }, { model, datasetPath: "datasets/ioi.jsonl", datasetSha: "abc" });
  if ("error" in out) throw new Error(out.error);
  const spec: Spec = out.spec;
  return { id: "run-1", spec, summary: null, manifest: null, listing: null, folder: "experiments/run-1" };
}

test("the methods text says every choice in words, with no raw values", () => {
  const text = methodsText(run({ prependBos: false, limit: 40, batchSize: 16 }));
  expect(text).not.toMatch(/\btrue\b|\bfalse\b|[{}]|undefined|null/);
  expect(text).toContain("tokenized without a beginning-of-sequence token");
  expect(text).toContain("the first 40 of the file");
  expect(text).toContain("batches of 16");
  expect(text).toContain("SHA-256 abc");
  expect(text).toContain("copied from the clean prompt into the corrupt one");
  expect(text).toContain("95% percentile bootstrap intervals over prompts, from 1000 resamples with seed 0");
});

test("an ablation's methods text names its baseline in words", () => {
  const text = methodsText(run({ kind: "ablation", baseline: { kind: "zero" } }));
  expect(text).toContain("replaced by a baseline: zero");
  expect(text).not.toMatch(/[{}]/);
});

test("the methods text carries the spec's notes, and nothing when there are none", () => {
  expect(methodsText(run({}))).not.toContain("Notes");
  expect(methodsText(run({ notes: "  \n" }))).not.toContain("Notes");
  const text = methodsText(run({ notes: "Expect the name movers 9.9 and 10.0 to restore the answer.\n" }));
  expect(text).toContain("Notes saved with the spec: Expect the name movers 9.9 and 10.0 to restore the answer.\n\nThe run is run-1.");
});
