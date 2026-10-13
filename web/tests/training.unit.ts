import { test, expect } from "@playwright/test";
import type { ModelInfo, ModelRef } from "../src/api/types";
import { freeName, lensFields, lensSpec, planText, progressText, saeFields, saeSpec, trainedOnLoaded } from "../src/lib/training";

const info = {
  id: "openai-community/gpt2",
  revision: "abc",
  architecture: "GPT2LMHeadModel",
  n_layers: 12,
  n_heads: 12,
  d_model: 768,
  d_head: 64,
  d_mlp: 3072,
  d_vocab: 50257,
  n_ctx: 1024,
  n_params: 124_000_000,
  dtype: "float32",
  device: "cpu",
  device_name: "CPU",
  process_weights: true,
  site_kinds: ["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out", "head"],
  backend: "transformer_lens",
  backend_version: "4",
  extra: { bos: true, prediction_method: "final_norm_logit_lens" },
} as ModelInfo;
const model: ModelRef = { id: info.id, source: "hub", revision: "abc", sha256: null, dtype: "float32", device: "cpu", process_weights: true };

test("an SAE's training spec states every choice, starting from Gao et al.'s", () => {
  const fields = saeFields(info, "reviews", ["gpt2-resid_post-6"]);
  expect(fields.name).toBe("gpt2-resid_post-6-2");
  const built = saeSpec(fields, model);
  expect(built.error).toBeUndefined();
  expect(built.spec).toEqual({
    logogram_sae_training: 1,
    name: "gpt2-resid_post-6-2",
    model,
    corpus: { name: "reviews", sha256: null },
    tokenization: { prepend_bos: true, max_tokens: 128 },
    site: "resid_post",
    layer: 6,
    architecture: { kind: "topk", k: 32 },
    d_sae: 6144,
    steps: 2000,
    batch_tokens: 1024,
    lr: 0.00033,
    betas: [0.9, 0.999],
    eps: 6.25e-10,
    seed: 0,
    held_out: 0.1,
    normalize: "mean_squared_norm",
    auxk: { k: 512, coefficient: 0.03125, dead_after: 102400 },
    log_every: 20,
    model_batch: 16,
  });
  expect(saeSpec({ ...fields, auxk: false }, model).spec?.auxk).toBeNull();
});

test("what can't train is said before it starts", () => {
  const fields = saeFields(info, "reviews", []);
  expect(saeSpec({ ...fields, name: "../x" }, model).error).toContain("letters, digits");
  expect(saeSpec({ ...fields, corpus: "" }, model).error).toBe("Choose a corpus to train on.");
  expect(saeSpec({ ...fields, k: "7000" }, model).error).toContain("can't be more than the number of features");
  expect(saeSpec({ ...fields, lr: "2" }, model).error).toContain("The learning rate must be more than 0 and at most 1");
  expect(saeSpec({ ...fields, steps: "1.5" }, model).error).toContain("Steps must be a whole number");
  expect(saeSpec({ ...fields, heldOut: "" }, model).error).toContain("held-out share");
  expect(saeSpec({ ...fields, auxkK: "0" }, model).error).toContain("AuxK's latents");
});

test("a lens's translators go on every layer but the last, or the layers listed", () => {
  const fields = lensFields(info, "reviews", ["gpt2-lens"]);
  const built = lensSpec(fields, model, info.n_layers);
  expect(built.spec).toMatchObject({ logogram_lens_training: 1, name: "gpt2-lens-2", layers: null, weight_decay: 0.001, steps: 250 });
  expect(lensSpec({ ...fields, everyLayer: false, layers: "4, 0-2" }, model, 12).spec?.layers).toEqual([0, 1, 2, 4]);
  expect(lensSpec({ ...fields, everyLayer: false, layers: "11" }, model, 12).error).toContain("Translators go on layers 0 to 10");
  expect(lensSpec({ ...fields, everyLayer: false, layers: "" }, model, 12).error).toContain("List the layers");
  expect(lensSpec({ ...fields, weightDecay: false }, model, 12).spec?.weight_decay).toBeNull();
});

test("a plan and a training's progress read as sentences", () => {
  const text = planText({
    refusal: null,
    texts: { corpus: "reviews", sha256: "a", total: 100, train: 90, held_out: 10, read: 40 },
    tokens: { train: 9000, used: 4000, held_out: 1000, per_step: 64, epochs: 3.2 },
    memory: { total: 2 * 1024 ** 3, available: 16 * 1024 ** 3, verdict: "fits" },
    seconds: { total: 90, collect: 20, train: 60, evaluate: 10 },
  });
  expect(text.reads).toBe("Trains on 40 of the 90 training texts: 4,000 tokens, each read 3.2 times. Measures on 10 held-out texts (1,000 tokens).");
  expect(text.memory).toBe("Holds about 2.0 GB at once, of 16.0 GB free: it fits.");
  expect(text.time).toBe("Takes about 1 min 30 s here (collecting 20 s, training 1 min 00 s, measuring 10 s), from one forward pass and one step timed now.");
  expect(progressText({ stage: "training", done: 50, total: 200 }, "SAE", "x")).toEqual({ text: "Step 50 of 200", share: 0.25 });
  expect(progressText({ stage: "writing" }, "lens", "y").text).toBe("Writing lenses/y");
  expect(freeName("a", ["a", "a-2"])).toBe("a-3");
});

test("something trained reads the model it was trained on, with its weights processed the same way", () => {
  const trained = { id: info.id, source: "hub" as const, revision: "abc", sha256: null, process_weights: true };
  expect(trainedOnLoaded(trained, info)).toBe(true);
  expect(trainedOnLoaded({ ...trained, revision: "def" }, info)).toBe(false);
  expect(trainedOnLoaded({ ...trained, process_weights: false }, info)).toBe(false);
  expect(trainedOnLoaded(null, info)).toBe(false);
});
