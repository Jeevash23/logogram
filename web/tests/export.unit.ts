import { test, expect } from "@playwright/test";

import type { RunDetail, Spec, Summary } from "../src/api/types";
import { exportFigure } from "../src/lib/export";

const spec: Spec = {
  logogram_spec: 4, name: "Chat heads", notes: "",
  model: { id: "my-chat-model", source: "local", revision: null, sha256: "a".repeat(64), device: "auto", dtype: "float32", process_weights: true },
  dataset: { path: "datasets/ioi.jsonl", sha256: "abc", limit: null },
  tokenization: { prepend_bos: false, chat_template: { kind: "user_turn", system: null } },
  experiment: { kind: "activation_patching", direction: "clean_to_corrupt" }, scope: { kind: "heads", position: { kind: "all" } },
  metric: { kind: "logit_diff", normalization: "dataset_gap" }, statistics: { bootstrap: 1000, seed: 0, ci: 0.95, cluster: null }, execution: { batch_size: 64 },
};

const site = (layer: number, head: number) => ({
  index: layer * 2 + head, row: layer, col: head, layer, head, kind: "head", position_key: "all", label: `L${layer} H${head}`,
  effect: { mean: 0.5, lo: 0.4, hi: 0.6, sd: 0.1, n: 8 }, delta: { mean: 1, lo: 0.9, hi: 1.1, sd: 0.1, n: 8 },
});

/** A run of four layers of two heads that kept the first two layers when it was cancelled. */
const summary = {
  n_prompts: 8,
  layout: {
    kind: "heads", row_title: "Layer", col_title: "Head",
    rows: [0, 1, 2, 3].map((i) => ({ key: String(i), label: String(i) })), cols: [0, 1].map((i) => ({ key: String(i), label: String(i) })),
  },
  sites: [site(0, 0), site(0, 1), site(1, 0), site(1, 1)],
  metric: { kind: "logit_diff" }, statistics: { ci: 0.95, bootstrap: 1000, seed: 0 },
  partial: {
    status: "cancelled", unit: "layers", done: 2, total: 4, sites: 4, sites_total: 8, prompts: 8, prompts_total: 8,
    note: "Stopped after 2 of 4 layers: these are the layers that finished.", left_out: [],
  },
  chat_template: { kind: "user_turn", system: null, user: null, sha256: "b".repeat(64), before: "<|user|>", after: "<|assistant|>" },
  warnings: [],
} as unknown as Summary;

const run = (s: Spec, sum: Summary): RunDetail => ({ id: "run-1", spec: s, summary: sum, manifest: null, listing: null, folder: "experiments/run-1" });

/** The lines a PNG figure writes, from a canvas that keeps them. */
async function figureLines(detail: RunDetail): Promise<string[]> {
  const lines: string[] = [];
  const context = new Proxy({} as Record<string | symbol, unknown>, {
    get: (target, key) => (key === "fillText" ? (text: string) => lines.push(text) : key in target ? target[key] : () => undefined),
    set: (target, key, value) => {
      target[key] = value;
      return true;
    },
  });
  const canvas = { width: 0, height: 0, getContext: () => context, toBlob: (done: (blob: Blob) => void) => done(new Blob(["png"])) };
  const scope = globalThis as Record<string, unknown>;
  const before = { window: scope.window, document: scope.document };
  scope.window = { setTimeout: () => 0 };
  scope.document = { fonts: { ready: Promise.resolve() }, createElement: (tag: string) => (tag === "canvas" ? canvas : { click() {} }) };
  try {
    await exportFigure(detail, "effect", false);
  } finally {
    scope.window = before.window;
    scope.document = before.document;
  }
  return lines;
}

test("a figure names a local model by its files, the chat template the prompts were read in, and how far a stopped run got", async () => {
  const lines = await figureLines(run(spec, summary));
  expect(lines).toContain("my-chat-model, local files aaaaaaaaaaaa · float32 · n = 8");
  expect(lines).toContain(
    "Prompts in the chat template, as the person's message (template SHA-256 bbbbbbbbbbbb) · batch 64 · 95% CI, 1000 resamples, seed 0",
  );
  expect(lines).toContain(
    "Stopped after 2 of 4 layers: these are the layers that finished. The run was cancelled before the end; nothing is drawn for the rest.",
  );
  expect(lines.join("\n")).not.toMatch(/unpinned|BOS/);

  // A run that kept every site over the prompts that finished draws every site.
  const prompts = { ...summary.partial!, status: "failed" as const, unit: "prompts" as const, note: "Stopped after 5 of 8 prompts: every site is measured on the prompts that finished." };
  expect(await figureLines(run(spec, { ...summary, partial: prompts }))).toContain(
    "Stopped after 5 of 8 prompts: every site is measured on the prompts that finished. The run failed before the end.",
  );
});

test("a figure of a finished run of a Hugging Face model names its revision and whether prompts began with BOS", async () => {
  const hub: Spec = {
    ...spec,
    model: { ...spec.model, id: "openai-community/gpt2", source: "hub", sha256: null, revision: "607a30d783dfa663caf39e06633721c8d4cfcd7e" },
    tokenization: { prepend_bos: true, chat_template: null },
  };
  const lines = await figureLines(run(hub, { ...summary, partial: null, chat_template: null }));
  expect(lines).toContain("openai-community/gpt2 @ 607a30d783dfa663caf39e06633721c8d4cfcd7e · float32 · n = 8");
  expect(lines).toContain("BOS on · batch 64 · 95% CI, 1000 resamples, seed 0");
  expect(lines.join("\n")).not.toMatch(/Stopped|chat template/);
  const unpinned = await figureLines(run({ ...hub, model: { ...hub.model, revision: null } }, { ...summary, partial: null, chat_template: null }));
  expect(unpinned).toContain("openai-community/gpt2 @ unpinned · float32 · n = 8");
});
