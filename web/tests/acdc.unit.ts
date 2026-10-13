import { test, expect } from "@playwright/test";

import type { DatasetDetail, ModelStatus, PromptRecord, Spec } from "../src/api/types";
import { buildSpec, datasetFacts, parseThreshold } from "../src/lib/buildSpec";
import { receiverPlace } from "../src/lib/edges";
import { roughDuration } from "../src/lib/format";
import { parseForm } from "../src/lib/formDraft";
import { DEFAULT_FORM, formFromSpec, type FormState } from "../src/lib/formState";
import { edgeLabel, experimentText, measureWords, receiverLabel, receiverWords, scopeFor, sharesKeys, suggestName, workload } from "../src/lib/spec";

const info = {
  id: "tiny-gpt2", revision: "test", architecture: "GPT2LMHeadModel", n_layers: 2, n_heads: 4, d_model: 32, d_head: 8,
  d_mlp: 128, d_vocab: 1000, n_ctx: 64, n_params: 1000, dtype: "float32", device: "cpu", device_name: "CPU",
  process_weights: true, site_kinds: ["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out", "head"],
  backend: "transformer_lens", backend_version: "test",
};
const model: ModelStatus = {
  state: "ready",
  info,
  ref: { id: "tiny-gpt2", source: "hub", revision: "test", sha256: null, dtype: "float32", device: "cpu", process_weights: true },
};
/** A model whose heads share keys and values: four query heads, two key/value heads. */
const grouped: ModelStatus = { ...model, info: { ...info, id: "tiny-llama", extra: { n_key_value_heads: 2 } } };
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
/** The form switched to ACDC, as choosing the method does. */
const acdc = (patch: Partial<FormState> = {}): FormState => ({
  ...DEFAULT_FORM,
  kind: "acdc",
  scope: scopeFor("acdc", DEFAULT_FORM.scope),
  ...patch,
});

test("ACDC states its direction, its threshold and its rule, and neither threshold nor rule has a default", () => {
  expect(acdc().scope).toEqual({ kind: "graph", layers: null, logits: true });
  expect(refused(acdc())).toContain("Give ACDC a threshold");
  expect(refused(acdc({ acdcThreshold: "0.1" }))).toBe("Choose which change in the metric ACDC compares with its threshold.");
  const spec = built(acdc({ acdcThreshold: "0.1", acdcRule: "toward_source", direction: "corrupt_to_clean" }));
  expect(spec.experiment).toEqual({ kind: "acdc", direction: "corrupt_to_clean", threshold: 0.1, rule: "toward_source" });
  expect(spec.scope).toEqual({ kind: "graph", layers: null, logits: true });
  expect(spec.name).toBe("ACDC corrupt→clean, threshold 0.1 · the whole graph");
  expect(suggestName(spec)).toBe(spec.name);
  // A saved spec fills the form, and the form builds the same spec again.
  expect(formFromSpec(spec)).toMatchObject({ kind: "acdc", acdcThreshold: "0.1", acdcRule: "toward_source", direction: "corrupt_to_clean" });
  expect(built(formFromSpec(spec))).toEqual(spec);
  expect(experimentText(spec.experiment)).toBe("ACDC, threshold 0.1 on changes toward the source (corrupt → clean)");
  const either = built(acdc({ acdcThreshold: "2e-3", acdcRule: "either_way" }));
  expect(experimentText(either.experiment)).toBe("ACDC, threshold 0.002 on changes either way (clean → corrupt)");
  // Its values are changes between circuits, named as such.
  expect(measureWords(spec.experiment, spec.metric).effect).toBe("Change when removed");
  // The plan counts its passes, which depend on the model: no rows to count here.
  expect(workload(spec, 8, 2, 4, null, 0)).toBeNull();
  // Another method leaves the graph; ACDC keeps it, or takes the whole graph.
  expect(scopeFor("acdc", { kind: "graph", layers: [1], logits: false })).toEqual({ kind: "graph", layers: [1], logits: false });
  expect(scopeFor("acdc", { kind: "heads", position: { kind: "all" } })).toEqual({ kind: "graph", layers: null, logits: true });
  expect(refused(acdc({ acdcThreshold: "0.1", acdcRule: "either_way", scope: { kind: "heads", position: { kind: "all" } } }))).toContain(
    "ACDC removes the edges of the model's graph",
  );
});

test("ACDC's threshold is a number above 0, and answers of several tokens are refused", () => {
  expect(parseThreshold("0.25")).toEqual({ value: 0.25 });
  expect(parseThreshold(" 1e-3 ")).toEqual({ value: 0.001 });
  expect(parseThreshold("")).toHaveProperty("error");
  expect(parseThreshold("0")).toEqual({ error: "ACDC's threshold is a number above 0: with 0, no edge could be removed." });
  expect(parseThreshold("−1")).toHaveProperty("error");
  expect(parseThreshold("a lot")).toEqual({ error: "“a lot” isn't a number. Give the threshold as a number above 0, like 0.1." });
  const record = (patch: Partial<PromptRecord> = {}): PromptRecord => ({ clean: "When A and B", corrupt: "When A and C", answer: " B", distractor: " C", ...patch });
  const dataset: DatasetDetail = {
    name: "ioi.jsonl", path: "datasets/ioi.jsonl", sha256: "abc", n: 2,
    records: [record(), record({ answer: " Bob Smith" })], continuations: [1],
  };
  const form = acdc({ acdcThreshold: "0.1", acdcRule: "toward_source", metric: "logprob_diff" });
  expect(refused(form, { facts: datasetFacts(dataset, null) })).toContain("ACDC reads the metric at the last prompt position");
});

test("a form with ACDC is kept in the browser, and one kept before ACDC takes the form's starting values", () => {
  const form = acdc({ acdcThreshold: "0.05", acdcRule: "either_way", scope: { kind: "graph", layers: [1], logits: true } });
  expect(parseForm(JSON.parse(JSON.stringify(form)))).toEqual(form);
  const before: Record<string, unknown> = { ...DEFAULT_FORM };
  delete before.acdcThreshold;
  delete before.acdcRule;
  expect(parseForm(before)).toEqual(DEFAULT_FORM);
  expect(parseForm({ ...form, acdcRule: "absolute" })).toBeNull();
});

test("with grouped-query attention, a key or value receiver is a key/value head, named and drawn for its group", () => {
  expect(sharesKeys(model.info)).toBe(false);
  expect(sharesKeys(grouped.info)).toBe(true);
  expect(sharesKeys({ n_heads: 32, n_key_value_heads: 8 })).toBe(true);
  expect(sharesKeys({ n_heads: 12, n_key_value_heads: 12 })).toBe(false);
  const key = { kind: "head" as const, layer: 5, head: 1, input: "k" as const };
  expect(receiverLabel(key)).toBe("L5 H1 k");
  expect(receiverLabel(key, true)).toBe("L5 KV1 k");
  expect(receiverLabel({ ...key, input: "q" }, true)).toBe("L5 H1 q");
  expect(receiverWords({ ...key, input: "v" }, true)).toBe("L5 KV1 value");
  expect(edgeLabel({ sender: { kind: "head", layer: 2, head: 3 }, receiver: key }, true)).toBe("L2 H3 → L5 KV1 k");
  // Query heads 2 and 3 read key/value head 1: its key is drawn between them.
  expect(receiverPlace(key, 12, 2)).toEqual({ column: 11, head: 2.5 });
  expect(receiverPlace({ ...key, input: "q" }, 12, 2)).toEqual({ column: 11, head: 1 });
  expect(receiverPlace(key, 12)).toEqual({ column: 11, head: 1 });

  // A path into a key/value head the loaded model doesn't have is refused before it runs.
  const paths = (head: number): FormState => ({
    ...DEFAULT_FORM,
    kind: "path_patching",
    pathReceivers: [{ kind: "head", layer: 1, head, input: "k" }],
  });
  expect(built(paths(1), { model: grouped }).experiment).toMatchObject({ receivers: [{ kind: "head", layer: 1, head: 1, input: "k" }] });
  expect(refused(paths(2), { model: grouped })).toBe(
    "L1 KV2 doesn't exist: tiny-llama shares each key and value among 2 heads, so its layers have 2 key/value heads (0 to 1), and a key or value receiver names one of them.",
  );
  expect(built(paths(2)).experiment).toMatchObject({ receivers: [{ head: 2, input: "k" }] });
});

test("how long a run may take is said roughly", () => {
  expect(roughDuration(0.2)).toBe("about 1 second");
  expect(roughDuration(42)).toBe("about 42 seconds");
  expect(roughDuration(200)).toBe("about 3 minutes");
  expect(roughDuration(9000)).toBe("about 2.5 hours");
  expect(roughDuration(50000)).toBe("about 14 hours");
});
