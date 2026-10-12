import { test, expect } from "@playwright/test";

import type { ModelInfo } from "../src/api/types";
import { CHAT_ADVICE, chatReason } from "../src/lib/chat";

function info(chat: NonNullable<ModelInfo["extra"]>["chat"]): ModelInfo {
  return {
    id: "google/gemma-3-270m-it",
    revision: "abc",
    architecture: "Gemma3ForCausalLM",
    n_layers: 18,
    n_heads: 4,
    d_model: 640,
    d_head: 256,
    d_mlp: 2048,
    d_vocab: 262144,
    n_ctx: 32768,
    n_params: null,
    dtype: "float32",
    device: "cpu",
    device_name: "CPU",
    process_weights: false,
    site_kinds: [],
    backend: "transformer_lens",
    backend_version: "4",
    extra: { chat },
  };
}

test("a model made for chat says why, and the others say nothing", () => {
  expect(chatReason(info({ instruct: true, template: true, reason: "its name marks it as made for chat (it)" }))).toBe(
    "Its name marks it as made for chat (it).",
  );
  // A base model with a chat template, a model loaded before the check existed, and none at all.
  expect(chatReason(info({ instruct: false, template: true, reason: null }))).toBeNull();
  expect(chatReason(info(undefined))).toBeNull();
  expect(chatReason(null)).toBeNull();
  // What to do is said the same way everywhere, and promises nothing.
  expect(CHAT_ADVICE).toContain("doesn't apply chat templates yet");
  expect(CHAT_ADVICE).toContain("Check the baseline");
});
