import { test, expect } from "@playwright/test";

import type { ModelInfo } from "../src/api/types";
import { chatAdvice, chatReason, chatTemplateFor, hasChatTemplate, shownBreaks, systemMessage } from "../src/lib/chat";

function info(chat: NonNullable<ModelInfo["extra"]>["chat"], template: string | null = null): ModelInfo {
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
    extra: { chat, chat_template_sha256: template },
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
  // What to do is said the same way everywhere: choose its chat template, when it ships one.
  const chat = { instruct: true, template: true, reason: "its name marks it as made for chat (it)" };
  expect(hasChatTemplate(info(chat, "ab".repeat(32)))).toBe(true);
  expect(chatAdvice(info(chat, "ab".repeat(32)))).toContain("choose its chat template in the experiment form");
  expect(chatAdvice(info(chat))).toContain("ships no chat template");
  for (const advice of [chatAdvice(info(chat)), chatAdvice(info(chat, "ab".repeat(32)))]) expect(advice).toContain("Check the baseline");
});

test("choosing how prompts are wrapped keeps the messages written", () => {
  expect(chatTemplateFor("none", { kind: "user_turn", system: "Be brief." })).toBeNull();
  expect(chatTemplateFor("user_turn", null)).toEqual({ kind: "user_turn", system: null });
  const reply = chatTemplateFor("assistant_prefix", { kind: "user_turn", system: "Be brief." });
  expect(reply).toEqual({ kind: "assistant_prefix", system: "Be brief.", user: "" });
  const kept = chatTemplateFor("assistant_prefix", { kind: "assistant_prefix", system: null, user: "Continue the story." });
  expect(kept).toEqual({ kind: "assistant_prefix", system: null, user: "Continue the story." });
  expect(chatTemplateFor("user_turn", kept)).toEqual({ kind: "user_turn", system: null });
  // An empty system message is none; the text of one is kept as written.
  expect(systemMessage("   ")).toBeNull();
  expect(systemMessage(" Be brief.")).toBe(" Be brief.");
  expect(shownBreaks("<|im_start|>user\n")).toBe("<|im_start|>user⏎\n");
});
