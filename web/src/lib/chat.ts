import type { ChatTemplateSpec, ModelInfo } from "../api/types";
import { capitalize } from "./format";

/** Why a loaded model counts as made for chat, as a sentence; null when it doesn't. Without a
 * chat template, Logogram runs every prompt as plain text, not as such models are trained on. */
export function chatReason(info: ModelInfo | null | undefined): string | null {
  const chat = info?.extra?.chat;
  if (!chat?.instruct) return null;
  return `${capitalize(chat.reason ?? "it is made for chat")}.`;
}

/** Whether prompts can be wrapped in the model's own chat template: its tokenizer ships one. */
export function hasChatTemplate(info: ModelInfo | null | undefined): boolean {
  return !!info?.extra?.chat_template_sha256;
}

/** What to do about a model made for chat that reads plain text, the same everywhere it is said:
 * choose its chat template, when it ships one. */
export function chatAdvice(info: ModelInfo | null | undefined): string {
  return hasChatTemplate(info)
    ? "To read prompts as it reads a chat, choose its chat template in the experiment form (Configure, under Data and model). Check the baseline before trusting a run."
    : "It ships no chat template, so its prompts can only be read as plain text. Check the baseline before trusting a run; for tasks posed as plain text, the model's base version is the closer fit.";
}

/** The chat template choices, as the experiment form and Your text offer them. */
export type ChatChoice = "none" | ChatTemplateSpec["kind"];

export const CHAT_CHOICES: { value: ChatChoice; label: string }[] = [
  { value: "none", label: "None" },
  { value: "user_turn", label: "The person's message" },
  { value: "assistant_prefix", label: "The start of the reply" },
];

/** The template for a choice, keeping the messages already written. */
export function chatTemplateFor(choice: ChatChoice, current: ChatTemplateSpec | null): ChatTemplateSpec | null {
  if (choice === "none") return null;
  const system = current?.system ?? null;
  if (choice === "user_turn") return { kind: "user_turn", system };
  return { kind: "assistant_prefix", system, user: current?.kind === "assistant_prefix" ? current.user : "" };
}

/** A message as a spec holds it: an empty system message is none. */
export function systemMessage(text: string): string | null {
  return text.trim() ? text : null;
}

/** Text with its line breaks shown (⏎ before each), so a template's newlines can be seen. */
export function shownBreaks(text: string): string {
  return text.replace(/\n/g, "⏎\n");
}
