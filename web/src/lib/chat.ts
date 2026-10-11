import type { ModelInfo } from "../api/types";
import { capitalize } from "./format";

/** Why a loaded model counts as made for chat, as a sentence; null when it doesn't. Logogram runs
 * every prompt as plain text, without the chat template such models are trained on. */
export function chatReason(info: ModelInfo | null | undefined): string | null {
  const chat = info?.extra?.chat;
  if (!chat?.instruct) return null;
  return `${capitalize(chat.reason ?? "it is made for chat")}.`;
}

/** What to do about it, the same everywhere it is said. */
export const CHAT_ADVICE =
  "Logogram doesn't apply chat templates yet. Check the baseline before trusting a run; for tasks posed as plain text, the model's base version is the closer fit.";
