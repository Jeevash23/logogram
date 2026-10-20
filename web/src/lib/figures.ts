// What the figures of Start here and the guide draw, as data, so that tests/guide.unit.ts can check
// it against what the bundled IOI example and the documentation say: the example's prompt pair,
// and the heads its run finds on GPT-2 small, with their effects and their roles.

/** The IOI example's question as GPT-2 small reads it: a token each (a comma is a token of its
 * own), and the one the corrupt prompt changes. */
export const PAIR = {
  clean: ["When", "Mary", "and", "John", "went", "to", "the", "store", ",", "John", "gave", "a", "drink", "to"],
  corrupt: ["When", "Mary", "and", "John", "went", "to", "the", "store", ",", "Mary", "gave", "a", "drink", "to"],
  /** The token the corruption changes: the repeated name. */
  changed: 9,
  answer: "Mary",
  distractor: "John",
} as const;

/** The prompt as text, with each token's space where GPT-2's tokens have one. */
export function promptText(tokens: readonly string[]): string {
  return tokens.reduce((text, token, i) => (i === 0 || /^[,.;:!?]$/.test(token) ? text + token : `${text} ${token}`), "");
}

export type IoiRole = "induction" | "s-inhibition" | "name-mover" | "negative-name-mover";

/** The IOI example's strongest heads on GPT-2 small and their effects, as the guide to replicating
 * IOI lists them (docs/guides/replicate-ioi.md), with each head's role in Wang et al. (2022). */
export const IOI_HEADS: { layer: number; head: number; effect: number; role: IoiRole }[] = [
  { layer: 5, head: 5, effect: 0.33, role: "induction" },
  { layer: 8, head: 6, effect: 0.3, role: "s-inhibition" },
  { layer: 8, head: 10, effect: 0.25, role: "s-inhibition" },
  { layer: 7, head: 9, effect: 0.21, role: "s-inhibition" },
  { layer: 9, head: 9, effect: 0.17, role: "name-mover" },
  { layer: 10, head: 0, effect: 0.11, role: "name-mover" },
  { layer: 10, head: 7, effect: -0.45, role: "negative-name-mover" },
  { layer: 11, head: 10, effect: -0.21, role: "negative-name-mover" },
];

/** The head the patching figure copies across: a name mover. */
export const PATCHED_HEAD = { layer: 9, head: 9 } as const;

/** The patched head's effect, from IOI_HEADS. */
export function patchedEffect(): number {
  return IOI_HEADS.find((h) => h.layer === PATCHED_HEAD.layer && h.head === PATCHED_HEAD.head)!.effect;
}

/** Where the figures put the clean and corrupt prompts' logit differences, as shares of half the
 * scale: a sketch of their sides of zero, not a measurement. A patched run sits its head's effect
 * of the way from the corrupt prompt's to the clean prompt's, which is what an effect is. */
export const METER = { clean: 0.74, corrupt: -0.6 } as const;

/** Where a run patched with a head of this effect puts the logit difference on the figures' scale. */
export function patchedAt(effect: number): number {
  return METER.corrupt + effect * (METER.clean - METER.corrupt);
}
