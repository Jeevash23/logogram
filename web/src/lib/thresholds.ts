// The cutoffs behind the app's verdicts. They decide the words a result is described in ("the
// behavior is present", "the conclusions hold") and never change a number, and each verdict says
// the cutoff it used, so a reader can judge the result against their own.

/** The behavior is present when the clean prompts prefer the answer and the clean–corrupt gap in
 * the logit difference is above this: a smaller gap leaves little for an intervention to restore. */
export const PRESENT_GAP = 0.5;

/** Whether a baseline shows the behavior, by PRESENT_GAP. */
export function behaviorPresent(summary: { clean_logit_diff: number | null; gap: number | null } | null | undefined): boolean {
  return !!summary && (summary.clean_logit_diff ?? 0) > 0 && (summary.gap ?? 0) > PRESENT_GAP;
}

/** A comparison or robustness check holds when the sites' rank correlation is above this and no
 * top site changes. */
export const HOLDS_RHO = 0.8;

/** An SAE fits poorly when it explains less than this share of the activations' variance. */
export const POOR_FIT = 0.6;
