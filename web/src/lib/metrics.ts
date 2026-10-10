// What a run measures, in words (mirroring METRIC_LABELS and describe_metric in
// src/logogram/spec.py), and what each metric can read.

import type { MetricKind, MetricSpec, Side, TaskInfo } from "../api/types";

export interface MetricWords {
  /** In a sentence: "logit difference". */
  label: string;
  /** On a label or axis: "logit diff". */
  short: string;
  /** What is computed: "logit(answer) − logit(distractor)". */
  formula: string;
  /** For choosing it: one plain line, with what it computes. */
  detail: string;
}

export const METRICS: Record<MetricKind, MetricWords> = {
  logit_diff: {
    label: "logit difference",
    short: "logit diff",
    formula: "logit(answer) − logit(distractor)",
    detail: "logit(answer) − logit(distractor) at the last token. Reads single tokens, and sets of them.",
  },
  logprob_diff: {
    label: "log-probability difference",
    short: "log-prob diff",
    formula: "log P(answer) − log P(distractor)",
    detail: "log P(answer) − log P(distractor). Also reads answers of several tokens, token by token.",
  },
  logprob: {
    label: "answer log-probability",
    short: "log P(answer)",
    formula: "log P(answer)",
    detail: "log P(answer), ignoring the distractor. Reads answers of several tokens.",
  },
  prob: {
    label: "answer probability",
    short: "P(answer)",
    formula: "P(answer)",
    detail: "P(answer), summed over a set. Flattens where the model is already sure.",
  },
  prob_diff: {
    label: "probability difference",
    short: "prob diff",
    formula: "P(answer) − P(distractor)",
    detail: "P(answer) − P(distractor), as in the greater-than task's sets of years.",
  },
  kl: {
    label: "KL divergence",
    short: "KL",
    formula: "KL(P_target || P)",
    detail: "How far the whole next-token distribution is from a target prompt's. Reads no answer.",
  },
};

export const METRIC_KINDS = Object.keys(METRICS) as MetricKind[];

/** Metrics that read answers at the last position only, so not continuations of several tokens
 * (SINGLE_POSITION_METRICS in spec.py). */
export const SINGLE_POSITION_METRICS: readonly MetricKind[] = ["logit_diff"];

/** Whether a metric is a difference whose sign says which of answer and distractor is preferred. */
export function signedMetric(kind: string | null | undefined): boolean {
  return kind === "logit_diff" || kind === "logprob_diff" || kind === "prob_diff";
}

function known(kind: string | null | undefined): kind is MetricKind {
  return !!kind && kind in METRICS;
}

/** The words for a metric, or for a run's metric. A run from before metrics could be chosen
 * measured the logit difference. */
export function metricWords(metric: { kind?: string | null; target?: Side | null } | null | undefined): MetricWords {
  const kind = known(metric?.kind) ? metric.kind : "logit_diff";
  const words = METRICS[kind];
  if (kind === "kl" && metric?.target) {
    return {
      ...words,
      label: `KL divergence from the ${metric.target} prompt's prediction`,
      formula: `KL(P_${metric.target} || P)`,
    };
  }
  return words;
}

/** The metric in a sentence, as describe_metric in spec.py writes it. */
export function describeMetric(metric: MetricSpec): string {
  if (metric.kind === "kl") return `KL divergence from the ${metric.target} prompt's next-token distribution`;
  return metric.kind === "logit_diff" ? `${METRICS.logit_diff.formula} at the last position` : METRICS[metric.kind].formula;
}

/** The metric that reads a task's answers with these options (TaskInfo.recommended_metric in
 * src/logogram/tasks.py): an option left out takes its default. */
export function taskMetric(task: Pick<TaskInfo, "metric" | "metric_when" | "options">, options: Record<string, unknown>): MetricKind {
  for (const rule of task.metric_when) {
    const value = options[rule.option] ?? task.options.find((o) => o.name === rule.option)?.default;
    if (value === rule.value) return rule.metric;
  }
  return task.metric;
}
