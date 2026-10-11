import { test, expect } from "@playwright/test";

import type { ProjectInfo, TaskInfo } from "../src/api/types";
import { taskMetric } from "../src/lib/metrics";
import { useStore } from "../src/store/app";

const factual: TaskInfo = {
  id: "factual_recall",
  name: "Factual recall",
  description: "A country's capital.",
  metric: "logit_diff",
  metric_when: [{ option: "single_token_answers", value: false, metric: "logprob_diff" }],
  options: [
    { name: "templates", type: "choices", default: ["capital_of"], description: "", allowed: ["capital_of", "called"] },
    { name: "single_token_answers", type: "bool", default: true, description: "", allowed: [] },
  ],
  templates: [],
};

test("a task's options decide the metric that reads its answers", () => {
  expect(taskMetric(factual, {})).toBe("logit_diff");
  expect(taskMetric(factual, { single_token_answers: true })).toBe("logit_diff");
  expect(taskMetric(factual, { single_token_answers: false })).toBe("logprob_diff");
  expect(taskMetric({ ...factual, metric: "prob_diff", metric_when: [] }, {})).toBe("prob_diff");
});

test("a dataset read by another metric offers to switch the form to it, as one step of its history", () => {
  const project: ProjectInfo = { session_id: "tasks", name: "tasks", path: "projects/tasks", datasets: [], description: "" };
  const store = useStore.getState();
  store.leaveProject();
  useStore.setState({ project, screen: "workbench", view: "prompts" });
  try {
    // The same metric: nothing to offer.
    store.datasetSaved("datasets/ioi.jsonl", "logit_diff");
    expect(useStore.getState().notices.at(-1)).toMatchObject({ text: "Saved datasets/ioi.jsonl. Check the baseline next." });
    expect(useStore.getState().notices.at(-1)?.action).toBeUndefined();

    store.datasetSaved("datasets/greater-than.jsonl", "prob_diff");
    const offer = useStore.getState().notices.at(-1);
    expect(offer?.text).toBe("Saved datasets/greater-than.jsonl. Its answers are read with the probability difference; the experiment form uses the logit difference.");
    expect(offer?.action?.label).toBe("Use the probability difference");
    offer?.action?.run();
    expect(useStore.getState().form.metric).toBe("prob_diff");
    const replaced = useStore.getState().notices.find((n) => n.text === "The experiment form now uses the probability difference.");
    expect(replaced?.action?.label).toBe("Undo");
    // The switch is undone like any change to the form.
    store.undoForm();
    expect(useStore.getState().form.metric).toBe("logit_diff");
    store.redoForm();
    expect(useStore.getState().form.metric).toBe("prob_diff");
  } finally {
    store.leaveProject();
  }
});
