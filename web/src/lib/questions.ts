// The questions interpretability asks, and Logogram's methods grouped by them: where a behavior is
// computed, how it flows, what a site represents, and what adding a direction does. The experiment
// form lists its methods under these titles; the first screens and Start here say, for someone
// new, what each question finds out and where to begin with it. tests/guide.unit.ts checks that
// each question's page of the documentation exists, and that its title is the heading the page on
// choosing a method gives it.

import type { ExperimentKind } from "../api/types";
import type { DocLink } from "./help";

export type QuestionId = "where" | "flow" | "represent" | "steer";

export interface Question {
  id: QuestionId;
  /** The question, as the experiment form titles its group of methods. */
  title: string;
  /** What it finds out, in a sentence, for someone new. */
  text: string;
  /** Where to begin: a bundled example that asks it, else a page of the documentation. */
  example?: string;
  /** Where the documentation shows it worked through. */
  guide: DocLink;
}

export const QUESTIONS: Question[] = [
  {
    id: "where",
    title: "Where is it computed?",
    text: "Which heads, MLPs or positions an answer depends on.",
    example: "ioi-gpt2",
    guide: { title: "Replicate IOI on GPT-2", path: "docs/guides/replicate-ioi.md" },
  },
  {
    id: "flow",
    title: "How does it flow?",
    text: "Which paths carry the answer, and the circuit they form.",
    guide: { title: "Find a circuit with ACDC", path: "docs/concepts/acdc.md" },
  },
  {
    id: "represent",
    title: "What does a site represent?",
    text: "What a part of the model encodes, and whether the model uses it.",
    guide: { title: "Find a direction and test it", path: "docs/guides/find-and-test-a-direction.md" },
  },
  {
    id: "steer",
    title: "What does adding a direction do?",
    text: "How the answer moves when the model is pushed along a direction.",
    guide: { title: "Steering", path: "docs/concepts/steering.md" },
  },
];

/** The question with this id. */
export function question(id: QuestionId): Question {
  return QUESTIONS.find((q) => q.id === id)!;
}

/** The methods of the experiment form, each under the question it answers. */
export const METHODS: { value: ExperimentKind; title: string; detail: string; question: QuestionId }[] = [
  { value: "activation_patching", title: "Activation patching", detail: "Copy an activation from the other prompt of each pair.", question: "where" },
  { value: "ablation", title: "Ablation", detail: "Replace an activation with a baseline you choose.", question: "where" },
  {
    value: "attribution_patching",
    title: "Attribution patching",
    detail: "Estimate patching at every site from gradients, then verify the strongest.",
    question: "where",
  },
  {
    value: "direct_logit_attribution",
    title: "Direct logit attribution",
    detail: "Split the logit difference into what each component writes directly. No intervention.",
    question: "where",
  },
  {
    value: "path_patching",
    title: "Path patching",
    detail: "Patch a component's effect through chosen receivers only, holding the other heads.",
    question: "flow",
  },
  {
    value: "edge_attribution",
    title: "Edge attribution",
    detail: "Estimate every edge between components from gradients, then patch the strongest as a circuit.",
    question: "flow",
  },
  {
    value: "acdc",
    title: "Circuit discovery (ACDC)",
    detail: "Remove the edges between components one at a time, keeping those whose removal changes the metric.",
    question: "flow",
  },
  {
    value: "probe",
    title: "Linear probe",
    detail: "Train a probe at each site to read a label, and measure it on held-out pairs. Reads; doesn't intervene.",
    question: "represent",
  },
  {
    value: "das",
    title: "Learned subspace (DAS)",
    detail: "Learn a few directions at each site whose patching carries the effect, measured on held-out pairs.",
    question: "represent",
  },
  {
    value: "subspace_patching",
    title: "Subspace patching",
    detail: "Patch only the component of each site along a saved direction: how much of the effect it carries.",
    question: "represent",
  },
  {
    value: "steering",
    title: "Steering",
    detail: "Add the mean clean–corrupt difference, an SAE feature or a saved direction at several strengths, against a random control.",
    question: "steer",
  },
];

/** A question's methods, as a reader lists them: "Activation patching, ablation, …". */
export function methodsOf(id: QuestionId): string {
  return METHODS.filter((m) => m.question === id)
    .map((m, i) => (i === 0 ? m.title : m.title.charAt(0).toLowerCase() + m.title.slice(1)))
    .join(", ");
}
