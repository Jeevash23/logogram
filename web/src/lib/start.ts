// The guide's first page, Start here, for someone new to Logogram or to interpretability: one
// question asked with two prompts, what patching a head does and what its effect means, the parts
// of a model the map shows, the first hour with the IOI example, what that example finds, and what
// to try next. The words are data so that tests/guide.unit.ts can check them: their length and
// punctuation, that every control they name exists in the app, and that the IOI prompt and heads
// are the example's own (src/logogram/gallery.py). components/StartHere.tsx draws them. Text
// between ** and ** names a control, as in lib/walkthrough.ts.

import type { View } from "../store/app";
import { FIRST_RUN } from "./walkthrough";

/** A button on the page: the walkthrough, a view of the workbench, or the examples. */
export type StartAction =
  | { label: string; to: "tour" }
  | { label: string; to: "view"; view: View }
  | { label: string; to: "examples" };

export interface StartSection {
  id: "lead" | "pair" | "patch" | "parts" | "hour" | "finds" | "then";
  /** The section's heading; the lead has none. */
  title: string | null;
  paragraphs: string[];
  /** Terms and what each means, as a list of definitions. */
  terms?: { term: string; text: string }[];
  /** Numbered steps, each with a button if it has one. */
  steps?: { text: string; action?: StartAction }[];
  /** Buttons after the section's paragraphs. Those that open a view show only in the workbench. */
  actions?: StartAction[];
}

/** The line beside the Start here button on the first screens. */
export const START_NOTE = "New to Logogram, or to interpretability? What it measures, in five minutes, and a first experiment to run.";

/** The figure of the three runs, drawn after the section on patching, says this. */
export const FIGURE_CAPTION =
  "Three runs of GPT-2 small, each drawn as its 144 heads, a row per layer as on the map. The patched run is the corrupt one with head 9.9's output copied in from the clean run.";

export const START: StartSection[] = [
  {
    id: "lead",
    title: null,
    paragraphs: [
      "A language model answers by passing a prompt through a stack of layers. Logogram finds which parts of those layers an answer depends on: it changes one part at a time, and measures how far the answer moves. This page follows one example, in about five minutes.",
    ],
  },
  {
    id: "pair",
    title: "One question, two prompts",
    paragraphs: [
      "Given “When Mary and John went to the store, John gave a drink to”, GPT-2 small answers Mary. Repeat Mary instead of John, and it answers John. The first is the clean prompt, where the model shows the behavior; the second is the corrupt prompt, which changes only what the behavior depends on. Mary is the answer, and John the distractor.",
      "The model's score for the answer minus its score for the distractor is the logit difference. It is above zero on the clean prompt and below zero on the corrupt one; the gap between them, averaged over the example's pairs, is what every effect is measured against.",
    ],
  },
  {
    id: "patch",
    title: "Change one part, measure the answer",
    paragraphs: [
      "Run the corrupt prompt again, but copy one attention head's output in from the clean prompt, at every token, and let the rest run as usual. How far the answer moves back toward Mary shows what that head alone restores. This is activation patching, done for each of GPT-2 small's 144 heads in turn, on every pair.",
      "A head's effect is how far it moves the logit difference, as a share of the gap: 1 brings the clean answer back completely, 0 changes nothing, and a negative effect pushes further toward the distractor. Each pair's effect is kept, and the mean comes with an interval: where it would fall with other prompts like these.",
    ],
  },
  {
    id: "parts",
    title: "The parts of a model",
    paragraphs: [],
    terms: [
      {
        term: "Token",
        text: "A word or a piece of one. The model reads a prompt as tokens and predicts the next one.",
      },
      {
        term: "Layer",
        text: "One step of the computation, made of attention heads and an MLP. GPT-2 small has 12, layer 0 to layer 11, drawn on the map from the top down.",
      },
      {
        term: "Residual stream",
        text: "The running sum at each token that every layer reads and adds to. After the last layer, it is read out as the prediction.",
      },
      {
        term: "Attention head",
        text: "Part of a layer that moves information from one token to a later one, such as from John to the last word. Head 9 of layer 9 is written 9.9, or L9 H9.",
      },
      { term: "MLP", text: "The other part of a layer, which works on each token by itself." },
      {
        term: "Site",
        text: "Where an experiment intervenes, such as a head, an MLP's output or the residual stream, at one token or all of them.",
      },
    ],
  },
  {
    id: "hour",
    title: "Your first hour",
    paragraphs: [],
    steps: [
      {
        text: "Open the IOI example, 32 pairs like this one and the experiment above, and let the walkthrough show each step at its control.",
        action: { label: "Walk me through the example", to: "tour" },
      },
      {
        text: `Choose **Load a model**, beside the tabs, and load GPT-2 small. It downloads once from Hugging Face, ${FIRST_RUN.download}.`,
      },
      {
        text: "In Baseline, press **Check the baseline**: the clean prompts should prefer the answer on average, the corrupt ones the distractor. If they don't, nothing that follows means much.",
      },
      { text: `In Configure, press **Run**: the experiment takes ${FIRST_RUN.cpu}, a few seconds on a GPU.` },
      {
        text: "On the map, choose the strongest cell: the inspector shows the evidence behind its number, prompt by prompt.",
      },
      {
        text: "In Results, **Check robustness** reruns the experiment with one choice changed, and flags the heads whose conclusion changed. A finding should survive changes that shouldn't matter.",
      },
    ],
  },
  {
    id: "finds",
    title: "What the example finds",
    paragraphs: [
      "Expect about eight heads to stand out. The induction head 5.5 and the S-inhibition heads 8.6, 8.10 and 7.9 each bring back a fifth to a third of the gap, the name movers 9.9 and 10.0 less, and the negative name movers 10.7 and 11.10 push toward the distractor. Wang et al. (2022) found the same heads and traced their jobs: 5.5 notices that John repeats, the S-inhibition heads pass that on to the last word, and the name movers copy the other name, Mary, to the output.",
      "Two cautions come with every patching map. A head's effect counts everything after it: the S-inhibition heads restore a lot because the name movers act on what they write. And an effect describes these prompts on this model.",
    ],
  },
  {
    id: "then",
    title: "Then",
    paragraphs: [
      "**Your text**, in Explore, shows what the model predicts after each token of any text you type, and what each layer would predict: a quick way to find a behavior to test.",
      "The other examples ask other questions. In the greater-than example, MLPs 9 and 10 carry the largest effects, about a third of the gap each.",
      "For a project of your own, choose **Create project** on All projects. **Generate a task**, in Prompts, makes clean and corrupt pairs for a known behavior, and the experiment form groups its methods by the question each answers.",
    ],
    actions: [
      { label: "Open Your text", to: "view", view: "text" },
      { label: "Go to the examples", to: "examples" },
      { label: "Open Prompts", to: "view", view: "prompts" },
    ],
  },
];

/** Every piece of text on the page, as a reader sees it: titles, paragraphs, terms, steps, the
 * buttons and the figure's caption. */
export function startTexts(): string[] {
  return [
    ...START.flatMap((section) => [
      ...(section.title ? [section.title] : []),
      ...section.paragraphs,
      ...(section.terms ?? []).flatMap((t) => [t.term, t.text]),
      ...(section.steps ?? []).flatMap((step) => [step.text, ...(step.action ? [step.action.label] : [])]),
      ...(section.actions ?? []).map((a) => a.label),
    ]),
    FIGURE_CAPTION,
  ];
}

/** How many words the page has. */
export function startWords(): number {
  return startTexts()
    .join(" ")
    .replace(/\*\*/g, "")
    .split(/\s+/)
    .filter((word) => /[\p{L}\p{N}]/u.test(word)).length;
}
