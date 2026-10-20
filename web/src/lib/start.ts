// The guide's first page, Start here, for someone new to Logogram or to interpretability: what it
// can find out, one question asked with two prompts, what patching a head does and what its effect
// means, the parts of a model the map shows, the first hour with the IOI example, what that example
// finds, and what to try next. Figures carry most of it: each moves through stages, a short line
// each (components/figures). The words are data so that tests/guide.unit.ts can check them: their
// length and punctuation, that every control they name exists in the app, and that the IOI prompt
// and heads are the example's own (src/logogram/gallery.py). components/StartHere.tsx draws them.
// Text between ** and ** names a control, as in lib/walkthrough.ts.

import type { View } from "../store/app";
import { QUESTIONS } from "./questions";
import { FIRST_RUN } from "./walkthrough";

/** A button on the page: the walkthrough, a view of the workbench, or the examples. */
export type StartAction =
  | { label: string; to: "tour" }
  | { label: string; to: "view"; view: View }
  | { label: string; to: "examples" };

/** A figure after a section's words: what it is, and its stages. The parts of a model take theirs
 * from the section's terms. */
export interface StartFigure {
  id: "pair" | "patching" | "parts" | "finds";
  /** What it shows, for its play button and assistive technology. */
  label: string;
  /** A line about the whole figure, under its stages. */
  caption?: string;
  stages: { title: string; text: string }[];
}

export interface StartSection {
  id: "lead" | "ask" | "pair" | "patch" | "parts" | "hour" | "finds" | "then";
  /** The section's heading; the lead has none. */
  title: string | null;
  paragraphs: string[];
  /** Terms and what each means, as a list of definitions. */
  terms?: { term: string; text: string }[];
  /** Numbered steps, each with a button if it has one. */
  steps?: { text: string; action?: StartAction }[];
  /** Buttons after the section's paragraphs. Those that open a view show only in the workbench. */
  actions?: StartAction[];
  figure?: StartFigure;
}

/** What Logogram does, in two sentences: under the tagline on the first screens, and the start of
 * Start here. */
export const PURPOSE =
  "Logogram finds which parts of a language model an answer depends on. It changes one part at a time, inside the model, and measures how far the answer moves.";

/** The line beside the Start here button on the first screens. */
export const START_NOTE = "New to Logogram, or to interpretability? What it measures, in five minutes, and a first experiment to run.";

/** The patching figure's line about all three runs. */
export const FIGURE_CAPTION = "Each run is drawn as GPT-2 small's 144 heads, as on the map.";

export const START: StartSection[] = [
  {
    id: "lead",
    title: null,
    paragraphs: [
      `${PURPOSE} This page follows one example, in about five minutes.`,
    ],
  },
  {
    id: "ask",
    title: "What you can ask",
    paragraphs: [],
    terms: QUESTIONS.map((q) => ({ term: q.title, text: q.text })),
  },
  {
    id: "pair",
    title: "One question, two prompts",
    paragraphs: [
      "Given “When Mary and John went to the store, John gave a drink to”, GPT-2 small answers Mary. Repeat Mary instead of John, and it answers John. The first is the clean prompt; the second, the corrupt prompt, changes only what the answer depends on.",
      "The model's score for the answer, Mary, minus its score for the distractor, John, is the logit difference.",
    ],
    figure: {
      id: "pair",
      label: "The clean and corrupt prompts, and the logit difference each gives",
      stages: [
        { title: "Clean prompt", text: "On the clean prompt GPT-2 small prefers Mary: the logit difference is above zero." },
        { title: "Corrupt prompt", text: "One name changes, and it prefers John: below zero." },
        { title: "The gap", text: "The gap between the two is what every effect is measured against." },
      ],
    },
  },
  {
    id: "patch",
    title: "Change one part, measure the answer",
    paragraphs: [
      "This is activation patching, done for each of the 144 heads, on every pair. A head's effect is how far it moves the answer, as a share of the gap: 1 brings Mary back completely, 0 changes nothing, and a negative effect pushes toward John. Each mean comes with an interval: where it would fall with other prompts like these.",
    ],
    figure: {
      id: "patching",
      label: "Activation patching: a head's output copied from the clean run into the corrupt run",
      caption: FIGURE_CAPTION,
      stages: [
        { title: "Clean run", text: "Run the clean prompt: GPT-2 small computes layer by layer, and prefers Mary." },
        { title: "Corrupt run", text: "Run the corrupt prompt the same way: it prefers John." },
        { title: "Copy one head", text: "Run the corrupt prompt again, with head 9.9's output copied in from the clean run." },
        { title: "Measure", text: "The answer moves back toward Mary by about a sixth of the gap: head 9.9's effect." },
      ],
    },
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
    // Its stages are the terms above.
    figure: { id: "parts", label: "The parts of a model, and one layer as a row of the map", stages: [] },
  },
  {
    id: "hour",
    title: "Your first hour",
    paragraphs: [],
    steps: [
      {
        text: "Open the IOI example: 32 pairs like this one, and the experiment above. The walkthrough shows each step at its control.",
        action: { label: "Walk me through the example", to: "tour" },
      },
      {
        text: `Choose **Load a model**, beside the tabs, and load GPT-2 small. It downloads once from Hugging Face, ${FIRST_RUN.download}.`,
      },
      {
        text: "In Baseline, press **Check the baseline**: the clean prompts should prefer the answer, the corrupt ones the distractor. If they don't, nothing that follows means much.",
      },
      { text: `In Configure, press **Run**: the experiment takes ${FIRST_RUN.cpu}, a few seconds on a GPU.` },
      {
        text: "On the map, choose the strongest cell: the inspector shows the evidence behind its number, prompt by prompt.",
      },
      {
        text: "In Results, **Check robustness** reruns the experiment with one choice changed. A finding should survive changes that shouldn't matter.",
      },
    ],
  },
  {
    id: "finds",
    title: "What the example finds",
    paragraphs: [
      "The induction head 5.5 and the S-inhibition heads 8.6, 8.10 and 7.9 each bring back a fifth to a third of the gap, the name movers 9.9 and 10.0 less, and the negative name movers 10.7 and 11.10 push toward the distractor, as Wang et al. (2022) found.",
      "Two cautions come with every patching map. A head's effect counts everything after it: the S-inhibition heads restore a lot because the name movers act on what they write. And an effect describes these prompts on this model.",
    ],
    figure: {
      id: "finds",
      label: "What the IOI example finds on GPT-2 small's heads",
      stages: [
        { title: "Layer by layer", text: "Results paint in as each layer finishes: cobalt brings Mary back, amber pushes toward John." },
        { title: "Induction head", text: "5.5 notices that John repeats." },
        { title: "S-inhibition heads", text: "8.6, 8.10 and 7.9 pass that on to the last word." },
        { title: "Name movers", text: "9.9 and 10.0 copy the other name, Mary, to the output." },
        { title: "Negative name movers", text: "10.7 and 11.10 push the other way, toward John." },
      ],
    },
  },
  {
    id: "then",
    title: "Then",
    paragraphs: [
      "**Your text**, in Explore, shows what the model predicts after each token of any text, and what each layer would predict: a quick way to find a behavior to test.",
      "The other examples ask other questions. In the greater-than example, MLPs 9 and 10 carry the largest effects, about a third of the gap each.",
      "For a project of your own, choose **Create project** on All projects. **Generate a task**, in Prompts, makes clean and corrupt pairs for a known behavior.",
    ],
    actions: [
      { label: "Open Your text", to: "view", view: "text" },
      { label: "Go to the examples", to: "examples" },
      { label: "Open Prompts", to: "view", view: "prompts" },
    ],
  },
];

/** A figure's stages: its own, or for the parts of a model, its section's terms. */
export function figureStages(section: StartSection): { title: string; text: string }[] {
  if (!section.figure) return [];
  return section.figure.stages.length ? section.figure.stages : (section.terms ?? []).map((t) => ({ title: t.term, text: t.text }));
}

/** Every piece of text on the page, as a reader sees it: titles, paragraphs, terms, steps, the
 * buttons, and the figures' stages and captions. */
export function startTexts(): string[] {
  return START.flatMap((section) => [
    ...(section.title ? [section.title] : []),
    ...section.paragraphs,
    ...(section.terms ?? []).flatMap((t) => [t.term, t.text]),
    ...(section.steps ?? []).flatMap((step) => [step.text, ...(step.action ? [step.action.label] : [])]),
    ...(section.actions ?? []).map((a) => a.label),
    // The parts of a model draw their terms, counted above.
    ...(section.figure?.stages ?? []).flatMap((stage) => [stage.title, stage.text]),
    ...(section.figure?.caption ? [section.figure.caption] : []),
  ]);
}

/** How many words the page has. */
export function startWords(): number {
  return startTexts()
    .join(" ")
    .replace(/\*\*/g, "")
    .split(/\s+/)
    .filter((word) => /[\p{L}\p{N}]/u.test(word)).length;
}
