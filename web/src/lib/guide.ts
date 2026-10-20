// The in-app guide's pages: what each is called, the views it opens and the documentation it
// points to on GitHub, and the figures its pages draw. The first page's words are in lib/start.ts,
// the others' in components/GuideDialog.tsx; tests/guide.unit.ts checks that every page of the
// documentation named here exists, with its headings, and that every figure's line is a sentence.

import type { View } from "../store/app";
import type { DocLink } from "./help";

export type GuidePageId = "start" | "workspaces" | "map" | "numbers" | "keys";

export interface GuidePage {
  id: GuidePageId;
  title: string;
  /** Views the page talks about, each opened by a button. */
  views: { label: string; view: View }[];
  /** Where the documentation says more. */
  docs: DocLink[];
}

export const GUIDE_PAGES: GuidePage[] = [
  {
    id: "start",
    title: "Start here",
    // Its buttons are part of its text (lib/start.ts).
    views: [],
    docs: [
      { title: "Getting started", path: "docs/getting-started.md" },
      { title: "Activation patching", path: "docs/concepts/activation-patching.md" },
      { title: "Replicate IOI on GPT-2", path: "docs/guides/replicate-ioi.md" },
      { title: "A path through a study", path: "docs/concepts/choosing-a-method.md", anchor: "a-path-through-a-study" },
    ],
  },
  {
    id: "workspaces",
    title: "Workspaces",
    views: [
      { label: "Open Explore", view: "explore" },
      { label: "Open Experiment", view: "prompts" },
      { label: "Open Evidence", view: "results" },
    ],
    docs: [
      { title: "Getting started", path: "docs/getting-started.md" },
      { title: "The workbench", path: "docs/workbench.md" },
    ],
  },
  {
    id: "map",
    title: "The map and the logogram",
    views: [
      { label: "Open the model map", view: "explore" },
      { label: "Open the results", view: "results" },
    ],
    docs: [
      { title: "Reading the map", path: "docs/getting-started.md", anchor: "read-the-map" },
      { title: "The logogram", path: "docs/getting-started.md", anchor: "the-logogram" },
      { title: "Activation patching", path: "docs/concepts/activation-patching.md" },
      { title: "Edge attribution", path: "docs/concepts/edge-attribution.md" },
    ],
  },
  {
    id: "numbers",
    title: "Numbers and intervals",
    views: [{ label: "Open the experiment form", view: "experiment" }],
    docs: [
      { title: "Metrics and normalization", path: "docs/concepts/metrics.md" },
      { title: "Statistics", path: "docs/concepts/statistics.md" },
      { title: "Check a finding's robustness", path: "docs/guides/check-robustness.md" },
    ],
  },
  {
    id: "keys",
    title: "Keyboard",
    views: [],
    docs: [{ title: "Getting started: the keyboard", path: "docs/getting-started.md", anchor: "keyboard" }],
  },
];

/** A figure of the guide: what it shows, and the stages it moves through (components/figures). */
export interface GuideFigure {
  label: string;
  stages: { title: string; text: string }[];
}

export const GUIDE_FIGURES: Record<"workbench" | "map" | "logogram" | "interval", GuideFigure> = {
  workbench: {
    label: "The workbench, part by part",
    stages: [
      { title: "Workspaces", text: "Explore, Experiment and Evidence, across the top, each in its color." },
      { title: "Views", text: "The open workspace's views, as tabs. The number keys open them." },
      { title: "Model and prompts", text: "The loaded model, and the prompts the views read, beside the tabs." },
      { title: "The view", text: "What the open view shows: the map, the experiment form or a run's results." },
      { title: "Inspector", text: "The evidence behind the component chosen on the map." },
      { title: "Tokens", text: "The prompt pair shown, token by token, with the tokens that differ marked." },
      { title: "History and commands", text: "Every run of the project, and any action by name." },
    ],
  },
  map: {
    label: "How to read the model map",
    stages: [
      { title: "Layers", text: "A row per layer, from layer 0 at the top." },
      { title: "Components", text: "A cell per component: the residual stream, each head, the attention output and the MLP." },
      { title: "Effects", text: "Color is the effect: cobalt toward the other prompt, amber away from it, none where nothing changes." },
      { title: "Choose a cell", text: "Choose a cell to circle it, and the inspector shows the evidence behind its number." },
    ],
  },
  logogram: {
    label: "How a run's results write its logogram",
    stages: [
      { title: "No effect", text: "A run in which nothing matters writes a thin, even ring." },
      { title: "Positive layer", text: "Where a layer's strongest effect is positive, the ink swells outward, bleeding cobalt." },
      { title: "Negative layer", text: "Where it is negative, the ink swells inward, bleeding amber." },
      { title: "The IOI example", text: "Layers run clockwise from the top: the IOI example's heads write this glyph." },
    ],
  },
  interval: {
    label: "A bootstrap interval, built from the prompts' own effects",
    stages: [
      { title: "Each prompt", text: "Every prompt's own effect is kept." },
      { title: "The mean", text: "Their mean is the number the map shows." },
      { title: "Resampling", text: "Draw the prompts again, with replacement, many times, and take the mean of each draw." },
      { title: "The interval", text: "The middle 95% of those means is the interval: where the mean would fall with other prompts like these." },
    ],
  },
};
