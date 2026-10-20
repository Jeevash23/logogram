// The walkthrough of the bundled IOI example: six steps, each pointing at a real control, found by
// its data-tour attribute (never by a class name or a position, which change). A step says when
// what it asks for is done, and the experiment's step moves on by itself once the run starts.
// tests/guide.unit.ts checks that every control named here carries its attribute in the code.

import type { View } from "../store/app";

/** The bundled example the walkthrough is written for. */
export const TOUR_EXAMPLE = "ioi-gpt2";

/** What the example's first run costs, said the same way by the walkthrough and by the guide's
 * first page (lib/start.ts). */
export const FIRST_RUN = { download: "about 0.5 GB", cpu: "about a minute on a CPU" } as const;

/** What a step's view and words depend on. */
export interface TourContext {
  /** A run is open whose results can be drawn: running, or ended with results. */
  hasRun: boolean;
  /** The open run finished, so it can be checked for robustness. */
  finished: boolean;
  /** A model is loaded, so the model map can be drawn without a run. */
  modelLoaded: boolean;
  /** The window is wide enough for the inspector to sit beside the view. */
  wide: boolean;
  /** The baseline of the prompts shown, once checked: how many clean prompts prefer the answer,
   * of how many. */
  baseline: { prefer: number; n: number } | null;
  /** A component is chosen on the map, so the inspector shows its evidence. */
  chosen: boolean;
}

export interface TourStep {
  id: string;
  title: string;
  /** The view the step shows. */
  view: (ctx: TourContext) => View;
  /** The controls the step outlines, by their data-tour names; it points at the first. */
  anchors: string[];
  /** A control to point at instead when none of the anchors is shown (a narrow window, say). */
  alternatives?: string[];
  side: "top" | "right" | "bottom" | "left";
  align: "start" | "center" | "end";
  /** What the step says, a paragraph each. Text between ** and ** names a control. */
  text: (ctx: TourContext) => string[];
  /** Once what the step asks for is done: a line saying so. */
  done?: (ctx: TourContext) => string | null;
  /** The step moves on by itself when this turns true while it is shown. */
  next?: (ctx: TourContext) => boolean;
}

/** Where the map is: on the results once a run is open, else on the model map of a loaded model. */
function mapView(ctx: TourContext): View {
  return ctx.hasRun || !ctx.modelLoaded ? "results" : "explore";
}

export const STEPS: TourStep[] = [
  {
    id: "prompts",
    title: "The prompt pairs",
    view: () => "prompts",
    anchors: ["prompts"],
    side: "bottom",
    align: "end",
    text: () => [
      "Each row is a pair. In the clean prompt one of two names repeats, and GPT-2 small should answer with the other; the corrupt prompt repeats the other name, so the right answer changes.",
      "Every effect compares the two. Both prompts of a pair have the same number of tokens, so that each position means the same in both, and the tokens that differ are marked.",
    ],
  },
  {
    id: "baseline",
    title: "Check the baseline",
    view: () => "baseline",
    anchors: ["baseline"],
    side: "bottom",
    align: "end",
    text: (ctx) => [
      "Before measuring what carries a behavior, check that the model shows it: the clean prompts should prefer the answer, and the corrupt ones the other name. The gap between them is what every effect is measured against.",
      ctx.modelLoaded
        ? "**Check the baseline** runs the model on both prompts of each pair, without changing anything."
        : "The check needs the model: choose **Load a model** beside the tabs, or press **Run** in the next step, which loads it.",
    ],
    done: (ctx) => (ctx.baseline ? `Checked: ${ctx.baseline.prefer} of ${ctx.baseline.n} clean prompts prefer the answer.` : null),
  },
  {
    id: "experiment",
    title: "The experiment",
    view: () => "experiment",
    // Beside Run, so the form's name and notes, and the summary above Run, stay in view.
    anchors: ["run", "experiment-form"],
    side: "left",
    align: "end",
    text: () => [
      "The form is the experiment's spec: the method, what to sweep, the metric and the statistics. Every choice that can change a number is here, and none is filled in for you.",
      `This one patches each head's output from the clean prompt into the corrupt one, at every position. Press **Run**: GPT-2 small downloads once, ${FIRST_RUN.download}, and the sweep takes ${FIRST_RUN.cpu}.`,
    ],
    done: (ctx) => (ctx.hasRun ? (ctx.finished ? "Its run has finished: the next steps read it." : "The run has started: the next step shows its results.") : null),
    // Pressing Run opens the run's results: the walkthrough follows it there.
    next: (ctx) => ctx.hasRun,
  },
  {
    id: "map",
    title: "Read the map",
    view: mapView,
    anchors: ["map"],
    side: "bottom",
    align: "end",
    text: (ctx) =>
      ctx.hasRun
        ? [
            ctx.finished
              ? "Each row is a layer, from layer 0 at the top, and each cell a head. Color is the effect, with its scale beside the map: cobalt where patching that head brings the answer back, amber where it pushes the other way."
              : "The results paint in here as each layer finishes, from layer 0 at the top, a cell per head. Color is the effect, with its scale beside the map: cobalt where patching that head brings the answer back, amber where it pushes the other way.",
            "An effect of 1 restores the clean answer completely, and 0 changes nothing. **Show values** prints each cell's number.",
          ]
        : [
            "When the run finishes, its results paint here layer by layer, from layer 0 at the top: cobalt where patching a head brings the answer back, amber where it pushes the other way, with the scale beside them.",
            "An effect of 1 restores the clean answer completely, and 0 changes nothing.",
          ],
  },
  {
    id: "site",
    title: "Open a site",
    view: mapView,
    anchors: ["inspector"],
    alternatives: ["inspector-button"],
    side: "left",
    align: "start",
    text: (ctx) => [
      ctx.wide
        ? "Choose a cell, with the mouse or the arrow keys, to open it here in the inspector."
        : "Choose a cell, with the mouse or the arrow keys, then **Inspector** to open it in a panel.",
      "It shows everything behind the number: the exact intervention and metric, the effect with its confidence interval, how many prompts flipped, and each prompt's own effect, the strongest and weakest first.",
    ],
    done: (ctx) => (ctx.chosen ? "Chosen: the inspector shows its evidence." : null),
  },
  {
    id: "robustness",
    title: "Check that it holds",
    view: () => "results",
    anchors: ["robustness"],
    side: "bottom",
    align: "end",
    text: (ctx) => [
      ctx.finished
        ? "A finding should survive changes that shouldn't matter. **Check robustness** reruns the sweep with one choice changed, such as the other direction or another baseline, and flags the heads whose conclusion changed."
        : "A finding should survive changes that shouldn't matter. Once the run finishes, **Check robustness** here reruns it with one choice changed, such as the other direction or another baseline, and flags the heads whose conclusion changed.",
      "**Start here**, the first page of **Help → Guide**, says what this example finds and what to try next. The guide's other pages explain the map and the numbers whenever you need them.",
    ],
  },
];

/** A paragraph's parts: plain text, and the names of controls to set in bold. */
export function emphasis(paragraph: string): { text: string; strong: boolean }[] {
  return paragraph
    .split("**")
    .map((text, i) => ({ text, strong: i % 2 === 1 }))
    .filter((part) => part.text !== "");
}
