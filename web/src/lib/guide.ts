// The in-app guide's pages: what each is called, the views it opens and the documentation it
// points to on GitHub. The words are in components/GuideDialog.tsx; tests/guide.unit.ts checks that
// every page of the documentation named here exists, with its headings.

import type { View } from "../store/app";
import type { DocLink } from "./help";

export type GuidePageId = "workspaces" | "map" | "numbers" | "keys";

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
