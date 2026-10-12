// Runs as the history shows them: what each is called, how a search finds it, which runs a
// filter shows, and the order and nesting of the history's rows.

import type { DerivedFrom, RunListing } from "../api/types";
import { usesFeatures } from "./buildSpec";
import { modelName } from "./hooks";
import { experimentShort, scopeShort } from "./spec";
import { fieldWords, membershipText, sweepsOf, sweepsOfRun, valueName } from "./sweeps";

/** What a run is called: the title someone gave it, or its spec's name. */
export function runTitle(run: Pick<RunListing, "name" | "title"> | null | undefined, fallback = "Run"): string {
  return run?.title || run?.name || fallback;
}

/** What the history calls a run: its title, or for a robustness check without one, the choice
 * it changed, and for a run of a sweep nested under the run it was made from, its value. */
export function historyLabel(run: RunListing): string {
  if (run.title) return run.title;
  const from = run.derived_from;
  if (from?.kind === "robustness") return `Robustness · ${from.change ?? run.name}`;
  if (from?.kind === "sweep" && from.run) return `Sweep · ${valueName(from.field, from.value)}`;
  return run.name;
}

export type HistoryFilter = "all" | "starred" | "archived";

/** Whether a filter shows a run. Archived runs are shown only by "archived", which shows only
 * them; "all" is every run that isn't archived. */
export function inFilter(run: RunListing, filter: HistoryFilter): boolean {
  if (filter === "archived") return !!run.archived;
  if (run.archived) return false;
  return filter === "starred" ? !!run.starred : true;
}

/** How many runs each filter shows. */
export function filterCounts(runs: RunListing[]): Record<HistoryFilter, number> {
  return {
    all: runs.filter((r) => inFilter(r, "all")).length,
    starred: runs.filter((r) => inFilter(r, "starred")).length,
    archived: runs.filter((r) => inFilter(r, "archived")).length,
  };
}

/** A run's method in words, such as "Patch clean→corrupt · Layer × head" (empty for a run whose
 * spec can't be read). */
export function methodText(run: RunListing): string {
  if (run.damaged) return "";
  try {
    return `${experimentShort(run.experiment)} · ${scopeShort(run.scope)}`;
  } catch {
    return ""; // an experiment this version doesn't know
  }
}

/** What a search looks through: the run's title and name, its id, its method, its model and its
 * dataset (and, for a run derived from another, the choice it changed). */
export function searchText(run: RunListing): string {
  const parts = [run.title, run.name, run.id, run.derived_from?.change, run.dataset_source];
  if (!run.damaged) {
    const kind = (run.experiment as { kind?: string }).kind ?? "";
    parts.push(kind, kind.replace(/_/g, " "), methodText(run), run.model_id, modelName(run.model_id), run.dataset);
  }
  return parts.filter(Boolean).join("\n").toLowerCase();
}

/** Whether a run matches a search: every word typed appears somewhere in what is searched. */
export function matchesRun(run: RunListing, query: string): boolean {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (words.length === 0) return true;
  const text = searchText(run);
  return words.every((w) => text.includes(w));
}

export interface HistoryRow {
  run: RunListing;
  depth: number;
  /** It came from a run that has been deleted (not one the search or the filter hides). */
  orphan: boolean;
}

/**
 * The history's rows: the runs a filter and a search show, newest first, with each run derived
 * from another (a robustness check, a rerun) nested under it while that one is shown too.
 */
export function historyRows(runs: RunListing[], filter: HistoryFilter, query: string): HistoryRow[] {
  const known = new Set(runs.map((r) => r.id));
  const shown = runs.filter((r) => inFilter(r, filter) && matchesRun(r, query));
  const ids = new Set(shown.map((r) => r.id));
  const children = new Map<string, RunListing[]>();
  const roots: RunListing[] = [];
  for (const r of shown) {
    const parent = r.derived_from?.run;
    if (parent && ids.has(parent)) {
      const list = children.get(parent) ?? [];
      list.push(r);
      children.set(parent, list);
    } else roots.push(r);
  }
  const out: HistoryRow[] = [];
  const visited = new Set<string>();
  const walk = (r: RunListing, depth: number) => {
    if (visited.has(r.id)) return;
    visited.add(r.id);
    const parent = r.derived_from?.run;
    out.push({ run: r, depth: Math.min(depth, 6), orphan: !!parent && !known.has(parent) });
    for (const c of children.get(r.id) ?? []) walk(c, depth + 1);
  };
  roots.forEach((r) => walk(r, 0));
  // A shared project's provenance can contain a cycle. Keep every run reachable.
  shown.forEach((r) => walk(r, 0));
  return out;
}

/** The runs the command palette offers: with nothing typed, the starred runs and then the newest
 * others; once something is typed, every run, archived ones too, for the search to match. Runs
 * whose spec can't be read can't be opened, so they aren't offered. */
export function paletteRuns(runs: RunListing[], query: string, newest = 12): RunListing[] {
  const openable = runs.filter((r) => !r.damaged);
  if (query.trim()) return openable;
  return [
    ...openable.filter((r) => r.starred && !r.archived),
    ...openable.filter((r) => !r.starred && !r.archived).slice(0, newest),
  ];
}

/** Runs to choose from in a picker, grouped: starred first, then the rest, then archived ones.
 * A single group without a label when nothing is starred or archived. */
export function pickerGroups(runs: RunListing[]): { label: string | null; runs: RunListing[] }[] {
  const starred = runs.filter((r) => r.starred && !r.archived);
  const archived = runs.filter((r) => r.archived);
  const rest = runs.filter((r) => !r.starred && !r.archived);
  if (starred.length === 0 && archived.length === 0) return [{ label: null, runs: rest }];
  return [
    { label: "Starred", runs: starred },
    { label: starred.length ? "Other runs" : "Runs", runs: rest },
    { label: "Archived", runs: archived },
  ].filter((g) => g.runs.length > 0);
}

const DERIVED_WORDS: Record<DerivedFrom["kind"], [string, string]> = {
  robustness: ["robustness check", "robustness checks"],
  rerun: ["rerun", "reruns"],
  verification: ["verification by patching", "verifications by patching"],
  sweep: ["run of a sweep", "runs of a sweep"],
};

/** What a run keeps when it is cancelled, by how it measures (src/logogram/partial.py). SAE
 * features, every one or chosen ones, are reported only when the run ends. */
export function keptText(run: RunListing | undefined): string {
  const kept = "What it has finished is kept and marked as partial";
  if (!run) return `${kept}. What is in progress isn't kept.`;
  if (usesFeatures(run.scope)) return "A run of SAE features reports nothing before it ends, so it keeps no results.";
  if (run.experiment.kind === "attribution_patching" || run.experiment.kind === "direct_logit_attribution") {
    return `${kept}: every site, over the prompts that finished. The prompts in progress aren't kept.`;
  }
  if (run.scope.kind === "site_sets") return `${kept}: the sets that finished. The set in progress isn't kept.`;
  if (run.experiment.kind === "steering") {
    const control = run.experiment.control ? ", each with its random control" : "";
    return `${kept}: the sites that finished${control}. The site in progress isn't kept.`;
  }
  if (run.scope.kind === "sites") return `${kept}: the sites that finished. The site in progress isn't kept.`;
  return `${kept}: the layers that finished. The layer in progress isn't kept.`;
}

/** What deleting a run takes with it, and what stays: said before it is deleted. */
export function deleteWarning(run: RunListing, runs: RunListing[]): string {
  const what = run.damaged
    ? "everything in it"
    : run.status === "draft"
      ? "its saved spec"
      : run.status === "finished"
        ? "the spec it ran, its per-prompt results, its summary and its manifest"
        : run.partial
          ? "the spec it ran, what it kept of what finished (per-prompt results and a summary) and its manifest"
          : "the spec it ran and its manifest";
  const parts = [`This deletes its folder, experiments/${run.id}, with ${what}. It can't be undone.`];
  const derived = runs.filter((r) => r.derived_from?.run === run.id);
  if (derived.length > 0) {
    const kinds = (Object.keys(DERIVED_WORDS) as DerivedFrom["kind"][])
      .map((kind) => {
        const n = derived.filter((r) => r.derived_from?.kind === kind).length;
        return n === 0 ? null : n === 1 ? `a ${DERIVED_WORDS[kind][0]}` : `${n} ${DERIVED_WORDS[kind][1]}`;
      })
      .filter(Boolean)
      .join(" and ");
    const one = derived.length === 1;
    parts.push(
      `${one ? "The run" : `The ${derived.length} runs`} derived from it (${kinds}) ${one ? "stays" : "stay"}, but no longer ${one ? "links" : "link"} back to it.`,
    );
  }
  // A sweep then has no results at this run's value.
  for (const { sweep, index } of sweepsOfRun(sweepsOf(runs), run.id)) {
    parts.push(`${membershipText(sweep, index, "It")} The sweep will have no results for that ${fieldWords(sweep.field).label}.`);
  }
  if (run.status === "finished" && !run.archived) parts.push("To keep it out of the way instead, archive it.");
  return parts.join(" ");
}
