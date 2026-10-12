// Sweeps as the history knows them: the runs that record one (derived_from of kind "sweep"),
// grouped by the sweep's id, with the run it was made from where it stands in for a value. The
// sweep's table (its sites at every value) comes from the server (src/logogram/sweeps.py).

import type { QueuedRun, RunListing, SweepKind, SweepMember, SweepStatus, SweepValue } from "../api/types";

/** What each field a sweep can vary is called (src/logogram/sweeps.py FIELDS). */
export const SWEEP_FIELDS: Record<string, { label: string; plural: string; kind: SweepKind }> = {
  "statistics.seed": { label: "bootstrap seed", plural: "bootstrap seeds", kind: "seed" },
  "experiment.baseline.seed": { label: "donor seed", plural: "donor seeds", kind: "seed" },
  "experiment.seed": { label: "split seed", plural: "split seeds", kind: "seed" },
  "scope.seed": { label: "feature split seed", plural: "feature split seeds", kind: "seed" },
  "dataset.path": { label: "dataset", plural: "datasets", kind: "dataset" },
  "model.revision": { label: "revision", plural: "revisions", kind: "revision" },
};

/** The words for a field (a field a newer version sweeps is named by its path). */
export function fieldWords(field: string): { label: string; plural: string; kind: SweepKind } {
  return SWEEP_FIELDS[field] ?? { label: field, plural: field, kind: "seed" };
}

/** A value as the sweep shows it: a seed, a dataset's file name, a revision (a commit short). */
export function valueLabel(field: string, value: SweepValue): string {
  const kind = fieldWords(field).kind;
  if (kind === "dataset" && typeof value === "string") return value.split("/").pop() || value;
  if (kind === "revision" && typeof value === "string" && /^[0-9a-f]{40}$/.test(value)) return value.slice(0, 7);
  return String(value);
}

/** A value in words, such as "bootstrap seed 2". */
export function valueName(field: string, value: SweepValue): string {
  return `${fieldWords(field).label} ${valueLabel(field, value)}`;
}

export interface SweepGroup {
  id: string;
  field: string;
  values: SweepValue[];
  /** The run it was made from (if any), and the value that run stands in for (if any). */
  original: string | null;
  standsIn: number | null;
  /** Its runs by value index, the run it was made from among them where it stands in. */
  runs: Record<number, RunListing>;
  created: string | null;
  /** What its spec is called: the original run's title or name, or a run's name without its value. */
  name: string;
}

function isSweep(d: RunListing["derived_from"] | undefined): d is SweepMember {
  return d?.kind === "sweep";
}

/** The project's sweeps, newest first, from the runs that record them (as the server finds them:
 * the first run's record names the values; where two runs claim a value, a finished one wins). */
export function sweepsOf(runs: RunListing[]): SweepGroup[] {
  const byId = new Map(runs.map((r) => [r.id, r]));
  const groups = new Map<string, { record: SweepMember; run: RunListing }[]>();
  for (const run of runs) {
    const d = run.derived_from;
    if (!isSweep(d)) continue;
    const list = groups.get(d.sweep) ?? [];
    list.push({ record: d, run });
    groups.set(d.sweep, list);
  }
  const out: SweepGroup[] = [];
  for (const [id, members] of groups) {
    members.sort((a, b) => a.record.index - b.record.index || (a.run.created ?? "").localeCompare(b.run.created ?? ""));
    const record = members[0].record;
    const same = JSON.stringify(record.values);
    const kept: Record<number, RunListing> = {};
    for (const { record: m, run } of members) {
      if (m.field !== record.field || JSON.stringify(m.values) !== same) continue;
      const already = kept[m.index];
      if (!already || (already.status !== "finished" && run.status === "finished")) kept[m.index] = run;
    }
    const original = record.run ? byId.get(record.run) : undefined;
    if (record.original !== null && original && !kept[record.original]) kept[record.original] = original;
    const created = members.map((m) => m.run.created ?? "").filter(Boolean).sort()[0] ?? null;
    const first = Object.entries(kept).sort(([a], [b]) => Number(a) - Number(b))[0];
    const suffix = first ? ` · ${valueName(record.field, record.values[Number(first[0])])}` : "";
    const name =
      record.original !== null && original
        ? original.title || original.name
        : first
          ? first[1].name.endsWith(suffix) ? first[1].name.slice(0, -suffix.length) : first[1].name
          : id;
    out.push({ id, field: record.field, values: record.values, original: record.run, standsIn: record.original, runs: kept, created, name });
  }
  return out.sort((a, b) => (b.created ?? "").localeCompare(a.created ?? "") || b.id.localeCompare(a.id));
}

/** The sweeps a run belongs to, and its value in each: the sweep it is a run of, and those whose
 * value it stands in for (it isn't run again for them). */
export function sweepsOfRun(sweeps: SweepGroup[], runId: string): { sweep: SweepGroup; index: number }[] {
  const out: { sweep: SweepGroup; index: number }[] = [];
  for (const sweep of sweeps) {
    for (const [index, run] of Object.entries(sweep.runs)) {
      if (run.id === runId) out.push({ sweep, index: Number(index) });
    }
  }
  return out;
}

/** A run's place in a sweep, as a sentence: "This run is bootstrap seed 2 of a sweep over 4
 * bootstrap seeds." */
export function membershipText(sweep: SweepGroup, index: number, subject = "This run"): string {
  const words = fieldWords(sweep.field);
  const value = valueName(sweep.field, sweep.values[index]);
  const over = `a sweep over ${sweep.values.length} ${words.plural}`;
  return index === sweep.standsIn && sweep.runs[index]?.id === sweep.original
    ? `${subject} stands in for ${value} in ${over}.`
    : `${subject} is ${value} in ${over}.`;
}

/** The value indexes of a sweep still waiting in the queue: they have no run yet. */
export function queuedValues(queue: QueuedRun[], sweepId: string): Set<number> {
  const out = new Set<number>();
  for (const q of queue) {
    const d = q.derived_from;
    if (isSweep(d) && d.sweep === sweepId) out.add(d.index);
  }
  return out;
}

/** What a value's status is called, where it isn't a finished run. */
export function statusText(status: SweepStatus, queued: boolean): string | null {
  if (status === "missing") return queued ? "queued" : "not run";
  return { finished: null, failed: "failed", cancelled: "cancelled", running: "running", draft: "not run", deleted: "deleted" }[status];
}

/** Seeds or revisions typed in a field, in order: separated by commas, spaces or new lines. */
export function parseValues(kind: SweepKind, text: string): { values: SweepValue[]; error: string | null } {
  const parts = text.split(/[\s,]+/).filter(Boolean);
  if (kind !== "seed") return { values: parts, error: null };
  const values: number[] = [];
  for (const part of parts) {
    if (!/^\d+$/.test(part) || Number(part) >= 2 ** 32) {
      return { values: [], error: `Seeds are whole numbers from 0 to 4,294,967,295; “${part}” isn't one.` };
    }
    values.push(Number(part));
  }
  return { values, error: null };
}
