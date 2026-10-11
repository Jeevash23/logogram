// Unsaved edits to a dataset's prompt pairs. A dataset file never changes once written, since runs
// record its hash: edits are kept in the tab, shown in the prompts table, and saved as a new
// dataset (POST /api/datasets/derive), which the server checks as it checks an import.

import type { Answer, DeriveRequest, PromptRecord } from "../api/types";
import { count, plural, visibleToken } from "./format";

export interface PromptEdits {
  /** The version of the file the edits were made to: they line up with its prompts only. */
  sha256: string;
  /** Edited prompt pairs as they will be saved, by index in the file. */
  replaced: Record<number, PromptRecord>;
  /** Indices of the prompt pairs to leave out, in increasing order. */
  removed: number[];
}

export function noEdits(sha256: string): PromptEdits {
  return { sha256, replaced: {}, removed: [] };
}

/** JSON with object keys sorted, so equal values compare equal whatever order they were written in. */
function stableJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (value && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, v]) => v !== undefined)
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
    return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${stableJson(v)}`).join(",")}}`;
  }
  return JSON.stringify(value) ?? "null";
}

/** Whether two prompt pairs say the same thing (a set's tokens in the same order). */
export function sameRecord(a: PromptRecord, b: PromptRecord): boolean {
  return stableJson(a) === stableJson(b);
}

/** The edits with prompt `index` replaced by `record`. Editing a pair back to the file's own is
 * no edit at all. */
export function withEdit(edits: PromptEdits, index: number, record: PromptRecord, original: PromptRecord): PromptEdits {
  const replaced = { ...edits.replaced };
  if (sameRecord(record, original)) delete replaced[index];
  else replaced[index] = record;
  return { ...edits, replaced };
}

/** The edits with these prompts left out of the new dataset, or kept again. A removed pair keeps
 * its edits, so restoring it brings them back. */
export function withRemoved(edits: PromptEdits, indices: readonly number[], removed: boolean): PromptEdits {
  const set = new Set(edits.removed);
  for (const i of indices) {
    if (removed) set.add(i);
    else set.delete(i);
  }
  return { ...edits, removed: [...set].sort((a, b) => a - b) };
}

/** How many prompt pairs the edits change: edited (and kept), and removed. */
export function editCounts(edits: PromptEdits | undefined): { edited: number; removed: number } {
  if (!edits) return { edited: 0, removed: 0 };
  const removed = new Set(edits.removed);
  const edited = Object.keys(edits.replaced).filter((k) => !removed.has(Number(k))).length;
  return { edited, removed: removed.size };
}

export function hasEdits(edits: PromptEdits | undefined): boolean {
  const { edited, removed } = editCounts(edits);
  return edited + removed > 0;
}

/** The edits in words: "2 prompt pairs edited and 1 removed". */
export function editsText(edits: PromptEdits | undefined): string {
  const { edited, removed } = editCounts(edits);
  if (edited && removed) return `${plural(edited, "prompt pair")} edited and ${count(removed)} removed`;
  if (edited) return `${plural(edited, "prompt pair")} edited`;
  return `${plural(removed, "prompt pair")} removed`;
}

/** The prompt pairs as they will be saved, by index in the file (removed ones too, to show them). */
export function editedRecords(records: PromptRecord[], edits: PromptEdits | undefined): PromptRecord[] {
  if (!edits || Object.keys(edits.replaced).length === 0) return records;
  return records.map((r, i) => edits.replaced[i] ?? r);
}

/** The request that saves the edits to `dataset` as a new dataset called `name`. A removed pair's
 * edits stay behind. */
export function deriveRequest(dataset: string, edits: PromptEdits, name: string): DeriveRequest {
  const removed = new Set(edits.removed);
  const replace: Record<number, PromptRecord> = {};
  for (const [key, record] of Object.entries(edits.replaced)) {
    if (!removed.has(Number(key))) replace[Number(key)] = record;
  }
  return { dataset, dataset_sha256: edits.sha256, name: name.trim(), remove: [...edits.removed], replace };
}

// -- names ------------------------------------------------------------------------------------

/** A dataset name's stem, as the server checks it (DATASET_NAME_RE in src/logogram/datasets.py). */
const NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$/;

/** Why a name can't be used for a new dataset, or null when it can. `taken` holds the file names
 * in the project. */
export function nameProblem(name: string, taken: ReadonlySet<string>): string | null {
  const stem = name.trim().replace(/\.jsonl$/, "");
  if (!stem) return "Give the new dataset a name.";
  if (!NAME.test(stem)) return "Use letters, digits, '.', '_' and '-' in the name, starting with a letter or digit.";
  if (taken.has(`${stem}.jsonl`)) return `${stem}.jsonl already exists. Choose another name.`;
  return null;
}

/** The name suggested for a dataset's edits: its own with "-edited", not yet taken. A run's saved
 * copy of its prompts is named by its hash: the dataset the run read is named instead, if known. */
export function suggestName(file: string, taken: ReadonlySet<string>): string {
  let stem = (file.split("/").pop() ?? "").replace(/\.jsonl$/, "");
  if (/^[0-9a-f]{64}$/.test(stem)) stem = "prompts";
  stem = stem.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[^A-Za-z0-9]+/, "") || "prompts";
  for (let k = 1; ; k++) {
    const suffix = k === 1 ? "-edited" : `-edited-${k}`;
    const name = `${stem.slice(0, 100 - suffix.length)}${suffix}`;
    if (!taken.has(`${name}.jsonl`)) return name;
  }
}

// -- the editor's fields ----------------------------------------------------------------------

export interface AnswerDraft {
  /** One answer (a token, or a continuation of several read in order), or a set of single tokens,
   * any of which counts. */
  kind: "text" | "set";
  text: string;
  /** A set's tokens, one per line. */
  members: string;
}

export interface SpanDraft {
  label: string;
  start: string;
  end: string;
}

/** A prompt pair as the editor's fields hold it. */
export interface RecordDraft {
  clean: string;
  corrupt: string;
  answer: AnswerDraft;
  distractor: AnswerDraft;
  positions: SpanDraft[];
  id: string;
  /** JSON text of an object; empty for none. */
  meta: string;
}

function answerDraft(answer: Answer): AnswerDraft {
  return typeof answer === "string"
    ? { kind: "text", text: answer, members: answer }
    : { kind: "set", text: answer[0] ?? "", members: answer.join("\n") };
}

/** An object as one line of JSON, spaced to read: {"template": "went", "pattern": "ABBA"}. */
function jsonLine(value: Record<string, unknown>): string {
  return `{${Object.entries(value).map(([k, v]) => `${JSON.stringify(k)}: ${JSON.stringify(v)}`).join(", ")}}`;
}

export function draftOf(record: PromptRecord): RecordDraft {
  return {
    clean: record.clean,
    corrupt: record.corrupt,
    answer: answerDraft(record.answer),
    distractor: answerDraft(record.distractor),
    positions: Object.entries(record.positions ?? {}).map(([label, [start, end]]) => ({ label, start: String(start), end: String(end) })),
    id: record.id ?? "",
    meta: record.meta ? jsonLine(record.meta) : "",
  };
}

/** A set's tokens from its field: one per line, blank lines left out. Spaces count. */
export function setMembers(text: string): string[] {
  return text.split("\n").filter((line) => line !== "");
}

/** The characters of the clean prompt a span covers, or null when it isn't a span of it. */
export function spanText(clean: string, span: SpanDraft): string | null {
  const start = Number(span.start);
  const end = Number(span.end);
  if (span.start.trim() === "" || span.end.trim() === "" || !Number.isInteger(start) || !Number.isInteger(end)) return null;
  if (start < 0 || end <= start || end > clean.length) return null;
  return clean.slice(start, end);
}

/**
 * The prompt pair the editor's fields describe, or the first thing to fix in them. These are the
 * server's checks in plainer words; the server checks the pair again when it is saved.
 */
export function recordOf(draft: RecordDraft): { record: PromptRecord; problem: null } | { record: null; problem: string } {
  const fail = (problem: string) => ({ record: null, problem }) as const;
  if (draft.clean === "") return fail("Write the clean prompt.");
  if (draft.corrupt === "") return fail("Write the corrupt prompt.");
  const answers: Record<"answer" | "distractor", Answer> = { answer: "", distractor: "" };
  for (const what of ["answer", "distractor"] as const) {
    const field = draft[what];
    if (field.kind === "text") {
      if (field.text === "") return fail(`Write the ${what}.`);
      answers[what] = field.text;
      continue;
    }
    const members = setMembers(field.members);
    if (members.length === 0) return fail(`List the tokens of the ${what}'s set, one per line.`);
    if (members.length > 1000) return fail(`The ${what}'s set has ${count(members.length)} tokens; it can have at most 1,000.`);
    const seen = new Set<string>();
    for (const member of members) {
      if (seen.has(member)) return fail(`The ${what}'s set lists “${visibleToken(member)}” twice. List each token once.`);
      seen.add(member);
    }
    answers[what] = members;
  }
  const spans: [string, [number, number]][] = [];
  const labels = new Set<string>();
  for (const span of draft.positions) {
    const label = span.label;
    if (!label.trim()) return fail("Name each position, or remove it.");
    if (label === "all" || label === "last" || /^-?\d+$/.test(label)) {
      return fail(`“${label}” can't name a position: “all”, “last” and numbers already name positions.`);
    }
    if (labels.has(label)) return fail(`Two positions are called “${label}”. Give each its own name.`);
    labels.add(label);
    const start = Number(span.start);
    const end = Number(span.end);
    if (span.start.trim() === "" || span.end.trim() === "" || !Number.isInteger(start) || !Number.isInteger(end)) {
      return fail(`Position “${label}” needs a start and an end: character offsets in the clean prompt.`);
    }
    if (start < 0 || end <= start) return fail(`Position “${label}” must end after it starts, at 0 or later.`);
    if (end > draft.clean.length) {
      return fail(`Position “${label}” ends at character ${end}, but the clean prompt has ${draft.clean.length}.`);
    }
    spans.push([label, [start, end]]);
  }
  let meta: Record<string, unknown> | undefined;
  if (draft.meta.trim()) {
    let value: unknown;
    try {
      value = JSON.parse(draft.meta);
    } catch {
      return fail("Meta isn't valid JSON. Write an object, such as {\"template\": \"went\"}, or leave it empty.");
    }
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      return fail("Meta must be a JSON object, such as {\"template\": \"went\"}, or empty.");
    }
    meta = value as Record<string, unknown>;
  }
  // In the file's order of fields; the empty ones are left out, as the server writes them.
  const record: PromptRecord = { clean: draft.clean, corrupt: draft.corrupt, answer: answers.answer, distractor: answers.distractor };
  if (spans.length) record.positions = Object.fromEntries(spans);
  if (draft.id !== "") record.id = draft.id;
  if (meta) record.meta = meta;
  return { record, problem: null };
}
