// Finding a dataset's prompt pairs: a search through their text, and pages of rows.

import type { Answer, PromptRecord } from "../api/types";
import { count, plural } from "./format";

/** Rows on a page of the prompts table. */
export const PAGE_SIZE = 100;

function answerParts(answer: Answer): string[] {
  return typeof answer === "string" ? [answer] : answer;
}

/** A prompt pair's meta as one line: "template: went · pattern: ABBA". Empty without meta. */
export function metaText(record: PromptRecord): string {
  return Object.entries(record.meta ?? {})
    .map(([key, value]) => `${key}: ${typeof value === "string" ? value : JSON.stringify(value)}`)
    .join(" · ");
}

/**
 * What a search looks through for one prompt pair: the clean and corrupt prompts, the answer and
 * distractor (each token of a set), and meta's keys and values. Lowercase, with a line between
 * fields, so a search never matches across two of them.
 */
export function searchText(record: PromptRecord): string {
  const fields = [record.clean, record.corrupt, ...answerParts(record.answer), ...answerParts(record.distractor)];
  const meta = metaText(record);
  if (meta) fields.push(meta);
  return fields.join("\n").toLowerCase();
}

/** A search as it is matched: without surrounding spaces, and in lowercase. Empty finds every row. */
export function searchQuery(query: string): string {
  return query.trim().toLowerCase();
}

/** Indices of the rows whose search text contains the query, ignoring case; every index when the
 * query is empty. `texts` are the rows' searchText, in order. */
export function matchTexts(texts: readonly string[], query: string): number[] {
  const q = searchQuery(query);
  const out: number[] = [];
  for (let i = 0; i < texts.length; i++) if (!q || texts[i].includes(q)) out.push(i);
  return out;
}

/** Indices of the prompt pairs that contain the query in their clean or corrupt prompt, answer,
 * distractor or meta, ignoring case. */
export function matchRows(records: readonly PromptRecord[], query: string): number[] {
  return matchTexts(records.map(searchText), query);
}

/** Whether a search finds a prompt pair by its meta alone, which the table doesn't show: the row
 * then shows it, so the reason it was found can be seen. */
export function foundByMeta(record: PromptRecord, query: string): boolean {
  const q = searchQuery(query);
  if (!q) return false;
  const shown = [record.clean, record.corrupt, ...answerParts(record.answer), ...answerParts(record.distractor)];
  return !shown.some((field) => field.toLowerCase().includes(q)) && metaText(record).toLowerCase().includes(q);
}

export interface RowPage {
  /** The rows shown, by index in the dataset. */
  rows: number[];
  /** The page shown, counted from 0 and kept within the pages there are. */
  page: number;
  /** How many pages there are: at least one, empty when there are no rows. */
  pages: number;
  /** How many rows there are to page through. */
  total: number;
  /** Where the page starts among all the rows, counted from 0, and where it ends (exclusive). */
  start: number;
  end: number;
}

/** One page of rows (dataset indices, in order). A page past the last shows the last. */
export function pageRows(rows: readonly number[], page: number, size = PAGE_SIZE): RowPage {
  const pages = Math.max(1, Math.ceil(rows.length / size));
  const shown = Math.min(Math.max(0, Math.floor(Number.isFinite(page) ? page : 0)), pages - 1);
  const start = shown * size;
  const end = Math.min(rows.length, start + size);
  return { rows: rows.slice(start, end), page: shown, pages, total: rows.length, start, end };
}

/** Where a key moves to among `total` rows, from the row at `at`: the arrows one row, Page Up and
 * Page Down a page, Home and End to the first and last. Null for any other key. */
export function rowAfterKey(key: string, at: number, total: number, size = PAGE_SIZE): number | null {
  let to: number;
  switch (key) {
    case "ArrowDown":
      to = at + 1;
      break;
    case "ArrowUp":
      to = at - 1;
      break;
    case "PageDown":
      to = at + size;
      break;
    case "PageUp":
      to = at - size;
      break;
    case "Home":
      to = 0;
      break;
    case "End":
      to = total - 1;
      break;
    default:
      return null;
  }
  return total > 0 ? Math.max(0, Math.min(total - 1, to)) : null;
}

/** The page that shows a row, or null when the row isn't among them. */
export function pageOfRow(rows: readonly number[], row: number, size = PAGE_SIZE): number | null {
  const at = rows.indexOf(row);
  return at < 0 ? null : Math.floor(at / size);
}

/** Where a page is: "Rows 101–200 of 2,000", or during a search, "Matches 1–34 of 34". */
export function rangeText(page: RowPage, searching: boolean): string {
  if (page.total === 0) return searching ? "No matches" : "No rows";
  return `${searching ? "Matches" : "Rows"} ${count(page.start + 1)}–${count(page.end)} of ${count(page.total)}`;
}

// -- showing a pair ---------------------------------------------------------------------------

/** Word pairs compared at most when two prompts have different numbers of words: past it, words are
 * compared in place, as for prompts of the same length. */
const DIFF_CELLS = 200_000;

/**
 * One prompt's words (and the spaces between them), each marked where the other prompt doesn't
 * have it. Prompts with as many words compare them in place; otherwise the words in common are
 * the longest sequence of them both prompts share, so a name of two words doesn't mark every word
 * after it.
 */
export function wordDiff(text: string, other: string): { text: string; differs: boolean }[] {
  const a = text.split(/(\s+)/);
  const b = other.split(/(\s+)/);
  if (a.length === b.length || a.length * b.length > DIFF_CELLS) {
    return a.map((w, i) => ({ text: w, differs: w.trim() !== "" && w !== b[i] }));
  }
  const shared = sharedWords(a, b);
  return a.map((w, i) => ({ text: w, differs: w.trim() !== "" && !shared.has(i) }));
}

/** Positions in `a` of a longest common subsequence of `a` and `b`. */
function sharedWords(a: string[], b: string[]): Set<number> {
  const m = b.length;
  // longest[i][j]: the longest common subsequence of a[i:] and b[j:], one row after another.
  const longest = new Int32Array((a.length + 1) * (m + 1));
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      longest[i * (m + 1) + j] =
        a[i] === b[j]
          ? longest[(i + 1) * (m + 1) + j + 1] + 1
          : Math.max(longest[(i + 1) * (m + 1) + j], longest[i * (m + 1) + j + 1]);
    }
  }
  const shared = new Set<number>();
  let i = 0;
  let j = 0;
  while (i < a.length && j < m) {
    if (a[i] === b[j]) {
      shared.add(i);
      i++;
      j++;
    } else if (longest[(i + 1) * (m + 1) + j] >= longest[i * (m + 1) + j + 1]) i++;
    else j++;
  }
  return shared;
}

/** A prompt's labelled positions in reading order, each with the text it names: "IO “Mary” · S2
 * “John”". Empty without positions. */
export function labelsText(text: string, spans: Record<string, [number, number]> | undefined): string {
  return Object.entries(spans ?? {})
    .sort(([la, [a]], [lb, [b]]) => a - b || (la < lb ? -1 : la > lb ? 1 : 0))
    .map(([label, [start, end]]) => `${label} “${text.slice(start, end)}”`)
    .join(" · ");
}

/** What a search found: "34 of 2,000 rows match", "1 of 2,000 rows matches", "No rows match". */
export function matchText(matches: number, total: number): string {
  if (matches === 0) return "No rows match";
  return `${count(matches)} of ${plural(total, "row")} ${matches === 1 ? "matches" : "match"}`;
}
