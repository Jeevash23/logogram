// Corpora (plain texts in the project) and where a unit is most active in them: reading an
// uploaded file's fields, naming a corpus, describing a unit, and the scale its values are shown on.

import type { CorpusFormat, CorpusTop, DirectionListing, ModelInfo, ReadPosition, ReadSite, StreamKind, TopExample } from "../api/types";
import { count } from "./format";

/** What a corpus may be called, as the server checks it (logogram.corpora). */
export const CORPUS_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$/;

/** The format a file's name says it has; a text file's texts are split as `split` says. Null for
 * a file the app can't upload (compressed, or Parquet, which is read from Hugging Face or the
 * command line). */
export function formatOf(filename: string, split: "lines" | "paragraphs" = "lines"): CorpusFormat | null {
  const name = filename.toLowerCase();
  if (name.endsWith(".jsonl") || name.endsWith(".ndjson")) return "jsonl";
  if (name.endsWith(".json")) return "json";
  if (name.endsWith(".csv")) return "csv";
  if (name.endsWith(".tsv")) return "tsv";
  if (/\.(gz|zst|bz2|xz|zip|arrow|parquet)$/.test(name)) return null;
  return split;
}

/** Whether a format's rows have fields (JSON) or columns (CSV, TSV), so a text field is chosen. */
export function hasFields(format: CorpusFormat): boolean {
  return format !== "lines" && format !== "paragraphs";
}

/** A CSV or TSV line's cells, quotes read as CSV quotes them. */
export function splitRow(line: string, delimiter: string): string[] {
  const cells: string[] = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (quoted) {
      if (c === '"' && line[i + 1] === '"') {
        cell += '"';
        i++;
      } else if (c === '"') quoted = false;
      else cell += c;
    } else if (c === '"' && cell === "") quoted = true;
    else if (c === delimiter) {
      cells.push(cell);
      cell = "";
    } else cell += c;
  }
  cells.push(cell);
  return cells.map((x) => x.replace(/\r$/, ""));
}

/** The fields of a file's first rows, in order of first appearance: a JSON object's keys, or a
 * CSV or TSV header's columns. Empty for plain text, or when the rows can't be read (the server
 * says why when the file is imported). */
export function fieldsOf(text: string, format: CorpusFormat, rows = 50): string[] {
  const seen = new Set<string>();
  const add = (row: unknown) => {
    if (row && typeof row === "object" && !Array.isArray(row)) for (const key of Object.keys(row)) seen.add(key);
  };
  if (format === "csv" || format === "tsv") {
    const header = text.replace(/^﻿/, "").split("\n", 1)[0] ?? "";
    return header.trim() ? splitRow(header, format === "csv" ? "," : "\t").filter((h) => h.trim()) : [];
  }
  if (format === "json") {
    try {
      const value: unknown = JSON.parse(text);
      if (Array.isArray(value)) value.slice(0, rows).forEach(add);
      return [...seen];
    } catch {
      // Many datasets' .json files are JSON Lines: read them as such.
    }
  }
  if (format === "json" || format === "jsonl") {
    let read = 0;
    for (const line of text.split("\n")) {
      if (!line.trim()) continue;
      try {
        add(JSON.parse(line));
      } catch {
        break;
      }
      if (++read >= rows) break;
    }
  }
  return [...seen];
}

const TEXT_FIELDS = ["text", "sentence", "content", "document", "review", "prompt", "body", "question"];
const LABEL_FIELDS = ["label", "labels", "class", "category", "sentiment", "stars", "target", "topic"];

/** The field most likely to hold each text, or each text's label, among a file's fields. */
export function guessField(fields: string[], what: "text" | "label"): string | null {
  const wanted = what === "text" ? TEXT_FIELDS : LABEL_FIELDS;
  for (const name of wanted) {
    const found = fields.find((f) => f.toLowerCase() === name);
    if (found) return found;
  }
  return what === "text" ? (fields[0] ?? null) : null;
}

/** A corpus name made from a file's name: its stem, with characters a name can't have replaced. */
export function nameFromFile(filename: string): string {
  const base = (filename.split(/[\\/]/).pop() ?? "").replace(/\.[A-Za-z0-9]+$/, "");
  const name = base.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[^A-Za-z0-9]+/, "").replace(/[^A-Za-z0-9]+$/, "").slice(0, 80);
  return name || "corpus";
}

/** What is wrong with a name for a new corpus (or an export, or a direction), or null. */
export function nameProblem(name: string, taken: ReadonlySet<string>, what = "corpus"): string | null {
  const stem = name.trim();
  if (!stem) return `Give the ${what} a name.`;
  if (!CORPUS_NAME.test(stem)) return "Use letters, digits, '.', '_' and '-', starting with a letter or digit, at most 80 characters.";
  if (taken.has(stem)) return `A ${what} called ${stem} already exists. Choose another name.`;
  return null;
}

/** What a dataset's prompt pairs offer to read them by: the position labels every pair has, and
 * the fields of meta every pair has with a string, number or true/false (to mark them by). */
export function datasetFacts(records: { positions?: Record<string, unknown> | null; meta?: Record<string, unknown> | null }[]): { labels: string[]; metaFields: string[] } {
  if (records.length === 0) return { labels: [], metaFields: [] };
  const scalar = (v: unknown) => typeof v === "string" || typeof v === "number" || typeof v === "boolean";
  const labels = Object.keys(records[0].positions ?? {}).filter((label) => records.every((r) => r.positions && label in r.positions));
  const metaFields = Object.keys(records[0].meta ?? {}).filter((field) => records.every((r) => r.meta && scalar(r.meta[field])));
  return { labels, metaFields };
}

/** A whole number typed into a field, or null while it is empty or not a whole number. */
export function wholeNumber(text: string): number | null {
  if (!/^-?\d+$/.test(text.trim())) return null;
  return Number(text.trim());
}

// -- sites and positions, in words ----------------------------------------------------------------

export const STREAM_KINDS: StreamKind[] = ["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"];

/** A stream site in words, as the server writes it in a direction's description. */
export function siteWords(site: ReadSite): string {
  switch (site.kind) {
    case "resid_pre":
      return `the residual stream entering layer ${site.layer}`;
    case "resid_mid":
      return `the residual stream between layer ${site.layer}'s attention and MLP`;
    case "resid_post":
      return `the residual stream leaving layer ${site.layer}`;
    case "attn_out":
      return `layer ${site.layer}'s attention output`;
    case "mlp_out":
      return `layer ${site.layer}'s MLP output`;
  }
}

/** A site as the command line writes it, such as resid_post:6. */
export function siteName(site: ReadSite): string {
  return `${site.kind}:${site.layer}`;
}

export function positionWords(position: ReadPosition): string {
  switch (position.kind) {
    case "last":
      return "the last token";
    case "index":
      return position.index < 0 ? `token ${position.index} from the end` : `token ${position.index}`;
    case "label":
      return `the token labelled ${position.label}`;
    case "mean":
      return "the mean over the tokens after the beginning-of-sequence token";
  }
}

// -- units -------------------------------------------------------------------------------------------

/** Whether a direction was found on another model (or revision, or local model) than the one
 * loaded, where it may not mean the same thing. */
export function foundElsewhere(direction: DirectionListing, info: ModelInfo | undefined | null): boolean {
  const found = direction.model;
  if (!found || !info) return false;
  const source = info.source ?? "hub";
  if (found.source !== source) return true;
  if (found.source === "local") return found.sha256 !== (info.sha256 ?? null);
  return found.id !== info.id || (found.revision !== null && found.revision !== info.revision);
}

/** A direction's model in words: its id at a short revision, or a local model's name. */
export function directionModelText(direction: DirectionListing): string {
  const m = direction.model;
  if (!m) return "";
  return m.source === "hub" && m.revision ? `${m.id} at ${m.revision.slice(0, 7)}` : m.id;
}

/** Why a direction can't be read at the loaded model's residual stream, or null when it can. */
export function directionProblem(direction: DirectionListing, info: ModelInfo | undefined | null): string | null {
  if (direction.error) return direction.error;
  if (info && direction.width !== undefined && direction.width !== info.d_model) {
    return `It is ${count(direction.width)} wide, and ${info.id}'s residual stream is ${count(info.d_model)}.`;
  }
  return null;
}

// -- examples -------------------------------------------------------------------------------------

/** The largest magnitude among the values the examples show: the top of the ink scale their
 * tokens are shaded on (a most negative example is shaded by its magnitude). */
export function exampleBound(top: TopExample[], bottom: TopExample[] | null = null): number {
  let m = 0;
  for (const example of [...top, ...(bottom ?? [])]) {
    for (const v of example.values) if (v !== null && Number.isFinite(v)) m = Math.max(m, Math.abs(v));
  }
  return m > 0 ? m : 1;
}

/** How a reading went, in a sentence: how much was read, and how often the unit is above zero. */
export function readingText(result: CorpusTop): string {
  const cut = result.truncated ? `, ${count(result.truncated)} of them cut to ${count(result.settings.max_tokens)} tokens` : "";
  const share = (100 * result.active_share).toFixed(result.active_share < 0.01 && result.active_share > 0 ? 2 : 1);
  return `Over ${count(result.tokens)} tokens of ${count(result.texts_read)} ${result.texts_read === 1 ? "text" : "texts"}${cut}, it is above zero at ${share}% of them.`;
}

/** A histogram's bars as rectangles in a box of `width` × `height`, the tallest filling it. */
export function histogramBars(histogram: CorpusTop["histogram"], width: number, height: number): { x: number; y: number; w: number; h: number; count: number }[] {
  const { edges, counts } = histogram;
  const lo = edges[0];
  const hi = edges[edges.length - 1];
  const span = hi - lo || 1;
  const top = Math.max(1, ...counts);
  return counts.map((c, i) => {
    const x = ((edges[i] - lo) / span) * width;
    const w = Math.max(1, ((edges[i + 1] - edges[i]) / span) * width);
    const h = c === 0 ? 0 : Math.max(1, (c / top) * height);
    return { x, y: height - h, w, h, count: c };
  });
}
