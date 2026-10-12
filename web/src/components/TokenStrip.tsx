import { Fragment, useEffect, useRef, useState, type KeyboardEvent } from "react";

import { api } from "../api/client";
import type { Answer, AnswerTokens, TokenStripData } from "../api/types";
import { answerText, capitalize, visibleToken } from "../lib/format";
import { useAnalysisContext } from "../lib/hooks";
import { useStore } from "../store/app";
import { Button, Icon } from "./ui";
import s from "./TokenStrip.module.css";

const EXPANDED_KEY = "logogram.tokensShown";

/** Whether the tokens are shown: as this viewer last left them, else unless the window is short. */
function initiallyExpanded(): boolean {
  try {
    const stored = localStorage.getItem(EXPANDED_KEY);
    if (stored !== null) return stored === "1";
  } catch {
    /* storage unavailable: use the default */
  }
  return window.innerHeight >= 760;
}

export function TokenStrip() {
  const dataset = useStore((st) => st.dataset);
  const datasetPath = useStore((st) => st.datasetPath);
  const index = useStore((st) => st.promptIndex);
  const setIndex = useStore((st) => st.setPromptIndex);
  const modelReady = useStore((st) => st.model.state === "ready");
  const context = useAnalysisContext();
  const [data, setData] = useState<TokenStripData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpandedState] = useState(initiallyExpanded);
  const setExpanded = (next: boolean) => {
    setExpandedState(next);
    try {
      localStorage.setItem(EXPANDED_KEY, next ? "1" : "0");
    } catch {
      /* remembered for this page only */
    }
  };
  const tokenPosition = useStore(st => st.tokenPosition);
  const singleKeys = useStore((st) => st.singleKeys);

  const record = dataset?.records[index];

  useEffect(() => {
    setData(null);
    setError(null);
    if (!datasetPath || !modelReady || !record || context.error) return;
    let cancelled = false;
    api.tokenize({ dataset: datasetPath, index, metric: context.tokenMetric, ...context.options }).then(
      (d) => !cancelled && setData(d),
      (e: Error) => !cancelled && setError(e.message),
    );
    return () => {
      cancelled = true;
    };
  }, [datasetPath, index, modelReady, context.key, context.tokenMetric.kind, record]);

  if (!dataset || !record) {
    return (
      <div className={s.strip}>
        <div className={s.placeholder}>
          {datasetPath ? "Loading prompts…" : "No prompts yet. Set up a prompt pair or a dataset in Prompts."}
        </div>
      </div>
    );
  }

  const meta = record.meta as { template?: unknown; pattern?: unknown } | undefined;
  // Where a generated prompt came from: its template, and for IOI its name order.
  const origin = [meta?.pattern, meta?.template].filter((v): v is string => typeof v === "string" && v !== "").join(" · ");

  return (
    <div className={s.strip}>
      <div className={s.top}>
        <div className={s.nav}>
          <Button size="small" variant="ghost" icon="chevronLeft" aria-label={singleKeys ? "Previous prompt ([)" : "Previous prompt"} disabled={index === 0} onClick={() => setIndex(index - 1)} />
          <span className={s.counter}>
            Prompt <strong>{index}</strong> <span className={s.of}>of {context.n}</span>
          </span>
          <Button size="small" variant="ghost" icon="chevronRight" aria-label={singleKeys ? "Next prompt (])" : "Next prompt"} disabled={index >= context.n - 1} onClick={() => setIndex(index + 1)} />
        </div>
        {origin && <span className={s.meta}>{origin}</span>}
        <div className={s.spacer} />
        {tokenPosition !== null && <Button size="small" variant="ghost" onClick={() => useStore.getState().setTokenPosition(null)}>Token {tokenPosition} · clear selection</Button>}
        <span className={s.answer}>
          answer <AnswerReading read={data?.answer} written={record.answer} />
          <span className={s.vs}>vs</span>
          distractor <AnswerReading read={data?.distractor} written={record.distractor} />
        </span>
        <Button size="small" variant="ghost" aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>{expanded ? "Hide tokens" : "Show tokens"}</Button>
      </div>

      {expanded && (!modelReady ? (
        <div className={s.raw}>
          <div>
            <span className={s.rowLabel}>clean</span> {record.clean}
          </div>
          <div>
            <span className={s.rowLabel}>corrupt</span> {record.corrupt}
          </div>
          <div className={s.hint}>Load a model to see how it tokenizes these prompts.</div>
        </div>
      ) : error || context.error ? (
        <div className={s.issue}>
          <Icon name="alert" size={14} /> {context.error ?? error}
        </div>
      ) : data ? (
        <TokenGrid data={data} />
      ) : (
        <div className={s.raw}>
          <div className="faint">Tokenizing…</div>
        </div>
      ))}
    </div>
  );
}

const ROWS = ["clean", "corrupt"] as const;

/** The grid's columns: in each, a clean token's index and a corrupt token's, null where that
 * prompt has none. Prompts of equal length line up position by position; the server lines up
 * prompts of different lengths by their labels and last tokens, which a run matches. */
function gridColumns(data: TokenStripData): [number | null, number | null][] {
  if (data.columns) return data.columns;
  const n = Math.max(data.clean.tokens.length, data.corrupt.tokens.length);
  return Array.from({ length: n }, (_, i) => [i < data.clean.tokens.length ? i : null, i < data.corrupt.tokens.length ? i : null]);
}

/** Each column's labels: a label whose tokens share the column once, one on a single prompt's token
 * there with that prompt named. */
function columnLabels(data: TokenStripData, columns: [number | null, number | null][]): string[] {
  const corrupt = data.corrupt_labels ?? data.labels;
  return columns.map(([ci, di]) => {
    const names: string[] = [];
    for (const label of new Set([...Object.keys(data.labels), ...Object.keys(corrupt)])) {
      const inClean = ci !== null && data.labels[label] === ci;
      const inCorrupt = di !== null && corrupt[label] === di;
      if (inClean && inCorrupt) names.push(label);
      else if (inClean) names.push(`${label} (clean)`);
      else if (inCorrupt) names.push(`${label} (corrupt)`);
    }
    return names.join(" ");
  });
}

export function TokenGrid({ data }: { data: TokenStripData }) {
  const selected = useStore(st => st.tokenPosition);
  const select = useStore(st => st.setTokenPosition);
  // Prompts of different lengths: each row keeps its own positions, and a selection its row.
  const uneven = !!data.columns;
  const columns = gridColumns(data);
  const n = columns.length;
  const [pickedRow, setPickedRow] = useState(0);
  const grid = useRef<HTMLDivElement>(null);
  const at = (row: number, col: number) => (col >= 0 && col < n ? columns[col][row] : null);
  const columnOf = (row: number, index: number) => columns.findIndex((c) => c[row] === index);
  // The tokens are one tab stop; the arrow keys move between them. The stop is the token last
  // focused, else the selected position, else the first token.
  const present = (row: number, col: number) => at(row, col) !== null;
  const [last, setLast] = useState<{ row: number; col: number } | null>(null);
  const selectedRow = uneven ? pickedRow : 0;
  const selectedCol = selected === null ? -1 : columnOf(selectedRow, selected);
  const stop = last && present(last.row, last.col)
    ? last
    : selectedCol >= 0
      ? { row: selectedRow, col: selectedCol }
      : { row: 0, col: Math.max(0, columns.findIndex((c) => c[0] !== null)) };
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const cell = (e.target as HTMLElement).closest<HTMLElement>("[data-col]");
    if (!cell || e.altKey || e.ctrlKey || e.metaKey) return;
    let row = Number(cell.dataset.row);
    let col = Number(cell.dataset.col);
    const along = (from: number, step: number) => {
      for (let c = from + step; c >= 0 && c < n; c += step) if (present(row, c)) return c;
      return from;
    };
    if (e.key === "ArrowLeft") col = along(col, -1);
    else if (e.key === "ArrowRight") col = along(col, 1);
    else if (e.key === "Home") col = along(-1, 1);
    else if (e.key === "End") col = along(n, -1);
    else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
      const other = row === 0 ? 1 : 0;
      if (!present(other, col)) return;
      row = other;
    } else return;
    e.preventDefault();
    setLast({ row, col });
    grid.current?.querySelector<HTMLElement>(`[data-row="${row}"][data-col="${col}"]`)?.focus();
  };
  // A column differs where its two tokens do, or where one prompt has none there.
  const differs = uneven
    ? new Set(columns.flatMap(([ci, di], c) => (ci === null || di === null || data.clean.ids[ci] !== data.corrupt.ids[di] ? [c] : [])))
    : new Set(data.differs);
  // With a chat template, the tokens outside the prompt are the template's (the clean prompt's
  // positions say which; the corrupt prompt's are the same when the two line up).
  const template = (row: number, i: number) =>
    !!data.prompt && (row === 0 || !uneven) && (i < data.prompt[0] || i >= data.prompt[1]);
  const labels = columnLabels(data, columns);
  const cols = Array.from({ length: n }, (_, c) => c);
  /** A row of positions: each column's token index in one prompt (in both, when they line up). */
  const positions = (row: number) => (
    <>
      <span className={s.rowLabel} />
      {cols.map((c) => (
        <span key={`p${row}${c}`} className={`${s.pos} ${row === 1 ? s.posBelow : ""} ${differs.has(c) ? s.diffCol : ""}`}>
          {columns[c][row] ?? ""}
        </span>
      ))}
    </>
  );
  return (
    <>
      <div className={s.scroller}>
        <div
          ref={grid}
          className={s.grid}
          style={{ gridTemplateColumns: `auto repeat(${n}, auto)` }}
          role="group"
          aria-label={
            uneven
              ? "Tokens of the clean and corrupt prompt, lined up by their labels, each prompt with its own positions. Arrow keys move between them; Enter selects a position."
              : "Tokens of the clean and corrupt prompt. Arrow keys move between them; Enter selects a position."
          }
          data-arrow-keys
          onKeyDown={onKeyDown}
        >
          {positions(0)}
          {ROWS.map((row, r) => (
            <Fragment key={row}>
              <span className={s.rowLabel}>{row}</span>
              {cols.map((c) => {
                const i = columns[c][r];
                const token = i === null ? undefined : data[row].tokens[i];
                const chat = i !== null && template(r, i);
                return (
                  <button type="button"
                    key={`${row}${c}`}
                    className={`${s.cell} ${differs.has(c) ? s.diff : ""} ${token === undefined ? s.missing : ""} ${chat ? s.template : ""}`}
                    title={i === null || token === undefined ? `No ${row} token here` : `${JSON.stringify(token)} · id ${data[row].ids[i]}${chat ? " · written by the chat template" : ""}`}
                    disabled={token === undefined}
                    aria-label={i === null ? `${row}: no token here` : `${row} token ${i}: ${token === undefined ? "missing" : visibleToken(token)}${chat ? ", the chat template's" : ""}`}
                    aria-pressed={i !== null && selected === i && (!uneven || pickedRow === r)}
                    data-row={r}
                    data-col={c}
                    tabIndex={stop.row === r && stop.col === c ? 0 : -1}
                    onFocus={() => setLast({ row: r, col: c })}
                    onClick={() => {
                      if (i === null) return;
                      const again = selected === i && (!uneven || pickedRow === r);
                      setPickedRow(r);
                      select(again ? null : i);
                    }}
                  >
                    {token === undefined ? "∅" : visibleToken(token) || "∅"}
                  </button>
                );
              })}
            </Fragment>
          ))}
          {uneven && positions(1)}
          {labels.some(Boolean) && (
            <>
              <span className={s.rowLabel} />
              {cols.map((c) => (
                <span key={`l${c}`} className={`${s.labels} ${differs.has(c) ? s.diffCol : ""}`}>
                  {labels[c]}
                </span>
              ))}
            </>
          )}
        </div>
      </div>
      {uneven && (
        <p className={s.note}>
          {data.clean.tokens.length} and {data.corrupt.tokens.length} tokens, lined up by{" "}
          {data.corrupt_labels ? "their labels and last tokens" : "their last tokens (only the clean prompt is labelled)"}. A run
          matches the two prompts there, each at its own position.
        </p>
      )}
      {data.issues.length > 0 && (
        <div className={s.issues}>
          {data.issues.map((issue) => (
            <div key={issue.kind + issue.message} className={s.issue} role="alert">
              <Icon name="alert" size={14} />
              <span>{capitalize(issue.message) || "This prompt has a tokenization issue the server didn't describe."}</span>
            </div>
          ))}
        </div>
      )}
    </>
  );
}

/** How many members of a set to show before saying how many more there are. */
const SHOWN = 4;

/**
 * An answer or distractor as the loaded model reads it: one token, any token of a set, or a
 * continuation of several tokens read in order. Without a model, as the dataset writes it.
 */
export function AnswerReading({ read, written }: { read?: AnswerTokens; written: Answer }) {
  const tokens = read?.tokens ?? (Array.isArray(written) ? written : null);
  const set = read ? !!read.alternatives : Array.isArray(written);
  if (set && tokens) {
    return (
      <span className={s.reading} title={tokens.map(visibleToken).join(" ")}>
        {tokens.slice(0, SHOWN).map((t, i) => <code key={i} className={s.tok}>{visibleToken(t)}</code>)}
        <span className={s.readingNote}>{tokens.length > SHOWN ? `+${tokens.length - SHOWN} more · ` : ""}any of {tokens.length}</span>
      </span>
    );
  }
  if (read && read.tokens.length > 1) {
    return (
      <span className={s.reading}>
        {read.tokens.map((t, i) => <code key={i} className={s.tok}>{visibleToken(t)}</code>)}
        <span className={s.readingNote}>{read.tokens.length} tokens, in order</span>
      </span>
    );
  }
  return <code className={s.tok}>{answerText(written)}</code>;
}
