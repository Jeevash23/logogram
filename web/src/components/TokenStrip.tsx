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

export function TokenGrid({ data }: { data: TokenStripData }) {
  const selected = useStore(st => st.tokenPosition);
  const select = useStore(st => st.setTokenPosition);
  const n = Math.max(data.clean.tokens.length, data.corrupt.tokens.length);
  const grid = useRef<HTMLDivElement>(null);
  // The tokens are one tab stop; the arrow keys move between them. The stop is the token last
  // focused, else the selected position, else the first token.
  const present = (row: number, col: number) => col >= 0 && col < n && data[ROWS[row]].tokens[col] !== undefined;
  const [last, setLast] = useState<{ row: number; col: number } | null>(null);
  const stop = last && present(last.row, last.col)
    ? last
    : selected !== null && present(0, selected)
      ? { row: 0, col: selected }
      : { row: 0, col: Math.max(0, data.clean.tokens.findIndex((t) => t !== undefined)) };
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
  const differs = new Set(data.differs);
  const labelAt = new Map<number, string[]>();
  for (const [label, pos] of Object.entries(data.labels)) {
    labelAt.set(pos, [...(labelAt.get(pos) ?? []), label]);
  }
  const cols = Array.from({ length: n }, (_, i) => i);
  return (
    <>
      <div className={s.scroller}>
        <div
          ref={grid}
          className={s.grid}
          style={{ gridTemplateColumns: `auto repeat(${n}, auto)` }}
          role="group"
          aria-label="Tokens of the clean and corrupt prompt. Arrow keys move between them; Enter selects a position."
          data-arrow-keys
          onKeyDown={onKeyDown}
        >
          <span className={s.rowLabel} />
          {cols.map((i) => (
            <span key={`p${i}`} className={`${s.pos} ${differs.has(i) ? s.diffCol : ""}`}>
              {i}
            </span>
          ))}
          {ROWS.map((row, r) => (
            <Fragment key={row}>
              <span className={s.rowLabel}>{row}</span>
              {cols.map((i) => {
                const token = data[row].tokens[i];
                return (
                  <button type="button"
                    key={`${row}${i}`}
                    className={`${s.cell} ${differs.has(i) ? s.diff : ""} ${token === undefined ? s.missing : ""}`}
                    title={token === undefined ? "No token at this position" : `${JSON.stringify(token)} · id ${data[row].ids[i]}`}
                    disabled={token === undefined}
                    aria-label={`${row} token ${i}: ${token === undefined ? "missing" : visibleToken(token)}`}
                    aria-pressed={selected === i}
                    data-row={r}
                    data-col={i}
                    tabIndex={stop.row === r && stop.col === i ? 0 : -1}
                    onFocus={() => setLast({ row: r, col: i })}
                    onClick={() => select(selected === i ? null : i)}
                  >
                    {token === undefined ? "∅" : visibleToken(token) || "∅"}
                  </button>
                );
              })}
            </Fragment>
          ))}
          {labelAt.size > 0 && (
            <>
              <span className={s.rowLabel} />
              {cols.map((i) => (
                <span key={`l${i}`} className={`${s.labels} ${differs.has(i) ? s.diffCol : ""}`}>
                  {labelAt.get(i)?.join(" ")}
                </span>
              ))}
            </>
          )}
        </div>
      </div>
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
