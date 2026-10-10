import { Fragment, useEffect, useState } from "react";

import { api } from "../api/client";
import type { TokenStripData } from "../api/types";
import { answerText, capitalize, visibleToken } from "../lib/format";
import { useAnalysisContext } from "../lib/hooks";
import { useStore } from "../store/app";
import { Button, Icon } from "./ui";
import s from "./TokenStrip.module.css";

export function TokenStrip() {
  const dataset = useStore((st) => st.dataset);
  const datasetPath = useStore((st) => st.datasetPath);
  const index = useStore((st) => st.promptIndex);
  const setIndex = useStore((st) => st.setPromptIndex);
  const modelReady = useStore((st) => st.model.state === "ready");
  const context = useAnalysisContext();
  const [data, setData] = useState<TokenStripData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(true);
  const tokenPosition = useStore(st => st.tokenPosition);

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

  const meta = record.meta as { template?: string; pattern?: string } | undefined;

  return (
    <div className={s.strip}>
      <div className={s.top}>
        <div className={s.nav}>
          <Button size="small" variant="ghost" icon="chevronLeft" aria-label="Previous prompt ([)" disabled={index === 0} onClick={() => setIndex(index - 1)} />
          <span className={s.counter}>
            Prompt <strong>{index}</strong> <span className={s.of}>of {context.n}</span>
          </span>
          <Button size="small" variant="ghost" icon="chevronRight" aria-label="Next prompt (])" disabled={index >= context.n - 1} onClick={() => setIndex(index + 1)} />
        </div>
        {meta?.pattern && (
          <span className={s.meta}>
            {meta.pattern}
            {meta.template ? ` · ${meta.template}` : ""}
          </span>
        )}
        <div className={s.spacer} />
        {tokenPosition !== null && <Button size="small" variant="ghost" onClick={() => useStore.getState().setTokenPosition(null)}>Token {tokenPosition} · clear selection</Button>}
        <span className={s.answer}>
          answer <code className={s.tok}>{answerText(record.answer)}</code>
          <span className={s.vs}>vs</span>
          distractor <code className={s.tok}>{answerText(record.distractor)}</code>
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

export function TokenGrid({ data }: { data: TokenStripData }) {
  const selected = useStore(st => st.tokenPosition);
  const select = useStore(st => st.setTokenPosition);
  const n = Math.max(data.clean.tokens.length, data.corrupt.tokens.length);
  const differs = new Set(data.differs);
  const labelAt = new Map<number, string[]>();
  for (const [label, pos] of Object.entries(data.labels)) {
    labelAt.set(pos, [...(labelAt.get(pos) ?? []), label]);
  }
  const cols = Array.from({ length: n }, (_, i) => i);
  return (
    <>
      <div className={s.scroller}>
        <div className={s.grid} style={{ gridTemplateColumns: `auto repeat(${n}, auto)` }}>
          <span className={s.rowLabel} />
          {cols.map((i) => (
            <span key={`p${i}`} className={`${s.pos} ${differs.has(i) ? s.diffCol : ""}`}>
              {i}
            </span>
          ))}
          {(["clean", "corrupt"] as const).map((row) => (
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
      {(data.issues.length > 0 || data.answer.id === null || data.distractor.id === null) && (
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
