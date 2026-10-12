import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";

import { api, type AnalysisOptions } from "../api/client";
import type { LookAttention, LookContinuation, LookFeatureAlong, LookFeatures, LookReport } from "../api/types";
import { MOD } from "../components/Header";
import { Heatmap, type Axis } from "../components/Heatmap/Heatmap";
import { ScaleBar } from "../components/Heatmap/ScaleBar";
import { Button, Callout, Checkbox, Empty, Field, Input, Segmented, Spinner, TextArea } from "../components/ui";
import { inkScale } from "../lib/color";
import { count, num, pct, visibleToken } from "../lib/format";
import { useAnalysisContext } from "../lib/hooks";
import { saeWritesText } from "../lib/spec";
import { useStore } from "../store/app";
import { SaidLines } from "./Continuations";
import s from "./views.module.css";
import t from "./TextView.module.css";

const EXAMPLE = "When Mary and John went to the store, John gave a drink to";

// What was typed and read, kept while the app is open so coming back shows it again.
const kept: { text: string; bos: boolean | null; topK: number; report: LookReport | null } = {
  text: EXAMPLE,
  bos: null,
  topK: 5,
  report: null,
};

/** Any text, read without making a dataset: the model's predictions after each token, what each
 * layer would predict at one of them, where a head attends and which features fire. */
export function TextView() {
  const model = useStore((st) => st.model);
  const sae = useStore((st) => st.sae);
  const selection = useStore((st) => st.selection);
  const theme = useStore((st) => st.theme);
  const context = useAnalysisContext();
  const [text, setText] = useState(kept.text);
  const [bos, setBos] = useState(kept.bos ?? context.options.prepend_bos);
  const [topK, setTopK] = useState(kept.topK);
  const [report, setReport] = useState<LookReport | null>(kept.report);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const options = { ...context.options, prepend_bos: bos };

  const read = async (position: number | null = null) => {
    if (!text.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const out = await api.look({ text, top_k: topK, position, ...options });
      kept.text = text;
      kept.bos = bos;
      kept.topK = topK;
      kept.report = out;
      setReport(out);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  if (model.state !== "ready") {
    return (
      <div className={s.view}>
        <Empty title="Load a model to read your text" action={<Button onClick={() => useStore.setState({ modelDialogOpen: true })}>Load a model</Button>}>
          The model reads the text live; nothing is saved.
        </Empty>
      </div>
    );
  }

  const stale = !!report && (kept.text !== text || kept.bos !== bos || kept.topK !== topK);
  return (
    <div className={s.view}>
      <div className={s.head}>
        <div className={s.titleBlock}>
          <h1 className={s.title}>What the model makes of your text</h1>
          <p className={s.subtitle}>
            Type any text: see what the model predicts after each token, what each layer would predict at one of them, where
            a head attends, and which SAE features fire. Nothing is saved, and nothing is changed in the model.
          </p>
        </div>
      </div>
      <div className={t.input}>
        <Field label="Text">
          <TextArea
            rows={3}
            value={text}
            maxLength={4000}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                e.preventDefault();
                void read();
              }
            }}
          />
        </Field>
        <div className={t.controls}>
          <Checkbox checked={bos} onChange={setBos}>
            Begin with the beginning-of-sequence token
          </Checkbox>
          <Segmented
            label="Predictions to show"
            value={String(topK)}
            onChange={(v) => setTopK(Number(v))}
            options={[
              { value: "3", label: "Top 3" },
              { value: "5", label: "Top 5" },
              { value: "10", label: "Top 10" },
            ]}
          />
          <Button variant="primary" onClick={() => void read()} disabled={busy || !text.trim()} tip={`Read the text (${MOD}+Enter in the text)`}>
            {busy ? "Reading…" : "Read"}
          </Button>
          {busy && <Spinner label="Reading the text" />}
        </div>
        {stale && <p className={s.small}>The text or the choices changed since it was read: press Read to read it again.</p>}
      </div>
      {error && <Callout tone="error">{error}</Callout>}
      {report && (
        <>
          <Tokens report={report} onChoose={(p) => void read(p)} busy={busy} />
          <NextTokens report={report} />
          <Continue key={`${kept.text}|${report.prepend_bos}`} text={kept.text} report={report} options={{ ...options, prepend_bos: report.prepend_bos }} />
          <Lens report={report} />
          <Attention text={kept.text} report={report} options={{ ...options, prepend_bos: report.prepend_bos }} theme={theme} initial={selection?.part === "head" && selection.head !== undefined ? { layer: selection.layer, head: selection.head } : null} />
          {sae.state === "ready" ? (
            <Features text={kept.text} options={{ ...options, prepend_bos: report.prepend_bos }} position={report.position} />
          ) : (
            <p className={s.small}>Load an SAE in Features to see which of its features fire on each token.</p>
          )}
        </>
      )}
    </div>
  );
}

/** The text's tokens, each with the model's most likely next token. One tab stop; the arrow keys
 * move between tokens and Enter reads that token layer by layer. */
function Tokens({ report, onChoose, busy }: { report: LookReport; onChoose: (position: number) => void; busy: boolean }) {
  const [focus, setFocus] = useState(report.position);
  const row = useRef<HTMLDivElement>(null);
  useEffect(() => setFocus(report.position), [report]);
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const n = report.tokens.length;
    let next = focus;
    if (e.key === "ArrowLeft") next = Math.max(0, focus - 1);
    else if (e.key === "ArrowRight") next = Math.min(n - 1, focus + 1);
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = n - 1;
    else return;
    e.preventDefault();
    setFocus(next);
    row.current?.querySelector<HTMLElement>(`[data-position="${next}"]`)?.focus();
  };
  return (
    <section className={t.section} aria-labelledby="text-tokens">
      <h2 id="text-tokens" className={s.panelTitle}>
        Tokens, and what the model predicts after each
      </h2>
      <div ref={row} className={t.tokens} role="group" aria-label="Tokens of the text. The arrow keys move between them; Enter reads one layer by layer." data-arrow-keys onKeyDown={onKeyDown}>
        {report.tokens.map((token, i) => {
          const next = report.predictions[i][0];
          return (
            <button
              key={i}
              type="button"
              className={t.token}
              data-position={i}
              tabIndex={i === focus ? 0 : -1}
              aria-pressed={i === report.position}
              disabled={busy}
              onFocus={() => setFocus(i)}
              onClick={() => onChoose(i)}
              aria-label={`Token ${i}, ${visibleToken(token)}: next most likely ${visibleToken(next.token)}, ${pct(next.prob)}`}
            >
              <span className={t.tokenText}>{visibleToken(token) || "∅"}</span>
              <span className={t.prediction}>
                {visibleToken(next.token)} <span className={t.prob}>{pct(next.prob)}</span>
              </span>
            </button>
          );
        })}
      </div>
      <p className={s.small}>Choose a token to read it layer by layer below.</p>
    </section>
  );
}

function ProbabilityRows({ tokens, label }: { tokens: { id: number; token: string; prob: number }[]; label: string }) {
  return (
    <ol className={t.probs} aria-label={label}>
      {tokens.map((x) => (
        <li key={x.id} className={t.probRow}>
          <code className={t.tok}>{visibleToken(x.token) || "∅"}</code>
          <span className={t.bar} aria-hidden="true">
            <span style={{ width: `${Math.max(1, x.prob * 100)}%` }} />
          </span>
          <span className={t.probValue}>{pct(x.prob, 1)}</span>
        </li>
      ))}
    </ol>
  );
}

function NextTokens({ report }: { report: LookReport }) {
  const token = report.tokens[report.position];
  return (
    <section className={t.section} aria-labelledby="text-next">
      <h2 id="text-next" className={s.panelTitle}>
        After token {report.position}, <code className={t.tok}>{visibleToken(token) || "∅"}</code>, the model predicts
      </h2>
      <ProbabilityRows tokens={report.predictions[report.position]} label={`The most likely next tokens after token ${report.position}`} />
    </section>
  );
}

/** A whole number typed into a field, or null while it is empty. */
const typedNumber = (text: string): number | null => (text.trim() === "" ? null : Number(text));

/** What the model writes after the text, greedily; optionally also with an SAE feature added
 * where the SAE writes, from a chosen token on, the new tokens included. One above the other. */
function Continue({ text, report, options }: { text: string; report: LookReport; options: AnalysisOptions }) {
  const sae = useStore((st) => st.sae);
  const info = sae.state === "ready" ? sae.info ?? null : null;
  const first = report.prepend_bos ? 1 : 0;
  const [tokens, setTokens] = useState<number | null>(20);
  const [steer, setSteer] = useState(false);
  const [feature, setFeature] = useState<number | null>(null);
  const [activation, setActivation] = useState("");
  const [start, setStart] = useState<number | null>(first);
  const [along, setAlong] = useState<LookFeatureAlong | null>(null);
  const [result, setResult] = useState<LookContinuation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const n = report.tokens.length;
  const validFeature = !!info && feature !== null && Number.isInteger(feature) && feature >= 0 && feature < info.d_sae;
  const strength = Number(activation);

  // The feature's largest activation on this text: a scale to choose how strongly to add it.
  useEffect(() => {
    setAlong(null);
    if (!steer || !validFeature || feature === null) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      api.lookFeature({ ...options, text, feature }).then(
        (d) => !cancelled && setAlong(d),
        () => undefined,
      );
    }, 200);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [steer, validFeature, feature, text, options.prepend_bos, info?.repo, info?.path, info?.revision, JSON.stringify(options.model)]);

  const problem =
    tokens === null || !Number.isInteger(tokens) || tokens < 1 || tokens > 64
      ? "Continue it by 1 to 64 tokens."
      : !steer
        ? null
        : !info
          ? "Load an SAE in Features to add one of its features."
          : feature === null || !Number.isInteger(feature) || feature < 0
            ? "Choose the feature to add: a whole number from 0."
            : feature >= info.d_sae
              ? `The SAE has ${count(info.d_sae)} features, 0 to ${count(info.d_sae - 1)}.`
              : activation.trim() === "" || !Number.isFinite(strength)
                ? "Give the activation to add it at: a number, in the feature's own units."
                : start === null || !Number.isInteger(start) || start < 0 || start >= n
                  ? `Choose a token from 0 to ${n - 1} to add it from.`
                  : null;

  const run = async () => {
    if (problem || tokens === null) return;
    setBusy(true);
    setError(null);
    try {
      const out = await api.lookContinue({
        ...options,
        text,
        tokens,
        steer: steer && feature !== null && start !== null ? { feature, activation: strength, start } : null,
      });
      setResult(out);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const added = result?.steer;
  return (
    <section className={t.section} aria-labelledby="text-continue">
      <h2 id="text-continue" className={s.panelTitle}>
        What the model writes next
      </h2>
      <div className={t.controls}>
        <Field label="New tokens" className={t.number}>
          <Input type="number" min={1} max={64} step={1} value={tokens ?? ""} onChange={(e) => setTokens(typedNumber(e.target.value))} />
        </Field>
        <Checkbox checked={steer} disabled={!info && !steer} onChange={setSteer}>
          Add an SAE feature while it continues
        </Checkbox>
        <Button onClick={() => void run()} disabled={busy || !!problem}>
          {busy ? "Continuing…" : "Continue the text"}
        </Button>
        {busy && <Spinner label="Continuing the text" />}
      </div>
      {steer && info && (
        <div className={t.controls}>
          <Field label="Feature" className={t.number}>
            <Input type="number" min={0} max={info.d_sae - 1} step={1} value={feature ?? ""} placeholder="Number" onChange={(e) => setFeature(typedNumber(e.target.value))} />
          </Field>
          <Field label="Activation" className={t.number}>
            <Input type="number" step="any" value={activation} placeholder="Units" onChange={(e) => setActivation(e.target.value)} />
          </Field>
          <Field label="From token" className={t.number}>
            <Input type="number" min={0} max={n - 1} step={1} value={start ?? ""} onChange={(e) => setStart(typedNumber(e.target.value))} />
          </Field>
        </div>
      )}
      <p className={s.small}>
        {!info
          ? "The most likely token each time (ties to the lowest token id). Load an SAE in Features to add one of its features while it continues."
          : steer
            ? `The most likely token each time (ties to the lowest token id). The feature's decoder direction times the activation is what it writes when that active; it is added where the SAE writes (${saeWritesText(info)}) at the token chosen and every token after it, the new ones included.${along ? ` Feature ${along.feature} is at most ${num(along.largest, 2)} on this text, at token ${along.position} (${visibleToken(along.tokens[along.position]) || "∅"}).` : ""}`
            : "The most likely token each time (ties to the lowest token id), the whole text read again for each."}
      </p>
      {problem && (steer || tokens === null || tokens < 1 || tokens > 64) && <p className={t.problem}>{problem}</p>}
      {error && <Callout tone="error">{error}</Callout>}
      {result && (
        <SaidLines
          lines={[
            { label: "Without steering", said: result.plain },
            ...(result.steered && added
              ? [{ label: `F${added.feature} at ${num(added.activation, 2)} from token ${added.start}`, said: result.steered }]
              : []),
          ]}
        />
      )}
    </section>
  );
}

function Lens({ report }: { report: LookReport }) {
  return (
    <section className={t.section} aria-labelledby="text-lens">
      <h2 id="text-lens" className={s.panelTitle}>
        What each layer would predict at token {report.position}
      </h2>
      {report.lens ? (
        <>
          <p className={s.small}>
            Each layer's residual stream read through the model's final normalization and unembedding (the logit lens). It
            describes what is readable at that layer; it isn't a causal effect.
          </p>
          <table className={s.table}>
            <thead>
              <tr>
                <th>Layer</th>
                {Array.from({ length: Math.min(3, report.top_k) }, (_, i) => (
                  <th key={i}>{i === 0 ? "Most likely" : `${i + 1}${i === 1 ? "nd" : "rd"}`}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {report.lens.map((row) => (
                <tr key={row.layer}>
                  <td>L{row.layer}</td>
                  {row.top.slice(0, 3).map((x) => (
                    <td key={x.id}>
                      <code className={t.tok}>{visibleToken(x.token) || "∅"}</code> <span className="faint">{pct(x.prob, 1)}</span>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </>
      ) : (
        <p className={s.small}>{report.lens_note}</p>
      )}
    </section>
  );
}

function Attention({
  text,
  report,
  options,
  theme,
  initial,
}: {
  text: string;
  report: LookReport;
  options: AnalysisOptions;
  theme: "light" | "dark";
  initial: { layer: number; head: number } | null;
}) {
  const info = useStore((st) => st.model.info);
  const [layer, setLayer] = useState(initial?.layer ?? 0);
  const [head, setHead] = useState(initial?.head ?? 0);
  const [data, setData] = useState<LookAttention | null>(null);
  const [error, setError] = useState<string | null>(null);
  const valid = !!info && layer >= 0 && layer < info.n_layers && head >= 0 && head < info.n_heads;
  useEffect(() => {
    setData(null);
    setError(null);
    if (!valid) return;
    let cancelled = false;
    api.lookAttention({ ...options, text, layer, head }).then(
      (d) => !cancelled && setData(d),
      (e: Error) => !cancelled && setError(e.message),
    );
    return () => {
      cancelled = true;
    };
  }, [text, layer, head, valid, options.prepend_bos, JSON.stringify(options.model)]);
  const axis: Axis[] = useMemo(() => report.tokens.map((tok, i) => ({ key: String(i), label: visibleToken(tok) || "∅" })), [report]);
  const color = useMemo(() => inkScale(1, theme), [theme]);
  const value = useCallback((r: number, c: number) => (c > r || !data ? undefined : data.pattern[r]?.[c]), [data]);
  const selected = useMemo(() => ({ r: report.position, c: report.position }), [report.position]);
  return (
    <section className={t.section} aria-labelledby="text-attention">
      <h2 id="text-attention" className={s.panelTitle}>
        Where a head attends in this text
      </h2>
      <div className={t.controls}>
        <Field label="Layer" className={t.number}>
          <Input type="number" min={0} max={(info?.n_layers ?? 1) - 1} value={layer} onChange={(e) => setLayer(Number(e.target.value))} />
        </Field>
        <Field label="Head" className={t.number}>
          <Input type="number" min={0} max={(info?.n_heads ?? 1) - 1} value={head} onChange={(e) => setHead(Number(e.target.value))} />
        </Field>
        <ScaleBar bound={1} theme={theme} kind="ink" label="Attention weight" />
      </div>
      {!valid && info && <p className={s.small}>Choose a layer from 0 to {info.n_layers - 1} and a head from 0 to {info.n_heads - 1}.</p>}
      {error && <Callout tone="error">{error}</Callout>}
      {data && (
        <Heatmap
          rows={axis}
          cols={axis}
          value={value}
          color={color}
          theme={theme}
          selected={selected}
          tokens
          cellMin={8}
          cellMax={24}
          rowTitle="Query"
          colTitle="Key"
          ariaLabel={`Attention of L${data.layer} H${data.head} in your text`}
        />
      )}
    </section>
  );
}

function Features({ text, options, position }: { text: string; options: AnalysisOptions; position: number }) {
  const [data, setData] = useState<LookFeatures | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    setData(null);
    setError(null);
    let cancelled = false;
    api.lookFeatures({ ...options, text, top_k: 5 }).then(
      (d) => !cancelled && setData(d),
      (e: Error) => !cancelled && setError(e.message),
    );
    return () => {
      cancelled = true;
    };
  }, [text, options.prepend_bos, JSON.stringify(options.model)]);
  return (
    <section className={t.section} aria-labelledby="text-features">
      <h2 id="text-features" className={s.panelTitle}>
        The SAE's strongest features at token {position}
      </h2>
      {error && <Callout tone="error">{error}</Callout>}
      {data &&
        (position < data.first_real_token ? (
          <p className={s.small}>SAEs aren't trained on the beginning-of-sequence token; choose another token.</p>
        ) : data.features[position]?.length ? (
          <ol className={t.features}>
            {data.features[position].map((f) => (
              <li key={f.feature}>
                <span className={t.featureName}>F{f.feature}</span> <span className="faint">{f.activation.toFixed(2)}</span>
              </li>
            ))}
          </ol>
        ) : (
          <p className={s.small}>No feature fires on this token.</p>
        ))}
      {data && <p className={s.small}>{data.active[position] ?? 0} features fire on this token. Open Features to read one on the prompts.</p>}
    </section>
  );
}
