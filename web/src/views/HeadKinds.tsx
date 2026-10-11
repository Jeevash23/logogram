import { useCallback, useMemo, useState } from "react";

import { api } from "../api/client";
import type { HeadKindName, HeadKindsReport } from "../api/types";
import { Heatmap, type Axis } from "../components/Heatmap/Heatmap";
import { ScaleBar } from "../components/Heatmap/ScaleBar";
import { Button, Callout, Field, Input, Segmented, Spinner } from "../components/ui";
import { inkScale } from "../lib/color";
import { ci, count, num } from "../lib/format";
import { useAnalysisContext } from "../lib/hooks";
import { useStore } from "../store/app";
import s from "./views.module.css";
import k from "./HeadKinds.module.css";

const KINDS: { value: HeadKindName; label: string }[] = [
  { value: "induction", label: "Induction" },
  { value: "duplicate_token", label: "Duplicate token" },
  { value: "previous_token", label: "Previous token" },
];

const BOOTSTRAP = 1000;
const LEVEL = 0.95;
const TOP = 5;

// The last report, the request it answered and the choices made, so coming back to the view (or
// choosing a head, which redraws it) shows them again.
let last: { key: string; report: HeadKindsReport } | null = null;
const chosen: { kind: HeadKindName; sequences: number; length: number | null; seed: number } = {
  kind: "induction",
  sequences: 32,
  length: null,
  seed: 0,
};

/** Every head scored by what it attends to, on random tokens repeated twice; choosing a head
 * selects it, so the Attention view shows its pattern on the prompts. */
export function HeadKinds() {
  const info = useStore((st) => st.model.info);
  const theme = useStore((st) => st.theme);
  const selection = useStore((st) => st.selection);
  const select = useStore((st) => st.select);
  const context = useAnalysisContext();
  const bos = context.options.prepend_bos;
  const longest = info ? Math.max(2, Math.floor((info.n_ctx - (bos ? 1 : 0)) / 2)) : 32;
  const [kind, setKindState] = useState<HeadKindName>(chosen.kind);
  const [sequences, setSequencesState] = useState(chosen.sequences);
  const [length, setLengthState] = useState(() => chosen.length ?? Math.min(32, longest));
  const [seed, setSeedState] = useState(chosen.seed);
  const setKind = (v: HeadKindName) => {
    chosen.kind = v;
    setKindState(v);
  };
  const setSequences = (v: number) => {
    chosen.sequences = v;
    setSequencesState(v);
  };
  const setLength = (v: number) => {
    chosen.length = v;
    setLengthState(v);
  };
  const setSeed = (v: number) => {
    chosen.seed = v;
    setSeedState(v);
  };
  const request = { seed, sequences, length, bootstrap: BOOTSTRAP, ci: LEVEL, ...context.options };
  const key = JSON.stringify([info?.id, info?.revision, info?.dtype, info?.process_weights, request]);
  const [found, setFound] = useState(last);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // A report for other settings (or another model) stays in view, marked as such.
  const shown = found?.report ?? null;
  const stale = !!found && found.key !== key;

  const problem =
    !Number.isInteger(sequences) || sequences < 2 || sequences > 512
      ? "Use 2 to 512 sequences."
      : !Number.isInteger(length) || length < 2 || length > longest
        ? `Use sequences of 2 to ${longest} tokens: two copies must fit in the model's context of ${info?.n_ctx ?? "?"} tokens.`
        : !Number.isInteger(seed) || seed < 0
          ? "The seed is a whole number from 0."
          : null;

  const find = async () => {
    if (problem) return;
    setBusy(true);
    setError(null);
    try {
      const out = await api.headKinds(request);
      last = { key, report: out };
      setFound(last);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const scores = shown?.kinds[kind];
  const rows: Axis[] = useMemo(() => Array.from({ length: shown?.n_layers ?? 0 }, (_, i) => ({ key: `L${i}`, label: `L${i}` })), [shown]);
  const cols: Axis[] = useMemo(() => Array.from({ length: shown?.n_heads ?? 0 }, (_, i) => ({ key: `H${i}`, label: `H${i}` })), [shown]);
  const color = useMemo(() => inkScale(1, theme), [theme]);
  const value = useCallback((r: number, c: number) => scores?.mean[r]?.[c] ?? null, [scores]);
  const tooltip = useCallback(
    (r: number, c: number) => (
      <>
        <div style={{ fontWeight: 650 }}>
          L{r} H{c}
        </div>
        <div>score {num(scores?.mean[r]?.[c], 3)}</div>
        <div style={{ opacity: 0.72 }}>
          {Math.round(LEVEL * 100)}% CI {ci(scores?.lo[r]?.[c], scores?.hi[r]?.[c], 3)}
        </div>
      </>
    ),
    [scores],
  );
  const onSelect = useCallback((r: number, c: number) => select({ layer: r, part: "head", head: c }), [select]);
  const selected = useMemo(
    () => (selection?.part === "head" && selection.head !== undefined ? { r: selection.layer, c: selection.head } : null),
    [selection?.part, selection?.layer, selection?.head],
  );

  return (
    <section className={k.section} aria-labelledby="head-kinds-title">
      <div className={k.head}>
        <div>
          <h2 id="head-kinds-title" className={s.panelTitle}>
            Find heads by what they attend to
          </h2>
          <p className={s.small}>
            Each head is read on sequences of random tokens, each repeated twice. A previous-token head attends to the token
            just before; a duplicate-token head to the earlier copy of the current token; an induction head to the token
            that followed that copy, which lets it predict what comes next. A score is the attention a head pays to that
            token, averaged over the second copy.
          </p>
        </div>
      </div>
      <div className={k.controls}>
        <Field label="Sequences" className={k.number}>
          <Input type="number" min={2} max={512} value={sequences} onChange={(e) => setSequences(Number(e.target.value))} />
        </Field>
        <Field label="Tokens in each copy" className={k.number}>
          <Input type="number" min={2} max={longest} value={length} onChange={(e) => setLength(Number(e.target.value))} />
        </Field>
        <Field label="Seed" className={k.number}>
          <Input type="number" min={0} value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
        </Field>
        <Button onClick={() => void find()} disabled={busy || !!problem || !info} tip={problem ?? (info ? undefined : "Load a model first")}>
          {busy ? "Finding heads…" : shown ? "Find again" : "Find heads"}
        </Button>
        {busy && <Spinner label="Scoring heads" />}
      </div>
      {problem && <p className={s.small}>{problem}</p>}
      {stale && !problem && <p className={s.small}>These scores are for other settings or another model. Press Find again to score the heads with these.</p>}
      {error && <Callout tone="error">{error}</Callout>}
      {shown && scores && (
        <div className={k.result}>
          <div className={k.map}>
            <div className={k.mapHead}>
              <Segmented label="Kind of head" value={kind} onChange={setKind} options={KINDS} />
              <ScaleBar bound={1} theme={theme} kind="ink" label="Score" />
            </div>
            <Heatmap
              rows={rows}
              cols={cols}
              value={value}
              color={color}
              theme={theme}
              selected={selected}
              onSelect={onSelect}
              tooltip={tooltip}
              rowTitle="Layer"
              colTitle="Head"
              cellMax={30}
              ariaLabel={`${KINDS.find((x) => x.value === kind)?.label} scores of every head`}
            />
          </div>
          <div className={k.top}>
            <span className={s.panelTitle}>Strongest {KINDS.find((x) => x.value === kind)?.label.toLowerCase()} heads</span>
            <ol className={k.list}>
              {shown.top[kind].slice(0, TOP).map((t) => (
                <li key={`${t.layer}-${t.head}`}>
                  <button
                    type="button"
                    className={k.headButton}
                    aria-pressed={selected?.r === t.layer && selected?.c === t.head}
                    onClick={() => onSelect(t.layer, t.head)}
                  >
                    <span className={k.headName}>
                      L{t.layer} H{t.head}
                    </span>
                    <span className={k.score}>{num(t.mean, 2)}</span>
                    <span className="faint">[{ci(t.lo, t.hi)}]</span>
                  </button>
                </li>
              ))}
            </ol>
            <p className={s.small}>
              {count(shown.settings.sequences)} sequences of {shown.settings.length} tokens repeated twice, seed{" "}
              {shown.settings.seed}, {shown.settings.prepend_bos ? "after a beginning-of-sequence token" : "without a beginning-of-sequence token"};{" "}
              {Math.round(shown.settings.ci * 100)}% intervals from {count(shown.settings.bootstrap)} bootstrap resamples of the
              sequences. Choose a head to see its attention on the prompts.
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
