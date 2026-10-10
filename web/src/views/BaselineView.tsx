import { scaleLinear } from "d3-scale";
import { useRef, useState } from "react";

import { api } from "../api/client";
import type { BaselinePrompt, BaselineReport, Normalization, PromptRecord, TopToken } from "../api/types";
import { Button, Callout, Empty } from "../components/ui";
import { useElementSize } from "../lib/canvas";
import { divergingScale, markColor, niceBound } from "../lib/color";
import { answerText, capitalize, num, plural, prob, signed, visibleToken } from "../lib/format";
import { modelName, useAnalysisContext } from "../lib/hooks";
import { analysisContext } from "../lib/analysis";
import { gapCheck, metricWords, signedMetric } from "../lib/metrics";
import { useStore } from "../store/app";
import s from "./views.module.css";
import b from "./BaselineView.module.css";

export function BaselineView() {
  const datasetPath = useStore((st) => st.datasetPath);
  const model = useStore((st) => st.model);
  const context = useAnalysisContext();
  const report = useStore((st) => st.baselines[context.baselineKey]);
  const index = useStore((st) => st.promptIndex);
  const [busy, setBusy] = useState(false);
  const guard = useStore((st) => st.guard);

  const check = async () => {
    const metric = context.metric;
    if (!datasetPath || !metric) return;
    setBusy(true);
    const out = await guard(() => api.baseline(datasetPath, context.options, metric));
    setBusy(false);
    const key = context.baselineKey;
    if (out && analysisContext(useStore.getState()).baselineKey === key) useStore.setState((st) => ({ baselines: { ...st.baselines, [key]: out } }));
  };

  const ready = model.state === "ready" && !!datasetPath && !context.error && !!context.metric;
  const verdict = report?.summary ? (behaviorPresent(report) ? "The behavior is present" : "The behavior is weak or missing") : null;
  const measured = metricWords(context.metric ?? context.tokenMetric);

  return (
    <div className={s.view}>
      <div className={s.head}>
        <div className={s.titleBlock}>
          <span className="eyebrow">Baseline</span>
          <h2 className={s.title}>{verdict ?? "Does the model show the behavior?"}</h2>
          <p className={s.subtitle}>
            Run the model on the clean and corrupt prompts without intervening, to confirm the behavior exists before
            measuring what carries it.
          </p>
        </div>
        <div className={s.headActions}>
          <Button variant={report ? "secondary" : "primary"} icon="play" disabled={!ready || busy} onClick={check}>
            {busy ? "Running…" : report ? "Run again" : "Check the baseline"}
          </Button>
        </div>
      </div>

      <p className={s.small}>{context.label} · {measured.label}</p>
      {context.error && <Callout tone="error" title="The loaded model differs">{context.error}</Callout>}
      {!context.error && context.metricError && <Callout tone="error" title="The metric isn't complete">{context.metricError}</Callout>}

      {!ready && !context.error && !context.metricError && (
        <Empty title={model.state !== "ready" ? "Load a model first" : "Choose prompts first"}>
          {model.state !== "ready"
            ? "The baseline runs the loaded model on the current dataset."
            : "Set up a prompt pair or a dataset in Prompts."}
        </Empty>
      )}
      {ready && !report && !busy && (
        <p className={s.small}>
          {modelName(model.info?.id)} on {datasetPath}. This takes a moment on a CPU.
        </p>
      )}
      {report && <Report report={report} index={index} normalization={context.metric?.normalization ?? "dataset_gap"} />}
    </div>
  );
}

function Report({ report, index, normalization }: { report: BaselineReport; index: number; normalization: Normalization }) {
  const dataset = useStore((st) => st.dataset);
  const sum = report.summary;
  if (!sum) {
    return (
      <Callout tone="error" title="None of these prompts can be used">
        {report.issues[0]?.message}
      </Callout>
    );
  }
  const prompt = report.prompts.find((p) => p.index === index) ?? report.prompts[0];
  const metric = sum.metric;
  // A server from before metrics could be chosen measured the logit difference: the preference.
  const kind = metric?.kind ?? "logit_diff";
  const words = metricWords(metric ?? { kind });
  const logitDiff = kind === "logit_diff";
  const gap = (logitDiff ? sum.gap : metric?.gap) ?? 0;
  // Whether a run can normalize effects by this gap, as the run itself will judge it. A server
  // from before metrics could be chosen reports the logit difference as the preference.
  const check = gapCheck(
    report.prompts.map((p) => ({
      index: p.index,
      clean: p.clean_metric ?? (logitDiff ? p.clean_logit_diff : null),
      corrupt: p.corrupt_metric ?? (logitDiff ? p.corrupt_logit_diff : null),
    })),
    normalization,
    words.label,
  );
  return (
    <>
      <p className={s.sentence}>
        Clean prompts prefer {preference(sum.clean_logit_diff)} on average ({sum.clean_prefers_answer} of {report.n}{" "}
        prefer the answer); corrupt prompts prefer {preference(sum.corrupt_logit_diff)} ({sum.corrupt_prefers_answer} of{" "}
        {report.n}).{" "}
        {logitDiff ? (
          <>
            The clean–corrupt gap is <strong>{num(gap)}</strong>.
          </>
        ) : (
          <>
            In the {words.label}, clean prompts score <strong>{num(metric?.clean, 3)}</strong> and corrupt prompts{" "}
            <strong>{num(metric?.corrupt, 3)}</strong>: a gap of <strong>{signed(gap, 3)}</strong>.
          </>
        )}
        {check.tone === "ok" && <> {check.text}</>}
      </p>
      {check.tone !== "ok" && (
        <Callout
          tone={check.tone === "error" ? "error" : "info"}
          title={check.tone === "error" ? "Effects can't be normalized by this gap" : "Effects normalized by this gap will be unreliable"}
        >
          {check.text}
        </Callout>
      )}
      <p className={s.small}>
        A preference is log P(answer) − log P(distractor), the logit difference for single tokens. A set of answers counts
        their probabilities together; an answer of several tokens multiplies its tokens' probabilities, each read after the
        ones before it.
      </p>
      {report.issues.length > 0 && (
        <Callout tone="error" title={`${plural(report.issues.length, "issue")} in this dataset`}>
          {report.issues.slice(0, 3).map((i) => (
            <div key={i.index + i.kind}>
              Prompt {i.index}: {i.message}
            </div>
          ))}
        </Callout>
      )}
      <div className={b.stats}>
        <Stat label={`Clean ${words.short}`} value={signed(logitDiff ? sum.clean_logit_diff : metric?.clean, logitDiff ? 2 : 3)} />
        <Stat label={`Corrupt ${words.short}`} value={signed(logitDiff ? sum.corrupt_logit_diff : metric?.corrupt, logitDiff ? 2 : 3)} />
        <Stat label="Clean answer prob." value={prob(sum.clean_answer_prob)} />
        <Stat label="Corrupt answer prob." value={prob(sum.corrupt_answer_prob)} />
      </div>
      <DotPlot prompts={report.prompts} selected={index} kind={kind} label={words.label} formula={words.formula} />
      {prompt && <TopTokens prompt={prompt} record={dataset?.records[prompt.index]} metricLabel={logitDiff ? null : words.short} />}
    </>
  );
}

/** Which token a mean preference favors, and by how much: "the answer by 2.90". */
function preference(ld: number | null) {
  const value = ld ?? 0;
  return (
    <>
      the {value >= 0 ? "answer" : <strong>distractor</strong>} by <strong>{num(Math.abs(value))}</strong>
    </>
  );
}

/** The clean prompts prefer the answer, by a gap large enough to measure interventions against. */
function behaviorPresent(report: BaselineReport): boolean {
  const sum = report.summary;
  return !!sum && (sum.clean_logit_diff ?? 0) > 0 && (sum.gap ?? 0) > 0.5;
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className={b.stat}>
      <div className={b.statLabel}>{label}</div>
      <div className={`${b.statValue} figure`}>{value}</div>
    </div>
  );
}

/** Each prompt's value in the metric, clean and corrupt. A difference's sign says which of answer
 * and distractor the model prefers, in the effect colors; other metrics are drawn in ink. */
function DotPlot({ prompts, selected, kind, label, formula }: { prompts: BaselinePrompt[]; selected: number; kind: string; label: string; formula: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const { width } = useElementSize(ref);
  const setIndex = useStore((st) => st.setPromptIndex);
  const valueOf = (p: BaselinePrompt, which: "clean" | "corrupt") =>
    kind === "logit_diff" ? p[`${which}_logit_diff`] : (p[`${which}_metric`] ?? null);
  const values = prompts.flatMap((p) => [valueOf(p, "clean") ?? 0, valueOf(p, "corrupt") ?? 0, 0]);
  const theme = useStore((st) => st.theme);
  const colored = signedMetric(kind);
  // Positive: the model prefers the answer (cobalt); negative: the distractor (amber).
  const signScale = divergingScale(niceBound(values, 1), theme);
  const x = scaleLinear()
    .domain([Math.min(...values), Math.max(...values)])
    .nice()
    .range([60, Math.max(120, width - 12)]);
  const rows: { which: "clean" | "corrupt"; y: number }[] = [
    { which: "clean", y: 22 },
    { which: "corrupt", y: 52 },
  ];
  return (
    <div ref={ref} className={b.plot}>
      <div className={s.small}>
        {capitalize(label)} per prompt ({formula}).{colored ? " Positive values prefer the answer." : ""} Click a dot to show that prompt.
      </div>
      {width > 0 && (
        <svg width={width} height={92} role="img" aria-label={`Clean and corrupt ${label} per prompt`}>
          <line className={b.zero} x1={x(0)} x2={x(0)} y1={6} y2={68} />
          {rows.map((row) => (
            <g key={row.which}>
              <text className={b.rowLabel} x={0} y={row.y + 4}>
                {row.which}
              </text>
              <line className={b.track} x1={60} x2={width - 12} y1={row.y} y2={row.y} />
              {prompts.map((p) => {
                const v = valueOf(p, row.which);
                if (v === null) return null;
                const isSel = p.index === selected;
                return (
                  <circle
                    tabIndex={0}
                    role="button"
                    aria-label={`Show prompt ${p.index}: ${row.which} ${label} ${signed(v, 3)}`}
                    onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setIndex(p.index); } }}
                    key={p.index}
                    className={isSel ? b.dotSelected : b.dot}
                    style={isSel || !colored ? undefined : { fill: markColor(signScale, v) }}
                    cx={x(v)}
                    cy={row.y}
                    r={isSel ? 5 : 3.5}
                    onClick={() => setIndex(p.index)}
                  >
                    <title>{`Prompt ${p.index}: ${signed(v, 3)}`}</title>
                  </circle>
                );
              })}
            </g>
          ))}
          {x.ticks(6).map((t) => (
            <text key={t} className={b.tick} x={x(t)} y={86} textAnchor="middle">
              {x.tickFormat(6)(t)}
            </text>
          ))}
        </svg>
      )}
    </div>
  );
}

/** log P from a probability; nothing for none. */
function logOf(p: number | null | undefined): number | null {
  return p === null || p === undefined || !(p > 0) ? null : Math.log(p);
}

/** How an answer is read: one token, any of a set, or several tokens in order. */
function reading(answer: PromptRecord["answer"] | undefined): string {
  if (Array.isArray(answer)) return ` · any of ${answer.length}`;
  if (typeof answer === "string" && /\S\s+\S/.test(answer.trim())) return " · in order";
  return "";
}

function TopTokens({ prompt, record, metricLabel }: { prompt: BaselinePrompt; record: PromptRecord | undefined; metricLabel: string | null }) {
  // The tokens that count for the answer and for the distractor: one token, or a set's members.
  const members = (a: PromptRecord["answer"] | undefined) => new Set(a === undefined ? [] : Array.isArray(a) ? a : [a]);
  const answers = members(record?.answer);
  const distractors = members(record?.distractor);
  return (
    <div className={s.grid2}>
      {(["clean", "corrupt"] as const).map((which) => {
        const top: TopToken[] = which === "clean" ? prompt.clean_top : prompt.corrupt_top;
        const pref = which === "clean" ? prompt.clean_logit_diff : prompt.corrupt_logit_diff;
        const metric = which === "clean" ? prompt.clean_metric : prompt.corrupt_metric;
        // log P(distractor) = log P(answer) − (log P(answer) − log P(distractor)).
        const logAnswer = logOf(which === "clean" ? prompt.clean_answer_prob : prompt.corrupt_answer_prob);
        const logDistractor = logAnswer === null || pref === null ? null : logAnswer - pref;
        return (
          <div key={which} className={s.panel}>
            <div className={b.topHead}>
              <span className={s.panelTitle}>
                Prompt {prompt.index} · {which}
              </span>
              <span className={s.small}>
                logit diff {signed(pref)}
                {metricLabel && metric !== undefined && metric !== null && <> · {metricLabel} {signed(metric, 3)}</>}
              </span>
            </div>
            <p className={b.promptText}>{which === "clean" ? prompt.clean : prompt.corrupt}</p>
            <dl className={b.answers}>
              <dt>log P(answer)</dt>
              <dd>
                {signed(logAnswer)}{" "}
                <span className={b.answerText}>
                  {record ? answerText(record.answer, 6) : prompt.answer}
                  {reading(record?.answer)}
                </span>
              </dd>
              <dt>log P(distractor)</dt>
              <dd>
                {signed(logDistractor)}{" "}
                <span className={b.answerText}>
                  {record ? answerText(record.distractor, 6) : prompt.distractor}
                  {reading(record?.distractor)}
                </span>
              </dd>
            </dl>
            <div className={b.top}>
              {top.map((t) => {
                const role = answers.has(t.token) ? "answer" : distractors.has(t.token) ? "distractor" : null;
                return (
                  <div key={t.id} className={b.topRow}>
                    <span className={`${b.topToken} ${role ? b.marked : ""}`}>{visibleToken(t.token)}</span>
                    <span className={b.topBar}>
                      <span style={{ width: `${Math.max(1, t.prob * 100)}%` }} />
                    </span>
                    <span className={b.topProb}>{prob(t.prob)}</span>
                    <span className={b.topRole}>{role ?? ""}</span>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
