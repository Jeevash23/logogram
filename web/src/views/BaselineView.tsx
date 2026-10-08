import { scaleLinear } from "d3-scale";
import { useRef, useState } from "react";

import { api } from "../api/client";
import type { BaselinePrompt, BaselineReport, TopToken } from "../api/types";
import { Button, Callout, Empty } from "../components/ui";
import { useElementSize } from "../lib/canvas";
import { num, plural, prob, signed, visibleToken } from "../lib/format";
import { modelName, useAnalysisContext } from "../lib/hooks";
import { analysisContext } from "../lib/analysis";
import { useStore } from "../store/app";
import s from "./views.module.css";
import b from "./BaselineView.module.css";

export function BaselineView() {
  const datasetPath = useStore((st) => st.datasetPath);
  const model = useStore((st) => st.model);
  const context = useAnalysisContext();
  const report = useStore((st) => st.baselines[context.key]);
  const index = useStore((st) => st.promptIndex);
  const [busy, setBusy] = useState(false);
  const guard = useStore((st) => st.guard);

  const check = async () => {
    if (!datasetPath) return;
    setBusy(true);
    const out = await guard(() => api.baseline(datasetPath, context.options));
    setBusy(false);
    if (out && analysisContext(useStore.getState()).key === context.key) useStore.setState((st) => ({ baselines: { ...st.baselines, [context.key]: out } }));
  };

  const ready = model.state === "ready" && !!datasetPath && !context.error;

  return (
    <div className={s.view}>
      <div className={s.head}>
        <div className={s.titleBlock}>
          <h2 className={s.title}>Baseline</h2>
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

      <p className={s.small}>{context.label}</p>
      {context.error && <Callout tone="error" title="The loaded model differs">{context.error}</Callout>}

      {!ready && !context.error && (
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
      {report && <Report report={report} index={index} />}
    </div>
  );
}

function Report({ report, index }: { report: BaselineReport; index: number }) {
  const sum = report.summary;
  if (!sum) {
    return (
      <Callout tone="error" title="None of these prompts can be used">
        {report.issues[0]?.message}
      </Callout>
    );
  }
  const gap = sum.gap ?? 0;
  const present = (sum.clean_logit_diff ?? 0) > 0 && gap > 0.5;
  const prompt = report.prompts.find((p) => p.index === index) ?? report.prompts[0];
  return (
    <>
      <p className={s.sentence}>
        {present ? <strong>The behavior is present.</strong> : <strong>The behavior is weak or missing.</strong>}{" "}
        Clean prompts prefer the answer by <strong>{signed(sum.clean_logit_diff)}</strong> logits on average (in{" "}
        {sum.clean_prefers_answer} of {report.n}); corrupt prompts by <strong>{signed(sum.corrupt_logit_diff)}</strong>{" "}
        (in {sum.corrupt_prefers_answer} of {report.n}). The clean–corrupt gap is <strong>{num(gap)}</strong>
        {present ? ", which is what interventions are measured against." : ". Normalized effects will be noisy or undefined."}
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
        <Stat label="Clean logit diff" value={signed(sum.clean_logit_diff)} />
        <Stat label="Corrupt logit diff" value={signed(sum.corrupt_logit_diff)} />
        <Stat label="Clean answer prob." value={prob(sum.clean_answer_prob)} />
        <Stat label="Corrupt answer prob." value={prob(sum.corrupt_answer_prob)} />
      </div>
      <DotPlot prompts={report.prompts} selected={index} />
      {prompt && <TopTokens prompt={prompt} />}
    </>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className={b.stat}>
      <div className={b.statLabel}>{label}</div>
      <div className={b.statValue}>{value}</div>
    </div>
  );
}

function DotPlot({ prompts, selected }: { prompts: BaselinePrompt[]; selected: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const { width } = useElementSize(ref);
  const setIndex = useStore((st) => st.setPromptIndex);
  const values = prompts.flatMap((p) => [p.clean_logit_diff ?? 0, p.corrupt_logit_diff ?? 0, 0]);
  const x = scaleLinear()
    .domain([Math.min(...values), Math.max(...values)])
    .nice()
    .range([60, Math.max(120, width - 12)]);
  const rows: { key: "clean_logit_diff" | "corrupt_logit_diff"; label: string; y: number }[] = [
    { key: "clean_logit_diff", label: "clean", y: 22 },
    { key: "corrupt_logit_diff", label: "corrupt", y: 52 },
  ];
  return (
    <div ref={ref} className={b.plot}>
      <div className={s.small}>Logit difference per prompt (answer − distractor). Click a dot to show that prompt.</div>
      {width > 0 && (
        <svg width={width} height={92} role="img" aria-label="Clean and corrupt logit differences per prompt">
          <line className={b.zero} x1={x(0)} x2={x(0)} y1={6} y2={68} />
          {rows.map((row) => (
            <g key={row.key}>
              <text className={b.rowLabel} x={0} y={row.y + 4}>
                {row.label}
              </text>
              <line className={b.track} x1={60} x2={width - 12} y1={row.y} y2={row.y} />
              {prompts.map((p) => {
                const v = p[row.key];
                if (v === null) return null;
                const isSel = p.index === selected;
                return (
                  <circle
                    tabIndex={0}
                    role="button"
                    aria-label={`Show prompt ${p.index}: ${row.label} logit difference ${signed(v, 3)}`}
                    onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setIndex(p.index); } }}
                    key={p.index}
                    className={isSel ? b.dotSelected : b.dot}
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

function TopTokens({ prompt }: { prompt: BaselinePrompt }) {
  return (
    <div className={s.grid2}>
      {(["clean", "corrupt"] as const).map((which) => {
        const top: TopToken[] = which === "clean" ? prompt.clean_top : prompt.corrupt_top;
        const ld = which === "clean" ? prompt.clean_logit_diff : prompt.corrupt_logit_diff;
        return (
          <div key={which} className={s.panel}>
            <div className={b.topHead}>
              <span className={s.panelTitle}>
                Prompt {prompt.index} · {which}
              </span>
              <span className={s.small}>logit diff {signed(ld)}</span>
            </div>
            <p className={b.promptText}>{which === "clean" ? prompt.clean : prompt.corrupt}</p>
            <div className={b.top}>
              {top.map((t) => {
                const role = t.token === prompt.answer ? "answer" : t.token === prompt.distractor ? "distractor" : null;
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
