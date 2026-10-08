import { useEffect, useMemo, useRef, useState } from "react";

import { api } from "../api/client";
import type { RunDetail, SiteDetail, SiteKind, SiteResult, Spec, Summary } from "../api/types";
import { ci, count, num, plural, prob, signed } from "../lib/format";
import { siteValue, useActiveRun, useArchitecture } from "../lib/hooks";
import {
  componentLabel,
  findSite,
  isResidKind,
  KIND_NAMES,
  selectionOfSite,
  sitesOnComponent,
  type ResidKind,
  type Selection,
} from "../lib/sites";
import { baselineText, experimentText, measureOf, measureWords, positionText } from "../lib/spec";
import { useStore } from "../store/app";
import { Distribution } from "./Distribution";
import { Button, Icon, Spinner } from "./ui";
import s from "./Inspector.module.css";

export function Inspector() {
  const selection = useStore((st) => st.selection);
  const run = useActiveRun();
  const site = useMemo(() => {
    if (!selection) return null;
    const base = findSite(run.sites, selection);
    return base ? (run.results[base.index] ?? null) : null;
  }, [selection, run.sites, run.results]);
  const pendingBase = selection ? findSite(run.sites, selection) : null;

  return (
    <div className={s.inspector}>
      <div className={s.header}>
        <span className={s.headerTitle}>Inspector</span>
      </div>
      <div className={s.body}>
        {!selection ? (
          <RunOverview />
        ) : (
          <>
            <ComponentHeader selection={selection} />
            {run.detail?.spec && pendingBase && <p className={s.note}>{experimentText(run.detail.spec.experiment)} · {positionText(pendingBase.position)}</p>}
            {site && run.detail ? (
              <Evidence site={site} runId={run.id as string} summary={run.detail.summary} finished={!!run.detail.summary} />
            ) : pendingBase && run.running ? (
              <p className={s.note}>
                <Spinner /> Waiting for layer {pendingBase.layer}.
              </p>
            ) : run.live?.status === "running" && run.sites.length === 0 ? (
              <p className={s.note}>
                <Spinner /> Waiting for the run to start.
              </p>
            ) : run.id ? (
              <NotMeasured selection={selection} />
            ) : (
              <p className={s.note}>Run an experiment to measure this component.</p>
            )}
            {run.detail?.spec && (site || pendingBase) && <Method spec={run.detail.spec} summary={run.detail.summary} selection={selection} />}
            <Actions selection={selection} />
            <AcrossRuns selection={selection} />
          </>
        )}
      </div>
    </div>
  );
}

function ComponentHeader({ selection }: { selection: Selection }) {
  const arch = useArchitecture();
  const run = useActiveRun();
  let kind = "";
  if (selection.part === "head") kind = `Attention head ${selection.head} in layer ${selection.layer}. Its output z, before the output projection.`;
  else if (selection.part === "attn") kind = `The attention output of layer ${selection.layer}: all heads, after the output projection.`;
  else if (selection.part === "mlp") kind = `The MLP output of layer ${selection.layer}.`;
  else if (selection.part === "feature") kind = `Feature ${selection.feature} of the SAE on layer ${selection.layer}: a direction in the model the SAE finds active on some tokens.`;
  else {
    const resid = selection.kind ?? sitesOnComponent(run.sites, selection)[0]?.kind;
    kind = resid ? `The ${KIND_NAMES[resid]} (layer ${selection.layer}).` : `The residual stream at layer ${selection.layer}.`;
  }
  const label = componentLabel(
    selection.part === "resid" && !selection.kind
      ? { ...selection, kind: residKindOnly(sitesOnComponent(run.sites, selection)) }
      : selection,
  );
  return (
    <div className={s.component}>
      <h2 className={s.componentTitle}>{label}</h2>
      <p className={s.componentText}>{kind}</p>
      {arch && selection.layer >= arch.nLayers && <p className={s.note}>This model has only {arch.nLayers} layers.</p>}
    </div>
  );
}

function Method({ spec, summary, selection }: { spec: Spec; summary: Summary | null; selection: Selection }) {
  const exp = spec.experiment;
  const base = summary ? findSite(summary.sites, selection) : null;
  const position = base ? positionText(base.position) : selection.positionKey ? `position ${selection.positionKey}` : "the positions in the spec";
  const resid = base?.kind ?? selection.kind;
  const what =
    selection.part === "head"
      ? "this head's output"
      : selection.part === "attn"
        ? "the attention output"
        : selection.part === "mlp"
          ? "the MLP output"
          : selection.part === "feature"
            ? `feature ${selection.feature} of the SAE`
            : resid
            ? `the ${KIND_NAMES[resid]}`
            : "the residual stream";
  let how: string;
  if (exp.kind === "direct_logit_attribution") {
    const written = selection.part === "head" ? "this head's output, through its share of the output projection," : `${what}`;
    return (
      <section className={s.section}>
        <h4 className={s.sectionTitle}>Method</h4>
        <p className={s.strong}>{experimentText(exp)}</p>
        <p className={s.text}>
          Runs each {exp.prompts} prompt once and reads {written} at the last token through the final normalization, with its
          scale held at its value in the run. This is the component's direct effect: it leaves out what the component does
          through later components, which patching measures. Nothing is replaced.
        </p>
        <h4 className={s.sectionTitle}>Metric</h4>
        <p className={s.text}>Logit difference at the last token: logit(answer) − logit(distractor).</p>
        <p className={s.text}>
          Share = direct effect ÷{" "}
          {spec.metric.normalization === "dataset_gap" ? (
            <>
              the mean {exp.prompts} logit difference{summary?.metric.denominator != null && <> ({num(summary.metric.denominator, 3)})</>}
            </>
          ) : (
            <>each prompt's own {exp.prompts} logit difference</>
          )}
          . Over all components, with the embeddings and biases, the shares add up to one.
        </p>
      </section>
    );
  }
  if (selection.part === "feature" && (exp.kind === "activation_patching" || exp.kind === "ablation")) {
    const [receiver, source] = exp.kind === "activation_patching" && exp.direction === "clean_to_corrupt" ? ["corrupt", "clean"] : ["clean", "corrupt"];
    const to = exp.kind === "ablation" ? "zero" : `its value in the paired ${source} prompt`;
    how =
      `Runs each ${receiver} prompt, encodes the activation the SAE reads, and changes only ${what} at ${position} to ${to}. ` +
      "The activation moves along the feature's decoder direction, and what the SAE misses (its error) is kept as it was.";
  } else if (selection.part === "feature" && exp.kind === "attribution_patching") {
    const [receiver, source] = exp.direction === "clean_to_corrupt" ? ["corrupt", "clean"] : ["clean", "corrupt"];
    how =
      `Estimates, to first order, what changing ${what} at ${position} in each ${receiver} prompt to its value in the paired ${source} ` +
      `prompt would do: (${source} − ${receiver} feature activation) × the gradient of the logit difference along the feature's ` +
      "decoder direction. Nothing is changed; verify the strongest by patching.";
  } else if (exp.kind === "attribution_patching") {
    const [receiver, source] = exp.direction === "clean_to_corrupt" ? ["corrupt", "clean"] : ["clean", "corrupt"];
    how =
      `Estimates, to first order, what replacing ${what} at ${position} in each ${receiver} prompt with its value from the paired ` +
      `${source} prompt would do: (${source} activation − ${receiver} activation) · the gradient of the logit difference at the ` +
      `${receiver} run. Nothing is replaced. The estimate misses saturation and can miss or even invert an effect; verify it by patching.`;
  } else if (exp.kind === "path_patching") {
    const [receiver, source] = exp.direction === "clean_to_corrupt" ? ["corrupt", "clean"] : ["clean", "corrupt"];
    const into = exp.receivers.map((r) => (r.kind === "logits" ? "the logits" : `L${r.layer} H${r.head}'s ${{ q: "query", k: "key", v: "value" }[r.input]}`)).join(", ");
    how =
      `Runs each ${receiver} prompt with ${what} at ${position} taken from the paired ${source} prompt and every other attention head ` +
      `held at its own value${exp.freeze_mlps ? ", MLPs too" : ""}, records what ${into} read, and patches only that into an unchanged ` +
      `${receiver} run. The effect is what travels from this component to the receivers${exp.freeze_mlps ? " directly" : ", directly or through MLPs"}.`;
  } else if (exp.kind === "steering") {
    const toward = exp.apply_to === "clean" ? "corrupt" : "clean";
    const split = summary?.steering;
    how =
      `Adds a direction to ${what} at ${position} in each held-out ${exp.apply_to} prompt: the mean of (${toward} − ${exp.apply_to}) ` +
      `there over the training pairs, times each strength${exp.control ? ", and a random direction of the same length at the same strengths as a control" : ""}. ` +
      (split ? `${split.train.length} pairs computed the directions; the ${split.test.length} held-out pairs are measured.` : "");
  } else if (exp.kind === "activation_patching") {
    how =
      exp.direction === "clean_to_corrupt"
        ? `Runs each corrupt prompt and replaces ${what} at ${position} with its value from the paired clean prompt. Does this restore the behavior?`
        : `Runs each clean prompt and replaces ${what} at ${position} with its value from the paired corrupt prompt. Does this break it?`;
  } else {
    const b = exp.baseline;
    const source =
      b.kind === "zero"
        ? "zeros"
        : b.kind === "mean"
          ? `its mean over the dataset's ${b.reference} prompts`
          : `its value in ${b.donors} randomly drawn ${b.pool} prompts (seed ${b.seed}), averaging the result`;
    how = `Runs each clean prompt and replaces ${what} at ${position} with ${source}.`;
  }
  const receiver = summary?.receiver ?? ((exp.kind === "activation_patching" || exp.kind === "attribution_patching" || exp.kind === "path_patching") && exp.direction === "clean_to_corrupt" ? "corrupt" : "clean");
  const reference = summary?.reference ?? (receiver === "corrupt" ? "clean" : "corrupt");
  const gap = summary?.metric.denominator;
  return (
    <section className={s.section}>
      <h4 className={s.sectionTitle}>{exp.kind === "attribution_patching" ? "Estimate" : "Intervention"}</h4>
      <p className={s.strong}>
        {experimentText(exp)}
        {exp.kind === "ablation" && <span className={s.muted}> · baseline: {baselineText(exp.baseline)}</span>}
      </p>
      <p className={s.text}>{how}</p>
      <h4 className={s.sectionTitle}>Metric</h4>
      <p className={s.text}>
        Logit difference at the last token: logit(answer) − logit(distractor).
      </p>
      <p className={s.text}>
        {exp.kind === "attribution_patching" ? "Estimated effect = (estimated patched" : exp.kind === "steering" ? "Normalized effect = (steered" : "Normalized effect = (patched"} − {receiver}) ÷{" "}
        {spec.metric.normalization === "dataset_gap" ? (
          <>
            mean({reference} − {receiver}){gap !== null && gap !== undefined && <> (denominator {num(gap, 3)})</>}
          </>
        ) : (
          <>each prompt's own ({reference} − {receiver})</>
        )}
        . 0 is no change; 1 is a change as large as switching to the {reference} prompt.
      </p>
    </section>
  );
}

function Evidence({
  site,
  runId,
  summary,
  finished,
}: {
  site: SiteResult;
  runId: string;
  summary: Summary | null;
  finished: boolean;
}) {
  const [detail, setDetail] = useState<SiteDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const setPromptIndex = useStore((st) => st.setPromptIndex);

  useEffect(() => {
    setDetail(null);
    setError(null);
    if (!finished) return;
    let cancelled = false;
    api.siteDetail(runId, site.index).then(
      (d) => !cancelled && setDetail(d),
      (e: Error) => !cancelled && setError(e.message),
    );
    return () => {
      cancelled = true;
    };
  }, [runId, site.index, finished]);

  const stats = summary?.statistics;
  const ciLevel = stats ? Math.round(stats.ci * 100) : 95;
  const exp = useActiveRun().detail?.spec.experiment;
  const words = measureWords(exp);
  const intervention = measureOf(exp) === "intervention";
  const noInterval = site.effect.lo === null || site.effect.hi === null;
  const byIndex = useMemo(() => new Map(detail?.prompts.map((p) => [p.index, p]) ?? []), [detail]);

  return (
    <section className={s.section}>
      <h4 className={s.sectionTitle}>Evidence</h4>
      <div className={s.headline}>
        <span className={s.big}>{signed(site.effect.mean, 3)}</span>
        <span className={s.bigLabel}>{words.mean}</span>
      </div>
      <dl className={s.stats}>
        <dt>{ciLevel}% CI</dt>
        <dd>
          {noInterval ? <span className={s.muted}>needs at least 2 prompts</span> : ci(site.effect.lo, site.effect.hi, 3)}
        </dd>
        <dt>n</dt>
        <dd>{plural(site.n, "prompt")}</dd>
        <dt>SD</dt>
        <dd>{num(site.effect.sd, 3)}</dd>
        <dt>{words.delta}</dt>
        <dd>
          {signed(site.delta.mean, 3)} <span className={s.muted}>({ci(site.delta.lo, site.delta.hi)})</span>
        </dd>
        {intervention && (
          <>
            <dt>Answer prob.</dt>
            <dd>
              {prob(site.answer_prob)} <span className={s.muted}>({signed(site.answer_prob_delta, 3)})</span>
            </dd>
            <dt>Sign flips</dt>
            <dd>
              {count(site.sign_flips)} of {count(site.n)}{" "}
              <span className={s.muted}>prompts changed which name the model prefers</span>
            </dd>
          </>
        )}
        <dt>Opposite sign</dt>
        <dd>
          {count(site.opposite_sign)} of {count(site.n)}{" "}
          <span className={s.muted}>{intervention ? "prompts moved the other way" : "prompts point the other way"}</span>
        </dd>
      </dl>
      {stats && (
        <p className={s.fine}>
          {stats.method[0].toUpperCase() + stats.method.slice(1)}, {count(stats.bootstrap)} resamples, seed {stats.seed}.
        </p>
      )}

      {!finished && <p className={s.note}>Per-prompt evidence appears when the run finishes.</p>}
      {error && <p className={s.note}>{error}</p>}
      {detail && (
        <>
          {detail.dataset_changed && (
            <p className={s.warn}>
              <Icon name="alert" size={13} /> The dataset file changed after this run. Prompt texts may not match.
            </p>
          )}
          {site.variant && <DoseResponse site={site} ciLevel={ciLevel} />}
          <Distribution
            label={`Per-prompt ${words.effect.toLowerCase()}`}
            values={detail.prompts.map((p) => ({ index: p.index, value: p.effect }))}
            mean={site.effect.mean}
            lo={site.effect.lo}
            hi={site.effect.hi}
            onPick={(i) => setPromptIndex(i)}
          />
          <PromptList title="Strongest" indices={detail.strongest} byIndex={byIndex} onPick={setPromptIndex} intervention={intervention} />
          <PromptList title="Weakest" indices={detail.weakest} byIndex={byIndex} onPick={setPromptIndex} intervention={intervention} />
        </>
      )}
    </section>
  );
}

/** A steered site at every strength: the mean effect and its interval, along the direction and
 * along the random control. */
function DoseResponse({ site, ciLevel }: { site: SiteResult; ciLevel: number }) {
  const run = useActiveRun();
  const select = useStore((st) => st.select);
  const row = Object.values(run.results).filter((x) => x.row === site.row && x.variant);
  const series = [false, true].map((control) =>
    row.filter((x) => x.variant?.control === control).sort((a, b) => (a.variant?.coefficient ?? 0) - (b.variant?.coefficient ?? 0)),
  );
  if (series[0].length < 2) return null;
  const xs = row.map((x) => x.variant?.coefficient ?? 0);
  const ys = row.flatMap((x) => [x.effect.lo ?? x.effect.mean ?? 0, x.effect.hi ?? x.effect.mean ?? 0]).concat([0, 1]);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const W = 280, H = 150, L = 34, R = 8, T = 10, B = 24;
  const x = (v: number) => L + ((v - x0) / Math.max(1e-9, x1 - x0)) * (W - L - R);
  const y = (v: number) => T + (1 - (v - y0) / Math.max(1e-9, y1 - y0)) * (H - T - B);
  const description = series[0].map((p) => `${p.variant_key}: ${signed(p.effect.mean, 2)}`).join(", ");
  return (
    <div className={s.dose}>
      <h4 className={s.sectionTitle}>Effect at each strength</h4>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`Mean normalized effect at each strength, with ${ciLevel}% intervals. ${description}.`}>
        <line x1={L} x2={W - R} y1={y(0)} y2={y(0)} className={s.doseZero} />
        <line x1={L} x2={W - R} y1={y(1)} y2={y(1)} className={s.doseOne} />
        <text x={L - 6} y={y(0)} className={s.doseTick} textAnchor="end" dominantBaseline="middle">0</text>
        <text x={L - 6} y={y(1)} className={s.doseTick} textAnchor="end" dominantBaseline="middle">1</text>
        {[x0, x1].map((v) => (
          <text key={v} x={x(v)} y={H - 6} className={s.doseTick} textAnchor="middle">{`×${num(v, Math.abs(v) < 1 ? 1 : 0)}`}</text>
        ))}
        {series.map((points, k) => (
          <g key={k} className={k ? s.doseControl : s.doseDirection}>
            <polyline points={points.map((p) => `${x(p.variant?.coefficient ?? 0)},${y(p.effect.mean ?? 0)}`).join(" ")} />
            {points.map((p) => (
              <g key={p.index}>
                {p.effect.lo !== null && p.effect.hi !== null && (
                  <line x1={x(p.variant?.coefficient ?? 0)} x2={x(p.variant?.coefficient ?? 0)} y1={y(p.effect.lo)} y2={y(p.effect.hi)} />
                )}
                <circle
                  cx={x(p.variant?.coefficient ?? 0)}
                  cy={y(p.effect.mean ?? 0)}
                  r={p.index === site.index ? 4.5 : 3}
                  onClick={() => select(selectionOfSite(p))}
                >
                  <title>{`${p.label}: ${signed(p.effect.mean, 3)} (${ci(p.effect.lo, p.effect.hi)})`}</title>
                </circle>
              </g>
            ))}
          </g>
        ))}
      </svg>
      <p className={s.fine}>
        Along the direction (solid){series[1].length ? " and a random direction of the same length (faint)" : ""}, with {ciLevel}%
        intervals. 1 means the prompts moved as far as switching to the other prompt.
      </p>
    </div>
  );
}

function PromptList({
  title,
  indices,
  byIndex,
  onPick,
  intervention,
}: {
  title: string;
  indices: number[];
  byIndex: Map<number, SiteDetail["prompts"][number]>;
  onPick: (i: number) => void;
  intervention: boolean;
}) {
  return (
    <div className={s.promptList}>
      <h4 className={s.sectionTitle}>{title}</h4>
      {indices.map((i) => {
        const p = byIndex.get(i);
        if (!p) return null;
        return (
          <button key={i} type="button" className={s.prompt} onClick={() => onPick(i)} title="Show this prompt in the token strip">
            <span className={s.promptIndex}>{i}</span>
            <span className={s.promptText}>{p.clean ?? `Prompt ${i}`}</span>
            <span className={s.promptValue}>{signed(p.effect, 2)}</span>
            <span className={s.promptMeta}>
              {p.answer && p.distractor ? `${p.answer.trim()} vs ${p.distractor.trim()} · ` : ""}
              {intervention ? (
                <>
                  logit diff {signed(p.receiver_logit_diff, 2)} → {signed(p.patched_logit_diff, 2)}
                  {p.flipped ? " · flipped" : ""}
                </>
              ) : (
                <>
                  writes {signed(p.delta, 2)} of a logit diff of {signed(p.receiver_logit_diff, 2)}
                </>
              )}
            </span>
          </button>
        );
      })}
    </div>
  );
}

function NotMeasured({ selection }: { selection: Selection }) {
  const run = useActiveRun();
  const others = sitesOnComponent(run.sites, selection);
  if (others.length > 1) {
    const positions = new Set(others.map((x) => x.position_key)).size;
    const what = positions === others.length ? `at ${others.length} positions` : `as ${others.length} sites`;
    return <p className={s.note}>This run measured this component {what}. Choose one in the results.</p>;
  }
  if (others.length === 0 && selection.positionKey !== undefined && sitesOnComponent(run.sites, { ...selection, positionKey: undefined }).length) {
    return <p className={s.note}>The active run didn't measure this component at position {selection.positionKey}.</p>;
  }
  return <p className={s.note}>The active run didn't measure this component.</p>;
}

/** The residual kind of these sites, if they all share one. */
function residKindOnly(sites: { kind: SiteKind }[]): ResidKind | undefined {
  const kinds = new Set(sites.map((x) => x.kind));
  const [only] = kinds;
  return kinds.size === 1 && isResidKind(only) ? only : undefined;
}

function Actions({ selection }: { selection: Selection }) {
  const prefill = useStore((st) => st.prefillExperiment);
  const setView = useStore((st) => st.setView);
  const modelReady = useStore((st) => st.model.state === "ready");
  return (
    <div className={s.actions}>
      <Button size="small" onClick={() => prefill("activation_patching", selection)}>
        Patch here
      </Button>
      <Button size="small" onClick={() => prefill("ablation", selection)}>
        Ablate here
      </Button>
      {selection.part === "head" && (
        <Button size="small" disabled={!modelReady} onClick={() => setView("attention")} title={modelReady ? undefined : "Load a model first"}>
          Open attention
        </Button>
      )}
    </div>
  );
}

function AcrossRuns({ selection }: { selection: Selection }) {
  const runs = useStore((st) => st.runs);
  const runDetails = useStore((st) => st.runDetails);
  const loadRun = useStore((st) => st.loadRun);
  const openRun = useStore((st) => st.openRun);
  const activeRunId = useStore((st) => st.activeRunId);
  const focus = useStore((st) => st.inspectorFocus);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (focus?.section === "runs") {
      setOpen(true);
      useStore.setState({ inspectorFocus: null }); // once: reselecting later doesn't reopen it
      window.setTimeout(() => ref.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
    }
  }, [focus]);

  const finished = useMemo(() => runs.filter((r) => r.status === "finished"), [runs]);
  useEffect(() => {
    if (!open) return;
    // loadRun shares one request per run, so repeated renders don't pile up requests.
    for (const r of finished) if (!runDetails[r.id]?.summary) void loadRun(r.id);
  }, [open, finished, runDetails, loadRun]);

  const rows = finished
    .map((r) => {
      const d: RunDetail | undefined = runDetails[r.id];
      const site = d?.summary ? findSite(d.summary.sites, selection) : null;
      return { run: r, site };
    })
    .filter((x) => x.site);

  const all = rows.flatMap(({ site }) => [site?.effect.lo ?? site?.effect.mean ?? 0, site?.effect.hi ?? site?.effect.mean ?? 0, 0]);
  const lo = Math.min(...all, -0.1);
  const hi = Math.max(...all, 0.1);
  const x = (v: number) => ((v - lo) / (hi - lo)) * 100;

  return (
    <section className={s.section} ref={ref}>
      <button type="button" className={s.disclosure} onClick={() => setOpen(!open)} aria-expanded={open}>
        <Icon name={open ? "chevronDown" : "chevronRight"} size={14} />
        This component across runs
      </button>
      {open && (
        <div className={s.across}>
          {rows.length === 0 && <p className={s.note}>No finished run measured {componentLabel(selection)}.</p>}
          {rows.map(({ run, site }) => (
            <button
              key={run.id}
              type="button"
              className={s.acrossRow}
              aria-current={run.id === activeRunId ? "true" : undefined}
              onClick={() => void openRun(run.id, "results")}
            >
              <span className={s.acrossName}>{run.name}</span>
              <span className={s.acrossValue}>{signed(siteValue(site ?? undefined, "effect"), 2)}</span>
              <span className={s.forest} aria-hidden="true">
                <span className={s.zero} style={{ left: `${x(0)}%` }} />
                {site?.effect.lo != null && site.effect.hi != null && (
                  <span
                    className={s.interval}
                    style={{ left: `${x(site.effect.lo)}%`, width: `${x(site.effect.hi) - x(site.effect.lo)}%` }}
                  />
                )}
                <span className={s.point} style={{ left: `${x(site?.effect.mean ?? 0)}%` }} />
              </span>
            </button>
          ))}
          {rows.length > 0 && (
            <p className={s.fine}>
              Mean normalized effect with its confidence interval in each run. Runs can differ in direction and
              baseline; open one to see its method.
            </p>
          )}
        </div>
      )}
    </section>
  );
}

function RunOverview() {
  const run = useActiveRun();
  const select = useStore((st) => st.select);
  const summary = run.detail?.summary;
  const top = useMemo(() => {
    const list = Object.values(run.results).filter((x) => x.effect.mean !== null);
    return list.sort((a, b) => Math.abs(b.effect.mean ?? 0) - Math.abs(a.effect.mean ?? 0)).slice(0, 5);
  }, [run.results]);

  if (!run.id) {
    return (
      <div className={s.overview}>
        <p className={s.note}>
          Select a component on the map to inspect it. Use the arrow keys to move between cells, and right-click a
          cell for actions.
        </p>
      </div>
    );
  }
  return (
    <div className={s.overview}>
      <span className="eyebrow">Selected run</span>
      <h3 className={s.overviewTitle}>{run.detail?.spec.name ?? "Run"}</h3>
      {run.detail?.spec && <p className={s.componentText}>{experimentText(run.detail.spec.experiment)}</p>}
      {summary && (
        <dl className={s.stats}>
          <dt>Prompts</dt>
          <dd>{count(summary.n_prompts)}</dd>
          <dt>Clean</dt>
          <dd>
            logit diff {signed(summary.baseline.clean.logit_diff.mean)}{" "}
            <span className={s.muted}>· prefers answer in {summary.baseline.clean.prefers_answer}</span>
          </dd>
          <dt>Corrupt</dt>
          <dd>
            logit diff {signed(summary.baseline.corrupt.logit_diff.mean)}{" "}
            <span className={s.muted}>· prefers answer in {summary.baseline.corrupt.prefers_answer}</span>
          </dd>
        </dl>
      )}
      {top.length > 0 && (
        <section className={s.section}>
          <h4 className={s.sectionTitle}>Largest effects</h4>
          {top.map((site) => (
            <button
              key={site.index}
              type="button"
              className={s.topRow}
              onClick={() => select(selectionOfSite(site))}
            >
              <span>{site.label}</span>
              <span className={s.topValue}>{signed(site.effect.mean, 3)}</span>
              <span className={s.muted}>{ci(site.effect.lo, site.effect.hi)}</span>
            </button>
          ))}
        </section>
      )}
      {summary?.warnings.map((w) => (
        <p key={w} className={s.warn}>
          <Icon name="alert" size={13} /> {w}
        </p>
      ))}
      <p className={s.fine}>Select a cell on the map or in the results for its evidence.</p>
    </div>
  );
}
