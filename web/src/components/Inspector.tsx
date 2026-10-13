import { useEffect, useMemo, useRef, useState } from "react";

import { api } from "../api/client";
import type { PositionSpec, ResultKind, RunDetail, SAERef, SiteDetail, SiteResult, Spec, Summary } from "../api/types";
import { capitalize, ci, count, num, plural, prob, qText, signed } from "../lib/format";
import { siteValue, useActiveRun, useAnalysisContext, useArchitecture } from "../lib/hooks";
import {
  componentLabel,
  findSite,
  isResidKind,
  KIND_NAMES,
  queryHeadsOf,
  queryHeadsText,
  selectionOfSite,
  sitesOnComponent,
  type ResidKind,
  type Selection,
} from "../lib/sites";
import { setWhat, universeText } from "../lib/circuits";
import { metricWords } from "../lib/metrics";
import { runTitle } from "../lib/runs";
import { siteText } from "../lib/spec";
import { baselineText, experimentText, measureOf, measureWords, positionText, receiverInputWords, senderWords, sharesKeys } from "../lib/spec";
import { prefersReducedMotion } from "../lib/motion";
import { useStore } from "../store/app";
import { Distribution } from "./Distribution";
import { FeatureWrites } from "./FeatureWrites";
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
            {run.detail?.spec && pendingBase && (
              <p className={s.note}>
                {experimentText(run.detail.spec.experiment)} ·{" "}
                {pendingBase.kind === "site_set"
                  ? "every site of the set at once"
                  : pendingBase.kind === "edge"
                    ? run.detail.spec.experiment.kind === "edge_attribution"
                      ? "summed over every position"
                      : "at every position"
                    : positionText(pendingBase.position)}
              </p>
            )}
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
            {site && selection.part === "set" && run.detail?.summary?.circuit && <SetReading site={site} summary={run.detail.summary} />}
            {run.detail?.spec && (site || pendingBase) && <Method spec={run.detail.spec} summary={run.detail.summary} selection={selection} />}
            {selection.part === "feature" && selection.feature !== undefined && run.detail?.spec?.sae && (
              <WhatItWrites
                saeRef={run.detail.spec.sae}
                feature={selection.feature}
                position={pendingBase?.position ?? null}
                receiver={run.detail.summary?.receiver ?? "clean"}
              />
            )}
            {selection.part !== "set" && selection.part !== "edge" && <Actions selection={selection} />}
            {/* A set's label names different sites in different runs: no comparison across runs. */}
            {selection.part !== "set" && <AcrossRuns selection={selection} />}
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
  else if (selection.part === "input") kind = inputText(selection, arch);
  else if (selection.part === "neuron") {
    kind = `Neuron ${selection.neuron} of the MLP in layer ${selection.layer}: its activation after the nonlinearity (for a gated MLP, the gate's activation times the up projection), the input to the MLP's output projection. It writes that activation times its row of the projection.`;
  }
  else if (selection.part === "edge") {
    const edge = findSite(run.sites, selection)?.edge;
    const grouped = sharesKeys(run.detail?.summary?.model);
    kind = edge
      ? `An edge of the model's graph: ${senderWords(edge.sender)}, read directly by ${receiverInputWords(edge.receiver, grouped)} through the residual stream.${grouped && edge.receiver.kind === "head" && edge.receiver.input !== "q" ? " The key/value head's key and value are read by every query head of its group." : ""}`
      : "An edge of the model's graph: a sender's output, read directly by a later component's input.";
  }
  else if (selection.part === "set") {
    const set = findSite(run.sites, selection);
    const members = set?.members ?? [];
    const universe = run.detail?.summary?.circuit?.universe ?? run.layout?.universe;
    const what = set ? setWhat(!!set.variant?.complement, set.variant?.size ?? members.length, universe) : "intervenes on a set of sites";
    kind = `A set of sites, intervened on together in one forward pass: it ${what}${members.length ? `. Its sites: ${members.map(siteText).join(", ")}.` : "."}`;
  }
  else {
    const found = sitesOnComponent(run.sites, selection)[0]?.kind;
    const resid = selection.kind ?? (found && isResidKind(found) ? found : undefined);
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
      {arch && selection.part !== "set" && selection.part !== "edge" && selection.layer >= arch.nLayers && <p className={s.note}>This model has only {arch.nLayers} layers.</p>}
    </div>
  );
}

/** What a head's query, key or value is, in words: with grouped-query attention a key or value
 * belongs to a key/value head, which every query head of its group reads. */
function inputText(selection: Selection, arch: { nHeads: number; nKvHeads?: number } | null): string {
  const layer = selection.layer;
  if (selection.kind === "q") {
    return `The query of attention head ${selection.head} in layer ${layer}: what the head compares with every key to choose where to attend, as its projection computes it (before any rotary embedding).`;
  }
  const what = selection.kind === "v" ? "value" : "key";
  const role = selection.kind === "v" ? "what the heads that read it take from each position they attend to" : "what each query compares with to choose where to attend";
  const kv = arch?.nKvHeads ?? arch?.nHeads;
  const readers =
    arch && kv && kv < arch.nHeads
      ? ` It is read by ${queryHeadsText(queryHeadsOf(selection.head ?? 0, arch.nHeads, kv))} (grouped-query attention), so changing it changes all of them.`
      : "";
  return `The ${what} of key/value head ${selection.head} in layer ${layer}: ${role}.${readers}`;
}

/** A set against replacing everything: its share, its faithfulness, what a site adds to it, and
 * how its two sites interact. */
function SetReading({ site, summary }: { site: SiteResult; summary: Summary }) {
  const row = summary.circuit?.rows.find((r) => r.index === site.index);
  if (!row) return null;
  const everything = summary.circuit?.everything === site.index;
  return (
    <section className={s.section}>
      <h3 className={s.sectionTitle}>Against replacing everything</h3>
      <dl className={s.stats}>
        {everything ? (
          <>
            <dt>Share</dt>
            <dd>
              1 <span className={s.muted}>this set replaces everything; the others are read against it</span>
            </dd>
          </>
        ) : (
          <>
            <dt>Share</dt>
            <dd>
              {num(row.share?.mean, 3)} <span className={s.muted}>({ci(row.share?.lo, row.share?.hi)})</span>
            </dd>
          </>
        )}
        {row.faithfulness && (
          <>
            <dt>Faithfulness</dt>
            <dd>
              {num(row.faithfulness.mean, 3)} <span className={s.muted}>({ci(row.faithfulness.lo, row.faithfulness.hi)})</span>
            </dd>
          </>
        )}
        {row.without && (
          <>
            <dt>Without {row.without.site}</dt>
            <dd>
              −{num(row.without.drop.mean, 3)} <span className={s.muted}>faithfulness of {row.without.of} ({ci(row.without.drop.lo, row.without.drop.hi)})</span>
            </dd>
          </>
        )}
        {row.interaction && (
          <>
            <dt>Beyond the sum</dt>
            <dd>
              {signed(row.interaction.effect.mean, 3)} <span className={s.muted}>{row.interaction.a} and {row.interaction.b} together ({ci(row.interaction.effect.lo, row.interaction.effect.hi)})</span>
            </dd>
          </>
        )}
      </dl>
      <p className={s.fine}>
        Share: this set's change in the metric as a share of replacing everything. Faithfulness: 1 − share, for a set that keeps
        its sites; 1 means they carry the behavior alone.
      </p>
    </section>
  );
}

function Method({ spec, summary, selection }: { spec: Spec; summary: Summary | null; selection: Selection }) {
  const exp = spec.experiment;
  const measured = metricWords(summary?.metric ?? spec.metric);
  const grouped = sharesKeys(summary?.model);
  const base = summary ? findSite(summary.sites, selection) : null;
  const position = base ? positionText(base.position) : selection.positionKey ? `position ${selection.positionKey}` : "the positions in the spec";
  const kind = base?.kind ?? selection.kind;
  const resid = kind && isResidKind(kind) ? kind : undefined;
  const what =
    selection.part === "head"
      ? "this head's output"
      : selection.part === "attn"
        ? "the attention output"
        : selection.part === "mlp"
          ? "the MLP output"
          : selection.part === "feature"
            ? `feature ${selection.feature} of the SAE`
            : selection.part === "input"
              ? selection.kind === "q"
                ? "this head's query"
                : `this key/value head's ${selection.kind === "v" ? "value" : "key"}`
              : selection.part === "neuron"
                ? "this neuron's activation"
                : resid
                  ? `the ${KIND_NAMES[resid]}`
                  : "the residual stream";
  let how: string;
  if (selection.part === "set" && (exp.kind === "activation_patching" || exp.kind === "ablation")) {
    const set = base ?? null;
    const keeps = !!set?.variant?.complement;
    const universe = universeText(summary?.circuit?.universe ?? summary?.layout.universe);
    const [receiver, source] = exp.kind === "activation_patching" && exp.direction === "clean_to_corrupt" ? ["corrupt", "clean"] : ["clean", "corrupt"];
    const value = exp.kind === "ablation" ? baselineText(exp.baseline) : `its value from the paired ${source} prompt`;
    how = keeps
      ? `Runs each ${receiver} prompt and replaces ${universe} except the set's sites, all at once, with ${value}. A site at one position keeps only that position. What changes is what the rest of the model carried.`
      : `Runs each ${receiver} prompt and replaces every site of the set, all at once, with ${value}.`;
  } else if (exp.kind === "direct_logit_attribution") {
    const written =
      selection.part === "head"
        ? "this head's output, through its share of the output projection,"
        : selection.part === "neuron"
          ? "this neuron's activation times its row of the MLP's output projection"
          : `${what}`;
    const value = selection.part === "input" && selection.kind === "v";
    return (
      <section className={s.section}>
        <h3 className={s.sectionTitle}>Method</h3>
        <p className={s.strong}>{experimentText(exp)}</p>
        <p className={s.text}>
          {value ? (
            <>
              Runs each {exp.prompts} prompt once and reads what this value at {position} adds to the last token: through every
              query head that reads it, its attention to that position times the value, through the head's share of the output
              projection (the attention pattern doesn't depend on the values), then through the final normalization with its scale
              held at its value in the run. This is its direct effect. Nothing is replaced.
            </>
          ) : (
            <>
              Runs each {exp.prompts} prompt once and reads {written} at the last token through the final normalization, with its
              scale held at its value in the run. This is the component's direct effect: it leaves out what the component does
              through later components, which patching measures. Nothing is replaced.
            </>
          )}
        </p>
        <h3 className={s.sectionTitle}>Metric</h3>
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
  } else if (selection.part === "edge" && exp.kind === "acdc") {
    const [receiver, source] = exp.direction === "clean_to_corrupt" ? ["corrupt", "clean"] : ["clean", "corrupt"];
    const edge = base?.edge;
    const into = edge ? receiverInputWords(edge.receiver, grouped) : "its receiver";
    const kept = base && summary?.acdc ? summary.acdc.kept.includes(base.index) : null;
    const counted = exp.rule === "toward_source" ? `toward the ${source} prompts'` : "either way";
    how =
      `ACDC tested the edges from the logits back toward the embeddings, this one after every edge downstream of ${into}. It ran each ${receiver} prompt with the ` +
      `edges removed so far, and this one, carrying their senders' outputs from the paired ${source} prompt, and compared the mean ` +
      `of the metric with the circuit's just before. ` +
      (kept === null
        ? `An edge stays removed when that moves the metric ${counted} by less than ${exp.threshold}.`
        : kept
          ? `It moved the metric ${counted} by ${exp.threshold} or more, so the edge was put back: it is in the circuit.`
          : `It moved the metric ${counted} by less than ${exp.threshold}, so the edge stayed removed.`);
  } else if (selection.part === "edge" && (exp.kind === "edge_attribution" || exp.kind === "path_patching")) {
    const [receiver, source] = exp.direction === "clean_to_corrupt" ? ["corrupt", "clean"] : ["clean", "corrupt"];
    const edge = base?.edge;
    const from = edge ? senderWords(edge.sender) : "the sender's output";
    const into = edge ? receiverInputWords(edge.receiver, grouped) : "the receiver's input";
    how =
      exp.kind === "edge_attribution"
        ? `Estimates, to first order, what letting only ${into} read ${from} from the paired ${source} prompt would do in each ${receiver} prompt: ` +
          `(${source} − ${receiver} output of the sender) · ${edgeGradient(exp, receiver, measured.label)}, summed over the positions. ` +
          "Nothing is replaced; verify the strongest edges by path patching."
        : `Runs each ${receiver} prompt with ${from} taken from the paired ${source} prompt and every other head and MLP held at its own value, ` +
          `records what ${into} reads, and patches only that into an unchanged ${receiver} run. The effect is what travels along this edge ` +
          "alone, directly through the residual stream.";
  } else if (selection.part === "feature" && exp.kind === "attribution_patching") {
    const [receiver, source] = exp.direction === "clean_to_corrupt" ? ["corrupt", "clean"] : ["clean", "corrupt"];
    how =
      `Estimates, to first order, what changing ${what} at ${position} in each ${receiver} prompt to its value in the paired ${source} ` +
      `prompt would do: (${source} − ${receiver} feature activation) × ${gradient(exp, receiver, measured.label)} along the feature's ` +
      "decoder direction. Nothing is changed; verify the strongest by patching.";
  } else if (exp.kind === "attribution_patching") {
    const [receiver, source] = exp.direction === "clean_to_corrupt" ? ["corrupt", "clean"] : ["clean", "corrupt"];
    how =
      `Estimates, to first order, what replacing ${what} at ${position} in each ${receiver} prompt with its value from the paired ` +
      `${source} prompt would do: (${source} activation − ${receiver} activation) · ${gradient(exp, receiver, measured.label)}. ` +
      `Nothing is replaced. ${exp.method === "integrated_gradients" ? "Averaging the gradient along the way corrects much of the saturation one gradient misses, but it is still an estimate" : "The estimate misses saturation and can miss or even invert an effect"}; verify it by patching.`;
  } else if (exp.kind === "path_patching") {
    const [receiver, source] = exp.direction === "clean_to_corrupt" ? ["corrupt", "clean"] : ["clean", "corrupt"];
    const into = exp.receivers.map((r) => receiverInputWords(r, grouped)).join(", ");
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
  } else if (exp.kind === "ablation") {
    const b = exp.baseline;
    const source =
      b.kind === "zero"
        ? "zeros"
        : b.kind === "mean"
          ? `its mean over the dataset's ${b.reference} prompts`
          : `its value in ${b.donors} randomly drawn ${b.pool} prompts (seed ${b.seed}), averaging the result`;
    how = `Runs each clean prompt and replaces ${what} at ${position} with ${source}.`;
  } else {
    how = "";
  }
  const receiver =
    summary?.receiver ??
    ((exp.kind === "activation_patching" || exp.kind === "attribution_patching" || exp.kind === "path_patching" || exp.kind === "edge_attribution" || exp.kind === "acdc") &&
    exp.direction === "clean_to_corrupt"
      ? "corrupt"
      : "clean");
  const estimate = exp.kind === "attribution_patching" || exp.kind === "edge_attribution";
  const metricText = `${measured.formula}${(summary?.metric ?? spec.metric).kind === "logit_diff" ? " at the last token" : ""}`;
  const reference = summary?.reference ?? (receiver === "corrupt" ? "clean" : "corrupt");
  const gap = summary?.metric.denominator;
  return (
    <section className={s.section}>
      <h3 className={s.sectionTitle}>{estimate ? "Estimate" : "Intervention"}</h3>
      <p className={s.strong}>
        {experimentText(exp)}
        {exp.kind === "ablation" && <span className={s.muted}> · baseline: {baselineText(exp.baseline)}</span>}
      </p>
      <p className={s.text}>{how}</p>
      <h3 className={s.sectionTitle}>Metric</h3>
      <p className={s.text}>
        {capitalize(measured.label)}: {metricText}.
      </p>
      <p className={s.text}>
        {estimate
          ? `Estimated effect = (estimated patched − ${receiver})`
          : exp.kind === "acdc"
            ? "Normalized change = (circuit without the edge − circuit with it)"
            : exp.kind === "steering"
              ? `Normalized effect = (steered − ${receiver})`
              : `Normalized effect = (patched − ${receiver})`}{" "}
        ÷{" "}
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

/** For a feature of the run's SAE: the tokens it promotes and suppresses directly, read with the
 * loaded model and SAE at the site's position in the chosen prompt of the run's receiver side. */
function WhatItWrites({
  saeRef,
  feature,
  position,
  receiver,
}: {
  saeRef: SAERef;
  feature: number;
  position: PositionSpec | null;
  receiver: "clean" | "corrupt";
}) {
  const sae = useStore((st) => st.sae);
  const modelReady = useStore((st) => st.model.state === "ready");
  const dataset = useStore((st) => st.datasetPath);
  const index = useStore((st) => st.promptIndex);
  const jobRunning = useStore((st) => st.job?.status === "running");
  const guard = useStore((st) => st.guard);
  const context = useAnalysisContext();
  const info = sae.state === "ready" ? sae.info : undefined;
  const trim = (path: string) => path.replace(/^\/+|\/+$/g, "");
  const loaded =
    !!info && info.repo === saeRef.repo && trim(info.path) === trim(saeRef.path) && (!saeRef.revision || saeRef.revision === info.revision);
  const load = async () => {
    const out = await guard(() => api.loadSae({ repo: saeRef.repo, path: saeRef.path, revision: saeRef.revision }));
    if (out) useStore.getState().applyJob(out);
  };
  return (
    <section className={s.section}>
      <h3 className={s.sectionTitle}>Tokens it promotes and suppresses directly</h3>
      {!modelReady ? (
        <p className={s.note}>Load the run's model to read what this feature writes.</p>
      ) : context.error ? (
        <p className={s.note}>{context.error}</p>
      ) : !loaded ? (
        <>
          <p className={s.note}>
            Load this run's SAE ({saeRef.repo}
            {saeRef.path ? ` · ${saeRef.path}` : ""}) to read what this feature writes.
          </p>
          <div>
            <Button size="small" disabled={sae.state === "loading" || jobRunning} onClick={() => void load()}>
              {sae.state === "loading" ? "Loading the SAE…" : "Load the SAE"}
            </Button>
          </div>
        </>
      ) : dataset ? (
        <>
          <FeatureWrites
            dataset={dataset}
            index={index}
            which={receiver}
            feature={feature}
            position={position?.kind === "all" ? null : position}
            options={context.options}
            optionsKey={context.key}
            sae={`${info.repo}/${info.path}@${info.revision}`}
            standardized={info.normalize === "layer_norm"}
            compact
          />
          <p className={s.fine}>
            Read in {receiver} prompt {index}, where the run measures the feature's change. Choose another prompt in the
            evidence to read it there.
          </p>
        </>
      ) : null}
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
  const active = useActiveRun().detail;
  const exp = active?.spec.experiment;
  const measuredBy = summary?.metric ?? active?.spec.metric;
  const words = measureWords(exp, measuredBy);
  const short = metricWords(measuredBy).short;
  const intervention = measureOf(exp) === "intervention";
  const noInterval = site.effect.lo === null || site.effect.hi === null;
  const byIndex = useMemo(() => new Map(detail?.prompts.map((p) => [p.index, p]) ?? []), [detail]);
  const values = useMemo(() => detail?.prompts.map((p) => ({ index: p.index, value: p.effect })) ?? [], [detail]);
  // A steered site and strength against its random control, paired over the same resamples.
  const control = summary?.steering?.control?.find((c) => c.index === site.index || c.control_index === site.index);
  // An edge attribution run lists only its strongest edges; its corrections cover every edge.
  const nSites = summary?.edges?.n_edges ?? summary?.sites.length ?? 0;
  const unit = summary?.layout.kind === "site_sets" ? "sets" : summary?.layout.kind === "edges" ? "edges" : "sites";

  return (
    <section className={s.section}>
      <h3 className={s.sectionTitle}>Evidence</h3>
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
              <span className={s.muted}>prompts changed whether the model prefers the answer</span>
            </dd>
          </>
        )}
        <dt>Opposite sign</dt>
        <dd>
          {count(site.opposite_sign)} of {count(site.n)}{" "}
          <span className={s.muted}>{intervention ? "prompts moved the other way" : "prompts point the other way"}</span>
        </dd>
        {site.band && (
          <>
            <dt>Band, all {count(nSites)} {unit}</dt>
            <dd>
              {ci(site.band.lo, site.band.hi, 3)}{" "}
              <span className={s.muted}>{site.band.lo > 0 || site.band.hi < 0 ? "excludes zero" : "includes zero"}</span>
            </dd>
          </>
        )}
        {site.q !== null && site.q !== undefined && (
          <>
            <dt>q-value</dt>
            <dd>
              {qText(site.q)} <span className={s.muted}>the share of false findings expected among {unit} this strong</span>
            </dd>
          </>
        )}
        {control && (
          <>
            <dt>Against the random control</dt>
            <dd>
              {signed(control.difference, 3)} <span className={s.muted}>({ci(control.lo, control.hi)}) {control.beats_control ? "beats it" : "doesn't beat it"}</span>
            </dd>
          </>
        )}
      </dl>
      {stats && (
        <p className={s.fine}>
          {[
            capitalize(stats.method),
            stats.cluster && stats.clusters ? `${count(stats.clusters)} clusters` : null,
            `${count(stats.bootstrap)} resamples`,
            `seed ${stats.seed}`,
          ].filter(Boolean).join(", ")}.
          {site.band && ` The band holds for all ${count(nSites)} ${unit} at once, with ${ciLevel}% confidence; the q-value is Benjamini–Hochberg's.`}
          {control && ` Against the control: |effect| along the direction − |effect| along a random direction of the same length, at the same strength.`}
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
          {site.variant?.coefficient !== undefined && <DoseResponse site={site} ciLevel={ciLevel} />}
          <Distribution
            label={`Per-prompt ${words.effect.toLowerCase()}`}
            values={values}
            mean={site.effect.mean}
            lo={site.effect.lo}
            hi={site.effect.hi}
            onPick={(i) => setPromptIndex(i)}
          />
          <PromptList title="Strongest" indices={detail.strongest} byIndex={byIndex} onPick={setPromptIndex} intervention={intervention} short={short} />
          <PromptList title="Weakest" indices={detail.weakest} byIndex={byIndex} onPick={setPromptIndex} intervention={intervention} short={short} />
          <p className={s.fine}>Choose a prompt to show it in the token strip.</p>
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
      <h3 className={s.sectionTitle}>Effect at each strength</h3>
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
  short,
}: {
  title: string;
  indices: number[];
  byIndex: Map<number, SiteDetail["prompts"][number]>;
  onPick: (i: number) => void;
  intervention: boolean;
  /** The run's metric, as a label. */
  short: string;
}) {
  return (
    <div className={s.promptList}>
      <h3 className={s.sectionTitle}>{title}</h3>
      {indices.map((i) => {
        const p = byIndex.get(i);
        if (!p) return null;
        return (
          <button key={i} type="button" className={s.prompt} onClick={() => onPick(i)}>
            <span className={s.promptIndex}>{i}</span>
            <span className={s.promptText}>{p.clean ?? `Prompt ${i}`}</span>
            <span className={s.promptValue}>{signed(p.effect, 2)}</span>
            <span className={s.promptMeta}>
              {p.answer && p.distractor ? `${p.answer.trim()} vs ${p.distractor.trim()} · ` : ""}
              {intervention ? (
                <>
                  {p.receiver_metric !== undefined && p.receiver_metric !== null ? (
                    <>{short} {signed(p.receiver_metric, 2)} → {signed(p.patched_metric, 2)}</>
                  ) : (
                    <>logit diff {signed(p.receiver_logit_diff, 2)} → {signed(p.patched_logit_diff, 2)}</>
                  )}
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
function residKindOnly(sites: { kind: ResultKind }[]): ResidKind | undefined {
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
        <Button size="small" disabled={!modelReady} onClick={() => setView("attention")} tip={modelReady ? undefined : "Load a model first"}>
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
      window.setTimeout(() => ref.current?.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" }), 50);
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
              <span className={s.acrossName}>{runTitle(run)}</span>
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
  const short = metricWords(summary?.metric ?? run.detail?.spec.metric).short;
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
      <h2 className={s.overviewTitle}>{(run.detail && run.title) || "Run"}</h2>
      {run.detail?.spec && <p className={s.componentText}>{experimentText(run.detail.spec.experiment)}</p>}
      {summary && (
        <dl className={s.stats}>
          <dt>Prompts</dt>
          <dd>{count(summary.n_prompts)}</dd>
          <dt>Clean</dt>
          <dd>
            {short} {signed((summary.baseline.clean.metric ?? summary.baseline.clean.logit_diff).mean)}{" "}
            <span className={s.muted}>· prefers answer in {summary.baseline.clean.prefers_answer}</span>
          </dd>
          <dt>Corrupt</dt>
          <dd>
            {short} {signed((summary.baseline.corrupt.metric ?? summary.baseline.corrupt.logit_diff).mean)}{" "}
            <span className={s.muted}>· prefers answer in {summary.baseline.corrupt.prefers_answer}</span>
          </dd>
        </dl>
      )}
      {top.length > 0 && (
        <section className={s.section}>
          <h3 className={s.sectionTitle}>Largest effects</h3>
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

/** The gradient an estimate multiplies by: at the receiver run, or averaged along the way. */
/** The gradient an edge's estimate takes: with respect to the receiver's own copy of the residual
 * stream, through its normalization with the scale recomputed (src/logogram/edges.py). */
function edgeGradient(exp: { method?: string; steps?: number | null }, receiver: string, metric: string): string {
  const where =
    "with respect to the receiver's own copy of the residual stream, before its normalization and through it with its scale recomputed, as patching the edge would recompute it";
  return exp.method === "integrated_gradients"
    ? `the gradient of the ${metric} ${where}, averaged over ${exp.steps} runs between the ${receiver} prompt and the other (integrated gradients)`
    : `the gradient of the ${metric} ${where}, at the ${receiver} run`;
}

function gradient(exp: { method?: string; steps?: number | null }, receiver: string, metric: string): string {
  return exp.method === "integrated_gradients"
    ? `the gradient of the ${metric}, averaged over ${exp.steps} runs between the ${receiver} prompt and the other (integrated gradients)`
    : `the gradient of the ${metric} at the ${receiver} run`;
}
