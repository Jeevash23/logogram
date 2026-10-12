import { useEffect, useId, useState } from "react";

import { api } from "../api/client";
import type { ExperimentKind, RunMemory, Spec } from "../api/types";
import { MOD } from "../components/Header";
import { RunButton } from "../components/RunButton";
import { Button, Callout, Kbd } from "../components/ui";
import { chatAdvice, chatReason } from "../lib/chat";
import { bytes, count, plural } from "../lib/format";
import type { EdgePlanState } from "../lib/edgePlan";
import { modelName, useAnalysisContext } from "../lib/hooks";
import { metricWords } from "../lib/metrics";
import { experimentText, scopeText, workloadText } from "../lib/spec";
import { behaviorPresent, PRESENT_GAP } from "../lib/thresholds";
import { runnerName, runWouldQueue, useStore } from "../store/app";
import e from "./ExperimentView.module.css";

/**
 * Beside the experiment form: what will run and what it takes, what is still missing, and the
 * button that runs it. It stays in view while the form scrolls, so the reason Run can't be
 * pressed is always next to it.
 */
export function RunSummary({
  spec,
  error,
  method,
  batchSize,
  n,
  rows,
  differences,
  onRun,
  onQueue,
  edgePlan = null,
}: {
  spec: Spec | null;
  /** Why the form can't run yet. */
  error: string | null;
  method: ExperimentKind;
  batchSize: number;
  /** Prompts the run uses. */
  n: number;
  /** Rows the sweep runs, when known. */
  rows: number | null;
  /** How the run would differ from the saved spec it was opened from. */
  differences: string[];
  onRun: (spec: Spec) => void;
  onQueue: (spec: Spec) => void;
  /** Edge attribution: the graph and what a run on the loaded model costs. */
  edgePlan?: EdgePlanState | null;
}) {
  const titleId = useId();
  const waiting = useStore(runWouldQueue);
  const runner = useStore((st) => st.queueRunner);
  const loaded = useStore((st) => (st.model.state === "ready" ? st.model.info : undefined));
  // Wrapped in the model's chat template, its prompts read as a chat: nothing to point out.
  const wrapped = useStore((st) => st.form.chatTemplate !== null);
  // The loaded model runs the spec, so what is known about it applies when the spec names it.
  const chat = loaded && !wrapped && (!spec || spec.model.id === loaded.id) ? chatReason(loaded) : null;
  return (
    <section className={e.panel} aria-labelledby={titleId}>
      <h2 id={titleId} className={e.panelTitle}>
        {spec ? "Ready to run" : "Not ready to run"}
      </h2>
      {spec ? (
        <p className={e.summaryText}>
          <strong>{experimentText(spec.experiment)}</strong> at {scopeText(spec.scope)}, over {plural(n, "prompt")}, measured
          by the {metricWords(spec.metric).label}.{rows !== null && <> {workloadText(spec, rows)}</>}
        </p>
      ) : (
        <p className={e.missing}>{error}</p>
      )}
      <dl className={e.facts}>
        {method === "edge_attribution" ? (
          <EdgeFacts plan={edgePlan} />
        ) : (
          <>
            <dt>Memory</dt>
            <dd>
              <MemoryFit method={method} batchSize={batchSize} />
            </dd>
          </>
        )}
        <dt>Baseline</dt>
        <dd>
          <BaselineState />
        </dd>
        {chat && (
          <>
            <dt>Prompt format</dt>
            <dd>
              <span>
                Plain text, without the chat template {modelName(loaded?.id)} is made for: results describe how it behaves on
                plain text, which can differ from how it behaves in a chat.
              </span>
              <span className={e.factNote}>
                {chat} {chatAdvice(loaded)}
              </span>
            </dd>
          </>
        )}
      </dl>
      {differences.length > 0 && (
        <Callout title="This runs with something other than the saved spec">
          {differences.join(" ")} Running uses what is loaded and chosen now, and records it in the new spec.
        </Callout>
      )}
      <div className={e.runRow}>
        <RunButton spec={spec} reason={error} onRun={onRun} onQueue={onQueue} />
        <span className={e.runHint}>
          {waiting ? (
            runner ? (
              `The queue is being run by ${runnerName(runner)}: this one joins its end.`
            ) : (
              "Another job is running: this one waits in the queue for its turn."
            )
          ) : (
            <>
              or <Kbd>{MOD}</Kbd> <Kbd>Enter</Kbd>
            </>
          )}
        </span>
      </div>
    </section>
  );
}

/** Whether a batch of this size fits in the memory free with the loaded model, by the server's
 * estimate for the method (some hold every layer's activations at once). */
function MemoryFit({ method, batchSize }: { method: ExperimentKind; batchSize: number }) {
  const ready = useStore((st) => st.model.state === "ready");
  const lengths = useStore((st) => st.dataset?.lengths);
  const seqLen = lengths && lengths.length > 0 ? Math.max(...lengths) : null;
  const usable = Number.isFinite(batchSize) && batchSize >= 1 && batchSize <= 4096;
  const [memory, setMemory] = useState<RunMemory | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setMemory(null);
    setError(null);
    if (!ready || seqLen === null || !usable) return;
    let cancelled = false;
    // While a number is typed, ask once it settles.
    const timer = window.setTimeout(() => {
      api.runMemory({ batch_size: Math.round(batchSize), seq_len: seqLen, method }).then(
        (m) => !cancelled && setMemory(m),
        (err: Error) => !cancelled && setError(err.message),
      );
    }, 250);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [ready, seqLen, batchSize, usable, method]);

  if (!ready) return <span className="faint">Known once a model is loaded.</span>;
  if (seqLen === null) return <span className="faint">Known once the prompts are read by the model.</span>;
  if (!usable) return <span className="faint">Choose a batch size from 1 to 4,096.</span>;
  if (error) return <span className="faint">{error}</span>;
  if (!memory) return <span className="faint">Estimating…</span>;
  const verdict =
    memory.verdict === "fits"
      ? "it fits."
      : memory.verdict === "tight"
        ? "it's tight: a smaller batch size leaves room."
        : "it won't fit: use a smaller batch size.";
  return (
    <>
      <span className={memory.verdict === "wont_fit" ? e.alert : undefined}>
        About {bytes(memory.total)} of the {bytes(memory.available)} free: {verdict}
      </span>
      <span className={e.factNote}>{memory.explanation}</span>
    </>
  );
}

/** An edge attribution run on the loaded model: its graph's size, the passes it takes and the
 * memory a batch needs against what is free, from the checks the run makes. */
function EdgeFacts({ plan }: { plan: EdgePlanState | null }) {
  const ready = useStore((st) => st.model.state === "ready");
  const p = plan?.plan;
  const pending = <span className="faint">{plan?.pending ? "Counting…" : !ready ? "Known once a model is loaded." : plan?.error ?? "Known once the experiment is ready."}</span>;
  return (
    <>
      <dt>Graph</dt>
      <dd>{p?.n_edges != null ? <span>{count(p.n_edges)} edges, from {count(p.n_senders)} senders into {count(p.n_receivers)} inputs</span> : pending}</dd>
      <dt>Cost</dt>
      <dd>
        {p?.cost ? (
          <>
            <span className={p.refusal ? e.alert : undefined}>
              {plural(p.cost.batches, "batch", "batches")} of {count(p.cost.forward_per_batch)} forward and{" "}
              {plural(p.cost.backward_per_batch, "backward pass", "backward passes")} each
              {p.cost.patched_per_batch ? `, then ${count(p.cost.patched_per_batch)} edge-patched forward passes each for the curve` : ""}.
            </span>
            <span className={e.factNote}>
              About {bytes(p.cost.memory_per_batch)} a batch{p.free != null ? ` of the ${bytes(p.free)} free` : ""}, and{" "}
              {count(p.cost.values)} per-prompt values kept.{p.refusal ? ` ${p.refusal}` : ""}
            </span>
          </>
        ) : p?.refusal ? (
          <span className={e.alert}>{p.refusal}</span>
        ) : (
          pending
        )}
      </dd>
    </>
  );
}

/** Whether the baseline was checked for these prompts, and what it found. */
function BaselineState() {
  const context = useAnalysisContext();
  const report = useStore((st) => st.baselines[context.baselineKey]);
  const open = () => useStore.getState().setView("baseline");
  if (!report?.summary) {
    return (
      <>
        <span>Not checked for these prompts yet.</span>
        <Button size="small" variant="ghost" className={e.factAction} onClick={open}>
          Check the baseline
        </Button>
      </>
    );
  }
  return behaviorPresent(report.summary) ? (
    <span>Checked: the behavior is present.</span>
  ) : (
    <>
      <span>Checked: the behavior is weak or missing (a gap of {PRESENT_GAP} or less, or the clean prompts don't prefer the answer).</span>
      <Button size="small" variant="ghost" className={e.factAction} onClick={open}>
        Open the baseline
      </Button>
    </>
  );
}
