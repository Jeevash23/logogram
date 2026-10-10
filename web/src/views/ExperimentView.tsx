import { useEffect, useMemo, useState } from "react";

import type { BaselineSpec, ExperimentKind, PathReceiverSpec, PositionSpec, SAEInfo, ScopeSpec, StreamKind } from "../api/types";
import { Button, Callout, Checkbox, Choices, Field, Input, Kbd, Segmented, Select, TextArea } from "../components/ui";
import { MOD } from "../components/Header";
import { plural, shortRevision } from "../lib/format";
import { modelName } from "../lib/hooks";
import { KIND_SHORT, parseHeadQuery } from "../lib/sites";
import { buildSpec, datasetFacts, parseStrengths, savedDifferences } from "../lib/buildSpec";
import { experimentText, receiverLabel, scopeFor, scopeShort, scopeText, siteText, suggestName, workload, workloadText } from "../lib/spec";
import { useStore, type FormState } from "../store/app";
import { metricWords } from "../lib/metrics";
import { MetricSettings } from "./MetricSettings";
import s from "./views.module.css";
import e from "./ExperimentView.module.css";

const STREAM_KINDS: StreamKind[] = ["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"];

const METHODS: { value: ExperimentKind; title: string; detail: string }[] = [
  { value: "activation_patching", title: "Activation patching", detail: "Copy an activation from the other prompt of each pair." },
  { value: "ablation", title: "Ablation", detail: "Replace an activation with a baseline you choose." },
  {
    value: "attribution_patching",
    title: "Attribution patching",
    detail: "Estimate patching at every site from one gradient, then verify the strongest.",
  },
  {
    value: "direct_logit_attribution",
    title: "Direct logit attribution",
    detail: "Split the logit difference into what each component writes directly. No intervention.",
  },
  {
    value: "path_patching",
    title: "Path patching",
    detail: "Patch a component's effect through chosen receivers only, holding the other heads.",
  },
  {
    value: "steering",
    title: "Steering",
    detail: "Add the mean clean–corrupt difference at several strengths, against a random control.",
  },
];

export function ExperimentView() {
  const form = useStore((st) => st.form);
  const setForm = useStore((st) => st.setForm);
  const model = useStore((st) => st.model);
  const dataset = useStore((st) => st.dataset);
  const datasetPath = useStore((st) => st.datasetPath);
  const job = useStore((st) => st.job);
  const startRun = useStore((st) => st.startRun);

  const labels = useMemo(() => {
    if (!dataset) return [];
    const sets = dataset.records.map((r) => new Set(Object.keys(r.positions ?? {})));
    if (sets.length === 0) return [];
    return [...sets[0]].filter((l) => sets.every((x) => x.has(l)));
  }, [dataset]);
  const lengths = dataset?.lengths ?? null;
  const uniformLength = lengths ? lengths.length === 1 : null;

  const sae = useStore((st) => st.sae);
  const facts = useMemo(() => datasetFacts(dataset, form.limit), [dataset, form.limit]);
  const built = buildSpec(form, { model, datasetPath, datasetSha: dataset?.sha256 ?? null, sae, facts });
  const differences = savedDifferences(form, model, datasetPath, dataset?.sha256 ?? null, modelName);
  const spec = "spec" in built ? built.spec : null;
  const busy = job?.status === "running";

  const suggested = spec
    ? suggestName(spec)
    : form.kind === "ablation" && !form.baseline
      ? `Ablation · ${scopeShort(form.scope).toLowerCase()}`
      : form.kind === "direct_logit_attribution" && !form.dlaPrompts
        ? `Direct attribution · ${scopeShort(form.scope).toLowerCase()}`
        : form.kind === "steering" && !form.steerApplyTo
          ? `Steering · ${scopeShort(form.scope).toLowerCase()}`
          : "";
  useEffect(() => {
    // The suggested name follows the rest of the form: not a change of its own to undo.
    if (!form.nameEdited && suggested && form.name !== suggested) setForm({ name: suggested }, { record: false });
  }, [suggested, form.nameEdited, form.name, setForm]);

  const n = dataset ? Math.min(dataset.n, form.limit ?? dataset.n) : 0;
  const arch = model.info ?? null;
  const noBos = model.info?.extra?.bos === false;
  const rows =
    spec && arch
      ? workload(spec, n, arch.n_layers, arch.n_heads, uniformLength && lengths ? lengths[0] : null, labels.length)
      : null;

  const run = () => {
    if (spec && !busy) void startRun(spec);
  };

  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") {
        ev.preventDefault();
        run();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  return (
    <div className={s.view}>
      <div className={s.head}>
        <div className={s.titleBlock}>
          <h2 className={s.title}>{form.draftId ? "Experiment" : "New experiment"}</h2>
          <p className={s.subtitle}>
            {form.draftId ? (
              <>
                Saved in experiments/{form.draftId}, not run yet. Running it writes the results into that folder.
              </>
            ) : (
              <>
                Every choice that changes a number is part of the spec. The same spec runs from the command line
                with <code>logogram run</code>.
              </>
            )}
          </p>
        </div>
        <div className={s.headActions}>
          <FormHistoryButtons />
          <Button
            variant="ghost"
            onClick={() => {
              useStore.setState({ specSource: "draft" });
              useStore.getState().setView("spec");
            }}
            disabled={!spec}
          >
            Preview spec
          </Button>
          <Button variant="primary" icon="play" onClick={run} disabled={!spec || busy} title={`Run (${MOD}+Enter)`}>
            Run
          </Button>
        </div>
      </div>

      <div className={e.form}>
        {form.predictions && <section className={e.section} aria-label="Included prediction diagnostic">
          <h3 className={e.sectionTitle}>Layer prediction diagnostic</h3>
          <p className={s.small}>Final-norm logit lens · {form.predictions.which} prompt {form.predictions.prompt_index} · {form.predictions.position.kind === "last" ? "last token" : `token index ${form.predictions.position.index}`} · top {form.predictions.top_k} tokens. Saved as predictions.json beside this run’s intervention results.</p>
          <div className={s.headActions}><Button size="small" onClick={() => useStore.getState().setView("predictions")}>Inspect predictions</Button><Button size="small" onClick={() => setForm({ predictions: null })}>Remove diagnostic</Button></div>
        </section>}
        <Field label="Name">
          <Input
            value={form.name}
            onChange={(ev) => setForm({ name: ev.target.value, nameEdited: true })}
            placeholder={suggested}
          />
        </Field>
        <Field label="Notes" help="Saved in the spec. Say what you expect, so the result can be read against it.">
          <TextArea
            rows={form.notes ? 3 : 1}
            value={form.notes}
            onChange={(ev) => setForm({ notes: ev.target.value })}
            placeholder="Hypothesis, context, links"
          />
        </Field>

        <section className={e.section}>
          <h3 className={e.sectionTitle}>Method</h3>
          <Choices
            label="Method"
            value={form.kind}
            onChange={(kind) => setForm({ kind, scope: scopeFor(kind, form.scope) })}
            columns={2}
            options={METHODS}
          />
          {form.kind === "steering" ? (
            <SteeringSettings form={form} onChange={setForm} />
          ) : form.kind === "direct_logit_attribution" ? (
            <div className={e.stack}>
              <Choices
                label="Prompts to split"
                value={form.dlaPrompts}
                onChange={(dlaPrompts) => setForm({ dlaPrompts })}
                columns={2}
                options={[
                  { value: "clean", title: "Clean prompts", detail: "Where the model should prefer the answer." },
                  { value: "corrupt", title: "Corrupt prompts", detail: "After the corruption has changed what it prefers." },
                ]}
              />
              {!form.dlaPrompts && <p className={e.required}>Required: the two prompts split differently, so there's no default.</p>}
              <p className={s.small}>
                Each component's term is its direct effect: what it writes into the residual stream at the last token, read
                through the final normalization with its scale held at its value in the run. It leaves out everything the
                component does through later components, which patching measures.
              </p>
            </div>
          ) : form.kind === "path_patching" ? (
            <PathSettings form={form} onChange={setForm} />
          ) : form.kind === "activation_patching" || form.kind === "attribution_patching" ? (
            <div className={e.stack}>
              <Choices
                label="Direction"
                value={form.direction}
                onChange={(direction) => setForm({ direction })}
                columns={2}
                options={[
                  {
                    value: "clean_to_corrupt",
                    title: "Clean → corrupt",
                    detail: "Run the corrupt prompt and patch in clean activations. Does this restore the behavior?",
                  },
                  {
                    value: "corrupt_to_clean",
                    title: "Corrupt → clean",
                    detail: "Run the clean prompt and patch in corrupt activations. Does this break it?",
                  },
                ]}
              />
              {form.kind === "attribution_patching" && (
                <p className={s.small}>
                  Estimates what patching each site would do, to first order: (source activation − receiver activation) · the
                  gradient of the metric at the receiver run. One gradient covers every site, so large sweeps take seconds. It
                  misses saturation and can miss or even invert an effect: verify the strongest sites by patching from the
                  results.
                </p>
              )}
            </div>
          ) : (
            <BaselineChooser baseline={form.baseline} onChange={(baseline) => setForm({ baseline })} />
          )}
        </section>

        <section className={e.section}>
          <h3 className={e.sectionTitle}>What to sweep</h3>
          <ScopeEditor
            method={form.kind}
            saeLoaded={sae.state === "ready" ? sae.info ?? null : null}
            scope={form.scope}
            onChange={(scope) => setForm({ scope })}
            labels={labels}
            uniformLength={uniformLength}
            lengths={lengths}
          />
        </section>

        <section className={e.section}>
          <h3 className={e.sectionTitle}>Metric</h3>
          <MetricSettings form={form} onChange={setForm} facts={facts} />
        </section>

        <section className={e.section}>
          <h3 className={e.sectionTitle}>Statistics and execution</h3>
          <div className={s.grid4}>
            <Field label="Bootstrap resamples">
              <Input type="number" min={100} max={100000} step={100} value={form.bootstrap} onChange={(ev) => setForm({ bootstrap: Number(ev.target.value) })} />
            </Field>
            <Field label="Confidence level">
              <Select value={String(form.ci)} onChange={(ev) => setForm({ ci: Number(ev.target.value) })}>
                <option value="0.9">90%</option>
                <option value="0.95">95%</option>
                <option value="0.99">99%</option>
              </Select>
            </Field>
            <Field label="Execution and bootstrap seed">
              <Input type="number" value={form.statSeed} onChange={(ev) => setForm({ statSeed: Number(ev.target.value) })} />
            </Field>
            <Field label="Batch size" help="Rows per forward pass. Recorded because it can change the last digits.">
              <Input type="number" min={1} max={4096} value={form.batchSize} onChange={(ev) => setForm({ batchSize: Number(ev.target.value) })} />
            </Field>
          </div>
        </section>

        <section className={e.section}>
          <h3 className={e.sectionTitle}>Data and model</h3>
          <dl className={s.kv}>
            <dt>Prompts</dt>
            <dd>
              {datasetPath ? (
                <>
                  {datasetPath} · {dataset ? plural(dataset.n, "prompt") : "…"}
                  {dataset && <span className="faint"> · sha256 {dataset.sha256.slice(0, 12)}</span>}
                </>
              ) : (
                "none chosen"
              )}
            </dd>
            <dt>Model</dt>
            <dd>
              {model.info ? (
                <>
                  {modelName(model.info.id)} <span className="faint">@ {shortRevision(model.info.revision)} · {model.info.dtype} · {model.ref?.device ?? model.info.device}{model.info.process_weights ? " · weights processed" : ""}</span>
                </>
              ) : (
                <>
                  {modelName(form.modelRef?.id ?? "openai-community/gpt2")}{" "}
                  <span className="faint">@ {shortRevision(form.modelRef?.revision)} · loads when you run</span>
                </>
              )}
            </dd>
          </dl>
          <div className={s.row}>
            <Checkbox checked={form.limit !== null} onChange={(v) => setForm({ limit: v ? Math.min(dataset?.n ?? 16, 16) : null })}>
              Use only the first
            </Checkbox>
            <Input
              type="number"
              min={1}
              style={{ width: 90 }}
              disabled={form.limit === null}
              value={form.limit ?? dataset?.n ?? 0}
              onChange={(ev) => setForm({ limit: Math.max(1, Number(ev.target.value)) })}
              aria-label="Number of prompts to use"
            />
            <span className={s.small}>prompts</span>
            <span style={{ flex: 1 }} />
            <Checkbox checked={form.prependBos} disabled={noBos && !form.prependBos} onChange={(prependBos) => setForm({ prependBos })}>
              {noBos ? "Prepend a beginning-of-sequence token (this model has none)" : "Prepend the beginning-of-sequence token"}
            </Checkbox>
          </div>
        </section>

        <div className={e.summary}>
          {differences.length > 0 && (
            <Callout title="This runs with something other than the saved spec">
              {differences.join(" ")} Running uses what is loaded and chosen now, and records it in the new spec.
            </Callout>
          )}
          {"error" in built ? (
            <Callout tone="error" title="Not ready to run">
              {built.error}
            </Callout>
          ) : (
            spec && (
              <p className={e.summaryText}>
                <strong>{experimentText(spec.experiment)}</strong> at {scopeText(spec.scope)}, over {plural(n, "prompt")},
                measured by the {metricWords(spec.metric).label}.
                {rows !== null && <> {workloadText(spec, rows)}</>}{" "}
                <span className="faint">
                  Press <Kbd>{MOD}</Kbd> <Kbd>Enter</Kbd> to run.
                </span>
              </p>
            )
          )}
          {busy && <p className={s.small}>Another job is running; this one can start when it finishes.</p>}
        </div>

      </div>
    </div>
  );
}

/** Undo and redo for the whole form, beside its title. Text fields keep their own undo. */
function FormHistoryButtons() {
  const canUndo = useStore((st) => st.formHistory.past.length > 0);
  const canRedo = useStore((st) => st.formHistory.future.length > 0);
  const undo = useStore((st) => st.undoForm);
  const redo = useStore((st) => st.redoForm);
  return (
    <div className={e.history} role="group" aria-label="Form history">
      <Button size="small" variant="ghost" icon="undo" disabled={!canUndo} onClick={undo} title={`Undo the last change to the form (${MOD}+Z)`} aria-keyshortcuts="Control+Z Meta+Z">
        Undo
      </Button>
      <Button size="small" variant="ghost" icon="redo" disabled={!canRedo} onClick={redo} title={`Redo (${MOD}+Shift+Z)`} aria-keyshortcuts="Control+Shift+Z Meta+Shift+Z Control+Y">
        Redo
      </Button>
    </div>
  );
}

const DIRECTIONS = [
  {
    value: "clean_to_corrupt" as const,
    title: "Clean → corrupt",
    detail: "Run the corrupt prompt and patch in clean activations. Does this restore the behavior?",
  },
  {
    value: "corrupt_to_clean" as const,
    title: "Corrupt → clean",
    detail: "Run the clean prompt and patch in corrupt activations. Does this break it?",
  },
];

function PathSettings({ form, onChange }: { form: FormState; onChange: (patch: Partial<FormState>) => void }) {
  const [query, setQuery] = useState("");
  const [part, setPart] = useState<"q" | "k" | "v">("q");
  const head = parseHeadQuery(query);
  const has = (r: PathReceiverSpec) => form.pathReceivers.some((x) => JSON.stringify(x) === JSON.stringify(r));
  const add = (r: PathReceiverSpec) => {
    if (!has(r)) onChange({ pathReceivers: [...form.pathReceivers, r] });
  };
  return (
    <div className={e.stack}>
      <Choices label="Direction" value={form.direction} onChange={(direction) => onChange({ direction })} columns={2} options={DIRECTIONS} />
      <Field label="Receivers" help="Where the paths end. Each sender of the sweep is patched through these inputs only.">
        <div className={e.receivers}>
          {form.pathReceivers.map((r) => (
            <span key={receiverLabel(r)} className={e.receiver}>
              {r.kind === "logits" ? "Logits, read directly" : `L${r.layer} H${r.head} ${{ q: "query", k: "key", v: "value" }[r.input]}`}
              <button type="button" aria-label={`Remove ${receiverLabel(r)}`} onClick={() => onChange({ pathReceivers: form.pathReceivers.filter((x) => x !== r) })}>
                ×
              </button>
            </span>
          ))}
          {form.pathReceivers.length === 0 && <span className={e.required}>Required: add at least one.</span>}
        </div>
      </Field>
      <div className={s.row}>
        <Input value={query} onChange={(ev) => setQuery(ev.target.value)} placeholder="L9 H9" aria-label="Receiver head, like L9 H9" style={{ width: 110 }} spellCheck={false} />
        <Select value={part} onChange={(ev) => setPart(ev.target.value as "q" | "k" | "v")} aria-label="Which input of the head">
          <option value="q">Query</option>
          <option value="k">Key</option>
          <option value="v">Value</option>
        </Select>
        <Button size="small" disabled={!head} onClick={() => head && (add({ kind: "head", layer: head.layer, head: head.head, input: part }), setQuery(""))}>
          Add head
        </Button>
        <Button size="small" disabled={has({ kind: "logits" })} onClick={() => add({ kind: "logits" })}>
          Add the logits
        </Button>
      </div>
      <Checkbox checked={form.pathFreezeMlps} onChange={(pathFreezeMlps) => onChange({ pathFreezeMlps })}>
        Hold the MLPs as well as the other heads, leaving only the direct path through the residual stream
      </Checkbox>
      <p className={s.small}>
        Each sender is patched while every other attention head keeps its own value, so its effect reaches the receivers only
        through the residual stream{form.pathFreezeMlps ? "" : " and the MLPs"}. What the receivers read in that run is then patched
        into an unchanged run. Senders in or after the last receiver's layer have no path and are left out.
      </p>
    </div>
  );
}

function SteeringSettings({ form, onChange }: { form: FormState; onChange: (patch: Partial<FormState>) => void }) {
  const strengths = parseStrengths(form.steerStrengths);
  return (
    <div className={e.stack}>
      <Choices
        label="Prompts to steer"
        value={form.steerApplyTo}
        onChange={(steerApplyTo) => onChange({ steerApplyTo })}
        columns={2}
        options={[
          {
            value: "corrupt",
            title: "Steer corrupt prompts toward clean",
            detail: "Adds the mean of (clean − corrupt). Does this direction bring the behavior back?",
          },
          {
            value: "clean",
            title: "Steer clean prompts toward corrupt",
            detail: "Adds the mean of (corrupt − clean). Does it take the behavior away?",
          },
        ]}
      />
      {!form.steerApplyTo && <p className={e.required}>Required: it decides the direction, so there's no default.</p>}
      <div className={s.grid3}>
        <Field
          label="Strengths"
          help={"error" in strengths ? strengths.error : "Multiples of the mean difference. 1 adds the whole difference; negative values push the other way."}
        >
          <Input value={form.steerStrengths} onChange={(ev) => onChange({ steerStrengths: ev.target.value })} spellCheck={false} aria-invalid={"error" in strengths} />
        </Field>
        <Field label="Pairs that compute the direction" help="The rest are held out, and only they are measured.">
          <Select value={String(form.steerTrain)} onChange={(ev) => onChange({ steerTrain: Number(ev.target.value) })}>
            <option value="0.25">25%</option>
            <option value="0.5">50%</option>
            <option value="0.75">75%</option>
          </Select>
        </Field>
        <Field label="Split seed">
          <Input type="number" value={form.steerSeed} onChange={(ev) => onChange({ steerSeed: Number(ev.target.value) })} />
        </Field>
      </div>
      <Checkbox checked={form.steerControl} onChange={(steerControl) => onChange({ steerControl })}>
        Run a random direction of the same length, at the same strengths, as a control
      </Checkbox>
    </div>
  );
}

function BaselineChooser({ baseline, onChange }: { baseline: BaselineSpec | null; onChange: (b: BaselineSpec) => void }) {
  return (
    <div className={e.stack}>
      <Choices
        label="Baseline"
        value={baseline?.kind ?? null}
        onChange={(kind) => {
          if (kind === "zero") onChange({ kind: "zero" });
          else if (kind === "mean") onChange({ kind: "mean", reference: "corrupt" });
          else onChange({ kind: "resample", pool: "corrupt", donors: 10, seed: 0 });
        }}
        columns={3}
        options={[
          { value: "zero", title: "Zero", detail: "Replace the activation with zeros." },
          { value: "mean", title: "Mean", detail: "Replace it with its mean over a stated reference set." },
          { value: "resample", title: "Resample", detail: "Replace it with its value in other prompts, averaged over donors." },
        ]}
      />
      {!baseline && <p className={e.required}>Required: the baseline changes the result, so there's no default.</p>}
      {baseline?.kind === "mean" && (
        <div className={s.grid3}>
          <Field label="Reference set" help="Per position when every position is replaced.">
            <Select value={baseline.reference} onChange={(ev) => onChange({ kind: "mean", reference: ev.target.value as "clean" | "corrupt" })}>
              <option value="corrupt">Corrupt prompts of this dataset</option>
              <option value="clean">Clean prompts of this dataset</option>
            </Select>
          </Field>
        </div>
      )}
      {baseline?.kind === "resample" && (
        <div className={s.grid3}>
          <Field label="Donor pool">
            <Select value={baseline.pool} onChange={(ev) => onChange({ ...baseline, pool: ev.target.value as "clean" | "corrupt" })}>
              <option value="corrupt">Corrupt prompts</option>
              <option value="clean">Clean prompts</option>
            </Select>
          </Field>
          <Field label="Donors per prompt" help="Never the prompt itself.">
            <Input type="number" min={1} max={1000} value={baseline.donors} onChange={(ev) => onChange({ ...baseline, donors: Math.max(1, Number(ev.target.value)) })} />
          </Field>
          <Field label="Donor seed">
            <Input type="number" value={baseline.seed} onChange={(ev) => onChange({ ...baseline, seed: Number(ev.target.value) })} />
          </Field>
        </div>
      )}
    </div>
  );
}

function PositionPicker({
  value,
  onChange,
  labels,
  allowAll = true,
}: {
  value: PositionSpec;
  onChange: (p: PositionSpec) => void;
  labels: string[];
  /** Steering adds at one token of each prompt, so "all positions" doesn't apply. */
  allowAll?: boolean;
}) {
  return (
    <div className={s.row}>
      <Segmented
        label="Position"
        value={value.kind}
        onChange={(kind) => {
          if (kind === "all") onChange({ kind: "all" });
          else if (kind === "last") onChange({ kind: "last" });
          else if (kind === "label") onChange({ kind: "label", label: labels[0] ?? "" });
          else onChange({ kind: "index", index: -1 });
        }}
        options={[
          { value: "all", label: "All positions", disabled: !allowAll, title: allowAll ? undefined : "Steering adds the direction at one token of each prompt" },
          { value: "last", label: "Last token" },
          { value: "label", label: "Named position", disabled: labels.length === 0, title: labels.length ? undefined : "This dataset has no named positions" },
          { value: "index", label: "Token index" },
        ]}
      />
      {value.kind === "label" && (
        <Select value={value.label} onChange={(ev) => onChange({ kind: "label", label: ev.target.value })} aria-label="Named position">
          {labels.map((l) => (
            <option key={l} value={l}>
              {l}
            </option>
          ))}
        </Select>
      )}
      {value.kind === "index" && (
        <Input
          type="number"
          style={{ width: 90 }}
          value={value.index}
          onChange={(ev) => onChange({ kind: "index", index: Number(ev.target.value) })}
          aria-label="Token index (negative counts from the end)"
        />
      )}
    </div>
  );
}

const LAST_TOKEN_NOTE = "Read at the last token, where the logit difference is measured.";

function ScopeEditor({
  method,
  saeLoaded,
  scope,
  onChange,
  labels,
  uniformLength,
  lengths,
}: {
  method: ExperimentKind;
  /** The loaded SAE: attribution patching can then estimate every one of its features. */
  saeLoaded: SAEInfo | null;
  scope: ScopeSpec;
  onChange: (s: ScopeSpec) => void;
  labels: string[];
  uniformLength: boolean | null;
  lengths: number[] | null;
}) {
  const kind = scope.kind;
  // Direct attribution splits what components write, at the last token only.
  const direct = method === "direct_logit_attribution";
  // Steering adds to one residual stream site per layer, at one token of each prompt.
  const steering = method === "steering";
  // Paths start at heads, attention outputs or MLP outputs.
  const paths = method === "path_patching";
  const verb = method === "ablation" ? "Ablate" : "Patch";
  const position = (p: PositionSpec) => (direct ? { kind: "last" } as PositionSpec : p);
  return (
    <div className={e.stack}>
      <Segmented
        label="Sweep"
        value={kind}
        onChange={(k) => {
          if (k === "features") onChange({ kind: "features", position: { kind: "last" }, top: 50, choose_on: null, seed: null });
          else if (k === "layer_components" && steering) onChange({ kind: "layer_components", components: ["resid_pre"], position: { kind: "last" } });
          else if (k === "heads") onChange({ kind: "heads", position: position({ kind: "all" }) });
          else if (k === "layer_position")
            onChange({ kind: "layer_position", site: "resid_pre", positions: uniformLength === false && labels.length ? "labels" : "each" });
          else if (k === "layer_components") onChange({ kind: "layer_components", components: ["attn_out", "mlp_out"], position: position({ kind: "all" }) });
        }}
        options={[
          { value: "heads", label: "Layer × head", disabled: steering, title: steering ? "Steering adds to the residual stream" : undefined },
          {
            value: "layer_position",
            label: "Layer × position",
            disabled: direct || steering || paths,
            title: direct
              ? "Direct effects are read at the last token only"
              : steering
                ? "Steering adds at one token"
                : paths
                  ? "Paths start at heads, attention outputs or MLP outputs"
                  : undefined,
          },
          { value: "layer_components", label: steering ? "One site in every layer" : "Attention and MLP per layer" },
          ...(method === "attribution_patching" && (saeLoaded || kind === "features")
            ? [{ value: "features" as const, label: "Every SAE feature" }]
            : []),
          ...(kind === "sites" ? [{ value: "sites" as const, label: scope.sites.length === 1 ? "Single site" : "Chosen sites" }] : []),
        ]}
      />
      {scope.kind === "features" && (
        <div className={s.grid2}>
          <Field label="Estimate each feature at">
            <PositionPicker value={scope.position} onChange={(position) => onChange({ ...scope, position })} labels={labels} />
          </Field>
          <Field
            label="Keep the strongest"
            help={saeLoaded ? `Of the ${saeLoaded.d_sae.toLocaleString("en-US")} features of the SAE on layer ${saeLoaded.layer}, by the size of their estimated effect.` : "By the size of their estimated effect."}
          >
            <Input type="number" min={1} max={500} value={scope.top} onChange={(ev) => onChange({ ...scope, top: Math.max(1, Math.min(500, Number(ev.target.value) || 1)) })} aria-label="Number of features to keep" />
          </Field>
        </div>
      )}
      {scope.kind === "heads" &&
        (direct ? (
          <Field label="Each head's output (z), through its share of the output projection">
            <p className={s.small}>{LAST_TOKEN_NOTE}</p>
          </Field>
        ) : (
          <Field label={`${verb} each head's output (z) at`}>
            <PositionPicker value={scope.position} onChange={(position) => onChange({ ...scope, position })} labels={labels} />
          </Field>
        ))}
      {scope.kind === "layer_components" && steering && (
        <div className={s.grid2}>
          <Field label="Residual stream site">
            <Select
              value={scope.components[0] ?? "resid_pre"}
              onChange={(ev) => onChange({ ...scope, components: [ev.target.value as StreamKind] })}
            >
              <option value="resid_pre">Before each layer</option>
              <option value="resid_mid">Between attention and MLP</option>
              <option value="resid_post">After each layer</option>
            </Select>
          </Field>
          <Field label="Add the direction at">
            <PositionPicker value={scope.position} onChange={(position) => onChange({ ...scope, position })} labels={labels} allowAll={false} />
          </Field>
        </div>
      )}
      {scope.kind === "layer_components" && !steering && (
        <>
          <Field label="Components">
            <div className={s.row}>
              {STREAM_KINDS.map((k) => (
                <Checkbox
                  key={k}
                  checked={scope.components.includes(k)}
                  disabled={(direct || paths) && k !== "attn_out" && k !== "mlp_out"}
                  onChange={(on) =>
                    onChange({
                      ...scope,
                      components: on ? STREAM_KINDS.filter((x) => x === k || scope.components.includes(x)) : scope.components.filter((x) => x !== k),
                    })
                  }
                >
                  {KIND_SHORT[k]}
                </Checkbox>
              ))}
            </div>
          </Field>
          <Field label="At">
            {direct ? (
              <p className={s.small}>{LAST_TOKEN_NOTE} Residual stream states aren't written by a component, so they have no direct effect.</p>
            ) : (
              <PositionPicker value={scope.position} onChange={(position) => onChange({ ...scope, position })} labels={labels} />
            )}
          </Field>
        </>
      )}
      {scope.kind === "layer_position" && (
        <div className={s.grid2}>
          <Field label="Site">
            <Select value={scope.site} onChange={(ev) => onChange({ ...scope, site: ev.target.value as StreamKind })}>
              <option value="resid_pre">Residual stream before each layer</option>
              <option value="resid_mid">Residual stream between attention and MLP</option>
              <option value="resid_post">Residual stream after each layer</option>
              <option value="attn_out">Attention output</option>
              <option value="mlp_out">MLP output</option>
            </Select>
          </Field>
          <Field
            label="Positions"
            help={
              uniformLength === false
                ? `Prompts have ${lengths?.[0]}–${lengths?.[lengths.length - 1]} tokens, so only named positions line up.`
                : undefined
            }
          >
            <Segmented
              label="Positions"
              value={scope.positions}
              onChange={(positions) => onChange({ ...scope, positions })}
              options={[
                { value: "each", label: "Every token", disabled: uniformLength === false },
                { value: "labels", label: "Named positions", disabled: labels.length === 0 },
              ]}
            />
          </Field>
        </div>
      )}
      {scope.kind === "sites" && (
        <div className={e.stack}>
          {direct && <p className={s.small}>{LAST_TOKEN_NOTE}</p>}
          {!direct && scope.sites.map((site, i) => (
            <Field key={i} label={siteText({ ...site, position: { kind: "all" } })}>
              <PositionPicker
                value={site.position}
                labels={labels}
                allowAll={!steering}
                onChange={(position) =>
                  onChange({ kind: "sites", sites: scope.sites.map((x, j) => (j === i ? { ...x, position } : x)) })
                }
              />
            </Field>
          ))}
          <p className={s.faint}>Chosen from the map. Pick another sweep above to go back to a full grid.</p>
        </div>
      )}
    </div>
  );
}
