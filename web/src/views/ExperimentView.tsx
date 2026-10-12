import { useEffect, useMemo, useState } from "react";

import type { BaselineSpec, ExperimentKind, HeadInputKind, ModelInfo, PathReceiverSpec, PositionSpec, SAEInfo, ScopeSpec, StreamKind } from "../api/types";
import { Button, Checkbox, Choices, Field, Input, Segmented, Select, TextArea } from "../components/ui";
import { MOD } from "../components/Header";
import { count, plural, shortRevision } from "../lib/format";
import { modelName } from "../lib/hooks";
import { EVERYTHING, SET_SITE_KINDS } from "../lib/circuits";
import { KIND_SHORT, parseHeadQuery, queryHeadsOf, queryHeadsText } from "../lib/sites";
import { buildSpec, datasetFacts, parseStrengths, savedDifferences, usesFeatures } from "../lib/buildSpec";
import { parseLayers, receiverLabel, saeWritesText, scopeFor, scopeShort, siteText, steerSiteOf, suggestName, workload } from "../lib/spec";
import { useStore, type FormState } from "../store/app";
import { ChatTemplateSettings } from "./ChatTemplateSettings";
import { MetricSettings } from "./MetricSettings";
import { ClusterChooser } from "./StatisticsSettings";
import { SiteSetsEditor } from "./SiteSetsEditor";
import { RunSummary } from "./RunSummary";
import s from "./views.module.css";
import e from "./ExperimentView.module.css";

const STREAM_KINDS: StreamKind[] = ["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"];

const METHODS: { value: ExperimentKind; title: string; detail: string }[] = [
  { value: "activation_patching", title: "Activation patching", detail: "Copy an activation from the other prompt of each pair." },
  { value: "ablation", title: "Ablation", detail: "Replace an activation with a baseline you choose." },
  {
    value: "attribution_patching",
    title: "Attribution patching",
    detail: "Estimate patching at every site from gradients, then verify the strongest.",
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
    detail: "Add the mean clean–corrupt difference or an SAE feature at several strengths, against a random control.",
  },
];

export function ExperimentView() {
  const form = useStore((st) => st.form);
  const setForm = useStore((st) => st.setForm);
  const model = useStore((st) => st.model);
  const dataset = useStore((st) => st.dataset);
  const datasetPath = useStore((st) => st.datasetPath);
  const startRun = useStore((st) => st.startRun);
  const queueRun = useStore((st) => st.queueRun);

  const labels = useMemo(() => {
    if (!dataset) return [];
    const sets = dataset.records.map((r) => new Set(Object.keys(r.positions ?? {})));
    if (sets.length === 0) return [];
    return [...sets[0]].filter((l) => sets.every((x) => x.has(l)));
  }, [dataset]);
  const lengths = dataset?.lengths ?? null;
  const uniformLength = lengths ? lengths.length === 1 : null;

  const sae = useStore((st) => st.sae);
  const saeInfo = sae.state === "ready" ? sae.info ?? null : null;
  // Steering along an SAE feature adds it where the SAE writes, at one site.
  const steersFeature = form.steerDirection === "sae_feature";
  const facts = useMemo(() => datasetFacts(dataset, form.limit), [dataset, form.limit]);
  const built = buildSpec(form, { model, datasetPath, datasetSha: dataset?.sha256 ?? null, sae, facts });
  const differences = savedDifferences(form, model, datasetPath, dataset?.sha256 ?? null, modelName);
  const spec = "spec" in built ? built.spec : null;

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
      ? workload(
          spec,
          n,
          arch.n_layers,
          arch.n_heads,
          uniformLength && lengths ? lengths[0] : null,
          labels.length,
          arch.extra?.n_key_value_heads ?? arch.n_heads,
          arch.d_mlp ?? 0,
        )
      : null;

  return (
    <div className={s.view}>
      <div className={s.head}>
        <div className={s.titleBlock}>
          <h1 className={s.title}>{form.draftId ? form.name || "A saved experiment" : "New experiment"}</h1>
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
        </div>
      </div>

      <div className={e.layout}>
      <div className={e.form}>
        {form.predictions && <section className={e.section} aria-label="Included prediction diagnostic">
          <h2 className={e.sectionTitle}>Layer prediction diagnostic</h2>
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
          <h2 className={e.sectionTitle}>Method</h2>
          <Choices
            label="Method"
            value={form.kind}
            onChange={(kind) => setForm({ kind, scope: scopeFor(kind, form.scope, kind === "steering" && steersFeature && saeInfo ? steerSiteOf(saeInfo) : null) })}
            columns={2}
            options={METHODS}
          />
          {form.kind === "steering" ? (
            <SteeringSettings form={form} onChange={setForm} sae={saeInfo} />
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
              {form.kind === "attribution_patching" && <AttributionSettings form={form} onChange={setForm} features={usesFeatures(form.scope)} />}
            </div>
          ) : (
            <BaselineChooser baseline={form.baseline} onChange={(baseline) => setForm({ baseline })} />
          )}
        </section>

        <section className={e.section}>
          <h2 className={e.sectionTitle}>What to sweep</h2>
          <ScopeEditor
            method={form.kind}
            model={arch}
            saeLoaded={saeInfo}
            steersFeature={form.kind === "steering" && steersFeature}
            scope={form.scope}
            onChange={(scope) => setForm({ scope })}
            labels={labels}
            uniformLength={uniformLength}
            lengths={lengths}
          />
        </section>

        <section className={e.section}>
          <h2 className={e.sectionTitle}>Metric</h2>
          <MetricSettings form={form} onChange={setForm} facts={facts} />
        </section>

        <section className={e.section}>
          <h2 className={e.sectionTitle}>Statistics and execution</h2>
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
          <ClusterChooser cluster={form.cluster} facts={facts} onChange={(cluster) => setForm({ cluster })} />
        </section>

        <section className={e.section}>
          <h2 className={e.sectionTitle}>Data and model</h2>
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
                  {modelName(model.info.id)}{" "}
                  <span className="faint">
                    {model.info.source === "local" ? `· local folder, files ${model.info.sha256?.slice(0, 12)}…` : `@ ${shortRevision(model.info.revision)}`} · {model.info.dtype} · {model.ref?.device ?? model.info.device}{model.info.process_weights ? " · weights processed" : ""}
                  </span>
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
            <Checkbox
              checked={form.prependBos && !form.chatTemplate}
              disabled={!!form.chatTemplate || (noBos && !form.prependBos)}
              onChange={(prependBos) => setForm({ prependBos })}
            >
              {form.chatTemplate
                ? "Prepend the beginning-of-sequence token (the chat template writes the beginning)"
                : noBos
                  ? "Prepend a beginning-of-sequence token (this model has none)"
                  : "Prepend the beginning-of-sequence token"}
            </Checkbox>
          </div>
          <ChatTemplateSettings
            value={form.chatTemplate}
            onChange={(chatTemplate) => setForm({ chatTemplate, ...(chatTemplate || !noBos ? {} : { prependBos: false }) })}
            firstPrompt={dataset?.records[0]?.clean ?? null}
          />
        </section>

      </div>
      <RunSummary
        spec={spec}
        error={"error" in built ? built.error : null}
        method={form.kind}
        scope={form.scope}
        bootstrap={form.bootstrap}
        batchSize={form.batchSize}
        n={n}
        rows={rows}
        differences={differences}
        onRun={startRun}
        onQueue={queueRun}
      />
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
      <Button size="small" variant="ghost" icon="undo" disabled={!canUndo} onClick={undo} tip={`Undo the last change to the form (${MOD}+Z)`} aria-keyshortcuts="Control+Z Meta+Z">
        Undo
      </Button>
      <Button size="small" variant="ghost" icon="redo" disabled={!canRedo} onClick={redo} tip={`Redo the change undone (${MOD}+Shift+Z)`} aria-keyshortcuts="Control+Shift+Z Meta+Shift+Z Control+Y">
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

/** How attribution patching estimates: one gradient, or integrated gradients over some steps. */
function AttributionSettings({ form, onChange, features }: { form: FormState; onChange: (patch: Partial<FormState>) => void; features: boolean }) {
  const steps = form.atpSteps;
  const valid = Number.isInteger(steps) && steps >= 2 && steps <= 64;
  return (
    <>
      <Choices
        label="Estimate with"
        value={form.atpMethod}
        onChange={(atpMethod) => onChange({ atpMethod })}
        columns={2}
        options={[
          {
            value: "gradient",
            title: "One gradient",
            detail: "The gradient at the receiver run: one forward and backward pass per batch. Misses saturation, where the metric stops changing in step with the activation.",
          },
          {
            value: "integrated_gradients",
            title: "Integrated gradients",
            detail: features
              ? "SAE features are estimated from one gradient only. Estimate the model's components to use integrated gradients."
              : "The gradient averaged over runs between the two prompts, which corrects much of that saturation. Each step costs another forward and backward pass.",
            disabled: features,
          },
        ]}
      />
      {features && form.atpMethod === "integrated_gradients" && (
        <p className={e.required}>Required: SAE features are estimated from one gradient. Choose it above.</p>
      )}
      {form.atpMethod === "integrated_gradients" && (
        <div className={s.grid3}>
          <Field
            label="Steps"
            help={valid ? `${steps} forward and backward passes per batch: ${steps} times the cost of one gradient.` : "A whole number from 2 to 64."}
          >
            <Input type="number" min={2} max={64} step={1} value={steps} onChange={(ev) => onChange({ atpSteps: Number(ev.target.value) })} aria-invalid={!valid} />
          </Field>
        </div>
      )}
      <p className={s.small}>
        Estimates what patching each site would do, to first order: (source activation − receiver activation) · the gradient
        of the metric{form.atpMethod === "integrated_gradients" ? ", averaged over runs whose inputs lie evenly between the receiver's and the source's" : " at the receiver run"}.
        One estimate covers every site, so large sweeps take seconds. Saturation (in attention, normalization and the softmax)
        can make one gradient miss or even invert an effect; integrated gradients correct much of it. Either way, verify the
        strongest sites by patching from the results.
      </p>
    </>
  );
}

/** A whole number typed into a field, or null while it is empty. */
const wholeOrNull = (text: string): number | null => (text.trim() === "" ? null : Number(text));

function SteeringSettings({ form, onChange, sae }: { form: FormState; onChange: (patch: Partial<FormState>) => void; sae: SAEInfo | null }) {
  const strengths = parseStrengths(form.steerStrengths);
  const feature = form.steerDirection === "sae_feature";
  const named = form.saeRef ? `${form.saeRef.repo}${form.saeRef.path ? `/${form.saeRef.path}` : ""}` : null;
  const chooseDirection = (steerDirection: FormState["steerDirection"]) => {
    if (steerDirection === "sae_feature") {
      // Added where the loaded SAE writes: the scope becomes that site.
      onChange({
        steerDirection,
        ...(sae ? { saeRef: { repo: sae.repo, path: sae.path, revision: sae.revision }, scope: scopeFor("steering", form.scope, steerSiteOf(sae)) } : {}),
      });
    } else {
      onChange({ steerDirection, scope: scopeFor("steering", form.scope) });
    }
  };
  return (
    <div className={e.stack}>
      <Choices
        label="Direction"
        value={form.steerDirection}
        onChange={chooseDirection}
        columns={2}
        options={[
          {
            value: "mean_difference",
            title: "The mean difference of the pairs",
            detail: "The mean of (other prompt − steered prompt) over the training pairs, added to the residual stream.",
          },
          {
            value: "sae_feature",
            title: "An SAE feature",
            detail: sae
              ? `Its decoder direction, added where the SAE writes: ${saeWritesText(sae)}.`
              : "Its decoder direction, added where its SAE writes. Needs a loaded SAE.",
          },
        ]}
      />
      {feature && (
        <div className={s.grid3}>
          <Field
            label="Feature"
            help={
              sae
                ? `One of the ${count(sae.d_sae)} features of ${sae.repo}${sae.path ? `/${sae.path}` : ""}.`
                : named
                  ? `A feature of ${named}, which isn't loaded. Load it in Explore → Features to check the feature and where it writes.`
                  : "Load an SAE in Explore → Features first."
            }
          >
            <Input
              type="number"
              min={0}
              max={sae ? sae.d_sae - 1 : undefined}
              step={1}
              value={form.steerFeature ?? ""}
              placeholder="Number"
              onChange={(ev) => onChange({ steerFeature: wholeOrNull(ev.target.value) })}
              aria-invalid={form.steerFeature === null}
            />
          </Field>
        </div>
      )}
      {feature && form.steerFeature === null && <p className={e.required}>Required: the feature to steer along. Features shows where each one fires.</p>}
      <Choices
        label="Prompts to steer"
        value={form.steerApplyTo}
        onChange={(steerApplyTo) => onChange({ steerApplyTo })}
        columns={2}
        options={
          feature
            ? [
                { value: "corrupt", title: "Steer corrupt prompts", detail: "1 means they moved as far as switching to the clean prompt. Does the feature bring the behavior back?" },
                { value: "clean", title: "Steer clean prompts", detail: "1 means they moved as far as switching to the corrupt prompt. Does it take the behavior away?" },
              ]
            : [
                { value: "corrupt", title: "Steer corrupt prompts toward clean", detail: "Adds the mean of (clean − corrupt). Does this direction bring the behavior back?" },
                { value: "clean", title: "Steer clean prompts toward corrupt", detail: "Adds the mean of (corrupt − clean). Does it take the behavior away?" },
              ]
        }
      />
      {!form.steerApplyTo && <p className={e.required}>Required: it decides what 1 means, so there's no default.</p>}
      <div className={s.grid3}>
        <Field
          label="Strengths"
          help={
            "error" in strengths
              ? strengths.error
              : feature
                ? "Multiples of the feature's typical activation: the mean of its positive activations at the steered token of the training pairs' prompts. 1 adds what the feature writes when that active; negative values push the other way."
                : "Multiples of the mean difference. 1 adds the whole difference; negative values push the other way."
          }
        >
          <Input value={form.steerStrengths} onChange={(ev) => onChange({ steerStrengths: ev.target.value })} spellCheck={false} aria-invalid={"error" in strengths} />
        </Field>
        <Field label={feature ? "Pairs that measure its typical activation" : "Pairs that compute the direction"} help="The rest are held out, and only they are measured.">
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
      <Checkbox checked={form.steerContinue} onChange={(steerContinue) => onChange({ steerContinue })}>
        Record what the steered model says
      </Checkbox>
      {form.steerContinue && (
        <>
          <div className={s.grid3}>
            <Field label="Held-out prompts to continue" help="The first ones in the split, 1 to 8.">
              <Input type="number" min={1} max={8} step={1} value={form.steerContinuePrompts} onChange={(ev) => onChange({ steerContinuePrompts: Number(ev.target.value) })} />
            </Field>
            <Field label="New tokens" help="For each prompt, 1 to 64.">
              <Input type="number" min={1} max={64} step={1} value={form.steerContinueTokens} onChange={(ev) => onChange({ steerContinueTokens: Number(ev.target.value) })} />
            </Field>
          </div>
          <p className={s.small}>
            Greedy: the most likely token each time, ties to the lowest token id. Each prompt is continued without steering, at
            each strength{form.steerControl ? " and along the random control" : ""}. Every forward pass adds the vector at the
            steered token of the prompt, as when it is measured; the tokens the model generates aren't steered themselves.
          </p>
        </>
      )}
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
          { value: "all", label: "All positions", disabled: !allowAll, reason: "Steering adds the direction at one token of each prompt, so it can't use all positions." },
          { value: "last", label: "Last token" },
          { value: "label", label: "Named position", disabled: labels.length === 0, reason: "This dataset has no named positions." },
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

type FeaturesScope = Extract<ScopeSpec, { kind: "features" }>;

/** Sets of sites to start from: the chosen sites as one set, or one set that replaces every head
 * (what a circuit is read against) to add sets beside. */
function firstSets(scope: ScopeSpec): ScopeSpec {
  if (scope.kind === "sites") {
    const sites = scope.sites.filter((x) => SET_SITE_KINDS.includes(x.kind));
    if (sites.length) return { kind: "site_sets", universe: null, sets: [{ label: sites.length === 1 ? siteText(sites[0]) : "Chosen sites", sites, complement: false }] };
  }
  return { kind: "site_sets", universe: ["head"], sets: [{ label: EVERYTHING, sites: [], complement: true }] };
}

/** Which prompts choose the strongest features, and which report them: the same prompts, or a
 * seeded split, so the reported effects aren't the ones that happened to look strongest. */
function HeldOutChoice({ scope, onChange }: { scope: FeaturesScope; onChange: (s: ScopeSpec) => void }) {
  const shares = [0.25, 0.5, 0.75];
  const share = scope.choose_on;
  return (
    <div className={e.stack}>
      <Choices
        label="Prompts that choose the features"
        value={share === null ? "every" : "split"}
        onChange={(v) => onChange(v === "every" ? { ...scope, choose_on: null, seed: null } : { ...scope, choose_on: 0.5, seed: scope.seed ?? 0 })}
        columns={2}
        options={[
          { value: "every", title: "Every prompt chooses and reports", detail: "The strongest features are reported on the same prompts that chose them." },
          { value: "split", title: "Some choose, the rest report", detail: "A seeded share of the prompts chooses the features; only the others report their effects." },
        ]}
      />
      {share !== null && (
        <div className={s.grid3}>
          <Field label="Share that chooses">
            <Select value={String(share)} onChange={(ev) => onChange({ ...scope, choose_on: Number(ev.target.value) })}>
              {(shares.includes(share) ? shares : [...shares, share].sort((a, b) => a - b)).map((v) => (
                <option key={v} value={String(v)}>
                  {Math.round(v * 100)}% of the prompts
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Split seed">
            <Input type="number" min={0} step={1} value={scope.seed ?? 0} onChange={(ev) => onChange({ ...scope, seed: Number(ev.target.value) })} />
          </Field>
        </div>
      )}
      <p className={s.small}>
        Features that look strongest on some prompts are partly lucky there, so reporting them on the same prompts overstates
        their effects (the winner's curse). Reporting them on prompts that didn't choose them doesn't.
      </p>
    </div>
  );
}

/** Why a head input or a neuron can't be a site of the loaded model, as it was measured when it
 * loaded; null when it can (or no model is loaded to say). */
function missingNote(model: ModelInfo | null, kind: HeadInputKind | "neuron"): string | null {
  if (!model || model.site_kinds.includes(kind)) return null;
  return model.extra?.site_notes?.[kind] ?? `${model.id} has no ${kind} sites.`;
}

/** What a key or value site is in the loaded model: with grouped-query attention, a key/value head
 * that every query head of its group reads. */
function groupText(model: ModelInfo | null): string {
  if (!model) {
    return "With grouped-query attention (Llama, Qwen, Gemma), a model has fewer key and value heads than query heads: a key or value site is a key/value head, and every query head of its group reads it.";
  }
  const kv = model.extra?.n_key_value_heads ?? model.n_heads;
  if (kv === model.n_heads) return `Each of the ${count(model.n_heads)} heads of a layer has its own query, key and value.`;
  const group = model.n_heads / kv;
  return `This model has ${count(kv)} key/value heads for its ${count(model.n_heads)} query heads: a key or value site is a key/value head, and every query head of its group reads it (${count(group)} each; key/value head 0 is read by ${queryHeadsText(queryHeadsOf(0, model.n_heads, kv))}). Patching one changes all of them.`;
}

/** The layers whose neurons a sweep covers, typed as a list ("3, 4" or "0-5"). The text is kept as
 * typed; the scope changes once it reads as layers. */
function LayersInput({
  value,
  onChange,
  nLayers,
}: {
  value: number[];
  onChange: (layers: number[]) => void;
  nLayers: number | null;
}) {
  const [text, setText] = useState(value.join(", "));
  const parsed = parseLayers(text);
  // Undo, or another change of the scope, shows its layers.
  useEffect(() => {
    if (JSON.stringify(parseLayers(text) ?? []) !== JSON.stringify(value)) setText(value.join(", "));
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only the scope's layers move the text
  }, [value]);
  const invalid = text.trim() !== "" && parsed === null;
  return (
    <Input
      value={text}
      placeholder={nLayers ? `such as ${Math.max(0, nLayers - 2)}, ${nLayers - 1}` : "such as 3, 4"}
      aria-label="Layers whose neurons to sweep"
      aria-invalid={invalid}
      spellCheck={false}
      onChange={(ev) => {
        setText(ev.target.value);
        const layers = parseLayers(ev.target.value);
        if (layers) onChange([...new Set(layers)]);
        else if (ev.target.value.trim() === "") onChange([]);
      }}
    />
  );
}

function ScopeEditor({
  method,
  model,
  saeLoaded,
  steersFeature,
  scope,
  onChange,
  labels,
  uniformLength,
  lengths,
}: {
  method: ExperimentKind;
  /** The loaded model: its key/value heads and neurons, and the sites it offers. */
  model: ModelInfo | null;
  /** The loaded SAE: attribution patching can then estimate every one of its features. */
  saeLoaded: SAEInfo | null;
  /** Steering along an SAE feature: added where its SAE writes, at one site. */
  steersFeature: boolean;
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
  // Sets of sites are intervened on together, which patching and ablation do.
  const setsAllowed = method === "activation_patching" || method === "ablation";
  // Why some scopes can't be chosen with this method, said once under the choices.
  const unavailable = steersFeature
    ? "Steering along an SAE feature adds it where the SAE writes, at one token: one site of one layer."
    : steering
    ? "Steering adds a direction to the residual stream at one token, so it sweeps no heads, positions, queries, keys, values, neurons or sets of sites."
    : direct
      ? "Direct effects are read at the last token only, and sets of sites need activation patching or ablation."
      : paths
        ? "Paths start at heads' outputs, attention outputs or MLP outputs, not at queries, keys, values or neurons, and sets of sites need activation patching or ablation."
        : "Sets of sites need activation patching or ablation.";
  // Queries, keys and values (the loaded model may offer only some), and neurons.
  const inputsMissing = missingNote(model, "q") && missingNote(model, "k") && missingNote(model, "v");
  const neuronsMissing = missingNote(model, "neuron");
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
          else if (k === "site_sets") onChange(firstSets(scope));
          else if (k === "head_inputs") {
            const input: HeadInputKind = direct || missingNote(model, "q") ? "v" : "q";
            onChange({ kind: "head_inputs", input, position: { kind: "all" } });
          } else if (k === "neurons") onChange({ kind: "neurons", layers: [], position: position({ kind: "all" }) });
        }}
        options={[
          { value: "heads", label: "Layer × head", disabled: steering, reason: unavailable },
          { value: "layer_position", label: "Layer × position", disabled: direct || steering || paths, reason: unavailable },
          { value: "layer_components", label: steering ? "One site in every layer" : "Attention and MLP per layer", disabled: steersFeature, reason: unavailable },
          {
            value: "head_inputs" as const,
            label: "Queries, keys, values",
            disabled: steering || paths || !!inputsMissing,
            reason: inputsMissing || unavailable,
          },
          { value: "neurons" as const, label: "Neurons", disabled: steering || paths || !!neuronsMissing, reason: neuronsMissing || unavailable },
          ...(method === "attribution_patching" && (saeLoaded || kind === "features")
            ? [{ value: "features" as const, label: "Every SAE feature" }]
            : []),
          ...(kind === "sites" ? [{ value: "sites" as const, label: scope.sites.length === 1 ? "Single site" : "Chosen sites" }] : []),
          {
            value: "site_sets" as const,
            label: "Sets of sites",
            disabled: !setsAllowed,
            reason: unavailable,
          },
        ]}
      />
      {scope.kind === "site_sets" && (
        <>
          <p className={s.small}>
            Each set's sites are {method === "ablation" ? "ablated" : "patched"} together in one forward pass, and each set is one
            result. A set can replace its sites, or keep them and replace the rest of the model: keeping a circuit alone shows how
            much of the behavior it carries.
          </p>
          <SiteSetsEditor scope={scope} onChange={onChange} labels={labels} />
        </>
      )}
      {scope.kind === "features" && (
        <div className={e.stack}>
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
          <HeldOutChoice scope={scope} onChange={onChange} />
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
      {scope.kind === "head_inputs" && (
        <div className={e.stack}>
          <Field label="Each head's">
            <Segmented
              label="Each head's query, key or value"
              value={scope.input}
              onChange={(input) => onChange({ ...scope, input })}
              options={(["q", "k", "v"] as const).map((k) => {
                const missing = missingNote(model, k);
                const noDirect = direct && k !== "v";
                return {
                  value: k,
                  label: { q: "Query", k: "Key", v: "Value" }[k],
                  disabled: noDirect || !!missing,
                  reason: noDirect
                    ? "Queries and keys have no direct effect: they act on the logits only through the softmax of their head's attention. A value's direct effect is what it adds through the heads that read it."
                    : missing ?? undefined,
                };
              })}
            />
          </Field>
          <p className={s.small}>{scope.input === "q" ? "A query site is one head's query, as its projection computes it, before any rotary embedding." : groupText(model)}</p>
          <Field label={`${direct ? "Read" : verb} each ${scope.input === "q" ? "query" : scope.input === "k" ? "key" : "value"} at`}>
            <PositionPicker value={scope.position} onChange={(position) => onChange({ ...scope, position })} labels={labels} />
          </Field>
          {direct && (
            <p className={s.small}>
              A value at any position reaches the last token through the attention of the heads that read it, with their
              attention patterns as they are in the run (the patterns don't depend on the values). All positions together are
              the direct effect of the heads that read the value.
            </p>
          )}
        </div>
      )}
      {scope.kind === "neurons" && (
        <div className={s.grid2}>
          <Field
            label="Layers"
            help={
              model?.d_mlp
                ? `Every neuron of each layer: ${count(model.d_mlp)} per layer, layers 0 to ${model.n_layers - 1}.`
                : "Every neuron of each layer listed, such as 3, 4 or 0-5."
            }
          >
            <LayersInput value={scope.layers} onChange={(layers) => onChange({ ...scope, layers })} nLayers={model?.n_layers ?? null} />
          </Field>
          <Field label={direct ? "Read each neuron at" : `${verb} each neuron at`}>
            {direct ? (
              <p className={s.small}>{LAST_TOKEN_NOTE} A neuron writes its activation times its row of the MLP's output projection.</p>
            ) : (
              <PositionPicker value={scope.position} onChange={(position) => onChange({ ...scope, position })} labels={labels} />
            )}
          </Field>
          <p className={`${s.small} ${e.wide}`}>
            A neuron's activation is the MLP's hidden activation after its nonlinearity (for a gated MLP, the gate's activation times
            the up projection): the input to the MLP's output projection.
          </p>
        </div>
      )}
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
          <p className={s.faint}>
            {steersFeature
              ? saeLoaded
                ? `Where the SAE writes: ${saeWritesText(saeLoaded)}.`
                : "Where the feature's SAE writes. Load it in Explore → Features to check the site."
              : "Chosen from the map. Pick another sweep above to go back to a full grid."}
          </p>
        </div>
      )}
    </div>
  );
}
