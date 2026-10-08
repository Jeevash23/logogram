import { useEffect, useMemo } from "react";

import type { BaselineSpec, PositionSpec, ScopeSpec, Spec, StreamKind } from "../api/types";
import { Button, Callout, Checkbox, Choices, Field, Input, Kbd, Segmented, Select, TextArea } from "../components/ui";
import { MOD } from "../components/Header";
import { count, plural, shortRevision } from "../lib/format";
import { modelName } from "../lib/hooks";
import { KIND_SHORT } from "../lib/sites";
import { experimentText, scopeShort, scopeText, siteText, suggestName, workload } from "../lib/spec";
import { useStore, type FormState } from "../store/app";
import s from "./views.module.css";
import e from "./ExperimentView.module.css";

const STREAM_KINDS: StreamKind[] = ["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"];

/** Build a spec from the form, or explain what's missing. */
export function buildSpec(
  form: FormState,
  ctx: {
    model: ReturnType<typeof useStore.getState>["model"];
    datasetPath: string | null;
    datasetSha: string | null;
  },
): { spec: Spec } | { error: string } {
  if (!ctx.datasetPath) return { error: "Choose prompts first (Prompts view)." };
  let experiment: Spec["experiment"];
  if (form.kind === "activation_patching") {
    experiment = { kind: "activation_patching", direction: form.direction };
  } else {
    if (!form.baseline) return { error: "Choose a baseline for the ablation. Logogram never assumes one." };
    experiment = { kind: "ablation", baseline: form.baseline };
  }
  const scope = form.scope;
  const positionError = (p: PositionSpec) =>
    p.kind === "label" && !p.label ? "Choose which named position to use." : null;
  if (scope.kind === "heads" || scope.kind === "layer_components") {
    const err = positionError(scope.position);
    if (err) return { error: err };
  }
  if (scope.kind === "layer_components" && scope.components.length === 0) {
    return { error: "Choose at least one component." };
  }
  if (scope.kind === "sites" && scope.sites.some((x) => positionError(x.position))) {
    return { error: "Choose which named position to use." };
  }
  const info = ctx.model.info;
  const ref = ctx.model.ref;
  const model: Spec["model"] = info
    ? {
        id: info.id,
        revision: info.revision,
        dtype: info.dtype,
        device: ref?.device ?? info.device,
        process_weights: info.process_weights,
      }
    : (form.modelRef ?? {
        id: "openai-community/gpt2",
        revision: null,
        dtype: "float32",
        device: "auto",
        process_weights: true,
      });
  const spec: Spec = {
    logogram_spec: 1,
    name: form.name.trim() || suggestName({ experiment, scope }),
    notes: form.notes,
    model,
    dataset: { path: ctx.datasetPath, sha256: ctx.datasetSha, limit: form.limit },
    tokenization: { prepend_bos: form.prependBos },
    experiment,
    scope,
    metric: { kind: "logit_diff", normalization: form.normalization },
    statistics: { bootstrap: form.bootstrap, ci: form.ci, seed: form.statSeed },
    execution: { batch_size: form.batchSize },
    predictions: form.predictions,
  };
  return { spec };
}

/** How what will run differs from the saved spec the form was opened from. */
export function savedDifferences(
  form: FormState,
  model: ReturnType<typeof useStore.getState>["model"],
  datasetPath: string | null,
  datasetSha: string | null,
): string[] {
  const out: string[] = [];
  const ref = form.modelRef;
  const info = model.info;
  if (ref && info) {
    if (ref.id !== info.id) {
      out.push(`It was written for ${modelName(ref.id)}; the loaded model is ${modelName(info.id)}.`);
    } else {
      if (ref.revision && ref.revision !== info.revision) {
        out.push(`It was written for revision ${shortRevision(ref.revision)}; the loaded model is ${shortRevision(info.revision)}.`);
      }
      if (ref.dtype !== info.dtype) out.push(`It was written for ${ref.dtype}; the model is loaded in ${info.dtype}.`);
      if (ref.process_weights !== info.process_weights) out.push("Weight processing differs from the saved spec.");
      if (ref.device !== "auto" && ref.device !== info.device) out.push(`It was written for ${ref.device}; the model is loaded on ${info.device}.`);
    }
  }
  const saved = form.savedDataset;
  if (saved && datasetPath) {
    if (saved.path !== datasetPath) out.push(`It used ${saved.path}; the prompts chosen now are ${datasetPath}.`);
    else if (saved.sha256 && datasetSha && saved.sha256 !== datasetSha) out.push(`${saved.path} has changed since it was saved.`);
  }
  return out;
}

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

  const built = buildSpec(form, { model, datasetPath, datasetSha: dataset?.sha256 ?? null });
  const differences = savedDifferences(form, model, datasetPath, dataset?.sha256 ?? null);
  const spec = "spec" in built ? built.spec : null;
  const busy = job?.status === "running";

  const suggested = spec
    ? suggestName(spec)
    : form.kind === "ablation" && !form.baseline
      ? `Ablation · ${scopeShort(form.scope).toLowerCase()}`
      : "";
  useEffect(() => {
    if (!form.nameEdited && suggested && form.name !== suggested) setForm({ name: suggested });
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
          <Button
            variant="ghost"
            onClick={() => useStore.setState({ view: "spec", specSource: "draft" })}
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
          <h3 className={e.sectionTitle}>Intervention</h3>
          <Segmented
            label="Experiment type"
            value={form.kind}
            onChange={(kind) => setForm({ kind })}
            options={[
              { value: "activation_patching", label: "Activation patching" },
              { value: "ablation", label: "Ablation" },
            ]}
          />
          {form.kind === "activation_patching" ? (
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
          ) : (
            <BaselineChooser baseline={form.baseline} onChange={(baseline) => setForm({ baseline })} />
          )}
        </section>

        <section className={e.section}>
          <h3 className={e.sectionTitle}>What to sweep</h3>
          <ScopeEditor
            verb={form.kind === "ablation" ? "Ablate" : "Patch"}
            scope={form.scope}
            onChange={(scope) => setForm({ scope })}
            labels={labels}
            uniformLength={uniformLength}
            lengths={lengths}
          />
        </section>

        <section className={e.section}>
          <h3 className={e.sectionTitle}>Metric</h3>
          <p className={e.text}>
            Logit difference at the last token: logit(answer) − logit(distractor). Per-prompt values are always kept.
          </p>
          <Field
            label="Normalize the effect by"
            help={
              form.normalization === "dataset_gap"
                ? "Each prompt's change divided by the dataset's mean clean–corrupt gap. Stable, and its mean is the usual normalized metric."
                : "Each prompt's change divided by its own clean–corrupt gap. Exactly 0 to 1 per prompt, but unstable when a gap is small."
            }
          >
            <Segmented
              label="Normalization"
              value={form.normalization}
              onChange={(normalization) => setForm({ normalization })}
              options={[
                { value: "dataset_gap", label: "Dataset mean gap" },
                { value: "prompt_gap", label: "Each prompt's gap" },
              ]}
            />
          </Field>
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
                <strong>{experimentText(spec.experiment)}</strong> at {scopeText(spec.scope)}, over {plural(n, "prompt")}.
                {rows !== null && <> That is {count(rows)} patched forward passes.</>}{" "}
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
}: {
  value: PositionSpec;
  onChange: (p: PositionSpec) => void;
  labels: string[];
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
          { value: "all", label: "All positions" },
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

function ScopeEditor({
  verb,
  scope,
  onChange,
  labels,
  uniformLength,
  lengths,
}: {
  verb: string;
  scope: ScopeSpec;
  onChange: (s: ScopeSpec) => void;
  labels: string[];
  uniformLength: boolean | null;
  lengths: number[] | null;
}) {
  const kind = scope.kind;
  return (
    <div className={e.stack}>
      <Segmented
        label="Sweep"
        value={kind}
        onChange={(k) => {
          if (k === "heads") onChange({ kind: "heads", position: { kind: "all" } });
          else if (k === "layer_position")
            onChange({ kind: "layer_position", site: "resid_pre", positions: uniformLength === false && labels.length ? "labels" : "each" });
          else if (k === "layer_components") onChange({ kind: "layer_components", components: ["attn_out", "mlp_out"], position: { kind: "all" } });
        }}
        options={[
          { value: "heads", label: "Layer × head" },
          { value: "layer_position", label: "Layer × position" },
          { value: "layer_components", label: "Attention and MLP per layer" },
          ...(kind === "sites" ? [{ value: "sites" as const, label: scope.sites.length === 1 ? "Single site" : "Chosen sites" }] : []),
        ]}
      />
      {scope.kind === "heads" && (
        <Field label={`${verb} each head's output (z) at`}>
          <PositionPicker value={scope.position} onChange={(position) => onChange({ ...scope, position })} labels={labels} />
        </Field>
      )}
      {scope.kind === "layer_components" && (
        <>
          <Field label="Components">
            <div className={s.row}>
              {STREAM_KINDS.map((k) => (
                <Checkbox
                  key={k}
                  checked={scope.components.includes(k)}
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
            <PositionPicker value={scope.position} onChange={(position) => onChange({ ...scope, position })} labels={labels} />
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
          {scope.sites.map((site, i) => (
            <Field key={i} label={siteText({ ...site, position: { kind: "all" } })}>
              <PositionPicker
                value={site.position}
                labels={labels}
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
