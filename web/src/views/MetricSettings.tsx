import { useState } from "react";

import type { MetricKind, ReadoutUnit, StreamKind } from "../api/types";
import { Button, Callout, Choices, Field, Input, Segmented, Select } from "../components/ui";
import { exampleText, READOUT_METHODS, type DatasetFacts } from "../lib/buildSpec";
import { plural } from "../lib/format";
import { METRIC_KINDS, METRICS } from "../lib/metrics";
import { useStore, type FormState } from "../store/app";
import { DirectionPicker } from "./DirectionPicker";
import s from "./views.module.css";
import e from "./ExperimentView.module.css";

const TITLES: Record<MetricKind, string> = {
  logit_diff: "Logit difference",
  logprob_diff: "Log-probability difference",
  logprob: "Answer log-probability",
  prob: "Answer probability",
  prob_diff: "Probability difference",
  kl: "KL divergence",
  readout: "A value inside the model",
};

/** The metric every forward pass is measured by, and how effects are normalized. */
export function MetricSettings({
  form,
  onChange,
  facts,
  labels = [],
  width,
}: {
  form: FormState;
  onChange: (patch: Partial<FormState>) => void;
  /** What is known about the prompts: whether their answers need a metric that reads continuations. */
  facts: DatasetFacts | null;
  /** The prompts' named positions, where a readout can read. */
  labels?: string[];
  /** The loaded model's residual stream width, for the directions a readout can read. */
  width?: number;
}) {
  // Direct logit attribution splits the logit difference, the one metric that is a sum of what
  // each component writes.
  const direct = form.kind === "direct_logit_attribution";
  // A readout is read by the methods that run the model forward with one site changed.
  const readable = (READOUT_METHODS as readonly string[]).includes(form.kind);
  const several = facts?.continuations ?? 0;
  const example = facts?.example ? ` (such as ${exampleText(facts.example)})` : "";
  return (
    <div className={e.stack}>
      <Choices
        label="Metric"
        value={form.metric}
        onChange={(metric) => onChange({ metric })}
        columns={2}
        options={METRIC_KINDS.map((kind) => ({
          value: kind,
          title: TITLES[kind],
          detail: METRICS[kind].detail,
          disabled: (direct && kind !== "logit_diff") || (kind === "readout" && !readable),
        }))}
      />
      {form.metric === "readout" && !readable && (
        <p className={e.required}>A readout is read by activation patching, ablation and subspace patching. Choose one of those, or another metric.</p>
      )}
      {form.metric === "readout" && readable && <ReadoutSettings form={form} onChange={onChange} labels={labels} width={width} />}
      {direct && form.metric !== "logit_diff" && (
        <p className={e.required}>Required: direct logit attribution splits the logit difference. Choose it above.</p>
      )}
      {direct && form.metric === "logit_diff" && (
        <p className={s.small}>Direct logit attribution splits the logit difference, the one metric that is a sum of what each component writes.</p>
      )}
      {!direct && form.metric === "kl" && (
        <>
          <Choices
            label="KL divergence measured from"
            value={form.klTarget}
            onChange={(klTarget) => onChange({ klTarget })}
            columns={2}
            options={[
              { value: "clean", title: "The clean prompt's prediction", detail: "KL(P_clean || P): how far each run's next-token distribution is from the clean prompt's." },
              { value: "corrupt", title: "The corrupt prompt's prediction", detail: "KL(P_corrupt || P): how far it is from the corrupt prompt's." },
            ]}
          />
          {!form.klTarget && <p className={e.required}>Required: the target changes every value, so there's no default.</p>}
        </>
      )}
      {!direct && several > 0 && form.metric === "logit_diff" && (
        <Callout title={`${plural(several, "prompt")} ${several === 1 ? "has an answer or distractor" : "have answers or distractors"} of several tokens`}>
          The logit difference reads one token at the last position, so it can't score them{example}. The log-probability
          difference reads them one token at a time, and equals the logit difference for single tokens.
          <div className={e.calloutAction}>
            <Button size="small" onClick={() => onChange({ metric: "logprob_diff" })}>
              Use the log-probability difference
            </Button>
          </div>
        </Callout>
      )}
      {direct ? (
        <Field
          label="Express each direct effect as a share of"
          help={
            form.normalization === "dataset_gap"
              ? "Each prompt's term divided by the mean logit difference of the chosen prompts. With the embeddings and biases, the shares add up to one."
              : "Each prompt's term divided by its own logit difference. Unstable when a prompt's logit difference is near zero."
          }
        >
          <Segmented
            label="Normalization"
            value={form.normalization}
            onChange={(normalization) => onChange({ normalization })}
            options={[
              { value: "dataset_gap", label: "Mean logit difference" },
              { value: "prompt_gap", label: "Each prompt's" },
            ]}
          />
        </Field>
      ) : (
        <Field
          label="Normalize the effect by"
          help={
            form.normalization === "dataset_gap"
              ? "Each prompt's change divided by the dataset's mean clean–corrupt gap in the metric. Stable, and its mean is the usual normalized metric."
              : "Each prompt's change divided by its own clean–corrupt gap. Exactly 0 to 1 per prompt, but unstable when a gap is small."
          }
        >
          <Segmented
            label="Normalization"
            value={form.normalization}
            onChange={(normalization) => onChange({ normalization })}
            options={[
              { value: "dataset_gap", label: "Dataset mean gap" },
              { value: "prompt_gap", label: "Each prompt's gap" },
            ]}
          />
        </Field>
      )}
      <p className={s.small}>Per-prompt values are always kept.</p>
    </div>
  );
}

const STREAM_SITES: { value: StreamKind; label: string }[] = [
  { value: "resid_pre", label: "Residual stream before the layer" },
  { value: "resid_mid", label: "Residual stream between attention and MLP" },
  { value: "resid_post", label: "Residual stream after the layer" },
  { value: "attn_out", label: "Attention output" },
  { value: "mlp_out", label: "MLP output" },
];

/** What a readout reads (a neuron, a saved direction at a site, a feature of the loaded SAE) and
 * where. Nothing is read until it is chosen. */
function ReadoutSettings({
  form,
  onChange,
  labels,
  width,
}: {
  form: FormState;
  onChange: (patch: Partial<FormState>) => void;
  labels: string[];
  width?: number;
}) {
  const sae = useStore((st) => (st.sae.state === "ready" ? st.sae.info : null));
  const layers = useStore((st) => st.model.info?.n_layers);
  const [kind, setKind] = useState<ReadoutUnit["kind"]>(form.readout?.kind ?? "neuron");
  const unit = form.readout?.kind === kind ? form.readout : null;
  const choose = (next: ReadoutUnit["kind"]) => {
    setKind(next);
    if (next === "neuron") onChange({ readout: { kind: "neuron", layer: 0, neuron: 0 } });
    else if (next === "sae_feature") onChange({ readout: sae ? { kind: "sae_feature", feature: 0 } : null });
    else onChange({ readout: null });
  };
  const number = (text: string) => Math.max(0, Math.floor(Number(text) || 0));
  const position = form.readoutPosition;
  return (
    <div className={e.stack}>
      <Segmented
        label="Read"
        value={kind}
        onChange={choose}
        options={[
          { value: "neuron", label: "A neuron" },
          { value: "direction", label: "A saved direction" },
          { value: "sae_feature", label: "An SAE feature", disabled: !sae, reason: "Load an SAE in Explore → Features first." },
        ]}
      />
      {kind === "neuron" && (
        <div className={s.grid3}>
          <Field label="Layer" help={layers ? `0 to ${layers - 1}.` : undefined}>
            <Input type="number" min={0} value={unit?.kind === "neuron" ? unit.layer : 0} onChange={(ev) => onChange({ readout: { kind: "neuron", layer: number(ev.target.value), neuron: unit?.kind === "neuron" ? unit.neuron : 0 } })} />
          </Field>
          <Field label="Neuron" help="Its activation after the nonlinearity.">
            <Input type="number" min={0} value={unit?.kind === "neuron" ? unit.neuron : 0} onChange={(ev) => onChange({ readout: { kind: "neuron", layer: unit?.kind === "neuron" ? unit.layer : 0, neuron: number(ev.target.value) } })} />
          </Field>
        </div>
      )}
      {kind === "direction" && (
        <div className={s.grid3}>
          <DirectionPicker
            label="Direction"
            value={unit?.kind === "direction" ? unit.direction : null}
            oneDimension
            width={width}
            onChange={(direction) =>
              onChange({ readout: direction ? { kind: "direction", direction, site: unit?.kind === "direction" ? unit.site : "resid_post", layer: unit?.kind === "direction" ? unit.layer : 0 } : null })
            }
          />
          <Field label="Read at">
            <Select
              value={unit?.kind === "direction" ? unit.site : "resid_post"}
              disabled={unit?.kind !== "direction"}
              onChange={(ev) => unit?.kind === "direction" && onChange({ readout: { ...unit, site: ev.target.value as StreamKind } })}
            >
              {STREAM_SITES.map((x) => (
                <option key={x.value} value={x.value}>
                  {x.label}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Layer">
            <Input type="number" min={0} disabled={unit?.kind !== "direction"} value={unit?.kind === "direction" ? unit.layer : 0} onChange={(ev) => unit?.kind === "direction" && onChange({ readout: { ...unit, layer: number(ev.target.value) } })} />
          </Field>
        </div>
      )}
      {kind === "sae_feature" && sae && (
        <div className={s.grid3}>
          <Field label="Feature" help={`One of the ${sae.d_sae.toLocaleString("en-US")} features of the loaded SAE, as it encodes what it reads.`}>
            <Input type="number" min={0} max={sae.d_sae - 1} value={unit?.kind === "sae_feature" ? unit.feature : 0} onChange={(ev) => onChange({ readout: { kind: "sae_feature", feature: Math.min(sae.d_sae - 1, number(ev.target.value)) } })} />
          </Field>
        </div>
      )}
      <Field label="At" help="The token of each prompt where it is read.">
        <div className={s.row}>
          <Segmented
            label="Readout position"
            value={position.kind}
            onChange={(k) =>
              onChange({ readoutPosition: k === "last" ? { kind: "last" } : k === "label" ? { kind: "label", label: labels[0] ?? "" } : { kind: "index", index: -1 } })
            }
            options={[
              { value: "last", label: "Last token" },
              { value: "label", label: "Named position", disabled: labels.length === 0, reason: "These prompts name no positions." },
              { value: "index", label: "Token index" },
            ]}
          />
          {position.kind === "label" && (
            <Select value={position.label} onChange={(ev) => onChange({ readoutPosition: { kind: "label", label: ev.target.value } })} aria-label="Named position">
              {labels.map((l) => (
                <option key={l} value={l}>
                  {l}
                </option>
              ))}
            </Select>
          )}
          {position.kind === "index" && (
            <Input type="number" value={position.index} onChange={(ev) => onChange({ readoutPosition: { kind: "index", index: Math.trunc(Number(ev.target.value)) } })} aria-label="Token index" />
          )}
        </div>
      </Field>
      {!form.readout && <p className={e.required}>Required: choose what the readout reads.</p>}
      <p className={s.small}>
        Effects are normalized as for any metric: 1 means the intervention moves the value as far as switching prompts does. Sites
        computed after the value can't change it.
      </p>
    </div>
  );
}
