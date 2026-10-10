import type { MetricKind } from "../api/types";
import { Button, Callout, Choices, Field, Segmented } from "../components/ui";
import type { DatasetFacts } from "../lib/buildSpec";
import { plural, visibleToken } from "../lib/format";
import { METRIC_KINDS, METRICS } from "../lib/metrics";
import type { FormState } from "../store/app";
import s from "./views.module.css";
import e from "./ExperimentView.module.css";

const TITLES: Record<MetricKind, string> = {
  logit_diff: "Logit difference",
  logprob_diff: "Log-probability difference",
  logprob: "Answer log-probability",
  prob: "Answer probability",
  prob_diff: "Probability difference",
  kl: "KL divergence",
};

/** The metric every forward pass is measured by, and how effects are normalized. */
export function MetricSettings({
  form,
  onChange,
  facts,
}: {
  form: FormState;
  onChange: (patch: Partial<FormState>) => void;
  /** What is known about the prompts: whether their answers need a metric that reads continuations. */
  facts: DatasetFacts | null;
}) {
  // Direct logit attribution splits the logit difference, the one metric that is a sum of what
  // each component writes.
  const direct = form.kind === "direct_logit_attribution";
  const several = facts?.continuations ?? 0;
  const example = facts?.example ? ` such as “${visibleToken(facts.example)}”` : "";
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
          detail: `${METRICS[kind].formula}. ${METRICS[kind].detail}`,
          disabled: direct && kind !== "logit_diff",
        }))}
      />
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
              { value: "clean", title: "The clean prompt's prediction", detail: "KL(P_clean ‖ P): how far each run's next-token distribution is from the clean prompt's." },
              { value: "corrupt", title: "The corrupt prompt's prediction", detail: "KL(P_corrupt ‖ P): how far it is from the corrupt prompt's." },
            ]}
          />
          {!form.klTarget && <p className={e.required}>Required: the target changes every value, so there's no default.</p>}
        </>
      )}
      {!direct && several > 0 && form.metric === "logit_diff" && (
        <Callout title={`${plural(several, "prompt")} ${several === 1 ? "has an answer" : "have answers"} of several tokens`}>
          The logit difference reads one token at the last position, so it can't score answers{example}. The log-probability
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
