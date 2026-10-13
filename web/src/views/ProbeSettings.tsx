import { Checkbox, Choices, Field, Input, Select } from "../components/ui";
import type { FormState } from "../store/app";
import s from "./views.module.css";
import e from "./ExperimentView.module.css";

type Change = (patch: Partial<FormState>) => void;

/** A share of the pairs, as a percent field: 1 to 99. */
function ShareInput({ value, onChange, label }: { value: number; onChange: (v: number) => void; label: string }) {
  return (
    <Input
      type="number"
      min={1}
      max={99}
      step={5}
      value={Math.round(value * 100)}
      onChange={(ev) => onChange(Math.min(99, Math.max(1, Number(ev.target.value))) / 100)}
      aria-label={label}
    />
  );
}

/** A linear probe: what it reads (no default), the split, the fit and its control. */
export function ProbeSettings({ form, onChange, fields }: { form: FormState; onChange: Change; fields: string[] }) {
  return (
    <div className={e.stack}>
      <Choices
        label="The probe reads"
        value={form.probeLabels}
        onChange={(probeLabels) => onChange({ probeLabels, probeField: probeLabels === "meta" ? (form.probeField ?? fields[0] ?? null) : form.probeField })}
        columns={2}
        options={[
          {
            value: "side",
            title: "Which prompt it is",
            detail: "Each pair gives two examples, its clean and its corrupt prompt. Where can the difference between them be read?",
          },
          {
            value: "meta",
            title: "A field of the prompts' meta",
            detail: fields.length ? `Such as ${fields.slice(0, 3).join(", ")}: a template, a number, a gender.` : "These prompts' meta has no field with two values or more.",
            disabled: fields.length === 0,
          },
        ]}
      />
      {!form.probeLabels && <p className={e.required}>Required: what the probe reads is what the run measures, so there's no default.</p>}
      {form.probeLabels === "meta" && (
        <div className={s.grid2}>
          <Field label="Field">
            <Select value={form.probeField ?? ""} onChange={(ev) => onChange({ probeField: ev.target.value || null })}>
              {fields.map((f) => (
                <option key={f} value={f}>
                  {f}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Examples" help="Each example carries its pair's label.">
            <Select value={form.probePrompts} onChange={(ev) => onChange({ probePrompts: ev.target.value as FormState["probePrompts"] })}>
              <option value="both">Both prompts of each pair</option>
              <option value="clean">The clean prompts</option>
              <option value="corrupt">The corrupt prompts</option>
            </Select>
          </Field>
        </div>
      )}
      <div className={s.grid4}>
        <Field label="Pairs that train (%)" help="The rest measure the probes.">
          <ShareInput value={form.probeTrain} onChange={(probeTrain) => onChange({ probeTrain })} label="Share of pairs that train the probes" />
        </Field>
        <Field label="Split seed">
          <Input type="number" min={0} value={form.probeSeed} onChange={(ev) => onChange({ probeSeed: Number(ev.target.value) })} />
        </Field>
        <Field label="L2 penalty" help="On the weights; 0 for none.">
          <Input type="number" min={0} step={0.01} value={form.probeL2} onChange={(ev) => onChange({ probeL2: Number(ev.target.value) })} />
        </Field>
        <Field label="Iterations" help="Of L-BFGS, from zero.">
          <Input type="number" min={1} max={1000} value={form.probeIterations} onChange={(ev) => onChange({ probeIterations: Number(ev.target.value) })} />
        </Field>
      </div>
      <Checkbox checked={form.probeControl} onChange={(probeControl) => onChange({ probeControl })}>
        Also train a probe on shuffled labels, a control for what a probe can memorize
      </Checkbox>
      <p className={s.small}>
        A probe shows that a label can be read from a site, not that the model reads it there. Patch the site, or save the probe's
        direction and patch along it, to test that.
      </p>
    </div>
  );
}

const DAS_DIRECTIONS = [
  { value: "clean_to_corrupt" as const, title: "Clean → corrupt", detail: "Patch clean values into the corrupt run, within the subspace: does it restore the behavior?" },
  { value: "corrupt_to_clean" as const, title: "Corrupt → clean", detail: "Patch corrupt values into the clean run, within the subspace: does it break it?" },
];

/** Distributed alignment search: the direction, the subspace's size, its training and control. */
export function DASSettings({ form, onChange, width }: { form: FormState; onChange: Change; width?: number }) {
  return (
    <div className={e.stack}>
      <Choices label="Direction" value={form.direction} onChange={(direction) => onChange({ direction })} columns={2} options={DAS_DIRECTIONS} />
      <div className={s.grid3}>
        <Field label="Dimensions" help={width ? `Of the ${width}-wide activation.` : "Of the activation."}>
          <Input type="number" min={1} max={64} value={form.dasDimensions} onChange={(ev) => onChange({ dasDimensions: Number(ev.target.value) })} />
        </Field>
        <Field label="Pairs that train (%)" help="The rest measure the subspace.">
          <ShareInput value={form.dasTrain} onChange={(dasTrain) => onChange({ dasTrain })} label="Share of pairs that train the subspace" />
        </Field>
        <Field label="Seed" help="The split, the starting point and the control.">
          <Input type="number" min={0} value={form.dasSeed} onChange={(ev) => onChange({ dasSeed: Number(ev.target.value) })} />
        </Field>
        <Field label="Steps" help="Of Adam, each over every training pair.">
          <Input type="number" min={1} max={10000} value={form.dasSteps} onChange={(ev) => onChange({ dasSteps: Number(ev.target.value) })} />
        </Field>
        <Field label="Learning rate">
          <Input type="number" min={0} max={1} step={0.001} value={form.dasRate} onChange={(ev) => onChange({ dasRate: Number(ev.target.value) })} />
        </Field>
      </div>
      <Checkbox checked={form.dasControl} onChange={(dasControl) => onChange({ dasControl })}>
        Also patch a random subspace of the same size
      </Checkbox>
      <p className={s.small}>
        Each site is patched three ways on the held-out pairs: within the learned subspace, as a whole, and within a random subspace.
        A learned subspace can carry an effect the model doesn't use as a variable: read it against the whole site, and on other
        prompts.
      </p>
    </div>
  );
}
