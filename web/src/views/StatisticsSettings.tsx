import { Choices, Field, Select } from "../components/ui";
import type { DatasetFacts } from "../lib/buildSpec";
import { count } from "../lib/format";

/** The choice of none, which no field of the meta can be named (a field name has a character). */
const NONE = "";

/**
 * What the bootstrap resamples: single prompts, or clusters of prompts that share a field of their
 * meta, such as the template they were made from. Offers the fields every used prompt has.
 */
export function ClusterChooser({
  cluster,
  cross,
  facts,
  onChange,
  onCross,
}: {
  cluster: string | null;
  /** A second field crossed with the first, read only when there is a first. */
  cross: string | null;
  facts: DatasetFacts | null;
  onChange: (cluster: string | null) => void;
  onCross: (cross: string | null) => void;
}) {
  const fields = facts?.fields ?? [];
  const known = cluster === null || fields.some((f) => f.field === cluster);
  const options = [
    { value: NONE, title: "None: each prompt is independent", detail: "Resample single prompts." },
    ...fields.map((f) => ({
      value: f.field,
      title: `Prompts with the same ${f.field}`,
      detail: f.clusters < 2 ? "Every prompt has the same value: one cluster, nothing to resample." : `${count(f.clusters)} clusters, each resampled whole.`,
      disabled: f.clusters < 2 && f.field !== cluster,
    })),
    // A choice the current prompts can't make stays visible, so it can be changed.
    ...(known || cluster === null ? [] : [{ value: cluster, title: `Prompts with the same ${cluster}`, detail: facts ? "Not every prompt has this field." : "Choose prompts to check this field." }]),
  ];
  const others = fields.filter((f) => f.field !== cluster && (f.clusters >= 2 || f.field === cross));
  const crossKnown = cross === null || others.some((f) => f.field === cross);
  return (
    <>
    <Field
      label="Resample"
      help={
        fields.length === 0 && facts
          ? "These prompts have no field in their meta that every one of them shares, so each prompt is resampled on its own."
          : "Prompts made from one template share much of their wording, so they aren't independent. Resampling whole clusters gives intervals that allow for that; they are usually wider."
      }
    >
      <Choices
        label="Resample"
        value={cluster ?? NONE}
        onChange={(value) => onChange(value === NONE ? null : value)}
        columns={2}
        options={options}
      />
    </Field>
    {cluster !== null && (others.length > 0 || cross !== null) && (
      <Field
        label="Crossed with"
        help={`For prompts built from two lists, such as names and places: a prompt shares its ${cluster} with some prompts and the second field with others, and a bootstrap over one of the two is too narrow. Each field's values are resampled on their own, and a prompt counts as often as its two values' draws multiplied.`}
      >
        <Select value={cross ?? NONE} onChange={(ev) => onCross(ev.target.value === NONE ? null : ev.target.value)}>
          <option value={NONE}>Nothing: resample {cluster} alone</option>
          {others.map((f) => (
            <option key={f.field} value={f.field}>
              {f.field} ({count(f.clusters)} values)
            </option>
          ))}
          {!crossKnown && cross !== null && <option value={cross}>{cross} (not in every prompt)</option>}
        </Select>
      </Field>
    )}
    </>
  );
}
