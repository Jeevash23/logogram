import type { DirectionRefSpec } from "../api/types";
import { Field, Select } from "../components/ui";
import { directionText, useSavedDirections } from "../lib/directions";
import s from "./views.module.css";

/** Choose one of the project's saved directions. Only those that fit are offered: of the model's
 * width, and of one dimension where a single direction is read or added. A spec's own choice that
 * no longer fits stays shown, marked, so the form never drops it silently. */
export function DirectionPicker({
  value,
  onChange,
  label,
  help,
  oneDimension = false,
  width,
}: {
  value: DirectionRefSpec | null;
  onChange: (d: DirectionRefSpec | null) => void;
  label: string;
  help?: string;
  /** Only directions of one dimension (steering adds one, a readout reads one). */
  oneDimension?: boolean;
  /** The loaded model's residual stream width, when a model is loaded. */
  width?: number;
}) {
  const { directions, error } = useSavedDirections();
  const fits = (directions ?? []).filter(
    (d) => !d.error && (!oneDimension || d.dimensions === 1) && (width === undefined || d.width === width),
  );
  const chosen = directions?.find((d) => d.name === value?.name);
  const missing = value && !fits.some((d) => d.name === value.name);
  const note = error
    ? `The directions can't be listed: ${error}`
    : directions && fits.length === 0
      ? "No direction here fits yet. Save one from a probe or learned-subspace run (Save as direction), or from Explore → Representations."
      : chosen
        ? directionText(chosen)
        : help;
  return (
    <Field label={label} help={note}>
      <Select
        value={value?.name ?? ""}
        onChange={(ev) => {
          const name = ev.target.value;
          // Choosing again leaves a pinned direction pinned; another one is pinned when it runs.
          onChange(name ? (name === value?.name ? value : { name, sha256: null }) : null);
        }}
      >
        <option value="">Choose a saved direction</option>
        {fits.map((d) => (
          <option key={d.name} value={d.name}>
            {d.name} · {directionText(d)}
          </option>
        ))}
        {missing && value && <option value={value.name}>{value.name} (doesn't fit this model, or is gone)</option>}
      </Select>
      {missing && value && <p className={s.small}>The spec names {value.name}, which this project's directions don't offer here.</p>}
    </Field>
  );
}
