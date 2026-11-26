import type { GateSpec } from "../api/types";
import { Checkbox, Field, Input } from "../components/ui";
import s from "./views.module.css";
import e from "./ExperimentView.module.css";

/** A bar of the gate as typed: empty for none. */
function typed(value: number | null, scale = 1): string {
  return value === null ? "" : String(Math.round(value * scale * 1e6) / 1e6);
}

function parsed(text: string, scale = 1): number | null {
  if (text.trim() === "") return null;
  const value = Number(text);
  return Number.isFinite(value) ? value / scale : null;
}

/**
 * The gate: what the baseline must show before anything is intervened on. Set before the results
 * are seen, it stops a run on prompts the model doesn't solve, where every effect would describe
 * something else.
 */
export function GateSettings({ gate, onChange }: { gate: GateSpec | null; onChange: (gate: GateSpec | null) => void }) {
  return (
    <div className={e.stack}>
      <Checkbox checked={gate !== null} onChange={(on) => onChange(on ? { min_correct: 0.9, min_gap: null } : null)}>
        Stop the run unless the model shows the behavior
      </Checkbox>
      <p className={s.small}>
        {gate === null
          ? "A gate measures the baseline first and stops the run, before any intervention, if the model doesn't do the task on these prompts. Without one, a run warns and goes on."
          : "The baseline is measured first. A run that falls short stops there and says what it measured. The gate is saved in the spec, so it is set before the results are seen."}
      </p>
      {gate !== null && (
        <div className={s.grid3}>
          <Field label="Clean prompts answered, at least (%)" help="The share on which the model prefers the answer to the distractor. Empty for no such bar.">
            <Input
              type="number"
              min={1}
              max={100}
              step={1}
              value={typed(gate.min_correct, 100)}
              onChange={(ev) => onChange({ ...gate, min_correct: parsed(ev.target.value, 100) })}
            />
          </Field>
          <Field label="Mean clean–corrupt gap, at least" help="In the metric's own units, as a size. Empty for no such bar.">
            <Input type="number" min={0} step={0.1} value={typed(gate.min_gap)} onChange={(ev) => onChange({ ...gate, min_gap: parsed(ev.target.value) })} />
          </Field>
        </div>
      )}
    </div>
  );
}
