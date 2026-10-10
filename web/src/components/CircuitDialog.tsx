import { useEffect, useMemo, useState } from "react";

import type { SiteSetSpec, SiteSpec, UniverseKind } from "../api/types";
import {
  circuitSets,
  MAX_SETS,
  setWhat,
  siteSetsError,
  topKSets,
  universeFor,
  universeText,
} from "../lib/circuits";
import { plural } from "../lib/format";
import { siteText } from "../lib/spec";
import { useStore } from "../store/app";
import { UniverseChooser } from "../views/SiteSetsEditor";
import c from "../views/SiteSets.module.css";
import { Button, Callout, Checkbox, Dialog, Field, Input } from "./ui";

/** Sets shown in the dialog's plan before saying how many more there are. */
const SHOWN = 8;

/**
 * Test sites as a circuit: the staged sites kept alone, removed, and against everything replaced
 * (with a set per site for minimality), or a finished run's strongest sites as nested circuits for
 * a curve of faithfulness against size. The sets go into the experiment form, one undo step.
 */
export function CircuitDialog({
  open,
  onOpenChange,
  sites,
  ranked,
  runId,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** The staged sites, tested as one circuit. */
  sites?: SiteSpec[];
  /** Or a run's sites, strongest first: its top k as nested circuits. */
  ranked?: SiteSpec[];
  runId?: string;
}) {
  const model = useStore((st) => st.model.info);
  const configureSets = useStore((st) => st.configureSets);
  const [minimality, setMinimality] = useState(true);
  const [k, setK] = useState(8);
  const [universe, setUniverse] = useState<UniverseKind[] | null>(null);
  const [touched, setTouched] = useState(false);

  const top = ranked !== undefined;
  const available = ranked?.length ?? 0;
  const size = top ? Math.max(1, Math.min(available, Math.round(k) || 1)) : (sites?.length ?? 0);
  const circuit = useMemo(() => (top ? (ranked ?? []).slice(0, size) : (sites ?? [])), [top, ranked, sites, size]);
  const kinds = universeFor(circuit);

  // The rest of the model starts as the kinds of component the circuit is made of, and follows
  // the circuit until it is chosen by hand.
  useEffect(() => {
    if (!open) return;
    setTouched(false);
    if (top) setK(Math.min(8, Math.max(1, available)));
  }, [open, top, available]);
  const suggested = "universe" in kinds ? kinds.universe.join(",") : "";
  useEffect(() => {
    if (open && !touched) setUniverse(suggested ? (suggested.split(",") as UniverseKind[]) : null);
  }, [open, touched, suggested]);

  const sets: SiteSetSpec[] = top ? topKSets(ranked ?? [], size) : circuitSets(sites ?? [], { minimality });
  const error = "error" in kinds ? kinds.error : siteSetsError(universe, sets, model ?? null);
  const single = !top && (sites?.length ?? 0) === 1;

  const setUp = () => {
    if (error) return;
    configureSets({ kind: "site_sets", universe, sets }, top ? { runId } : { staged: true });
    onOpenChange(false);
  };

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title={top ? "Test the strongest sites as a circuit" : `Test ${plural(sites?.length ?? 0, "site")} as a circuit`}
      description={
        top
          ? "Keep the top 1, 2, 4, 8 … sites alone and replace the rest of the model, to see how faithfulness grows with the circuit's size."
          : "Keep these sites alone and replace the rest of the model, remove them, and replace everything, each in one forward pass."
      }
      wide
      footer={
        <>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button variant="primary" disabled={!!error} onClick={setUp}>
            Set up the test
          </Button>
        </>
      }
    >
      {top ? (
        <Field
          label="Sites in the largest circuit"
          help={`The run's strongest heads, attention outputs or MLP outputs, by the size of their mean effect: ${plural(available, "site")} to choose from.`}
        >
          <Input type="number" min={1} max={available} step={1} value={k} onChange={(ev) => setK(Number(ev.target.value))} style={{ width: 100 }} />
        </Field>
      ) : (
        <p className={c.note}>{(sites ?? []).map(siteText).join(", ")}</p>
      )}
      <Field
        label="The rest of the model"
        help={`Keeping the circuit replaces ${universeText(universe)} except the circuit's sites, at every position; replacing everything replaces all of them. A layer's attention output is the sum of its heads, so the rest is made of one or the other.`}
      >
        <UniverseChooser
          value={universe}
          onChange={(next) => {
            setTouched(true);
            setUniverse(next);
          }}
        />
      </Field>
      {!top && (
        <Checkbox checked={minimality && !single} disabled={single} onChange={setMinimality}>
          {single
            ? "Test each site's part (a circuit of one site has none to test)"
            : `Also keep the circuit without each site, one at a time: what each of the ${sites?.length ?? 0} sites adds`}
        </Checkbox>
      )}
      <div>
        <div className={c.chartTitle}>
          {plural(sets.length, "set")}, each intervened on at once with the form's method
        </div>
        <ul className={c.plan}>
          {sets.slice(0, SHOWN).map((set) => (
            <li key={set.label}>
              <strong>{set.label}</strong>{" "}
              <span>{setWhat(set.complement, set.sites.length, universe)}</span>
            </li>
          ))}
          {sets.length > SHOWN && <li><span>and {plural(sets.length - SHOWN, "more set")}</span></li>}
        </ul>
      </div>
      {sets.length > MAX_SETS && (
        <Callout tone="error" title={`A run holds at most ${MAX_SETS} sets`}>
          This needs {sets.length}. Test fewer sites{!top ? ", or leave out what each site adds" : ""}.
        </Callout>
      )}
      {error && sets.length <= MAX_SETS && <Callout tone="error" title="These sets can't run">{error}</Callout>}
    </Dialog>
  );
}
