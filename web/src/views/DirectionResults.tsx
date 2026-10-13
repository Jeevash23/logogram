import { useCallback, useMemo, useState } from "react";

import { api } from "../api/client";
import type { DASRun, ProbeRun, SiteBase, SiteResult } from "../api/types";
import { Heatmap } from "../components/Heatmap/Heatmap";
import { Button, Input } from "../components/ui";
import { inkScale, type ResolvedTheme } from "../lib/color";
import { directionsChanged } from "../lib/directions";
import { num, signed } from "../lib/format";
import { useStore } from "../store/app";
import c from "./SiteSets.module.css";
import s from "./views.module.css";

/** An interval as text: "0.61 to 0.83". */
const ci = (lo: number | null | undefined, hi: number | null | undefined) =>
  lo === null || lo === undefined || hi === null || hi === undefined ? "no interval" : `${num(lo, 2)} to ${num(hi, 2)}`;

/** A mean with its interval as a bar on a track from `lo` to `hi` (0 marked). */
function IntervalBar({ mean, low, high, lo, hi, signs }: { mean: number | null; low: number | null; high: number | null; lo: number; hi: number; signs?: boolean }) {
  const x = (v: number) => ((v - lo) / (hi - lo || 1)) * 100;
  if (mean === null) return <span className={c.faint}>waiting</span>;
  return (
    <span className={c.interval}>
      <span className={c.value}>{signs ? signed(mean, 2) : num(mean, 2)}</span>
      <span className={c.track} title={`${num(mean, 3)} (${ci(low, high)})`}>
        <span className={c.zero} style={{ left: `${x(0)}%` }} />
        {low !== null && high !== null && <span className={c.ci} style={{ left: `${x(low)}%`, width: `${Math.max(0.5, x(high) - x(low))}%` }} />}
        <span className={c.point} style={{ left: `${x(mean)}%` }} />
      </span>
    </span>
  );
}

/** Save what a run found at one of its sites as a direction of the project. */
function SaveDirection({ runId, site, suggested, row }: { runId: string; site: number; suggested: string; row?: number | null }) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState(suggested);
  const [state, setState] = useState<{ busy: boolean; error: string | null; saved: string | null }>({ busy: false, error: null, saved: null });
  const notify = useStore((st) => st.notify);
  const save = async () => {
    setState({ busy: true, error: null, saved: null });
    try {
      const saved = await api.saveRunDirection(runId, { site, name: name.trim(), row: row ?? null });
      directionsChanged();
      setState({ busy: false, error: null, saved: saved.name });
      setOpen(false);
      notify(`Saved the direction ${saved.name}. Steering, directional ablation, subspace patching and readouts can use it.`);
    } catch (err) {
      setState({ busy: false, error: err instanceof Error ? err.message : String(err), saved: null });
    }
  };
  if (!open) {
    return (
      <Button size="small" variant="ghost" onClick={() => setOpen(true)}>
        {state.saved ? `Saved as ${state.saved}` : "Save as direction"}
      </Button>
    );
  }
  return (
    <span className={s.row}>
      <Input value={name} onChange={(ev) => setName(ev.target.value)} aria-label="Direction name" style={{ width: 160 }} />
      <Button size="small" variant="primary" disabled={!name.trim() || state.busy} onClick={() => void save()}>
        Save
      </Button>
      <Button size="small" variant="ghost" onClick={() => setOpen(false)}>
        Cancel
      </Button>
      {state.error && <span className={s.small}>{state.error}</span>}
    </span>
  );
}

/** A name for a site's direction: the run's site, with letters, digits and dashes only. */
function suggestedName(prefix: string, site: SiteBase): string {
  return `${prefix}-${site.label}`.toLowerCase().replace(/[^a-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 80);
}

/**
 * A probe run: each site's held-out accuracy with its interval, beside the control trained on
 * shuffled labels and their difference (selectivity), pair by pair. On a sweep of more than one
 * column (layers × positions, layers × heads), the accuracies are also drawn as a map, in ink
 * from chance (blank) to 1: accuracy isn't an effect, so it never takes the effect colors.
 */
export function ProbeResults({
  runId,
  probe,
  sites,
  results,
  ciLevel,
  theme,
  selectedIndex,
  onSelect,
  finished,
}: {
  runId: string;
  probe: ProbeRun | null | undefined;
  sites: SiteBase[];
  results: Record<number, SiteResult>;
  ciLevel: number;
  theme: ResolvedTheme;
  selectedIndex: number | null;
  onSelect: (site: SiteResult) => void;
  finished: boolean;
}) {
  const probes = sites.filter((x) => !x.variant?.control);
  const controlOf = useMemo(() => new Map(sites.filter((x) => x.variant?.control).map((x) => [x.row, x.index])), [sites]);
  const selectivity = useMemo(() => new Map((probe?.selectivity ?? []).map((x) => [x.index, x])), [probe]);
  const chance = probe?.chance ?? 0;
  const grid = probe?.grid;
  const scale = useMemo(() => inkScale(1, theme), [theme]);
  // Where each probe sits in the scope's own layout (a row of the run's layout is a base site).
  const cells = useMemo(() => {
    const out = new Map<string, number>();
    if (!grid) return out;
    for (const site of probes) {
      const at = grid.sites[site.row];
      if (at) out.set(`${at.row}:${at.col}`, site.index);
    }
    return out;
  }, [grid, probes]);
  const value = useCallback(
    (r: number, col: number) => {
      const index = cells.get(`${r}:${col}`);
      if (index === undefined) return undefined;
      const mean = results[index]?.effect.mean;
      return mean === undefined ? null : mean;
    },
    [cells, results],
  );
  const color = useCallback((v: number) => scale(Math.max(0, (v - chance) / (1 - chance || 1))), [scale, chance]);
  const showMap = !!grid && grid.kind !== "sites" && grid.cols.length > 1;
  return (
    <div className={c.results}>
      {probe && (
        <p className={s.small}>
          The probes read {probe.labels}: {probe.classes.join(", ")}. Always answering the most common training class scores{" "}
          {num(chance, 2)} on the {probe.examples.test} held-out examples ({probe.test.length} pairs); {probe.examples.train} examples
          from {probe.train.length} pairs trained them.
        </p>
      )}
      {showMap && grid && (
        <div>
          <Heatmap
            rows={grid.rows}
            cols={grid.cols}
            value={value}
            color={color}
            theme={theme}
            rowTitle={grid.row_title}
            colTitle={grid.col_title}
            onSelect={(r, col) => {
              const index = cells.get(`${r}:${col}`);
              const result = index === undefined ? undefined : results[index];
              if (result) onSelect(result);
            }}
            showValues
            ariaLabel="Held-out accuracy of the probe at each site"
          />
          <p className={s.small}>Ink from chance ({num(chance, 2)}, blank) to every held-out example right (1).</p>
        </div>
      )}
      <table className={s.table}>
        <caption className="visually-hidden">Each site's probe, its control and their difference</caption>
        <thead>
          <tr>
            <th>Site</th>
            <th>Held-out accuracy · {ciLevel}% CI</th>
            <th>Control (shuffled labels)</th>
            <th>Selectivity · {ciLevel}% CI</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {probes.map((site) => {
            const result = results[site.index];
            const controlIndex = controlOf.get(site.row);
            const control = controlIndex === undefined ? undefined : results[controlIndex];
            const sel = selectivity.get(site.index);
            return (
              <tr key={site.index} aria-selected={site.index === selectedIndex}>
                <td>
                  <button type="button" className={c.setButton} onClick={() => result && onSelect(result)} aria-pressed={site.index === selectedIndex}>
                    <span className={c.setName}>{site.label}</span>
                  </button>
                </td>
                <td>
                  <IntervalBar mean={result?.effect.mean ?? null} low={result?.effect.lo ?? null} high={result?.effect.hi ?? null} lo={0} hi={1} />
                </td>
                <td>{control ? <IntervalBar mean={control.effect.mean} low={control.effect.lo} high={control.effect.hi} lo={0} hi={1} /> : <span className={c.faint}>—</span>}</td>
                <td>
                  {sel ? (
                    <span title={ci(sel.lo, sel.hi)}>
                      {signed(sel.difference, 2)} <span className={c.faint}>({ci(sel.lo, sel.hi)})</span>
                      {sel.selective ? " · selective" : ""}
                    </span>
                  ) : (
                    <span className={c.faint}>{finished ? "—" : "after the run"}</span>
                  )}
                </td>
                <td>{finished && <SaveDirection runId={runId} site={site.index} suggested={suggestedName("probe", site)} />}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className={s.small}>
        Selectivity is the probe's accuracy minus its control's, measured on the same held-out pairs: how much of what the probe
        reads it couldn't have memorized. A probe shows what can be read at a site, not what the model uses: save its direction and
        patch along it, or ablate it, to test that. With two classes, the saved direction points from the first class toward the
        second.
      </p>
    </div>
  );
}

/** A learned subspace run: at each site, patching within the learned subspace beside the whole
 * site and a random subspace, the share of the whole site's effect the subspace carries, and how
 * training went. */
export function DASResults({
  runId,
  das,
  sites,
  results,
  selectedIndex,
  onSelect,
  finished,
}: {
  runId: string;
  das: DASRun | null | undefined;
  sites: SiteBase[];
  results: Record<number, SiteResult>;
  selectedIndex: number | null;
  onSelect: (site: SiteResult) => void;
  finished: boolean;
}) {
  const rows = useMemo(() => {
    const byRow = new Map<number, Partial<Record<"learned" | "full" | "random", SiteBase>>>();
    for (const site of sites) {
      const key = site.variant?.subspace;
      if (!key) continue;
      byRow.set(site.row, { ...(byRow.get(site.row) ?? {}), [key]: site });
    }
    return [...byRow.entries()].sort((a, b) => a[0] - b[0]);
  }, [sites]);
  const training = useMemo(() => new Map((das?.sites ?? []).map((x) => [x.row, x])), [das]);
  const effects = rows.flatMap(([, v]) => [v.learned, v.full, v.random].map((x) => (x ? results[x.index]?.effect : undefined)));
  const ext = effects.flatMap((e) => (e ? [e.lo ?? e.mean ?? 0, e.hi ?? e.mean ?? 0] : [])).concat([0]);
  const lo = Math.min(...ext);
  const hi = Math.max(...ext);
  const cell = (site: SiteBase | undefined) => {
    const r = site ? results[site.index] : undefined;
    return r ? <IntervalBar mean={r.effect.mean} low={r.effect.lo} high={r.effect.hi} lo={lo} hi={hi} signs /> : <span className={c.faint}>—</span>;
  };
  return (
    <div className={c.results}>
      {das && (
        <p className={s.small}>
          A subspace of {das.dimensions} dimension{das.dimensions === 1 ? "" : "s"} learned at each site on {das.train.length} pairs
          ({das.steps} steps of Adam, learning rate {das.learning_rate}), measured on the {das.test.length} held-out pairs.
        </p>
      )}
      <table className={s.table}>
        <caption className="visually-hidden">Each site patched within the learned subspace, as a whole and within a random subspace</caption>
        <thead>
          <tr>
            <th>Site</th>
            <th>Learned subspace</th>
            <th>Whole site</th>
            <th>Random subspace</th>
            <th className={s.num}>Share of the whole</th>
            <th className={s.num}>Training effect</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {rows.map(([row, v]) => {
            const learned = v.learned ? results[v.learned.index] : undefined;
            const full = v.full ? results[v.full.index] : undefined;
            const share = learned?.effect.mean != null && full?.effect.mean ? learned.effect.mean / full.effect.mean : null;
            const t = training.get(row);
            const label = v.learned?.label.replace(/ learned$/, "") ?? `Row ${row}`;
            return (
              <tr key={row} aria-selected={v.learned?.index === selectedIndex}>
                <td>
                  <button type="button" className={c.setButton} onClick={() => learned && onSelect(learned)} aria-pressed={v.learned?.index === selectedIndex}>
                    <span className={c.setName}>{label}</span>
                  </button>
                </td>
                <td>{cell(v.learned)}</td>
                <td>{cell(v.full)}</td>
                <td>{cell(v.random)}</td>
                <td className={s.num}>{share === null ? "—" : num(share, 2)}</td>
                <td className={s.num}>{t?.train_effect != null ? num(t.train_effect, 2) : "—"}</td>
                <td>{finished && v.learned && <SaveDirection runId={runId} site={v.learned.index} suggested={suggestedName("das", v.learned)} />}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className={s.small}>
        The whole site is the most any subspace of it can carry; the random subspace, what a subspace of that size carries by chance.
        A learned subspace that carries much more than the random one can still be an illusion: it may steer the computation from
        outside what the model uses. Check it on other prompts, and against other methods.
      </p>
    </div>
  );
}
