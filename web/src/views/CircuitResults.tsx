import { useMemo, useRef } from "react";

import type { CircuitInfo, CircuitRow, SiteBase, SiteResult, Stat, UniverseKind } from "../api/types";
import { useElementSize } from "../lib/canvas";
import { setWhat } from "../lib/circuits";
import { ci, count, num, plural, signed } from "../lib/format";
import { logogram, RING_RADIUS } from "../lib/logogram";
import { siteText } from "../lib/spec";
import s from "./views.module.css";
import c from "./SiteSets.module.css";

/** A set of a run: its result, and what the run says about it against replacing everything. */
interface SetRow {
  site: SiteBase;
  result: SiteResult | undefined;
  row: CircuitRow | undefined;
  complement: boolean;
  size: number;
}

/**
 * The results of a run over sets of sites: each set's effect with its interval, its share of
 * replacing everything, the faithfulness of the sets that keep their sites (also against their
 * size), what each site adds to a circuit, and how pairs of sites interact.
 */
export function CircuitResults({
  sites,
  results,
  circuit,
  universe,
  effectLabel,
  ciLevel,
  selectedIndex,
  onSelect,
}: {
  sites: SiteBase[];
  results: Record<number, SiteResult>;
  circuit: CircuitInfo | null | undefined;
  universe: UniverseKind[] | null | undefined;
  effectLabel: string;
  ciLevel: number;
  selectedIndex: number | null;
  onSelect: (site: SiteBase) => void;
}) {
  const rows: SetRow[] = useMemo(() => {
    const byIndex = new Map(circuit?.rows.map((r) => [r.index, r]) ?? []);
    return [...sites]
      .sort((a, b) => a.index - b.index)
      .map((site) => ({
        site,
        result: results[site.index],
        row: byIndex.get(site.index),
        complement: !!(site.variant?.complement ?? byIndex.get(site.index)?.complement),
        size: site.variant?.size ?? site.members?.length ?? byIndex.get(site.index)?.size ?? 0,
      }));
  }, [sites, results, circuit]);
  const everything = circuit?.everything ?? null;
  const finished = !!circuit;
  // One scale for every set's interval, through zero.
  const ext = rows.flatMap((r) => [r.result?.effect.lo ?? r.result?.effect.mean ?? 0, r.result?.effect.hi ?? r.result?.effect.mean ?? 0, 0]);
  const lo = Math.min(...ext);
  const hi = Math.max(...ext);
  const x = (v: number) => ((v - lo) / (hi - lo || 1)) * 100;
  const without = rows.filter((r) => r.row?.without);
  const interactions = rows.filter((r) => r.row?.interaction);
  const curve = rows.filter((r) => r.row?.faithfulness && r.complement && r.size > 0);

  return (
    <div className={c.results}>
      {curve.length > 0 && <FaithfulnessChart rows={curve} ciLevel={ciLevel} selectedIndex={selectedIndex} onSelect={onSelect} />}
      <table className={`${s.table} ${c.table}`}>
        <caption className="visually-hidden">Each set of sites, intervened on at once</caption>
        <thead>
          <tr>
            <th>Set</th>
            <th>{effectLabel} · {ciLevel}% CI</th>
            <th className={s.num}>Share of everything</th>
            <th className={s.num}>Faithfulness</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ site, result, row, complement, size }) => (
            <tr key={site.index} aria-selected={site.index === selectedIndex}>
              <td>
                <button type="button" className={c.setButton} onClick={() => onSelect(site)} aria-pressed={site.index === selectedIndex}>
                  <span className={c.setName}>{site.label}</span>
                </button>
                <span className={c.setWhat} title={(site.members ?? []).map(siteText).join(", ")}>
                  {setWhat(complement, size, universe)}
                  {site.members && site.members.length > 0 && site.members.length <= 4 ? `: ${site.members.map(siteText).join(", ")}` : ""}
                </span>
              </td>
              <td>
                {result ? (
                  <span className={c.interval}>
                    <span className={c.value}>{signed(result.effect.mean, 2)}</span>
                    <span className={c.track} title={`${signed(result.effect.mean, 3)} (${ci(result.effect.lo, result.effect.hi, 3)})`}>
                      <span className={c.zero} style={{ left: `${x(0)}%` }} />
                      {result.effect.lo !== null && result.effect.hi !== null && (
                        <span className={c.ci} style={{ left: `${x(result.effect.lo)}%`, width: `${Math.max(0.5, x(result.effect.hi) - x(result.effect.lo))}%` }} />
                      )}
                      <span className={c.point} style={{ left: `${x(result.effect.mean ?? 0)}%` }} />
                    </span>
                  </span>
                ) : (
                  <span className={c.faint}>waiting</span>
                )}
              </td>
              <td className={s.num}>{site.index === everything ? <span className={c.faint}>everything</span> : <StatCell stat={row?.share} finished={finished} />}</td>
              <td className={s.num}>{complement && size > 0 ? <StatCell stat={row?.faithfulness} finished={finished} /> : <span className={c.faint}>—</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className={c.notes}>
        <p className={s.small}>
          Share: a set's change in the metric as a share of replacing everything. Faithfulness, for a set that keeps its
          sites: 1 − share, the part of the behavior they carry alone. 1 means keeping them alone loses nothing; 0 means
          it does no better than replacing everything. Intervals come from the same resamples as the effects.
        </p>
        {finished && everything === null && (
          <p className={s.small}>
            Shares and faithfulness are read against a set that keeps nothing and replaces the whole rest of the model. Add one
            to read them.
          </p>
        )}
      </div>
      {without.length > 0 && (
        <section className={c.notes} aria-label="What each site adds">
          <h3 className={s.panelTitle}>What each site adds</h3>
          {without.map(({ site, row }) => {
            const w = row?.without;
            if (!w) return null;
            const needed = (w.drop.lo ?? 0) > 0;
            return (
              <p key={site.index} className={c.note}>
                Leaving <strong>{w.site}</strong> out of {w.of} lowers its faithfulness by {num(w.drop.mean, 2)}{" "}
                <span className={c.faint}>({ci(w.drop.lo, w.drop.hi)})</span>: {needed ? "the circuit needs it." : "not clearly needed."}
              </p>
            );
          })}
        </section>
      )}
      {interactions.length > 0 && (
        <section className={c.notes} aria-label="How sites interact">
          <h3 className={s.panelTitle}>How sites interact</h3>
          {interactions.map(({ site, row }) => {
            const it = row?.interaction;
            if (!it) return null;
            const clear = (it.effect.lo ?? 0) > 0 || (it.effect.hi ?? 0) < 0;
            return (
              <p key={site.index} className={c.note}>
                <strong>{it.a}</strong> and <strong>{it.b}</strong> together: {signed(it.effect.mean, 2)} beyond the sum of each
                alone <span className={c.faint}>({ci(it.effect.lo, it.effect.hi)})</span>
                {clear ? ((it.effect.mean ?? 0) > 0 ? ", more than the sum." : ", less than the sum.") : ", no clear interaction."}
              </p>
            );
          })}
          <p className={s.small}>Effect of both − effect of the first − effect of the second, normalized like the effects.</p>
        </section>
      )}
    </div>
  );
}

function StatCell({ stat, finished }: { stat: Stat | null | undefined; finished: boolean }) {
  if (!stat || stat.mean === null) return <span className={c.faint}>{finished ? "undefined" : "when finished"}</span>;
  return (
    <>
      {num(stat.mean, 2)} <span className={c.faint}>[{ci(stat.lo, stat.hi)}]</span>
    </>
  );
}

const RING = logogram({ seed: "selection", detail: 0, weight: 0.4 }).d;

/** Faithfulness against the number of sites kept: one point per set that keeps its sites, with its
 * interval; nested circuits of different sizes are joined. Drawn in ink. */
function FaithfulnessChart({
  rows,
  ciLevel,
  selectedIndex,
  onSelect,
}: {
  rows: SetRow[];
  ciLevel: number;
  selectedIndex: number | null;
  onSelect: (site: SiteBase) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const { width } = useElementSize(host);
  const sizes = [...new Set(rows.map((r) => r.size))].sort((a, b) => a - b);
  const values = rows.flatMap((r) => [r.row?.faithfulness?.lo, r.row?.faithfulness?.hi, r.row?.faithfulness?.mean]).filter((v): v is number => v !== null && v !== undefined && Number.isFinite(v));
  const y0 = Math.min(0, ...values);
  const y1 = Math.max(1, ...values);
  const W = Math.max(280, Math.min(width, 640));
  const H = 200;
  const L = 44;
  const R = 16;
  const T = 12;
  const B = 40;
  const xAt = (size: number) => (sizes.length === 1 ? L + (W - L - R) / 2 : L + (sizes.indexOf(size) / (sizes.length - 1)) * (W - L - R));
  const yAt = (v: number) => T + (1 - (v - y0) / (y1 - y0 || 1)) * (H - T - B);
  // Nested circuits have one set per size: join them into a curve.
  const joined = sizes.length === rows.length && rows.length > 1;
  const ordered = [...rows].sort((a, b) => a.size - b.size);
  const ticks = [y0, 0, 0.5, 1, y1].filter((v, i, a) => a.indexOf(v) === i && v >= y0 && v <= y1);
  const description = ordered.map((r) => `${r.site.label}, ${plural(r.size, "site")}: ${num(r.row?.faithfulness?.mean, 2)}`).join("; ");
  return (
    <figure className={c.chart} ref={host}>
      <figcaption className={c.chartTitle}>Faithfulness against the number of sites kept</figcaption>
      {width > 0 && (
        <svg width={W} height={H} role="img" aria-label={`Faithfulness of each set that keeps its sites, with ${ciLevel}% intervals. ${description}.`}>
          <line className={c.axis} x1={L} x2={W - R} y1={H - B} y2={H - B} />
          {[0, 1].map((v) => (
            <line key={v} className={c.reference} x1={L} x2={W - R} y1={yAt(v)} y2={yAt(v)} />
          ))}
          {ticks.map((v) => (
            <text key={v} className={c.tick} x={L - 8} y={yAt(v)} textAnchor="end" dominantBaseline="middle">
              {num(v, v === 0 || v === 1 ? 0 : 1)}
            </text>
          ))}
          {sizes.map((size) => (
            <text key={size} className={c.tick} x={xAt(size)} y={H - B + 16} textAnchor="middle">
              {count(size)}
            </text>
          ))}
          <text className={c.axisLabel} x={(L + W - R) / 2} y={H - 6} textAnchor="middle">
            Sites kept
          </text>
          <text className={c.axisLabel} x={-((T + H - B) / 2)} y={11} textAnchor="middle" transform="rotate(-90)">
            Faithfulness
          </text>
          {joined && (
            <polyline className={c.curve} points={ordered.map((r) => `${xAt(r.size)},${yAt(r.row?.faithfulness?.mean ?? 0)}`).join(" ")} />
          )}
          {ordered.map((r) => {
            const f = r.row?.faithfulness;
            if (!f || f.mean === null) return null;
            const cx = xAt(r.size);
            const cy = yAt(f.mean);
            const selected = r.site.index === selectedIndex;
            const radius = 9;
            return (
              <g key={r.site.index}>
                {f.lo !== null && f.hi !== null && <line className={c.whisker} x1={cx} x2={cx} y1={yAt(f.lo)} y2={yAt(f.hi)} />}
                {selected && (
                  <path className={c.ring} d={RING} transform={`translate(${cx} ${cy}) scale(${radius / RING_RADIUS}) translate(-50 -50)`} strokeWidth={2.6 / (radius / RING_RADIUS)} />
                )}
                <circle
                  className={c.dot}
                  cx={cx}
                  cy={cy}
                  r={4}
                  tabIndex={0}
                  role="button"
                  aria-label={`${r.site.label}: faithfulness ${num(f.mean, 2)}, ${ci(f.lo, f.hi)}`}
                  onClick={() => onSelect(r.site)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      onSelect(r.site);
                    }
                  }}
                >
                  <title>{`${r.site.label}: ${num(f.mean, 3)} (${ci(f.lo, f.hi, 3)})`}</title>
                </circle>
              </g>
            );
          })}
        </svg>
      )}
      <p className={s.small}>
        Each point is a set that keeps its sites and replaces the rest of the model, with its {ciLevel}% interval. Dashed
        lines: 1, as faithful as the whole model; 0, no better than replacing everything.
      </p>
    </figure>
  );
}
