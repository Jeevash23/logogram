import { useMemo } from "react";

import type { SiteResult } from "../api/types";
import { signed } from "../lib/format";
import f from "./Forest.module.css";

/** The strongest sites by |effect|, each with its mean and confidence interval. */
export function Forest({
  results,
  selectedIndex,
  onSelect,
  flagged,
  limit = 12,
  ciLevel,
}: {
  results: SiteResult[];
  selectedIndex: number | null;
  onSelect: (site: SiteResult) => void;
  flagged?: Set<number>;
  limit?: number;
  ciLevel: number;
}) {
  const top = useMemo(
    () =>
      results
        .filter((x) => x.effect.mean !== null)
        .sort((a, b) => Math.abs(b.effect.mean ?? 0) - Math.abs(a.effect.mean ?? 0))
        .slice(0, limit),
    [results, limit],
  );
  if (top.length === 0) return null;
  // A finished run also has each site's simultaneous band: the interval that holds for all its
  // sites together, wider than the site's own.
  const banded = top.some((x) => x.band);
  const ext = top.flatMap((x) => [
    x.effect.lo ?? x.effect.mean ?? 0,
    x.effect.hi ?? x.effect.mean ?? 0,
    x.band?.lo ?? 0,
    x.band?.hi ?? 0,
    0,
  ]);
  const lo = Math.min(...ext);
  const hi = Math.max(...ext);
  const span = hi - lo || 1;
  const x = (v: number) => ((v - lo) / span) * 100;
  const longest = Math.max(...top.map((site) => site.label.length));
  const labelWidth = `${Math.min(26, Math.max(8, longest + 1))}ch`;
  return (
    <div className={f.forest} style={{ ["--label-width" as string]: labelWidth }}>
      <div className={f.head}>
        <span>Largest effects</span>
        <span className={f.headNote}>mean · {ciLevel}% CI{banded ? " · band for all sites" : ""}</span>
      </div>
      {top.map((site) => (
        <button
          key={site.index}
          type="button"
          className={f.row}
          aria-current={site.index === selectedIndex ? "true" : undefined}
          onClick={() => onSelect(site)}
        >
          <span className={f.label} title={site.label}>
            <span className={f.labelText}>{site.label}</span>
            {flagged?.has(site.index) && <span className={f.flag} title="Changes with the baseline" />}
          </span>
          <span className={f.track} aria-hidden="true">
            <span className={f.zero} style={{ left: `${x(0)}%` }} />
            {site.band && (
              <span className={f.band} style={{ left: `${x(site.band.lo)}%`, width: `${Math.max(0.5, x(site.band.hi) - x(site.band.lo))}%` }} />
            )}
            {site.effect.lo !== null && site.effect.hi !== null && (
              <span
                className={f.ci}
                style={{ left: `${x(site.effect.lo)}%`, width: `${Math.max(0.5, x(site.effect.hi) - x(site.effect.lo))}%` }}
              />
            )}
            <span className={f.point} style={{ left: `${x(site.effect.mean ?? 0)}%` }} />
          </span>
          <span className={f.value}>{signed(site.effect.mean, 2)}</span>
        </button>
      ))}
    </div>
  );
}
