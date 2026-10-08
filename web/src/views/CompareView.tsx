import { useEffect, useMemo, useState } from "react";

import { api } from "../api/client";
import type { Comparison, ComparisonChange, RunDetail } from "../api/types";
import { Heatmap, type Axis } from "../components/Heatmap/Heatmap";
import { ScaleBar } from "../components/Heatmap/ScaleBar";
import { Button, Callout, Empty, Field, Progress, Segmented, Select } from "../components/ui";
import { divergingScale, niceBound, SCALE_FLOOR } from "../lib/color";
import { ci, count, num, pct, signed } from "../lib/format";
import { siteValue } from "../lib/hooks";
import { selectionOfSite, siteAt, siteKey } from "../lib/sites";
import { runView, useStore } from "../store/app";
import s from "./views.module.css";
import c from "./CompareView.module.css";

function describeValue(v: unknown): string {
  if (v === null || v === undefined) return "none";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

export function CompareView() {
  const [aId, bId] = useStore((st) => st.compareIds);
  const runs = useStore((st) => st.runs);
  const setCompare = useStore((st) => st.setCompare);
  const mode = useStore((st) => st.compareMode);
  const theme = useStore((st) => st.theme);
  const loadRun = useStore((st) => st.loadRun);
  const details = useStore((st) => st.runDetails);
  const live = useStore((st) => st.live);
  const select = useStore((st) => st.select);
  const openRun = useStore((st) => st.openRun);
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [error, setError] = useState<string | null>(null);

  const da: RunDetail | undefined = aId ? details[aId] : undefined;
  const db: RunDetail | undefined = bId ? details[bId] : undefined;
  const va = useMemo(() => runView(da, aId ? live[aId] : undefined), [da, aId, live]);
  const vb = useMemo(() => runView(db, bId ? live[bId] : undefined), [db, bId, live]);
  const bothFinished = !!da?.summary && !!db?.summary;

  useEffect(() => {
    if (aId) void loadRun(aId);
    if (bId) void loadRun(bId);
  }, [aId, bId, loadRun]);

  useEffect(() => {
    setComparison(null);
    setError(null);
    if (!aId || !bId || !bothFinished) return;
    let cancelled = false;
    api.compare(aId, bId).then(
      (cmp) => !cancelled && setComparison(cmp),
      (e: Error) => !cancelled && setError(e.message),
    );
    return () => {
      cancelled = true;
    };
  }, [aId, bId, bothFinished]);

  const finished = runs.filter((r) => r.status === "finished" || r.id === aId || r.id === bId);
  const robust = db?.manifest?.derived_from?.kind === "robustness" && db.manifest.derived_from.run === aId;
  const bLive = bId ? live[bId] : undefined;

  const bound = useMemo(() => {
    const values = [...Object.values(va.results), ...Object.values(vb.results)].map((x) => siteValue(x, "effect"));
    return niceBound(values, SCALE_FLOOR.effect);
  }, [va.results, vb.results]);
  const color = useMemo(() => divergingScale(bound, theme), [bound, theme]);
  const diffBound = useMemo(
    () => niceBound(comparison?.diff.map((d) => d.value) ?? [], SCALE_FLOOR.effect),
    [comparison],
  );
  const diffColor = useMemo(() => divergingScale(diffBound, theme), [diffBound, theme]);
  // Flags name sites of run A; match them to either run's sites by what they measure.
  const flaggedKeys = useMemo(() => {
    const ids = new Set(comparison?.flagged ?? []);
    return new Set(va.sites.filter((x) => ids.has(x.index)).map(siteKey));
  }, [comparison, va.sites]);

  const layout = va.layout;
  const rows: Axis[] = layout?.rows.map((x) => ({ key: x.key, label: x.label })) ?? [];
  const cols: Axis[] = layout?.cols.map((x) => ({ key: x.key, label: x.label, emphasis: x.differs })) ?? [];

  return (
    <div className={s.view}>
      <div className={s.head}>
        <div className={s.titleBlock}>
          <h2 className={s.title}>{robust ? "Robustness check" : "Compare runs"}</h2>
          <p className={s.subtitle}>
            {robust
              ? "The same sweep rerun with one methodological choice changed. Conclusions that change are flagged on the map."
              : "Two runs over the same sites, for example the same sweep on two datasets or on two models of the same architecture."}
          </p>
        </div>
        <div className={s.headActions}>
          <Segmented
            label="Display"
            value={mode}
            onChange={(m) => useStore.setState({ compareMode: m })}
            options={[
              { value: "side", label: "Side by side" },
              { value: "diff", label: "Difference (B − A)", disabled: !comparison?.same_layout },
            ]}
          />
        </div>
      </div>

      <div className={s.grid2}>
        <Field label="Run A">
          <Select value={aId ?? ""} onChange={(e) => setCompare(e.target.value || null, bId)}>
            <option value="">Choose a run</option>
            {finished.map((r) => (
              <option key={r.id} value={r.id}>
                {r.name} · {r.id}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Run B">
          <Select value={bId ?? ""} onChange={(e) => setCompare(aId, e.target.value || null)}>
            <option value="">Choose a run</option>
            {finished.map((r) => (
              <option key={r.id} value={r.id}>
                {r.name} · {r.id}
              </option>
            ))}
          </Select>
        </Field>
      </div>

      {(!aId || !bId) && (
        <Empty title="Choose two runs">
          Tick two finished runs in the history, or pick them here. A robustness check from the results view opens
          here automatically.
        </Empty>
      )}

      {bLive?.status === "running" && (
        <div className={c.running}>
          <span className={s.small}>
            Running run B
            {bLive.progress ? ` · layer ${bLive.progress.layer} · ${pct(bLive.progress.done / bLive.progress.total)}` : "…"}
          </span>
          <Progress value={bLive.progress ? bLive.progress.done / bLive.progress.total : null} />
        </div>
      )}

      {comparison && comparison.spec_differences.length > 0 && (
        <div className={c.differences}>
          <span className={s.small}>What differs</span>
          {comparison.spec_differences.map((d) => (
            <div key={d.path} className={c.difference}>
              <span className={c.path}>{d.path}</span>
              <span>{describeValue(d.a)}</span>
              <span className="faint">→</span>
              <span>{describeValue(d.b)}</span>
            </div>
          ))}
        </div>
      )}
      {error && <Callout tone="error">{error}</Callout>}

      {comparison && <Verdict comparison={comparison} robust={robust} />}

      {layout && layout.kind !== "sites" && aId && bId && (
        mode === "diff" && comparison?.same_layout ? (
          <div className={c.panel}>
            <div className={c.panelHead}>
              <span className={s.panelTitle}>Difference in normalized effect, B − A</span>
              <ScaleBar bound={diffBound} theme={theme} label="B − A" />
            </div>
            <Heatmap
              rows={rows}
              cols={cols}
              value={(r, col) => comparison.diff.find((d) => d.row === r && d.col === col)?.value ?? undefined}
              color={diffColor}
              theme={theme}
              rowTitle={layout.row_title}
              colTitle={layout.col_title}
              tokens={layout.kind === "layer_position" && layout.cols[0]?.clean !== undefined}
              flagged={(r, col) => {
                const site = siteAt(va.sites, r, col);
                return !!site && flaggedKeys.has(siteKey(site));
              }}
              onSelect={(r, col) => {
                const site = siteAt(va.sites, r, col);
                if (site) select(selectionOfSite(site));
              }}
              tooltip={(r, col) => {
                const d = comparison.diff.find((x) => x.row === r && x.col === col);
                const site = siteAt(va.sites, r, col);
                return (
                  <>
                    <div style={{ fontWeight: 650 }}>{site?.label}</div>
                    <div>B − A {signed(d?.value, 3)}</div>
                  </>
                );
              }}
              ariaLabel="Difference map"
            />
          </div>
        ) : (
          <div className={c.pair}>
            {([["A", va, aId], ["B", vb, bId]] as const).map(([label, v, id]) => (
              <div key={label} className={c.panel}>
                <div className={c.panelHead}>
                  <span className={s.panelTitle}>
                    {label} · {(label === "A" ? da : db)?.spec.name ?? id}
                  </span>
                  <Button size="small" variant="ghost" onClick={() => void openRun(id, "results")}>
                    Open
                  </Button>
                </div>
                <Heatmap
                  rows={(v.layout ?? layout).rows.map((x) => ({ key: x.key, label: x.label }))}
                  cols={(v.layout ?? layout).cols.map((x) => ({ key: x.key, label: x.label, emphasis: x.differs }))}
                  value={(r, col) => {
                    const site = siteAt(v.sites, r, col);
                    if (!site) return v.layout ? undefined : null;
                    return siteValue(v.results[site.index], "effect");
                  }}
                  color={color}
                  theme={theme}
                  cellMax={26}
                  rowTitle={layout.row_title}
                  tokens={layout.kind === "layer_position" && layout.cols[0]?.clean !== undefined}
                  fade={label === "B" && bLive?.status === "running"}
                  flagged={(r, col) => {
                    const site = siteAt(v.sites, r, col);
                    return !!site && flaggedKeys.has(siteKey(site));
                  }}
                  onSelect={(r, col) => {
                    const site = siteAt(v.sites, r, col);
                    if (site) select(selectionOfSite(site));
                  }}
                  tooltip={(r, col) => {
                    const site = siteAt(v.sites, r, col);
                    const res = site ? v.results[site.index] : undefined;
                    return (
                      <>
                        <div style={{ fontWeight: 650 }}>{site?.label}</div>
                        <div>effect {signed(res?.effect.mean, 3)}</div>
                        <div style={{ opacity: 0.72 }}>CI {ci(res?.effect.lo, res?.effect.hi)}</div>
                      </>
                    );
                  }}
                  ariaLabel={`Run ${label}`}
                />
              </div>
            ))}
            <div className={c.scale}>
              <ScaleBar bound={bound} theme={theme} label="Normalized effect (shared scale)" />
            </div>
          </div>
        )
      )}

      {comparison && <ChangesTable changes={comparison.changes} topK={comparison.top_k} />}
    </div>
  );
}

function Verdict({ comparison, robust }: { comparison: Comparison; robust: boolean }) {
  const rho = comparison.spearman;
  const k = comparison.top_k;
  const overlap = comparison.top_overlap;
  const flagged = comparison.flagged.length;
  const stable = flagged === 0 && (rho ?? 0) > 0.8;
  return (
    <div className={c.verdict}>
      <div className={c.metrics}>
        <div>
          <div className={c.metricValue}>{rho === null ? "—" : num(rho, 2)}</div>
          <div className={c.metricLabel}>Spearman rank correlation over {count(comparison.n_common)} sites</div>
        </div>
        <div>
          <div className={c.metricValue}>
            {overlap} of {k}
          </div>
          <div className={c.metricLabel}>top-{k} components in common (by |effect|)</div>
        </div>
        <div>
          <div className={c.metricValue}>{comparison.n_sign_changes}</div>
          <div className={c.metricLabel}>sign reversals, with both confidence intervals excluding zero</div>
        </div>
      </div>
      <p className={s.sentence}>
        {stable ? (
          <>
            <strong>The conclusions hold.</strong> The ranking agrees closely and no top component changes.
          </>
        ) : (
          <>
            <strong>
              {flagged} conclusion{flagged === 1 ? "" : "s"} {robust ? "depend on this choice" : "differ between the runs"}.
            </strong>{" "}
            {flagged > 0 ? "They are listed below and flagged on the map." : "The overall ranking differs."}
          </>
        )}
      </p>
    </div>
  );
}

function ChangesTable({ changes, topK }: { changes: ComparisonChange[]; topK: number }) {
  const select = useStore((st) => st.select);
  if (changes.length === 0) return null;
  const flagText = (f: ComparisonChange["flags"][number]) =>
    f === "sign" ? "sign changed" : f === "left_top" ? `left the top ${topK}` : `entered the top ${topK}`;
  return (
    <div className={c.changes}>
      <h3 className={s.panelTitle}>Top components in either run</h3>
      <table className={s.table}>
        <thead>
          <tr>
            <th>Component</th>
            <th className={s.num}>A</th>
            <th className={s.num}>B</th>
            <th className={s.num}>Rank</th>
            <th>Change</th>
          </tr>
        </thead>
        <tbody>
          {changes.map((ch) => (
            <tr
              key={siteKey(ch)}
              data-clickable
              onClick={() =>
                select({
                  layer: ch.layer,
                  part: ch.kind === "head" ? "head" : ch.kind === "attn_out" ? "attn" : ch.kind === "mlp_out" ? "mlp" : "resid",
                  head: ch.head ?? undefined,
                  positionKey: ch.position_key === "all" ? undefined : ch.position_key,
                })
              }
            >
              <td>{ch.label}</td>
              <td className={s.num}>
                {signed(ch.effect_a.mean)} <span className="faint">[{ci(ch.effect_a.lo, ch.effect_a.hi)}]</span>
              </td>
              <td className={s.num}>
                {signed(ch.effect_b.mean)} <span className="faint">[{ci(ch.effect_b.lo, ch.effect_b.hi)}]</span>
              </td>
              <td className={s.num}>
                {ch.rank_a} → {ch.rank_b}
              </td>
              <td className={ch.flags.length ? c.flagged : "faint"}>
                {ch.flags.length ? ch.flags.map(flagText).join(", ") : "stable"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
