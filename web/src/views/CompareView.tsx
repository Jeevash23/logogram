import { useCallback, useEffect, useMemo, useState } from "react";

import { api } from "../api/client";
import type { Comparison, ComparisonChange, EdgeSpec, FaithfulnessCurveSpec, Layout, PathReceiverSpec, RunDetail, RunListing, SiteBase, SiteResult, SiteSpec } from "../api/types";
import { Heatmap, type Axis } from "../components/Heatmap/Heatmap";
import { ScaleBar } from "../components/Heatmap/ScaleBar";
import { Button, Callout, Empty, Field, Progress, Segmented, Select } from "../components/ui";
import { divergingScale, niceBound, SCALE_FLOOR, type ColorScale, type ResolvedTheme } from "../lib/color";
import { differenceText } from "../lib/environment";
import { ci, count, num, pct, signed } from "../lib/format";
import { pickerGroups, runTitle } from "../lib/runs";
import { siteValue } from "../lib/hooks";
import { findSite, gridKey, isResidKind, partOfKind, selectionOfSite, siteGrid, siteKey, type Selection } from "../lib/sites";
import { edgeLabel, receiverLabel, siteText } from "../lib/spec";
import { fieldWords, membershipText, sweepsOf, sweepsOfRun, valueName } from "../lib/sweeps";
import { HOLDS_RHO } from "../lib/thresholds";
import { EdgeVerificationPanel } from "./EdgeResults";
import { FlagMark } from "./Forest";
import { runView, useStore } from "../store/app";
import s from "./views.module.css";
import c from "./CompareView.module.css";

function describeValue(v: unknown): string {
  if (v === null || v === undefined) return "none";
  if (Array.isArray(v) && v.length > 0 && v.every((x) => x && typeof x === "object" && "sender" in x && "receiver" in x)) {
    // A list of edges: name them, as the results do.
    const names = (v as EdgeSpec[]).map((e) => edgeLabel(e));
    return `${names.length} edge${names.length === 1 ? "" : "s"}: ${names.slice(0, 6).join(", ")}${names.length > 6 ? ", …" : ""}`;
  }
  if (v && typeof v === "object" && "counts" in v && "patch" in v && "control" in v) {
    // A faithfulness curve.
    const curve = v as FaithfulnessCurveSpec;
    return `${curve.counts.join(", ")} edges, patching ${curve.patch === "chosen" ? "the chosen edges" : "every edge but the chosen"}; ${curve.control.draws} random draw${curve.control.draws === 1 ? "" : "s"}, seed ${curve.control.seed}`;
  }
  if (Array.isArray(v) && v.length > 0 && v.every((x) => x && typeof x === "object" && "kind" in x && ["head", "mlp", "logits"].includes(String(x.kind)) && !("position" in x))) {
    // Path patching's receivers.
    const names = (v as PathReceiverSpec[]).map((r) => receiverLabel(r));
    return `${names.slice(0, 6).join(", ")}${names.length > 6 ? `, and ${names.length - 6} more` : ""}`;
  }
  if (Array.isArray(v) && v.length > 0 && v.every((x) => x && typeof x === "object" && "kind" in x && "layer" in x)) {
    // A list of sites: name them, as the map does.
    const names = (v as SiteSpec[]).map(siteText);
    return `${names.length} site${names.length === 1 ? "" : "s"}: ${names.slice(0, 6).join(", ")}${names.length > 6 ? ", …" : ""}`;
  }
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
  const da: RunDetail | undefined = useStore((st) => (aId ? st.runDetails[aId] : undefined));
  const db: RunDetail | undefined = useStore((st) => (bId ? st.runDetails[bId] : undefined));
  // Only these two runs: another run's progress doesn't redraw the comparison.
  const liveA = useStore((st) => (aId ? st.live[aId] : undefined));
  const liveB = useStore((st) => (bId ? st.live[bId] : undefined));
  const select = useStore((st) => st.select);
  const selection = useStore((st) => st.selection);
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [error, setError] = useState<string | null>(null);

  const va = useMemo(() => runView(da, liveA), [da, liveA]);
  const vb = useMemo(() => runView(db, liveB), [db, liveB]);
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

  // Runs with results: finished, or stopped early with what finished kept.
  const finished = runs.filter((r) => r.status === "finished" || r.partial || r.id === aId || r.id === bId);
  // What each run is called: the title given to it, or its spec's name.
  const titleOf = (id: string | null, detail: RunDetail | undefined) => {
    const listed = id ? runs.find((r) => r.id === id) : undefined;
    return listed ? runTitle(listed) : detail?.listing?.title || detail?.spec.name || id || "";
  };
  const titleA = titleOf(aId, da);
  const titleB = titleOf(bId, db);
  const robust = db?.manifest?.derived_from?.kind === "robustness" && db.manifest.derived_from.run === aId;
  // B patches, for real, the strongest sites that A (attribution patching) estimated.
  const verification = db?.manifest?.derived_from?.kind === "verification" && db.manifest.derived_from.run === aId;
  const bLive = liveB;

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
  const rows: Axis[] = useMemo(() => layout?.rows.map((x) => ({ key: x.key, label: x.label })) ?? [], [layout]);
  const cols: Axis[] = useMemo(() => layout?.cols.map((x) => ({ key: x.key, label: x.label, emphasis: x.differs })) ?? [], [layout]);
  // One lookup per cell rather than a search through every site or difference.
  const gridA = useMemo(() => siteGrid(va.sites), [va.sites]);
  const diffs = useMemo(() => new Map((comparison?.diff ?? []).map((d) => [gridKey(d.row, d.col), d])), [comparison]);
  const diffValue = useCallback((r: number, col: number) => diffs.get(gridKey(r, col))?.value ?? undefined, [diffs]);
  const diffFlagged = useCallback((r: number, col: number) => {
    const site = gridA.get(gridKey(r, col));
    return !!site && flaggedKeys.has(siteKey(site));
  }, [gridA, flaggedKeys]);
  const diffSelect = useCallback((r: number, col: number) => {
    const site = gridA.get(gridKey(r, col));
    if (site) select(selectionOfSite(site));
  }, [gridA, select]);
  // The selected site, as each run measured it. The difference map is drawn on run A's cells.
  const siteA = findSite(va.sites, selection);
  const siteB = findSite(vb.sites, selection);
  const selectedR = siteA?.row ?? null;
  const selectedC = siteA?.col ?? null;
  const diffSelected = useMemo(() => (selectedR !== null && selectedC !== null ? { r: selectedR, c: selectedC } : null), [selectedR, selectedC]);
  const diffTooltip = useCallback((r: number, col: number) => (
    <>
      <div style={{ fontWeight: 650 }}>{gridA.get(gridKey(r, col))?.label}</div>
      <div>B − A {signed(diffs.get(gridKey(r, col))?.value, 3)}</div>
    </>
  ), [gridA, diffs]);

  return (
    <div className={s.view}>
      <div className={s.head}>
        <div className={s.titleBlock}>
          <h1 className={s.title}>
            {verification
              ? "Estimates checked by patching"
              : robust
                ? "Robustness check"
                : da && db
                  ? titleA === titleB
                    ? `Two runs of ${titleA}`
                    : `${titleA} against ${titleB}`
                  : "Two runs side by side"}
          </h1>
          <p className={s.subtitle}>
            {verification && layout?.kind === "edges"
              ? "A estimates every edge to first order; B path-patches the strongest of them, one at a time. Where they disagree, the estimate is flagged."
              : verification
              ? "A estimates every site to first order; B patches the strongest of them for real. Where they disagree, the estimate is flagged."
              : robust
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
            <RunOptions runs={finished} />
          </Select>
        </Field>
        <Field label="Run B">
          <Select value={bId ?? ""} onChange={(e) => setCompare(aId, e.target.value || null)}>
            <option value="">Choose a run</option>
            <RunOptions runs={finished} />
          </Select>
        </Field>
      </div>

      {(aId || bId) && <SweepLinks aId={aId} bId={bId} />}

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
          <Progress value={bLive.progress ? bLive.progress.done / bLive.progress.total : null} label="Progress of run B" />
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
      {comparison && (comparison.environment_differences?.length ?? 0) > 0 && (
        <section className={c.differences} aria-label="Where the runs ran">
          <span className={s.small}>They ran with other versions or on other devices, which alone can change results in the last digits</span>
          {comparison.environment_differences?.map((d) => {
            const { a, b } = differenceText(d);
            return (
              <div key={d.key} className={c.difference}>
                <span className={c.path}>{d.label}</span>
                <span>{a}</span>
                <span className="faint">→</span>
                <span>{b}</span>
              </div>
            );
          })}
        </section>
      )}
      {error && <Callout tone="error">{error}</Callout>}
      {comparison && (comparison.partial_a || comparison.partial_b) && (
        <Callout title="A run here stopped before it finished">
          {[
            comparison.partial_a ? `Run A: ${comparison.partial_a.note}` : null,
            comparison.partial_b ? `Run B: ${comparison.partial_b.note}` : null,
          ]
            .filter(Boolean)
            .join(" ")}{" "}
          Only what both runs kept is compared.
        </Callout>
      )}

      {comparison && <Verdict comparison={comparison} robust={robust} verification={verification} />}

      {verification && layout?.kind === "edges" && aId && (
        <EdgeVerificationPanel
          runId={aId}
          ciLevel={Math.round((da?.summary?.statistics.ci ?? 0.95) * 100)}
          selectedIndex={siteA?.index ?? null}
          onSelect={(index) => {
            const site = va.sites.find((x) => x.index === index);
            if (site) select(selectionOfSite(site));
          }}
        />
      )}
      {/* Edges are drawn in each run's results; here they are compared in the tables. */}
      {layout && layout.kind !== "sites" && layout.kind !== "edges" && aId && bId && (
        mode === "diff" && comparison?.same_layout ? (
          <div className={c.panel}>
            <div className={c.panelHead}>
              <span className={s.panelTitle}>Difference in normalized effect, B − A</span>
              {flaggedKeys.size > 0 && <span className={c.flagNote}><FlagMark /> flagged: the conclusion differs</span>}
              <ScaleBar bound={diffBound} theme={theme} label="B − A" />
            </div>
            <Heatmap
              rows={rows}
              cols={cols}
              value={diffValue}
              color={diffColor}
              theme={theme}
              rowTitle={layout.row_title}
              colTitle={layout.col_title}
              tokens={layout.kind === "layer_position" && layout.cols[0]?.clean !== undefined}
              flagged={diffFlagged}
              selected={diffSelected}
              onSelect={diffSelect}
              tooltip={diffTooltip}
              ariaLabel="Difference map"
            />
          </div>
        ) : (
          <div className={c.pair}>
            {([["A", va, aId], ["B", vb, bId]] as const).map(([label, v, id]) => (
              <ComparePanel
                key={label}
                label={label}
                id={id}
                name={label === "A" ? titleA : titleB}
                layout={v.layout ?? layout}
                measured={!!v.layout}
                sites={v.sites}
                results={v.results}
                color={color}
                theme={theme}
                running={label === "B" && bLive?.status === "running"}
                flaggedKeys={flaggedKeys}
                selection={selection}
              />
            ))}
            <div className={c.scale}>
              <ScaleBar bound={bound} theme={theme} label="Normalized effect (shared scale)" />
              {flaggedKeys.size > 0 && <span className={c.flagNote}><FlagMark /> flagged: the conclusion differs</span>}
            </div>
          </div>
        )
      )}

      {aId && bId && (siteA || siteB) && (
        <SelectedSite
          siteA={siteA}
          siteB={siteB}
          resultA={siteA ? va.results[siteA.index] : undefined}
          resultB={siteB ? vb.results[siteB.index] : undefined}
          comparison={comparison}
          aId={aId}
          bId={bId}
          flagged={!!siteA && flaggedKeys.has(siteKey(siteA))}
        />
      )}

      {comparison && <ChangesTable changes={comparison.changes} topK={comparison.top_k} selected={siteA ?? siteB} />}
    </div>
  );
}

/** Where a compared run is a value of a sweep: which value, and a way to the sweep. Two values of
 * one sweep are said together. */
function SweepLinks({ aId, bId }: { aId: string | null; bId: string | null }) {
  const runs = useStore((st) => st.runs);
  const openSweep = useStore((st) => st.openSweep);
  const sweeps = useMemo(() => sweepsOf(runs), [runs]);
  const inA = aId ? sweepsOfRun(sweeps, aId) : [];
  const inB = bId ? sweepsOfRun(sweeps, bId) : [];
  const shared = inA.find((a) => inB.some((b) => b.sweep.id === a.sweep.id));
  const lines: { key: string; text: string; sweep: string }[] = [];
  if (shared) {
    const b = inB.find((x) => x.sweep.id === shared.sweep.id);
    const { sweep } = shared;
    const words = fieldWords(sweep.field);
    lines.push({
      key: sweep.id,
      sweep: sweep.id,
      text: `A and B are ${valueName(sweep.field, sweep.values[shared.index])} and ${valueName(sweep.field, sweep.values[b?.index ?? 0])} of one sweep over ${sweep.values.length} ${words.plural}.`,
    });
  }
  for (const [label, memberships] of [["A", inA], ["B", inB]] as const) {
    for (const { sweep, index } of memberships) {
      if (shared && sweep.id === shared.sweep.id) continue;
      lines.push({ key: `${label}:${sweep.id}`, sweep: sweep.id, text: membershipText(sweep, index, label) });
    }
  }
  if (lines.length === 0) return null;
  return (
    <div className={c.sweeps}>
      {lines.map((line) => (
        <p key={line.key} className={c.sweepLine}>
          {line.text}{" "}
          <button type="button" className={c.sweepLink} onClick={() => openSweep(line.sweep)}>
            Show the sweep
          </button>
        </p>
      ))}
    </div>
  );
}

/** The runs to choose from, by title (and id, which tells reruns apart): starred first, then the
 * rest, then archived ones. */
function RunOptions({ runs }: { runs: RunListing[] }) {
  const option = (r: RunListing) => (
    <option key={r.id} value={r.id}>
      {runTitle(r)} · {r.id}
      {r.partial ? " · partial" : ""}
    </option>
  );
  return (
    <>
      {pickerGroups(runs).map((group) =>
        group.label === null ? (
          group.runs.map(option)
        ) : (
          <optgroup key={group.label} label={group.label}>
            {group.runs.map(option)}
          </optgroup>
        ),
      )}
    </>
  );
}

/** One run of a side-by-side comparison, on the shared scale. Its accessors keep their identity
 * while the other run streams, so its cells aren't drawn again for nothing. */
function ComparePanel({
  label,
  id,
  name,
  layout,
  measured,
  sites,
  results,
  color,
  theme,
  running,
  flaggedKeys,
  selection,
}: {
  label: "A" | "B";
  id: string;
  name: string;
  layout: Layout;
  /** False while the run hasn't reported its own layout: every cell is then pending. */
  measured: boolean;
  sites: ReturnType<typeof runView>["sites"];
  results: ReturnType<typeof runView>["results"];
  color: ColorScale;
  theme: ResolvedTheme;
  running: boolean;
  flaggedKeys: Set<string>;
  selection: Selection | null;
}) {
  const select = useStore((st) => st.select);
  const openRun = useStore((st) => st.openRun);
  const grid = useMemo(() => siteGrid(sites), [sites]);
  const rows: Axis[] = useMemo(() => layout.rows.map((x) => ({ key: x.key, label: x.label })), [layout]);
  const cols: Axis[] = useMemo(() => layout.cols.map((x) => ({ key: x.key, label: x.label, emphasis: x.differs })), [layout]);
  const value = useCallback((r: number, col: number) => {
    const site = grid.get(gridKey(r, col));
    if (!site) return measured ? undefined : null;
    return siteValue(results[site.index], "effect");
  }, [grid, results, measured]);
  const flagged = useCallback((r: number, col: number) => {
    const site = grid.get(gridKey(r, col));
    return !!site && flaggedKeys.has(siteKey(site));
  }, [grid, flaggedKeys]);
  const onSelect = useCallback((r: number, col: number) => {
    const site = grid.get(gridKey(r, col));
    if (site) select(selectionOfSite(site));
  }, [grid, select]);
  const selectedSite = findSite(sites, selection);
  const selectedR = selectedSite?.row ?? null;
  const selectedC = selectedSite?.col ?? null;
  const selected = useMemo(() => (selectedR !== null && selectedC !== null ? { r: selectedR, c: selectedC } : null), [selectedR, selectedC]);
  const tooltip = useCallback((r: number, col: number) => {
    const site = grid.get(gridKey(r, col));
    const res = site ? results[site.index] : undefined;
    return (
      <>
        <div style={{ fontWeight: 650 }}>{site?.label}</div>
        <div>effect {signed(res?.effect.mean, 3)}</div>
        <div style={{ opacity: 0.72 }}>CI {ci(res?.effect.lo, res?.effect.hi)}</div>
      </>
    );
  }, [grid, results]);
  return (
    <div className={c.panel}>
      <div className={c.panelHead}>
        <span className={s.panelTitle}>
          {label} · {name}
        </span>
        <Button size="small" variant="ghost" onClick={() => void openRun(id, "results")}>
          Open
        </Button>
      </div>
      <Heatmap
        rows={rows}
        cols={cols}
        value={value}
        color={color}
        theme={theme}
        cellMax={26}
        rowTitle={layout.row_title}
        tokens={layout.kind === "layer_position" && layout.cols[0]?.clean !== undefined}
        fade={running}
        flagged={flagged}
        selected={selected}
        onSelect={onSelect}
        tooltip={tooltip}
        ariaLabel={`Run ${label}`}
      />
    </div>
  );
}

function Verdict({ comparison, robust, verification }: { comparison: Comparison; robust: boolean; verification: boolean }) {
  const rho = comparison.spearman;
  const k = comparison.top_k;
  const overlap = comparison.top_overlap;
  const flagged = comparison.flagged.length;
  const stable = flagged === 0 && (rho ?? 0) > HOLDS_RHO;
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
        {comparison.paired && (
          <div>
            <div className={c.metricValue}>{comparison.n_differs ?? 0}</div>
            <div className={c.metricLabel}>sites that differ prompt by prompt: the paired difference's interval excludes zero</div>
          </div>
        )}
      </div>
      <p className={s.small}>
        {comparison.paired
          ? "Both runs measured the same prompts, so each site's effects are compared prompt by prompt, from resamples shared by every site. A paired interval is much narrower than the two runs' intervals side by side."
          : "These runs measured different prompts (or the dataset isn't pinned), so their effects can't be paired prompt by prompt; only their separate intervals are compared."}
      </p>
      <p className={s.sentence}>
        {verification ? (
          comparison.n_sign_changes === 0 ? (
            <>
              <strong>No estimate has the wrong sign.</strong> Where both are confident, patching agrees with the estimate's
              direction{rho !== null && <>, and their rankings correlate at {num(rho, 2)}</>}.
            </>
          ) : (
            <>
              <strong>
                {comparison.n_sign_changes} estimate{comparison.n_sign_changes === 1 ? " has" : "s have"} the wrong sign.
              </strong>{" "}
              Patching confidently reverses {comparison.n_sign_changes === 1 ? "it" : "them"}: trust the patched value, listed below.
            </>
          )
        ) : stable ? (
          <>
            <strong>The conclusions hold.</strong> The ranking agrees closely (a rank correlation of {num(rho, 2)}, above the{" "}
            {HOLDS_RHO} this check asks for) and no top component changes.
          </>
        ) : (
          <>
            <strong>
              {flagged} conclusion{flagged === 1 ? "" : "s"} {robust ? "depend on this choice" : "differ between the runs"}.
            </strong>{" "}
            {flagged > 0
              ? "They are listed below and flagged on the map."
              : `The overall ranking differs: a rank correlation of ${rho === null ? "unknown" : num(rho, 2)}, where the conclusions hold above ${HOLDS_RHO}.`}
          </>
        )}
      </p>
    </div>
  );
}

function flagText(f: ComparisonChange["flags"][number], topK: number): string {
  return f === "sign" ? "sign changed" : f === "left_top" ? `left the top ${topK}` : f === "entered_top" ? `entered the top ${topK}` : "differs prompt by prompt";
}

/** The selected site in both runs: what each measured there, how they differ, and a way into
 * each run at that site. */
function SelectedSite({
  siteA,
  siteB,
  resultA,
  resultB,
  comparison,
  aId,
  bId,
  flagged,
}: {
  siteA: SiteBase | null;
  siteB: SiteBase | null;
  resultA?: SiteResult;
  resultB?: SiteResult;
  comparison: Comparison | null;
  aId: string;
  bId: string;
  flagged: boolean;
}) {
  const openRun = useStore((st) => st.openRun);
  const select = useStore((st) => st.select);
  const site = siteA ?? siteB;
  if (!site) return null;
  const key = siteKey(site);
  const change = comparison?.changes.find((ch) => siteKey(ch) === key);
  const paired = change?.difference ?? null;
  const diff = siteA ? comparison?.diff.find((d) => d.index_a === siteA.index)?.value : undefined;
  const runs = [
    { label: "A", id: aId, site: siteA, result: resultA },
    { label: "B", id: bId, site: siteB, result: resultB },
  ] as const;
  return (
    <section className={c.selected} aria-label="The selected site in both runs">
      <div className={c.panelHead}>
        <h2 className={s.panelTitle}>{site.label}</h2>
        <Button size="small" variant="ghost" onClick={() => select(null)}>
          Clear selection
        </Button>
      </div>
      <div className={c.metrics}>
        {runs.map(({ label, id, site: here, result }) => (
          <div key={label}>
            <div className={c.metricValue}>{result ? signed(result.effect.mean, 3) : "—"}</div>
            <div className={c.metricLabel}>
              {!here
                ? `Run ${label} didn't measure this site.`
                : result
                  ? `Run ${label}: normalized effect, CI ${ci(result.effect.lo, result.effect.hi)}`
                  : `Run ${label} hasn't reached this site yet.`}
            </div>
            {here && (
              <Button size="small" variant="ghost" className={c.open} onClick={() => void openRun(id, "results")}>
                Open run {label} at this site
              </Button>
            )}
          </div>
        ))}
        <div>
          <div className={c.metricValue}>{paired ? signed(paired.mean, 3) : signed(diff, 3)}</div>
          <div className={c.metricLabel}>
            {paired
              ? `B − A, paired prompt by prompt, CI ${ci(paired.lo, paired.hi)}`
              : diff !== undefined
                ? "B − A, the difference of the two means"
                : comparison
                  ? "B − A: only one run measured this site."
                  : "B − A: shown once both runs have finished."}
          </div>
        </div>
      </div>
      {comparison && siteA && siteB && (
        <p className={s.small}>
          {change && change.flags.length > 0
            ? `Flagged: ${change.flags.map((f) => flagText(f, comparison.top_k)).join(", ")}.`
            : flagged
              ? "Flagged."
              : change
                ? `Among the top ${comparison.top_k} components (rank ${change.rank_a} in A, ${change.rank_b} in B), and not flagged.`
                : `Not flagged, and outside the top ${comparison.top_k} components of both runs.`}
        </p>
      )}
    </section>
  );
}

function ChangesTable({ changes, topK, selected }: { changes: ComparisonChange[]; topK: number; selected: SiteBase | null }) {
  const select = useStore((st) => st.select);
  if (changes.length === 0) return null;
  const selectedKey = selected ? siteKey(selected) : null;
  const paired = changes.some((ch) => ch.difference);
  return (
    <div className={c.changes}>
      <h2 className={s.panelTitle}>Top components in either run</h2>
      <p className={s.small}>Choose a component to see it in both runs.</p>
      <table className={s.table}>
        <thead>
          <tr>
            <th>Component</th>
            <th className={s.num}>A</th>
            <th className={s.num}>B</th>
            {paired && <th className={s.num}>B − A, paired</th>}
            <th className={s.num}>Rank</th>
            <th>Change</th>
          </tr>
        </thead>
        <tbody>
          {changes.map((ch) => {
            const choose = () =>
              select({
                layer: ch.layer,
                part: partOfKind(ch.kind),
                head: ch.head ?? undefined,
                feature: ch.feature ?? undefined,
                positionKey: ch.position_key === "all" ? undefined : ch.position_key,
                kind: isResidKind(ch.kind) ? ch.kind : undefined,
                variantKey: ch.variant_key ?? undefined,
              });
            const key = siteKey(ch);
            return (
            <tr key={key} data-clickable aria-selected={key === selectedKey} onClick={choose}>
              <td>
                <button type="button" className={c.rowButton} onClick={(e) => { e.stopPropagation(); choose(); }}>
                  {ch.label}
                </button>
              </td>
              <td className={s.num}>
                {signed(ch.effect_a.mean)} <span className="faint">[{ci(ch.effect_a.lo, ch.effect_a.hi)}]</span>
              </td>
              <td className={s.num}>
                {signed(ch.effect_b.mean)} <span className="faint">[{ci(ch.effect_b.lo, ch.effect_b.hi)}]</span>
              </td>
              {paired && (
                <td className={s.num}>
                  {ch.difference ? (
                    <>
                      {signed(ch.difference.mean)} <span className="faint">[{ci(ch.difference.lo, ch.difference.hi)}]</span>
                    </>
                  ) : (
                    <span className="faint">—</span>
                  )}
                </td>
              )}
              <td className={s.num}>
                {ch.rank_a} → {ch.rank_b}
              </td>
              <td className={ch.flags.length ? c.flagged : "faint"}>
                {ch.flags.length ? ch.flags.map((f) => flagText(f, topK)).join(", ") : "stable"}
              </td>
            </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
