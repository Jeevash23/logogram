import { useMemo } from "react";

import type { SiteResult, Spec } from "../api/types";
import { Heatmap, type Axis } from "../components/Heatmap/Heatmap";
import { ScaleBar } from "../components/Heatmap/ScaleBar";
import { Button, Callout, Chip, Empty, Progress, Segmented } from "../components/ui";
import { divergingScale, niceBound, SCALE_FLOOR } from "../lib/color";
import { ago, ci, count, duration, num, pct, shortRevision, signed } from "../lib/format";
import { modelName, siteValue, useActiveRun } from "../lib/hooks";
import { findSite, layoutTitle, selectionOfSite, siteAt } from "../lib/sites";
import { baselineText, experimentText, positionText, scopeText } from "../lib/spec";
import { formFromSpec, useStore } from "../store/app";
import { Distribution } from "../components/Distribution";
import { api } from "../api/client";
import { copyText } from "../components/CopyCommand";
import { downloadBlob, exportFigure, methodsText } from "../lib/export";
import { Forest } from "./Forest";
import s from "./views.module.css";
import r from "./ResultsView.module.css";

export function ResultsView() {
  const run = useActiveRun();
  const selection = useStore((st) => st.selection);
  const select = useStore((st) => st.select);
  const metric = useStore((st) => st.mapMetric);
  const scaleMode = useStore((st) => st.scaleMode);
  const theme = useStore((st) => st.theme);
  const flags = useStore((st) => (run.id ? st.flags[run.id] : undefined));
  const runs = useStore((st) => st.runs);
  const listing = runs.find((x) => x.id === run.id);

  const bound = useMemo(() => {
    if (scaleMode === "unit" && metric === "effect") return 1;
    return niceBound(Object.values(run.results).map((x) => siteValue(x, metric)), SCALE_FLOOR[metric]);
  }, [run.results, metric, scaleMode]);
  const color = useMemo(() => divergingScale(bound, theme), [bound, theme]);
  const flagged = useMemo(() => new Set(flags?.sites ?? []), [flags]);

  if (!run.id) {
    return (
      <div className={s.view}>
        <Empty
          title="No run selected"
          action={
            <Button variant="primary" onClick={() => useStore.getState().setView("experiment")}>
              Set up an experiment
            </Button>
          }
        >
          Start an experiment, or pick a run from the history.
        </Empty>
      </div>
    );
  }
  const spec = run.detail?.spec;
  const status = run.live?.status === "running" ? "running" : listing?.status ?? (run.detail?.summary ? "finished" : "draft");
  const summary = run.detail?.summary;
  const manifest = run.detail?.manifest;
  const layout = run.layout;
  const selectedSite = findSite(run.sites, selection);

  const rows: Axis[] = layout?.rows.map((x) => ({ key: x.key, label: x.label })) ?? [];
  const cols: Axis[] =
    layout?.cols.map((x) => ({ key: x.key, label: x.label, emphasis: x.differs })) ?? [];
  const value = (ri: number, ci_: number) => {
    const site = siteAt(run.sites, ri, ci_);
    if (!site) return undefined;
    return siteValue(run.results[site.index], metric);
  };

  return (
    <div className={s.view}>
      <div className={s.head}>
        <div className={s.titleBlock}>
          <h2 className={s.title}>{spec?.name ?? run.id}</h2>
          {spec && (
            <p className={s.subtitle}>
              {experimentText(spec.experiment)} at {scopeText(spec.scope)}
              {spec.experiment.kind === "ablation" && <> · baseline {baselineText(spec.experiment.baseline)}</>}.
            </p>
          )}
        </div>
        <div className={s.headActions}>
          {status === "finished" && (
            <>
              <Button onClick={() => useStore.setState({ robustnessDialogOpen: true })} icon="refresh">
                Check robustness
              </Button>
              <Button variant="ghost" onClick={() => useStore.setState({ view: "spec", specSource: "run" })}>
                Spec
              </Button>
            </>
          )}
          {spec && (status === "failed" || status === "cancelled" || status === "draft") && (
            <Button
              variant={status === "draft" ? "primary" : "secondary"}
              onClick={() =>
                status === "draft" && run.id
                  ? void useStore.getState().openDraft(run.id)
                  : useStore.setState({ form: formFromSpec(spec), view: "experiment", analysisSource: "form" })
              }
            >
              {status === "draft" ? "Open in the form" : "Edit and rerun"}
            </Button>
          )}
        </div>
      </div>

      {spec && <MethodChips spec={spec} n={summary?.n_prompts ?? run.live?.nPrompts} />}

      {status === "running" && <RunningBanner />}
      {status === "failed" && (
        <Callout tone="error" title="This run failed">
          {manifest?.error ?? listing?.error ?? "No error message was recorded."}
        </Callout>
      )}
      {status === "cancelled" && <Callout title="This run was cancelled">Its spec and any dataset snapshot were kept. Partial results were discarded.</Callout>}
      {status === "draft" && !run.live && (
        <Callout title="This experiment hasn't run yet">Open it in the form and press Run.</Callout>
      )}

      {summary && (
        <p className={r.baseline}>
          Unpatched logit difference: clean <strong>{signed(summary.baseline.clean.logit_diff.mean)}</strong>, corrupt{" "}
          <strong>{signed(summary.baseline.corrupt.logit_diff.mean)}</strong>, gap{" "}
          <strong>{num(summary.baseline.gap.mean)}</strong>.{" "}
          <span className="faint">
            {manifest?.model && <>{modelName(manifest.model.id)} @ {shortRevision(manifest.model.revision)} · </>}
            {manifest?.device && <>{manifest.device.name} · </>}
            {manifest?.wall_time_s !== undefined && <>{duration(manifest.wall_time_s)} · </>}
            {ago(manifest?.finished_at)}
          </span>
        </p>
      )}

      {summary && run.detail && <div className={s.row} style={{ flexWrap: "wrap", marginBottom: 16 }} aria-label="Export results">
        <Button size="small" onClick={() => void useStore.getState().guard(async () => {
          const blob = await api.exportRun(run.id!);
          downloadBlob(blob, `${run.id}.csv`);
        })}>Export per-prompt CSV</Button>
        <Button size="small" onClick={() => void useStore.getState().guard(() => exportFigure(run.detail!, metric, scaleMode === "unit"))}>Export figure PNG</Button>
        <Button size="small" onClick={() => void useStore.getState().guard(async () => {
          if (!await copyText(methodsText(run.detail!))) throw new Error("Couldn't access the clipboard. Use Download methods instead.");
          useStore.getState().notify("Methods copied.");
        })}>Copy methods</Button>
        <Button size="small" variant="ghost" onClick={() => downloadBlob(new Blob([methodsText(run.detail!)], { type: "text/plain;charset=utf-8" }), `${run.id}-methods.txt`)}>Download methods</Button>
      </div>}

      {layout && <div className={s.headActions}>
        <Segmented label="Result values" value={metric} onChange={mapMetric => useStore.setState({ mapMetric })} options={[{ value: "effect", label: "Normalized effect" }, { value: "delta", label: "Δ logit diff" }]} />
        <Segmented label="Result scale" value={scaleMode} onChange={scaleMode => useStore.setState({ scaleMode })} options={[{ value: "auto", label: "Fit" }, { value: "unit", label: "±1", disabled: metric !== "effect" }]} />
        <Button size="small" onClick={() => useStore.setState({ view: "explore", exploreMode: "atlas" })}>Open model atlas</Button>
        {run.detail?.predictions && <Button size="small" onClick={() => useStore.getState().setView("predictions")}>Open saved predictions</Button>}
      </div>}
      {layout && layout.kind !== "sites" && (
        <div className={r.grid}>
          <div className={r.heat}>
            <div className={r.heatHead}>
              <span className={s.panelTitle}>{layoutTitle(layout)}</span>
              <ScaleBar bound={bound} theme={theme} label={metric === "effect" ? "Normalized effect" : "Δ logit diff"} />
            </div>
            <Heatmap
              rows={rows}
              cols={cols}
              value={value}
              color={color}
              theme={theme}
              rowTitle={layout.row_title}
              colTitle={layout.kind === "layer_position" ? (layout.cols[0]?.clean !== undefined ? "Position (clean tokens of prompt 0)" : "Named position") : layout.col_title}
              tokens={layout.kind === "layer_position" && layout.cols[0]?.clean !== undefined}
              cellMax={layout.kind === "layer_components" ? 60 : layout.kind === "heads" ? 44 : 34}
              aspect={layout.kind === "layer_components" ? 0.45 : 1}
              showValues
              format={(v) => num(v, Math.abs(v) >= 10 ? 0 : 2)}
              fade={status === "running"}
              selected={selectedSite ? { r: selectedSite.row, c: selectedSite.col } : null}
              flagged={(ri, ci_) => {
                const site = siteAt(run.sites, ri, ci_);
                return !!site && flagged.has(site.index);
              }}
              onSelect={(ri, ci_) => {
                const site = siteAt(run.sites, ri, ci_);
                if (site) select(selectionOfSite(site));
              }}
              tooltip={(ri, ci_) => <CellTooltip site={siteAt(run.sites, ri, ci_)} results={run.results} metric={metric} />}
              ariaLabel={`${layoutTitle(layout)} results`}
            />
          </div>
          <div className={r.side}>
            {layout.kind === "heads" && <SweepSummary results={Object.values(run.results)} total={run.sites.length} />}
            <Forest ciLevel={run.ciLevel} results={Object.values(run.results)} selectedIndex={selectedSite?.index ?? null} onSelect={(site) => select(selectionOfSite(site))} flagged={flagged} />
          </div>
        </div>
      )}
      {layout?.kind === "sites" && (
        <Forest ciLevel={run.ciLevel} results={Object.values(run.results)} selectedIndex={selectedSite?.index ?? null} onSelect={(site) => select(selectionOfSite(site))} flagged={flagged} limit={50} />
      )}

      {summary?.warnings.map((w) => (
        <Callout key={w} title="Note">
          {w}
        </Callout>
      ))}
      {summary && <p className={s.faint}>{summary.metric.normalized_effect}.</p>}
    </div>
  );
}

function MethodChips({ spec, n }: { spec: Spec; n: number | undefined }) {
  const e = spec.experiment;
  const scope = spec.scope;
  const position =
    scope.kind === "heads" || scope.kind === "layer_components"
      ? positionText(scope.position)
      : scope.kind === "layer_position"
        ? scope.positions === "each"
          ? "every position"
          : "named positions"
        : scope.sites.map((x) => positionText(x.position)).filter((v, i, a) => a.indexOf(v) === i).join(", ");
  const site =
    scope.kind === "heads"
      ? "head output (z)"
      : scope.kind === "layer_position"
        ? scope.site
        : scope.kind === "layer_components"
          ? scope.components.join(", ")
          : `${scope.sites.length} chosen`;
  return (
    <div className={s.chips}>
      {e.kind === "activation_patching" ? (
        <Chip k="direction">{e.direction === "clean_to_corrupt" ? "clean → corrupt" : "corrupt → clean"}</Chip>
      ) : (
        <Chip k="baseline">{baselineText(e.baseline)}</Chip>
      )}
      <Chip k="site">{site}</Chip>
      <Chip k="position">{position}</Chip>
      <Chip k="metric">
        logit diff · {spec.metric.normalization === "dataset_gap" ? "dataset gap" : "per-prompt gap"}
      </Chip>
      {n !== undefined && <Chip k="n">{count(n)}</Chip>}
      <Chip k="CI">
        {Math.round(spec.statistics.ci * 100)}% · {count(spec.statistics.bootstrap)} resamples · seed {spec.statistics.seed}
      </Chip>
      <Chip k="model" title={spec.model.revision ?? undefined}>
        {modelName(spec.model.id)} · {spec.model.dtype}
      </Chip>
    </div>
  );
}

function RunningBanner() {
  const run = useActiveRun();
  const p = run.live?.progress;
  const model = useStore((st) => st.model);
  return (
    <div className={r.running}>
      <div className={r.runningText}>
        {p
          ? `Layer ${p.layer} · ${count(p.done)} of ${count(p.total)} rows · ${pct(p.done / p.total)}`
          : model.state === "loading"
            ? `Loading ${modelName(model.id)}…`
            : "Preparing prompts…"}
      </div>
      <Progress value={p ? p.done / p.total : null} />
    </div>
  );
}

function CellTooltip({
  site,
  results,
  metric,
}: {
  site: { index: number; label: string } | null;
  results: Record<number, SiteResult>;
  metric: "effect" | "delta";
}) {
  if (!site) return null;
  const res = results[site.index];
  return (
    <>
      <div style={{ fontWeight: 650 }}>{site.label}</div>
      {res ? (
        <>
          <div>
            {metric === "effect" ? "effect" : "Δ logit diff"} <strong>{signed(siteValue(res, metric), 3)}</strong>
          </div>
          <div style={{ opacity: 0.72 }}>
            CI {metric === "effect" ? ci(res.effect.lo, res.effect.hi) : ci(res.delta.lo, res.delta.hi)} · flipped{" "}
            {res.sign_flips} of {res.n}
          </div>
        </>
      ) : (
        <div style={{ opacity: 0.72 }}>Waiting for this layer</div>
      )}
    </>
  );
}


function SweepSummary({ results, total }: { results: SiteResult[]; total: number }) {
  const select = useStore((st) => st.select);
  const done = results.filter((x) => x.effect.mean !== null);
  const pos = done.filter((x) => (x.effect.lo ?? 0) > 0);
  const neg = done.filter((x) => (x.effect.hi ?? 0) < 0);
  const byIndex = new Map(done.map((x) => [x.index, x]));
  return (
    <div className={r.summary}>
      <div className={r.summaryHead}>Across all {count(total)} heads</div>
      <dl className={r.summaryStats}>
        <dt>CI above zero</dt>
        <dd>{count(pos.length)}</dd>
        <dt>CI below zero</dt>
        <dd>{count(neg.length)}</dd>
        <dt>CI includes zero</dt>
        <dd>{count(done.length - pos.length - neg.length)}</dd>
        {done.length < total && (
          <>
            <dt>Still running</dt>
            <dd>{count(total - done.length)}</dd>
          </>
        )}
      </dl>
      <Distribution
        values={done.map((x) => ({ index: x.index, value: x.effect.mean }))}
        mean={null}
        lo={null}
        hi={null}
        label="Mean normalized effect of each head"
        pointName="Site"
        showKey={false}
        onPick={(i) => {
          const site = byIndex.get(i);
          if (site) select(selectionOfSite(site));
        }}
      />
      <p className={r.summaryNote}>Select a dot or a heatmap cell to inspect its intervention evidence.</p>
    </div>
  );
}
