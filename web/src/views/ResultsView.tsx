import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { useMemo } from "react";

import type { SiteResult, Spec } from "../api/types";
import { Heatmap, type Axis } from "../components/Heatmap/Heatmap";
import { ScaleBar } from "../components/Heatmap/ScaleBar";
import { Logogram } from "../components/Logogram";
import { Button, Callout, Checkbox, Empty, Icon, menuClasses, Progress, Segmented } from "../components/ui";
import { divergingScale, niceBound, SCALE_FLOOR } from "../lib/color";
import { ago, capitalize, ci, count, duration, num, pct, shortRevision, signed } from "../lib/format";
import { modelName, siteValue, useActiveRun, useRunProfile } from "../lib/hooks";
import { findSite, layoutTitle, selectionOfSite, siteAt } from "../lib/sites";
import { baselineText, measureOf, measureWords, positionText } from "../lib/spec";
import { useStore } from "../store/app";
import { Distribution } from "../components/Distribution";
import { api } from "../api/client";
import { copyText } from "../components/CopyCommand";
import { downloadBlob, exportFigure, methodsText } from "../lib/export";
import { Forest } from "./Forest";
import s from "./views.module.css";
import r from "./ResultsView.module.css";

/** How many of an estimate's strongest sites "Verify by patching" patches for real. */
const VERIFY_TOP = 10;

export function ResultsView() {
  const run = useActiveRun();
  const selection = useStore((st) => st.selection);
  const select = useStore((st) => st.select);
  const metric = useStore((st) => st.mapMetric);
  const scaleMode = useStore((st) => st.scaleMode);
  const theme = useStore((st) => st.theme);
  const flags = useStore((st) => (run.id ? st.flags[run.id] : undefined));
  const cellValues = useStore((st) => st.cellValues);
  const profile = useRunProfile(run);
  const runs = useStore((st) => st.runs);
  const job = useStore((st) => st.job);
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
    layout?.cols.map((x) => ({
      key: x.key,
      label: x.label,
      // Steering: strengths along the direction in ink, the random control muted.
      emphasis: layout.kind === "steering" ? !x.control : x.differs,
    })) ?? [];
  const value = (ri: number, ci_: number) => {
    const site = siteAt(run.sites, ri, ci_);
    if (!site) return undefined;
    return siteValue(run.results[site.index], metric);
  };

  const words = measureWords(spec?.experiment);
  const attribution = measureOf(spec?.experiment) === "attribution";
  const strongest = Object.values(run.results)
    .filter((x) => x.effect.mean !== null && !x.variant?.control)
    .sort((a, b) => Math.abs(b.effect.mean ?? 0) - Math.abs(a.effect.mean ?? 0))[0];
  const provenance = [
    status === "running" ? "Running" : status === "finished" ? `Finished ${ago(manifest?.finished_at)}` : status === "draft" ? "Not run yet" : status === "failed" ? "Failed" : "Cancelled",
    manifest?.wall_time_s !== undefined && manifest?.wall_time_s !== null ? duration(manifest.wall_time_s) : null,
    manifest?.device?.name ? `on ${manifest.device.name}` : null,
  ].filter(Boolean).join(" · ");

  return (
    <div className={s.view}>
      <div className={r.hero}>
        <Logogram
          seed={run.id}
          profile={profile}
          size={92}
          className={r.glyph}
          title={profile ? "This run's logogram: layers run clockwise from the top; the ink swells outward where a layer's strongest effect is positive and inward where it is negative." : undefined}
        />
        <div className={r.heroText}>
          <span className="eyebrow">{provenance}</span>
          <h1 className={r.title}>{spec?.name ?? run.id}</h1>
          {spec && <MethodsLine spec={spec} n={summary?.n_prompts ?? run.live?.nPrompts} />}
        </div>
        <div className={r.heroActions}>
          {status === "finished" && (
            <>
              {spec?.experiment.kind === "attribution_patching" && (
                <Button
                  variant="primary"
                  disabled={job?.status === "running"}
                  onClick={() => void useStore.getState().startVerification(run.id as string, VERIFY_TOP)}
                  title="Patch the sites with the largest estimated effects, for real, and compare"
                >
                  Verify top {VERIFY_TOP} by patching
                </Button>
              )}
              <Button onClick={() => useStore.setState({ robustnessDialogOpen: true })} icon="refresh">
                Check robustness
              </Button>
              {run.detail && <ExportMenu runId={run.id} detail={run.detail} />}
              <Button variant="ghost" onClick={() => { useStore.setState({ specSource: "run" }); useStore.getState().setView("spec"); }}>
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
                  : useStore.getState().editRun(spec)
              }
            >
              {status === "draft" ? "Open in the form" : "Edit and rerun"}
            </Button>
          )}
        </div>
      </div>

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
        <dl className={r.figures} aria-label="The run at a glance">
          {summary.features ? (
            <>
              <div>
                <dt>SAE variance explained</dt>
                <dd className="figure">{pct(summary.features.fit.variance_explained)}</dd>
              </div>
              {summary.features.site_estimate != null && summary.features.features_estimate != null ? (
                <>
                  <div>
                    <dt>Estimated effect of the whole site</dt>
                    <dd className="figure">{signed(summary.features.site_estimate)}</dd>
                  </div>
                  <div>
                    <dt>Of it, through the features</dt>
                    <dd className="figure">{signed(summary.features.features_estimate)}</dd>
                  </div>
                </>
              ) : (
                <div>
                  <dt>Features per token</dt>
                  <dd className="figure">{num(summary.features.fit.l0, 1)}</dd>
                </div>
              )}
            </>
          ) : summary.steering ? (
            <>
              <div>
                <dt>Held-out prompts measured</dt>
                <dd className="figure">{count(summary.steering.test.length)}</dd>
              </div>
              <div>
                <dt>Pairs that computed it</dt>
                <dd className="figure">{count(summary.steering.train.length)}</dd>
              </div>
              {(() => {
                const control = Object.values(run.results)
                  .filter((x) => x.variant?.control && x.effect.mean !== null)
                  .sort((a, b) => Math.abs(b.effect.mean ?? 0) - Math.abs(a.effect.mean ?? 0))[0];
                return control ? (
                  <div>
                    <dt>Largest random-control effect</dt>
                    <dd className="figure">{signed(control.effect.mean, 2)}</dd>
                  </div>
                ) : null;
              })()}
            </>
          ) : summary.direct ? (
            <>
              <div>
                <dt>{summary.direct.prompts === "clean" ? "Clean" : "Corrupt"} logit diff</dt>
                <dd className="figure">{signed(summary.direct.logit_diff)}</dd>
              </div>
              <div>
                <dt>Attention writes</dt>
                <dd className="figure">{signed(summary.direct.attention)}</dd>
              </div>
              <div>
                <dt>MLPs write</dt>
                <dd className="figure">{signed(summary.direct.mlp)}</dd>
              </div>
              <div>
                <dt>Embeddings and biases</dt>
                <dd className="figure">{signed((summary.direct.embeddings ?? 0) + (summary.direct.biases ?? 0))}</dd>
              </div>
            </>
          ) : (
            <>
              <div>
                <dt>Clean logit diff</dt>
                <dd className="figure">{signed(summary.baseline.clean.logit_diff.mean)}</dd>
              </div>
              <div>
                <dt>Corrupt logit diff</dt>
                <dd className="figure">{signed(summary.baseline.corrupt.logit_diff.mean)}</dd>
              </div>
              <div>
                <dt>Gap the effects are measured against</dt>
                <dd className="figure">{num(summary.baseline.gap.mean)}</dd>
              </div>
            </>
          )}
          {strongest && (
            <div className={r.wide}>
              <dt>{attribution ? "Largest direct effect" : "Strongest effect"}</dt>
              <dd className="figure">
                <button type="button" className={r.figureLink} onClick={() => select(selectionOfSite(strongest))}>
                  <span className={r.swatch} style={{ background: color(siteValue(strongest, metric) ?? 0) }} aria-hidden="true" />
                  {strongest.label} <span className={r.figureValue}>{signed(strongest.effect.mean, 2)}</span>
                </button>
              </dd>
            </div>
          )}
        </dl>
      )}

      {layout && <div className={r.controls}>
        <Segmented label="Result values" value={metric} onChange={mapMetric => useStore.setState({ mapMetric })} options={[{ value: "effect", label: words.effect }, { value: "delta", label: words.delta }]} />
        <Segmented label="Result scale" value={scaleMode} onChange={scaleMode => useStore.setState({ scaleMode })} options={[{ value: "auto", label: "Fit" }, { value: "unit", label: "±1", disabled: metric !== "effect" }]} />
        <Checkbox checked={cellValues} onChange={(v) => useStore.setState({ cellValues: v })}>Show values</Checkbox>
        <span className={r.controlsSpacer} />
        <Button size="small" variant="ghost" icon="explore" onClick={() => { useStore.setState({ exploreMode: "atlas" }); useStore.getState().setView("explore"); }}>Open model atlas</Button>
        {run.detail?.predictions && <Button size="small" variant="ghost" onClick={() => useStore.getState().setView("predictions")}>Open saved predictions</Button>}
      </div>}
      {layout && layout.kind !== "sites" && (
        <div className={r.grid}>
          <div className={r.heat}>
            <div className={r.heatHead}>
              <span className={s.panelTitle}>{layoutTitle(layout)}</span>
              <ScaleBar bound={bound} theme={theme} label={metric === "effect" ? words.effect : words.delta} />
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
              showValues={cellValues}
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
              tooltip={(ri, ci_) => <CellTooltip site={siteAt(run.sites, ri, ci_)} results={run.results} metric={metric} words={words} attribution={attribution} />}
              ariaLabel={`${layoutTitle(layout)} results`}
            />
          </div>
          <div className={r.side}>
            {layout.kind === "heads" && <SweepSummary results={Object.values(run.results)} total={run.sites.length} label={`${capitalize(words.mean)} of each head`} />}
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

/** The method as one quiet line: every choice that can change a number, in reading order. */
function MethodsLine({ spec, n }: { spec: Spec; n: number | undefined }) {
  const e = spec.experiment;
  const scope = spec.scope;
  const site =
    scope.kind === "heads"
      ? "each head's output (z)"
      : scope.kind === "layer_position"
        ? scope.site.replace("_", " ")
        : scope.kind === "layer_components"
          ? scope.components.map((c) => c.replace("_", " ")).join(", ")
          : scope.kind === "features"
            ? `every feature of the SAE (keeping the top ${scope.top})`
            : `${scope.sites.length} chosen site${scope.sites.length === 1 ? "" : "s"}`;
  const position =
    scope.kind === "heads" || scope.kind === "layer_components" || scope.kind === "features"
      ? positionText(scope.position)
      : scope.kind === "layer_position"
        ? scope.positions === "each"
          ? "every position"
          : "named positions"
        : scope.sites.map((x) => positionText(x.position)).filter((v, i, a) => a.indexOf(v) === i).join(", ");
  return (
    <p className={r.methods}>
      <strong>
        {e.kind === "activation_patching"
          ? e.direction === "clean_to_corrupt" ? "Patch clean → corrupt" : "Patch corrupt → clean"
          : e.kind === "direct_logit_attribution"
            ? `Direct logit attribution of the ${e.prompts} prompts`
            : e.kind === "attribution_patching"
              ? `Estimate patching ${e.direction === "clean_to_corrupt" ? "clean → corrupt" : "corrupt → clean"} (attribution patching)`
              : e.kind === "path_patching"
                ? `Path patching ${e.direction === "clean_to_corrupt" ? "clean → corrupt" : "corrupt → clean"} into ${e.receivers.map((x) => (x.kind === "logits" ? "logits" : `L${x.layer} H${x.head} ${x.input}`)).join(", ")}${e.freeze_mlps ? ", MLPs held" : ""}`
                : e.kind === "steering"
                ? `Steer ${e.apply_to} → ${e.apply_to === "clean" ? "corrupt" : "clean"} (${e.coefficients.map((c) => `×${c}`.replace("-", "−")).join(", ")}${e.control ? ", random control" : ""})`
                : `Ablate (${baselineText(e.baseline)})`}
      </strong>
      {" "}{site} at {position}
      <span className={r.sep}>·</span>
      {e.kind === "direct_logit_attribution"
        ? `logit difference, as a share of ${spec.metric.normalization === "dataset_gap" ? "the mean" : "each prompt's"}`
        : `logit difference, normalized by ${spec.metric.normalization === "dataset_gap" ? "the dataset gap" : "each prompt's gap"}`}
      {n !== undefined && <><span className={r.sep}>·</span>n = {count(n)}</>}
      <span className={r.sep}>·</span>{Math.round(spec.statistics.ci * 100)}% CI, {count(spec.statistics.bootstrap)} resamples, seed {spec.statistics.seed}
      <span className={r.sep}>·</span>{modelName(spec.model.id)}{spec.model.revision ? ` @ ${shortRevision(spec.model.revision)}` : ""}, {spec.model.dtype}
    </p>
  );
}

/** Exports in one place: the per-prompt table, the figure, and the methods text. */
function ExportMenu({ runId, detail }: { runId: string; detail: NonNullable<ReturnType<typeof useActiveRun>["detail"]> }) {
  const metric = useStore((st) => st.mapMetric);
  const scaleMode = useStore((st) => st.scaleMode);
  const guard = useStore((st) => st.guard);
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <Button iconAfter="chevronDown">Export</Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className={menuClasses.menu} sideOffset={6} align="end">
          <DropdownMenu.Item className={menuClasses.item} onSelect={() => void guard(async () => downloadBlob(await api.exportRun(runId), `${runId}.csv`))}>
            <Icon name="download" size={14} /> Export per-prompt CSV
          </DropdownMenu.Item>
          <DropdownMenu.Item className={menuClasses.item} onSelect={() => void guard(() => exportFigure(detail, metric, scaleMode === "unit"))}>
            <Icon name="download" size={14} /> Export figure PNG
          </DropdownMenu.Item>
          <DropdownMenu.Separator className={menuClasses.separator} />
          <DropdownMenu.Item
            className={menuClasses.item}
            onSelect={() =>
              void guard(async () => {
                if (!(await copyText(methodsText(detail)))) throw new Error("Couldn't access the clipboard. Use Download methods instead.");
                useStore.getState().notify("Methods copied.");
              })
            }
          >
            <Icon name="copy" size={14} /> Copy methods
          </DropdownMenu.Item>
          <DropdownMenu.Item
            className={menuClasses.item}
            onSelect={() => downloadBlob(new Blob([methodsText(detail)], { type: "text/plain;charset=utf-8" }), `${runId}-methods.txt`)}
          >
            <Icon name="download" size={14} /> Download methods
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
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
  words,
  attribution,
}: {
  site: { index: number; label: string } | null;
  results: Record<number, SiteResult>;
  metric: "effect" | "delta";
  words: ReturnType<typeof measureWords>;
  attribution: boolean;
}) {
  if (!site) return null;
  const res = results[site.index];
  return (
    <>
      <div style={{ fontWeight: 650 }}>{site.label}</div>
      {res ? (
        <>
          <div>
            {metric === "effect" ? words.effect : words.delta} <strong>{signed(siteValue(res, metric), 3)}</strong>
          </div>
          <div style={{ opacity: 0.72 }}>
            CI {metric === "effect" ? ci(res.effect.lo, res.effect.hi) : ci(res.delta.lo, res.delta.hi)}
            {!attribution && <> · flipped {res.sign_flips} of {res.n}</>}
          </div>
        </>
      ) : (
        <div style={{ opacity: 0.72 }}>Waiting for this layer</div>
      )}
    </>
  );
}


function SweepSummary({ results, total, label }: { results: SiteResult[]; total: number; label: string }) {
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
        label={label}
        pointName="Site"
        showKey={false}
        onPick={(i) => {
          const site = byIndex.get(i);
          if (site) select(selectionOfSite(site));
        }}
      />
      <p className={r.summaryNote}>Select a dot or a heatmap cell to inspect its per-prompt evidence.</p>
    </div>
  );
}
