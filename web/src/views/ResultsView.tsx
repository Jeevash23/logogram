import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { useCallback, useMemo, useState } from "react";

import type { ExperimentSpec, Layout, Side, SiteBase, SiteResult, Spec, Summary } from "../api/types";
import { Heatmap, type Axis } from "../components/Heatmap/Heatmap";
import { ScaleBar } from "../components/Heatmap/ScaleBar";
import { Logogram } from "../components/Logogram";
import { Button, Callout, Checkbox, Empty, Icon, menuClasses, Progress, Segmented } from "../components/ui";
import { divergingScale, niceBound, SCALE_FLOOR, type ColorScale, type ResolvedTheme } from "../lib/color";
import { ago, capitalize, ci, count, duration, num, pct, qText, shortRevision, signed } from "../lib/format";
import { modelName, siteValue, useActiveRun, useRunProfile } from "../lib/hooks";
import { findSite, gridKey, layoutTitle, selectionOfSite, siteGrid, type Selection } from "../lib/sites";
import { metricWords } from "../lib/metrics";
import { baselineText, experimentText, measureOf, measureWords, positionText } from "../lib/spec";
import { useStore } from "../store/app";
import { Distribution } from "../components/Distribution";
import { api } from "../api/client";
import { copyText } from "../components/CopyCommand";
import { downloadBlob, exportFigure, methodsText } from "../lib/export";
import { CircuitDialog } from "../components/CircuitDialog";
import { rankedSites, universeText } from "../lib/circuits";
import { CircuitResults } from "./CircuitResults";
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
  // Stays the same array while a run streams progress, so the lists below aren't sorted again.
  const resultList = useMemo(() => Object.values(run.results), [run.results]);
  // A finished run's single heads, attention or MLP outputs, strongest first: the top of them can
  // be tested as a circuit.
  const ranked = useMemo(() => rankedSites(run.detail?.summary?.sites ?? []), [run.detail?.summary]);
  const [circuitOpen, setCircuitOpen] = useState(false);

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

  const metricSpec = summary?.metric ?? spec?.metric;
  const words = measureWords(spec?.experiment, metricSpec);
  const measured = metricWords(metricSpec);
  const attribution = measureOf(spec?.experiment) === "attribution";
  const strongest = resultList
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
          {spec && <MethodsLine spec={spec} n={summary?.n_prompts ?? run.live?.nPrompts} clusters={summary?.statistics.clusters} />}
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
              {ranked.length > 0 && (
                <Button onClick={() => setCircuitOpen(true)} title="Keep the strongest sites alone, 1, 2, 4, 8 … of them, and replace the rest of the model">
                  Test the top sites as a circuit
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
              {summary.features.chosen_on && summary.features.reported_on && (
                <>
                  <div>
                    <dt>Prompts that chose the features</dt>
                    <dd className="figure">{count(summary.features.chosen_on.length)}</dd>
                  </div>
                  <div>
                    <dt>Prompts they are reported on</dt>
                    <dd className="figure">{count(summary.features.reported_on.length)}</dd>
                  </div>
                </>
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
              {summary.steering.control && summary.steering.control.length > 0 && (
                <div>
                  <dt>Beat their random control</dt>
                  <dd className="figure">
                    {count(summary.steering.control.filter((c) => c.beats_control).length)}
                    <span className={r.figureOf}> of {count(summary.steering.control.length)}</span>
                  </dd>
                </div>
              )}
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
                <dt>Clean {measured.short}</dt>
                <dd className="figure">{signed((summary.baseline.clean.metric ?? summary.baseline.clean.logit_diff).mean)}</dd>
              </div>
              <div>
                <dt>Corrupt {measured.short}</dt>
                <dd className="figure">{signed((summary.baseline.corrupt.metric ?? summary.baseline.corrupt.logit_diff).mean)}</dd>
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
      {summary && <Corrections summary={summary} />}

      {run.id && ranked.length > 0 && <CircuitDialog open={circuitOpen} onOpenChange={setCircuitOpen} ranked={ranked} runId={run.id} />}
      {layout && layout.kind !== "site_sets" && <div className={r.controls}>
        <Segmented label="Result values" value={metric} onChange={mapMetric => useStore.setState({ mapMetric })} options={[{ value: "effect", label: words.effect }, { value: "delta", label: words.delta }]} />
        <Segmented label="Result scale" value={scaleMode} onChange={scaleMode => useStore.setState({ scaleMode })} options={[{ value: "auto", label: "Fit" }, { value: "unit", label: "±1", disabled: metric !== "effect" }]} />
        <Checkbox checked={cellValues} onChange={(v) => useStore.setState({ cellValues: v })}>Show values</Checkbox>
        <span className={r.controlsSpacer} />
        <Button size="small" variant="ghost" icon="explore" onClick={() => { useStore.setState({ exploreMode: "atlas" }); useStore.getState().setView("explore"); }}>Open model atlas</Button>
        {run.detail?.predictions && <Button size="small" variant="ghost" onClick={() => useStore.getState().setView("predictions")}>Open saved predictions</Button>}
      </div>}
      {layout?.kind === "site_sets" && (
        <CircuitResults
          sites={run.sites}
          results={run.results}
          circuit={summary?.circuit}
          universe={summary?.circuit?.universe ?? layout.universe}
          effectLabel={words.effect}
          ciLevel={run.ciLevel}
          selectedIndex={selectedSite?.index ?? null}
          onSelect={(site) => select(selectionOfSite(site))}
        />
      )}
      {layout && layout.kind !== "sites" && layout.kind !== "site_sets" && (
        <div className={r.grid}>
          <div className={r.heat}>
            <div className={r.heatHead}>
              <span className={s.panelTitle}>{layoutTitle(layout)}</span>
              <ScaleBar bound={bound} theme={theme} label={metric === "effect" ? words.effect : words.delta} />
            </div>
            <ResultsHeatmap
              layout={layout}
              sites={run.sites}
              results={run.results}
              metric={metric}
              experiment={spec?.experiment}
              measuredBy={metricSpec}
              color={color}
              theme={theme}
              flagged={flagged}
              selection={selection}
              showValues={cellValues}
              running={status === "running"}
            />
          </div>
          <div className={r.side}>
            {layout.kind === "heads" && <SweepSummary results={resultList} total={run.sites.length} label={`${capitalize(words.mean)} of each head`} />}
            <Forest ciLevel={run.ciLevel} results={resultList} selectedIndex={selectedSite?.index ?? null} onSelect={(site) => select(selectionOfSite(site))} flagged={flagged} />
          </div>
        </div>
      )}
      {layout?.kind === "sites" && (
        <Forest ciLevel={run.ciLevel} results={resultList} selectedIndex={selectedSite?.index ?? null} onSelect={(site) => select(selectionOfSite(site))} flagged={flagged} limit={50} />
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

const formatCell = (v: number) => num(v, Math.abs(v) >= 10 ? 0 : 2);

/**
 * Corrections for testing many sites at once: how many sites exclude zero with a band that holds
 * for all of them together, and how many are discoveries at the Benjamini–Hochberg q-value.
 */
function Corrections({ summary }: { summary: Summary }) {
  const st = summary.statistics;
  if (st.band_level === null || st.band_level === undefined) return null;
  const alpha = Math.round((1 - st.ci) * 1000) / 1000;
  const sites = summary.sites.filter((x) => !x.variant?.control);
  const banded = sites.filter((x) => x.band && (x.band.lo > 0 || x.band.hi < 0)).length;
  const discoveries = sites.filter((x) => x.q !== null && x.q !== undefined && x.q < alpha).length;
  const single = sites.filter((x) => x.effect.lo !== null && x.effect.hi !== null && (x.effect.lo > 0 || x.effect.hi < 0)).length;
  const unit = summary.layout.kind === "site_sets" ? "set" : "site";
  return (
    <section className={r.corrections} aria-label={`Corrected for the number of ${unit}s`}>
      <p className={r.correctionsText}>
        Corrected for {count(summary.sites.length)} {unit}s: <strong>{count(banded)}</strong> exclude zero with simultaneous
        bands and <strong>{count(discoveries)}</strong> have q below {alpha}, where {count(single)} intervals exclude zero one
        {" "}{unit} at a time.
      </p>
      <p className={s.small}>
        With many sites, some intervals exclude zero by chance alone. A simultaneous band holds for every site at once, so a
        site whose band excludes zero still stands out after the number of sites is counted; a q-value is the share of
        false findings expected among the sites at least as strong.{st.multiple_comparisons ? ` ${st.multiple_comparisons}` : ""}
      </p>
    </section>
  );
}

/**
 * The sweep as a heatmap. Each cell finds its site in a map built once per run, and the accessors
 * keep their identity while the run streams progress, so cells are drawn again only when results
 * arrive or the display changes.
 */
function ResultsHeatmap({
  layout,
  sites,
  results,
  metric,
  experiment,
  measuredBy,
  color,
  theme,
  flagged,
  selection,
  showValues,
  running,
}: {
  layout: Layout;
  sites: SiteBase[];
  results: Record<number, SiteResult>;
  metric: "effect" | "delta";
  experiment: ExperimentSpec | undefined;
  /** The run's metric, which names its values. */
  measuredBy: { kind?: string | null; target?: Side | null } | undefined;
  color: ColorScale;
  theme: ResolvedTheme;
  flagged: Set<number>;
  selection: Selection | null;
  showValues: boolean;
  running: boolean;
}) {
  const select = useStore((st) => st.select);
  const grid = useMemo(() => siteGrid(sites), [sites]);
  const rows: Axis[] = useMemo(() => layout.rows.map((x) => ({ key: x.key, label: x.label })), [layout]);
  const cols: Axis[] = useMemo(
    () =>
      layout.cols.map((x) => ({
        key: x.key,
        label: x.label,
        // Steering: strengths along the direction in ink, the random control muted.
        emphasis: layout.kind === "steering" ? !x.control : x.differs,
      })),
    [layout],
  );
  const words = useMemo(() => measureWords(experiment, measuredBy), [experiment, measuredBy]);
  const attribution = measureOf(experiment) === "attribution";
  const value = useCallback(
    (ri: number, ci_: number) => {
      const site = grid.get(gridKey(ri, ci_));
      return site ? siteValue(results[site.index], metric) : undefined;
    },
    [grid, results, metric],
  );
  const isFlagged = useCallback(
    (ri: number, ci_: number) => {
      const site = grid.get(gridKey(ri, ci_));
      return !!site && flagged.has(site.index);
    },
    [grid, flagged],
  );
  const onSelect = useCallback(
    (ri: number, ci_: number) => {
      const site = grid.get(gridKey(ri, ci_));
      if (site) select(selectionOfSite(site));
    },
    [grid, select],
  );
  const tooltip = useCallback(
    (ri: number, ci_: number) => (
      <CellTooltip site={grid.get(gridKey(ri, ci_)) ?? null} results={results} metric={metric} words={words} attribution={attribution} />
    ),
    [grid, results, metric, words, attribution],
  );
  const selectedSite = findSite(sites, selection);
  const selectedR = selectedSite?.row ?? null;
  const selectedC = selectedSite?.col ?? null;
  const selected = useMemo(() => (selectedR !== null && selectedC !== null ? { r: selectedR, c: selectedC } : null), [selectedR, selectedC]);
  const tokenAxis = layout.kind === "layer_position" && layout.cols[0]?.clean !== undefined;
  return (
    <Heatmap
      rows={rows}
      cols={cols}
      value={value}
      color={color}
      theme={theme}
      rowTitle={layout.row_title}
      colTitle={layout.kind === "layer_position" ? (tokenAxis ? "Position (clean tokens of prompt 0)" : "Named position") : layout.col_title}
      tokens={tokenAxis}
      cellMax={layout.kind === "layer_components" ? 60 : layout.kind === "heads" ? 44 : 34}
      aspect={layout.kind === "layer_components" ? 0.45 : 1}
      showValues={showValues}
      format={formatCell}
      fade={running}
      selected={selected}
      flagged={isFlagged}
      onSelect={onSelect}
      tooltip={tooltip}
      ariaLabel={`${layoutTitle(layout)} results`}
    />
  );
}

/** The method as one quiet line: every choice that can change a number, in reading order. */
function MethodsLine({ spec, n, clusters }: { spec: Spec; n: number | undefined; clusters?: number | null }) {
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
            ? `every feature of the SAE (keeping the top ${scope.top}${scope.choose_on !== null ? `, chosen on ${Math.round(scope.choose_on * 100)}% of the prompts with seed ${scope.seed} and reported on the rest` : ""})`
            : scope.kind === "site_sets"
              ? `${scope.sets.length} set${scope.sets.length === 1 ? "" : "s"} of sites${scope.universe ? ` (the rest of the model: ${universeText(scope.universe)})` : ""}`
              : `${scope.sites.length} chosen site${scope.sites.length === 1 ? "" : "s"}`;
  const position =
    scope.kind === "heads" || scope.kind === "layer_components" || scope.kind === "features"
      ? positionText(scope.position)
      : scope.kind === "layer_position"
        ? scope.positions === "each"
          ? "every position"
          : "named positions"
        : scope.kind === "site_sets"
          ? "once per set"
          : scope.sites.map((x) => positionText(x.position)).filter((v, i, a) => a.indexOf(v) === i).join(", ");
  return (
    <p className={r.methods}>
      <strong>
        {e.kind === "activation_patching"
          ? e.direction === "clean_to_corrupt" ? "Patch clean → corrupt" : "Patch corrupt → clean"
          : e.kind === "direct_logit_attribution"
            ? `Direct logit attribution of the ${e.prompts} prompts`
            : e.kind === "attribution_patching"
              ? `${experimentText(e)} (attribution patching)`
              : e.kind === "path_patching"
                ? `Path patching ${e.direction === "clean_to_corrupt" ? "clean → corrupt" : "corrupt → clean"} into ${e.receivers.map((x) => (x.kind === "logits" ? "logits" : `L${x.layer} H${x.head} ${x.input}`)).join(", ")}${e.freeze_mlps ? ", MLPs held" : ""}`
                : e.kind === "steering"
                ? `Steer ${e.apply_to} → ${e.apply_to === "clean" ? "corrupt" : "clean"} (${e.coefficients.map((c) => `×${c}`.replace("-", "−")).join(", ")}${e.control ? ", random control" : ""})`
                : `Ablate (${baselineText(e.baseline)})`}
      </strong>
      {" "}{site}{scope.kind === "site_sets" ? ", each intervened on at once" : ` at ${position}`}
      <span className={r.sep}>·</span>
      {e.kind === "direct_logit_attribution"
        ? `logit difference, as a share of ${spec.metric.normalization === "dataset_gap" ? "the mean" : "each prompt's"}`
        : `${metricWords(spec.metric).label}, normalized by ${spec.metric.normalization === "dataset_gap" ? "the dataset gap" : "each prompt's gap"}`}
      {n !== undefined && <><span className={r.sep}>·</span>n = {count(n)}</>}
      <span className={r.sep}>·</span>{Math.round(spec.statistics.ci * 100)}% CI{spec.statistics.cluster ? `, resampling ${clusters ? `${count(clusters)} clusters` : "clusters"} of prompts with the same ${spec.statistics.cluster}` : ""}, {count(spec.statistics.bootstrap)} resamples, seed {spec.statistics.seed}
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
            {res.q !== null && res.q !== undefined && <> · q {qText(res.q)}</>}
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
  const done = useMemo(() => results.filter((x) => x.effect.mean !== null), [results]);
  const pos = done.filter((x) => (x.effect.lo ?? 0) > 0);
  const neg = done.filter((x) => (x.effect.hi ?? 0) < 0);
  const byIndex = useMemo(() => new Map(done.map((x) => [x.index, x])), [done]);
  // The same array while nothing new arrives: the beeswarm is laid out again only for new values.
  const values = useMemo(() => done.map((x) => ({ index: x.index, value: x.effect.mean })), [done]);
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
        values={values}
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
