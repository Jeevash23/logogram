import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";

import type { ExperimentSpec, Layout, Side, SiteBase, SiteResult, Spec, SteeringFeature, Summary } from "../api/types";
import { Heatmap, type Axis } from "../components/Heatmap/Heatmap";
import { ScaleBar } from "../components/Heatmap/ScaleBar";
import { Logogram } from "../components/Logogram";
import { Button, Callout, Checkbox, Empty, Icon, menuClasses, Progress, Segmented } from "../components/ui";
import { divergingScale, niceBound, SCALE_FLOOR, type ColorScale, type ResolvedTheme } from "../lib/color";
import { ago, capitalize, ci, count, duration, num, pct, plural, qText, shortRevision, signed } from "../lib/format";
import { modelName, siteValue, useActiveRun, useRunProfile } from "../lib/hooks";
import { findSite, gridKey, layoutTitle, selectionOfSite, siteGrid, type Selection } from "../lib/sites";
import { metricWords } from "../lib/metrics";
import { baselineText, experimentText, measureOf, measureWords, positionText, saeWritesText } from "../lib/spec";
import { useStore } from "../store/app";
import { Distribution } from "../components/Distribution";
import { api } from "../api/client";
import { copyText } from "../components/CopyCommand";
import { downloadBlob, exportFigure } from "../lib/export";
import { environmentChanges, environmentNote } from "../lib/environment";
import { CircuitDialog } from "../components/CircuitDialog";
import { rankedSites, universeText } from "../lib/circuits";
import { membershipText, sweepsOf, sweepsOfRun } from "../lib/sweeps";
import { CircuitResults } from "./CircuitResults";
import { Continuations } from "./Continuations";
import { FlagMark, Forest } from "./Forest";
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
  const environment = useStore((st) => st.environment);
  useEffect(() => useStore.getState().loadEnvironment(), []);
  const listing = runs.find((x) => x.id === run.id);
  // Where a finished run ran, against where a rerun here would run.
  const environmentText = useMemo(
    () => (run.detail?.summary ? environmentNote(environmentChanges(run.detail.manifest, run.detail.spec, environment)) : null),
    [run.detail, environment],
  );
  // A title someone gave the run is shown instead of its spec's name.
  const title = listing ? listing.title : run.detail?.listing?.title;

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
  // Running in another process: nothing streams here until it ends.
  const elsewhere = status === "running" && run.live?.status !== "running" && !!listing?.elsewhere;
  const summary = run.detail?.summary;
  const manifest = run.detail?.manifest;
  // A cancelled or failed run that kept what finished: its results are shown like a finished
  // run's, with a note saying how far it got.
  const partial = status === "cancelled" || status === "failed" ? summary?.partial ?? listing?.partial ?? null : null;
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
    status === "running" ? (elsewhere ? "Running in another process" : "Running") : status === "finished" ? `Finished ${ago(manifest?.finished_at)}` : status === "draft" ? "Not run yet" : status === "failed" ? "Failed" : "Cancelled",
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
          <h1 className={r.title}>{title || spec?.name || run.id}</h1>
          {title && spec && title !== spec.name && <p className={r.specName}>Spec name: {spec.name}</p>}
          {spec && <MethodsLine spec={spec} n={summary?.n_prompts ?? run.live?.nPrompts} clusters={summary?.statistics.clusters} />}
          {spec?.notes?.trim() && <RunNotes text={spec.notes.trim()} />}
          <SweepNote runId={run.id} />
          {environmentText && (
            <p className={r.environment} role="note">
              <Icon name="info" size={14} />
              {environmentText}
            </p>
          )}
        </div>
        <div className={r.heroActions}>
          {status === "finished" && (
            <>
              {spec?.experiment.kind === "attribution_patching" && (
                <Button
                  variant="primary"
                  disabled={job?.status === "running"}
                  onClick={() => void useStore.getState().startVerification(run.id as string, VERIFY_TOP)}
                  tip={job?.status === "running" ? "A run is in progress. This can start when it finishes." : "Patch the sites with the largest estimated effects, for real, and compare"}
                >
                  Verify top {VERIFY_TOP} by patching
                </Button>
              )}
              {ranked.length > 0 && (
                <Button onClick={() => setCircuitOpen(true)} tip="Keep the strongest sites alone, 1, 2, 4, 8 … of them, and replace the rest of the model">
                  Test the top sites as a circuit
                </Button>
              )}
              <Button onClick={() => useStore.setState({ robustnessDialogOpen: true })} icon="refresh">
                Check robustness
              </Button>
              <Button
                onClick={() => useStore.setState({ sweepDialogRun: run.id })}
                tip="Run this spec once for each of several seeds, datasets or revisions, and see which sites hold"
              >
                Sweep…
              </Button>
              {run.detail && <ExportMenu runId={run.id} detail={run.detail} />}
              <Button variant="ghost" onClick={() => { useStore.setState({ specSource: "run" }); useStore.getState().setView("spec"); }}>
                Spec
              </Button>
            </>
          )}
          {partial && run.detail && <ExportMenu runId={run.id} detail={run.detail} />}
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

      {status === "running" && !elsewhere && <RunningBanner />}
      {elsewhere && (
        <Callout title="This run is running in another process">
          Another process runs it (<code>logogram run</code>, a notebook or another Logogram), so its progress isn&rsquo;t shown here.
          Its results appear here when it finishes.
        </Callout>
      )}
      {partial && (
        <Callout
          tone={status === "failed" ? "error" : undefined}
          title={status === "failed" ? "This run failed before it finished" : "This run was cancelled before it finished"}
        >
          <strong>{partial.note}</strong> Nothing is drawn for the rest. Their statistics are computed as for a finished run.
          {(manifest?.error ?? listing?.error) && <> {manifest?.error ?? listing?.error}</>}
          {partial.left_out.length > 0 && (
            <ul className={r.leftOut} aria-label="What this run doesn't keep">
              {partial.left_out.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          )}
          {status === "failed" && run.detail?.traceback && <FailureDetails key={run.id} traceback={run.detail.traceback} />}
        </Callout>
      )}
      {status === "failed" && !partial && (
        <Callout tone="error" title="This run failed">
          {manifest?.error ?? listing?.error ?? "No error message was recorded."}
          {run.detail?.traceback && <FailureDetails key={run.id} traceback={run.detail.traceback} />}
        </Callout>
      )}
      {status === "cancelled" && !partial && (
        <Callout title="This run was cancelled">
          {manifest?.error ? `${manifest.error} ` : ""}Its spec and any dataset snapshot were kept. It has no results: either
          nothing had finished when it stopped, or it ran on a version of Logogram before 0.5, which didn't keep what
          finished.
        </Callout>
      )}
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
                  {summary.features.skip_estimate != null && (
                    <div>
                      <dt>Through the skip connection</dt>
                      <dd className="figure">{signed(summary.features.skip_estimate)}</dd>
                    </div>
                  )}
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
                <dt>{summary.steering.feature ? "Pairs that measured strength 1" : "Pairs that computed it"}</dt>
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
              {summary.steering.feature && summary.steering.feature.activations.length > 0 && (
                <div>
                  <dt>Strength 1: F{summary.steering.feature.feature}'s typical activation</dt>
                  <dd className="figure">{summary.steering.feature.activations.map((a) => num(a, 2)).join(", ")}</dd>
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
      {run.id && ranked.length > 0 && <CircuitDialog open={circuitOpen} onOpenChange={setCircuitOpen} ranked={ranked} runId={run.id} />}
      {layout && layout.kind !== "site_sets" && <div className={r.controls}>
        <Segmented label="Result values" value={metric} onChange={mapMetric => useStore.setState({ mapMetric })} options={[{ value: "effect", label: words.effect }, { value: "delta", label: words.delta }]} />
        <Segmented label="Result scale" value={scaleMode} onChange={scaleMode => useStore.setState({ scaleMode })} options={[{ value: "auto", label: "Fit" }, { value: "unit", label: "±1", disabled: metric !== "effect" }]} />
        <Checkbox checked={cellValues} onChange={(v) => useStore.setState({ cellValues: v })}>Show values</Checkbox>
        <span className={r.controlsSpacer} />
        <Button size="small" variant="ghost" icon="explore" onClick={() => { useStore.setState({ exploreMode: "atlas" }); useStore.getState().setView("explore"); }}>Show on the model map</Button>
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
              {flagged.size > 0 && (
                <span className={r.flagNote}>
                  <FlagMark /> {flagged.size} conclusion{flagged.size === 1 ? "" : "s"} changed in the robustness check
                </span>
              )}
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
              partial={!!partial}
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
      {summary?.steering?.feature && <SteeredFeature feature={summary.steering.feature} trained={summary.steering.train.length} />}
      {run.detail?.continuations && <Continuations data={run.detail.continuations} selectedRow={selectedSite?.row ?? null} />}
      {summary && <Corrections summary={summary} />}

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

/** What steering along an SAE feature added: its decoder direction where the SAE writes, with
 * strength 1 at its typical activation. */
function SteeredFeature({ feature, trained }: { feature: SteeringFeature; trained: number }) {
  const typical = feature.activations.map((a) => num(a, 3)).join(", ");
  const fired = feature.fired.reduce((a, b) => a + b, 0);
  const measured = feature.measured.reduce((a, b) => a + b, 0);
  return (
    <p className={s.small}>
      Steered along feature {feature.feature} of {feature.sae.repo}
      {feature.sae.path ? `/${feature.sae.path}` : ""}: its decoder direction, added where the SAE writes,{" "}
      {saeWritesText(feature)}. Strength 1 adds what the feature writes at its typical activation, {typical}: the mean of its
      positive activations at the steered token of the clean and corrupt prompts of the {plural(trained, "training pair")}{" "}
      (positive in {count(fired)} of {count(measured)}).
      {feature.standardized ? " The SAE standardizes its inputs, so each prompt's vector is also scaled by the spread of its own activation there." : ""}
    </p>
  );
}

/**
 * Corrections for testing many sites at once: how many sites exclude zero with a band that holds
 * for all of them together, and how many are discoveries at the Benjamini–Hochberg q-value.
 */
function Corrections({ summary }: { summary: Summary }) {
  const st = summary.statistics;
  if (st.band_level === null || st.band_level === undefined) return null;
  const alpha = Math.round((1 - st.ci) * 1000) / 1000;
  // The corrections cover every row the run reports, steering's random controls included, as the
  // server computes them and the command line counts them.
  const sites = summary.sites;
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
      <details className={r.correctionsMore}>
        <summary>What the corrections mean</summary>
        <p className={s.small}>
          With many sites, some intervals exclude zero by chance alone. A simultaneous band holds for every site at once, so a
          site whose band excludes zero still stands out after the number of sites is counted; a q-value is the share of
          false findings expected among the sites at least as strong.{st.multiple_comparisons ? ` ${st.multiple_comparisons}` : ""}
        </p>
      </details>
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
  partial,
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
  /** A run that stopped early: its missing sites weren't measured. */
  partial: boolean;
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
      missing={partial ? "Not measured" : undefined}
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
                ? `Steer ${e.apply_to} ${e.direction?.kind === "sae_feature" ? `along SAE feature ${e.direction.feature}` : `→ ${e.apply_to === "clean" ? "corrupt" : "clean"}`} (${e.coefficients.map((c) => `×${c}`.replace("-", "−")).join(", ")}${e.control ? ", random control" : ""})`
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

/** The sweeps this run is a value of, each with a way to it. */
function SweepNote({ runId }: { runId: string }) {
  const runs = useStore((st) => st.runs);
  const openSweep = useStore((st) => st.openSweep);
  const memberships = useMemo(() => sweepsOfRun(sweepsOf(runs), runId), [runs, runId]);
  if (memberships.length === 0) return null;
  return (
    <div className={r.sweeps}>
      {memberships.map(({ sweep, index }) => (
        <p key={sweep.id} className={r.sweepLine}>
          {membershipText(sweep, index)}{" "}
          <button type="button" className={r.sweepLink} onClick={() => openSweep(sweep.id)}>
            Show the sweep
          </button>
        </p>
      ))}
    </div>
  );
}

/**
 * The spec's notes, which say what was expected so the result can be read against them: three
 * lines at most until shown in full.
 */
function RunNotes({ text }: { text: string }) {
  const id = useId();
  const ref = useRef<HTMLParagraphElement>(null);
  const [open, setOpen] = useState(false);
  const [long, setLong] = useState(false);
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element || open) return;
    const measure = () => setLong(element.scrollHeight > element.clientHeight + 1);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [text, open]);
  return (
    <div className={r.notes}>
      <p ref={ref} id={id} className={open ? r.notesText : `${r.notesText} ${r.notesClamped}`}>
        <span className={r.notesLabel}>Notes</span> {text}
      </p>
      {(long || open) && (
        <button type="button" className={r.notesToggle} aria-expanded={open} aria-controls={id} onClick={() => setOpen(!open)}>
          {open ? "Show less" : "Show all notes"}
        </button>
      )}
    </div>
  );
}

/** Exports in one place: the per-prompt table, the figure, and the methods text. */
function ExportMenu({ runId, detail }: { runId: string; detail: NonNullable<ReturnType<typeof useActiveRun>["detail"]> }) {
  const kind = detail.summary?.layout.kind;
  const hasMap = !!kind && kind !== "sites" && kind !== "site_sets";
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
          <DropdownMenu.Item className={menuClasses.item} onSelect={() => void guard(async () => downloadBlob(await api.runReport(runId), `${runId}.html`))}>
            <Icon name="download" size={14} /> Download the report (HTML)
          </DropdownMenu.Item>
          <DropdownMenu.Separator className={menuClasses.separator} />
          <DropdownMenu.Item className={menuClasses.item} onSelect={() => void guard(async () => downloadBlob(await api.exportSites(runId), `${runId}-sites.csv`))}>
            <Icon name="download" size={14} /> Export per-site CSV
          </DropdownMenu.Item>
          <DropdownMenu.Item className={menuClasses.item} onSelect={() => void guard(async () => downloadBlob(await api.exportRun(runId), `${runId}.csv`))}>
            <Icon name="download" size={14} /> Export per-prompt CSV
          </DropdownMenu.Item>
          <DropdownMenu.Item className={menuClasses.item} onSelect={() => void guard(async () => downloadBlob(await api.exportRun(runId, true), `${runId}-with-prompts.csv`))}>
            <Icon name="download" size={14} /> Export per-prompt CSV with prompt text
          </DropdownMenu.Item>
          {hasMap && (
            <>
              <DropdownMenu.Item className={menuClasses.item} onSelect={() => void guard(() => exportFigure(detail, metric, scaleMode === "unit"))}>
                <Icon name="download" size={14} /> Export figure PNG
              </DropdownMenu.Item>
              <DropdownMenu.Item
                className={menuClasses.item}
                onSelect={() => void guard(async () => downloadBlob(await api.runFigure(runId, metric, scaleMode === "unit"), `${runId}.svg`))}
              >
                <Icon name="download" size={14} /> Export figure SVG
              </DropdownMenu.Item>
            </>
          )}
          <DropdownMenu.Separator className={menuClasses.separator} />
          <DropdownMenu.Item
            className={menuClasses.item}
            onSelect={() =>
              void guard(async () => {
                const { text } = await api.methods(runId);
                if (!(await copyText(text))) throw new Error("Couldn't access the clipboard. Use Download methods instead.");
                useStore.getState().notify("Methods copied.");
              })
            }
          >
            <Icon name="copy" size={14} /> Copy methods
          </DropdownMenu.Item>
          <DropdownMenu.Item
            className={menuClasses.item}
            onSelect={() =>
              void guard(async () => {
                const { text } = await api.methods(runId);
                downloadBlob(new Blob([text], { type: "text/plain;charset=utf-8" }), `${runId}-methods.txt`);
              })
            }
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
      <Progress value={p ? p.done / p.total : null} label="Progress of this run" />
    </div>
  );
}

/** A failed run's traceback, behind a disclosure, with a way to copy it into a bug report. It
 * was saved without the paths and names of the machine the run ran on. */
function FailureDetails({ traceback }: { traceback: string }) {
  const [open, setOpen] = useState(false);
  const notify = useStore((st) => st.notify);
  const copy = async () => {
    if (await copyText(traceback)) notify("Traceback copied.");
    else notify("Couldn't reach the clipboard. Select the traceback and copy it instead.", "error");
  };
  return (
    <details className={r.failure} onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>{open ? "Hide details" : "Show details"}</summary>
      <div className={r.failureBody}>
        <pre className={r.traceback} tabIndex={0} aria-label="Traceback">
          {traceback}
        </pre>
        <Button size="small" icon="copy" aria-label="Copy the traceback" onClick={() => void copy()}>
          Copy
        </Button>
      </div>
    </details>
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
