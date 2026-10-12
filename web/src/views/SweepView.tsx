import { useCallback, useEffect, useMemo, useState } from "react";

import { api } from "../api/client";
import type { SweepSite, SweepTable, SweepValueRow } from "../api/types";
import { ScaleBar } from "../components/Heatmap/ScaleBar";
import { SweepChart, type ChartCell, type ChartValue } from "../components/SweepChart/SweepChart";
import { Button, Callout, Empty, Field, Segmented, Select } from "../components/ui";
import { divergingScale, niceBound, SCALE_FLOOR } from "../lib/color";
import { downloadBlob } from "../lib/export";
import { capitalize, ci, count, num, plural, shortRevision, signed } from "../lib/format";
import { selectionOfSite } from "../lib/sites";
import { measureWords } from "../lib/spec";
import { fieldWords, queuedValues, statusText, sweepsOf, type SweepGroup } from "../lib/sweeps";
import { useStore } from "../store/app";
import s from "./views.module.css";
import w from "./SweepView.module.css";

/** How many of the strongest sites to show; 0 shows every site. */
const TOPS = [5, 10, 20, 50, 0];
/** Up to this many values, the table has a column for each. */
const VALUE_COLUMNS = 8;

const VERDICT_WORDS: Record<SweepSite["verdict"], string> = {
  holds: "Keeps its sign and size",
  size: "Changes size",
  sign: "Flips sign",
  unknown: "Too few values",
};

/** A list in words: "0, 1, 2 and 3", or the first few and how many more. */
function listText(items: string[], most = 8): string {
  if (items.length <= 1) return items.join("");
  if (items.length > most) return `${items.slice(0, most - 1).join(", ")} … and ${items.length - most + 1} more`;
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}

/**
 * One spec run across several values of one choice: which values, their runs, and for the
 * strongest sites, each site's effect with its interval at every value, beside the spread of its
 * means across values, and whether it keeps its sign and size.
 */
export function SweepView() {
  const sweepId = useStore((st) => st.sweepId);
  const runs = useStore((st) => st.runs);
  const queue = useStore((st) => st.queue);
  const theme = useStore((st) => st.theme);
  const guard = useStore((st) => st.guard);
  const sweeps = useMemo(() => sweepsOf(runs), [runs]);
  const current = sweeps.find((x) => x.id === sweepId) ?? null;
  const [choose, setChoose] = useState<"first" | "all">("first");
  const [top, setTop] = useState(10);
  const [table, setTable] = useState<SweepTable | null>(null);
  const [error, setError] = useState<string | null>(null);
  // The selected site, by what it measures (choosing the sites another way reorders them), and
  // value.
  const [cell, setCell] = useState<{ key: string; value: number } | null>(null);

  // With no sweep chosen, the newest is shown.
  useEffect(() => {
    if (!sweepId && sweeps[0]) useStore.setState({ sweepId: sweeps[0].id });
  }, [sweepId, sweeps]);

  // Read the table again whenever one of the sweep's runs changes: it starts, ends, or is deleted.
  const version = current ? Object.entries(current.runs).map(([i, r]) => `${i}:${r.id}:${r.status}`).join(",") : "";
  useEffect(() => {
    if (!sweepId) return;
    let cancelled = false;
    api.sweep(sweepId, choose, top).then(
      (t) => {
        if (cancelled) return;
        setTable(t);
        setError(null);
      },
      (e: Error) => {
        if (cancelled) return;
        setTable(null);
        setError(e.message);
      },
    );
    return () => {
      cancelled = true;
    };
  }, [sweepId, choose, top, version]);

  const shown = table && table.sweep === sweepId ? table : null;
  const queued = useMemo(() => queuedValues(queue, sweepId ?? ""), [queue, sweepId]);
  // The server knows a sweep by its runs: until one of them starts, there is no table, only runs
  // waiting in the queue.
  const waiting = !shown && !current && queued.size > 0;
  const paused = useStore((st) => st.queueState === "paused");
  const values: ChartValue[] = useMemo(
    () => (shown?.values ?? []).map((v) => ({ label: v.label, note: statusText(v.status, queued.has(v.index)) })),
    [shown, queued],
  );
  const bound = useMemo(() => {
    const ends = (shown?.sites ?? []).flatMap((site) => [
      ...site.points.map((p) => p?.mean),
      site.spread ? site.spread.mean + (site.spread.sd ?? 0) : null,
      site.spread ? site.spread.mean - (site.spread.sd ?? 0) : null,
    ]);
    return niceBound(ends, SCALE_FLOOR.effect);
  }, [shown]);
  const color = useMemo(() => divergingScale(bound, theme), [bound, theme]);
  const effect = measureWords(shown?.experiment, shown?.metric).effect;
  const level = shown?.ci ? `${Math.round(shown.ci * 100)}%` : "";
  const pickedSite = shown && cell ? shown.sites.findIndex((x) => x.key === cell.key) : -1;
  const picked: ChartCell | null =
    shown && cell && pickedSite >= 0 && cell.value < shown.values.length ? { site: pickedSite, value: cell.value } : null;
  const pick = useCallback(
    (c: ChartCell) => {
      const site = shown?.sites[c.site];
      if (site) setCell({ key: site.key, value: c.value });
    },
    [shown],
  );

  const describe = useCallback(
    ({ site, value }: ChartCell) => {
      if (!shown) return "";
      const row = shown.values[value];
      const s0 = shown.sites[site];
      const p = s0?.points[value];
      const at = `${s0?.label} at ${shown.label} ${row?.label}`;
      if (!p || p.mean === null) {
        const why = row?.status === "finished" ? "not among that run's sites" : statusText(row?.status ?? "missing", queued.has(value)) ?? "";
        return `${at}: no results (${why}).`;
      }
      const interval = p.lo !== null && p.hi !== null ? `, ${level} interval ${ci(p.lo, p.hi, 3)}` : "";
      return `${at}: ${signed(p.mean, 3)}${interval}. ${VERDICT_WORDS[s0.verdict]}.`;
    },
    [shown, level, queued],
  );

  const openAt = useCallback(
    async ({ site, value }: ChartCell) => {
      const row = shown?.values[value];
      const s0 = shown?.sites[site];
      if (!row?.run || !s0) return;
      // At the site, chosen once the run is open: a run of another revision drops the selection.
      await useStore.getState().openRun(row.run, "results", selectionOfSite(s0));
    },
    [shown],
  );

  if (!sweepId && sweeps.length === 0) {
    return (
      <div className={s.view}>
        <div className={s.head}>
          <div className={s.titleBlock}>
            <h1 className={s.title}>Whether a finding holds across seeds, datasets or checkpoints</h1>
          </div>
        </div>
        <Empty title="No sweeps yet">
          A sweep runs one spec across several values of one choice: seeds, datasets or a model&rsquo;s revisions. Open a finished
          run&rsquo;s results and choose <strong>Sweep…</strong>, then see here which of its strongest sites keep their sign and size.
        </Empty>
      </div>
    );
  }

  const words = fieldWords(shown?.field ?? current?.field ?? "");
  const exportTable = () =>
    void guard(async () => {
      if (!sweepId) return;
      downloadBlob(await api.exportSweep(sweepId, choose, top), `sweep-${sweepId}.csv`);
    });

  return (
    <div className={s.view}>
      <div className={s.head}>
        <div className={s.titleBlock}>
          <h1 className={s.title}>
            {shown ? shown.headline : waiting ? "This sweep's runs are waiting in the queue" : error ? "This sweep can't be shown" : "Reading the sweep…"}
          </h1>
          {shown && (
            <p className={s.subtitle}>
              {shown.name}, run once for each of {plural(shown.values.length, words.label, words.plural)}:{" "}
              {listText(shown.values.map((v) => v.label))}. {shown.description}
            </p>
          )}
          {waiting && (
            <p className={s.subtitle}>
              {queued.size === 1 ? "Its run waits" : `Its ${count(queued.size)} runs wait`} in the queue
              {paused ? ", which is paused: nothing in it starts until you start it" : ""}. The sweep is shown here once its
              first run starts.
            </p>
          )}
        </div>
        <div className={s.headActions}>
          {(sweeps.length > 1 || (sweeps.length > 0 && !current)) && (
            <Select aria-label="Sweep" value={current ? (sweepId ?? "") : ""} onChange={(e) => useStore.getState().openSweep(e.target.value || null)}>
              {!current && <option value="">Choose a sweep</option>}
              {sweeps.map((x) => (
                <option key={x.id} value={x.id}>
                  {sweepOption(x)}
                </option>
              ))}
            </Select>
          )}
          {waiting && paused && <Button onClick={() => void useStore.getState().startQueue()}>Start the queue</Button>}
          <Button icon="download" disabled={!shown || shown.sites.length === 0} onClick={exportTable}>
            Export CSV
          </Button>
        </div>
      </div>
      {error && !waiting && <Callout tone="error">{error}</Callout>}
      {shown && <ValueRuns table={shown} queued={queued} />}
      {shown && (
        <div className={w.controls}>
          <Segmented
            label="Choose the sites on"
            value={choose}
            onChange={setChoose}
            options={[
              { value: "first", label: `The first value (${shown.label} ${shown.values[0]?.label ?? ""})` },
              { value: "all", label: "All values" },
            ]}
          />
          <Field label="Sites shown" className={w.top}>
            <Select value={top} onChange={(e) => setTop(Number(e.target.value))}>
              {TOPS.map((t) => (
                <option key={t} value={t}>
                  {t === 0 ? "Every site" : `The strongest ${t}`}
                </option>
              ))}
            </Select>
          </Field>
        </div>
      )}
      {shown && !shown.problem && <p className={w.chosen}>{shown.chosen}</p>}
      {shown?.problem && (
        <Callout title="Sites can't be chosen this way">
          <p>{shown.problem}</p>
          <Button size="small" className={w.calloutAction} onClick={() => setChoose("all")}>
            Choose on all values
          </Button>
        </Callout>
      )}
      {shown && shown.sites.length > 0 && (
        <section className={w.figure} aria-labelledby="sweep-figure-title">
          <div className={w.figureHead}>
            <h2 id="sweep-figure-title" className={s.panelTitle}>
              {effect} at each {words.label}
            </h2>
            <ScaleBar bound={bound} theme={theme} label={effect} />
          </div>
          <SweepChart
            values={values}
            sites={shown.sites}
            ordered={shown.kind === "revision"}
            bound={bound}
            color={color}
            selected={picked}
            onSelect={pick}
            onOpen={(c) => void openAt(c)}
            describe={describe}
            ariaLabel={`${effect} of the ${plural(shown.sites.length, "site")} at each ${words.label}`}
          />
          <Key table={shown} level={level} />
        </section>
      )}
      {shown && picked && (
        <SelectedCell table={shown} cell={picked} level={level} queued={queued} onOpen={() => void openAt(picked)} onClear={() => setCell(null)} />
      )}
      {shown && shown.sites.length > 0 && <Verdicts table={shown} />}
      {shown && shown.sites.length > 0 && (
        <SitesTable table={shown} level={level} selected={picked?.site ?? null} onSelect={(site) => pick({ site, value: picked?.value ?? 0 })} />
      )}
      {shown?.notes.map((note) => (
        <Callout key={note} title="Note">
          {note}
        </Callout>
      ))}
    </div>
  );
}

function sweepOption(sweep: SweepGroup): string {
  const words = fieldWords(sweep.field);
  return `${sweep.name} · ${plural(sweep.values.length, words.label, words.plural)}`;
}

/** Which values the sweep varied, and each value's run: open it, or what became of it. */
function ValueRuns({ table, queued }: { table: SweepTable; queued: Set<number> }) {
  const openRun = useStore((st) => st.openRun);
  const rows = (
    <table className={s.table}>
      <thead>
        <tr>
          <th scope="col">{capitalize(table.label)}</th>
          <th scope="col">Run</th>
          <th scope="col">Results</th>
        </tr>
      </thead>
      <tbody>
        {table.values.map((v) => (
          <ValueRow key={v.index} table={table} value={v} queued={queued.has(v.index)} onOpen={(run) => void openRun(run, "results")} />
        ))}
      </tbody>
    </table>
  );
  const finished = table.finished;
  const summary = `${plural(table.values.length, table.label, table.plural)} and their runs: ${count(finished)} finished`;
  return table.values.length <= 12 ? (
    <section className={w.values} aria-label="The values and their runs">
      {rows}
    </section>
  ) : (
    <details className={w.values}>
      <summary>{summary}</summary>
      {rows}
    </details>
  );
}

function ValueRow({ table, value, queued, onOpen }: { table: SweepTable; value: SweepValueRow; queued: boolean; onOpen: (run: string) => void }) {
  const note = statusText(value.status, queued);
  const pinned =
    table.kind === "revision" && value.revision && value.revision !== value.value ? ` · commit ${shortRevision(value.revision)}` : "";
  return (
    <tr>
      <th scope="row" className={`${w.rowHead} ${w.valueCell}`}>
        {value.label}
        {pinned && <span className="faint">{pinned}</span>}
      </th>
      <td>
        {value.run ? (
          <button type="button" className={w.runLink} onClick={() => onOpen(value.run as string)}>
            {value.title ?? value.run}
          </button>
        ) : (
          <span className="faint">{value.status === "deleted" ? "Deleted" : queued ? "Waiting in the queue" : "No run"}</span>
        )}
        {value.original && <span className={w.original}> · the run the sweep was made from</span>}
      </td>
      <td className={note ? w.missingCell : undefined}>
        {note ? `None: ${note}` : value.n_prompts !== null ? `${count(value.n_prompts)} prompts` : "Finished"}
        {value.error && <span className={w.error}> · {value.error}</span>}
      </td>
    </tr>
  );
}

/** What the marks are: which variability each shows. */
function Key({ table, level }: { table: SweepTable; level: string }) {
  const ordered = table.kind === "revision";
  return (
    <div className={w.key}>
      <span className={w.keyItem}>
        <svg width="14" height="18" aria-hidden="true" className={w.keyMark}>
          <line x1="7" y1="2" x2="7" y2="16" />
          <line x1="3.5" y1="2" x2="10.5" y2="2" />
          <line x1="3.5" y1="16" x2="10.5" y2="16" />
          <circle cx="7" cy="9" r="3.5" className={w.keyDot} />
        </svg>
        Each run&rsquo;s mean and its {level} interval over its prompts
      </span>
      {ordered ? (
        <span className={w.keyItem}>
          <svg width="22" height="12" aria-hidden="true" className={w.keyMark}>
            <polyline points="1,10 8,4 14,7 21,2" className={w.keyLine} />
          </svg>
          The means in the order given
        </span>
      ) : (
        <span className={w.keyItem}>
          <span className={w.keyBand} aria-hidden="true" />
          The spread across {table.plural}: the means&rsquo; mean ± 1 SD
        </span>
      )}
      <span className={w.keyItem}>
        <svg width="10" height="16" aria-hidden="true" className={w.keyMark}>
          <line x1="5" y1="1" x2="5" y2="15" className={w.keyDash} />
        </svg>
        No results at that value: missing, not zero
      </span>
      <p className={w.keyText}>{table.spread}</p>
    </div>
  );
}

/** The selected site at the selected value, with the spread across values beside a run's own
 * interval, said apart. */
function SelectedCell({
  table,
  cell,
  level,
  queued,
  onOpen,
  onClear,
}: {
  table: SweepTable;
  cell: ChartCell;
  level: string;
  queued: Set<number>;
  onOpen: () => void;
  onClear: () => void;
}) {
  const site = table.sites[cell.site];
  const value = table.values[cell.value];
  const p = site.points[cell.value];
  const at = `${table.label} ${value.label}`;
  return (
    <section className={w.selected} aria-labelledby="sweep-selected-title">
      <div className={w.selectedHead}>
        <h2 id="sweep-selected-title" className={s.panelTitle}>
          {site.label} at {at}
        </h2>
        <Button size="small" variant="ghost" onClick={onClear}>
          Clear selection
        </Button>
      </div>
      <dl className={w.selectedFacts}>
        <div>
          <dt>At {at}</dt>
          <dd className="figure">{p && p.mean !== null ? signed(p.mean, 3) : "—"}</dd>
          <dd className={w.factNote}>
            {p && p.mean !== null
              ? p.lo !== null && p.hi !== null
                ? `This run's ${level} interval over its prompts: ${ci(p.lo, p.hi, 3)}${p.n !== null ? `, n = ${count(p.n)}` : ""}.`
                : "This run has no interval: it measured one prompt."
              : `No results: ${value.status === "finished" ? "not among this run's sites" : statusText(value.status, queued.has(cell.value))}.`}
          </dd>
        </div>
        <div>
          <dt>Across the {table.plural}</dt>
          <dd className="figure">{site.spread?.sd != null ? `± ${num(site.spread.sd, 3)}` : "—"}</dd>
          <dd className={w.factNote}>
            {site.spread
              ? `The spread of the means: their SD around ${signed(site.spread.mean, 3)}, from ${signed(site.spread.min, 3)} to ${signed(site.spread.max, 3)}. No single run's interval includes it.`
              : "No value has results for this site yet."}
          </dd>
        </div>
        <div>
          <dt>Within a run</dt>
          <dd className="figure">{site.half_width !== null ? `± ${num(site.half_width, 3)}` : "—"}</dd>
          <dd className={w.factNote}>The median half-width of the runs&rsquo; own {level} intervals: what resampling one run&rsquo;s prompts allows.</dd>
        </div>
      </dl>
      <p className={s.sentence}>{site.text}</p>
      {value.run && value.status === "finished" && (
        <div>
          <Button size="small" onClick={onOpen}>
            Open the {at} run at {site.label}
          </Button>
        </div>
      )}
    </section>
  );
}

/** Plainly: how many of the sites keep their sign and size, and how many don't, and why. */
function Verdicts({ table }: { table: SweepTable }) {
  const { holds, size, sign, unknown } = table.counts;
  const parts = [
    `${count(holds)} keep${holds === 1 ? "s" : ""} ${holds === 1 ? "its" : "their"} sign and size at every ${table.label} that measured ${holds === 1 ? "it" : "them"}`,
    size > 0 ? `${count(size)} change${size === 1 ? "s" : ""} size: two of ${size === 1 ? "its" : "their"} intervals exclude each other` : null,
    sign > 0 ? `${count(sign)} flip${sign === 1 ? "s" : ""} sign: ${sign === 1 ? "its interval lies" : "their intervals lie"} above zero at one value and below it at another` : null,
    unknown > 0 ? `${count(unknown)} ${unknown === 1 ? "was" : "were"} measured at too few values to compare` : null,
  ].filter(Boolean);
  return (
    <p className={s.sentence}>
      <strong>Of the {plural(table.sites.length, "site")} shown,</strong> {parts.join("; ")}.
    </p>
  );
}

/** Every shown site: whether it holds, the spread of its means beside a run's own interval, and
 * (for a few values) its mean at each. */
function SitesTable({
  table,
  level,
  selected,
  onSelect,
}: {
  table: SweepTable;
  level: string;
  selected: number | null;
  onSelect: (site: number) => void;
}) {
  const columns = table.values.length <= VALUE_COLUMNS;
  return (
    <section className={w.sites} aria-labelledby="sweep-sites-title">
      <h2 id="sweep-sites-title" className={s.panelTitle}>
        The sites across the {table.plural}
      </h2>
      <div className={w.tableScroll}>
        <table className={s.table}>
          <thead>
            <tr>
              <th scope="col">Site</th>
              <th scope="col">Across the {table.plural}</th>
              <th scope="col" className={s.num}>
                Spread of the means (SD)
              </th>
              <th scope="col" className={s.num}>
                A run&rsquo;s {level} interval (± median)
              </th>
              {columns &&
                table.values.map((v) => (
                  <th key={v.index} scope="col" className={s.num}>
                    {v.label}
                  </th>
                ))}
            </tr>
          </thead>
          <tbody>
            {table.sites.map((site, i) => (
              <tr key={site.key} data-clickable aria-selected={i === selected} onClick={() => onSelect(i)}>
                <th scope="row" className={w.rowHead}>
                  <button
                    type="button"
                    className={w.runLink}
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelect(i);
                    }}
                  >
                    {site.label}
                  </button>
                </th>
                <td className={site.verdict === "sign" || site.verdict === "size" ? w.flagged : undefined}>{VERDICT_WORDS[site.verdict]}</td>
                <td className={s.num}>{site.spread?.sd != null ? num(site.spread.sd, 3) : "—"}</td>
                <td className={s.num}>{site.half_width !== null ? num(site.half_width, 3) : "—"}</td>
                {columns &&
                  site.points.map((p, j) => (
                    <td key={j} className={s.num}>
                      {p && p.mean !== null ? (
                        <>
                          {signed(p.mean)} <span className="faint">[{ci(p.lo, p.hi)}]</span>
                        </>
                      ) : (
                        <span className="faint">missing</span>
                      )}
                    </td>
                  ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!columns && <p className={s.small}>Export the CSV for every site at every value.</p>}
    </section>
  );
}
