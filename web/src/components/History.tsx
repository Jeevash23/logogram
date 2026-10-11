import { useEffect, useId, useMemo, useRef, useState } from "react";

import type { RunListing } from "../api/types";
import { ago, pct, plural } from "../lib/format";
import { experimentShort, scopeShort } from "../lib/spec";
import { layerProfile } from "../lib/logogram";
import { useStore } from "../store/app";
import { Logogram } from "./Logogram";
import { Button, Spinner } from "./ui";
import s from "./History.module.css";

export function History({ onOpen }: { onOpen?: () => void } = {}) {
  const runs = useStore((st) => st.runs);
  const activeRunId = useStore((st) => st.activeRunId);
  const openRun = useStore((st) => st.openRun);
  const openDraft = useStore((st) => st.openDraft);
  const live = useStore((st) => st.live);
  const projectPath = useStore((st) => st.project?.path);
  const [checked, setChecked] = useState<string[]>([]);
  useEffect(() => setChecked([]), [projectPath]); // ticks belong to the project they were made in

  // Variants (robustness checks, reruns) sit under the run they came from.
  const ordered = useMemo(() => {
    const children = new Map<string, RunListing[]>();
    const roots: RunListing[] = [];
    const ids = new Set(runs.map((r) => r.id));
    for (const r of runs) {
      const parent = r.derived_from?.run;
      if (parent && ids.has(parent)) {
        const list = children.get(parent) ?? [];
        list.push(r);
        children.set(parent, list);
      } else roots.push(r);
    }
    const out: { run: RunListing; depth: number }[] = [];
    const visited = new Set<string>();
    const walk = (r: RunListing, depth: number) => {
      if (visited.has(r.id)) return;
      visited.add(r.id);
      out.push({ run: r, depth: Math.min(depth, 6) });
      for (const c of children.get(r.id) ?? []) walk(c, depth + 1);
    };
    roots.forEach((r) => walk(r, 0));
    // A shared project's provenance can contain a cycle. Keep every run reachable.
    runs.forEach((r) => walk(r, 0));
    return out;
  }, [runs]);

  const toggle = (id: string) =>
    setChecked((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur.slice(-1), id]));

  const open = async (run: RunListing) => {
    if (run.status === "draft") {
      await openDraft(run.id);
      onOpen?.();
      return;
    }
    await openRun(run.id, run.status === "finished" || run.status === "running" ? "results" : undefined);
    onOpen?.();
  };

  const compare = () => {
    if (checked.length !== 2) return;
    useStore.setState({ compareIds: [checked[0], checked[1]] });
    useStore.getState().setView("compare");
  };

  return (
    <div className={s.history}>
      <div className={s.header}>
        <span className={s.title}>History</span>
        <span className={s.count}>{runs.length || ""}</span>
        <div className={s.spacer} />
        {checked.length > 0 && (
          <Button size="small" variant={checked.length === 2 ? "primary" : "secondary"} disabled={checked.length !== 2} onClick={compare}>
            {checked.length === 2 ? "Compare" : "Pick one more"}
          </Button>
        )}
      </div>
      <QueuedRuns />
      {runs.length === 0 ? (
        <p className={s.empty}>Runs appear here as you start them. Each one is a folder in experiments/.</p>
      ) : (
        <ul className={s.list}>
          {ordered.map(({ run, depth }) => {
            const progress = live[run.id]?.progress;
            const status = run.status === "running" || live[run.id]?.status === "running" ? "running" : run.status;
            return (
              <li key={run.id} className={s.row} style={{ paddingLeft: 8 + depth * 14 }}>
                {depth > 0 && <span className={s.branch} style={{ left: depth * 14 - 2 }} aria-hidden="true" />}
                <input
                  type="checkbox"
                  className={s.check}
                  checked={checked.includes(run.id)}
                  onChange={() => toggle(run.id)}
                  disabled={run.status !== "finished"}
                  aria-label={`Select ${run.name} to compare`}
                />
                <button
                  type="button"
                  className={s.item}
                  aria-current={run.id === activeRunId ? "true" : undefined}
                  onClick={() => void open(run)}
                >
                  <RunGlyph run={run} />
                  <span className={s.name}>
                    {run.derived_from?.kind === "robustness" ? "Robustness · " : ""}
                    {run.derived_from?.kind === "robustness" ? (run.derived_from.change ?? run.name) : run.name}
                  </span>
                  <span className={s.state}>
                    {status === "running" ? (
                      <>
                        <Spinner />
                        {progress ? pct(progress.done / progress.total) : ""}
                      </>
                    ) : status === "draft" ? (
                      "not run"
                    ) : status === "finished" ? (
                      ago(run.finished ?? run.created)
                    ) : (
                      status
                    )}
                  </span>
                  <span className={s.meta}>
                    {experimentShort(run.experiment)} · {scopeShort(run.scope)}
                    {run.n_prompts ? ` · ${plural(run.n_prompts, "prompt")}` : ""}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

/** The runs waiting for the job in progress to end, first in line first. Each can be taken out
 * of the queue until it starts. */
function QueuedRuns() {
  const queue = useStore((st) => st.queue);
  const remove = useStore((st) => st.removeQueuedRun);
  const list = useRef<HTMLOListElement>(null);
  const label = useId();
  if (queue.length === 0) return null;

  const removeAt = async (index: number, id: string) => {
    const focused = list.current?.contains(document.activeElement) ?? false;
    await remove(id);
    // Keep the keyboard in the list: on the run that took this one's place, or the one before it.
    if (focused) {
      requestAnimationFrame(() => {
        const buttons = list.current?.querySelectorAll("button");
        if (buttons?.length) buttons[Math.min(index, buttons.length - 1)].focus();
      });
    }
  };

  return (
    <div className={s.queued}>
      <div className={s.groupHeader}>
        <span className={s.groupTitle} id={label}>
          Queued
        </span>
        <span className={s.count}>{queue.length}</span>
      </div>
      <ol className={s.queuedList} aria-labelledby={label} ref={list}>
        {queue.map((q, i) => (
          <li key={q.id} className={s.queuedRow}>
            <span className={s.name}>{q.title}</span>
            <span className={s.state}>{i === 0 ? "next" : ""}</span>
            <span className={s.meta}>
              {experimentShort(q.spec.experiment)} · {scopeShort(q.spec.scope)}
            </span>
            <Button
              size="small"
              variant="ghost"
              icon="close"
              className={s.remove}
              aria-label={`Remove “${q.title}” from the queue`}
              title="Remove from the queue"
              onClick={() => void removeAt(i, q.id)}
            />
          </li>
        ))}
      </ol>
    </div>
  );
}

/** A run's logogram in the list: written by its results, or by the layers streamed so far. */
function RunGlyph({ run }: { run: RunListing }) {
  const live = useStore((st) => st.live[run.id]);
  const profile = useMemo(() => {
    // Sets of sites belong to no one layer: their glyph marks the run's identity only.
    if (run.layout_kind === "site_sets" || run.scope.kind === "site_sets") return null;
    if (run.profile) return run.profile;
    if (!live || live.sites.length === 0) return null;
    const layers = live.model?.n_layers ?? Math.max(...live.sites.map((x) => x.layer)) + 1;
    return layerProfile(live.sites, (i) => live.cells[i]?.effect.mean ?? null, layers);
  }, [run.profile, live]);
  return <Logogram seed={run.id} profile={profile} size={30} className={s.glyph} />;
}
