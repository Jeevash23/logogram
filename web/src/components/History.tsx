import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";

import type { RunListing } from "../api/types";
import { ago, pct, plural } from "../lib/format";
import { experimentShort, scopeShort } from "../lib/spec";
import { layerProfile } from "../lib/logogram";
import {
  deleteWarning,
  filterCounts,
  historyLabel,
  historyRows,
  matchesRun,
  methodText,
  runTitle,
  type HistoryFilter,
} from "../lib/runs";
import { useStore } from "../store/app";
import { Logogram } from "./Logogram";
import { Button, ConfirmDialog, Icon, Input, menuClasses, Segmented, Spinner } from "./ui";
import s from "./History.module.css";

export function History({ onOpen }: { onOpen?: () => void } = {}) {
  const runs = useStore((st) => st.runs);
  const openRun = useStore((st) => st.openRun);
  const openDraft = useStore((st) => st.openDraft);
  const annotate = useStore((st) => st.annotateRun);
  const deleteRun = useStore((st) => st.deleteRun);
  const projectPath = useStore((st) => st.project?.path);
  const [checked, setChecked] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<HistoryFilter>("all");
  const [renaming, setRenaming] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<RunListing | null>(null);
  const deletingAt = useRef<{ id: string; index: number } | null>(null);
  const list = useRef<HTMLUListElement>(null);
  const search = useRef<HTMLInputElement>(null);
  const found = useId();
  // Ticks, searches and filters belong to the project they were made in.
  useEffect(() => {
    setChecked([]);
    setQuery("");
    setFilter("all");
    setRenaming(null);
    setDeleting(null);
  }, [projectPath]);
  // A ticked run that was deleted, here or in another tab, is no longer one to compare, nor one to
  // ask about deleting.
  useEffect(() => {
    setChecked((cur) => (cur.every((id) => runs.some((r) => r.id === id)) ? cur : cur.filter((id) => runs.some((r) => r.id === id))));
    setDeleting((cur) => (cur && !runs.some((r) => r.id === cur.id) ? null : cur));
  }, [runs]);

  const rows = useMemo(() => historyRows(runs, filter, query), [runs, filter, query]);
  const counts = useMemo(() => filterCounts(runs), [runs]);
  const searching = query.trim() !== "";
  // Archived runs a search finds while they are hidden: say so, rather than finding nothing.
  const archivedMatches = useMemo(
    () => (searching && filter !== "archived" ? runs.filter((r) => r.archived && matchesRun(r, query)).length : 0),
    [runs, query, filter, searching],
  );

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

  const rowOf = (id: string) => list.current?.querySelector<HTMLElement>(`:scope > li[data-run="${CSS.escape(id)}"]`) ?? null;
  const focusIn = (row: HTMLElement | null | undefined) =>
    (row?.querySelector<HTMLElement>("[data-run-main]") ?? row?.querySelector<HTMLElement>("[data-run-actions]"))?.focus();
  /** The keyboard goes to the row at `index` now, or the last row if fewer are left, or to the
   * search when none are. */
  const focusRow = (index: number) =>
    requestAnimationFrame(() => {
      const items = list.current?.querySelectorAll<HTMLElement>(":scope > li");
      if (!items || items.length === 0) search.current?.focus();
      else focusIn(items[Math.min(index, items.length - 1)]);
    });
  /** After a change that can take a row out of view (archived, or unstarred while only starred
   * runs show), the keyboard stays in the list rather than falling out of it: a focused control
   * that is removed leaves the focus on the page, or on the dialog around the history. */
  const keepFocus = (index: number) =>
    requestAnimationFrame(() => {
      const active = document.activeElement;
      if (!active || active === document.body || active.matches("[role=dialog]")) focusRow(index);
    });

  const star = async (run: RunListing, index: number) => {
    await annotate(run.id, { starred: !run.starred });
    keepFocus(index);
  };
  const archive = async (run: RunListing, index: number) => {
    await annotate(run.id, { archived: !run.archived });
    keepFocus(index);
  };
  // Where the keyboard goes once a title has been edited: back to its run, once the run's line
  // is drawn again in place of the field.
  const afterRename = useRef<{ id: string; index: number } | null>(null);
  const renamed = (id: string, index: number, refocus: boolean) => {
    afterRename.current = refocus ? { id, index } : null;
    setRenaming(null);
  };
  useLayoutEffect(() => {
    const at = afterRename.current;
    if (renaming !== null || !at) return;
    afterRename.current = null;
    const row = rowOf(at.id);
    if (row) focusIn(row);
    else focusRow(at.index); // a new title the search no longer finds
  });
  const confirmDelete = async () => {
    const run = deleting;
    if (!run) return;
    setDeleting(null);
    if (await deleteRun(run.id)) focusRow(deletingAt.current?.index ?? 0);
  };

  const emptyText = searching
    ? `No ${filter === "all" ? "" : `${filter} `}runs match “${query.trim()}”.`
    : filter === "starred"
      ? "No starred runs. Star a run to find it here quickly."
      : filter === "archived"
        ? "No archived runs. Archive a run to keep it out of the history without deleting it."
        : "Every run is archived. Show the archived runs to see them.";

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
        <>
          <div className={s.tools}>
            <div className={s.search}>
              <Icon name="search" size={14} />
              <Input
                ref={search}
                type="search"
                aria-label="Search runs"
                aria-describedby={found}
                placeholder="Title, name, id, method, model or dataset"
                spellCheck={false}
                value={query}
                // Escape clears a search before it closes the history.
                data-owns-escape={query ? "" : undefined}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Escape" && query) {
                    e.preventDefault();
                    setQuery("");
                  }
                }}
              />
              {query && (
                <Button
                  size="small"
                  variant="ghost"
                  icon="close"
                  className={s.clear}
                  aria-label="Clear the search"
                  onClick={() => {
                    setQuery("");
                    search.current?.focus();
                  }}
                />
              )}
            </div>
            <Segmented
              label="Show"
              value={filter}
              onChange={setFilter}
              options={[
                { value: "all", label: <>All <span className={s.n}>{counts.all}</span></> },
                { value: "starred", label: <>Starred <span className={s.n}>{counts.starred}</span></> },
                { value: "archived", label: <>Archived <span className={s.n}>{counts.archived}</span></> },
              ]}
            />
          </div>
          <p id={found} className={s.found} aria-live="polite">
            {searching ? `${plural(rows.length, "run")} found.` : ""}
            {archivedMatches > 0 ? ` ${plural(archivedMatches, "archived run")} ${archivedMatches === 1 ? "matches" : "match"} too.` : ""}
          </p>
          {archivedMatches > 0 && (
            <Button size="small" variant="ghost" className={s.showArchived} onClick={() => setFilter("archived")}>
              Show archived runs
            </Button>
          )}
          {rows.length === 0 ? (
            <p className={s.empty}>{emptyText}</p>
          ) : (
            <ul className={s.list} aria-label="Runs" ref={list}>
              {rows.map(({ run, depth, orphan }, index) => (
                <RunRow
                  key={run.id}
                  run={run}
                  depth={depth}
                  orphan={orphan}
                  checked={checked.includes(run.id)}
                  renaming={renaming === run.id}
                  onCheck={() => toggle(run.id)}
                  onOpen={() => void open(run)}
                  onRename={() => setRenaming(run.id)}
                  onRenamed={(refocus) => renamed(run.id, index, refocus)}
                  onStar={() => void star(run, index)}
                  onArchive={() => void archive(run, index)}
                  onDelete={() => {
                    deletingAt.current = { id: run.id, index };
                    setDeleting(run);
                  }}
                />
              ))}
            </ul>
          )}
        </>
      )}
      <ConfirmDialog
        open={deleting !== null}
        onOpenChange={(value) => {
          if (!value) setDeleting(null);
        }}
        title={`Delete “${runTitle(deleting)}”?`}
        keepLabel="Keep it"
        confirmLabel={deleting?.status === "draft" ? "Delete the draft" : "Delete the run"}
        onConfirm={() => void confirmDelete()}
        returnFocus={() => {
          // Kept: back to the row's actions. Deleted: confirmDelete moves on once it is gone.
          const at = deletingAt.current;
          const actions = at ? rowOf(at.id)?.querySelector<HTMLElement>("[data-run-actions]") : null;
          actions?.focus();
        }}
      >
        {deleting ? deleteWarning(deleting, runs) : ""}
      </ConfirmDialog>
    </div>
  );
}

/** One run in the history: open it, tick it to compare, star it, and more in its menu. A run
 * whose spec can't be read says why instead, and can only be renamed, starred, archived or
 * deleted. */
function RunRow({
  run,
  depth,
  orphan,
  checked,
  renaming,
  onCheck,
  onOpen,
  onRename,
  onRenamed,
  onStar,
  onArchive,
  onDelete,
}: {
  run: RunListing;
  depth: number;
  orphan: boolean;
  checked: boolean;
  renaming: boolean;
  onCheck: () => void;
  onOpen: () => void;
  onRename: () => void;
  onRenamed: (refocus: boolean) => void;
  onStar: () => void;
  onArchive: () => void;
  onDelete: () => void;
}) {
  const live = useStore((st) => st.live[run.id]);
  const active = useStore((st) => st.activeRunId === run.id);
  const progress = live?.progress;
  const status = run.status === "running" || live?.status === "running" ? "running" : run.status;
  const label = historyLabel(run);
  return (
    <li className={s.row} style={{ paddingLeft: 8 + depth * 14 }} data-run={run.id}>
      {depth > 0 && <span className={s.branch} style={{ left: depth * 14 - 2 }} aria-hidden="true" />}
      <input
        type="checkbox"
        className={s.check}
        checked={checked}
        onChange={onCheck}
        disabled={run.status !== "finished"}
        aria-label={`Select ${label} to compare`}
      />
      {renaming ? (
        <RenameField run={run} label={label} onDone={onRenamed} />
      ) : run.damaged ? (
        <div className={`${s.item} ${s.damaged}`}>
          <RunGlyph run={run} />
          <span className={s.name}>{label}</span>
          <span className={s.state}>unreadable</span>
          <span className={s.problem}>{run.error}</span>
          {run.annotations_error && <span className={s.problem}>{run.annotations_error}</span>}
        </div>
      ) : (
        <button
          type="button"
          className={s.item}
          data-run-main=""
          aria-current={active ? "true" : undefined}
          aria-keyshortcuts="F2"
          onClick={onOpen}
          onKeyDown={(e) => {
            if (e.key === "F2" && !e.altKey && !e.ctrlKey && !e.metaKey && !e.shiftKey) {
              e.preventDefault();
              onRename();
            }
          }}
        >
          <RunGlyph run={run} />
          <span className={s.name}>{label}</span>
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
            {orphan ? "From a deleted run · " : ""}
            {methodText(run)}
            {run.n_prompts ? ` · ${plural(run.n_prompts, "prompt")}` : ""}
          </span>
          {run.annotations_error && <span className={s.problem}>{run.annotations_error}</span>}
        </button>
      )}
      {!renaming && (
        <>
          <button
            type="button"
            className={s.star}
            aria-pressed={!!run.starred}
            aria-label={`Star “${label}”`}
            onClick={onStar}
          >
            <Icon name="star" size={15} fill={run.starred ? "currentColor" : "none"} />
          </button>
          <RunMenu run={run} label={label} onRename={onRename} onStar={onStar} onArchive={onArchive} onDelete={onDelete} />
        </>
      )}
    </li>
  );
}

/** A run's actions. Rename and Delete start once the menu has closed, so the keyboard lands in
 * the title or the question rather than back on the menu's button. */
function RunMenu({
  run,
  label,
  onRename,
  onStar,
  onArchive,
  onDelete,
}: {
  run: RunListing;
  label: string;
  onRename: () => void;
  onStar: () => void;
  onArchive: () => void;
  onDelete: () => void;
}) {
  const pending = useRef<(() => void) | null>(null);
  const running = useStore((st) => st.live[run.id]?.status === "running" || (st.job?.status === "running" && st.job.run_id === run.id));
  const queued = useStore((st) => st.queue.some((q) => q.draft_id === run.id));
  const busy = run.status === "running" || running ? "Running" : queued ? "Queued" : null;
  const afterClosing = (action: () => void) => () => {
    pending.current = action;
  };
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <Button size="small" variant="ghost" icon="more" className={s.actions} data-run-actions="" aria-label={`Actions for “${label}”`} />
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          className={menuClasses.menu}
          align="end"
          sideOffset={4}
          onCloseAutoFocus={(e) => {
            const action = pending.current;
            pending.current = null;
            if (!action) return;
            e.preventDefault();
            action();
          }}
        >
          <DropdownMenu.Item className={menuClasses.item} onSelect={afterClosing(onRename)}>
            Rename…
            <span className={menuClasses.shortcut}>F2</span>
          </DropdownMenu.Item>
          <DropdownMenu.Item className={menuClasses.item} onSelect={onStar}>
            {run.starred ? "Remove the star" : "Star"}
          </DropdownMenu.Item>
          <DropdownMenu.Item className={menuClasses.item} onSelect={onArchive}>
            {run.archived ? "Move out of the archive" : "Archive"}
          </DropdownMenu.Item>
          <DropdownMenu.Separator className={menuClasses.separator} />
          <DropdownMenu.Item className={menuClasses.item} disabled={busy !== null} onSelect={afterClosing(onDelete)}>
            Delete…
            {busy && <span className={menuClasses.shortcut}>{busy}</span>}
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}

/** A run's title, edited in place. Enter keeps it, Escape leaves it as it was, and leaving the
 * field keeps what was typed. An empty title shows the spec's name again. */
function RenameField({ run, label, onDone }: { run: RunListing; label: string; onDone: (refocus: boolean) => void }) {
  const annotate = useStore((st) => st.annotateRun);
  const [value, setValue] = useState(run.title ?? run.name);
  const [saving, setSaving] = useState(false);
  const done = useRef(false);
  const hint = useId();
  const commit = async (refocus: boolean) => {
    if (done.current) return;
    done.current = true;
    const text = value.trim().split(/\s+/).join(" ");
    if (text !== (run.title ?? run.name)) {
      setSaving(true);
      await annotate(run.id, { title: text });
    }
    onDone(refocus);
  };
  return (
    <form
      className={s.rename}
      onSubmit={(e) => {
        e.preventDefault();
        void commit(true);
      }}
    >
      <RunGlyph run={run} />
      <Input
        className={s.renameInput}
        autoFocus
        value={value}
        maxLength={200}
        readOnly={saving}
        spellCheck={false}
        aria-label={`Title of “${label}”`}
        aria-describedby={hint}
        data-owns-escape=""
        onFocus={(e) => e.currentTarget.select()}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            e.preventDefault();
            if (done.current) return;
            done.current = true;
            onDone(true);
          }
        }}
        onBlur={() => void commit(false)}
      />
      <span id={hint} className={s.renameHint}>
        Enter keeps it, Esc leaves it as it was. Empty shows the spec's name{run.damaged ? "" : `, “${run.name}”`}.
      </span>
    </form>
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
  }, [run.profile, run.layout_kind, run.scope.kind, live]);
  return <Logogram seed={run.id} profile={profile} size={30} className={s.glyph} />;
}
