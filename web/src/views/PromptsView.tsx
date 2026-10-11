import { Fragment, useDeferredValue, useEffect, useId, useMemo, useRef, useState, type KeyboardEvent } from "react";

import { api } from "../api/client";
import type { Answer, DatasetDetail, PromptRecord, RunDetail, TaskDatasetCreated, TokenStripData } from "../api/types";
import { TokenGrid } from "../components/TokenStrip";
import { Button, Callout, ConfirmDialog, Dialog, Field, Icon, Input, Segmented, TextArea, Tip } from "../components/ui";
import { answerText, count, plural, visibleToken } from "../lib/format";
import { modelName, useAnalysisContext } from "../lib/hooks";
import {
  deriveRequest,
  draftOf,
  editedRecords,
  editsText,
  hasEdits,
  nameProblem,
  noEdits,
  recordOf,
  sameRecord,
  setMembers,
  spanText,
  suggestName,
  withEdit,
  withRemoved,
  type AnswerDraft,
  type PromptEdits,
  type RecordDraft,
  type SpanDraft,
} from "../lib/promptEdits";
import {
  PAGE_SIZE,
  foundByMeta,
  matchText,
  matchTexts,
  metaText,
  pageOfRow,
  pageRows,
  rangeText,
  rowAfterKey,
  searchQuery,
  searchText,
  type RowPage,
} from "../lib/promptRows";
import { useStore } from "../store/app";
import p from "./PromptsView.module.css";
import { TaskGenerator } from "./TaskGenerator";
import s from "./views.module.css";

type Mode = "none" | "pair" | "task" | "import";

/** Run a form action; return its result, or record its error to show next to the form. */
async function attempt<T>(fn: () => Promise<T>, setError: (e: string | null) => void): Promise<T | undefined> {
  setError(null);
  try {
    return await fn();
  } catch (e) {
    setError(e instanceof Error ? e.message : String(e));
    return undefined;
  }
}

/** A run's saved copy of the prompts it read (datasets/snapshots/<sha256>.jsonl). */
function isSnapshot(path: string): boolean {
  return path.startsWith("datasets/snapshots/");
}

/** What the view is for, or what it shows: the prompt pairs in the dataset chosen. */
function titleOf(dataset: DatasetDetail | null): string {
  if (!dataset) return "The prompt pairs a run measures";
  return `${plural(dataset.n, "prompt pair")} ${isSnapshot(dataset.path) ? "saved with this run" : `in ${dataset.name}`}`;
}

export function PromptsView() {
  const project = useStore((st) => st.project);
  const datasetPath = useStore((st) => st.datasetPath);
  const dataset = useStore((st) => st.dataset);
  const datasetEdits = useStore((st) => st.datasetEdits);
  const selectDataset = useStore((st) => st.selectDataset);
  const datasets = project?.datasets ?? [];
  const [mode, setMode] = useState<Mode>(datasets.length ? "none" : "task");

  const created = async (path: string, task?: TaskDatasetCreated) => {
    await useStore.getState().refreshProject();
    await selectDataset(path);
    setMode("none");
    // A task's answers may need another metric than the form's: the notice offers it.
    useStore.getState().datasetSaved(path, task?.metric);
  };

  return (
    <div className={s.view}>
      <div className={s.head}>
        <div className={s.titleBlock}>
          <h1 className={s.title}>{titleOf(dataset)}</h1>
          <p className={s.subtitle}>
            Each prompt is a pair: a clean prompt where the model shows the behavior, and a corrupt prompt where it
            shouldn't. The experiment's metric compares the answer with a distractor, such as their logit difference at the
            last token. An answer can be one token, a set of tokens any of which counts, or several tokens read in order.
          </p>
        </div>
      </div>

      <div className={s.row} style={{ alignItems: "center" }}>
        <div className={s.tabsInline} role="group" aria-label="Datasets">
          {datasets.map((d) => {
            const button = (
            <button
              key={d.path}
              type="button"
              className={s.datasetButton}
              aria-pressed={d.path === datasetPath}
              onClick={() => void selectDataset(d.path)}
            >
              {d.name}
              <span>
                {d.error ? "unreadable" : plural(d.n ?? 0, "prompt")}
                {hasEdits(datasetEdits[d.path]) && " · unsaved edits"}
              </span>
            </button>
            );
            // Why a file can't be read, on hover and on keyboard focus.
            return d.error ? <Tip key={d.path} text={d.error}>{button}</Tip> : button;
          })}
        </div>
        <div style={{ flex: 1 }} />
        <Segmented<Mode>
          label="Add prompts"
          value={mode}
          onChange={(m) => setMode(m === mode ? "none" : m)}
          options={[
            { value: "pair", label: "Single pair" },
            { value: "task", label: "Generate a task" },
            { value: "import", label: "Import JSONL" },
          ]}
        />
      </div>

      {mode === "pair" && <PairForm onCreated={created} />}
      {mode === "task" && <TaskGenerator onCreated={(path, task) => void created(path, task)} />}
      {mode === "import" && <ImportForm onCreated={created} />}

      {dataset && datasetPath && <DatasetPrompts key={datasetPath} path={datasetPath} dataset={dataset} onSaved={created} />}
    </div>
  );
}

/** How the loaded model tokenizes a prompt pair, read as the experiment form reads prompts (its
 * BOS choice, and its metric's kind, which says whether answers of several tokens can be read).
 * Null without a model, or while the pair isn't complete. */
function useTokenPreview(record: PromptRecord | null): TokenStripData | null {
  const [preview, setPreview] = useState<TokenStripData | null>(null);
  const modelReady = useStore((st) => st.model.state === "ready");
  const context = useAnalysisContext();
  const { options, tokenMetric } = context;
  const key = record ? JSON.stringify(record) : "";

  useEffect(() => {
    if (!modelReady || !record) return;
    let current = true;
    const t = window.setTimeout(() => {
      api.tokenize({ record, metric: tokenMetric, ...options }).then(
        (data) => current && setPreview(data),
        () => current && setPreview(null),
      );
    }, 250);
    return () => {
      current = false;
      window.clearTimeout(t);
    };
    // The record's key stands for the record, and the context's key for its options.
  }, [key, modelReady, context.key, tokenMetric.kind]);
  return modelReady && record ? preview : null;
}

function PairForm({ onCreated }: { onCreated: (path: string) => void }) {
  const [record, setRecord] = useState<PromptRecord>({
    clean: "When Mary and John went to the store, John gave a drink to",
    corrupt: "When Mary and John went to the store, Mary gave a drink to",
    answer: " Mary",
    distractor: " John",
  });
  const [name, setName] = useState("pair");
  const [error, setError] = useState<string | null>(null);
  const modelReady = useStore((st) => st.model.state === "ready");
  const preview = useTokenPreview(record);

  const set = (k: keyof PromptRecord) => (v: string) => setRecord((r) => ({ ...r, [k]: v }));

  return (
    <div className={s.panel}>
      <div className={s.panelTitle}>A single prompt pair</div>
      <div className={s.grid2}>
        <Field label="Clean prompt" help="The model should show the behavior here.">
          <TextArea rows={2} value={record.clean} onChange={(e) => set("clean")(e.target.value)} />
        </Field>
        <Field label="Corrupt prompt" help="Same length in tokens, with the behavior removed or reversed.">
          <TextArea rows={2} value={record.corrupt} onChange={(e) => set("corrupt")(e.target.value)} />
        </Field>
        <Field label="Answer" help="One token, usually with a leading space. Several tokens are read in order by the log-probability metrics.">
          <Input value={record.answer} onChange={(e) => set("answer")(e.target.value)} />
        </Field>
        <Field label="Distractor" help="The wrong answer the metric compares the answer with.">
          <Input value={record.distractor} onChange={(e) => set("distractor")(e.target.value)} />
        </Field>
      </div>
      {preview && (
        <div>
          <div className={s.small} style={{ marginBottom: 6 }}>
            How the loaded model tokenizes it
          </div>
          <TokenGrid data={preview} />
        </div>
      )}
      {!modelReady && <p className={s.faint}>Load a model to check token alignment before saving.</p>}
      <div className={s.row}>
        <Field label="File name">
          <Input value={name} onChange={(e) => setName(e.target.value)} style={{ width: 200 }} />
        </Field>
        <Button
          variant="primary"
          disabled={!name.trim()}
          onClick={async () => {
            const out = await attempt(() => api.savePair({ ...record, name }), setError);
            if (out) onCreated(out.path);
          }}
        >
          Save pair
        </Button>
      </div>
      {error && <Callout tone="error">{error}</Callout>}
    </div>
  );
}

function ImportForm({ onCreated }: { onCreated: (path: string) => void }) {
  const [text, setText] = useState("");
  const [name, setName] = useState("imported");
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  return (
    <div className={s.panel}>
      <div className={s.panelTitle}>Import JSONL</div>
      <p className={s.small}>
        One JSON object per line with <code>clean</code>, <code>corrupt</code>, <code>answer</code> and{" "}
        <code>distractor</code>. Optional <code>positions</code> names character spans in the clean prompt, for
        example <code>{'{"S2": [38, 42]}'}</code>.
      </p>
      <Field label="Paste lines, or choose a file">
        <TextArea
          rows={6}
          value={text}
          spellCheck={false}
          placeholder={'{"clean": "...", "corrupt": "...", "answer": " Mary", "distractor": " John"}'}
          onChange={(e) => setText(e.target.value)}
          style={{ fontSize: "var(--text-xs)" }}
        />
      </Field>
      <div className={s.row}>
        <input
          ref={fileRef}
          type="file"
          accept=".jsonl,.json,.txt"
          className="visually-hidden"
          tabIndex={-1}
          onChange={async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            setText(await file.text());
            setName(file.name.replace(/\.(jsonl|json|txt)$/i, ""));
          }}
        />
        <Button icon="file" onClick={() => fileRef.current?.click()}>
          Choose a file…
        </Button>
        <Field label="File name">
          <Input value={name} onChange={(e) => setName(e.target.value)} style={{ width: 200 }} />
        </Field>
        <Button
          variant="primary"
          disabled={!text.trim() || !name.trim()}
          onClick={async () => {
            const out = await attempt(() => api.importDataset(name, text), setError);
            if (out) onCreated(out.path);
          }}
        >
          Import
        </Button>
      </div>
      {error && <Callout tone="error">{error}</Callout>}
    </div>
  );
}

function wordDiff(a: string, b: string): { text: string; differs: boolean }[] {
  const wa = a.split(/(\s+)/);
  const wb = b.split(/(\s+)/);
  return wa.map((w, i) => ({ text: w, differs: w.trim() !== "" && w !== wb[i] }));
}

function Diff({ text, other }: { text: string; other: string }) {
  return (
    <>
      {wordDiff(text, other).map((w, k) =>
        w.differs ? (
          <span key={k} className={s.diffWord}>
            {w.text}
          </span>
        ) : (
          <Fragment key={k}>{w.text}</Fragment>
        ),
      )}
    </>
  );
}

/** The dataset a run's saved copy of its prompts was made from, as its manifest records it. */
function copiedFrom(runs: Record<string, RunDetail>, path: string): string | null {
  for (const detail of Object.values(runs)) {
    const read = detail.manifest?.dataset;
    if (read?.path === path && read.source_path) return read.source_path;
  }
  return null;
}

/**
 * The prompt pairs of the dataset chosen: searched, a page at a time, and edited. The file never
 * changes, since runs record its hash: edits stay in the tab until they are saved as a new
 * dataset, which is then chosen.
 */
function DatasetPrompts({ path, dataset, onSaved }: { path: string; dataset: DatasetDetail; onSaved: (path: string) => Promise<void> }) {
  const index = useStore((st) => st.promptIndex);
  const setIndex = useStore((st) => st.setPromptIndex);
  const model = useStore((st) => st.model);
  const project = useStore((st) => st.project);
  const runDetails = useStore((st) => st.runDetails);
  const stored = useStore((st) => st.datasetEdits[path]);
  const setDatasetEdits = useStore((st) => st.setDatasetEdits);

  // Edits made to another version of the file no longer line up with its prompts.
  const stale = stored !== undefined && stored.sha256 !== dataset.sha256;
  const edits = stored && !stale ? stored : undefined;
  const base = edits ?? noEdits(dataset.sha256);
  const update = (next: PromptEdits) => setDatasetEdits(path, next);
  const records = useMemo(() => editedRecords(dataset.records, edits), [dataset.records, edits]);
  const removed = useMemo(() => new Set(edits?.removed ?? []), [edits]);
  // The file, as it reads within a sentence and at the start of one. File names keep their case.
  const file = isSnapshot(path) ? "the run's saved copy" : dataset.name;
  const fileAtStart = isSnapshot(path) ? "The run's saved copy" : dataset.name;

  const [query, setQuery] = useState("");
  const deferred = useDeferredValue(query);
  const q = searchQuery(deferred);
  const searching = q !== "";
  const texts = useMemo(() => (searching ? records.map(searchText) : null), [records, searching]);
  const matches = useMemo(() => (texts ? matchTexts(texts, q) : records.map((_, i) => i)), [texts, q, records]);
  const [page, setPage] = useState(() => Math.floor(index / PAGE_SIZE));
  const shown = pageRows(matches, page);
  // One stop for the Tab key in the table: the chosen row, or the page's first.
  const active = shown.rows.includes(index) ? index : shown.rows[0];
  // A prompt chosen elsewhere (the token strip, or [ and ]) brings its page; paging away doesn't
  // bring it back.
  useEffect(() => {
    const at = pageOfRow(matches, index);
    if (at !== null && at !== shown.page) setPage(at);
  }, [index]);

  const top = useRef<HTMLDivElement>(null);
  const search = useRef<HTMLInputElement>(null);
  const buttons = useRef(new Map<number, HTMLButtonElement>());
  const focusAfterPaging = useRef<number | null>(null);
  useEffect(() => {
    const row = focusAfterPaging.current;
    const button = row === null ? undefined : buttons.current.get(row);
    if (button) {
      button.focus();
      focusAfterPaging.current = null;
    }
  });

  /** Search for this text, from the first page of what it finds. */
  const findText = (text: string) => {
    setQuery(text);
    setPage(0);
  };
  const goTo = (next: number, scroll = false) => {
    setPage(next);
    if (scroll) top.current?.scrollIntoView({ block: "start" });
  };
  /** Arrow keys, Page Up and Down, Home and End move to another row and choose it. */
  const onRowKey = (e: KeyboardEvent<HTMLButtonElement>, at: number) => {
    if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
    const next = rowAfterKey(e.key, at, matches.length);
    if (next === null) return;
    e.preventDefault();
    const row = matches[next];
    setIndex(row);
    const target = Math.floor(next / PAGE_SIZE);
    if (target !== shown.page) {
      focusAfterPaging.current = row;
      setPage(target);
    } else buttons.current.get(row)?.focus();
  };

  const issuesByIndex = useMemo(() => {
    const map = new Map<number, string[]>();
    for (const i of dataset.issues ?? []) map.set(i.index, [...(map.get(i.index) ?? []), i.message]);
    return map;
  }, [dataset]);
  const continued = useMemo(() => new Set(dataset.continuations ?? []), [dataset]);
  const [busy, setBusy] = useState(false);
  const nIssues = issuesByIndex.size;
  const unusable = [...issuesByIndex.keys()];

  const [editing, setEditing] = useState<number | null>(null);
  const [confirmDiscard, setConfirmDiscard] = useState(false);
  const taken = useMemo(() => new Set((project?.datasets ?? []).map((d) => d.name)), [project]);
  const suggestion = suggestName((isSnapshot(path) && copiedFrom(runDetails, path)) || dataset.name, taken);
  const [name, setName] = useState<string | null>(null);
  const newName = name ?? suggestion;
  const problem = nameProblem(newName, taken);
  // A dataset needs at least one prompt pair; the server says so too.
  const emptied = edits !== undefined && edits.removed.length >= dataset.n;
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const editsTitle = useId();

  const save = async () => {
    if (!edits || problem || emptied) return;
    setSaving(true);
    setSaveError(null);
    try {
      const out = await api.deriveDataset(deriveRequest(path, edits, newName));
      setDatasetEdits(path, null);
      await onSaved(out.path);
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };
  const discard = () => {
    setDatasetEdits(path, null);
    setConfirmDiscard(false);
    setName(null);
    setSaveError(null);
    // Once the question has closed: its button has gone with the edits.
    window.setTimeout(() => search.current?.focus(), 0);
  };

  const lengths = dataset.lengths ?? [];
  return (
    <div className={p.prompts} ref={top}>
      <p className={s.faint}>
        {dataset.path}
        {lengths.length > 0 && <> · {lengths.length === 1 ? `${lengths[0]} tokens each` : `${lengths[0]}–${lengths[lengths.length - 1]} tokens`}</>}
        {" · "}sha256 {dataset.sha256.slice(0, 12)}
      </p>
      {nIssues > 0 && (
        <Callout tone="error" title={`${plural(nIssues, "prompt")} can't be used with this model`}>
          Experiments refuse datasets with unusable prompts, so results never silently skip any. Edit or remove them and
          save the prompts as a new dataset, or regenerate the dataset with the model loaded.
          <div className={s.row} style={{ marginTop: 8 }}>
            <Button size="small" disabled={stale || unusable.every((i) => removed.has(i))} onClick={() => update(withRemoved(base, unusable, true))}>
              {nIssues === 1 ? "Remove this prompt" : `Remove these ${count(nIssues)} prompts`}
            </Button>
            {model.state === "ready" && model.info && dataset.records.some((r) => r.meta?.template && r.meta?.pattern) && (
              <Button size="small" disabled={busy} onClick={() => void regenerate(dataset.n, model.info?.id, setBusy)}>
                {busy ? "Generating…" : `Generate IOI prompts for ${modelName(model.info.id)}`}
              </Button>
            )}
          </div>
        </Callout>
      )}

      {(stale || hasEdits(edits)) && (
        <section className={p.edits} aria-labelledby={editsTitle}>
          <div>
            <h2 id={editsTitle} className={p.editsTitle}>
              {stale ? "These edits no longer line up with the file" : "Unsaved edits"}
            </h2>
            <p className={s.small}>
              {stale
                ? `${fileAtStart} changed since the edits were made, so they no longer match its prompts. Discard them to edit it as it is now.`
                : `${editsText(edits)}. ${fileAtStart} stays as it is, since runs record its hash: save the edits as a new dataset to use them.`}
            </p>
          </div>
          {stale ? (
            <div className={p.saveRow}>
              <Button onClick={() => setConfirmDiscard(true)}>Discard edits</Button>
            </div>
          ) : (
            <form
              className={p.saveRow}
              onSubmit={(e) => {
                e.preventDefault();
                void save();
              }}
            >
              <Field label="New dataset name" help={problem ? <span className={p.problem}>{problem}</span> : undefined}>
                <Input
                  value={newName}
                  spellCheck={false}
                  onChange={(e) => {
                    setName(e.target.value);
                    setSaveError(null);
                  }}
                />
              </Field>
              <Button variant="primary" type="submit" disabled={!!problem || emptied || saving}>
                {saving ? "Saving…" : "Save as a new dataset"}
              </Button>
              <Button disabled={saving} onClick={() => setConfirmDiscard(true)}>
                Discard edits
              </Button>
            </form>
          )}
          {emptied && <p className={p.blocked}>Every prompt pair is removed. Restore at least one to save a new dataset.</p>}
          {saveError && <Callout tone="error">{saveError}</Callout>}
        </section>
      )}

      <div className={p.toolbar}>
        <div className={p.search}>
          <Icon name="search" size={14} />
          <Input
            ref={search}
            type="search"
            aria-label="Search the prompt pairs"
            placeholder="Search prompts, answers and meta"
            spellCheck={false}
            value={query}
            onChange={(e) => findText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Escape" && query) {
                e.preventDefault();
                findText("");
              }
            }}
          />
          {query && (
            <Button
              size="small"
              variant="ghost"
              icon="close"
              className={p.clear}
              aria-label="Clear the search"
              onClick={() => {
                findText("");
                search.current?.focus();
              }}
            />
          )}
        </div>
        <span className={p.found} aria-live="polite">
          {searching ? matchText(matches.length, records.length) : ""}
        </span>
        <Pager page={shown} searching={searching} onPage={(n) => goTo(n)} />
      </div>

      {shown.rows.length === 0 ? (
        <div className={p.none}>
          <span>No prompt pair contains “{deferred.trim()}”.</span>
          <Button
            size="small"
            onClick={() => {
              findText("");
              search.current?.focus();
            }}
          >
            Show every row
          </Button>
        </div>
      ) : (
        <table className={`${s.table} ${p.table}`}>
          <caption className="visually-hidden">
            Prompt pairs in {file}: {rangeText(shown, searching)}
          </caption>
          <thead>
            <tr>
              <th scope="col" className={s.num}>
                #
              </th>
              <th scope="col">Clean</th>
              <th scope="col">Corrupt</th>
              <th scope="col">Answer</th>
              <th scope="col">Distractor</th>
              <th scope="col">
                <span className="visually-hidden">Edit</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {shown.rows.map((i, k) => {
              const record = records[i];
              const isRemoved = removed.has(i);
              const isEdited = !isRemoved && edits?.replaced[i] !== undefined;
              const tab = i === active ? 0 : -1;
              const issues = isEdited ? undefined : issuesByIndex.get(i);
              const meta = foundByMeta(record, q) ? metaText(record) : "";
              return (
                <tr key={i} data-clickable aria-selected={i === index} data-edit={isRemoved ? "removed" : isEdited ? "edited" : undefined} onClick={() => setIndex(i)}>
                  <th scope="row" className={s.num}>
                    <button
                      ref={(el) => {
                        if (el) buttons.current.set(i, el);
                        else buttons.current.delete(i);
                      }}
                      type="button"
                      className={p.select}
                      aria-pressed={i === index}
                      aria-label={`Select prompt ${i}`}
                      tabIndex={tab}
                      onKeyDown={(e) => onRowKey(e, shown.start + k)}
                    >
                      {i}
                    </button>
                    {isRemoved && <span className={p.mark}>removed</span>}
                    {isEdited && <span className={p.mark}>edited</span>}
                  </th>
                  <td>
                    <Diff text={record.clean} other={record.corrupt} />
                    {issues && <div className={p.issue}>{issues.join(" ")}</div>}
                    {meta && <div className={p.meta}>{meta}</div>}
                  </td>
                  <td>
                    <Diff text={record.corrupt} other={record.clean} />
                  </td>
                  {/* The loaded model splits this pair's answer or distractor: read token by token. */}
                  <td>
                    <AnswerCell answer={record.answer} note={!isEdited && continued.has(i) ? "several tokens in this pair, read in order" : undefined} />
                  </td>
                  <td>
                    <AnswerCell answer={record.distractor} />
                  </td>
                  <td className={p.actions}>
                    {!isRemoved && (
                      <Button size="small" variant="ghost" tabIndex={tab} disabled={stale} aria-label={`Edit prompt ${i}`} onClick={() => setEditing(i)}>
                        Edit
                      </Button>
                    )}
                    <Button
                      size="small"
                      variant="ghost"
                      tabIndex={tab}
                      disabled={stale}
                      aria-label={`${isRemoved ? "Restore" : "Remove"} prompt ${i}`}
                      onClick={() => update(withRemoved(base, [i], !isRemoved))}
                    >
                      {isRemoved ? "Restore" : "Remove"}
                    </Button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      {shown.pages > 1 && <Pager page={shown} searching={searching} onPage={(n) => goTo(n, true)} />}

      {editing !== null && (
        <PromptEditor
          index={editing}
          file={fileAtStart}
          original={dataset.records[editing]}
          current={records[editing]}
          onApply={(record) => {
            update(withEdit(base, editing, record, dataset.records[editing]));
            setEditing(null);
          }}
          onClose={() => setEditing(null)}
        />
      )}
      <ConfirmDialog
        open={confirmDiscard}
        onOpenChange={setConfirmDiscard}
        title="Discard the unsaved edits?"
        keepLabel="Keep the edits"
        confirmLabel="Discard edits"
        onConfirm={discard}
      >
        {hasEdits(edits) ? `${editsText(edits)}: the edits will be lost.` : "The edits will be lost."} {fileAtStart} stays as it is.
      </ConfirmDialog>
    </div>
  );
}

/** Where a page of rows is, and the way to the previous and next pages. */
function Pager({ page, searching, onPage }: { page: RowPage; searching: boolean; onPage: (page: number) => void }) {
  return (
    <div className={p.pager}>
      <span className={p.range}>{rangeText(page, searching)}</span>
      {page.pages > 1 && (
        <>
          <Button size="small" icon="chevronLeft" aria-label="Previous page" disabled={page.page === 0} onClick={() => onPage(page.page - 1)}>
            Previous
          </Button>
          <Button size="small" iconAfter="chevronRight" aria-label="Next page" disabled={page.page >= page.pages - 1} onClick={() => onPage(page.page + 1)}>
            Next
          </Button>
        </>
      )}
    </div>
  );
}

/** An answer or distractor in the table: as written, or a set's members, any of which counts. */
function AnswerCell({ answer, note }: { answer: Answer; note?: string }) {
  if (Array.isArray(answer)) {
    return (
      <span title={answer.map(visibleToken).join(" ")}>
        <span className={s.tok}>{answerText(answer, 6)}</span>
        <span className={s.answerNote}>any of {answer.length}</span>
      </span>
    );
  }
  return (
    <span>
      <span className={s.tok}>{answer}</span>
      {note && <span className={s.answerNote}>{note}</span>}
    </span>
  );
}

/**
 * Edit one prompt pair: each field of the dataset's format. Applying keeps the edit with the
 * others, unsaved; the file doesn't change.
 */
function PromptEditor({
  index,
  file,
  original,
  current,
  onApply,
  onClose,
}: {
  index: number;
  /** The file edited, as a sentence starts with it. */
  file: string;
  original: PromptRecord;
  current: PromptRecord;
  onApply: (record: PromptRecord) => void;
  onClose: () => void;
}) {
  const [draft, setDraft] = useState<RecordDraft>(() => draftOf(current));
  const { record, problem } = recordOf(draft);
  const preview = useTokenPreview(record);
  const formId = useId();
  const set = (patch: Partial<RecordDraft>) => setDraft((d) => ({ ...d, ...patch }));
  const asInFile = record !== null && sameRecord(record, original);
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
      title={`Edit prompt ${index}`}
      description={`Edits stay in this tab until you save them as a new dataset. ${file} doesn't change.`}
      wide
      footer={
        <>
          {!asInFile && (
            <Button variant="ghost" style={{ marginRight: "auto" }} onClick={() => setDraft(draftOf(original))}>
              Restore the file's values
            </Button>
          )}
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" type="submit" form={formId} disabled={!record}>
            Apply
          </Button>
        </>
      }
    >
      <form
        id={formId}
        className={p.editor}
        onSubmit={(e) => {
          e.preventDefault();
          if (record) onApply(record);
        }}
      >
        <div className={s.grid2}>
          <Field label="Clean prompt" help="The model should show the behavior here.">
            <TextArea rows={3} autoFocus value={draft.clean} onChange={(e) => set({ clean: e.target.value })} />
          </Field>
          <Field label="Corrupt prompt" help="Same length in tokens, with the behavior removed or reversed.">
            <TextArea rows={3} value={draft.corrupt} onChange={(e) => set({ corrupt: e.target.value })} />
          </Field>
          <AnswerField label="Answer" value={draft.answer} onChange={(answer) => set({ answer })} />
          <AnswerField label="Distractor" value={draft.distractor} onChange={(distractor) => set({ distractor })} />
        </div>
        <PositionsField clean={draft.clean} value={draft.positions} onChange={(positions) => set({ positions })} />
        <div className={s.grid2}>
          <Field label="ID" help="Optional: a name for this pair, kept in the file.">
            <Input value={draft.id} spellCheck={false} onChange={(e) => set({ id: e.target.value })} />
          </Field>
          <Field label="Meta" help={'Optional: a JSON object, such as {"template": "went"}. The bootstrap can resample prompts by one of its fields.'}>
            <TextArea rows={2} spellCheck={false} value={draft.meta} onChange={(e) => set({ meta: e.target.value })} />
          </Field>
        </div>
        {preview && (
          <div className={p.preview}>
            <div className={s.small}>How the loaded model tokenizes it</div>
            <TokenGrid data={preview} />
          </div>
        )}
        {problem && <Callout tone="error">{problem}</Callout>}
      </form>
    </Dialog>
  );
}

/** An answer or distractor: one token or a continuation read in order, or a set of tokens. */
function AnswerField({ label, value, onChange }: { label: string; value: AnswerDraft; onChange: (value: AnswerDraft) => void }) {
  const members = setMembers(value.members);
  const help =
    value.kind === "text"
      ? `Usually one token with a leading space; several tokens are read in order. Reads as “${visibleToken(value.text)}”.`
      : `One token per line, any of which counts; spaces count. ${plural(members.length, "token")}${members.length ? `: ${answerText(members, 6)}` : ""}.`;
  return (
    <Field label={label} help={help}>
      <Segmented<AnswerDraft["kind"]>
        label={`${label}: one answer or a set`}
        value={value.kind}
        onChange={(kind) => onChange({ ...value, kind })}
        options={[
          { value: "text", label: "Token or continuation" },
          { value: "set", label: "Set of tokens" },
        ]}
      />
      {value.kind === "text" ? (
        <Input value={value.text} spellCheck={false} onChange={(e) => onChange({ ...value, text: e.target.value })} />
      ) : (
        <TextArea rows={4} spellCheck={false} value={value.members} onChange={(e) => onChange({ ...value, members: e.target.value })} />
      )}
    </Field>
  );
}

/** Named character spans in the clean prompt, each with the text it covers. */
function PositionsField({ clean, value, onChange }: { clean: string; value: SpanDraft[]; onChange: (value: SpanDraft[]) => void }) {
  const set = (k: number, patch: Partial<SpanDraft>) => onChange(value.map((span, i) => (i === k ? { ...span, ...patch } : span)));
  return (
    <fieldset className={p.positions}>
      <legend className={p.legend}>Positions</legend>
      <p className={s.small}>
        Optional: named character spans in the clean prompt. A label points to the last token that overlaps its span, so a
        run can patch at, say, S2 in prompts of different lengths. Check them after editing the clean prompt.
      </p>
      {value.length > 0 && (
        <div className={p.spans}>
          <span className={p.spanHead} aria-hidden="true">
            Label
          </span>
          <span className={p.spanHead} aria-hidden="true">
            Start
          </span>
          <span className={p.spanHead} aria-hidden="true">
            End
          </span>
          <span className={p.spanHead} aria-hidden="true">
            Covers
          </span>
          <span />
          {value.map((span, k) => {
            const text = spanText(clean, span);
            const name = span.label.trim() || `${k + 1}`;
            return (
              <Fragment key={k}>
                <Input aria-label={`Position ${k + 1}: label`} value={span.label} spellCheck={false} onChange={(e) => set(k, { label: e.target.value })} />
                <Input aria-label={`Position ${name}: start`} inputMode="numeric" value={span.start} onChange={(e) => set(k, { start: e.target.value })} />
                <Input aria-label={`Position ${name}: end`} inputMode="numeric" value={span.end} onChange={(e) => set(k, { end: e.target.value })} />
                <span className={p.covers}>{text === null ? "not a span of the clean prompt" : `“${text}”`}</span>
                <Button size="small" variant="ghost" icon="close" aria-label={`Remove position ${name}`} onClick={() => onChange(value.filter((_, i) => i !== k))} />
              </Fragment>
            );
          })}
        </div>
      )}
      <Button size="small" icon="plus" onClick={() => onChange([...value, { label: "", start: "", end: "" }])}>
        Add a position
      </Button>
    </fieldset>
  );
}

/** A new IOI dataset from the default templates, with only names that are single tokens for the
 * loaded model (the server filters them), saved beside the others and selected. */
async function regenerate(size: number, modelId: string | undefined, setBusy: (busy: boolean) => void) {
  const st = useStore.getState();
  setBusy(true);
  const out = await st.guard(async () => {
    const taken = new Set((st.project?.datasets ?? []).map((d) => d.name));
    const base = `ioi-${modelName(modelId).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "")}`;
    let name = base;
    for (let k = 2; taken.has(name) || taken.has(`${name}.jsonl`); k++) name = `${base}-${k}`;
    // The task's own defaults: its default templates, both name orders, the subject flipped.
    return api.generateTask({ task: "ioi", name, n: Math.max(32, size), seed: 0, options: {} });
  });
  setBusy(false);
  if (!out) return;
  await st.refreshProject();
  await st.selectDataset(out.path);
  st.notify(`Saved ${out.path}, with names that are single tokens for ${modelName(modelId)}. Check the baseline next.`);
}
