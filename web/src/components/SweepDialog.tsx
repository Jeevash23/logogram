import { useEffect, useMemo, useState } from "react";

import { api } from "../api/client";
import type { Revisions, SweepField, SweepPlan, SweepValue, TaskInfo } from "../api/types";
import { capitalize, count, plural } from "../lib/format";
import { parseValues } from "../lib/sweeps";
import { useStore } from "../store/app";
import { Button, Callout, Checkbox, Choices, Dialog, Field, Input, Select } from "./ui";
import d from "./SweepDialog.module.css";

/** A list in words: "1, 2 and 3". */
function listText(items: string[]): string {
  return items.length <= 1 ? items.join("") : `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}

/** Sweep a run: run its spec once for each of several values of one choice, one run after
 * another, through the queue. Opened from the run's results or the command palette. */
export function SweepDialog() {
  const runId = useStore((st) => st.sweepDialogRun);
  return runId ? <SweepForm key={runId} runId={runId} /> : null;
}

function SweepForm({ runId }: { runId: string }) {
  const modelId = useStore((st) => st.runDetails[runId]?.spec.model.id);
  const startSweep = useStore((st) => st.startSweep);
  const close = () => useStore.setState({ sweepDialogRun: null });
  const [fields, setFields] = useState<SweepField[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [fieldPath, setFieldPath] = useState("");
  // What is typed for each field, so switching between them keeps it.
  const [texts, setTexts] = useState<Record<string, string>>({});
  const [datasets, setDatasets] = useState<string[]>([]);
  // The server's plan for the values as they are now, or why there is none: an error, or a hint
  // while fewer than two values are given.
  const [planned, setPlanned] = useState<{ key: string; plan: SweepPlan | null; error: string | null; hint?: string } | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api.sweepFields(runId).then(
      (list) => {
        if (cancelled) return;
        setFields(list);
        // The seed of the run's own method first (its donors, its split), else the bootstrap's.
        const first = list.find((f) => f.kind === "seed" && f.field !== "statistics.seed") ?? list[0];
        if (first) setFieldPath(first.field);
        setTexts(Object.fromEntries(list.filter((f) => f.kind !== "dataset").map((f) => [f.field, f.suggested.join(", ")])));
        setDatasets((list.find((f) => f.kind === "dataset")?.suggested ?? []).map(String));
      },
      (e: Error) => !cancelled && setLoadError(e.message),
    );
    return () => {
      cancelled = true;
    };
  }, [runId]);

  const field = fields?.find((f) => f.field === fieldPath) ?? null;
  const parsed = useMemo(
    () =>
      !field ? { values: [] as SweepValue[], error: null } : field.kind === "dataset" ? { values: datasets as SweepValue[], error: null } : parseValues(field.kind, texts[field.field] ?? ""),
    [field, texts, datasets],
  );
  const key = JSON.stringify([fieldPath, parsed.values]);

  // What the sweep would run, or why it can't, from the server: asked again a moment after typing
  // stops.
  useEffect(() => {
    if (!field) return;
    if (parsed.error) {
      setPlanned({ key, plan: null, error: parsed.error });
      return;
    }
    if (parsed.values.length < 2) {
      const give = field.kind === "dataset" ? "Tick" : "Give";
      setPlanned({ key, plan: null, error: null, hint: `${give} at least two ${field.plural} to sweep over.` });
      return;
    }
    let cancelled = false;
    const timer = window.setTimeout(() => {
      api.planSweep({ run: runId, field: field.field, values: parsed.values }).then(
        (plan) => !cancelled && setPlanned({ key, plan, error: null }),
        (e: Error) => !cancelled && setPlanned({ key, plan: null, error: e.message }),
      );
    }, 250);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
    // The key holds the field and the values.
  }, [key, field, runId, parsed]);

  const current = planned && planned.key === key ? planned : null;
  const plan = current?.plan ?? null;
  const made = plan ? plan.runs.filter((r) => !r.original).length : 0;
  const queue = async () => {
    if (!plan || !field) return;
    setBusy(true);
    const ok = await startSweep({ run: runId, field: field.field, values: parsed.values });
    setBusy(false);
    if (ok) close();
  };

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) close();
      }}
      title="Sweep this run"
      description="Run its spec once for each of several values of one choice, one run after another. The sweep then shows which of its strongest sites keep their sign and size across the values."
      wide
      footer={
        <>
          <Button variant="ghost" onClick={close}>
            Cancel
          </Button>
          <Button variant="primary" disabled={!plan || busy} onClick={() => void queue()}>
            {made > 0 ? `Queue ${plural(made, "run")}` : "Queue the runs"}
          </Button>
        </>
      }
    >
      {loadError && <Callout tone="error">{loadError}</Callout>}
      {!fields && !loadError && <p className={d.note}>Reading the run&rsquo;s spec…</p>}
      {fields && (
        <>
          <Choices
            label="Vary"
            value={fieldPath}
            onChange={setFieldPath}
            columns={2}
            options={fields.map((f) => ({
              value: f.field,
              title: capitalize(f.label),
              detail: (
                <>
                  {f.description} This run: {f.current_label ?? "unpinned"}.
                </>
              ),
            }))}
          />
          {field?.kind === "seed" && (
            <Field
              label={`${capitalize(field.plural)}, in order`}
              help={`Whole numbers separated by commas. This run's own ${field.label}, ${field.current_label}, stands in for itself: it isn't run again.`}
            >
              <Input
                value={texts[field.field] ?? ""}
                spellCheck={false}
                onChange={(e) => setTexts({ ...texts, [field.field]: e.target.value })}
              />
            </Field>
          )}
          {field?.kind === "revision" && (
            <RevisionValues
              modelId={modelId}
              text={texts[field.field] ?? ""}
              onText={(text) => setTexts({ ...texts, [field.field]: text })}
            />
          )}
          {field?.kind === "dataset" && <DatasetValues chosen={datasets} suggested={field.suggested.map(String)} onChange={setDatasets} />}
          {field && <PlanText plan={plan} error={current?.error ?? null} hint={current?.hint ?? null} field={field} />}
        </>
      )}
    </Dialog>
  );
}

/** What Queue will do, in words, or why it can't. */
function PlanText({ plan, error, hint, field }: { plan: SweepPlan | null; error: string | null; hint: string | null; field: SweepField }) {
  if (error) return <Callout tone="error">{error}</Callout>;
  if (hint) return <p className={d.plan}>{hint}</p>;
  if (!plan) return <p className={d.note}>Checking the values…</p>;
  const runs = plan.runs.filter((r) => !r.original);
  const kept = plan.runs.find((r) => r.original);
  const own = field.current_label;
  return (
    <p className={d.plan} role="status">
      Queues {plural(runs.length, "run")}, one after another: {runs.length === 1 ? plan.label : plan.plural} {listText(runs.map((r) => r.label))}.{" "}
      {kept
        ? `${capitalize(plan.label)} ${kept.label} is this run, which stands in for it.`
        : own !== null
          ? `This run's ${plan.label}, ${own}, isn't one of them, so this run isn't part of the sweep.`
          : ""}
    </p>
  );
}

/** Revisions typed in order, or picked from the model's branches and tags. */
function RevisionValues({ modelId, text, onText }: { modelId: string | undefined; text: string; onText: (text: string) => void }) {
  const [list, setList] = useState<Revisions["revisions"] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [find, setFind] = useState("");
  const chosen = text.split(/[\s,]+/).filter(Boolean);
  const names = new Map((list ?? []).map((r, i) => [r.name, i]));
  // Picked revisions keep the list's order, which follows training for checkpoints like step1000;
  // anything typed that isn't on the list keeps its place before them.
  const toggle = (name: string) => {
    const next = chosen.includes(name) ? chosen.filter((x) => x !== name) : [...chosen, name];
    const typed = next.filter((x) => !names.has(x));
    const picked = next.filter((x) => names.has(x)).sort((a, b) => (names.get(a) ?? 0) - (names.get(b) ?? 0));
    onText([...typed, ...picked].join(", "));
  };
  const load = async () => {
    if (!modelId) return;
    setLoading(true);
    setError(null);
    try {
      setList((await api.revisions(modelId)).revisions);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };
  const shown = (list ?? []).filter((r) => r.name.toLowerCase().includes(find.trim().toLowerCase()));
  return (
    <div className={d.values}>
      <Field
        label="Revisions, in order"
        help="Commits, tags or branches of the model's repository on Hugging Face, separated by commas, such as Pythia's checkpoints step1000, step2000. Each run pins the commit its revision points to when it runs."
      >
        <Input value={text} spellCheck={false} onChange={(e) => onText(e.target.value)} />
      </Field>
      <div className={d.row}>
        <Button size="small" disabled={!modelId || loading} onClick={() => void load()}>
          {loading ? "Listing…" : list ? "List them again" : "List its branches and tags"}
        </Button>
        <span className={d.note}>Asks Hugging Face for {modelId ?? "the model"}&rsquo;s branches and tags, once.</span>
      </div>
      {error && <Callout tone="error">{error}</Callout>}
      {list && (
        <>
          <Input
            type="search"
            aria-label="Find a branch or tag"
            placeholder="Find a branch or tag, such as step1"
            value={find}
            spellCheck={false}
            onChange={(e) => setFind(e.target.value)}
          />
          <ul className={d.list} aria-label="Branches and tags">
            {shown.map((r) => (
              <li key={`${r.kind}:${r.name}`}>
                <Checkbox checked={chosen.includes(r.name)} onChange={() => toggle(r.name)}>
                  {r.name}{" "}
                  <span className={d.meta}>
                    {r.kind} · {r.commit.slice(0, 7)}
                  </span>
                </Checkbox>
              </li>
            ))}
          </ul>
          <p className={d.note}>
            {plural(list.length, "branch or tag", "branches and tags")}
            {find.trim() ? `, ${count(shown.length)} shown` : ""}.
          </p>
        </>
      )}
    </div>
  );
}

/** Datasets of the project, in the order they are ticked, and a way to generate one per seed. */
function DatasetValues({ chosen, suggested, onChange }: { chosen: string[]; suggested: string[]; onChange: (paths: string[]) => void }) {
  const datasets = useStore((st) => st.project?.datasets ?? []);
  // A finished run read a saved copy of its prompts: offered where the file it came from changed.
  const rows = [
    ...datasets.map((x) => ({ path: x.path, name: x.name, n: x.n ?? null, error: x.error ?? null })),
    ...suggested.filter((p) => !datasets.some((x) => x.path === p)).map((p) => ({ path: p, name: "This run's saved copy of its prompts", n: null, error: null })),
  ];
  const toggle = (path: string) => onChange(chosen.includes(path) ? chosen.filter((x) => x !== path) : [...chosen, path]);
  return (
    <div className={d.values}>
      <fieldset className={d.fieldset}>
        <legend className={d.legend}>Datasets, in the order ticked</legend>
        <ul className={d.list}>
          {rows.map((r) => {
            const at = chosen.indexOf(r.path);
            return (
              <li key={r.path}>
                <Checkbox checked={at >= 0} disabled={!!r.error} onChange={() => toggle(r.path)}>
                  {at >= 0 && <span className={d.order}>{at + 1}</span>}
                  {r.name} <span className={d.meta}>{r.error ? "can't be read" : r.n !== null ? plural(r.n, "prompt pair") : r.path}</span>
                </Checkbox>
              </li>
            );
          })}
        </ul>
      </fieldset>
      <GenerateDatasets onCreated={(paths) => onChange([...chosen, ...paths.filter((p) => !chosen.includes(p))])} />
    </div>
  );
}

/** One dataset of a task for each seed, with the task's default options: the same task's prompts
 * drawn again, to see what a sample of prompts changes. */
function GenerateDatasets({ onCreated }: { onCreated: (paths: string[]) => void }) {
  const [tasks, setTasks] = useState<TaskInfo[] | null>(null);
  const [taskId, setTaskId] = useState("ioi");
  const [n, setN] = useState(32);
  const [seeds, setSeeds] = useState("1, 2, 3");
  const [name, setName] = useState("ioi");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!open || tasks) return;
    api.tasks().then(setTasks, (e: Error) => setError(e.message));
  }, [open, tasks]);
  const task = tasks?.find((t) => t.id === taskId) ?? null;
  const parsed = parseValues("seed", seeds);
  const generate = async () => {
    if (!task || parsed.error || parsed.values.length === 0) return;
    setBusy(true);
    setError(null);
    const made: string[] = [];
    try {
      const options = Object.fromEntries(task.options.map((o) => [o.name, Array.isArray(o.default) ? [...o.default] : o.default]));
      for (const seed of parsed.values) {
        const out = await api.generateTask({ task: task.id, name: `${name.trim()}-seed${seed}`, n, seed: Number(seed), options });
        made.push(out.path);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      if (made.length) {
        await useStore.getState().refreshProject();
        onCreated(made);
      }
    }
  };
  return (
    <details className={d.generate} onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>Generate one dataset per seed</summary>
      {open && (
        <div className={d.generateBody}>
          {!tasks && !error && <p className={d.note}>Reading the tasks…</p>}
          {tasks && (
            <div className={d.grid}>
              <Field label="Task">
                <Select
                  value={taskId}
                  onChange={(e) => {
                    setTaskId(e.target.value);
                    setName(e.target.value.replace(/_/g, "-"));
                  }}
                >
                  {tasks.map((t) => (
                    <option key={t.id} value={t.id}>
                      {t.name}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label="Prompt pairs in each">
                <Input type="number" min={1} max={100000} value={n} onChange={(e) => setN(Math.max(1, Math.round(Number(e.target.value) || 1)))} />
              </Field>
              <Field label="Seeds" help={parsed.error ?? undefined}>
                <Input value={seeds} spellCheck={false} onChange={(e) => setSeeds(e.target.value)} />
              </Field>
              <Field label="File names" help={`${name.trim() || "name"}-seed${parsed.values[0] ?? 1}.jsonl, and so on`}>
                <Input value={name} spellCheck={false} onChange={(e) => setName(e.target.value)} />
              </Field>
            </div>
          )}
          {error && <Callout tone="error">{error}</Callout>}
          {tasks && (
            <div className={d.row}>
              <Button size="small" disabled={busy || !!parsed.error || parsed.values.length === 0 || !name.trim()} onClick={() => void generate()}>
                {busy ? "Generating…" : `Generate ${plural(parsed.values.length, "dataset")}`}
              </Button>
              <span className={d.note}>With the task&rsquo;s default options, for the loaded model&rsquo;s tokenizer. They are ticked for the sweep.</span>
            </div>
          )}
        </div>
      )}
    </details>
  );
}
