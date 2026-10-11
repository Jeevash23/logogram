import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { TaskDatasetCreated, TaskInfo, TaskOption } from "../api/types";
import { Button, Callout, Checkbox, Choices, Field, Input } from "../components/ui";
import { plural } from "../lib/format";
import { METRICS, taskMetric as metricFor } from "../lib/metrics";
import { useStore } from "../store/app";
import s from "./views.module.css";

/** Plainer names for some tasks' options, and their values (the server's names otherwise). */
const OPTION_LABELS: Record<string, string> = {
  templates: "Templates",
  patterns: "Name order",
  corruption: "Corruption",
  single_token_answers: "Single-token answers",
};
const VALUE_TEXT: Record<string, Record<string, { title: string; detail?: string }>> = {
  patterns: {
    ABBA: { title: "ABBA", detail: "When Mary and John …, John gave … to" },
    BABA: { title: "BABA", detail: "When John and Mary …, John gave … to" },
  },
  corruption: {
    flip: { title: "Flip the subject", detail: "S2 becomes the IO name, so the corrupt prompt favors the other name (a negative preference)." },
    abc: { title: "Three new names", detail: "All names are replaced by unrelated ones, so neither answer is supported." },
  },
};

const label = (option: TaskOption) => OPTION_LABELS[option.name] ?? option.name.charAt(0).toUpperCase() + option.name.slice(1).replace(/_/g, " ");

/** The options a task starts with: its defaults, every one stated in the request. */
function defaults(task: TaskInfo): Record<string, unknown> {
  return Object.fromEntries(task.options.map((o) => [o.name, Array.isArray(o.default) ? [...o.default] : o.default]));
}

/** A file name for a task's dataset, not yet taken in the project. */
function freeName(task: TaskInfo, taken: Set<string>): string {
  const base = task.id.replace(/_/g, "-");
  let name = base;
  for (let k = 2; taken.has(`${name}.jsonl`) || taken.has(name); k++) name = `${base}-${k}`;
  return name;
}

/**
 * Generate a seeded dataset for one of the classic tasks the server knows (GET /api/tasks). Each
 * option is shown by its type; the templates are checkboxes. A dataset whose answers another
 * metric reads offers to switch the experiment form to it.
 */
export function TaskGenerator({ onCreated }: { onCreated: (path: string, created: TaskDatasetCreated) => void }) {
  const [tasks, setTasks] = useState<TaskInfo[] | null>(null);
  const [taskId, setTaskId] = useState("ioi");
  const [options, setOptions] = useState<Record<string, unknown>>({});
  const [n, setN] = useState(32);
  const [seed, setSeed] = useState(0);
  const [name, setName] = useState("ioi");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const modelReady = useStore((st) => st.model.state === "ready");
  const datasets = useStore((st) => st.project?.datasets);
  const taken = new Set((datasets ?? []).map((d) => d.name));

  useEffect(() => {
    let cancelled = false;
    api.tasks().then(
      (list) => {
        if (cancelled) return;
        setTasks(list);
        const first = list.find((t) => t.id === "ioi") ?? list[0];
        if (first) choose(first, list);
      },
      (e: Error) => !cancelled && setError(e.message),
    );
    return () => {
      cancelled = true;
    };
    // Once, when the generator opens: the tasks don't change while the server runs.
  }, []);

  const task = tasks?.find((t) => t.id === taskId) ?? null;
  const choose = (next: TaskInfo, list = tasks) => {
    if (!list) return;
    setTaskId(next.id);
    setOptions(defaults(next));
    setName(freeName(next, taken));
    setError(null);
  };
  const set = (option: string, value: unknown) => setOptions((cur) => ({ ...cur, [option]: value }));
  const missing = task?.options.find((o) => o.type === "choices" && !(options[o.name] as string[] | undefined)?.length);
  const validN = Number.isInteger(n) && n >= 1 && n <= 100_000;
  const validSeed = Number.isInteger(seed) && seed >= 0 && seed < 2 ** 32;

  const generate = async () => {
    if (!task) return;
    setBusy(true);
    setError(null);
    try {
      const out = await api.generateTask({ task: task.id, name: name.trim(), n, seed, options });
      onCreated(out.path, out);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  if (!tasks) {
    return (
      <div className={s.panel}>
        <div className={s.panelTitle}>Generate a task</div>
        {error ? <Callout tone="error">{error}</Callout> : <p className={s.small}>Reading the tasks…</p>}
      </div>
    );
  }
  return (
    <div className={s.panel}>
      <div className={s.panelTitle}>Generate a task</div>
      <Choices
        label="Task"
        value={taskId}
        onChange={(id) => {
          const next = tasks.find((t) => t.id === id);
          if (next) choose(next);
        }}
        columns={3}
        options={tasks.map((t) => ({ value: t.id, title: t.name, detail: `Read with the ${METRICS[t.metric]?.label ?? t.metric}` }))}
      />
      {task && <p className={s.small}>{task.description}</p>}
      <div className={s.grid3}>
        <Field label="Prompts">
          <Input type="number" min={1} max={100000} value={n} onChange={(e) => setN(Number(e.target.value))} aria-invalid={!validN} />
        </Field>
        <Field label="Seed" help="The same seed and settings give the same prompts on every machine.">
          <Input type="number" min={0} value={seed} onChange={(e) => setSeed(Number(e.target.value))} aria-invalid={!validSeed} />
        </Field>
        <Field label="File name">
          <Input value={name} onChange={(e) => setName(e.target.value)} />
        </Field>
      </div>
      {task?.options.map((option) => (
        <OptionField key={option.name} task={task} option={option} value={options[option.name]} onChange={(v) => set(option.name, v)} />
      ))}
      {task && metricFor(task, options) !== task.metric && (
        <p className={s.small}>With these options, its answers are read with the {METRICS[metricFor(task, options)]?.label}.</p>
      )}
      {!modelReady && (
        <p className={s.faint}>
          With a model loaded, words that aren't single tokens for it are left out, and pairs keep one length. Without one,
          each word counts as a token: generate again once the model is loaded.
        </p>
      )}
      <div>
        <Button variant="primary" disabled={busy || !task || !!missing || !validN || !validSeed || !name.trim()} onClick={() => void generate()}>
          {busy ? "Generating…" : `Generate ${plural(validN ? n : 0, "prompt")}`}
        </Button>
      </div>
      {missing && <p className={s.small}>Choose at least one of the {label(missing).toLowerCase()}.</p>}
      {error && <Callout tone="error">{error}</Callout>}
    </div>
  );
}

function OptionField({ task, option, value, onChange }: { task: TaskInfo; option: TaskOption; value: unknown; onChange: (v: unknown) => void }) {
  if (option.type === "bool") {
    return (
      <Field label={label(option)} help={option.description}>
        <Checkbox checked={value === true} onChange={onChange}>
          {label(option)}
        </Checkbox>
      </Field>
    );
  }
  if (option.type === "choice") {
    const text = VALUE_TEXT[option.name] ?? {};
    return (
      <Field label={label(option)} help={text[option.allowed[0]] ? undefined : option.description}>
        <Choices
          label={label(option)}
          value={typeof value === "string" ? value : null}
          onChange={onChange}
          columns={Math.min(3, option.allowed.length)}
          options={option.allowed.map((v) => ({ value: v, title: text[v]?.title ?? v, detail: text[v]?.detail }))}
        />
      </Field>
    );
  }
  const chosen = Array.isArray(value) ? (value as string[]) : [];
  // Keep the order the task lists them in.
  const toggle = (v: string, on: boolean) => onChange(option.allowed.filter((x) => (x === v ? on : chosen.includes(x))));
  if (option.name === "templates") {
    return (
      <Field label={label(option)} help={option.description}>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {task.templates.map((t) => (
            <Checkbox key={t.id} top checked={chosen.includes(t.id)} onChange={(on) => toggle(t.id, on)}>
              <TemplateText text={t.text} task={task.id} />
            </Checkbox>
          ))}
        </div>
      </Field>
    );
  }
  const text = VALUE_TEXT[option.name] ?? {};
  return (
    <Field label={label(option)} help={option.description}>
      <div className={s.row}>
        {option.allowed.map((v) => (
          <Checkbox key={v} checked={chosen.includes(v)} onChange={(on) => toggle(v, on)}>
            {text[v]?.title ?? v} {text[v]?.detail && <span className="faint">({text[v]?.detail})</span>}
          </Checkbox>
        ))}
      </div>
    </Field>
  );
}

/** IOI's slots, by the names the paper gives them. */
const IOI_SLOTS: Record<string, string> = { A: "A", B: "B", C: "S", PLACE: "place", OBJECT: "object" };

/** A template with its slots marked: short codes as written, words in lower case. */
export function TemplateText({ text, task }: { text: string; task: string }) {
  const parts = text.split(/(\{[A-Z][A-Z0-9_]*\})/g);
  const name = (slot: string) =>
    task === "ioi" && IOI_SLOTS[slot] ? IOI_SLOTS[slot] : slot.length <= 3 ? slot : slot.toLowerCase().replace(/_/g, " ");
  return (
    <span className={s.template}>
      {parts.map((p, i) =>
        p.startsWith("{") ? (
          <span key={i} className={s.slot}>
            {name(p.slice(1, -1))}
          </span>
        ) : (
          <span key={i}>{p}</span>
        ),
      )}
    </span>
  );
}
