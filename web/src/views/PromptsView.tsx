import { useEffect, useMemo, useRef, useState } from "react";

import { api } from "../api/client";
import type { IOITemplate, PromptRecord, TokenStripData } from "../api/types";
import { TokenGrid } from "../components/TokenStrip";
import { Button, Callout, Checkbox, Choices, Field, Input, Segmented, TextArea } from "../components/ui";
import { plural } from "../lib/format";
import { useStore } from "../store/app";
import s from "./views.module.css";

type Mode = "none" | "pair" | "ioi" | "import";

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

export function PromptsView() {
  const project = useStore((st) => st.project);
  const datasetPath = useStore((st) => st.datasetPath);
  const selectDataset = useStore((st) => st.selectDataset);
  const datasets = project?.datasets ?? [];
  const [mode, setMode] = useState<Mode>(datasets.length ? "none" : "ioi");

  const created = async (path: string) => {
    await useStore.getState().refreshProject();
    await selectDataset(path);
    setMode("none");
    useStore.getState().notify(`Saved ${path}. Check the baseline next.`);
  };

  return (
    <div className={s.view}>
      <div className={s.head}>
        <div className={s.titleBlock}>
          <h2 className={s.title}>Prompts</h2>
          <p className={s.subtitle}>
            Each prompt is a pair: a clean prompt where the model shows the behavior, and a corrupt prompt where it
            shouldn't. The metric is the logit difference between the answer and a distractor at the last token.
          </p>
        </div>
      </div>

      <div className={s.row} style={{ alignItems: "center" }}>
        <div className={s.tabsInline} role="group" aria-label="Datasets">
          {datasets.map((d) => (
            <button
              key={d.path}
              type="button"
              className={s.datasetButton}
              aria-pressed={d.path === datasetPath}
              onClick={() => void selectDataset(d.path)}
              title={d.error}
            >
              {d.name}
              <span>{d.error ? "unreadable" : plural(d.n ?? 0, "prompt")}</span>
            </button>
          ))}
        </div>
        <div style={{ flex: 1 }} />
        <Segmented<Mode>
          label="Add prompts"
          value={mode}
          onChange={(m) => setMode(m === mode ? "none" : m)}
          options={[
            { value: "pair", label: "Single pair" },
            { value: "ioi", label: "IOI generator" },
            { value: "import", label: "Import JSONL" },
          ]}
        />
      </div>

      {mode === "pair" && <PairForm onCreated={created} />}
      {mode === "ioi" && <IOIForm onCreated={created} />}
      {mode === "import" && <ImportForm onCreated={created} />}

      <DatasetTable />
    </div>
  );
}

function PairForm({ onCreated }: { onCreated: (path: string) => void }) {
  const [record, setRecord] = useState<PromptRecord>({
    clean: "When Mary and John went to the store, John gave a drink to",
    corrupt: "When Mary and John went to the store, Mary gave a drink to",
    answer: " Mary",
    distractor: " John",
  });
  const [name, setName] = useState("pair");
  const [preview, setPreview] = useState<TokenStripData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const modelReady = useStore((st) => st.model.state === "ready");

  useEffect(() => {
    if (!modelReady) return;
    const t = window.setTimeout(() => {
      api.tokenize({ record }).then(setPreview, () => setPreview(null));
    }, 250);
    return () => window.clearTimeout(t);
  }, [record, modelReady]);

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
        <Field label="Answer" help="One token, usually with a leading space.">
          <Input value={record.answer} onChange={(e) => set("answer")(e.target.value)} />
        </Field>
        <Field label="Distractor" help="The wrong answer the logit difference is measured against.">
          <Input value={record.distractor} onChange={(e) => set("distractor")(e.target.value)} />
        </Field>
      </div>
      {modelReady && preview && (
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

function IOIForm({ onCreated }: { onCreated: (path: string) => void }) {
  const [templates, setTemplates] = useState<IOITemplate[]>([]);
  const [chosen, setChosen] = useState<string[]>([]);
  const [n, setN] = useState(32);
  const [seed, setSeed] = useState(0);
  const [patterns, setPatterns] = useState<("ABBA" | "BABA")[]>(["ABBA", "BABA"]);
  const [corruption, setCorruption] = useState<"flip" | "abc">("flip");
  const [name, setName] = useState("ioi");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const modelReady = useStore((st) => st.model.state === "ready");

  useEffect(() => {
    api.ioiTemplates().then((t) => {
      setTemplates(t);
      setChosen(t.filter((x) => x.default).map((x) => x.id));
    }, () => undefined);
  }, []);

  const togglePattern = (p: "ABBA" | "BABA", on: boolean) =>
    setPatterns((cur) => (on ? [...new Set([...cur, p])] : cur.filter((x) => x !== p)));

  return (
    <div className={s.panel}>
      <div className={s.panelTitle}>Indirect object identification</div>
      <p className={s.small}>
        Two people are named and one is mentioned again; the model should complete with the other. The answer is
        the indirect object (IO) and the distractor is the repeated subject (S). Named positions IO, S1, S2 and end
        are recorded for each prompt.
      </p>
      <div className={s.grid3}>
        <Field label="Prompts">
          <Input type="number" min={1} max={100000} value={n} onChange={(e) => setN(Number(e.target.value))} />
        </Field>
        <Field label="Seed">
          <Input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
        </Field>
        <Field label="File name">
          <Input value={name} onChange={(e) => setName(e.target.value)} />
        </Field>
      </div>
      <Field label="Name order">
        <div className={s.row}>
          <Checkbox checked={patterns.includes("ABBA")} onChange={(v) => togglePattern("ABBA", v)}>
            ABBA <span className="faint">(When Mary and John …, John gave … to)</span>
          </Checkbox>
          <Checkbox checked={patterns.includes("BABA")} onChange={(v) => togglePattern("BABA", v)}>
            BABA <span className="faint">(When John and Mary …, John gave … to)</span>
          </Checkbox>
        </div>
      </Field>
      <Field label="Templates" help="The first three share one token structure, so named positions line up exactly.">
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {templates.map((t) => (
            <Checkbox
              key={t.id}
              checked={chosen.includes(t.id)}
              onChange={(v) => setChosen((cur) => (v ? [...cur, t.id] : cur.filter((x) => x !== t.id)))}
            >
              <TemplateText text={t.text} />
            </Checkbox>
          ))}
        </div>
      </Field>
      <Field label="Corruption">
        <Choices
          label="Corruption"
          value={corruption}
          onChange={setCorruption}
          columns={2}
          options={[
            {
              value: "flip",
              title: "Flip the subject",
              detail: "S2 becomes the IO name, so the corrupt prompt favors the other name (negative logit difference).",
            },
            {
              value: "abc",
              title: "Three new names",
              detail: "All names are replaced by unrelated ones, so neither answer is supported.",
            },
          ]}
        />
      </Field>
      {!modelReady && (
        <p className={s.faint}>
          With a model loaded, names that aren't single tokens for it are left out automatically.
        </p>
      )}
      <div>
        <Button
          variant="primary"
          disabled={busy || !chosen.length || !patterns.length || n < 1 || !name.trim()}
          onClick={async () => {
            setBusy(true);
            const out = await attempt(
              () => api.generateIOI({ name, n, seed, templates: chosen, patterns, corruption }),
              setError,
            );
            setBusy(false);
            if (out) onCreated(out.path);
          }}
        >
          Generate {plural(n, "prompt")}
        </Button>
      </div>
      {error && <Callout tone="error">{error}</Callout>}
    </div>
  );
}

function TemplateText({ text }: { text: string }) {
  const parts = text.split(/(\{[A-Z]+\})/g);
  const names: Record<string, string> = { "{A}": "A", "{B}": "B", "{C}": "S", "{PLACE}": "place", "{OBJECT}": "object" };
  return (
    <span className={s.template}>
      {parts.map((p, i) => (p.startsWith("{") ? <span key={i} className={s.slot}>{names[p] ?? p}</span> : <span key={i}>{p}</span>))}
    </span>
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

function DatasetTable() {
  const dataset = useStore((st) => st.dataset);
  const index = useStore((st) => st.promptIndex);
  const setIndex = useStore((st) => st.setPromptIndex);
  const issuesByIndex = useMemo(() => {
    const map = new Map<number, string[]>();
    for (const i of dataset?.issues ?? []) map.set(i.index, [...(map.get(i.index) ?? []), i.message]);
    return map;
  }, [dataset]);
  if (!dataset) return null;
  const shown = dataset.records.slice(0, 300);
  const nIssues = issuesByIndex.size;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div className={s.row} style={{ alignItems: "baseline" }}>
        <h3 className={s.panelTitle}>{dataset.name}</h3>
        <span className={s.faint}>
          {plural(dataset.n, "prompt")}
          {dataset.lengths && dataset.lengths.length > 0 && (
            <>
              {" "}
              · {dataset.lengths.length === 1 ? `${dataset.lengths[0]} tokens each` : `${dataset.lengths[0]}–${dataset.lengths[dataset.lengths.length - 1]} tokens`}
            </>
          )}{" "}
          · sha256 {dataset.sha256.slice(0, 12)}
        </span>
      </div>
      {nIssues > 0 && (
        <Callout tone="error" title={`${plural(nIssues, "prompt")} can't be used with this model`}>
          Experiments refuse datasets with unusable prompts, so results never silently skip any. Fix these lines in
          the file, or regenerate the dataset with the model loaded.
        </Callout>
      )}
      <table className={s.table}>
        <thead>
          <tr>
            <th className={s.num}>#</th>
            <th>Clean</th>
            <th>Corrupt</th>
            <th>Answer</th>
            <th>Distractor</th>
          </tr>
        </thead>
        <tbody>
          {shown.map((r, i) => {
            const issues = issuesByIndex.get(i);
            return (
              <tr key={i} data-clickable aria-selected={i === index} onClick={() => setIndex(i)}>
                <td className={s.num}>{i}</td>
                <td>
                  {wordDiff(r.clean, r.corrupt).map((w, k) =>
                    w.differs ? <span key={k} className={s.diffWord}>{w.text}</span> : w.text,
                  )}
                  {issues && <div style={{ color: "var(--alert)", fontSize: "var(--text-xs)", marginTop: 2 }}>{issues.join(" ")}</div>}
                </td>
                <td>
                  {wordDiff(r.corrupt, r.clean).map((w, k) =>
                    w.differs ? <span key={k} className={s.diffWord}>{w.text}</span> : w.text,
                  )}
                </td>
                <td className={s.tok}>{r.answer}</td>
                <td className={s.tok}>{r.distractor}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {dataset.n > shown.length && <p className={s.faint}>Showing the first {shown.length} prompts.</p>}
    </div>
  );
}
