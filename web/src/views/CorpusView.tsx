import { useDeferredValue, useEffect, useId, useMemo, useRef, useState } from "react";

import { api, type CorpusTopRequest } from "../api/client";
import type { CorpusFormat, CorpusListing, CorpusPage, CorpusTop, DirectionListing, HubDatasetFiles, TopExample, TopUnit } from "../api/types";
import { ScaleBar } from "../components/Heatmap/ScaleBar";
import { Button, Callout, Checkbox, ConfirmDialog, Empty, Field, Icon, Input, Progress, Segmented, Select, Spinner, Tip } from "../components/ui";
import { font, prepareCanvas, useChromeColors, useElementSize } from "../lib/canvas";
import { inkScale, textOn } from "../lib/color";
import {
  datasetFacts,
  directionProblem,
  exampleBound,
  fieldsOf,
  formatOf,
  foundElsewhere,
  guessField,
  hasFields,
  histogramBars,
  nameFromFile,
  nameProblem,
  readingText,
  wholeNumber,
} from "../lib/corpora";
import { bytes, count, num, plural, visibleToken } from "../lib/format";
import { modelRefOf } from "../lib/spec";
import { useStore } from "../store/app";
import c from "./CorpusView.module.css";
import { openInText } from "./TextView";
import s from "./views.module.css";

type AddMode = "none" | "file" | "hub" | "prompts";
type UnitKind = TopUnit["kind"];

// What was chosen and found, kept while the app is open so coming back shows it again. A reading
// belongs to the request that made it (the model included): another request hides it.
const kept: {
  corpus: string | null;
  unit: UnitKind;
  layer: string;
  neuron: string;
  feature: string;
  direction: string | null;
  head: string;
  texts: string;
  maxTokens: string | null;
  bos: boolean | null;
  batchSize: string;
  topK: string;
  context: string;
  result: CorpusTop | null;
  request: string | null;
} = {
  corpus: null,
  unit: "neuron",
  layer: "0",
  neuron: "0",
  feature: "0",
  direction: null,
  head: "0",
  texts: "1000",
  maxTokens: null,
  bos: null,
  batchSize: "16",
  topK: "20",
  context: "8",
  result: null,
  request: null,
};

/** The project's corpora, newest list after every change (here, in another tab, or by a job). */
function useCorpora(): { corpora: CorpusListing[] | null; error: string | null; reload: () => void } {
  const project = useStore((st) => st.project?.session_id);
  const version = useStore((st) => st.corporaVersion);
  const [corpora, setCorpora] = useState<CorpusListing[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [asked, setAsked] = useState(0);
  useEffect(() => {
    let current = true;
    api.corpora().then(
      (list) => {
        if (!current) return;
        setCorpora(list);
        setError(null);
      },
      (e: Error) => current && setError(e.message),
    );
    return () => {
      current = false;
    };
  }, [project, version, asked]);
  return { corpora, error, reload: () => setAsked((n) => n + 1) };
}

/** The project's saved directions, read again whenever they change. */
export function useDirections(): DirectionListing[] {
  const project = useStore((st) => st.project?.session_id);
  const version = useStore((st) => st.directionsVersion);
  const [directions, setDirections] = useState<DirectionListing[]>([]);
  useEffect(() => {
    let current = true;
    api.directions().then(
      (list) => current && setDirections(list),
      () => current && setDirections([]),
    );
    return () => {
      current = false;
    };
  }, [project, version]);
  return directions;
}

/** Plain texts in the project, and where a unit (a neuron, an SAE feature, a direction, a head)
 * is most active in them. */
export function CorpusView() {
  const { corpora, error, reload } = useCorpora();
  const [chosen, setChosen] = useState<string | null>(kept.corpus);
  const [mode, setMode] = useState<AddMode>("none");
  const readable = (corpora ?? []).filter((x) => !x.error);
  const corpus = readable.find((x) => x.name === chosen) ?? null;
  const taken = useMemo(() => new Set((corpora ?? []).map((x) => x.name)), [corpora]);

  // The corpus chosen before, or else the first; one deleted elsewhere gives way to the next.
  useEffect(() => {
    if (!corpora) return;
    if (!chosen || !corpora.some((x) => x.name === chosen)) {
      const first = readable[0]?.name ?? null;
      setChosen(first);
      kept.corpus = first;
    }
  }, [corpora]);

  const choose = (name: string | null) => {
    setChosen(name);
    kept.corpus = name;
  };
  const created = (name: string) => {
    choose(name);
    setMode("none");
    reload();
  };
  const broken = (corpora ?? []).find((x) => x.name === chosen && x.error);

  return (
    <div className={s.view}>
      <div className={s.head}>
        <div className={s.titleBlock}>
          <h1 className={s.title}>{corpus ? `Where a unit is most active in ${corpus.name}` : "Where a unit is most active in plain text"}</h1>
          <p className={s.subtitle}>
            A corpus is plain text the model reads as it would any text: a file of yours, a Hugging Face dataset, or a
            dataset's prompts. Choose a unit (an MLP neuron, a feature of the loaded SAE, a saved direction or an attention
            head) to find the texts and tokens where it is most active.
          </p>
        </div>
      </div>

      <div className={s.row} style={{ alignItems: "center" }}>
        <div className={s.tabsInline} role="group" aria-label="Corpora">
          {(corpora ?? []).map((x) => {
            const button = (
              <button key={x.name} type="button" className={s.datasetButton} aria-pressed={x.name === chosen} onClick={() => choose(x.name)}>
                {x.name}
                <span>{x.error ? "unreadable" : plural(x.n ?? 0, "text")}</span>
              </button>
            );
            return x.error ? (
              <Tip key={x.name} text={x.error}>
                {button}
              </Tip>
            ) : (
              button
            );
          })}
        </div>
        <div style={{ flex: 1 }} />
        <Segmented<AddMode>
          label="Add texts"
          value={mode}
          onChange={(m) => setMode(m === mode ? "none" : m)}
          options={[
            { value: "file", label: "Upload a file" },
            { value: "hub", label: "Hugging Face" },
            { value: "prompts", label: "From prompts" },
          ]}
        />
      </div>
      {error && <Callout tone="error">{error}</Callout>}
      {broken && (
        <Callout tone="error" title={`${broken.name} can't be read`}>
          {broken.error} Import its texts again under another name, or delete corpora/{broken.name}.jsonl.
        </Callout>
      )}

      {mode === "file" && <FileImport taken={taken} onCreated={created} />}
      {mode === "hub" && <HubImport taken={taken} onCreated={created} />}
      {mode === "prompts" && <PromptsImport taken={taken} onCreated={created} />}

      {corpora && corpora.length === 0 && mode === "none" && (
        <Empty title="No corpus yet" action={<Button onClick={() => setMode("file")}>Upload a file</Button>}>
          Add texts with the buttons above: a text, JSON Lines or CSV file, one file of a Hugging Face dataset, or the prompts
          of a dataset in this project.
        </Empty>
      )}

      {corpus && <CorpusDetails key={corpus.name} corpus={corpus} taken={taken} onRenamed={choose} />}
      {corpus && <MostActive key={`top-${corpus.name}`} corpus={corpus} />}
    </div>
  );
}

// -- adding texts ------------------------------------------------------------------------------

/** Run a form's action; return its result, or keep its error to show beside the form. */
async function attempt<T>(fn: () => Promise<T>, setError: (e: string | null) => void): Promise<T | undefined> {
  setError(null);
  try {
    return await fn();
  } catch (e) {
    setError(e instanceof Error ? e.message : String(e));
    return undefined;
  }
}

const FORMAT_OPTIONS: { value: CorpusFormat; label: string }[] = [
  { value: "lines", label: "Text, one per line" },
  { value: "paragraphs", label: "Text, separated by blank lines" },
  { value: "jsonl", label: "JSON Lines" },
  { value: "json", label: "A JSON array" },
  { value: "csv", label: "CSV" },
  { value: "tsv", label: "TSV" },
];

function FieldChoices({ fields, text, label, onText, onLabel }: { fields: string[]; text: string; label: string | null; onText: (v: string) => void; onLabel: (v: string | null) => void }) {
  return (
    <>
      <Field label="Text field" help="The field or column that holds each text.">
        {fields.length ? (
          <Select value={text} onChange={(e) => onText(e.target.value)}>
            {fields.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </Select>
        ) : (
          <Input value={text} onChange={(e) => onText(e.target.value)} spellCheck={false} />
        )}
      </Field>
      <Field label="Label field" help="Optional: a class for each text, such as a sentiment.">
        {fields.length ? (
          <Select value={label ?? ""} onChange={(e) => onLabel(e.target.value || null)}>
            <option value="">No label</option>
            {fields.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </Select>
        ) : (
          <Input value={label ?? ""} placeholder="None" onChange={(e) => onLabel(e.target.value.trim() ? e.target.value : null)} spellCheck={false} />
        )}
      </Field>
    </>
  );
}

function FileImport({ taken, onCreated }: { taken: ReadonlySet<string>; onCreated: (name: string) => void }) {
  const [file, setFile] = useState<{ name: string; size: number; text: string } | null>(null);
  const [format, setFormat] = useState<CorpusFormat>("lines");
  const [textField, setTextField] = useState("text");
  const [labelField, setLabelField] = useState<string | null>(null);
  const [limit, setLimit] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const fields = useMemo(() => (file && hasFields(format) ? fieldsOf(file.text, format) : []), [file, format]);
  const problem = !file ? "Choose a file first." : nameProblem(name, taken) ?? (limit.trim() && (wholeNumber(limit) ?? 0) < 1 ? "Keep at least one text, or leave the limit empty." : null);

  const choose = async (chosen: File) => {
    setError(null);
    const fmt = formatOf(chosen.name);
    if (!fmt) {
      setError(`${chosen.name} can't be uploaded: Parquet and compressed files are read from Hugging Face, or with logogram corpora import.`);
      return;
    }
    const text = await chosen.text();
    setFile({ name: chosen.name, size: chosen.size, text });
    setFormat(fmt);
    const found = hasFields(fmt) ? fieldsOf(text, fmt) : [];
    setTextField(guessField(found, "text") ?? "text");
    setLabelField(guessField(found, "label"));
    setName(nameFromFile(chosen.name));
  };

  const submit = async () => {
    if (problem || !file) return;
    setBusy(true);
    const out = await attempt(
      () =>
        api.importCorpus({
          name: name.trim(),
          text: file.text,
          filename: file.name,
          format,
          text_field: textField,
          label_field: hasFields(format) ? labelField : null,
          limit: limit.trim() ? wholeNumber(limit) : null,
        }),
      setError,
    );
    setBusy(false);
    if (out) onCreated(out.name);
  };

  return (
    <section className={s.panel} aria-labelledby="corpus-file">
      <h2 id="corpus-file" className={s.panelTitle}>
        Upload a file
      </h2>
      <p className={s.small}>
        A text file (one text per line, or texts separated by blank lines), JSON Lines or a JSON array of objects, or CSV or
        TSV with a header row. Only the file's name is kept, never its folder.
      </p>
      <div className={s.row}>
        <input
          ref={input}
          type="file"
          accept=".txt,.text,.jsonl,.ndjson,.json,.csv,.tsv"
          className="visually-hidden"
          tabIndex={-1}
          aria-label="The file to import as a corpus"
          onChange={(e) => {
            const chosen = e.target.files?.[0];
            if (chosen) void choose(chosen);
            e.target.value = "";
          }}
        />
        <Button icon="file" onClick={() => input.current?.click()}>
          Choose a file…
        </Button>
        {file && (
          <span className={s.small}>
            {file.name} · {bytes(file.size)}
          </span>
        )}
      </div>
      {file && (
        <div className={c.grid}>
          <Field label="Laid out as">
            <Select value={format} onChange={(e) => setFormat(e.target.value as CorpusFormat)}>
              {FORMAT_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </Select>
          </Field>
          {hasFields(format) && <FieldChoices fields={fields} text={textField} label={labelField} onText={setTextField} onLabel={setLabelField} />}
          <Field label="Keep the first" help="Texts; empty keeps every one.">
            <Input inputMode="numeric" value={limit} placeholder="Every text" onChange={(e) => setLimit(e.target.value)} />
          </Field>
          <Field label="Corpus name">
            <Input value={name} spellCheck={false} onChange={(e) => setName(e.target.value)} />
          </Field>
        </div>
      )}
      <div className={s.row}>
        <Button variant="primary" disabled={!!problem || busy} tip={problem ?? undefined} onClick={() => void submit()}>
          {busy ? "Importing…" : "Import"}
        </Button>
        {busy && <Spinner label="Importing the file" />}
      </div>
      {error && <Callout tone="error">{error}</Callout>}
    </section>
  );
}

function HubImport({ taken, onCreated }: { taken: ReadonlySet<string>; onCreated: (name: string) => void }) {
  const job = useStore((st) => st.job);
  const [repo, setRepo] = useState("");
  const [revision, setRevision] = useState("");
  const [listing, setListing] = useState<HubDatasetFiles | null>(null);
  const [path, setPath] = useState("");
  const [split, setSplit] = useState<"lines" | "paragraphs">("lines");
  const [textField, setTextField] = useState("text");
  const [labelField, setLabelField] = useState("");
  const [limit, setLimit] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [listingBusy, setListingBusy] = useState(false);
  const [started, setStarted] = useState<string | null>(null);
  const mine = job && started === job.id ? job : null;
  const running = mine?.status === "running";
  const isText = /\.(txt|text)$/i.test(path);
  const problem =
    !repo.trim() ? "Type the dataset's id, such as stanfordnlp/imdb." : !path.trim() ? "Choose a file of the dataset (List its files)." : nameProblem(name, taken) ?? (limit.trim() && (wholeNumber(limit) ?? 0) < 1 ? "Keep at least one text, or leave the limit empty." : null);

  // The job ends: the corpus is there (the list is read again), or it says why not.
  useEffect(() => {
    if (!mine || mine.status === "running") return;
    if (mine.status === "finished") onCreated(String(mine.progress.corpus ?? name.trim()));
    else if (mine.status === "failed") setError(mine.error ?? "The import failed.");
    else if (mine.status === "cancelled") setError("The import was cancelled. Nothing was written; what was downloaded stays in the Hugging Face cache.");
    setStarted(null);
  }, [mine?.status]);

  const list = async () => {
    setListingBusy(true);
    const out = await attempt(() => api.hubDatasetFiles(repo.trim(), revision.trim() || null), setError);
    setListingBusy(false);
    if (!out) return;
    setListing(out);
    const first = out.files.find((f) => f.readable);
    if (first && !path) setPath(first.path);
    if (!name) setName(nameFromFile(repo.trim().split("/").pop() ?? "dataset"));
  };

  const submit = async () => {
    if (problem) return;
    const out = await attempt(
      () =>
        api.importHubCorpus({
          name: name.trim(),
          repo: repo.trim(),
          path: path.trim(),
          revision: revision.trim() || null,
          split,
          text_field: textField.trim() || "text",
          label_field: labelField.trim() || null,
          limit: limit.trim() ? wholeNumber(limit) : null,
        }),
      setError,
    );
    if (out) {
      setStarted(out.id);
      useStore.getState().applyJob(out);
    }
  };

  const progress = mine?.progress as { done?: number; total?: number } | undefined;
  return (
    <section className={s.panel} aria-labelledby="corpus-hub">
      <h2 id="corpus-hub" className={s.panelTitle}>
        A file of a Hugging Face dataset
      </h2>
      <p className={s.small}>
        Logogram asks Hugging Face for the dataset's files only when you list them, and downloads the one you choose once,
        at the exact commit its revision names, which the corpus records. Gated and private datasets need{" "}
        <code>hf auth login</code> on this machine first.
      </p>
      <div className={c.grid}>
        <Field label="Dataset">
          <Input value={repo} placeholder="owner/name" spellCheck={false} onChange={(e) => setRepo(e.target.value)} />
        </Field>
        <Field label="Revision" help="A commit, branch or tag; empty for main.">
          <Input value={revision} placeholder="main" spellCheck={false} onChange={(e) => setRevision(e.target.value)} />
        </Field>
        <div className={c.end}>
          <Button disabled={!repo.trim() || listingBusy} onClick={() => void list()}>
            {listingBusy ? "Listing…" : "List its files"}
          </Button>
        </div>
      </div>
      {listing && (
        <p className={s.small}>
          {listing.repo} at {listing.revision.slice(0, 7)}: {plural(listing.files.length, "file")}, {plural(listing.files.filter((f) => f.readable).length, "file")} Logogram reads.
        </p>
      )}
      <div className={c.grid}>
        <Field label="File" help="Parquet, JSON Lines, JSON, CSV, TSV or text, by its path in the dataset.">
          {listing ? (
            <Select value={path} onChange={(e) => setPath(e.target.value)}>
              {listing.files
                .filter((f) => f.readable)
                .map((f) => (
                  <option key={f.path} value={f.path}>
                    {f.path} · {bytes(f.size)}
                  </option>
                ))}
            </Select>
          ) : (
            <Input value={path} placeholder="data/train-00000-of-00001.parquet" spellCheck={false} onChange={(e) => setPath(e.target.value)} />
          )}
        </Field>
        {isText ? (
          <Field label="Texts">
            <Select value={split} onChange={(e) => setSplit(e.target.value as "lines" | "paragraphs")}>
              <option value="lines">One per line</option>
              <option value="paragraphs">Separated by blank lines</option>
            </Select>
          </Field>
        ) : (
          <>
            <Field label="Text column">
              <Input value={textField} spellCheck={false} onChange={(e) => setTextField(e.target.value)} />
            </Field>
            <Field label="Label column" help="Optional.">
              <Input value={labelField} placeholder="None" spellCheck={false} onChange={(e) => setLabelField(e.target.value)} />
            </Field>
          </>
        )}
        <Field label="Keep the first" help="Texts; empty keeps every one.">
          <Input inputMode="numeric" value={limit} placeholder="Every text" onChange={(e) => setLimit(e.target.value)} />
        </Field>
        <Field label="Corpus name">
          <Input value={name} spellCheck={false} onChange={(e) => setName(e.target.value)} />
        </Field>
      </div>
      <div className={s.row}>
        <Button variant="primary" disabled={!!problem || running || (!!job && job.status === "running")} tip={problem ?? (job?.status === "running" && !running ? `“${job.title}” is running. Wait for it to finish.` : undefined)} onClick={() => void submit()}>
          {running ? "Importing…" : "Download and import"}
        </Button>
        {running && (
          <>
            <div className={c.progress}>
              <Progress value={progress?.total ? (progress.done ?? 0) / progress.total : null} label="Downloading the file" />
            </div>
            <span className={s.small}>{progress?.total ? `${bytes(progress.done ?? 0)} of ${bytes(progress.total)}` : "Starting"}</span>
            <Button size="small" onClick={() => void api.cancelJob()} disabled={mine?.cancelling}>
              {mine?.cancelling ? "Cancelling…" : "Cancel"}
            </Button>
          </>
        )}
      </div>
      {error && <Callout tone="error">{error}</Callout>}
    </section>
  );
}

function PromptsImport({ taken, onCreated }: { taken: ReadonlySet<string>; onCreated: (name: string) => void }) {
  const datasets = useStore((st) => st.project?.datasets ?? []);
  const usable = datasets.filter((d) => !d.error);
  const [path, setPath] = useState(usable[0]?.path ?? "");
  const [sides, setSides] = useState<"clean" | "corrupt" | "both">("both");
  const [label, setLabel] = useState("side");
  const [limit, setLimit] = useState("");
  const [name, setName] = useState("");
  const [meta, setMeta] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const stem = (path.split("/").pop() ?? "").replace(/\.jsonl$/, "");
  const suggested = `${stem || "prompts"}-${sides === "both" ? "prompts" : sides}`;
  const shown = name || suggested;
  const problem = !path ? "Add a dataset in Prompts first." : nameProblem(shown, taken) ?? (limit.trim() && (wholeNumber(limit) ?? 0) < 1 ? "Take at least one pair, or leave the limit empty." : null);

  // The fields of meta every prompt has, to label the texts by.
  useEffect(() => {
    setMeta([]);
    if (!path) return;
    let current = true;
    api.dataset(path).then(
      (d) => current && setMeta(datasetFacts(d.records).metaFields),
      () => undefined,
    );
    return () => {
      current = false;
    };
  }, [path]);

  const submit = async () => {
    if (problem) return;
    setBusy(true);
    const out = await attempt(
      () => api.corpusFromPrompts({ name: shown.trim(), dataset: path, sides, label: label || null, limit: limit.trim() ? wholeNumber(limit) : null }),
      setError,
    );
    setBusy(false);
    if (out) onCreated(out.name);
  };

  return (
    <section className={s.panel} aria-labelledby="corpus-prompts">
      <h2 id="corpus-prompts" className={s.panelTitle}>
        A dataset's prompts as texts
      </h2>
      <div className={c.grid}>
        <Field label="Dataset">
          <Select value={path} onChange={(e) => setPath(e.target.value)}>
            {usable.map((d) => (
              <option key={d.path} value={d.path}>
                {d.name}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Prompts">
          <Segmented
            label="Prompts"
            value={sides}
            onChange={setSides}
            options={[
              { value: "clean", label: "Clean" },
              { value: "corrupt", label: "Corrupt" },
              { value: "both", label: "Both" },
            ]}
          />
        </Field>
        <Field label="Label each text by">
          <Select value={label} onChange={(e) => setLabel(e.target.value)}>
            <option value="side">Its side (clean or corrupt)</option>
            {meta.map((m) => (
              <option key={m} value={`meta:${m}`}>
                Its meta's {m}
              </option>
            ))}
            <option value="">No label</option>
          </Select>
        </Field>
        <Field label="Keep the first" help="Prompt pairs; empty keeps every one.">
          <Input inputMode="numeric" value={limit} placeholder="Every pair" onChange={(e) => setLimit(e.target.value)} />
        </Field>
        <Field label="Corpus name">
          <Input value={shown} spellCheck={false} onChange={(e) => setName(e.target.value)} />
        </Field>
      </div>
      <div className={s.row}>
        <Button variant="primary" disabled={!!problem || busy} tip={problem ?? undefined} onClick={() => void submit()}>
          {busy ? "Importing…" : "Import"}
        </Button>
      </div>
      {error && <Callout tone="error">{error}</Callout>}
    </section>
  );
}

// -- the corpus chosen -------------------------------------------------------------------------

function CorpusDetails({ corpus, taken, onRenamed }: { corpus: CorpusListing; taken: ReadonlySet<string>; onRenamed: (name: string | null) => void }) {
  const [renaming, setRenaming] = useState(false);
  const [name, setName] = useState(corpus.name);
  const [confirm, setConfirm] = useState(false);
  const [showTexts, setShowTexts] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const others = useMemo(() => new Set([...taken].filter((n) => n !== corpus.name)), [taken, corpus.name]);
  const problem = name.trim() === corpus.name ? "Type a new name." : nameProblem(name, others);
  const source = corpus.source;

  const rename = async () => {
    if (problem) return;
    const out = await attempt(() => api.renameCorpus(corpus.name, name.trim()), setError);
    if (out) {
      setRenaming(false);
      onRenamed(out.name);
    }
  };
  const remove = async () => {
    setConfirm(false);
    const out = await attempt(() => api.deleteCorpus(corpus.name), setError);
    if (out) onRenamed(null);
  };

  return (
    <section className={c.details} aria-label={`The corpus ${corpus.name}`}>
      <p className={s.faint}>
        {corpus.path} · {plural(corpus.n ?? 0, "text")}
        {corpus.labels?.length ? ` · labels ${corpus.labels.slice(0, 8).join(", ")}${corpus.labels.length > 8 ? ` and ${count(corpus.labels.length - 8)} more` : ""}` : " · no labels"}
        {" · "}sha256 {corpus.sha256?.slice(0, 12)}
      </p>
      {source && <p className={s.small}>From {source.detail}.</p>}
      {renaming ? (
        <form
          className={s.row}
          onSubmit={(e) => {
            e.preventDefault();
            void rename();
          }}
        >
          <Field label="New name" help={problem && name.trim() !== corpus.name ? problem : undefined}>
            <Input value={name} spellCheck={false} autoFocus onChange={(e) => setName(e.target.value)} />
          </Field>
          <Button variant="primary" type="submit" disabled={!!problem}>
            Rename
          </Button>
          <Button onClick={() => setRenaming(false)}>Keep the name</Button>
        </form>
      ) : (
        <div className={s.row}>
          <Button size="small" aria-expanded={showTexts} onClick={() => setShowTexts(!showTexts)}>
            {showTexts ? "Hide the texts" : "Show the texts"}
          </Button>
          <Button size="small" variant="ghost" onClick={() => setRenaming(true)}>
            Rename…
          </Button>
          <Button size="small" variant="ghost" onClick={() => setConfirm(true)}>
            Delete…
          </Button>
        </div>
      )}
      {error && <Callout tone="error">{error}</Callout>}
      {showTexts && <CorpusTexts name={corpus.name} />}
      <ConfirmDialog open={confirm} onOpenChange={setConfirm} title={`Delete the corpus ${corpus.name}?`} keepLabel="Keep it" confirmLabel="Delete the corpus" onConfirm={() => void remove()}>
        {corpus.path} and its description are deleted, {plural(corpus.n ?? 0, "text")}. It can't be undone.
      </ConfirmDialog>
    </section>
  );
}

/** A corpus's texts, a hundred at a time, found by a search through their text, label and meta. */
function CorpusTexts({ name }: { name: string }) {
  const [query, setQuery] = useState("");
  const deferred = useDeferredValue(query);
  const [page, setPage] = useState(0);
  const [data, setData] = useState<CorpusPage | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let current = true;
    const timer = window.setTimeout(() => {
      api.corpus(name, deferred, page).then(
        (d) => {
          if (!current) return;
          setData(d);
          setError(null);
        },
        (e: Error) => current && setError(e.message),
      );
    }, 150);
    return () => {
      current = false;
      window.clearTimeout(timer);
    };
  }, [name, deferred, page]);
  const range = data && data.matches > 0 ? `${count(data.start + 1)}–${count(data.start + data.texts.length)} of ${count(data.matches)}` : "";
  return (
    <div className={c.texts}>
      <div className={c.toolbar}>
        <div className={c.search}>
          <Icon name="search" size={14} />
          <Input
            type="search"
            aria-label="Search the texts"
            placeholder="Search texts, labels and meta"
            spellCheck={false}
            value={query}
            data-owns-escape
            onChange={(e) => {
              setQuery(e.target.value);
              setPage(0);
            }}
          />
        </div>
        <span className={s.small} aria-live="polite">
          {data ? (deferred.trim() ? `${plural(data.matches, "text")} found${range ? ` · ${range}` : ""}` : range && `Texts ${range}`) : ""}
        </span>
        {data && data.pages > 1 && (
          <div className={s.row}>
            <Button size="small" icon="chevronLeft" disabled={data.page === 0} onClick={() => setPage(data.page - 1)}>
              Previous
            </Button>
            <Button size="small" iconAfter="chevronRight" disabled={data.page >= data.pages - 1} onClick={() => setPage(data.page + 1)}>
              Next
            </Button>
          </div>
        )}
      </div>
      {error && <Callout tone="error">{error}</Callout>}
      {data && data.texts.length === 0 && <p className={s.small}>No text contains “{deferred.trim()}”.</p>}
      {data && data.texts.length > 0 && (
        <ol className={c.textList} aria-label={`Texts of ${name}`}>
          {data.texts.map((t) => (
            <li key={t.index} className={c.textRow}>
              <span className={c.textIndex}>{t.index}</span>
              <span className={c.text}>
                {t.text}
                {t.length > t.text.length && <span className={s.faint}> … ({count(t.length)} characters in all)</span>}
              </span>
              {t.label !== null && <span className={c.label}>{t.label}</span>}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

// -- where a unit is most active ------------------------------------------------------------------

const UNIT_OPTIONS: { value: UnitKind; label: string }[] = [
  { value: "neuron", label: "MLP neuron" },
  { value: "sae_feature", label: "SAE feature" },
  { value: "direction", label: "Direction" },
  { value: "head", label: "Attention head" },
];

function MostActive({ corpus }: { corpus: CorpusListing }) {
  const model = useStore((st) => st.model);
  const sae = useStore((st) => st.sae);
  const progress = useStore((st) => st.analysisProgress);
  const directions = useDirections();
  const info = model.info;
  const [unit, setUnit] = useState<UnitKind>(kept.unit);
  const [layer, setLayer] = useState(kept.layer);
  const [neuron, setNeuron] = useState(kept.neuron);
  const [feature, setFeature] = useState(kept.feature);
  const [direction, setDirection] = useState<string | null>(kept.direction);
  const [head, setHead] = useState(kept.head);
  const [texts, setTexts] = useState(kept.texts);
  const [maxTokens, setMaxTokens] = useState(kept.maxTokens ?? String(Math.min(128, info?.n_ctx ?? 128)));
  const [bos, setBos] = useState(kept.bos ?? info?.extra?.bos !== false);
  const [batchSize, setBatchSize] = useState(kept.batchSize);
  const [topK, setTopK] = useState(kept.topK);
  const [context, setContext] = useState(kept.context);
  const [result, setResult] = useState<CorpusTop | null>(kept.result);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const theme = useStore((st) => st.theme);

  const usable = directions.filter((d) => !directionProblem(d, info));
  const chosenDirection = usable.find((d) => d.name === direction) ?? usable[0] ?? null;
  const neurons = !!info?.site_kinds.includes("neuron");
  const saeReady = sae.state === "ready" && !!sae.info;
  const options = UNIT_OPTIONS.map((o) => {
    if (o.value === "neuron" && info && !neurons) return { ...o, disabled: true, reason: info.extra?.site_notes?.neuron ?? "This model's neurons weren't measured when it loaded." };
    if (o.value === "sae_feature" && !saeReady) return { ...o, disabled: true, reason: "Load an SAE in Features to read one of its features." };
    if (o.value === "direction" && usable.length === 0) return { ...o, disabled: true, reason: "Save a direction first: a principal component or a mean difference in Representations." };
    return o;
  });

  const request = useMemo((): { body: CorpusTopRequest | null; problem: string | null } => {
    if (!info) return { body: null, problem: "Load a model first." };
    const whole = (text: string, lo: number, hi: number, what: string): number | string => {
      const n = wholeNumber(text);
      return n === null || n < lo || n > hi ? `${what} is a whole number from ${count(lo)} to ${count(hi)}.` : n;
    };
    let chosen: TopUnit;
    const l = whole(layer, 0, info.n_layers - 1, "The layer");
    if (unit === "neuron") {
      const n = whole(neuron, 0, (info.d_mlp ?? 1) - 1, "The neuron");
      if (typeof l === "string") return { body: null, problem: l };
      if (typeof n === "string") return { body: null, problem: n };
      chosen = { kind: "neuron", layer: l, neuron: n };
    } else if (unit === "head") {
      const h = whole(head, 0, info.n_heads - 1, "The head");
      if (typeof l === "string") return { body: null, problem: l };
      if (typeof h === "string") return { body: null, problem: h };
      chosen = { kind: "head", layer: l, head: h };
    } else if (unit === "sae_feature") {
      if (!sae.info) return { body: null, problem: "Load an SAE in Features first." };
      const f = whole(feature, 0, sae.info.d_sae - 1, "The feature");
      if (typeof f === "string") return { body: null, problem: f };
      chosen = { kind: "sae_feature", feature: f };
    } else {
      if (!chosenDirection) return { body: null, problem: "Save a direction first." };
      if (!chosenDirection.site) return { body: null, problem: `The direction ${chosenDirection.name} doesn't say where it was found, so it can't be read here.` };
      chosen = { kind: "direction", name: chosenDirection.name, sha256: chosenDirection.sha256 ?? null };
    }
    const nTexts = texts.trim() ? whole(texts, 1, 10_000_000, "The number of texts") : null;
    const tokens = whole(maxTokens, bos ? 2 : 1, info.n_ctx, "Tokens of each text");
    const batch = whole(batchSize, 1, 4096, "The batch size");
    const k = whole(topK, 1, 50, "The number of examples");
    const ctx = whole(context, 0, 32, "The context");
    for (const v of [nTexts, tokens, batch, k, ctx]) if (typeof v === "string") return { body: null, problem: v };
    return {
      body: {
        corpus: corpus.name,
        corpus_sha256: corpus.sha256 ?? null,
        unit: chosen,
        texts: nTexts as number | null,
        max_tokens: tokens as number,
        top_k: k as number,
        context: ctx as number,
        prepend_bos: bos,
        batch_size: batch as number,
        model: modelRefOf(info),
      },
      problem: null,
    };
  }, [info, unit, layer, neuron, feature, head, chosenDirection, sae.info, texts, maxTokens, bos, batchSize, topK, context, corpus.name, corpus.sha256]);
  const key = request.body ? JSON.stringify(request.body) : null;
  // A reading of another corpus isn't this one's, stale or not.
  const shown = result && kept.request !== null && result.corpus.name === corpus.name ? result : null;
  const stale = !!shown && key !== kept.request;

  useEffect(() => {
    Object.assign(kept, { unit, layer, neuron, feature, direction: chosenDirection?.name ?? direction, head, texts, maxTokens, bos, batchSize, topK, context });
  }, [unit, layer, neuron, feature, direction, chosenDirection, head, texts, maxTokens, bos, batchSize, topK, context]);

  if (model.state !== "ready" || !info) {
    return (
      <section className={c.section} aria-labelledby="most-active">
        <h2 id="most-active" className={s.panelTitle}>
          Where a unit is most active
        </h2>
        <Empty title="Load a model to read the texts" align="start" action={<Button onClick={() => useStore.setState({ modelDialogOpen: true })}>Load a model</Button>}>
          The model reads the corpus's texts here; nothing in the model changes.
        </Empty>
      </section>
    );
  }

  const run = async () => {
    if (!request.body || !key) return;
    setBusy(true);
    setError(null);
    useStore.setState({ analysisProgress: null });
    try {
      const out = await api.corpusTop(request.body);
      kept.result = out;
      kept.request = key;
      setResult(out);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const reading = busy && progress?.what === "corpus.top" ? progress : null;
  return (
    <section className={c.section} aria-labelledby="most-active">
      <h2 id="most-active" className={s.panelTitle}>
        Where a unit is most active
      </h2>
      <Segmented<UnitKind> label="Unit" value={unit} onChange={setUnit} options={options} />
      <div className={c.grid}>
        {(unit === "neuron" || unit === "head") && (
          <Field label="Layer">
            <Input type="number" min={0} max={info.n_layers - 1} value={layer} onChange={(e) => setLayer(e.target.value)} />
          </Field>
        )}
        {unit === "neuron" && (
          <Field label="Neuron" help={info.d_mlp ? `0 to ${count(info.d_mlp - 1)}` : undefined}>
            <Input type="number" min={0} value={neuron} onChange={(e) => setNeuron(e.target.value)} />
          </Field>
        )}
        {unit === "head" && (
          <Field label="Head">
            <Input type="number" min={0} max={info.n_heads - 1} value={head} onChange={(e) => setHead(e.target.value)} />
          </Field>
        )}
        {unit === "sae_feature" && sae.info && (
          <Field label="Feature" help={`0 to ${count(sae.info.d_sae - 1)} of ${sae.info.repo}`}>
            <Input type="number" min={0} value={feature} onChange={(e) => setFeature(e.target.value)} />
          </Field>
        )}
        {unit === "direction" && (
          <Field label="Direction" help={chosenDirection?.site ? `Read at ${chosenDirection.site.kind}:${chosenDirection.site.layer}, where it was found${foundElsewhere(chosenDirection, info) ? "; found on another model" : ""}.` : undefined}>
            <Select value={chosenDirection?.name ?? ""} onChange={(e) => setDirection(e.target.value)}>
              {usable.map((d) => (
                <option key={d.name} value={d.name}>
                  {d.name}
                  {d.dimensions && d.dimensions > 1 ? ` (${d.dimensions} dimensions)` : ""}
                </option>
              ))}
            </Select>
          </Field>
        )}
      </div>
      <div className={c.grid}>
        <Field label="Texts to read" help={`The first ones; empty reads all ${count(corpus.n ?? 0)}.`}>
          <Input inputMode="numeric" value={texts} placeholder="Every text" onChange={(e) => setTexts(e.target.value)} />
        </Field>
        <Field label="Tokens of each text" help={`At most ${count(info.n_ctx)}; longer texts are cut.`}>
          <Input type="number" min={1} max={info.n_ctx} value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)} />
        </Field>
        <Field label="Batch size">
          <Input type="number" min={1} value={batchSize} onChange={(e) => setBatchSize(e.target.value)} />
        </Field>
        <Field label="Examples" help="The most active tokens shown.">
          <Input type="number" min={1} max={50} value={topK} onChange={(e) => setTopK(e.target.value)} />
        </Field>
        <Field label="Context" help="Tokens shown on each side.">
          <Input type="number" min={0} max={32} value={context} onChange={(e) => setContext(e.target.value)} />
        </Field>
      </div>
      <Checkbox checked={bos} onChange={setBos} disabled={info.extra?.bos === false}>
        Begin each text with the beginning-of-sequence token (read, but never counted)
      </Checkbox>
      <div className={s.row}>
        <Button variant="primary" disabled={!!request.problem || busy} tip={request.problem ?? undefined} onClick={() => void run()}>
          {busy ? "Reading…" : "Find where it is most active"}
        </Button>
        {busy && (
          <>
            <div className={c.progress}>
              <Progress value={reading && reading.total ? reading.done / reading.total : null} label="Reading the texts" />
            </div>
            <span className={s.small}>{reading ? `${count(reading.done)} of ${count(reading.total)} texts` : "Tokenizing"}</span>
          </>
        )}
      </div>
      {error && <Callout tone="error">{error}</Callout>}
      {stale && <p className={s.small}>The unit or the choices changed since this was read: press Find to read it again.</p>}
      {shown && !stale && <TopResult result={shown} theme={theme} />}
    </section>
  );
}

function TopResult({ result, theme }: { result: CorpusTop; theme: "light" | "dark" }) {
  const bound = exampleBound(result.top, result.bottom);
  const color = useMemo(() => inkScale(bound, theme), [bound, theme]);
  const bos = result.settings.prepend_bos;
  return (
    <div className={c.result}>
      <p className={s.sentence}>
        <strong>{result.description}.</strong> {readingText(result)} Its mean is {num(result.mean, 3)}, from {num(result.min, 3)} to {num(result.max, 3)}.
      </p>
      {result.notes.map((note) => (
        <Callout key={note}>{note}</Callout>
      ))}
      <Histogram result={result} />
      <div className={s.row}>
        <ScaleBar bound={bound} theme={theme} kind="ink" label={result.signed ? "Value at each token (its size, for the most negative)" : "Value at each token"} />
      </div>
      <Examples title={`The ${count(result.top.length)} tokens where it is most active`} examples={result.top} color={color} bos={bos} />
      {result.bottom && <Examples title={`The ${count(result.bottom.length)} tokens where it is most negative`} examples={result.bottom} color={color} bos={bos} />}
    </div>
  );
}

function Examples({ title, examples, color, bos }: { title: string; examples: TopExample[]; color: (v: number) => string; bos: boolean }) {
  const id = useId();
  return (
    <section className={c.examples} aria-labelledby={id}>
      <h3 id={id} className={c.examplesTitle}>
        {title}
      </h3>
      <ol className={c.exampleList}>
        {examples.map((e, rank) => {
          const at = e.position - e.start;
          const plain = e.tokens.map((t, i) => (i === at ? `[${t}]` : t)).join("");
          return (
            <li key={`${e.text}-${e.position}`} className={c.example}>
              <div className={c.exampleHead}>
                <span className={c.rank}>{rank + 1}</span>
                <span>
                  Text {e.text}
                  {e.label !== null ? ` · ${e.label}` : ""} · token {e.position}
                </span>
                <span className={c.value}>{num(e.value, 3)}</span>
                {e.open !== null && (
                  <Button size="small" variant="ghost" onClick={() => openInText(e.open ?? "", bos, e.position)} aria-label={`Open text ${e.text} in Your text, at token ${e.position}`}>
                    Open in Your text
                  </Button>
                )}
              </div>
              <p className={c.window}>
                <span className="visually-hidden">
                  The text around token {e.position}: {plain}
                </span>
                {e.tokens.map((token, i) => {
                  const v = e.values[i];
                  const bg = v === null ? "transparent" : color(Math.abs(v));
                  return (
                    <span
                      key={i}
                      aria-hidden="true"
                      className={i === at ? `${c.tok} ${c.focus}` : c.tok}
                      style={v === null ? undefined : { background: bg, color: textOn(bg) }}
                      title={v === null ? "Not counted (the beginning-of-sequence token)" : num(v, 3)}
                    >
                      {visibleToken(token) || "∅"}
                    </span>
                  );
                })}
              </p>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

/** The unit's values over every token counted, as a histogram in ink, with zero marked. */
function Histogram({ result }: { result: CorpusTop }) {
  const colors = useChromeColors();
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const { width } = useElementSize(wrap);
  const height = 84;
  const { edges } = result.histogram;
  const lo = edges[0];
  const hi = edges[edges.length - 1];
  useEffect(() => {
    const el = canvas.current;
    if (!el || width === 0) return;
    const ctx = prepareCanvas(el, width, height);
    if (!ctx) return;
    const plotH = height - 4;
    for (const bar of histogramBars(result.histogram, width, plotH)) {
      if (bar.h === 0) continue;
      ctx.fillStyle = colors.muted;
      ctx.fillRect(bar.x + 0.5, bar.y + 2, Math.max(1, bar.w - 1), bar.h);
    }
    ctx.strokeStyle = colors.line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, plotH + 2.5);
    ctx.lineTo(width, plotH + 2.5);
    ctx.stroke();
    if (lo < 0 && hi > 0) {
      const x = Math.round(((0 - lo) / (hi - lo)) * width) + 0.5;
      ctx.save();
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = colors.text;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, plotH + 2);
      ctx.stroke();
      ctx.restore();
      ctx.fillStyle = colors.muted;
      ctx.font = font(colors, 10, 500);
      ctx.fillText("0", x + 4, 10);
    }
  }, [result, width, colors, lo, hi]);
  const peak = result.histogram.counts.indexOf(Math.max(...result.histogram.counts));
  const label = `Histogram of its values over ${count(result.tokens)} tokens, from ${num(lo, 3)} to ${num(hi, 3)}; the most tokens are between ${num(edges[peak], 3)} and ${num(edges[peak + 1], 3)}.`;
  return (
    <figure className={c.histogram}>
      <div ref={wrap}>
        <canvas ref={canvas} role="img" aria-label={label} />
      </div>
      <figcaption className={c.axis}>
        <span>{num(lo, 3)}</span>
        <span>Its value at each token counted</span>
        <span>{num(hi, 3)}</span>
      </figcaption>
    </figure>
  );
}
