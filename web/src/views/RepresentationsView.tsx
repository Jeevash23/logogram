import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent, type MouseEvent } from "react";

import { api, type RepresentationsRequest } from "../api/client";
import type { ActivationsExported, CorpusListing, DirectionListing, ReadPosition, ReadSite, RepresentationPoint, Representations, RowsSource, StreamKind } from "../api/types";
import { Button, Callout, Checkbox, ConfirmDialog, Empty, Field, Input, Progress, Segmented, Select } from "../components/ui";
import { font, inkRing, prepareCanvas, useChromeColors, useElementSize, type ChromeColors } from "../lib/canvas";
import {
  datasetFacts,
  directionModelText,
  foundElsewhere,
  nameProblem,
  positionWords,
  siteName,
  siteWords,
  STREAM_KINDS,
  wholeNumber,
} from "../lib/corpora";
import { count, num, plural } from "../lib/format";
import {
  classCounts,
  classMarks,
  explainedText,
  exportSites,
  exportSize,
  keyboardOrder,
  MARKED_CLASSES,
  nearestPoint,
  pointLabel,
  pointText,
  scatterScale,
  type ClassMark,
} from "../lib/representations";
import { modelRefOf } from "../lib/spec";
import { useStore } from "../store/app";
import { useDirections } from "./CorpusView";
import r from "./RepresentationsView.module.css";
import s from "./views.module.css";

type SourceKind = "dataset" | "corpus";
type PositionKind = ReadPosition["kind"];

const METHOD_WORDS: Record<string, string> = {
  probe: "A linear probe's weights",
  das: "A subspace learned by distributed alignment search",
  pca: "A principal component",
  mean_difference: "The mean difference between two sets",
  import: "Vectors imported",
  sae_feature: "An SAE feature's decoder direction",
};

const KIND_WORDS: Record<StreamKind, string> = {
  resid_pre: "Residual stream entering the layer",
  resid_mid: "Residual stream between attention and MLP",
  resid_post: "Residual stream leaving the layer",
  attn_out: "Attention output",
  mlp_out: "MLP output",
};

// What was chosen and found, kept while the app is open. A reading belongs to the request that
// made it (the model included): another request hides it until it is read again.
const kept = {
  source: "dataset" as SourceKind,
  dataset: null as string | null,
  sides: "both" as "clean" | "corrupt" | "both",
  mark: "side",
  corpus: null as string | null,
  maxTokens: null as string | null,
  limit: "",
  kind: "resid_post" as StreamKind,
  layer: "0",
  position: "last" as PositionKind,
  index: "-1",
  label: "",
  bos: null as boolean | null,
  batchSize: "16",
  result: null as Representations | null,
  request: null as string | null,
  axes: [0, 1] as [number, number],
};

function useCorpusList(): CorpusListing[] {
  const project = useStore((st) => st.project?.session_id);
  const version = useStore((st) => st.corporaVersion);
  const [corpora, setCorpora] = useState<CorpusListing[]>([]);
  useEffect(() => {
    let current = true;
    api.corpora().then(
      (list) => current && setCorpora(list.filter((c) => !c.error)),
      () => current && setCorpora([]),
    );
    return () => {
      current = false;
    };
  }, [project, version]);
  return corpora;
}

/** The principal components of the activations at a site over prompts or texts, directions saved
 * from them, and the project's directions. */
export function RepresentationsView() {
  const model = useStore((st) => st.model);
  const datasets = useStore((st) => st.project?.datasets ?? []);
  const progress = useStore((st) => st.analysisProgress);
  const info = model.info;
  const corpora = useCorpusList();
  const usableDatasets = datasets.filter((d) => !d.error);
  const [source, setSource] = useState<SourceKind>(kept.source);
  const [dataset, setDataset] = useState<string | null>(kept.dataset ?? usableDatasets[0]?.path ?? null);
  const [sides, setSides] = useState(kept.sides);
  const [mark, setMark] = useState(kept.mark);
  const [corpus, setCorpus] = useState<string | null>(kept.corpus);
  const [maxTokens, setMaxTokens] = useState(kept.maxTokens ?? String(Math.min(128, info?.n_ctx ?? 128)));
  const [limit, setLimit] = useState(kept.limit);
  const [kind, setKind] = useState<StreamKind>(kept.kind);
  const [layer, setLayer] = useState(kept.layer);
  const [position, setPosition] = useState<PositionKind>(kept.position);
  const [index, setIndex] = useState(kept.index);
  const [label, setLabel] = useState(kept.label);
  const [bos, setBos] = useState(kept.bos ?? info?.extra?.bos !== false);
  const [batchSize, setBatchSize] = useState(kept.batchSize);
  const [result, setResult] = useState<Representations | null>(kept.result);
  const [facts, setFacts] = useState<{ labels: string[]; metaFields: string[] }>({ labels: [], metaFields: [] });
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const chosenCorpus = corpora.find((c) => c.name === corpus) ?? corpora[0] ?? null;
  const chosenDataset = usableDatasets.find((d) => d.path === dataset) ?? usableDatasets[0] ?? null;

  // A dataset's labelled positions and meta fields, to read and mark its prompts by.
  useEffect(() => {
    setFacts({ labels: [], metaFields: [] });
    if (!chosenDataset) return;
    let current = true;
    api.dataset(chosenDataset.path).then(
      (d) => current && setFacts(datasetFacts(d.records)),
      () => undefined,
    );
    return () => {
      current = false;
    };
  }, [chosenDataset?.path, chosenDataset?.sha256]);

  // A corpus's texts have no labelled positions: reading one reads the last token instead.
  useEffect(() => {
    if (source === "corpus" && position === "label") setPosition("last");
  }, [source]);

  const kinds = STREAM_KINDS.filter((k) => !info || info.site_kinds.includes(k));
  const request = useMemo((): { body: RepresentationsRequest | null; problem: string | null } => {
    if (!info) return { body: null, problem: "Load a model first." };
    const l = wholeNumber(layer);
    if (l === null || l < 0 || l >= info.n_layers) return { body: null, problem: `The layer is a whole number from 0 to ${info.n_layers - 1}.` };
    const n = limit.trim() ? wholeNumber(limit) : null;
    if (limit.trim() && (n === null || n < 1)) return { body: null, problem: "Read at least one, or leave the limit empty." };
    let rows: RowsSource;
    if (source === "dataset") {
      if (!chosenDataset) return { body: null, problem: "Add a dataset in Prompts first, or read a corpus." };
      rows = { kind: "dataset", path: chosenDataset.path, sha256: chosenDataset.sha256 ?? null, sides, mark: mark || null, limit: n };
    } else {
      if (!chosenCorpus) return { body: null, problem: "Add a corpus in Corpus first, or read a dataset's prompts." };
      const t = wholeNumber(maxTokens);
      if (t === null || t < (bos ? 2 : 1) || t > info.n_ctx) return { body: null, problem: `Tokens of each text is a whole number up to ${count(info.n_ctx)}.` };
      rows = { kind: "corpus", name: chosenCorpus.name, sha256: chosenCorpus.sha256 ?? null, limit: n, max_tokens: t };
    }
    let at: ReadPosition;
    if (position === "index") {
      const i = wholeNumber(index);
      if (i === null) return { body: null, problem: "The token index is a whole number; negative counts from the end." };
      at = { kind: "index", index: i };
    } else if (position === "label") {
      if (source !== "dataset" || !label) return { body: null, problem: "Choose a labelled position of the prompts." };
      at = { kind: "label", label };
    } else at = { kind: position };
    const batch = wholeNumber(batchSize);
    if (batch === null || batch < 1 || batch > 4096) return { body: null, problem: "The batch size is a whole number from 1 to 4,096." };
    return { body: { source: rows, site: { kind, layer: l }, position: at, prepend_bos: bos, batch_size: batch, model: modelRefOf(info) }, problem: null };
  }, [info, layer, limit, source, chosenDataset, sides, mark, chosenCorpus, maxTokens, bos, position, index, label, kind, batchSize]);
  const key = request.body ? JSON.stringify(request.body) : null;
  const shown = result && kept.request !== null ? result : null;
  const stale = !!shown && key !== kept.request;

  useEffect(() => {
    Object.assign(kept, {
      source,
      dataset: chosenDataset?.path ?? dataset,
      sides,
      mark,
      corpus: chosenCorpus?.name ?? corpus,
      maxTokens,
      limit,
      kind,
      layer,
      position,
      index,
      label,
      bos,
      batchSize,
    });
  }, [source, chosenDataset, dataset, sides, mark, chosenCorpus, corpus, maxTokens, limit, kind, layer, position, index, label, bos, batchSize]);

  if (model.state !== "ready" || !info) {
    return (
      <div className={s.view}>
        <Empty title="Load a model to read its representations" action={<Button onClick={() => useStore.setState({ modelDialogOpen: true })}>Load a model</Button>}>
          The model reads the prompts or texts; nothing in it changes.
        </Empty>
        <DirectionsList />
      </div>
    );
  }

  const run = async () => {
    if (!request.body || !key) return;
    setBusy(true);
    setError(null);
    useStore.setState({ analysisProgress: null });
    try {
      const out = await api.representations(request.body);
      kept.result = out;
      kept.request = key;
      kept.axes = [0, Math.min(1, out.components - 1)];
      setResult(out);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const reading = busy && progress?.what === "representations" ? progress : null;
  return (
    <div className={s.view}>
      <div className={s.head}>
        <div className={s.titleBlock}>
          <h1 className={s.title}>{shown && !stale ? "How the rows spread along their principal components" : "How prompts and texts sit in the model's activations"}</h1>
          <p className={s.subtitle}>
            Read the activation at one site and token of each prompt or text, then find the directions along which they vary
            most: their principal components, centered and computed exactly. Save a component, or the difference between two
            classes' means, as a direction for the methods that use one, or export the activations themselves.
          </p>
        </div>
      </div>

      <section className={r.settings} aria-labelledby="representations-read">
        <h2 id="representations-read" className={s.panelTitle}>
          What to read
        </h2>
        <Segmented<SourceKind>
          label="Read"
          value={source}
          onChange={setSource}
          options={[
            { value: "dataset", label: "A dataset's prompts" },
            { value: "corpus", label: "A corpus's texts", disabled: corpora.length === 0, reason: corpora.length === 0 ? "Add a corpus in Corpus to read its texts." : undefined },
          ]}
        />
        <div className={r.grid}>
          {source === "dataset" ? (
            <>
              <Field label="Dataset">
                <Select value={chosenDataset?.path ?? ""} onChange={(e) => setDataset(e.target.value)}>
                  {usableDatasets.map((d) => (
                    <option key={d.path} value={d.path}>
                      {d.name}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label="Prompts">
                <Select value={sides} onChange={(e) => setSides(e.target.value as "clean" | "corrupt" | "both")}>
                  <option value="both">Clean and corrupt</option>
                  <option value="clean">Clean</option>
                  <option value="corrupt">Corrupt</option>
                </Select>
              </Field>
              <Field label="Mark each by">
                <Select value={mark} onChange={(e) => setMark(e.target.value)}>
                  <option value="side">Its side (clean or corrupt)</option>
                  {facts.metaFields.map((m) => (
                    <option key={m} value={`meta:${m}`}>
                      Its meta's {m}
                    </option>
                  ))}
                  <option value="">Nothing</option>
                </Select>
              </Field>
              <Field label="Read the first" help="Prompt pairs; empty reads every one.">
                <Input inputMode="numeric" value={limit} placeholder="Every pair" onChange={(e) => setLimit(e.target.value)} />
              </Field>
            </>
          ) : (
            <>
              <Field label="Corpus" help="Its texts are marked by their labels.">
                <Select value={chosenCorpus?.name ?? ""} onChange={(e) => setCorpus(e.target.value)}>
                  {corpora.map((c) => (
                    <option key={c.name} value={c.name}>
                      {c.name}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label="Tokens of each text" help={`At most ${count(info.n_ctx)}; longer texts are cut.`}>
                <Input type="number" min={1} max={info.n_ctx} value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)} />
              </Field>
              <Field label="Read the first" help="Texts; empty reads every one.">
                <Input inputMode="numeric" value={limit} placeholder="Every text" onChange={(e) => setLimit(e.target.value)} />
              </Field>
            </>
          )}
        </div>
        <div className={r.grid}>
          <Field label="Site">
            <Select value={kind} onChange={(e) => setKind(e.target.value as StreamKind)}>
              {kinds.map((k) => (
                <option key={k} value={k}>
                  {KIND_WORDS[k]}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Layer">
            <Input type="number" min={0} max={info.n_layers - 1} value={layer} onChange={(e) => setLayer(e.target.value)} />
          </Field>
          <Field label="Token">
            <Select
              value={position}
              onChange={(e) => {
                const next = e.target.value as PositionKind;
                setPosition(next);
                if (next === "label" && !facts.labels.includes(label)) setLabel(facts.labels[0] ?? "");
              }}
            >
              <option value="last">The last</option>
              <option value="index">An index</option>
              {source === "dataset" && facts.labels.length > 0 && <option value="label">A labelled position</option>}
              <option value="mean">The mean over tokens</option>
            </Select>
          </Field>
          {position === "index" && (
            <Field label="Index" help="Negative counts from the end.">
              <Input type="number" value={index} onChange={(e) => setIndex(e.target.value)} />
            </Field>
          )}
          {position === "label" && source === "dataset" && (
            <Field label="Label">
              <Select value={label} onChange={(e) => setLabel(e.target.value)}>
                <option value="">Choose one</option>
                {facts.labels.map((l) => (
                  <option key={l} value={l}>
                    {l}
                  </option>
                ))}
              </Select>
            </Field>
          )}
          <Field label="Batch size">
            <Input type="number" min={1} value={batchSize} onChange={(e) => setBatchSize(e.target.value)} />
          </Field>
        </div>
        <Checkbox checked={bos} onChange={setBos} disabled={info.extra?.bos === false}>
          Begin each with the beginning-of-sequence token (the mean over tokens leaves it out)
        </Checkbox>
        <div className={s.row}>
          <Button variant="primary" disabled={!!request.problem || busy} tip={request.problem ?? undefined} onClick={() => void run()}>
            {busy ? "Reading…" : "Find the principal components"}
          </Button>
          {busy && (
            <>
              <div className={r.progress}>
                <Progress value={reading && reading.total ? reading.done / reading.total : null} label="Reading the activations" />
              </div>
              <span className={s.small}>{reading ? `${count(reading.done)} of ${count(reading.total)} read` : "Tokenizing"}</span>
            </>
          )}
        </div>
        {error && <Callout tone="error">{error}</Callout>}
        {stale && <p className={s.small}>The choices changed since these were found: press Find to read them again.</p>}
      </section>

      {/* Each reading starts its own panel: the classes and names it offers are its own. */}
      {shown && !stale && request.body && <Components key={kept.request ?? ""} result={shown} request={request.body} />}

      <DirectionsList />
    </div>
  );
}

// -- the components found ---------------------------------------------------------------------------

function Components({ result, request }: { result: Representations; request: RepresentationsRequest }) {
  const [axes, setAxes] = useState<[number, number]>(() => [Math.min(kept.axes[0], result.components - 1), Math.min(kept.axes[1], result.components - 1)]);
  const [selected, setSelected] = useState<number | null>(null);
  const marks = useMemo(() => classMarks(result.marks), [result.marks]);
  const counts = useMemo(() => classCounts(result.points), [result.points]);
  const choose = (which: 0 | 1, value: number) => {
    const next: [number, number] = which === 0 ? [value, axes[1]] : [axes[0], value];
    setAxes(next);
    kept.axes = next;
  };
  const point = selected !== null ? result.points[selected] : null;
  const unmarked = counts.get(null) ?? 0;
  const extra = result.marks.slice(MARKED_CLASSES);
  return (
    <section className={r.results} aria-labelledby="representations-found">
      <h2 id="representations-found" className={s.panelTitle}>
        {plural(result.n, "row")}: {result.description}
      </h2>
      <div className={r.split}>
        <div className={r.plot}>
          <div className={s.row}>
            <Field label="Across">
              <Select value={axes[0]} onChange={(e) => choose(0, Number(e.target.value))}>
                {result.explained.map((x, i) => (
                  <option key={i} value={i}>
                    Component {i + 1} ({explainedText(x)})
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Up">
              <Select value={axes[1]} onChange={(e) => choose(1, Number(e.target.value))}>
                {result.explained.map((x, i) => (
                  <option key={i} value={i}>
                    Component {i + 1} ({explainedText(x)})
                  </option>
                ))}
              </Select>
            </Field>
          </div>
          <Scatter points={result.points} marks={marks} xc={axes[0]} yc={axes[1]} explained={result.explained} selected={selected} onSelect={setSelected} />
          <ul className={r.legend} aria-label="Classes">
            {result.marks.slice(0, MARKED_CLASSES).map((m) => (
              <li key={m}>
                <MarkIcon mark={marks.get(m) ?? null} />
                {m} <span className={s.faint}>{count(counts.get(m) ?? 0)}</span>
              </li>
            ))}
            {(unmarked > 0 || extra.length > 0) && (
              <li>
                <MarkIcon mark={null} />
                {extra.length > 0 ? `${plural(extra.length, "more class", "more classes")}${unmarked ? ", and rows without one" : ""}` : "Rows without a class"}
              </li>
            )}
          </ul>
          {point && (
            <div className={r.point} aria-live="polite">
              <div className={s.small}>
                {pointLabel(point)} · component {axes[0] + 1} {num(point.coords[axes[0]], 3)}, component {axes[1] + 1} {num(point.coords[axes[1]], 3)}
              </div>
              <p className={r.pointText}>{point.text}</p>
            </div>
          )}
        </div>
        <div>
          <h3 className={r.subTitle}>Variance each component explains</h3>
          <ol className={r.variance} aria-label="The share of the variance each component explains">
            {result.explained.map((x, i) => (
              <li key={i} className={r.varianceRow}>
                <span>{i + 1}</span>
                <span className={r.bar} aria-hidden="true">
                  <span style={{ width: `${Math.max(0.5, x * 100)}%` }} />
                </span>
                <span className={r.share}>{explainedText(x)}</span>
              </li>
            ))}
          </ol>
          <p className={s.small}>
            Of a total variance of {num(result.total_variance, 3)} over {count(result.width)} dimensions. Each component's sign
            is fixed so its largest loading is positive.
          </p>
        </div>
      </div>
      <SaveDirections result={result} request={request} />
      <ExportActivations result={result} request={request} />
    </section>
  );
}

/** A class's shape, as the plot draws it, for the legend. */
function MarkIcon({ mark }: { mark: ClassMark | null }) {
  const paths: Record<string, string> = {
    circle: "M6 2a4 4 0 1 0 0.001 0Z",
    square: "M2.5 2.5h7v7h-7Z",
    triangle: "M6 1.5L10.5 9.5H1.5Z",
    diamond: "M6 1L11 6L6 11L1 6Z",
  };
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true" className={r.icon}>
      {mark ? (
        <path d={paths[mark.shape]} fill={mark.filled ? "currentColor" : "none"} stroke="currentColor" strokeWidth={mark.filled ? 0 : 1.3} />
      ) : (
        <circle cx="6" cy="6" r="2" fill="var(--faint)" />
      )}
    </svg>
  );
}

function drawMark(ctx: CanvasRenderingContext2D, x: number, y: number, mark: ClassMark | null, colors: ChromeColors): void {
  ctx.beginPath();
  if (!mark) {
    ctx.arc(x, y, 2, 0, Math.PI * 2);
    ctx.fillStyle = colors.faint;
    ctx.fill();
    return;
  }
  const size = 4;
  switch (mark.shape) {
    case "circle":
      ctx.arc(x, y, size, 0, Math.PI * 2);
      break;
    case "square":
      ctx.rect(x - size * 0.9, y - size * 0.9, size * 1.8, size * 1.8);
      break;
    case "triangle":
      ctx.moveTo(x, y - size * 1.2);
      ctx.lineTo(x + size * 1.15, y + size * 0.9);
      ctx.lineTo(x - size * 1.15, y + size * 0.9);
      ctx.closePath();
      break;
    case "diamond":
      ctx.moveTo(x, y - size * 1.3);
      ctx.lineTo(x + size * 1.3, y);
      ctx.lineTo(x, y + size * 1.3);
      ctx.lineTo(x - size * 1.3, y);
      ctx.closePath();
      break;
  }
  if (mark.filled) {
    ctx.fillStyle = colors.text;
    ctx.fill();
  } else {
    ctx.lineWidth = 1.3;
    ctx.strokeStyle = colors.text;
    ctx.stroke();
  }
}

const MARGIN = { left: 52, right: 14, top: 12, bottom: 34 };

/** The rows on two components, drawn on canvas: each class an ink shape, the chosen row circled.
 * One tab stop; the arrow keys move from row to row across the plot, Enter chooses one. */
function Scatter({
  points,
  marks,
  xc,
  yc,
  explained,
  selected,
  onSelect,
}: {
  points: RepresentationPoint[];
  marks: Map<string, ClassMark>;
  xc: number;
  yc: number;
  explained: number[];
  selected: number | null;
  onSelect: (row: number) => void;
}) {
  const colors = useChromeColors();
  const wrap = useRef<HTMLDivElement>(null);
  const base = useRef<HTMLCanvasElement>(null);
  const overlay = useRef<HTMLCanvasElement>(null);
  const { width } = useElementSize(wrap);
  const height = Math.round(Math.max(280, Math.min(460, width * 0.62)));
  const scale = useMemo(() => scatterScale(points, xc, yc, width, height, MARGIN), [points, xc, yc, width, height]);
  const order = useMemo(() => keyboardOrder(points, xc, yc), [points, xc, yc]);
  const [focused, setFocused] = useState(false);
  const [active, setActive] = useState<number | null>(null);
  const [hover, setHover] = useState<{ row: number; x: number; y: number } | null>(null);
  const helpId = useId();
  const current = active ?? selected ?? order[0] ?? null;

  useEffect(() => {
    const canvas = base.current;
    if (!canvas || width === 0) return;
    const ctx = prepareCanvas(canvas, width, height);
    if (!ctx) return;
    const { x, y, xDomain, yDomain } = scale;
    // Zero on each axis, dashed; the plot's frame as hairlines.
    ctx.strokeStyle = colors.line;
    ctx.lineWidth = 1;
    ctx.strokeRect(MARGIN.left + 0.5, MARGIN.top + 0.5, width - MARGIN.left - MARGIN.right - 1, height - MARGIN.top - MARGIN.bottom - 1);
    ctx.save();
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = colors.faint;
    ctx.beginPath();
    const x0 = Math.round(x(0)) + 0.5;
    const y0 = Math.round(y(0)) + 0.5;
    ctx.moveTo(x0, MARGIN.top);
    ctx.lineTo(x0, height - MARGIN.bottom);
    ctx.moveTo(MARGIN.left, y0);
    ctx.lineTo(width - MARGIN.right, y0);
    ctx.stroke();
    ctx.restore();
    // The axes' ends and names.
    ctx.fillStyle = colors.muted;
    ctx.font = font(colors, 10, 500);
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillText(num(xDomain[0], 2), MARGIN.left, height - MARGIN.bottom + 4);
    ctx.textAlign = "right";
    ctx.fillText(num(xDomain[1], 2), width - MARGIN.right, height - MARGIN.bottom + 4);
    ctx.textAlign = "center";
    ctx.fillStyle = colors.text;
    ctx.fillText(`Component ${xc + 1} (${explainedText(explained[xc] ?? 0)})`, MARGIN.left + (width - MARGIN.left - MARGIN.right) / 2, height - MARGIN.bottom + 16);
    ctx.fillStyle = colors.muted;
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText(num(yDomain[1], 2), MARGIN.left - 6, MARGIN.top);
    ctx.textBaseline = "bottom";
    ctx.fillText(num(yDomain[0], 2), MARGIN.left - 6, height - MARGIN.bottom);
    ctx.save();
    ctx.translate(12, MARGIN.top + (height - MARGIN.top - MARGIN.bottom) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = colors.text;
    ctx.fillText(`Component ${yc + 1} (${explainedText(explained[yc] ?? 0)})`, 0, 0);
    ctx.restore();
    // Rows without a class first, so the classes' shapes stay on top.
    for (const p of points) if (p.mark === null || !marks.has(p.mark)) drawMark(ctx, x(p.coords[xc] ?? 0), y(p.coords[yc] ?? 0), null, colors);
    for (const p of points) if (p.mark !== null && marks.has(p.mark)) drawMark(ctx, x(p.coords[xc] ?? 0), y(p.coords[yc] ?? 0), marks.get(p.mark) ?? null, colors);
  }, [points, marks, scale, width, height, colors, xc, yc, explained]);

  const ringed = focused ? current : selected;
  useEffect(() => {
    const canvas = overlay.current;
    if (!canvas || width === 0) return;
    const ctx = prepareCanvas(canvas, width, height);
    if (!ctx) return;
    const at = (row: number) => [scale.x(points[row].coords[xc] ?? 0), scale.y(points[row].coords[yc] ?? 0)] as const;
    if (hover && hover.row !== ringed) {
      const [hx, hy] = at(hover.row);
      ctx.strokeStyle = colors.muted;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(hx, hy, 7, 0, Math.PI * 2);
      ctx.stroke();
    }
    if (ringed !== null && points[ringed]) {
      const [px, py] = at(ringed);
      inkRing(ctx, px - 5, py - 5, 10, 10, colors.text, colors.surface);
    }
  }, [hover, ringed, scale, points, xc, yc, width, height, colors]);

  const hit = (e: MouseEvent): number | null => {
    const canvas = base.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    return nearestPoint(points, scale, xc, yc, e.clientX - rect.left, e.clientY - rect.top);
  };
  const onKeyDown = (e: KeyboardEvent) => {
    if (current === null || order.length === 0) return;
    const at = order.indexOf(current);
    let next: number | null = null;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") next = order[Math.min(order.length - 1, at + 1)];
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = order[Math.max(0, at - 1)];
    else if (e.key === "Home") next = order[0];
    else if (e.key === "End") next = order[order.length - 1];
    else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onSelect(current);
      return;
    }
    if (next === null) return;
    e.preventDefault();
    setActive(next);
  };

  return (
    <div className={r.scatter} ref={wrap}>
      <div className={r.stage}>
        <canvas
          ref={base}
          tabIndex={0}
          role="group"
          aria-roledescription="interactive chart"
          aria-label={`${plural(points.length, "row")} on components ${xc + 1} and ${yc + 1}`}
          aria-describedby={helpId}
          className={r.canvas}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          onKeyDown={onKeyDown}
          onMouseMove={(e) => {
            const row = hit(e);
            setHover(row === null ? null : { row, x: e.clientX, y: e.clientY });
          }}
          onMouseLeave={() => setHover(null)}
          onClick={(e) => {
            const row = hit(e);
            if (row !== null) {
              setActive(row);
              onSelect(row);
            }
          }}
        />
        <canvas ref={overlay} className={r.overlay} aria-hidden="true" />
      </div>
      <p id={helpId} className={r.help}>
        The arrow keys move from row to row across the plot; Enter shows one below.
      </p>
      <div className={r.readout} aria-live="polite" aria-atomic="true">
        {focused && current !== null && points[current] ? pointText(points[current]) : ""}
      </div>
      {hover && points[hover.row] && (
        <div className={r.tooltip} style={{ left: hover.x + 14, top: hover.y + 14 }}>
          {pointText(points[hover.row])}
        </div>
      )}
    </div>
  );
}

// -- directions and exports ---------------------------------------------------------------------------

function SaveDirections({ result, request }: { result: Representations; request: RepresentationsRequest }) {
  const directions = useDirections();
  const taken = useMemo(() => new Set(directions.map((d) => d.name)), [directions]);
  const site = siteName(result.site).replace(":", "-");
  const [component, setComponent] = useState(1);
  const [componentName, setComponentName] = useState<string | null>(null);
  const [positive, setPositive] = useState(result.marks[0] ?? "");
  const [negative, setNegative] = useState(result.marks[1] ?? "");
  const [differenceName, setDifferenceName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState<string | null>(null);
  const clean = (text: string) => text.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[^A-Za-z0-9]+/, "").slice(0, 80) || "direction";
  const pcName = componentName ?? `pc${component}-${site}`;
  const diffName = differenceName ?? clean(`${positive}-minus-${negative}-${site}`);
  const pcProblem = nameProblem(pcName, taken, "direction");
  const diffProblem = result.marks.length < 2 ? "Mark the rows with two classes or more to save the difference of their means." : positive === negative ? "Choose two different classes." : nameProblem(diffName, taken, "direction");

  const save = async (name: string, choice: { kind: "component"; component: number } | { kind: "mean_difference"; positive: string; negative: string }) => {
    setBusy(true);
    setError(null);
    setSaved(null);
    try {
      const out = await api.saveRepresentation({ ...request, name, save: choice });
      setSaved(out.name);
      useStore.getState().notify(`Saved the direction ${out.name}.`);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className={r.actions}>
      <form
        className={r.action}
        aria-labelledby="save-component"
        onSubmit={(e) => {
          e.preventDefault();
          if (!pcProblem) void save(pcName.trim(), { kind: "component", component });
        }}
      >
        <h3 id="save-component" className={r.subTitle}>
          Save a component as a direction
        </h3>
        <div className={s.row}>
          <Field label="Component">
            <Select value={component} onChange={(e) => setComponent(Number(e.target.value))}>
              {result.explained.map((x, i) => (
                <option key={i} value={i + 1}>
                  {i + 1} ({explainedText(x)})
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Direction name">
            <Input value={pcName} spellCheck={false} onChange={(e) => setComponentName(e.target.value)} />
          </Field>
          <Button type="submit" disabled={!!pcProblem || busy} tip={pcProblem ?? undefined}>
            Save the component
          </Button>
        </div>
      </form>
      <form
        className={r.action}
        aria-labelledby="save-difference"
        onSubmit={(e) => {
          e.preventDefault();
          if (!diffProblem) void save(diffName.trim(), { kind: "mean_difference", positive, negative });
        }}
      >
        <h3 id="save-difference" className={r.subTitle}>
          Save the difference between two classes' means as a direction
        </h3>
        <div className={s.row}>
          <Field label="Mean of">
            <Select value={positive} onChange={(e) => setPositive(e.target.value)} disabled={result.marks.length < 2}>
              {result.marks.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Minus the mean of">
            <Select value={negative} onChange={(e) => setNegative(e.target.value)} disabled={result.marks.length < 2}>
              {result.marks.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Direction name">
            <Input value={diffName} spellCheck={false} onChange={(e) => setDifferenceName(e.target.value)} />
          </Field>
          <Button type="submit" disabled={!!diffProblem || busy} tip={diffProblem ?? undefined}>
            Save the difference
          </Button>
        </div>
      </form>
      {saved && <p className={s.small}>Saved {saved} in the project's directions folder: it is listed below.</p>}
      {error && <Callout tone="error">{error}</Callout>}
    </div>
  );
}

function ExportActivations({ result, request }: { result: Representations; request: RepresentationsRequest }) {
  const nLayers = useStore((st) => st.model.info?.n_layers ?? 1);
  const progress = useStore((st) => st.analysisProgress);
  const [everyLayer, setEveryLayer] = useState(false);
  const [name, setName] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [written, setWritten] = useState<ActivationsExported | null>(null);
  const [error, setError] = useState<string | null>(null);
  const sites: ReadSite[] = exportSites(result.site, everyLayer, nLayers);
  const source = request.source;
  const stem = source.kind === "dataset" ? (source.path.split("/").pop() ?? "dataset").replace(/\.jsonl$/, "") : source.name;
  const suggested = `${stem}-${everyLayer ? result.site.kind : siteName(result.site).replace(":", "-")}`.replace(/[^A-Za-z0-9._-]+/g, "-");
  const shown = name ?? suggested;
  const problem = nameProblem(shown, new Set(), "export");
  const reading = busy && progress?.what === "export" ? progress : null;

  const run = async () => {
    if (problem) return;
    setBusy(true);
    setError(null);
    setWritten(null);
    useStore.setState({ analysisProgress: null });
    try {
      const { site: _site, ...rest } = request;
      setWritten(await api.exportActivations({ ...rest, sites, name: shown.trim() }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <form
      className={r.action}
      aria-labelledby="export-activations"
      onSubmit={(e) => {
        e.preventDefault();
        void run();
      }}
    >
      <h3 id="export-activations" className={r.subTitle}>
        Export the activations
      </h3>
      <p className={s.small}>
        {siteWords(result.site)} at {positionWords(result.position)}, one row for each of the {plural(result.n, "row")}, to
        activations/{shown || "NAME"}.safetensors in the project (float32), with a JSON file saying what each row is and how it
        was read. About {exportSize(result.n, result.width, sites.length)}.
      </p>
      <div className={s.row}>
        <Field label="Export name">
          <Input value={shown} spellCheck={false} onChange={(e) => setName(e.target.value)} />
        </Field>
        <Checkbox checked={everyLayer} onChange={setEveryLayer}>
          Every layer's {result.site.kind} ({plural(nLayers, "tensor")})
        </Checkbox>
        <Button type="submit" disabled={!!problem || busy} tip={problem ?? undefined}>
          {busy ? "Exporting…" : "Export"}
        </Button>
        {busy && (
          <div className={r.progress}>
            <Progress value={reading && reading.total ? reading.done / reading.total : null} label="Reading the activations to export" />
          </div>
        )}
      </div>
      {written && (
        <p className={s.small}>
          Wrote {plural(written.n, "row")} of {written.sites.join(", ")} to {written.path}, and what they are to {written.json_path}.
        </p>
      )}
      {error && <Callout tone="error">{error}</Callout>}
    </form>
  );
}

/** The project's saved directions, with what made each, where and on which model; a direction
 * found on another model is marked. Rename and delete them here. */
function DirectionsList() {
  const directions = useDirections();
  const info = useStore((st) => st.model.info);
  const taken = useMemo(() => new Set(directions.map((d) => d.name)), [directions]);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [deleting, setDeleting] = useState<DirectionListing | null>(null);
  const [error, setError] = useState<string | null>(null);
  const others = useMemo(() => new Set([...taken].filter((n) => n !== renaming)), [taken, renaming]);
  const problem = renaming && name.trim() === renaming ? "Type a new name." : nameProblem(name, others, "direction");

  const rename = async () => {
    if (!renaming || problem) return;
    setError(null);
    try {
      await api.renameDirection(renaming, name.trim());
      setRenaming(null);
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const remove = async (d: DirectionListing) => {
    setDeleting(null);
    setError(null);
    try {
      await api.deleteDirection(d.name);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <section className={r.directions} aria-labelledby="directions-title">
      <h2 id="directions-title" className={s.panelTitle}>
        Directions in this project
      </h2>
      <p className={s.small}>
        Vectors saved in the project's directions folder, each with the model and site it was found on: principal components
        and mean differences saved here, and what probes and learned subspaces save. Runs that use one keep their own copy,
        so renaming or deleting it changes no run.
      </p>
      {error && <Callout tone="error">{error}</Callout>}
      {directions.length === 0 ? (
        <p className={s.small}>None yet.</p>
      ) : (
        <table className={s.table}>
          <thead>
            <tr>
              <th scope="col">Name</th>
              <th scope="col">Made from</th>
              <th scope="col">Where</th>
              <th scope="col">Model</th>
              <th scope="col">
                <span className="visually-hidden">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {directions.map((d) => {
              const elsewhere = foundElsewhere(d, info);
              return (
                <tr key={d.name}>
                  <td>
                    {renaming === d.name ? (
                      <form
                        className={r.renameForm}
                        onSubmit={(e) => {
                          e.preventDefault();
                          void rename();
                        }}
                      >
                        <Input aria-label={`New name for ${d.name}`} value={name} spellCheck={false} autoFocus onChange={(e) => setName(e.target.value)} />
                        <Button size="small" type="submit" disabled={!!problem} tip={problem ?? undefined}>
                          Rename
                        </Button>
                        <Button size="small" variant="ghost" onClick={() => setRenaming(null)}>
                          Keep
                        </Button>
                      </form>
                    ) : (
                      <span className={r.directionName}>{d.name}</span>
                    )}
                    {!d.error && d.dimensions !== undefined && (
                      <div className={s.faint}>
                        {d.dimensions === 1 ? "1 dimension" : `${d.dimensions} dimensions`} of {count(d.width ?? 0)}
                      </div>
                    )}
                  </td>
                  <td>
                    {d.error ? (
                      <span className={r.unreadable}>{d.error}</span>
                    ) : (
                      <>
                        {METHOD_WORDS[d.made_by?.method ?? ""] ?? d.made_by?.method}
                        {d.made_by?.run ? ` (run ${d.made_by.run})` : ""}
                        <div className={s.faint}>{d.made_by?.detail}</div>
                      </>
                    )}
                  </td>
                  <td>{d.site ? siteName({ kind: d.site.kind as StreamKind, layer: d.site.layer }) : d.error ? "" : "No site"}</td>
                  <td>
                    {directionModelText(d)}
                    {elsewhere && <div className={r.elsewhere}>Found on another model than the one loaded</div>}
                  </td>
                  <td className={r.rowActions}>
                    {renaming !== d.name && (
                      <>
                        <Button
                          size="small"
                          variant="ghost"
                          aria-label={`Rename ${d.name}`}
                          onClick={() => {
                            setRenaming(d.name);
                            setName(d.name);
                          }}
                        >
                          Rename
                        </Button>
                        <Button size="small" variant="ghost" aria-label={`Delete ${d.name}`} onClick={() => setDeleting(d)}>
                          Delete
                        </Button>
                      </>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      <ConfirmDialog
        open={deleting !== null}
        onOpenChange={(open) => !open && setDeleting(null)}
        title={`Delete the direction ${deleting?.name ?? ""}?`}
        keepLabel="Keep it"
        confirmLabel="Delete the direction"
        onConfirm={() => deleting && void remove(deleting)}
      >
        Its files are deleted from the project's directions folder. Runs that used it keep their own copy. It can't be undone.
      </ConfirmDialog>
    </section>
  );
}
