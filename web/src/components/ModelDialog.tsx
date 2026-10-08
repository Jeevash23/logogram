import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { Device, Dtype, EstimateResponse } from "../api/types";
import { bytes, params, pct, shortRevision } from "../lib/format";
import { modelName, useActiveRun } from "../lib/hooks";
import { useStore } from "../store/app";
import { Button, Callout, Checkbox, Choices, Dialog, Field, Input, Progress, Segmented, Spinner } from "./ui";
import s from "./ModelDialog.module.css";

const GPT2 = "openai-community/gpt2";

export function ModelDialog() {
  const open = useStore((st) => st.modelDialogOpen);
  const model = useStore((st) => st.model);
  const job = useStore((st) => st.job);
  const guard = useStore((st) => st.guard);
  const [choice, setChoice] = useState<"gpt2" | "other">("gpt2");
  const [other, setOther] = useState("");
  const [dtype, setDtype] = useState<Dtype>("float32");
  const [device, setDevice] = useState<Device>("auto");
  const [processWeights, setProcessWeights] = useState(true);
  const [revision, setRevision] = useState("");
  const run = useActiveRun();
  const formRef = useStore((st) => st.form.modelRef);
  const savedRef = run.detail?.spec.model ?? formRef;
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [estimate, setEstimate] = useState<EstimateResponse | null>(null);
  const [estimating, setEstimating] = useState(false);
  const [estimateError, setEstimateError] = useState<string | null>(null);

  const id = choice === "gpt2" ? GPT2 : other.trim();
  const loading = model.state === "loading";

  useEffect(() => {
    if (!open) return;
    api.presets().then((p) => setSuggestions(p.suggestions), () => undefined);
  }, [open]);

  // Ask Hugging Face for the exact revision and size, and estimate memory, before loading.
  useEffect(() => {
    setEstimate(null);
    setEstimateError(null);
    setEstimating(false);
    if (!open || !id || !/^[\w.-]+(\/[\w.-]+)?$/.test(id)) return;
    let cancelled = false;
    const t = window.setTimeout(() => {
      setEstimating(true);
      api.estimate({ id, revision: revision.trim() || null, dtype, device }).then(
        (e) => {
          if (!cancelled) {
            setEstimate(e);
            setEstimating(false);
          }
        },
        (e: Error) => {
          if (!cancelled) {
            setEstimateError(e.message);
            setEstimating(false);
          }
        },
      );
    }, choice === "other" ? 500 : 0);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
  }, [open, id, revision, dtype, device, choice]);

  const load = async () => {
    const out = await guard(() =>
      api.loadModel({ id, revision: estimate?.revision ?? (revision.trim() || null), dtype, device, process_weights: processWeights }),
    );
    if (out) useStore.getState().applyJob(out);
  };

  const est = estimate?.estimate;
  const verdictText = est ? { fits: "Fits", tight: "Tight", wont_fit: "Won't fit" }[est.verdict] : null;
  const where = est ? ({ cuda: "on the GPU", mps: "in unified memory", cpu: "in RAM" } as Record<string, string>)[est.device] ?? "" : "";

  return (
    <Dialog
      open={open}
      onOpenChange={(v) => useStore.setState({ modelDialogOpen: v })}
      title="Load a model"
      description="Models are downloaded from Hugging Face once, with your own login if needed, then loaded from the local cache."
      wide
      footer={
        <>
          {model.info && model.state === "ready" && (
            <Button variant="ghost" onClick={() => void guard(() => api.unloadModel())} disabled={job?.status === "running"}>
              Unload {modelName(model.info.id)}
            </Button>
          )}
          <div style={{ flex: 1 }} />
          <Button variant="ghost" onClick={() => useStore.setState({ modelDialogOpen: false })}>
            Close
          </Button>
          <Button variant="primary" onClick={load} disabled={!id || loading || job?.status === "running" || est?.verdict === "wont_fit"}>
            {loading ? "Loading…" : "Load"}
          </Button>
        </>
      }
    >
      {savedRef && <Button size="small" onClick={() => {
        setChoice(savedRef.id === GPT2 ? "gpt2" : "other");
        setOther(savedRef.id); setRevision(savedRef.revision ?? "");
        setDtype(savedRef.dtype); setDevice(savedRef.device); setProcessWeights(savedRef.process_weights);
      }}>Use saved experiment model settings</Button>}
      <Choices
        label="Model"
        value={choice}
        onChange={setChoice}
        columns={2}
        options={[
          { value: "gpt2", title: "GPT-2 small", detail: "124M parameters · 12 layers × 12 heads · tested with Logogram" },
          { value: "other", title: "Another Hugging Face model", detail: "Any model TransformerLens supports, on a best-effort basis" },
        ]}
      />
      {choice === "other" && (
        <Field label="Hugging Face id" help="Like owner/name. Only safetensors weights are loaded.">
          <Input value={other} onChange={(e) => setOther(e.target.value)} placeholder="EleutherAI/pythia-160m" list="model-suggestions" spellCheck={false} autoFocus />
          <datalist id="model-suggestions">
            {suggestions.map((x) => (
              <option key={x} value={x} />
            ))}
          </datalist>
        </Field>
      )}
      <Field label="Model revision" help="A commit, tag or branch. Leave blank to resolve the current default; the exact commit is saved with every run.">
        <Input value={revision} onChange={(e) => setRevision(e.target.value)} placeholder="Default revision" spellCheck={false} />
      </Field>
      <div className={s.settings}>
        <Field label="Precision">
          <Segmented
            label="Precision"
            value={dtype}
            onChange={setDtype}
            options={[
              { value: "float32", label: "float32" },
              { value: "bfloat16", label: "bfloat16" },
              { value: "float16", label: "float16" },
            ]}
          />
        </Field>
        <Field label="Device">
          <Segmented
            label="Device"
            value={device}
            onChange={setDevice}
            options={[
              { value: "auto", label: "Automatic" },
              { value: "cuda", label: "CUDA" },
              { value: "mps", label: "MPS" },
              { value: "cpu", label: "CPU" },
            ]}
          />
        </Field>
      </div>
      <Checkbox checked={processWeights} onChange={setProcessWeights}>
        Process weights like TransformerLens: fold LayerNorm, center writing weights and the unembedding
      </Checkbox>

      <div className={s.estimate} aria-live="polite">
        {estimating && (
          <div className={s.row}>
            <Spinner /> Checking {id} on Hugging Face…
          </div>
        )}
        {estimateError && <Callout tone="error">{estimateError}</Callout>}
        {estimate && est && (
          <>
            <div className={s.verdictRow}>
              <span className={`${s.verdict} ${est.verdict === "wont_fit" ? s.bad : ""}`}>{verdictText}</span>
              <span>
                needs about <strong>{bytes(est.total)}</strong> of <strong>{bytes(est.available)}</strong> free {where}
              </span>
            </div>
            <div className={s.bar} aria-hidden="true">
              <span style={{ width: `${Math.min(100, (est.weights / est.available) * 100)}%` }} className={s.weights} />
              <span style={{ width: `${Math.min(100, (est.activations / est.available) * 100)}%` }} className={s.acts} />
              <span style={{ width: `${Math.min(100, (est.margin / est.available) * 100)}%` }} className={s.margin} />
            </div>
            <dl className={s.breakdown}>
              <dt>Weights</dt>
              <dd>{bytes(est.weights)}</dd>
              <dt>Activations</dt>
              <dd>{bytes(est.activations)}</dd>
              <dt>Margin</dt>
              <dd>{bytes(est.margin)}</dd>
              <dt title="Parameters and stored buffers, as listed in the safetensors files">Stored values</dt>
              <dd>{params(est.n_params)}</dd>
              <dt>Revision</dt>
              <dd title={estimate.revision}>{shortRevision(estimate.revision)}</dd>
              <dt>Download</dt>
              <dd>{estimate.download_bytes === 0 ? "already in the cache" : bytes(estimate.download_bytes)}</dd>
            </dl>
            <p className={s.fine}>{est.explanation}</p>
            {estimate.gated && <p className={s.fine}>This model is gated: accept its license on huggingface.co and log in with `hf auth login`.</p>}
          </>
        )}
      </div>

      {loading && (
        <div className={s.progress}>
          <div className={s.row}>
            <Spinner />
            {model.stage === "downloading" && model.total
              ? `Downloading ${bytes(model.done)} of ${bytes(model.total)} (${pct((model.done ?? 0) / model.total)})`
              : model.stage === "resolving"
                ? "Checking the revision"
                : model.stage === "loading"
                  ? "Loading weights"
                  : model.stage === "processing"
                    ? "Processing weights"
                    : "Starting"}
          </div>
          <Progress value={model.stage === "downloading" && model.total ? (model.done ?? 0) / model.total : null} />
        </div>
      )}
      {model.state === "error" && model.error && <Callout tone="error" title="The model didn't load">{model.error}</Callout>}
      {model.state === "ready" && model.info && (
        <p className={s.fine}>
          Loaded: {modelName(model.info.id)} @ {shortRevision(model.info.revision)} on {model.info.device_name} ({model.info.dtype}).
        </p>
      )}
    </Dialog>
  );
}
