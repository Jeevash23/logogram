import { useEffect, useRef, useState } from "react";

import { api } from "../api/client";
import type { Device, DeviceCheck, Dtype, EstimateResponse, LocalModel, ModelPreset, SystemReport } from "../api/types";
import { chatAdvice, chatReason } from "../lib/chat";
import { gpuChoices, whereText } from "../lib/devices";
import { bytes, params, pct, shortRevision } from "../lib/format";
import { modelName } from "../lib/hooks";
import { useStore } from "../store/app";
import { FolderPicker } from "./FolderPicker";
import { Button, Callout, Checkbox, Dialog, Field, Input, Progress, Segmented, Select, Spinner, useRadioGroup } from "./ui";
import s from "./ModelDialog.module.css";

const GPT2 = "openai-community/gpt2";
const OTHER = "other";
/** A local model's choice: by the hash of its files, which is what specs find it by. */
const LOCAL = "local:";

/** A folder's last name, as a local model's suggested name. */
function folderName(path: string): string {
  return path.replace(/[\\/]+$/, "").split(/[\\/]/).pop() ?? "";
}

export function ModelDialog() {
  const open = useStore((st) => st.modelDialogOpen);
  const model = useStore((st) => st.model);
  const job = useStore((st) => st.job);
  const guard = useStore((st) => st.guard);
  // Wrapped in its chat template, the prompts read as a chat: the warning below has nothing to say.
  const wrapped = useStore((st) => st.form.chatTemplate !== null);
  /** A preset's id, LOCAL + a local model's hash, or OTHER for an id typed in. */
  const [choice, setChoice] = useState<string>(GPT2);
  const [other, setOther] = useState("");
  const [dtype, setDtype] = useState<Dtype>("float32");
  const [device, setDevice] = useState<Device>("auto");
  const [processWeights, setProcessWeights] = useState(true);
  const [revision, setRevision] = useState("");
  // Only the active run's spec: the dialog doesn't follow a run's progress.
  const runModel = useStore((st) => (st.activeRunId ? st.runDetails[st.activeRunId]?.spec.model : undefined));
  const formRef = useStore((st) => st.form.modelRef);
  const savedRef = runModel ?? formRef;
  const [presets, setPresets] = useState<ModelPreset[]>([]);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [locals, setLocals] = useState<LocalModel[]>([]);
  const [system, setSystem] = useState<SystemReport | null>(null);
  const [estimate, setEstimate] = useState<EstimateResponse | null>(null);
  const [estimating, setEstimating] = useState(false);
  const [estimateError, setEstimateError] = useState<string | null>(null);
  // Adding a model folder: the folder chosen and the name it is added under.
  const [picking, setPicking] = useState(false);
  const [adding, setAdding] = useState<{ path: string; name: string } | null>(null);
  const [addError, setAddError] = useState<string | null>(null);
  // The saved experiment's local model, when this machine has no folder with its files.
  const [missing, setMissing] = useState<string | null>(null);
  useEffect(() => setMissing(null), [choice]);
  // Whether the device chosen is there: Load waits for one that is, and the dialog says why.
  const [deviceCheck, setDeviceCheck] = useState<DeviceCheck | null>(null);

  const local = choice.startsWith(LOCAL) ? (locals.find((m) => LOCAL + m.sha256 === choice) ?? null) : null;
  const id = local ? local.name : choice === OTHER ? other.trim() : choice;
  const loading = model.state === "loading";
  const busy = job?.status === "running";
  const addJob = job?.kind === "add_model" ? job : null;
  const hashing = addJob?.status === "running";
  // Bytes hashed so far, of all the folder's files that loading reads.
  const hashed = addJob ? { done: Number(addJob.progress.done ?? 0), total: Number(addJob.progress.total ?? 0) } : null;
  const shown = presets.length ? presets : [{ id: GPT2, label: "GPT-2 small", detail: "124M", tested: true, gated: false }];
  const gpus = gpuChoices(system);
  // The presets, the local models and "Another model" are one radio group: one tab stop, arrow
  // keys choose.
  const options = [...shown.map((p) => ({ value: p.id })), ...locals.map((m) => ({ value: LOCAL + m.sha256 })), { value: OTHER }];
  const radio = useRadioGroup(options, choice, setChoice);
  // Choosing "Another model" (by click, Enter or Space) moves on to typing its id; moving through
  // the options with the arrow keys doesn't take the focus out of the group.
  const otherInput = useRef<HTMLInputElement>(null);
  const chooseOther = () => {
    setChoice(OTHER);
    requestAnimationFrame(() => otherInput.current?.focus());
  };
  const nameInput = useRef<HTMLInputElement>(null);

  const refreshLocals = () => api.localModels().then(setLocals, () => undefined);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    api.checkDevice(device).then(
      (checked) => {
        if (!cancelled) setDeviceCheck(checked);
      },
      () => undefined,
    );
    return () => {
      cancelled = true;
    };
  }, [open, device]);
  const deviceProblem = deviceCheck?.device === device ? deviceCheck.error : null;

  useEffect(() => {
    if (!open) return;
    api.presets().then((p) => {
      setPresets(p.presets);
      setSuggestions(p.suggestions);
    }, () => undefined);
    void refreshLocals();
    api.system().then(setSystem, () => undefined);
  }, [open]);

  // A folder added: list it, and choose it.
  const addStatus = addJob?.status;
  useEffect(() => {
    if (!addStatus || addStatus === "running") return;
    if (addStatus === "failed") setAddError(addJob?.error ?? "The folder couldn't be added.");
    if (addStatus !== "finished") return;
    const name = adding?.name.trim();
    void api.localModels().then((listed) => {
      setLocals(listed);
      const added = listed.find((m) => m.name === name);
      if (added) setChoice(LOCAL + added.sha256);
      setAdding(null);
    }, () => undefined);
    // Only a change in the job's status matters here.
  }, [addStatus, addJob?.id]);

  // Ask Hugging Face for the exact revision and size (or read a local model's folder), and
  // estimate memory on the device chosen, before loading.
  useEffect(() => {
    setEstimate(null);
    setEstimateError(null);
    setEstimating(false);
    // Nothing to estimate on a device that isn't here: the dialog says so instead.
    if (!open || !id || deviceProblem || (!local && !/^[\w.-]+(\/[\w.-]+)?$/.test(id))) return;
    let cancelled = false;
    const t = window.setTimeout(() => {
      setEstimating(true);
      const request = local
        ? api.estimate({ id, source: "local", sha256: local.sha256, dtype, device, process_weights: processWeights })
        : api.estimate({ id, revision: revision.trim() || null, dtype, device, process_weights: processWeights });
      request.then(
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
    }, choice === OTHER ? 500 : 0);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
  }, [open, id, local?.sha256, revision, dtype, device, processWeights, choice, deviceProblem]);

  const load = async () => {
    const out = await guard(() =>
      api.loadModel(
        local
          ? { id: local.name, source: "local", revision: null, sha256: local.sha256, dtype, device, process_weights: processWeights }
          : { id, source: "hub", revision: estimate?.revision ?? (revision.trim() || null), sha256: null, dtype, device, process_weights: processWeights },
      ),
    );
    if (out) useStore.getState().applyJob(out);
  };

  const add = async () => {
    if (!adding?.name.trim()) return;
    setAddError(null);
    const out = await guard(() => api.addLocalModel(adding.path, adding.name.trim()));
    if (out) useStore.getState().applyJob(out);
  };

  const remove = async (m: LocalModel) => {
    const listed = await guard(() => api.removeLocalModel(m.name));
    if (!listed) return;
    setLocals(listed);
    setChoice(GPT2);
  };

  const useSaved = () => {
    if (!savedRef) return;
    if (savedRef.source === "local") {
      const found = locals.find((m) => m.sha256 === savedRef.sha256);
      if (!found) {
        setMissing(
          `This machine has no local model with the files the experiment names (${savedRef.id}, SHA-256 ${savedRef.sha256?.slice(0, 12)}…). Add its folder with Add a model folder.`,
        );
        return;
      }
      setChoice(LOCAL + found.sha256);
    } else {
      setChoice(presets.some((p) => p.id === savedRef.id) ? savedRef.id : OTHER);
      setOther(savedRef.id);
      setRevision(savedRef.revision ?? "");
    }
    setDtype(savedRef.dtype);
    setDevice(savedRef.device);
    setProcessWeights(savedRef.process_weights);
  };

  const est = estimate?.estimate;
  const verdictText = est ? { fits: "Fits", tight: "Tight", wont_fit: "Won't fit" }[est.verdict] : null;
  const where = est ? whereText(est.device) : "";
  const onGpu = device === "cuda" || device.startsWith("cuda:");

  return (
    <Dialog
      open={open}
      onOpenChange={(v) => useStore.setState({ modelDialogOpen: v })}
      title="Load a model"
      description="Models are downloaded from Hugging Face once, with your own login if needed, then loaded from the local cache, or loaded from a folder on this machine."
      wide
      footer={
        <>
          {model.info && model.state === "ready" && (
            <Button variant="ghost" onClick={() => void guard(() => api.unloadModel())} disabled={busy}>
              Unload {modelName(model.info.id)}
            </Button>
          )}
          <div style={{ flex: 1 }} />
          <Button variant="ghost" onClick={() => useStore.setState({ modelDialogOpen: false })}>
            Close
          </Button>
          <Button
            variant="primary"
            onClick={load}
            disabled={!id || loading || busy || deviceProblem !== null || est?.verdict === "wont_fit" || estimate?.supported === false}
          >
            {loading ? "Loading…" : "Load"}
          </Button>
        </>
      }
    >
      {savedRef && <Button size="small" onClick={useSaved}>Use saved experiment model settings</Button>}
      <div className={s.models} role="radiogroup" aria-label="Model">
        {shown.map((p, i) => (
          <button key={p.id} type="button" role="radio" aria-checked={choice === p.id} className={s.model} onClick={() => setChoice(p.id)} {...radio(i)}>
            <span className={s.dot} />
            <span className={s.modelName}>{p.label}</span>
            <span className={s.modelDetail}>{p.detail}</span>
            {p.tested && <span className={s.tag}>Tested</span>}
          </button>
        ))}
        {locals.map((m, i) => (
          <button
            key={m.sha256}
            type="button"
            role="radio"
            aria-checked={choice === LOCAL + m.sha256}
            className={s.model}
            onClick={() => setChoice(LOCAL + m.sha256)}
            {...radio(shown.length + i)}
          >
            <span className={s.dot} />
            <span className={s.modelName}>{m.name}</span>
            <span className={s.modelDetail}>
              {bytes(m.size)} in {m.folder}
            </span>
            <span className={s.tag}>Local</span>
          </button>
        ))}
        <button type="button" role="radio" aria-checked={choice === OTHER} className={s.model} onClick={chooseOther} {...radio(shown.length + locals.length)}>
          <span className={s.dot} />
          <span className={s.modelName}>Another model</span>
          <span className={s.modelDetail}>Any Hugging Face model TransformerLens can load, checked when it loads</span>
        </button>
      </div>
      {!adding && !hashing && (
        <div className={s.row}>
          <Button size="small" icon="folder" onClick={() => setPicking(true)} disabled={busy}>
            Add a model folder
          </Button>
          <span>A folder on this machine with config.json, the tokenizer's files and safetensors weights.</span>
        </div>
      )}
      <FolderPicker
        open={picking}
        onOpenChange={setPicking}
        title="Choose a model folder"
        chooseLabel="Choose this folder"
        onChoose={(path) => {
          setAddError(null);
          setAdding({ path, name: folderName(path) });
          requestAnimationFrame(() => nameInput.current?.focus());
        }}
      />
      {adding && (
        <section className={s.adding} aria-label="Add a model folder">
          <p className={s.fine}>
            Its files are hashed once: specs name the model by its name and that hash, never by its folder, and loading checks
            the files are still the same.
          </p>
          <dl className={s.breakdown}>
            <dt>Folder</dt>
            <dd className={s.arch} title={adding.path}>{adding.path}</dd>
          </dl>
          <div className={s.row}>
            <Field label="Name">
              <Input
                ref={nameInput}
                value={adding.name}
                onChange={(e) => setAdding({ ...adding, name: e.target.value })}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    void add();
                  }
                }}
                spellCheck={false}
                disabled={hashing}
              />
            </Field>
            <Button size="small" onClick={() => void add()} disabled={!adding.name.trim() || busy}>
              Add
            </Button>
            <Button size="small" variant="ghost" onClick={() => setAdding(null)} disabled={hashing}>
              Cancel
            </Button>
          </div>
          {hashing && hashed && (
            <div className={s.progress}>
              <div className={s.row}>
                <Spinner />
                {hashed.total > 0
                  ? `Hashing ${bytes(hashed.done)} of ${bytes(hashed.total)} (${pct(hashed.done / hashed.total)})`
                  : "Reading the folder"}
              </div>
              <Progress value={hashed.total > 0 ? hashed.done / hashed.total : null} label="Hashing the folder's files" />
            </div>
          )}
        </section>
      )}
      {addError && <Callout tone="error" title="The folder wasn't added">{addError}</Callout>}
      {missing && <Callout tone="error" title="This machine doesn't have the experiment's model">{missing}</Callout>}
      {choice === OTHER && (
        <Field label="Hugging Face id" help="Like owner/name. Only safetensors weights are loaded.">
          <Input ref={otherInput} value={other} onChange={(e) => setOther(e.target.value)} placeholder="EleutherAI/pythia-70m" list="model-suggestions" spellCheck={false} />
          <datalist id="model-suggestions">
            {suggestions.map((x) => (
              <option key={x} value={x} />
            ))}
          </datalist>
        </Field>
      )}
      {local ? (
        <div className={s.row}>
          <span>
            {local.name}: {local.files} files, SHA-256 {local.sha256.slice(0, 12)}…. A local model has no revisions: its files are
            pinned by their hash.
          </span>
          <Button size="small" variant="ghost" onClick={() => void remove(local)} disabled={busy} tip="Forget it here. Its folder and files are left as they are.">
            Remove from the list
          </Button>
        </div>
      ) : (
        <Field label="Model revision" help="A commit, tag or branch. Leave blank to resolve the current default; the exact commit is saved with every run.">
          <Input value={revision} onChange={(e) => setRevision(e.target.value)} placeholder="Default revision" spellCheck={false} />
        </Field>
      )}
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
            value={onGpu ? "cuda" : device}
            onChange={(v) => setDevice(v === "cuda" && onGpu ? device : v)}
            options={[
              { value: "auto", label: "Automatic" },
              { value: "cuda", label: "CUDA" },
              { value: "mps", label: "MPS" },
              { value: "cpu", label: "CPU" },
            ]}
          />
        </Field>
        {onGpu && gpus.length > 0 && (
          <Field label="GPU" help="This machine has several. Memory is estimated on the one chosen.">
            <Select value={device} onChange={(e) => setDevice(e.target.value as Device)}>
              {gpus.map((g) => (
                <option key={g.value} value={g.value}>
                  {g.label}
                </option>
              ))}
            </Select>
          </Field>
        )}
      </div>
      {deviceProblem && (
        <Callout tone="error" title="The device chosen isn't available here">
          {deviceProblem}
        </Callout>
      )}
      {device === "mps" && !deviceProblem && (
        <p className={s.fine}>
          TransformerLens reports that Apple's MPS can give silently wrong results, and Logogram hasn't been checked on it
          yet. MPS is faster; check results that matter on the CPU.
        </p>
      )}
      <Checkbox checked={processWeights} onChange={setProcessWeights}>
        Process weights like TransformerLens: fold LayerNorm, center writing weights and the unembedding
      </Checkbox>

      <div className={s.estimate} aria-live="polite">
        {estimating && (
          <div className={s.row}>
            <Spinner /> {local ? `Reading the folder of ${local.name}…` : `Checking ${id} on Hugging Face…`}
          </div>
        )}
        {estimateError && estimateError !== deviceProblem && <Callout tone="error">{estimateError}</Callout>}
        {estimate?.support_note && <Callout tone="error" title="This model can't be loaded">{estimate.support_note}</Callout>}
        {estimate && est && estimate.supported && (
          <>
            <div className={s.verdictRow}>
              <span className={`${s.verdict} ${est.verdict === "wont_fit" ? s.bad : ""}`}>{verdictText}</span>
              <span>
                needs about <strong>{bytes(est.total)}</strong> of <strong>{bytes(est.available)}</strong> free {where}
              </span>
            </div>
            {processWeights && est.verdict !== "fits" && est.processing > est.activations && (
              <div className={s.row}>
                <span>Processing the weights needs {bytes(est.processing)} for a moment while the model loads.</span>
                <Button size="small" onClick={() => setProcessWeights(false)}>
                  Turn processing off
                </Button>
              </div>
            )}
            <div className={s.bar} aria-hidden="true">
              <span style={{ width: `${Math.min(100, (est.weights / est.available) * 100)}%` }} className={s.weights} />
              {/* The larger need beside the weights: processing them while loading, or a batch's activations. */}
              <span style={{ width: `${Math.min(100, (Math.max(est.activations, est.processing) / est.available) * 100)}%` }} className={est.processing > est.activations ? s.processing : s.acts} />
              <span style={{ width: `${Math.min(100, (est.margin / est.available) * 100)}%` }} className={s.margin} />
            </div>
            <dl className={s.breakdown}>
              <dt>Weights</dt>
              <dd>{bytes(est.weights)}</dd>
              <dt>Activations</dt>
              <dd>{bytes(est.activations)}</dd>
              {est.processing > 0 && (
                <>
                  <dt>While loading</dt>
                  <dd>{bytes(est.processing)} of float32 copies of the weights</dd>
                </>
              )}
              <dt>Margin</dt>
              <dd>{bytes(est.margin)}</dd>
              <dt>Stored values</dt>
              <dd>{params(est.n_params)} parameters and buffers</dd>
              <dt>{local ? "Files" : "Revision"}</dt>
              <dd title={local ? local.sha256 : (estimate.revision ?? undefined)}>
                {local ? `SHA-256 ${local.sha256.slice(0, 12)}…` : shortRevision(estimate.revision)}
              </dd>
              <dt>Download</dt>
              <dd>{local ? "none: a local folder" : estimate.download_bytes === 0 ? "already in the cache" : bytes(estimate.download_bytes)}</dd>
              <dt>Shape</dt>
              <dd>{estimate.architecture.n_layers} layers × {estimate.architecture.n_heads} heads</dd>
              <dt>Architecture</dt>
              <dd className={s.arch} title={estimate.architecture.architecture}>{estimate.architecture.architecture.replace(/ForCausalLM$|LMHeadModel$/, "")}</dd>
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
              : model.stage === "checking"
                ? model.total
                  ? `Checking the folder's changed files: ${bytes(model.done)} of ${bytes(model.total)}`
                  : "Checking the folder's files"
                : model.stage === "resolving"
                  ? "Checking the revision"
                  : model.stage === "loading"
                    ? "Loading weights"
                    : model.stage === "processing"
                      ? "Processing weights"
                      : "Starting"}
          </div>
          <Progress
            value={(model.stage === "downloading" || model.stage === "checking") && model.total ? (model.done ?? 0) / model.total : null}
            label="Loading the model"
          />
        </div>
      )}
      {model.state === "error" && model.error && <Callout tone="error" title="The model didn't load">{model.error}</Callout>}
      {model.state === "ready" && model.info && (
        <p className={s.fine}>
          Loaded: {modelName(model.info.id)}
          {model.info.source === "local" ? `, a local model (files ${model.info.sha256?.slice(0, 12)}…)` : ` @ ${shortRevision(model.info.revision)}`} on{" "}
          {model.info.device_name} ({model.info.device}, {model.info.dtype}).
        </p>
      )}
      {model.state === "ready" && chatReason(model.info) && !wrapped && (
        <Callout title={`${modelName(model.info?.id)} is made for chat, and runs here on plain text`}>
          {chatReason(model.info)} Without its chat template, Logogram feeds every prompt as plain text, not as the model was
          trained to read it, so it can behave differently here than in a chat: a behavior can be weaker, missing or
          different. {chatAdvice(model.info)}
        </Callout>
      )}
    </Dialog>
  );
}
