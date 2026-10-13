import { useEffect, useRef, useState, type ReactNode } from "react";

import { api } from "../api/client";
import type { CorpusListing, CurvePoint, Job, TrainedLens, TrainedSAE, TrainingPlan, TrainingProgress } from "../api/types";
import { font, prepareCanvas, useChromeColors, useElementSize } from "../lib/canvas";
import { count, num } from "../lib/format";
import { modelName } from "../lib/hooks";
import { modelRefOf } from "../lib/spec";
import {
  type CommonFields,
  type LensFields,
  type SaeFields,
  SAE_SITES,
  SAE_SITE_WORDS,
  lensFields,
  lensSettingsText,
  lensSpec,
  planText,
  progressText,
  saeFields,
  saeSettingsText,
  saeSpec,
  share,
} from "../lib/training";
import { useStore } from "../store/app";
import { Button, Callout, Checkbox, Dialog, Empty, Field, Input, Progress, Segmented, Select, Spinner } from "./ui";
import t from "./TrainDialog.module.css";

const close = () => useStore.setState({ trainDialog: null });

/** Train an SAE or a tuned lens in the project (src/logogram/sae_training.py,
 * src/logogram/lenses.py): every choice of its training spec, what training would take on this
 * machine, then its progress and loss curve, and what it measured on the held-out texts. Opened
 * from Features and from Layer predictions. */
export function TrainDialog() {
  const which = useStore((st) => st.trainDialog);
  if (!which) return null;
  return which === "sae" ? <TrainSae /> : <TrainLens />;
}

/** The project's corpora, read again whenever a training ends. */
function useCorpora(): { corpora: CorpusListing[] | null; error: string | null } {
  const version = useStore((st) => st.trainedVersion);
  const [corpora, setCorpora] = useState<CorpusListing[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    api.corpora().then(
      (list) => !cancelled && setCorpora(list),
      (e: Error) => !cancelled && setError(e.message),
    );
    return () => {
      cancelled = true;
    };
  }, [version]);
  return { corpora, error };
}

/** What the server says training ``spec`` would take, asked again a moment after the fields stop
 * changing. */
function usePlan<S>(spec: S | null, ask: (spec: S) => Promise<TrainingPlan>): { plan: TrainingPlan | null; error: string | null; busy: boolean } {
  const key = spec ? JSON.stringify(spec) : "";
  const [state, setState] = useState<{ key: string; plan: TrainingPlan | null; error: string | null }>({ key: "", plan: null, error: null });
  useEffect(() => {
    if (!spec) return;
    let cancelled = false;
    const timer = window.setTimeout(() => {
      ask(spec).then(
        (plan) => !cancelled && setState({ key, plan, error: null }),
        (e: Error) => !cancelled && setState({ key, plan: null, error: e.message }),
      );
    }, 700);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
    // The key holds the spec.
  }, [key]);
  const current = state.key === key && !!spec;
  return { plan: current ? state.plan : null, error: current ? state.error : null, busy: !!spec && !current };
}

/** The training job this dialog shows: one of its kind that runs, or the one it started, once
 * ended. */
function useTraining(kind: Job["kind"], startedId: string | null): Job | null {
  const job = useStore((st) => st.job);
  return job && job.kind === kind && (job.status === "running" || job.id === startedId) ? job : null;
}

function TrainSae() {
  const model = useStore((st) => st.model);
  const info = model.info;
  const busy = useStore((st) => st.job?.status === "running");
  const guard = useStore((st) => st.guard);
  const version = useStore((st) => st.trainedVersion);
  const { corpora, error: corporaError } = useCorpora();
  const [trained, setTrained] = useState<TrainedSAE[] | null>(null);
  const [fields, setFields] = useState<SaeFields | null>(null);
  const [startedId, setStartedId] = useState<string | null>(null);
  const job = useTraining("train_sae", startedId);

  useEffect(() => {
    api.projectSaes().then(setTrained, () => setTrained([]));
  }, [version]);
  const readable = (corpora ?? []).filter((c) => !c.error);
  useEffect(() => {
    if (!fields && info && corpora && trained) setFields(saeFields(info, readable[0]?.name ?? "", trained.map((s) => s.name)));
  }, [info, corpora, trained, fields, readable]);

  const ref = info ? modelRefOf(info, model.ref?.device ?? info.device) : null;
  const built = fields && ref ? saeSpec(fields, ref) : null;
  const spec = built?.spec ?? null;
  const planned = usePlan(job ? null : spec, (s) => api.planSae(s));
  const set = (patch: Partial<SaeFields>) => setFields((f) => (f ? { ...f, ...patch } : f));
  const start = async () => {
    if (!spec) return;
    const out = await guard(() => api.trainSae(spec));
    if (out) {
      setStartedId(out.id);
      useStore.getState().applyJob(out);
    }
  };
  const load = async (name: string, sha256: string | null) => {
    const out = await guard(() => api.loadSae({ repo: name, path: `saes/${name}`, revision: null, source: "project", sha256 }));
    if (out) {
      useStore.getState().applyJob(out);
      close();
      useStore.getState().setView("features");
    }
  };

  const progress = (job?.progress ?? {}) as TrainingProgress;
  const result = job?.status === "finished" ? trained?.find((s) => s.name === progress.name) ?? null : null;
  return (
    <Dialog
      open
      onOpenChange={(open) => !open && close()}
      title="Train an SAE"
      description="A TopK sparse autoencoder on one site of one layer, trained on a corpus of this project and saved as saes/NAME, where specs find it by the hash of its files. The same settings on this machine give the same weights."
      wide
      footer={
        job ? (
          <TrainingFooter job={job} onBack={() => setStartedId(null)}>
            {result && result.sha256 && (
              <Button variant="primary" disabled={busy} onClick={() => void load(result.name, result.sha256 ?? null)}>
                Load this SAE
              </Button>
            )}
          </TrainingFooter>
        ) : (
          <>
            <Button variant="ghost" onClick={close}>
              Cancel
            </Button>
            <Button variant="primary" disabled={!spec || busy || !!planned.plan?.refusal} onClick={() => void start()}>
              Train
            </Button>
          </>
        )
      }
    >
      {!info ? (
        <Empty title="Load a model first">An SAE is trained on the activations of the loaded model.</Empty>
      ) : job ? (
        <TrainingView job={job} what="SAE" value="fvu">
          {result?.held_out && <SaeMeasured held={result.held_out} />}
        </TrainingView>
      ) : (
        <>
          {corporaError && <Callout tone="error">{corporaError}</Callout>}
          {corpora && readable.length === 0 && <NoCorpus />}
          {fields && (
            <>
              <Common fields={fields} corpora={readable} onChange={set} what="SAE" folder="saes" />
              <div className={t.grid}>
                <Field label="Site">
                  <Select value={fields.site} onChange={(e) => set({ site: e.target.value as SaeFields["site"] })}>
                    {SAE_SITES.filter((s) => info.site_kinds.includes(s)).map((s) => (
                      <option key={s} value={s}>
                        {SAE_SITE_WORDS[s]}
                      </option>
                    ))}
                  </Select>
                </Field>
                <Field label="Layer">
                  <Select value={fields.layer} onChange={(e) => set({ layer: Number(e.target.value) })}>
                    {Array.from({ length: info.n_layers }, (_, i) => (
                      <option key={i} value={i}>
                        Layer {i}
                      </option>
                    ))}
                  </Select>
                </Field>
                <Field label="Features" help={`The activation is ${count(info.d_model)} wide.`}>
                  <Input inputMode="numeric" value={fields.dSae} onChange={(e) => set({ dSae: e.target.value })} />
                </Field>
                <Field label="Features per token" help="TopK keeps each token's largest.">
                  <Input inputMode="numeric" value={fields.k} onChange={(e) => set({ k: e.target.value })} />
                </Field>
              </div>
              <Optimizer fields={fields} onChange={set} />
              <details className={t.more}>
                <summary>
                  Every other setting <span className={t.summary}>{saeSettingsText(fields, info.d_model)}</span>
                </summary>
                <div className={t.moreBody}>
                  <Field label="Activations">
                    <Segmented
                      label="Activations"
                      value={fields.normalize}
                      onChange={(normalize) => set({ normalize })}
                      options={[
                        { value: "mean_squared_norm", label: `Scaled to a mean squared norm of ${count(info.d_model)}` },
                        { value: "none", label: "As they are" },
                      ]}
                    />
                  </Field>
                  <Checkbox checked={fields.auxk} onChange={(auxk) => set({ auxk })}>
                    AuxK: dead latents reconstruct what the live ones miss (Gao et al. 2024)
                  </Checkbox>
                  {fields.auxk && (
                    <div className={t.grid3}>
                      <Field label="Dead latents used">
                        <Input inputMode="numeric" value={fields.auxkK} onChange={(e) => set({ auxkK: e.target.value })} />
                      </Field>
                      <Field label="Coefficient">
                        <Input inputMode="decimal" value={fields.auxkCoefficient} onChange={(e) => set({ auxkCoefficient: e.target.value })} />
                      </Field>
                      <Field label="Dead after, in tokens">
                        <Input inputMode="numeric" value={fields.auxkDeadAfter} onChange={(e) => set({ auxkDeadAfter: e.target.value })} />
                      </Field>
                    </div>
                  )}
                  <CommonMore fields={fields} onChange={set} />
                </div>
              </details>
              {built?.error ? <Callout tone="error">{built.error}</Callout> : <PlanPanel {...planned} />}
            </>
          )}
        </>
      )}
    </Dialog>
  );
}

function TrainLens() {
  const model = useStore((st) => st.model);
  const info = model.info;
  const busy = useStore((st) => st.job?.status === "running");
  const guard = useStore((st) => st.guard);
  const version = useStore((st) => st.trainedVersion);
  const { corpora, error: corporaError } = useCorpora();
  const [trained, setTrained] = useState<TrainedLens[] | null>(null);
  const [fields, setFields] = useState<LensFields | null>(null);
  const [startedId, setStartedId] = useState<string | null>(null);
  const job = useTraining("train_lens", startedId);

  useEffect(() => {
    api.lenses().then(setTrained, () => setTrained([]));
  }, [version]);
  const readable = (corpora ?? []).filter((c) => !c.error);
  useEffect(() => {
    if (!fields && info && corpora && trained) setFields(lensFields(info, readable[0]?.name ?? "", trained.map((s) => s.name)));
  }, [info, corpora, trained, fields, readable]);

  const ref = info ? modelRefOf(info, model.ref?.device ?? info.device) : null;
  const built = fields && ref && info ? lensSpec(fields, ref, info.n_layers) : null;
  const spec = built?.spec ?? null;
  const planned = usePlan(job ? null : spec, (s) => api.planLens(s));
  const set = (patch: Partial<LensFields>) => setFields((f) => (f ? { ...f, ...patch } : f));
  const start = async () => {
    if (!spec) return;
    const out = await guard(() => api.trainLens(spec));
    if (out) {
      setStartedId(out.id);
      useStore.getState().applyJob(out);
    }
  };
  const progress = (job?.progress ?? {}) as TrainingProgress;
  const result = job?.status === "finished" ? trained?.find((s) => s.name === progress.name) ?? null : null;
  const use = (lens: TrainedLens) => {
    useStore.setState({ trainDialog: null, chosenLens: { name: lens.name, sha256: lens.sha256 ?? null } });
    useStore.getState().setView("predictions");
  };
  const last = info ? info.n_layers - 1 : 0;
  return (
    <Dialog
      open
      onOpenChange={(open) => !open && close()}
      title="Train a tuned lens"
      description="An affine translator for each layer, trained so that the layer's residual stream, read through the final normalization and unembedding, predicts what the model predicts (Belrose et al. 2023). Saved as lenses/NAME; it reads only the model it was trained on."
      wide
      footer={
        job ? (
          <TrainingFooter job={job} onBack={() => setStartedId(null)}>
            {result && result.fits && (
              <Button variant="primary" onClick={() => use(result)}>
                Read the layer predictions with it
              </Button>
            )}
          </TrainingFooter>
        ) : (
          <>
            <Button variant="ghost" onClick={close}>
              Cancel
            </Button>
            <Button variant="primary" disabled={!spec || busy || !!planned.plan?.refusal} onClick={() => void start()}>
              Train
            </Button>
          </>
        )
      }
    >
      {!info ? (
        <Empty title="Load a model first">A lens is trained on the residual stream of the loaded model.</Empty>
      ) : info.extra?.prediction_method !== "final_norm_logit_lens" ? (
        <Callout>
          A lens reads the residual stream through the model&rsquo;s final normalization and unembedding, which didn&rsquo;t reproduce{" "}
          {modelName(info.id)}&rsquo;s output when it loaded, so no lens can be trained on it.
        </Callout>
      ) : job ? (
        <TrainingView job={job} what="lens" value="kl">
          {result?.held_out && <LensMeasured held={result.held_out} />}
        </TrainingView>
      ) : (
        <>
          {corporaError && <Callout tone="error">{corporaError}</Callout>}
          {corpora && readable.length === 0 && <NoCorpus />}
          {fields && (
            <>
              <Common fields={fields} corpora={readable} onChange={set} what="lens" folder="lenses" />
              <div className={t.grid}>
                <Field label="Layers">
                  <Segmented
                    label="Layers"
                    value={fields.everyLayer ? "every" : "chosen"}
                    onChange={(v) => set({ everyLayer: v === "every" })}
                    options={[
                      { value: "every", label: last > 1 ? `Every layer but the last (0 to ${last - 1})` : "Layer 0" },
                      { value: "chosen", label: "Chosen layers" },
                    ]}
                  />
                </Field>
                {!fields.everyLayer && (
                  <Field label="Chosen layers" help={`Such as 0, 2, 4 or 0-3; the last layer, ${last}, reads as the model's own prediction.`}>
                    <Input value={fields.layers} spellCheck={false} onChange={(e) => set({ layers: e.target.value })} />
                  </Field>
                )}
              </div>
              <Optimizer fields={fields} onChange={set} />
              <details className={t.more}>
                <summary>
                  Every other setting <span className={t.summary}>{lensSettingsText(fields)}</span>
                </summary>
                <div className={t.moreBody}>
                  <Checkbox checked={fields.weightDecay} onChange={(weightDecay) => set({ weightDecay })}>
                    Weight decay towards the logit lens (AdamW)
                  </Checkbox>
                  {fields.weightDecay && (
                    <Field label="Weight decay">
                      <Input inputMode="decimal" value={fields.decay} onChange={(e) => set({ decay: e.target.value })} />
                    </Field>
                  )}
                  <CommonMore fields={fields} onChange={set} />
                </div>
              </details>
              {built?.error ? <Callout tone="error">{built.error}</Callout> : <PlanPanel {...planned} />}
            </>
          )}
        </>
      )}
    </Dialog>
  );
}

function NoCorpus() {
  return (
    <Callout title="This project has no corpus yet">
      A corpus is the plain text a training reads: add one to the project first, from a text or JSONL file, a Hugging Face dataset or a
      dataset&rsquo;s prompts.
    </Callout>
  );
}

/** The fields both trainings have: the name and the corpus. */
function Common<F extends CommonFields>({
  fields,
  corpora,
  onChange,
  what,
  folder,
}: {
  fields: F;
  corpora: CorpusListing[];
  onChange: (patch: Partial<F>) => void;
  what: string;
  folder: string;
}) {
  return (
    <div className={t.grid}>
      <Field label="Name" help={`Saved as ${folder}/${fields.name || "NAME"}, never over another.`}>
        <Input value={fields.name} spellCheck={false} onChange={(e) => onChange({ name: e.target.value } as Partial<F>)} />
      </Field>
      <Field label="Corpus" help={`The texts the ${what} trains on.`}>
        <Select value={fields.corpus} onChange={(e) => onChange({ corpus: e.target.value } as Partial<F>)}>
          {corpora.length === 0 && <option value="">No corpus</option>}
          {corpora.map((c) => (
            <option key={c.name} value={c.name}>
              {c.name} · {count(c.n ?? 0)} texts
            </option>
          ))}
        </Select>
      </Field>
    </div>
  );
}

/** Steps, tokens per step, learning rate, seed and held-out share: what changes a training most. */
function Optimizer<F extends CommonFields>({ fields, onChange }: { fields: F; onChange: (patch: Partial<F>) => void }) {
  const field = (key: keyof CommonFields) => (e: { target: { value: string } }) => onChange({ [key]: e.target.value } as Partial<F>);
  return (
    <div className={t.grid3}>
      <Field label="Steps">
        <Input inputMode="numeric" value={fields.steps} onChange={field("steps")} />
      </Field>
      <Field label="Tokens per step">
        <Input inputMode="numeric" value={fields.batchTokens} onChange={field("batchTokens")} />
      </Field>
      <Field label="Learning rate" help="The same at every step.">
        <Input inputMode="decimal" value={fields.lr} onChange={field("lr")} />
      </Field>
      <Field label="Seed" help="Of the split, the orders and the start.">
        <Input inputMode="numeric" value={fields.seed} onChange={field("seed")} />
      </Field>
      <Field label="Held-out share" help="Of the texts: measured on, not trained on.">
        <Input inputMode="decimal" value={fields.heldOut} onChange={field("heldOut")} />
      </Field>
    </div>
  );
}

/** The settings both trainings have that rarely change: Adam's betas and epsilon, the curve's
 * points, and how texts become tokens. */
function CommonMore<F extends CommonFields>({ fields, onChange }: { fields: F; onChange: (patch: Partial<F>) => void }) {
  const field = (key: keyof CommonFields) => (e: { target: { value: string } }) => onChange({ [key]: e.target.value } as Partial<F>);
  return (
    <>
      <div className={t.grid3}>
        <Field label="Adam's first beta">
          <Input inputMode="decimal" value={fields.beta1} onChange={field("beta1")} />
        </Field>
        <Field label="Adam's second beta">
          <Input inputMode="decimal" value={fields.beta2} onChange={field("beta2")} />
        </Field>
        <Field label="Adam's epsilon">
          <Input inputMode="decimal" value={fields.eps} onChange={field("eps")} />
        </Field>
        <Field label="A point of the curve every" help="Steps; each point is the mean of the steps since the last.">
          <Input inputMode="numeric" value={fields.logEvery} onChange={field("logEvery")} />
        </Field>
        <Field label="Tokens per text" help="The rest of a longer text is left out.">
          <Input inputMode="numeric" value={fields.maxTokens} onChange={field("maxTokens")} />
        </Field>
        <Field label="Texts per forward pass" help="Part of the spec: batch shape can change the last digits.">
          <Input inputMode="numeric" value={fields.modelBatch} onChange={field("modelBatch")} />
        </Field>
      </div>
      <Checkbox checked={fields.prependBos} onChange={(prependBos) => onChange({ prependBos } as Partial<F>)}>
        Begin each text with the beginning-of-sequence token (its own position is never trained on)
      </Checkbox>
    </>
  );
}

/** What training would take here, or why it can't train. */
function PlanPanel({ plan, error, busy }: { plan: TrainingPlan | null; error: string | null; busy: boolean }) {
  if (error) return <Callout tone="error">{error}</Callout>;
  if (busy || !plan) {
    return (
      <p className={t.plan} role="status">
        <Spinner label="Estimating" /> Estimating what training takes on this machine…
      </p>
    );
  }
  if (plan.refusal) return <Callout tone="error" title="It can't train as set">{plan.refusal}</Callout>;
  const text = planText(plan);
  return (
    <div className={t.plan} role="status">
      <p>{text.reads}</p>
      <p>
        {text.memory} {text.time}
      </p>
      {plan.memory?.verdict === "wont_fit" && (
        <Callout title="It won't fit in memory">
          Lower the steps or tokens per step (the activations of every token it trains on are held at once), or the number of features.
        </Callout>
      )}
    </div>
  );
}

/** A training in progress, or how it ended, with its loss curve. */
function TrainingView({ job, what, value, children }: { job: Job; what: "SAE" | "lens"; value: "fvu" | "kl"; children?: ReactNode }) {
  const progress = (job.progress ?? {}) as TrainingProgress;
  const name = progress.name ?? job.title.replace(/^Train the (SAE|lens) /, "");
  const where = progressText(progress, what, name);
  const curve = progress.curve ?? [];
  const latest = curve[curve.length - 1];
  return (
    <div className={t.training}>
      {job.status === "running" && (
        <div className={t.progress} role="status" aria-live="polite">
          <p>
            <Spinner label="Training" /> {job.cancelling ? "Cancelling…" : where.text}
          </p>
          <Progress value={where.share} label={`Progress of the ${what} training`} />
        </div>
      )}
      {job.status === "failed" && <Callout tone="error" title="The training stopped">{job.error}</Callout>}
      {job.status === "cancelled" && <Callout title="The training was cancelled">Nothing was written: the {what} isn&rsquo;t saved.</Callout>}
      {job.status === "finished" && <p className={t.done}>{where.text}.</p>}
      {curve.length > 0 && (
        <section aria-label="Loss curve">
          <p className={t.note}>
            {value === "fvu"
              ? `Fraction of variance unexplained on the training batches${latest?.l0 !== undefined ? ` · ${num(latest.l0, 1)} features per token at step ${count(latest.step)}` : ""}`
              : "KL divergence from the model's prediction on the training batches, the mean over the trained layers"}
          </p>
          <LossCurve points={curve} value={value} />
        </section>
      )}
      {children}
    </div>
  );
}

function TrainingFooter({ job, onBack, children }: { job: Job; onBack: () => void; children?: ReactNode }) {
  if (job.status === "running") {
    return (
      <>
        <Button variant="ghost" onClick={close} tip="It goes on training; the status line shows how far it is.">
          Close
        </Button>
        <Button disabled={job.cancelling} onClick={() => void useStore.getState().cancelJob()}>
          {job.cancelling ? "Cancelling…" : "Cancel the training"}
        </Button>
      </>
    );
  }
  return (
    <>
      <Button variant="ghost" onClick={onBack}>
        {job.status === "finished" ? "Train another" : "Back to the settings"}
      </Button>
      {children ?? (
        <Button variant="primary" onClick={close}>
          Close
        </Button>
      )}
    </>
  );
}

/** What an SAE's training measured on its held-out texts. */
function SaeMeasured({ held }: { held: NonNullable<TrainedSAE["held_out"]> }) {
  const loss = held.loss;
  return (
    <section aria-label="Measured on the held-out texts">
      <h3 className={t.heading}>On {count(held.tokens)} held-out tokens</h3>
      <dl className={t.metrics}>
        <div>
          <dt>Variance explained</dt>
          <dd className="figure">{share(held.variance_explained)}</dd>
        </div>
        <div>
          <dt>Features per token</dt>
          <dd className="figure">{num(held.l0, 1)}</dd>
        </div>
        <div>
          <dt>Never fire</dt>
          <dd className="figure">{share(held.dead_share)}</dd>
        </div>
        {loss && (
          <div>
            <dt>Loss recovered</dt>
            <dd className="figure">{share(loss.recovered)}</dd>
          </div>
        )}
      </dl>
      {loss && (
        <p className={t.note}>
          Next-token loss {num(loss.clean, 3)} as the model runs, {num(loss.spliced, 3)} with the SAE&rsquo;s reconstruction in its place,{" "}
          {num(loss.zero, 3)} with the site set to zero. The loss recovered is how much of the gap from zero to the model the SAE closes.
        </p>
      )}
    </section>
  );
}

/** Each layer's held-out KL divergence from the model's prediction, before and after. */
function LensMeasured({ held }: { held: NonNullable<TrainedLens["held_out"]> }) {
  return (
    <section aria-label="Measured on the held-out texts">
      <h3 className={t.heading}>KL divergence from the model&rsquo;s prediction, on {count(held.tokens)} held-out tokens</h3>
      <table className={t.table}>
        <thead>
          <tr>
            <th scope="col">Layer</th>
            <th scope="col">Logit lens</th>
            <th scope="col">Tuned lens</th>
          </tr>
        </thead>
        <tbody>
          {held.layers.map((row) => (
            <tr key={row.layer}>
              <th scope="row">Layer {row.layer}</th>
              <td>{num(row.logit_lens, 4)}</td>
              <td>{row.tuned_lens === null ? "No translator" : num(row.tuned_lens, 4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

/** A training's loss curve, drawn in ink: its value at each recorded step, from zero up. */
export function LossCurve({ points, value }: { points: CurvePoint[]; value: "fvu" | "kl" }) {
  const host = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const { width } = useElementSize(host);
  const colors = useChromeColors();
  const values = points.map((p) => p[value]).filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  const latest = points[points.length - 1];
  useEffect(() => {
    if (!canvas.current || width < 1 || values.length === 0) return;
    const height = 150;
    const left = 52;
    const right = 14;
    const top = 10;
    const bottom = 26;
    const ctx = prepareCanvas(canvas.current, width, height);
    if (!ctx) return;
    const max = Math.max(...values) || 1;
    const last = Math.max(1, ...points.map((p) => p.step));
    const x = (step: number) => left + (step / last) * (width - left - right);
    const y = (v: number) => top + (1 - v / max) * (height - top - bottom);
    ctx.font = font(colors, 11, 450);
    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    for (const v of [0, max / 2, max]) {
      ctx.fillStyle = colors.muted;
      ctx.fillText(v === 0 ? "0" : v.toPrecision(2), left - 8, y(v));
      ctx.strokeStyle = colors.lineSoft;
      ctx.beginPath();
      ctx.moveTo(left, y(v));
      ctx.lineTo(width - right, y(v));
      ctx.stroke();
    }
    ctx.textAlign = "center";
    ctx.fillText("0", x(0), height - 10);
    ctx.fillText(`step ${count(last)}`, Math.min(x(last), width - right - 30), height - 10);
    ctx.strokeStyle = colors.text;
    ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach((p, i) => {
      const v = p[value];
      if (typeof v !== "number") return;
      if (i === 0) ctx.moveTo(x(p.step), y(v));
      else ctx.lineTo(x(p.step), y(v));
    });
    ctx.stroke();
    // A curve of few points shows each one.
    if (points.length <= 40) {
      ctx.fillStyle = colors.text;
      for (const p of points) {
        const v = p[value];
        if (typeof v !== "number") continue;
        ctx.beginPath();
        ctx.arc(x(p.step), y(v), 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, [width, points, values, value, colors]);
  const shown = latest?.[value];
  return (
    <div className={t.curve} ref={host}>
      <canvas
        ref={canvas}
        role="img"
        aria-label={`The loss curve over ${count(points.length)} points${typeof shown === "number" ? `: ${num(shown, 4)} at step ${count(latest.step)}` : ""}.`}
      />
    </div>
  );
}
