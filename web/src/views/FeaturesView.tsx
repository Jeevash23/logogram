import { useEffect, useRef, useState } from "react";

import { api } from "../api/client";
import type { CatalogueSAE, FeatureReport, SAEInfo, TokenFeatures, TrainedSAE } from "../api/types";
import { FeatureWrites } from "../components/FeatureWrites";
import { Button, Callout, Empty, Field, Input, Progress, Segmented, Select, Spinner, useRadioGroup } from "../components/ui";
import { inkScale, textOn } from "../lib/color";
import { bytes, count, num, pct, signed, visibleToken } from "../lib/format";
import { modelName, useAnalysisContext } from "../lib/hooks";
import { KIND_NAMES } from "../lib/sites";
import { saeName, saeRefOf, steerSiteOf } from "../lib/spec";
import { POOR_FIT } from "../lib/thresholds";
import { useStore } from "../store/app";
import { ProjectSAEs, SAEPicker } from "./SAEPicker";
import s from "./views.module.css";
import f from "./FeaturesView.module.css";

/** What an SAE reads (and a transcoder writes), in words. */
function siteOf(info: SAEInfo): string {
  if (info.transcoder) {
    return `layer ${info.layer}'s MLP input and writes its output${info.skip ? ", plus a skip connection from the input" : ""}`;
  }
  switch (info.site) {
    case "resid_pre":
      return `the residual stream entering layer ${info.layer}`;
    case "resid_post":
      return `the residual stream leaving layer ${info.layer}`;
    case "attn_out":
      return `layer ${info.layer}'s attention output`;
    case "mlp_out":
      return `layer ${info.layer}'s MLP output`;
    case "head":
      return `the outputs (z) of every attention head in layer ${info.layer}, side by side`;
    default:
      return `the ${KIND_NAMES[info.site]} of layer ${info.layer}`;
  }
}

/** Below this, an SAE describes too little of the activations for its features to mean much. */

export function FeaturesView() {
  const model = useStore((st) => st.model);
  const sae = useStore((st) => st.sae);
  // Choosing another SAE while one is loaded: it stays loaded until loading another begins.
  const [choosing, setChoosing] = useState(false);
  const loadedKey = sae.info ? `${sae.info.source ?? "hub"}:${sae.info.repo}/${sae.info.path}@${sae.info.revision ?? sae.info.sha256}` : null;
  useEffect(() => setChoosing(false), [loadedKey]);
  if (model.state !== "ready" || !model.info) {
    return (
      <div className={s.view}>
        <Empty
          title="Load a model to read its features"
          action={<Button variant="primary" onClick={() => useStore.setState({ modelDialogOpen: true })}>Load a model</Button>}
        >
          A sparse autoencoder (SAE) rewrites one of the model's activations as a few active features out of thousands.
        </Empty>
      </div>
    );
  }
  return (
    <div className={s.view}>
      <div className={s.head}>
        <div className={s.titleBlock}>
          <h1 className={s.title}>{sae.state === "ready" && sae.info ? "What the SAE reads in these prompts" : "Read features with a sparse autoencoder"}</h1>
          <p className={s.subtitle}>
            An SAE writes an activation as a few active features, each a direction in the model, plus an error it misses. Load
            one trained on {modelName(model.info.id)}, check that it fits these prompts, then look at, patch or estimate its features.
          </p>
        </div>
      </div>
      {sae.state === "ready" && sae.info && !choosing ? (
        <Loaded info={sae.info} onChoose={() => setChoosing(true)} />
      ) : (
        <Loader onKeep={sae.state === "ready" && sae.info ? () => setChoosing(false) : undefined} />
      )}
    </div>
  );
}

function Loader({ onKeep }: { onKeep?: () => void }) {
  const model = useStore((st) => st.model);
  const sae = useStore((st) => st.sae);
  const job = useStore((st) => st.job);
  const guard = useStore((st) => st.guard);
  const [suggestions, setSuggestions] = useState<{ repo: string; detail: string }[]>([]);
  const [repo, setRepo] = useState("");
  const [folders, setFolders] = useState<{ repo: string; revision: string; folders: string[] } | null>(null);
  const [folder, setFolder] = useState("");
  const [finding, setFinding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const id = model.info?.id ?? "";
  const suggestionRadio = useRadioGroup(suggestions.map((x) => ({ value: x.repo })), repo, (next) => {
    setRepo(next);
    setFolders(null);
  });

  useEffect(() => {
    if (!id) return;
    api.saeSuggestions(id).then((list) => {
      setSuggestions(list);
      if (list.length) setRepo((r) => r || list[0].repo);
    }, () => undefined);
  }, [id]);

  const find = async () => {
    setFinding(true);
    setError(null);
    setFolders(null);
    try {
      const found = await api.saeFolders(repo.trim());
      setFolders(found);
      setFolder(found.folders[0] ?? "");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setFinding(false);
    }
  };
  const load = async () => {
    if (!folders) return;
    const out = await guard(() => api.loadSae({ repo: folders.repo, path: folder, revision: folders.revision, source: "hub", sha256: null }));
    if (out) useStore.getState().applyJob(out);
  };
  // A published SAE by its release and id: its repository and folder, at the repository's current
  // commit, which loading pins and specs record.
  const loadPublished = async (x: CatalogueSAE) => {
    const out = await guard(() => api.loadSae({ repo: x.repo, path: x.folder, revision: null, source: "hub", sha256: null }));
    if (out) useStore.getState().applyJob(out);
  };
  // An SAE trained in this project, by its name and the hash of its files when it was trained.
  const loadTrained = async (x: TrainedSAE) => {
    const out = await guard(() => api.loadSae({ repo: x.name, path: x.path, revision: null, source: "project", sha256: x.sha256 ?? null }));
    if (out) useStore.getState().applyJob(out);
  };
  const loading = sae.state === "loading";
  const busy = loading || job?.status === "running";

  return (
    <section className={f.loader} aria-label="Load an SAE">
      {onKeep && (
        <div className={f.keep}>
          <span className={s.small}>Loading another SAE replaces {sae.info ? saeName(sae.info) : "the loaded one"}.</span>
          <Button size="small" variant="ghost" onClick={onKeep}>
            Keep the loaded SAE
          </Button>
        </div>
      )}
      <ProjectSAEs busy={busy} onLoad={(x) => void loadTrained(x)} />
      <SAEPicker modelId={id} busy={busy} onLoad={(x) => void loadPublished(x)} />
      <h2 className={s.panelTitle}>Or name a repository and folder</h2>
      {suggestions.length > 0 && (
        <div className={f.suggestions} role="radiogroup" aria-label="Repositories with SAEs for this model">
          {suggestions.map((x, i) => (
            <button key={x.repo} type="button" role="radio" aria-checked={repo === x.repo} className={f.suggestion} onClick={() => { setRepo(x.repo); setFolders(null); }} {...suggestionRadio(i)}>
              <span className={f.dot} />
              <span className={f.repo}>{x.repo}</span>
              <span className={f.detail}>{x.detail}</span>
            </button>
          ))}
        </div>
      )}
      <div className={s.row}>
        <Field label="Hugging Face repository" help="SAEs and transcoders in SAELens, EleutherAI or Gemma Scope 2 format, read from safetensors files only.">
          <Input value={repo} onChange={(e) => { setRepo(e.target.value); setFolders(null); }} placeholder="owner/name" spellCheck={false} style={{ width: 360 }} />
        </Field>
        <Button onClick={() => void find()} disabled={!/^[\w.-]+\/[\w.-]+$/.test(repo.trim()) || finding}>
          {finding ? "Looking…" : "Find SAEs"}
        </Button>
      </div>
      {error && <Callout tone="error">{error}</Callout>}
      {folders && (
        <div className={s.row}>
          <Field label="Which SAE" help={`${count(folders.folders.length)} in ${folders.repo} at ${folders.revision.slice(0, 10)}. Each reads one site of one layer.`}>
            <Select value={folder} onChange={(e) => setFolder(e.target.value)} aria-label="Which SAE">
              {folders.folders.map((x) => (
                <option key={x} value={x}>
                  {x || "(top level)"}
                </option>
              ))}
            </Select>
          </Field>
          <Button variant="primary" onClick={() => void load()} disabled={busy}>
            Load
          </Button>
        </div>
      )}
      {loading && (
        <div className={f.progress}>
          <span className={s.small}>
            <Spinner /> {sae.total ? `Downloading ${bytes(sae.done ?? 0)} of ${bytes(sae.total)}` : "Loading the SAE"}
          </span>
          <Progress value={sae.total ? (sae.done ?? 0) / sae.total : null} label="Loading the SAE" />
        </div>
      )}
      {sae.state === "error" && sae.error && <Callout tone="error" title="The SAE didn't load">{sae.error}</Callout>}
    </section>
  );
}

function Loaded({ info, onChoose }: { info: SAEInfo; onChoose: () => void }) {
  const dataset = useStore((st) => st.datasetPath);
  const context = useAnalysisContext();
  const guard = useStore((st) => st.guard);
  const [measuring, setMeasuring] = useState(false);
  const [feature, setFeature] = useState<number | null>(null);
  const fit = info.fit;

  const measure = async () => {
    if (!dataset) return;
    setMeasuring(true);
    await guard(() => api.saeFit(dataset, context.options));
    setMeasuring(false);
  };

  return (
    <>
      <section className={f.card} aria-label="The loaded SAE">
        <div className={f.cardHead}>
          <div>
            <p className={s.eyebrow}>{info.source === "project" ? "Trained in this project" : info.repo}</p>
            <h2 className={f.cardTitle}>{info.source === "project" ? info.repo : info.path || "(top level)"}</h2>
            <p className={s.small}>
              Reads {siteOf(info)} · {count(info.d_sae)} features ·{" "}
              {info.activation === "topk" ? `the top ${info.k} per token` : info.activation === "jumprelu" ? "JumpReLU" : "ReLU"}
              {info.normalize === "layer_norm" ? " · inputs standardized per token" : ""}
            </p>
          </div>
          <div className={s.headActions}>
            <Button size="small" onClick={() => void measure()} disabled={!dataset || measuring}>
              {measuring ? "Measuring…" : fit ? "Measure again" : "Measure the fit"}
            </Button>
            <Button size="small" onClick={onChoose}>
              Load another SAE
            </Button>
            <Button size="small" variant="ghost" onClick={() => void guard(() => api.unloadSae())}>
              Unload
            </Button>
          </div>
        </div>
        {(info.warnings ?? []).map((warning) => (
          <Callout key={warning}>{warning}</Callout>
        ))}
        {info.source === "project" && info.trained?.held_out && (
          <p className={s.small}>
            Trained on the corpus {info.trained.corpus ?? "of this project"}: on its held-out text it explains{" "}
            {pct(info.trained.held_out.variance_explained)} of the variance and recovers {pct(info.trained.held_out.loss?.recovered)} of the
            next-token loss. Its files hash to {(info.sha256 ?? "").slice(0, 12)}, which specs pin.
          </p>
        )}
        {fit ? (
          <dl className={f.fit}>
            <div>
              <dt>Variance explained</dt>
              <dd className="figure">{pct(fit.variance_explained)}</dd>
            </div>
            <div>
              <dt>Features per token</dt>
              <dd className="figure">{num(fit.l0, 1)}</dd>
            </div>
            {fit.logit_diff != null && (
              <div>
                <dt>Logit diff, model → SAE spliced in</dt>
                <dd className="figure">
                  {signed(fit.logit_diff)} → {signed(fit.spliced_logit_diff)}
                </dd>
              </div>
            )}
          </dl>
        ) : (
          <p className={s.small}>Not measured on these prompts yet. Measure it before reading the features: an SAE trained on another model, or with other weight processing, fits poorly.</p>
        )}
        {fit && fit.variance_explained < POOR_FIT && (
          <Callout title="This SAE fits these activations poorly">
            It explains {pct(fit.variance_explained)} of their variance, under the {pct(POOR_FIT)} counted as a fit, so its features
            describe little of what the model does here.
            SAEs only fit activations like the ones they were trained on: try loading the model with TransformerLens's weight
            processing switched the other way, or an SAE trained on this model.
          </Callout>
        )}
      </section>
      <div className={f.columns}>
        <TokenPanel info={info} selected={feature} onFeature={setFeature} />
        {feature !== null ? <FeaturePanel info={info} feature={feature} /> : <Empty title="Choose a feature">Pick one of the features on a token to see where it fires.</Empty>}
      </div>
    </>
  );
}

function TokenPanel({ info, selected, onFeature }: { info: SAEInfo; selected: number | null; onFeature: (f: number) => void }) {
  const dataset = useStore((st) => st.datasetPath);
  const index = useStore((st) => st.promptIndex);
  const token = useStore((st) => st.tokenPosition);
  const context = useAnalysisContext();
  const [which, setWhich] = useState<"clean" | "corrupt">("clean");
  const [data, setData] = useState<TokenFeatures | null>(null);
  const [error, setError] = useState<string | null>(null);
  const seq = useRef(0);

  useEffect(() => {
    if (!dataset) return;
    const mine = ++seq.current;
    setData(null);
    setError(null);
    api.saeTokens(dataset, index, which, context.options).then(
      (d) => mine === seq.current && setData(d),
      (e: Error) => mine === seq.current && setError(e.message),
    );
  }, [dataset, index, which, context.key, info.repo, info.path, info.revision, info.sha256]);

  return (
    <section className={f.panel} aria-label="Features on each token">
      <div className={f.panelHead}>
        <span className={s.panelTitle}>Prompt {index}, token by token</span>
        <Segmented label="Prompt" value={which} onChange={setWhich} options={[{ value: "clean", label: "Clean" }, { value: "corrupt", label: "Corrupt" }]} />
      </div>
      <p className={s.small}>Choose a token to patch features at its position; choose a feature to read it.</p>
      {error && <Callout tone="error">{error}</Callout>}
      {!data && !error && <p className={s.small}><Spinner /> Encoding the prompt</p>}
      {data && (
        <ol className={f.tokens}>
          {data.tokens.map((t, p) => (
            <li key={p} className={f.tokenRow} aria-current={token === p ? "true" : undefined}>
              <button type="button" className={f.token} onClick={() => useStore.getState().setTokenPosition(p)}>
                <span className={f.position}>{p}</span>
                <span className={f.text}>{visibleToken(t)}</span>
              </button>
              <span className={f.chips}>
                {p < data.first_real_token ? (
                  <span className={s.faint}>not read: SAEs aren't trained on this token</span>
                ) : !data.features[p]?.length ? (
                  <span className={s.faint}>no feature fires</span>
                ) : (
                  data.features[p].slice(0, 5).map((x) => (
                    <button key={x.feature} type="button" className={f.chip} aria-pressed={selected === x.feature} onClick={() => onFeature(x.feature)}>
                      F{x.feature} <span className={f.value}>{num(x.activation, 1)}</span>
                    </button>
                  ))
                )}
                {p >= data.first_real_token && data.active[p] > 5 && <span className={s.faint}>+{data.active[p] - 5}</span>}
              </span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

function FeaturePanel({ info, feature }: { info: SAEInfo; feature: number }) {
  const dataset = useStore((st) => st.datasetPath);
  const index = useStore((st) => st.promptIndex);
  const token = useStore((st) => st.tokenPosition);
  const context = useAnalysisContext();
  const theme = useStore((st) => st.theme);
  const [data, setData] = useState<FeatureReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const seq = useRef(0);

  useEffect(() => {
    if (!dataset) return;
    const mine = ++seq.current;
    setData(null);
    setError(null);
    api.saeFeature(dataset, index, "clean", feature, context.options).then(
      (d) => mine === seq.current && setData(d),
      (e: Error) => mine === seq.current && setError(e.message),
    );
  }, [dataset, index, feature, context.key, info.repo, info.path, info.revision, info.sha256]);

  const peak = data ? Math.max(1e-9, ...data.activations) : 1;
  // Feature activations are never negative: the single-hue ink ramp, as for attention.
  const ink = inkScale(peak, theme);
  const ref = saeRefOf(info);
  const patch = () => {
    const st = useStore.getState();
    st.setView("experiment");
    st.replaceForm({
      ...st.form,
      kind: "activation_patching",
      saeRef: ref,
      scope: {
        kind: "sites",
        sites: [{ kind: "sae_feature", layer: info.layer, feature, position: token === null ? { kind: "last" } : { kind: "index", index: token } }],
      },
      nameEdited: false,
      draftId: null,
    });
  };
  const estimate = () => {
    const st = useStore.getState();
    st.setView("experiment");
    st.replaceForm({ ...st.form, kind: "attribution_patching", atpMethod: "gradient", saeRef: ref, scope: { kind: "features", position: { kind: "last" }, top: 50, choose_on: null, seed: null }, nameEdited: false, draftId: null });
  };
  // Steering adds the feature's decoder direction where the SAE writes, at the chosen token.
  const steer = () => {
    const st = useStore.getState();
    const where = steerSiteOf(info);
    st.setView("experiment");
    st.replaceForm({
      ...st.form,
      kind: "steering",
      steerDirection: "sae_feature",
      steerFeature: feature,
      saeRef: ref,
      scope: { kind: "sites", sites: [{ ...where, head: null, position: token === null ? { kind: "last" } : { kind: "index", index: token } }] },
      nameEdited: false,
      draftId: null,
    });
  };

  return (
    <section className={f.panel} aria-label={`Feature ${feature}`}>
      <div className={f.panelHead}>
        <span className={s.panelTitle}>Feature {feature}</span>
        <div className={s.headActions}>
          <Button size="small" onClick={patch}>
            {token === null ? "Patch this feature at the last token" : `Patch this feature at token ${token}`}
          </Button>
          <Button
            size="small"
            onClick={steer}
            tip={`Fill the experiment form: add its decoder direction where the SAE writes, at ${token === null ? "the last token" : `token ${token}`}`}
          >
            Steer along this feature
          </Button>
          <Button size="small" variant="ghost" onClick={estimate}>
            Estimate every feature
          </Button>
        </div>
      </div>
      {error && <Callout tone="error">{error}</Callout>}
      {!data && !error && <p className={s.small}><Spinner /> Reading the feature on every prompt</p>}
      {data && (
        <>
          <p className={s.small}>Along clean prompt {index}: darker where it fires more (peak {num(peak, 2)}).</p>
          <div className={f.strip} role="img" aria-label={`Feature ${feature} along prompt ${index}: ${data.tokens.map((t, i) => `${visibleToken(t)} ${num(data.activations[i], 2)}`).join(", ")}`}>
            {data.tokens.map((t, i) => {
              const fill = ink(Math.max(0, data.activations[i]));
              return (
                <span key={i} className={f.stripToken} style={{ background: fill, color: textOn(fill) }} title={`${visibleToken(t)}: ${num(data.activations[i], 3)}`}>
                  {visibleToken(t)}
                </span>
              );
            })}
          </div>
          <h3 className={s.panelTitle}>Tokens it promotes and suppresses directly</h3>
          {dataset && (
            <FeatureWrites
              dataset={dataset}
              index={index}
              which="clean"
              feature={feature}
              position={token === null ? null : { kind: "index", index: token }}
              options={context.options}
              optionsKey={context.key}
              sae={`${info.source ?? "hub"}:${info.repo}/${info.path}@${info.revision ?? info.sha256}`}
              standardized={info.normalize === "layer_norm"}
            />
          )}
          <h3 className={s.panelTitle}>Where it fires most</h3>
          <p className={s.small}>
            It fires on {count(data.active_prompts)} of {count(data.n)} clean prompts. These are the strongest; descriptions of features live on
            sites like Neuronpedia, which Logogram doesn't contact.
          </p>
          <ol className={f.top}>
            {data.top.map((r) => (
              <li key={r.index}>
                <button type="button" onClick={() => useStore.getState().setPromptIndex(r.index)} className={f.topRow}>
                  <span className={f.position}>{r.index}</span>
                  <span className={f.text}>{r.text}</span>
                  <span className={f.peak}>
                    {visibleToken(r.token)} <strong>{num(r.max, 2)}</strong>
                  </span>
                </button>
              </li>
            ))}
          </ol>
        </>
      )}
    </section>
  );
}
