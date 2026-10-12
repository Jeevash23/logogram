import type { RunDetail } from "../api/types";
import { divergingScale, niceBound, SCALE_FLOOR, textOn } from "./color";
import { describeMetric } from "./metrics";
import { baselineText, experimentText, measureWords, scopeText } from "./spec";

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** A copyable methods paragraph derived only from the executed spec and saved provenance, in
 * sentences a reader can follow: every choice that can change a number, said in words. */
export function methodsText(run: RunDetail): string {
  const { spec: s, summary, manifest } = run;
  const n = summary?.n_prompts;
  const revision = s.model.revision ? `revision ${s.model.revision}` : "an unpinned revision (the main branch when it ran)";
  const device = manifest?.device?.type ?? s.model.device;
  const weights = s.model.process_weights
    ? "with TransformerLens's weight processing (LayerNorm weights folded, writing weights and the unembedding centered), which leaves logit differences unchanged"
    : "with its original weights, unprocessed";
  const prompts = `${n === undefined ? "The" : `${n}`} prompt pair${n === 1 ? " was" : "s were"} read from ${s.dataset.path}${
    s.dataset.sha256 ? ` (SHA-256 ${s.dataset.sha256})` : " (not pinned by a checksum)"
  }${s.dataset.limit ? `, the first ${s.dataset.limit} of the file` : ""}, and tokenized ${
    s.tokenization.prepend_bos ? "with" : "without"
  } a beginning-of-sequence token.`;
  const normalized =
    s.metric.normalization === "dataset_gap"
      ? "Effects were normalized by the mean clean–corrupt gap over the dataset"
      : "Effects were normalized by each prompt's own clean–corrupt gap";
  const formula = summary?.metric.normalized_effect;
  const notes = s.notes?.trim();
  return [
    `${experimentText(s.experiment)}, at ${scopeText(s.scope)}.`,
    `The model was ${s.model.id} at ${revision}, run in ${s.model.dtype} on ${device}, ${weights}.`,
    prompts,
    `The metric, for each prompt, was ${describeMetric(s.metric)}. ${normalized}${formula ? `: ${formula}` : ""}.`,
    `Prompts ran in batches of ${s.execution.batch_size}. ${statisticsText(run)}`,
    methodText(run),
    // What was expected, as the spec that ran states it, so the result can be read against it.
    ...(notes ? [`Notes saved with the spec: ${notes}`] : []),
    `The run is ${run.id}.${
      manifest?.versions
        ? ` Software: ${Object.entries(manifest.versions).filter(([, v]) => v).map(([k, v]) => `${k} ${v}`).join(", ")}.`
        : ""
    }`,
    ...(summary?.warnings.map((warning) => `Note from the run: ${warning}`) ?? []),
  ].join("\n\n");
}

/** What the method did, beyond its name: the baseline, the paths, the steering or the estimate. */
function methodText(run: RunDetail): string {
  const { spec: s, summary } = run;
  const e = s.experiment;
  const arrow = (d: "clean_to_corrupt" | "corrupt_to_clean") =>
    d === "clean_to_corrupt" ? "from the clean prompt into the corrupt one" : "from the corrupt prompt into the clean one";
  switch (e.kind) {
    case "ablation":
      return `Each site's activation was replaced by a baseline: ${baselineText(e.baseline)}.`;
    case "activation_patching":
      return `Each site's activation was copied ${arrow(e.direction)} of each pair.`;
    case "direct_logit_attribution":
      return `Direct logit attribution of the ${e.prompts} prompts: each component's output at the last token, read through the final normalization with its scale held at its value in the run.${
        summary?.direct
          ? ` The mean logit difference was ${summary.direct.logit_diff?.toFixed(4)}: attention ${summary.direct.attention?.toFixed(4)}, MLPs ${summary.direct.mlp?.toFixed(4)}, embeddings ${summary.direct.embeddings?.toFixed(4)} and biases ${summary.direct.biases?.toFixed(4)}.`
          : ""
      }`;
    case "path_patching":
      return `Path patching ${arrow(e.direction)}, into ${e.receivers.map((r) => (r.kind === "logits" ? "the logits" : `L${r.layer} H${r.head} ${r.input}`)).join(", ")}: each sender was patched while every other attention head was held at its value in the receiver run${
        e.freeze_mlps ? ", and the MLPs were held too" : ", with the MLPs recomputed"
      }; the receivers' recorded inputs were then patched into an unchanged run.`;
    case "steering": {
      const other = e.apply_to === "clean" ? "corrupt" : "clean";
      return `Steering: a direction was added to the ${e.apply_to} prompts, the mean of (${other} − ${e.apply_to}) activations at each site over a training split of ${Math.round(
        e.train_fraction * 100,
      )}% of the pairs (split seed ${e.seed}), at strengths ${e.coefficients.join(", ")}. ${
        e.control ? "A random direction of the same length was run as a control." : "No random control was run."
      }${summary?.steering ? ` The directions came from dataset prompts ${summary.steering.train.join(", ")}; effects were measured on the ${summary.steering.test.length} held-out prompts.` : ""}`;
    }
    case "attribution_patching":
      return `Attribution patching ${arrow(e.direction)}: a first-order estimate of each site's patching effect, (source − receiver activation) · ${
        e.method === "integrated_gradients"
          ? `the gradient of the metric averaged over ${e.steps} runs whose input embeddings lie evenly between the receiver's and the source's (integrated gradients)`
          : "the gradient of the metric at the receiver run"
      }, not a patched forward pass.`;
  }
}

/** The bootstrap as run: over prompts or clusters of them, with the corrections for many sites. */
function statisticsText(run: RunDetail): string {
  const st = run.spec.statistics;
  const summary = run.summary?.statistics;
  const over = st.cluster
    ? `over clusters of prompts with the same ${st.cluster}${summary?.clusters ? ` (${summary.clusters} clusters)` : ""}`
    : "over prompts";
  const corrected = summary?.multiple_comparisons ? ` ${summary.multiple_comparisons}` : "";
  return `Intervals are ${st.ci * 100}% percentile bootstrap intervals ${over}, from ${st.bootstrap} resamples with seed ${st.seed}.${corrected}`;
}

/** Render an export with a fixed light background, full axis labels and a symmetric legend. */
export async function exportFigure(run: RunDetail, metric: "effect" | "delta", unit: boolean) {
  const summary = run.summary;
  if (!summary) throw new Error("Select a finished run to export a figure.");
  await document.fonts.ready;
  const { layout, sites } = summary;
  const cell = 44;
  const left = 210;
  const top = 190;
  const width = Math.max(960, left + layout.cols.length * cell + 32);
  const height = top + layout.rows.length * cell + 170;
  // Browser canvas limits vary. Keep a large sweep readable in the CSV without silently
  // clipping a figure or exhausting memory while allocating its backing store.
  if (width * height > 16_000_000 || width > 12000 || height > 12000) throw new Error("This sweep is too large for one figure. Export its per-prompt CSV or run a smaller scope.");
  const canvas = document.createElement("canvas");
  const resolution = width * height < 4_000_000 ? 2 : 1;
  canvas.width = width * resolution;
  canvas.height = height * resolution;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Your browser couldn't create the figure canvas.");
  ctx.scale(resolution, resolution);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#1c2226";
  const font = '"Instrument Sans Variable", "Instrument Sans", sans-serif';
  ctx.font = `600 23px ${font}`;
  ctx.fillText(run.spec.name, 28, 38, width - 56);
  ctx.font = `14px ${font}`;
  ctx.fillText(`${experimentText(run.spec.experiment)} · ${scopeText(run.spec.scope)}`, 28, 66, width - 56);
  ctx.fillText(`${run.spec.model.id} @ ${run.spec.model.revision ?? "unpinned"} · ${run.spec.model.dtype} · n = ${summary.n_prompts}`, 28, 88, width - 56);
  const words = measureWords(run.spec.experiment, summary.metric);
  ctx.fillText(`${layout.row_title} × ${layout.col_title} · ${metric === "effect" ? words.effect : words.delta} · mean over prompts`, 28, 110, width - 56);
  const bound = unit && metric === "effect" ? 1 : niceBound(sites.map((s) => s[metric].mean), SCALE_FLOOR[metric]);
  const color = divergingScale(bound, "light");
  ctx.font = `13px ${font}`;
  layout.rows.forEach((row, r) => { ctx.fillStyle = "#1c2226"; ctx.textAlign = "right"; ctx.fillText(row.label, left - 14, top + r * cell + 27, left - 28); });
  layout.cols.forEach((col, c) => {
    ctx.save(); ctx.translate(left + c * cell + 24, top - 10); ctx.rotate(-Math.PI / 4); ctx.textAlign = "left";
    ctx.fillText(col.label, 0, 0, 92); ctx.restore();
  });
  for (const site of sites) {
    const value = site[metric].mean;
    const fill = color(value);
    ctx.fillStyle = fill;
    const x = left + site.col * cell, y = top + site.row * cell;
    ctx.fillRect(x, y, cell - 2, cell - 2);
    ctx.fillStyle = textOn(fill);
    ctx.textAlign = "center";
    ctx.font = `12px ${font}`;
    ctx.fillText(value === null ? "∅" : value.toFixed(2), x + (cell - 2) / 2, y + 26, cell - 6);
  }
  const legendY = height - 122;
  for (let x = 0; x < 300; x++) { ctx.fillStyle = color(bound * (2 * x / 299 - 1)); ctx.fillRect(28 + x, legendY, 1, 14); }
  ctx.fillStyle = "#1c2226"; ctx.font = `13px ${font}`; ctx.textAlign = "left";
  ctx.fillText(String(-bound), 28, legendY + 34); ctx.fillText("0", 174, legendY + 34);
  ctx.textAlign = "right"; ctx.fillText(String(bound), 328, legendY + 34); ctx.textAlign = "left";
  ctx.fillText(`BOS ${run.spec.tokenization.prepend_bos ? "on" : "off"} · batch ${run.spec.execution.batch_size} · ${summary.statistics.ci * 100}% CI, ${summary.statistics.bootstrap} resamples, seed ${summary.statistics.seed}`, 28, height - 55, width - 56);
  ctx.fillText(`Run ${run.id} · ∅ undefined · colors clipped at ±${bound}; full values and intervals in the saved results`, 28, height - 30, width - 56);
  const blob = await new Promise<Blob>((resolve, reject) => canvas.toBlob((result) => result ? resolve(result) : reject(new Error("The figure could not be encoded.")), "image/png"));
  downloadBlob(blob, `${run.id}-${metric}.png`);
}
