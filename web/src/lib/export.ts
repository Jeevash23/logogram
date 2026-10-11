import type { RunDetail } from "../api/types";
import { divergingScale, niceBound, SCALE_FLOOR, textOn } from "./color";
import { describeMetric } from "./metrics";
import { experimentText, measureWords, scopeText } from "./spec";

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** A copyable methods paragraph derived only from the executed spec and saved provenance. */
export function methodsText(run: RunDetail): string {
  const { spec: s, summary, manifest } = run;
  return [
    `${experimentText(s.experiment)} at ${scopeText(s.scope)}.`,
    `Model: ${s.model.id}, revision ${s.model.revision ?? "unpinned"}, ${s.model.dtype}, device ${manifest?.device?.type ?? s.model.device}; weight processing ${s.model.process_weights ? "enabled" : "disabled"}.`,
    `Dataset: ${s.dataset.path}; SHA-256 ${s.dataset.sha256 ?? "unpinned"}; n = ${summary?.n_prompts ?? "unknown"}; limit ${s.dataset.limit ?? "none"}; prepend BOS ${s.tokenization.prepend_bos}.`,
    `Metric: ${describeMetric(s.metric)}; normalization ${s.metric.normalization}. ${summary?.metric.normalized_effect ?? ""}.`,
    `Execution batch size ${s.execution.batch_size}. ${statisticsText(run)}`,
    s.experiment.kind === "ablation"
      ? `Ablation baseline: ${JSON.stringify(s.experiment.baseline)}.`
      : s.experiment.kind === "direct_logit_attribution"
        ? `Direct logit attribution of the ${s.experiment.prompts} prompts: each component's output at the last token, read through the final normalization with its scale held at its value in the run.${summary?.direct ? ` Mean logit difference ${summary.direct.logit_diff?.toFixed(4)}: attention ${summary.direct.attention?.toFixed(4)}, MLPs ${summary.direct.mlp?.toFixed(4)}, embeddings ${summary.direct.embeddings?.toFixed(4)}, biases ${summary.direct.biases?.toFixed(4)}.` : ""}`
        : s.experiment.kind === "path_patching"
          ? `Path patching (direction ${s.experiment.direction}) into ${s.experiment.receivers.map((r) => (r.kind === "logits" ? "the logits" : `L${r.layer} H${r.head} ${r.input}`)).join(", ")}: each sender patched with every other attention head held at its receiver-run value${s.experiment.freeze_mlps ? " and the MLPs held too" : ", MLPs recomputed"}; the receivers' recorded inputs then patched into an unchanged run.`
          : s.experiment.kind === "steering"
          ? `Steering: a direction added to the ${s.experiment.apply_to} prompts, the mean of (${s.experiment.apply_to === "clean" ? "corrupt" : "clean"} − ${s.experiment.apply_to}) at each site over a training split of ${Math.round(s.experiment.train_fraction * 100)}% of the pairs (split seed ${s.experiment.seed}); strengths ${s.experiment.coefficients.join(", ")}; random control of the same length ${s.experiment.control ? "included" : "not run"}.${summary?.steering ? ` Directions from dataset prompts ${summary.steering.train.join(", ")}; measured on the ${summary.steering.test.length} held-out prompts.` : ""}`
          : s.experiment.kind === "attribution_patching"
          ? `Attribution patching (direction ${s.experiment.direction}): a first-order estimate of each site's patching effect, (source − receiver activation) · ${
              s.experiment.method === "integrated_gradients"
                ? `the gradient of the metric averaged over ${s.experiment.steps} runs whose input embeddings lie evenly between the receiver's and the source's (integrated gradients)`
                : "the gradient of the metric at the receiver run"
            }, not a patched forward pass.`
          : `Direction: ${s.experiment.direction}.`,
    `Run: ${run.id}. ${manifest?.versions ? `Software: ${Object.entries(manifest.versions).filter(([, v]) => v).map(([k, v]) => `${k} ${v}`).join(", ")}.` : ""}`,
    ...(summary?.warnings.map((warning) => `Run note: ${warning}`) ?? []),
  ].join("\n\n");
}

/** The bootstrap as run: over prompts or clusters of them, with the corrections for many sites. */
function statisticsText(run: RunDetail): string {
  const st = run.spec.statistics;
  const summary = run.summary?.statistics;
  const over = st.cluster
    ? `over clusters of prompts with the same ${st.cluster}${summary?.clusters ? ` (${summary.clusters} clusters)` : ""}`
    : "over prompts";
  const corrected = summary?.multiple_comparisons ? ` ${summary.multiple_comparisons}` : "";
  return `Percentile bootstrap ${over}: ${st.bootstrap} resamples, ${st.ci * 100}% confidence interval, seed ${st.seed}.${corrected}`;
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
  ctx.fillStyle = "#f4f6f6";
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
    ctx.fillStyle = textOn(fill, "light");
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
