// Spec builders and plain-language descriptions (mirroring src/logogram/spec.py).

import type {
  BaselineSpec,
  ChatTemplateSpec,
  EdgeSenderSpec,
  EdgeSpec,
  ExperimentKind,
  PathReceiverSpec,
  ExperimentSpec,
  Measure,
  MetricSpec,
  ModelInfo,
  ProbeLabels,
  ModelRef,
  PositionSpec,
  SAEInfo,
  ScopeSpec,
  Side,
  SiteSpec,
  Spec,
  StreamKind,
} from "../api/types";
import { count, plural } from "./format";
import { metricWords } from "./metrics";
import { KIND_SHORT, type Selection } from "./sites";

export function positionText(p: PositionSpec): string {
  switch (p.kind) {
    case "all":
      return "all positions";
    case "last":
      return "the last token";
    case "index":
      return `token index ${p.index}`;
    case "label":
      return `position ${p.label}`;
  }
}

export function baselineText(b: BaselineSpec): string {
  switch (b.kind) {
    case "zero":
      return "zero";
    case "mean":
      return `mean over ${b.reference} prompts`;
    case "resample":
      return `resample from ${b.donors} ${b.pool} donor${b.donors === 1 ? "" : "s"}, seed ${b.seed}`;
    case "scale":
      return `scale by ${String(b.factor).replace("-", "−")}`;
    case "projection":
      return b.reference === null ? `remove the direction ${b.direction.name}` : `set the direction ${b.direction.name} to its ${b.reference} mean`;
  }
}

/** What a probe reads, in words (src/logogram/spec.py labels_text). */
export function labelsText(labels: ProbeLabels): string {
  if (labels.kind === "side") return "which prompt it is (clean or corrupt)";
  const prompts = { clean: "clean prompts", corrupt: "corrupt prompts", both: "both prompts" }[labels.prompts];
  return `meta.${labels.field} (${prompts})`;
}

/** Whether a model's heads share keys and values (grouped-query attention): it has fewer
 * key/value heads than query heads, as a run's summary or the loaded model says. */
export function sharesKeys(model: { n_heads: number; n_key_value_heads?: number | null; extra?: { n_key_value_heads?: number } } | null | undefined): boolean {
  if (!model) return false;
  const kv = model.n_key_value_heads ?? model.extra?.n_key_value_heads ?? model.n_heads;
  return kv < model.n_heads;
}

/** A head receiver's head as the results write it: "H9", or "KV1" for a key or value on a model
 * whose heads share keys and values (`grouped`: grouped-query attention), where it names the
 * key/value head. */
function receiverHead(r: Extract<PathReceiverSpec, { kind: "head" }>, grouped: boolean): string {
  return grouped && r.input !== "q" ? `KV${r.head}` : `H${r.head}`;
}

/** A path's receiver as the results write it, for example "L9 H9 q", "L3 MLP", "logits", or
 * "L9 KV1 k" for a key/value head's key with grouped-query attention (src/logogram/spec.py
 * receiver_text). */
export function receiverLabel(r: PathReceiverSpec, grouped = false): string {
  if (r.kind === "logits") return "logits";
  if (r.kind === "mlp") return `L${r.layer} MLP`;
  return `L${r.layer} ${receiverHead(r, grouped)} ${r.input}`;
}

/** A path's receiver in words, for example "L9 H9 query", "L9 KV1 key" or "L3 MLP input". */
export function receiverWords(r: PathReceiverSpec, grouped = false): string {
  if (r.kind === "logits") return "Logits, read directly";
  if (r.kind === "mlp") return `L${r.layer} MLP input`;
  return `L${r.layer} ${receiverHead(r, grouped)} ${{ q: "query", k: "key", v: "value" }[r.input]}`;
}

/** An edge's sender as the results write it, for example "L5 H5", "L0 MLP" or "embeddings". */
export function senderLabel(s: EdgeSenderSpec): string {
  if (s.kind === "embed") return "embeddings";
  if (s.kind === "mlp") return `L${s.layer} MLP`;
  return `L${s.layer} H${s.head}`;
}

/** What an edge's sender sends, in words, for example "L5 H5's output". */
export function senderWords(s: EdgeSenderSpec): string {
  return s.kind === "embed" ? "the embeddings" : `${senderLabel(s)}'s output`;
}

/** What an edge's receiver reads, in words, for example "L9 H6's query input". */
export function receiverInputWords(r: PathReceiverSpec, grouped = false): string {
  if (r.kind === "logits") return "the logits";
  if (r.kind === "mlp") return `L${r.layer} MLP's input`;
  return `L${r.layer} ${receiverHead(r, grouped)}'s ${{ q: "query", k: "key", v: "value" }[r.input]} input`;
}

/** An edge as the results write it, for example "L5 H5 → L9 H6 q", or "L5 H5 → L9 KV1 k" into a
 * key/value head's key with grouped-query attention (src/logogram/spec.py edge_text). */
export function edgeLabel(e: EdgeSpec, grouped = false): string {
  return `${senderLabel(e.sender)} → ${receiverLabel(e.receiver, grouped)}`;
}

/** The graph a graph scope covers, in words (as src/logogram/spec.py graph_text says it). */
export function graphText(scope: { layers: number[] | null; logits: boolean }): string {
  if (scope.layers === null) return scope.logits ? "every edge of the graph" : "every edge but those into the logits";
  const runs: number[][] = [];
  for (const layer of scope.layers) {
    const last = runs[runs.length - 1];
    if (last && layer === last[last.length - 1] + 1) last.push(layer);
    else runs.push([layer]);
  }
  const text = runs.map((r) => (r.length === 1 ? `${r[0]}` : `${r[0]}–${r[r.length - 1]}`)).join(", ");
  return `the edges into layer${scope.layers.length === 1 ? "" : "s"} ${text}${scope.logits ? " and the logits" : ""}`;
}

/** A steering strength as the results write it, for example ×2 or ×−0.5. */
export function strengthText(c: number): string {
  return `×${String(c).replace("-", "−")}`;
}

/** Whether a model loaded on `loaded` (cpu, mps, or cuda:N, one GPU) is on the device a spec asks
 * for. "auto" takes what is there; "cuda" is the default GPU, which the server knows: any GPU is
 * taken for it here, and the server checks it exactly. */
export function sameDevice(asked: string, loaded: string): boolean {
  if (asked === "auto" || asked === loaded) return true;
  return asked === "cuda" && (loaded === "cuda" || loaded.startsWith("cuda:"));
}

/** How a chat template wraps prompts, in a few words. */
export function chatTemplateText(t: ChatTemplateSpec | null | undefined): string {
  if (!t) return "plain text";
  return t.kind === "user_turn" ? "in the chat template, as the person's message" : "in the chat template, as the start of the reply";
}

/** The spec's model as the loaded one: where its weights came from, and the device asked for. */
export function modelRefOf(info: ModelInfo, device: ModelRef["device"] = info.device): ModelRef {
  return {
    id: info.id,
    source: info.source ?? "hub",
    revision: info.revision,
    sha256: info.sha256 ?? null,
    dtype: info.dtype,
    device,
    process_weights: info.process_weights,
  };
}

export function experimentText(e: ExperimentSpec): string {
  if (e.kind === "activation_patching") {
    return e.direction === "clean_to_corrupt" ? "Patch clean → corrupt" : "Patch corrupt → clean";
  }
  if (e.kind === "direct_logit_attribution") return `Direct logit attribution (${e.prompts} prompts)`;
  if (e.kind === "attribution_patching") {
    const arrow = e.direction === "clean_to_corrupt" ? "clean → corrupt" : "corrupt → clean";
    return e.method === "integrated_gradients"
      ? `Estimate patching ${arrow} with integrated gradients (${e.steps} steps)`
      : `Estimate patching ${arrow}`;
  }
  if (e.kind === "probe") {
    return `Linear probe for ${labelsText(e.labels)} (${Math.round(e.train_fraction * 100)}% of pairs train, seed ${e.seed}${e.control ? ", against shuffled labels" : ""})`;
  }
  if (e.kind === "das") {
    const arrow = e.direction === "clean_to_corrupt" ? "clean → corrupt" : "corrupt → clean";
    return `Learn a subspace of ${plural(e.dimensions, "dimension")} and patch ${arrow} within it (${e.steps} steps${e.control ? ", random control" : ""})`;
  }
  if (e.kind === "subspace_patching") {
    const arrow = e.direction === "clean_to_corrupt" ? "clean → corrupt" : "corrupt → clean";
    return `Patch ${arrow} along the direction ${e.subspace.name}`;
  }
  if (e.kind === "steering") {
    const strengths = `${e.coefficients.map(strengthText).join(", ")}${e.control ? ", random control" : ""}`;
    if (e.direction.kind === "sae_feature") return `Steer ${e.apply_to} prompts along SAE feature ${e.direction.feature} (${strengths})`;
    if (e.direction.kind === "direction") return `Steer ${e.apply_to} prompts along the direction ${e.direction.direction.name} (${strengths})`;
    const toward = e.apply_to === "clean" ? "corrupt" : "clean";
    return `Steer ${e.apply_to} prompts toward ${toward} (${strengths})`;
  }
  if (e.kind === "path_patching") {
    const arrow = e.direction === "clean_to_corrupt" ? "clean → corrupt" : "corrupt → clean";
    const names = e.receivers.slice(0, 4).map((r) => receiverLabel(r)).join(", ") + (e.receivers.length > 4 ? ` and ${e.receivers.length - 4} more` : "");
    return `Path patching ${arrow} into ${names} (${e.freeze_mlps ? "heads and MLPs" : "other heads"} held)`;
  }
  if (e.kind === "edge_attribution") {
    const arrow = e.direction === "clean_to_corrupt" ? "clean → corrupt" : "corrupt → clean";
    const how = e.method === "integrated_gradients" ? ` with integrated gradients (${e.steps} steps)` : "";
    return `Estimate each edge ${arrow}${how}${e.curve ? ", then patch the strongest for a faithfulness curve" : ""}`;
  }
  if (e.kind === "acdc") {
    const arrow = e.direction === "clean_to_corrupt" ? "clean → corrupt" : "corrupt → clean";
    return `ACDC, threshold ${e.threshold} on changes ${e.rule === "toward_source" ? "toward the source" : "either way"} (${arrow})`;
  }
  switch (e.baseline.kind) {
    case "zero":
      return "Zero-ablate";
    case "mean":
      return `Mean-ablate (${e.baseline.reference} mean)`;
    case "resample":
      return `Resample-ablate (${e.baseline.donors} ${e.baseline.pool} donors, seed ${e.baseline.seed})`;
    case "scale":
      return `Scale by ${String(e.baseline.factor).replace("-", "−")}`;
    case "projection":
      return `Ablate the direction ${e.baseline.direction.name} (${e.baseline.reference === null ? "removed" : `set to its ${e.baseline.reference} mean`})`;
  }
}

export function experimentShort(e: ExperimentSpec): string {
  if (e.kind === "activation_patching") {
    return e.direction === "clean_to_corrupt" ? "Patch clean→corrupt" : "Patch corrupt→clean";
  }
  if (e.kind === "direct_logit_attribution") return `Direct attribution, ${e.prompts}`;
  if (e.kind === "attribution_patching") {
    const arrow = e.direction === "clean_to_corrupt" ? "clean→corrupt" : "corrupt→clean";
    return e.method === "integrated_gradients" ? `Estimated ${arrow}, integrated gradients` : `Estimated ${arrow}`;
  }
  if (e.kind === "probe") return e.labels.kind === "side" ? "Probe for the prompt" : `Probe for ${e.labels.field}`;
  if (e.kind === "das") return `Learned subspace, ${e.dimensions}d`;
  if (e.kind === "subspace_patching") return `Patch along ${e.subspace.name}`;
  if (e.kind === "steering") {
    if (e.direction?.kind === "sae_feature") return `Steer ${e.apply_to} along F${e.direction.feature}`;
    if (e.direction?.kind === "direction") return `Steer ${e.apply_to} along ${e.direction.direction.name}`;
    return `Steer ${e.apply_to}→${e.apply_to === "clean" ? "corrupt" : "clean"}`;
  }
  if (e.kind === "path_patching") return `Paths ${e.direction === "clean_to_corrupt" ? "clean→corrupt" : "corrupt→clean"}`;
  if (e.kind === "edge_attribution") {
    const arrow = e.direction === "clean_to_corrupt" ? "clean→corrupt" : "corrupt→clean";
    return e.method === "integrated_gradients" ? `Edges ${arrow}, integrated gradients` : `Edges ${arrow}`;
  }
  if (e.kind === "acdc") return `ACDC ${e.direction === "clean_to_corrupt" ? "clean→corrupt" : "corrupt→clean"}, threshold ${e.threshold}`;
  return { zero: "Zero ablation", mean: "Mean ablation", resample: "Resample ablation", scale: "Scaled", projection: "Directional ablation" }[e.baseline.kind];
}

/** A readout metric's unit, in words (src/logogram/spec.py readout_text). */
export function readoutText(metric: Extract<MetricSpec, { kind: "readout" }>): string {
  const u = metric.unit;
  const what =
    u.kind === "neuron"
      ? `neuron ${u.neuron} of L${u.layer}'s MLP`
      : u.kind === "direction"
        ? `L${u.layer} ${KIND_SHORT[u.site]} along the direction ${u.direction.name}`
        : `SAE feature ${u.feature}`;
  return `${what} at ${positionText(metric.position)}`;
}

/** What a method's per-prompt values are (see Summary.measure). */
export function measureOf(e: ExperimentSpec | null | undefined): Measure {
  if (e?.kind === "probe") return "probe";
  if (e?.kind === "direct_logit_attribution") return "attribution";
  if (e?.kind === "attribution_patching" || e?.kind === "edge_attribution") return "estimate";
  return "intervention";
}

/** The names of a run's two values, and what they mean, for its method and metric. Direct logit
 * attribution always splits the logit difference. */
export function measureWords(e: ExperimentSpec | null | undefined, metric?: { kind?: string | null; target?: Side | null } | null) {
  if (measureOf(e) === "attribution") {
    return {
      effect: "Share of logit diff",
      delta: "Direct effect",
      mean: "mean share of the logit difference",
      effectLegend:
        "Share of the mean logit difference this component writes directly. Negative values push toward the distractor.",
      deltaLegend: "What the component writes directly into the logit difference (answer − distractor), in logits.",
    };
  }
  if (e?.kind === "probe") {
    return {
      effect: "Held-out accuracy",
      delta: "Held-out accuracy",
      mean: "held-out accuracy",
      effectLegend: "The share of held-out examples whose label the site's probe reads correctly. Compare it with chance and with the control probe, trained on shuffled labels.",
      deltaLegend: "The share of held-out examples whose label the site's probe reads correctly.",
    };
  }
  const m = metricWords(metric);
  const delta = `Δ ${m.short}`;
  if (e?.kind === "das") {
    return {
      effect: "Normalized effect",
      delta,
      mean: "mean normalized effect",
      effectLegend: `Normalized effect on held-out pairs of patching within the learned subspace, the whole site, or a random subspace: 1 means it ${e.direction === "clean_to_corrupt" ? "restores the clean behavior" : "shifts the output as far as the corrupt prompt does"}.`,
      deltaLegend: `Change in the ${m.label} (${m.formula}) caused by the patch.`,
    };
  }
  if (e?.kind === "steering") {
    return {
      effect: "Normalized effect",
      delta,
      mean: "mean normalized effect",
      effectLegend: `Normalized effect: 1 means steering moved the ${e.apply_to} prompts as far as switching to the ${e.apply_to === "clean" ? "corrupt" : "clean"} prompt; 0 means no change.`,
      deltaLegend: `Change in the ${m.label} (${m.formula}) caused by adding the direction.`,
    };
  }
  if (e?.kind === "path_patching") {
    return {
      effect: "Normalized effect",
      delta,
      mean: "mean normalized effect",
      effectLegend: `Normalized effect through the receivers only: 1 means the path alone ${e.direction === "clean_to_corrupt" ? "restores the clean behavior" : "shifts the output as far as the corrupt prompt does"}; 0 means nothing travels along it.`,
      deltaLegend: `Change in the ${m.label} (${m.formula}) when only the receivers' inputs are patched.`,
    };
  }
  if (e?.kind === "acdc") {
    const source = e.direction === "clean_to_corrupt" ? "clean" : "corrupt";
    return {
      effect: "Change when removed",
      delta: `${delta} when removed`,
      mean: "mean normalized change",
      effectLegend: `Normalized change in the metric when ACDC removed the edge from the circuit as it stood then: 1 means removing it moved the metric as far as switching to the ${source} prompt; 0 means no change.`,
      deltaLegend: `Change in the ${m.label} (${m.formula}) when ACDC removed the edge from the circuit as it stood then.`,
    };
  }
  const restores =
    (e?.kind === "activation_patching" || e?.kind === "attribution_patching" || e?.kind === "edge_attribution" || e?.kind === "subspace_patching") &&
    e.direction === "clean_to_corrupt";
  if (e?.kind === "edge_attribution") {
    return {
      effect: "Estimated effect",
      delta: `Estimated Δ ${m.short}`,
      mean: "mean estimated effect",
      effectLegend: `Estimated normalized effect of patching the edge alone, to first order: 1 would mean the edge alone ${restores ? "restores the clean behavior" : "shifts the output as far as the corrupt prompt does"}. Verify the strongest by path patching.`,
      deltaLegend: `First-order estimate of the change in the ${m.label} (${m.formula}) patching the edge alone would cause.`,
    };
  }
  if (measureOf(e) === "estimate") {
    return {
      effect: "Estimated effect",
      delta: `Estimated Δ ${m.short}`,
      mean: "mean estimated effect",
      effectLegend: `Estimated normalized effect, to first order: 1 would mean the site alone ${restores ? "restores the clean behavior" : "shifts the output as far as the corrupt prompt does"}. Verify the strongest by patching.`,
      deltaLegend: `First-order estimate of the change in the ${m.label} (${m.formula}) patching would cause.`,
    };
  }
  return {
    effect: "Normalized effect",
    delta,
    mean: "mean normalized effect",
    effectLegend: `Normalized effect: 1 means the site alone ${restores ? "restores the clean behavior" : "shifts the output as far as the corrupt prompt does"}; 0 means no change.`,
    deltaLegend: `Change in the ${m.label} (${m.formula}) caused by the intervention.`,
  };
}

export function scopeText(s: ScopeSpec): string {
  switch (s.kind) {
    case "heads":
      return `every head, ${positionText(s.position)}`;
    case "layer_position":
      return `${s.site} at every layer and ${s.positions === "each" ? "every position" : "each labelled position"}`;
    case "layer_components":
      return `${s.components.join(" and ")} per layer, ${positionText(s.position)}`;
    case "sites":
      return `${s.sites.length} chosen site${s.sites.length === 1 ? "" : "s"}`;
    case "features":
      return `every SAE feature, ${positionText(s.position)}, keeping the top ${s.top}${
        s.choose_on !== null ? ` chosen on ${Math.round(s.choose_on * 100)}% of the prompts (seed ${s.seed}) and reported on the rest` : ""
      }`;
    case "site_sets":
      return `${s.sets.length} set${s.sets.length === 1 ? "" : "s"} of sites, each at once`;
    case "head_inputs":
      return `${s.input === "q" ? "every head's query" : s.input === "k" ? "every key" : "every value"}, ${positionText(s.position)}`;
    case "neurons":
      return `every MLP neuron of ${layersText(s.layers)}, ${positionText(s.position)}`;
    case "graph":
      return graphText(s);
    case "edges":
      return `${s.edges.length} edge${s.edges.length === 1 ? "" : "s"}, one at a time`;
  }
}

/** "layer 3", "layers 3, 4 and 9". */
export function layersText(layers: number[]): string {
  if (layers.length === 1) return `layer ${layers[0]}`;
  return `layers ${layers.slice(0, -1).join(", ")} and ${layers[layers.length - 1]}`;
}

/** Layers typed as a list: "3, 4", "3 4", ranges like "0-5". Null when they can't be read. */
export function parseLayers(text: string): number[] | null {
  const parts = text.replace(/[–—]/g, "-").split(/[\s,;]+/).filter(Boolean);
  if (parts.length === 0) return null;
  const out: number[] = [];
  for (const part of parts) {
    const range = part.match(/^(\d+)-(\d+)$/);
    if (range) {
      const [a, b] = [Number(range[1]), Number(range[2])];
      if (b < a || b - a > 1024) return null;
      for (let layer = a; layer <= b; layer++) out.push(layer);
    } else if (/^\d+$/.test(part)) out.push(Number(part));
    else return null;
  }
  return out;
}

export function scopeShort(s: ScopeSpec): string {
  switch (s.kind) {
    case "heads":
      return "Layer × head";
    case "layer_position":
      return "Layer × position";
    case "layer_components":
      if (s.components.length === 1) return `${s.components[0].replace("_", " ")} per layer`;
      return s.components.length === 2 && s.components.includes("attn_out") && s.components.includes("mlp_out") ? "Attn and MLP" : "Components per layer";
    case "sites":
      return s.sites.length === 1 ? "Single site" : "Chosen sites";
    case "features":
      return "SAE features";
    case "site_sets":
      return "Sets of sites";
    case "head_inputs":
      return s.input === "q" ? "Heads' queries" : s.input === "k" ? "Keys" : "Values";
    case "neurons":
      return s.layers.length === 1 ? `Neurons of layer ${s.layers[0]}` : "Neurons";
    case "graph":
      return s.layers === null ? "The whole graph" : "Edges into chosen layers";
    case "edges":
      return s.edges.length === 1 ? "One edge" : "Chosen edges";
  }
}

export function siteFromSelection(sel: Selection, residKind: SiteSpec["kind"] = sel.kind ?? "resid_pre"): SiteSpec {
  const kind: SiteSpec["kind"] =
    sel.part === "head"
      ? "head"
      : sel.part === "attn"
        ? "attn_out"
        : sel.part === "mlp"
          ? "mlp_out"
          : sel.part === "feature"
            ? "sae_feature"
            : sel.part === "input"
              ? (sel.kind === "k" || sel.kind === "v" ? sel.kind : "q")
              : sel.part === "neuron"
                ? "neuron"
                : residKind;
  const position: PositionSpec =
    sel.positionKey === undefined
      ? { kind: "all" }
      : sel.positionKey === "last"
        ? { kind: "last" }
        : /^-?\d+$/.test(sel.positionKey)
          ? { kind: "index", index: Number(sel.positionKey) }
          : { kind: "label", label: sel.positionKey };
  return {
    kind,
    layer: sel.layer,
    head: sel.part === "head" || sel.part === "input" ? sel.head : null,
    ...(sel.part === "feature" ? { feature: sel.feature } : {}),
    ...(sel.part === "neuron" ? { neuron: sel.neuron } : {}),
    position,
  };
}

export function suggestName(spec: Pick<Spec, "experiment" | "scope">): string {
  const what = experimentShort(spec.experiment);
  const where = spec.scope.kind === "sites" && spec.scope.sites.length === 1
    ? siteText(spec.scope.sites[0])
    : spec.scope.kind === "features"
      ? "SAE features"
      : scopeShort(spec.scope).toLowerCase();
  return `${what} · ${where}`;
}

/** A site as the results write it (src/logogram/sites.py, site_label): a key or value names its
 * key/value head ("L9 KV2 k"), a query its query head ("L9 H6 q"). */
export function siteText(site: SiteSpec): string {
  const pos = site.position.kind === "all" ? "" : ` @ ${positionKey(site.position)}`;
  switch (site.kind) {
    case "head":
      return `L${site.layer} H${site.head}${pos}`;
    case "sae_feature":
      return `L${site.layer} F${site.feature}${pos}`;
    case "q":
      return `L${site.layer} H${site.head} q${pos}`;
    case "k":
    case "v":
      return `L${site.layer} KV${site.head} ${site.kind}${pos}`;
    case "neuron":
      return `L${site.layer} N${site.neuron}${pos}`;
    default: {
      const name = { resid_pre: "resid pre", resid_mid: "resid mid", resid_post: "resid post", attn_out: "attn", mlp_out: "mlp" }[site.kind];
      return `L${site.layer} ${name}${pos}`;
    }
  }
}

export function positionKey(p: PositionSpec): string {
  switch (p.kind) {
    case "all":
      return "all";
    case "last":
      return "last";
    case "index":
      return String(p.index);
    case "label":
      return p.label;
  }
}

/** How many of n prompts a seeded split gives its first part, as the server rounds it (Python's
 * round: halves go to the even neighbor). */
export function splitCount(n: number, share: number): number {
  const x = n * share;
  const floor = Math.floor(x);
  if (Math.abs(x - floor - 0.5) < 1e-9) return floor % 2 === 0 ? floor : floor + 1;
  return Math.round(x);
}

/** Rows the sweep will run: sites × prompts × donors. A decomposition runs each prompt once, an
 * estimate once per prompt (once per step of integrated gradients). Keys and values have
 * `nKvHeads` heads a layer (fewer than `nHeads` with grouped-query attention), an MLP `dMlp`
 * neurons. */
export function workload(
  spec: Spec,
  n: number,
  nLayers: number,
  nHeads: number,
  nPositions: number | null,
  nLabels: number,
  nKvHeads: number = nHeads,
  dMlp: number = 0,
): number | null {
  if (spec.experiment.kind === "direct_logit_attribution") return n;
  if (spec.experiment.kind === "attribution_patching" || spec.experiment.kind === "edge_attribution") {
    return spec.experiment.method === "integrated_gradients" ? n * (spec.experiment.steps ?? 1) : n;
  }
  const s = spec.scope;
  if (s.kind === "graph") return null; // ACDC's passes are the edge plan's to count
  if (s.kind === "edges") return s.edges.length * n;
  if (spec.experiment.kind === "steering") {
    const e = spec.experiment;
    const sites = s.kind === "layer_components" ? nLayers : s.kind === "sites" ? s.sites.length : 0;
    const test = n - splitCount(n, e.train_fraction);
    return sites * e.coefficients.length * (e.control ? 2 : 1) * Math.max(0, test);
  }
  let sites = 0;
  if (s.kind === "heads") sites = nLayers * nHeads;
  else if (s.kind === "head_inputs") sites = nLayers * (s.input === "q" ? nHeads : nKvHeads);
  else if (s.kind === "neurons") sites = s.layers.length * dMlp;
  else if (s.kind === "layer_components") sites = nLayers * s.components.length;
  else if (s.kind === "sites") sites = s.sites.length;
  else if (s.kind === "site_sets") sites = s.sets.length;
  else if (s.kind === "features") return n; // only attribution patching sweeps every feature
  else if (s.positions === "each") {
    if (nPositions === null) return null;
    sites = nLayers * nPositions;
  } else sites = nLayers * nLabels;
  const donors = spec.experiment.kind === "ablation" && spec.experiment.baseline.kind === "resample" ? spec.experiment.baseline.donors : 1;
  return sites * n * donors;
}

/** What running the spec costs, in words. */
export function workloadText(spec: Spec, rows: number): string {
  const e = spec.experiment;
  if (e.kind === "direct_logit_attribution") return `That is one forward and one backward pass for each of the ${count(rows)} prompts.`;
  if (e.kind === "attribution_patching") {
    if (e.method === "integrated_gradients") {
      const steps = e.steps ?? 1;
      return `That is ${count(rows)} forward and backward passes, ${steps} for each of the ${count(Math.round(rows / steps))} prompts, for every site at once.`;
    }
    return `That is two forward passes and one backward pass for each of the ${count(rows)} prompts, for every site at once.`;
  }
  if (e.kind === "edge_attribution") {
    const steps = e.method === "integrated_gradients" ? e.steps ?? 1 : 1;
    const n = Math.round(rows / steps);
    const estimate =
      steps > 1
        ? `That is two forward passes, then ${count(steps)} forward and backward passes, for each of the ${count(n)} prompts, for every edge at once.`
        : `That is three forward passes and one backward pass for each of the ${count(n)} prompts, for every edge at once.`;
    if (!e.curve) return estimate;
    return `${estimate} The faithfulness curve then patches the strongest edges for real, beside edges chosen at random.`;
  }
  if (e.kind === "steering") {
    const steered = `That is ${count(rows)} steered forward passes on the held-out prompts.`;
    if (!e.continuations) return steered;
    const { prompts, tokens } = e.continuations;
    return `${steered} Continuing the first ${plural(prompts, "held-out prompt")} by ${plural(tokens, "token")}, without steering and at each strength, takes a forward pass per token.`;
  }
  if (e.kind === "path_patching" && spec.scope.kind === "edges") {
    return `That is ${count(rows)} paths, each edge on each prompt alone, four forward passes each.`;
  }
  if (e.kind === "path_patching") return `That is ${count(rows)} paths, three forward passes each (fewer when senders come after the receivers).`;
  if (spec.scope.kind === "site_sets") return `That is ${count(rows)} patched forward passes, every site of a set at once.`;
  return `That is ${count(rows)} patched forward passes.`;
}

/** Where steering along a feature of this SAE adds its decoder direction, as a scope names it: the
 * site the SAE writes. An SAE on the heads' outputs (z) writes them before the output projection,
 * which turns them into the layer's attention output, so its scope names that. */
export function steerSiteOf(info: Pick<SAEInfo, "site" | "layer">): { kind: StreamKind; layer: number } {
  return { kind: info.site === "head" ? "attn_out" : info.site, layer: info.layer };
}

/** Where an SAE's output is read into the model, in words: where steering adds one of its
 * features (as src/logogram/spec.py describe_sae_site says it). */
export function saeWritesText(info: Pick<SAEInfo, "site" | "layer" | "transcoder">): string {
  const layer = info.layer;
  if (info.site === "head") return `the outputs (z) of layer ${layer}'s attention heads, side by side, before the output projection`;
  if (info.transcoder) return `layer ${layer}'s MLP output, which the transcoder writes`;
  switch (info.site) {
    case "resid_pre":
      return `the residual stream entering layer ${layer}`;
    case "resid_post":
      return `the residual stream leaving layer ${layer}`;
    case "attn_out":
      return `layer ${layer}'s attention output`;
    case "mlp_out":
      return `layer ${layer}'s MLP output`;
    default:
      return `the ${info.site} of layer ${layer}`;
  }
}

/** Fit a sweep to a method. Direct attribution reads what heads, attention and MLP outputs write,
 * at the last token where the logit difference is measured. Steering along an SAE feature
 * (`steerSite`, where its SAE writes) adds it at that one site. */
export function scopeFor(kind: ExperimentKind, scope: ScopeSpec, steerSite: { kind: StreamKind; layer: number } | null = null): ScopeSpec {
  const last: PositionSpec = { kind: "last" };
  // Edge attribution and ACDC measure the edges of the model's graph, and nothing else does.
  if (kind === "edge_attribution" || kind === "acdc") return scope.kind === "graph" ? scope : { kind: "graph", layers: null, logits: true };
  if (scope.kind === "graph") return scopeFor(kind, { kind: "heads", position: { kind: "all" } }, steerSite);
  // Edges one at a time are path patching's (a verification of edge attribution).
  if (scope.kind === "edges" && kind !== "path_patching") return scopeFor(kind, { kind: "heads", position: { kind: "all" } }, steerSite);
  // Only attribution patching sweeps every SAE feature; other methods patch chosen ones.
  if (scope.kind === "features" && kind !== "attribution_patching") return { kind: "heads", position: { kind: "all" } };
  // Queries, keys, values and neurons are patched, ablated or estimated one by one; steering adds
  // to the residual stream and paths start at components, so those methods sweep heads instead.
  // Direct effects: a neuron's at the last token, a value's at any position (through the heads
  // that read it); queries and keys have none.
  if (scope.kind === "head_inputs" || scope.kind === "neurons") {
    if (kind === "steering" || kind === "path_patching") return scopeFor(kind, { kind: "heads", position: { kind: "all" } }, steerSite);
    if (kind === "direct_logit_attribution") {
      if (scope.kind === "neurons") return { ...scope, position: last };
      return scope.input === "v" ? scope : { kind: "head_inputs", input: "v", position: scope.position };
    }
    return scope;
  }
  // Sets are intervened on by patching or ablation; other methods measure their sites one by one.
  if (scope.kind === "site_sets" && kind !== "activation_patching" && kind !== "ablation") {
    const sites = setMembers(scope.sets);
    return scopeFor(kind, sites.length ? { kind: "sites", sites } : { kind: "heads", position: { kind: "all" } });
  }
  if (kind === "steering") {
    // Steering adds to one residual stream site per layer, at one token.
    const single = (p: PositionSpec) => (p.kind === "all" ? last : p);
    if (steerSite) {
      // Along an SAE feature: where the SAE writes, at the tokens already chosen there.
      const here = scope.kind === "sites" ? scope.sites.filter((x) => x.kind === steerSite.kind && x.layer === steerSite.layer) : [];
      const positions = here.length ? here.map((x) => single(x.position)) : "position" in scope ? [single(scope.position)] : [last];
      return { kind: "sites", sites: positions.map((position) => ({ kind: steerSite.kind, layer: steerSite.layer, head: null, position })) };
    }
    if (scope.kind === "layer_components") {
      const resid = scope.components.find((c) => c === "resid_pre" || c === "resid_mid" || c === "resid_post") ?? "resid_pre";
      return { kind: "layer_components", components: [resid], position: single(scope.position) };
    }
    if (scope.kind === "sites") {
      const sites = scope.sites
        .filter((x) => x.kind === "resid_pre" || x.kind === "resid_mid" || x.kind === "resid_post")
        .map((x) => ({ ...x, position: single(x.position) }));
      if (sites.length) return { kind: "sites", sites };
    }
    return { kind: "layer_components", components: ["resid_pre"], position: last };
  }
  if (kind === "probe" || kind === "das" || kind === "subspace_patching") {
    // A probe and a learned subspace read one token of each prompt; a direction lives in the
    // stream, attention outputs and MLP outputs (a probe also reads heads' outputs).
    const streams = (c: string) => c === "resid_pre" || c === "resid_mid" || c === "resid_post" || c === "attn_out" || c === "mlp_out";
    const one = kind !== "subspace_patching";
    const single = (p: PositionSpec) => (one && p.kind === "all" ? last : p);
    if (scope.kind === "layer_position") return scope;
    if (scope.kind === "layer_components") {
      const components = scope.components.filter(streams);
      return { kind: "layer_components", components: components.length ? components : ["resid_post"], position: single(scope.position) };
    }
    if (scope.kind === "heads" && kind === "probe") return { kind: "heads", position: single(scope.position) };
    if (scope.kind === "sites") {
      const sites = scope.sites.filter((x) => streams(x.kind) || (kind === "probe" && x.kind === "head")).map((x) => ({ ...x, position: single(x.position) }));
      if (sites.length) return { kind: "sites", sites };
    }
    return { kind: "layer_components", components: ["resid_post"], position: last };
  }
  if (kind === "path_patching") {
    // Paths start at heads, attention outputs or MLP outputs.
    if (scope.kind === "layer_position") return { kind: "heads", position: { kind: "all" } };
    if (scope.kind === "layer_components") {
      const components = scope.components.filter((c) => c === "attn_out" || c === "mlp_out");
      return { ...scope, components: components.length ? components : ["attn_out", "mlp_out"] };
    }
    if (scope.kind === "sites") {
      const sites = scope.sites.filter((x) => x.kind === "head" || x.kind === "attn_out" || x.kind === "mlp_out");
      return sites.length ? { kind: "sites", sites } : { kind: "heads", position: { kind: "all" } };
    }
    return scope;
  }
  if (kind !== "direct_logit_attribution") return scope;
  switch (scope.kind) {
    case "heads":
      return { kind: "heads", position: last };
    case "layer_components": {
      const components = scope.components.filter((c) => c === "attn_out" || c === "mlp_out");
      return { kind: "layer_components", components: components.length ? components : ["attn_out", "mlp_out"], position: last };
    }
    case "layer_position":
      return { kind: "heads", position: last };
    case "sites": {
      // Values keep their positions (any token's value reaches the last through the attention);
      // queries and keys have no direct effect.
      const sites = scope.sites
        .filter((x) => x.kind === "head" || x.kind === "attn_out" || x.kind === "mlp_out" || x.kind === "neuron" || x.kind === "v")
        .map((x) => (x.kind === "v" ? x : { ...x, position: last }));
      return sites.length ? { kind: "sites", sites } : { kind: "heads", position: last };
    }
    case "features":
    case "site_sets":
    case "edges":
      return { kind: "heads", position: last };
    default:
      return scope;
  }
}

/** Every site of some sets, each once, in the order they first appear. */
export function setMembers(sets: { sites: SiteSpec[] }[]): SiteSpec[] {
  const seen = new Set<string>();
  const out: SiteSpec[] = [];
  for (const set of sets) {
    for (const site of set.sites) {
      const key = siteText(site);
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(site);
    }
  }
  return out;
}
