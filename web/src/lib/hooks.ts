import { useMemo } from "react";

import type { RunDetail, SiteResult } from "../api/types";
import { runView, useStore } from "../store/app";
import { analysisContext } from "./analysis";
import { layerProfile } from "./logogram";

export function useAnalysisContext() {
  const form = useStore((s) => s.form);
  const model = useStore((s) => s.model);
  const datasetPath = useStore((s) => s.datasetPath);
  const dataset = useStore((s) => s.dataset);
  const project = useStore((s) => s.project);
  const analysisSource = useStore((s) => s.analysisSource);
  const activeRunId = useStore((s) => s.activeRunId);
  const runDetails = useStore((s) => s.runDetails);
  const view = useStore((s) => s.view);
  const mapOverlay = useStore((s) => s.mapOverlay);
  return useMemo(() => analysisContext({ form, model, datasetPath, dataset, project, analysisSource, activeRunId, runDetails, view, mapOverlay }),
    [form, model, datasetPath, dataset, project, analysisSource, activeRunId, runDetails, view, mapOverlay]);
}

const MODEL_NAMES: Record<string, string> = {
  "openai-community/gpt2": "GPT-2 small",
  gpt2: "GPT-2 small",
  "openai-community/gpt2-medium": "GPT-2 medium",
  "openai-community/gpt2-large": "GPT-2 large",
  "openai-community/gpt2-xl": "GPT-2 XL",
  "EleutherAI/pythia-70m": "Pythia 70M",
  "EleutherAI/pythia-160m": "Pythia 160M",
  "EleutherAI/pythia-410m": "Pythia 410M",
  "EleutherAI/pythia-1b": "Pythia 1B",
  "HuggingFaceTB/SmolLM2-135M": "SmolLM2 135M",
  "HuggingFaceTB/SmolLM2-360M": "SmolLM2 360M",
  "Qwen/Qwen2.5-0.5B": "Qwen2.5 0.5B",
  "Qwen/Qwen3-0.6B-Base": "Qwen3 0.6B",
  "allenai/OLMo-2-0425-1B": "OLMo 2 1B",
  "meta-llama/Llama-3.2-1B": "Llama 3.2 1B",
  "google/gemma-2-2b": "Gemma 2 2B",
  "google/gemma-3-270m": "Gemma 3 270M",
  "microsoft/phi-1_5": "Phi-1.5",
};

export function modelName(id: string | null | undefined): string {
  if (!id) return "No model";
  return MODEL_NAMES[id] ?? id;
}

export function useActiveRun() {
  const id = useStore((s) => s.activeRunId);
  const detail = useStore((s) => (id ? s.runDetails[id] : undefined));
  const live = useStore((s) => (id ? s.live[id] : undefined));
  const view = useMemo(() => runView(detail, live), [detail, live]);
  return { id, detail, live, ...view, ciLevel: ciLevelOf(detail) };
}

/** The confidence level a run's intervals use, in percent. */
export function ciLevelOf(detail: RunDetail | undefined): number {
  return Math.round((detail?.summary?.statistics.ci ?? detail?.spec.statistics.ci ?? 0.95) * 100);
}

export interface Architecture {
  nLayers: number;
  nHeads: number;
  id: string;
  dModel: number | null;
  dHead: number | null;
  dMlp: number | null;
  siteKinds: string[];
  structure: string;
  normalization: string;
  activation: string;
}

export function useArchitecture(): Architecture | null {
  const info = useStore((s) => s.model.info);
  const run = useActiveRun();
  return useMemo(() => {
    const shape = run.model ?? info;
    if (!shape) return null;
    const matching = info && (!run.detail || (info.id === run.detail.spec.model.id && (!run.detail.spec.model.revision || info.revision === run.detail.spec.model.revision)));
    return {
      nLayers: shape.n_layers, nHeads: shape.n_heads, id: shape.id,
      dModel: shape.d_model ?? (matching ? info.d_model : null), dHead: shape.d_head ?? (matching ? info.d_head : null),
      dMlp: shape.d_mlp ?? (matching ? info.d_mlp : null),
      siteKinds: shape.site_kinds ?? (matching ? info.site_kinds : [...new Set(run.sites.map(s => s.kind))]),
      structure: run.model?.block_structure ?? (matching ? info.extra?.block_structure : null) ?? "components",
      normalization: run.model?.normalization ?? (matching ? info.extra?.normalization : null) ?? "Normalization",
      activation: run.model?.activation ?? (matching ? info.extra?.activation : null) ?? "",
    };
  }, [info, run.model, run.detail, run.sites]);
}

export function siteValue(site: SiteResult | undefined, metric: "effect" | "delta"): number | null {
  if (!site) return null;
  const v = metric === "effect" ? site.effect.mean : site.delta.mean;
  return v === null || v === undefined || !Number.isFinite(v) ? null : v;
}

/**
 * The per-layer profile that writes a run's logogram: in each layer, the measured site with the
 * largest normalized effect. Layers still running count as zero, so the glyph fills in live.
 */
export function useRunProfile(run: ReturnType<typeof useActiveRun>): number[] | null {
  return useMemo(() => {
    const layers = run.model?.n_layers ?? (run.sites.length ? Math.max(...run.sites.map((x) => x.layer)) + 1 : 0);
    if (!layers || Object.keys(run.results).length === 0) return null;
    return layerProfile(run.sites, (i) => siteValue(run.results[i], "effect"), layers);
  }, [run.model, run.sites, run.results]);
}
