// The devices a model can be loaded on, as the model dialog offers them: one GPU of several by
// its index (a spec's cuda:N), as the system report lists them.

import type { Device, SystemReport } from "../api/types";
import { bytes } from "./format";

/** The GPUs PyTorch sees, by index (Apple's GPU has none, and isn't chosen this way). */
export function indexedGpus(report: SystemReport | null | undefined): (SystemReport["gpus"][number] & { index: number })[] {
  return (report?.gpus ?? []).filter((g): g is SystemReport["gpus"][number] & { index: number } => typeof g.index === "number");
}

/** Each GPU to choose from when there is more than one: the default GPU ("cuda", as specs say it),
 * then each by its index. Empty with one GPU or none, when "cuda" says everything. */
export function gpuChoices(report: SystemReport | null | undefined): { value: Device; label: string }[] {
  const gpus = indexedGpus(report);
  if (gpus.length < 2) return [];
  const fallback = report?.default_gpu ?? 0;
  const first = gpus.find((g) => g.index === fallback) ?? gpus[0];
  return [
    { value: "cuda", label: `The default GPU (cuda:${first.index}, ${first.name})` },
    ...gpus.map((g) => ({
      value: `cuda:${g.index}` as Device,
      label: `GPU ${g.index}: ${g.name} · ${bytes(g.memory_total)}${g.memory_free !== null ? `, ${bytes(g.memory_free)} free` : ""}`,
    })),
  ];
}

/** Where a memory estimate compares against, after "free": "on GPU 1", "on the GPU", "in RAM". */
export function whereText(device: string): string {
  const gpu = /^cuda:(\d+)$/.exec(device);
  if (gpu) return `on GPU ${gpu[1]}`;
  return ({ cuda: "on the GPU", mps: "in unified memory", cpu: "in RAM" } as Record<string, string>)[device] ?? "";
}
