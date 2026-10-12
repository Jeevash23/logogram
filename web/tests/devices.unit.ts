import { test, expect } from "@playwright/test";

import type { SystemReport } from "../src/api/types";
import { gpuChoices, indexedGpus, whereText } from "../src/lib/devices";

const GB = 1024 ** 3;

function report(gpus: SystemReport["gpus"], defaultGpu: number | null = 0): SystemReport {
  return {
    os: "Linux", machine: "x86_64", python: "3.12", cpu: "CPU", cpu_cores: 4, cpu_threads: 8, memory_total: 16 * GB,
    memory_available: 8 * GB, gpus, backend: gpus.length ? "cuda" : "cpu", torch: "2", torch_cuda: null, transformer_lens: null,
    transformers: null, recommended_dtype: "float32", precision_note: "", hf_cache_free: null, mps_available: false, issues: [],
    default_gpu: defaultGpu,
  };
}

const A = { name: "Test GPU A", memory_total: 24 * GB, memory_free: 20 * GB, bf16: true, index: 0 };
const B = { name: "Test GPU B", memory_total: 48 * GB, memory_free: null, bf16: true, index: 1 };

test("with several GPUs, each is offered by the device a spec names it with", () => {
  expect(gpuChoices(report([A, B]))).toEqual([
    { value: "cuda", label: "The default GPU (cuda:0, Test GPU A)" },
    { value: "cuda:0", label: "GPU 0: Test GPU A · 24.0 GB, 20.0 GB free" },
    { value: "cuda:1", label: "GPU 1: Test GPU B · 48.0 GB" },
  ]);
  // The default GPU is the one torch names current, which needn't be the first.
  expect(gpuChoices(report([A, B], 1))[0].label).toBe("The default GPU (cuda:1, Test GPU B)");
});

test("one GPU, none, or Apple's offer no choice of GPU", () => {
  expect(gpuChoices(report([A]))).toEqual([]);
  expect(gpuChoices(report([]))).toEqual([]);
  expect(gpuChoices(null)).toEqual([]);
  const apple = { name: "Apple GPU (Metal)", memory_total: 16 * GB, memory_free: null, bf16: true, index: null };
  expect(indexedGpus(report([apple]))).toEqual([]);
});

test("a memory estimate says which memory it compares against", () => {
  expect(whereText("cuda:1")).toBe("on GPU 1");
  expect(whereText("cuda")).toBe("on the GPU");
  expect(whereText("cpu")).toBe("in RAM");
  expect(whereText("mps")).toBe("in unified memory");
});
