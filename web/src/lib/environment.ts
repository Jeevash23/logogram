// Where a run ran, against where a rerun would run. The same spec on the same machine with the
// same versions gives bit-identical results; other versions or another device may change the
// last digits, so the app says when they differ.

import type { Environment, EnvironmentDifference, Manifest, Spec } from "../api/types";

/** The versions a run's numbers can depend on beyond its spec, as manifests name them. */
export const ENVIRONMENT_VERSIONS: { key: string; label: string }[] = [
  { key: "logogram", label: "Logogram" },
  { key: "torch", label: "PyTorch" },
  { key: "transformer_lens", label: "TransformerLens" },
  { key: "transformers", label: "Transformers" },
  { key: "numpy", label: "NumPy" },
];

const DEVICES: Record<string, string> = { cpu: "the CPU", cuda: "CUDA", mps: "Apple's MPS" };

/** A device type in words: "the CPU", "CUDA". */
export function deviceText(type: string): string {
  return DEVICES[type] ?? type;
}

export interface EnvironmentChange {
  key: string;
  label: string;
  /** As the run's manifest recorded it. */
  recorded: string;
  /** What a rerun on this server would use. */
  here: string;
}

/**
 * What a run recorded about where it ran that differs from this server: the library versions,
 * and the device a rerun of its spec would use here ("auto" picks the server's default, as does
 * a device this machine doesn't have). Nothing when either side is unknown.
 */
export function environmentChanges(
  manifest: Manifest | null | undefined,
  spec: Spec | null | undefined,
  env: Environment | null | undefined,
): EnvironmentChange[] {
  if (!manifest || !env) return [];
  const out: EnvironmentChange[] = [];
  for (const { key, label } of ENVIRONMENT_VERSIONS) {
    const recorded = manifest.versions?.[key];
    const here = key === "logogram" ? env.logogram : env.versions[key];
    if (recorded && here && recorded !== here) out.push({ key, label, recorded, here });
  }
  const ran = manifest.device?.type;
  if (ran) {
    const wanted = spec?.model.device;
    const rerun = wanted && wanted !== "auto" && env.devices.includes(wanted) ? wanted : env.default_device;
    if (rerun !== ran) out.push({ key: "device", label: "Device", recorded: deviceText(ran), here: deviceText(rerun) });
  }
  return out;
}

function list(items: string[]): string {
  return items.length < 2 ? items.join("") : `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}

/** The note the Results view shows: which versions or device differ, and what that can mean. */
export function environmentNote(changes: EnvironmentChange[]): string | null {
  if (changes.length === 0) return null;
  const versions = changes.filter((c) => c.key !== "device").map((c) => `${c.label} ${c.recorded} (here ${c.here})`);
  const device = changes.find((c) => c.key === "device");
  const parts = [
    ...(versions.length ? [`with ${list(versions)}`] : []),
    ...(device ? [`on ${device.recorded} (here ${device.here})`] : []),
  ];
  return `This run ran ${parts.join(", ")}. A rerun here may differ in the last digits.`;
}

const DEVICE_NAMES: Record<string, string> = { cpu: "CPU", cuda: "CUDA", mps: "MPS" };

/** A difference between two runs' manifests, as Compare lists it: device types by name. */
export function differenceText(d: EnvironmentDifference): { a: string; b: string } {
  if (d.key !== "device.type") return { a: d.a, b: d.b };
  return { a: DEVICE_NAMES[d.a] ?? d.a, b: DEVICE_NAMES[d.b] ?? d.b };
}
