// The project's saved directions (src/logogram/directions.py): vectors in the model's activation
// space that a probe, a learned subspace, a principal component or a mean difference found, and
// that steering, directional ablation, subspace patching and readouts use.

import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { DirectionListing } from "../api/types";
import { useStore } from "../store/app";
import { plural } from "./format";
import { KIND_SHORT } from "./sites";

/** Tell every list of directions in this tab to read them again: one was saved, renamed or
 * deleted. (The server tells the other tabs, with a directions.changed event.) */
export function directionsChanged(): void {
  useStore.setState((st) => ({ directionsVersion: st.directionsVersion + 1 }));
}

/** The open project's saved directions, read again when they change or another project opens;
 * null until read. */
export function useSavedDirections(): { directions: DirectionListing[] | null; error: string | null } {
  const project = useStore((st) => st.project?.session_id);
  const version = useStore((st) => st.directionsVersion);
  const [state, setState] = useState<{ directions: DirectionListing[] | null; error: string | null }>({ directions: null, error: null });
  useEffect(() => {
    if (!project) return;
    let live = true;
    api
      .directions()
      .then((directions) => live && setState({ directions, error: null }))
      .catch((err: unknown) => live && setState({ directions: null, error: err instanceof Error ? err.message : String(err) }));
    return () => {
      live = false;
    };
  }, [project, version]);
  return state;
}

const MADE_BY: Record<NonNullable<DirectionListing["made_by"]>["method"], string> = {
  probe: "a probe",
  das: "a learned subspace",
  pca: "a principal component",
  mean_difference: "a mean difference",
  import: "an import",
  sae_feature: "an SAE feature",
};

/** A direction in a few words: its size, where it was found and how, for example
 * "1 dimension · L6 resid post · from a probe". */
export function directionText(d: DirectionListing): string {
  if (d.error) return d.error;
  const parts = [plural(d.dimensions ?? 1, "dimension")];
  if (d.site) parts.push(`L${d.site.layer} ${KIND_SHORT[d.site.kind as keyof typeof KIND_SHORT] ?? d.site.kind}`);
  if (d.made_by) parts.push(`from ${MADE_BY[d.made_by.method] ?? d.made_by.method}`);
  return parts.join(" · ");
}
