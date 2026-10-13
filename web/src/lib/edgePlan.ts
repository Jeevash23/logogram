import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { EdgePlan, Spec } from "../api/types";
import { useStore } from "../store/app";

export interface EdgePlanState {
  plan: EdgePlan | null;
  /** Why the server couldn't plan the run (no project, say); not a refusal of the run. */
  error: string | null;
  pending: boolean;
}

/**
 * What an edge attribution or ACDC spec would do on the loaded model, asked of the server once the
 * form settles: the graph's size, what a run costs (for ACDC, how long it may take here), and why
 * a run would be refused, from the checks the run makes (src/logogram/edges.py edge_plan,
 * src/logogram/acdc.py acdc_plan). Null without a model or another method.
 */
export function useEdgePlan(spec: Spec | null): EdgePlanState {
  const model = useStore((st) => (st.model.state === "ready" ? (st.model.info?.id ?? null) : null));
  // What can change the plan: the graph, the method and curve, the prompts and how they are read,
  // and the batch size. ACDC's threshold and rule don't change what a run costs. The spec is
  // built afresh on every render, so it is compared as text.
  const e = spec?.experiment;
  const method = e?.kind === "acdc" ? { kind: e.kind, direction: e.direction } : e;
  const key =
    spec && model && (e?.kind === "edge_attribution" || e?.kind === "acdc")
      ? JSON.stringify([model, method, spec.scope, spec.dataset, spec.tokenization, spec.metric, spec.execution])
      : null;
  const [state, setState] = useState<EdgePlanState>({ plan: null, error: null, pending: false });
  useEffect(() => {
    if (!key || !spec) {
      setState({ plan: null, error: null, pending: false });
      return;
    }
    let cancelled = false;
    setState((s) => ({ ...s, pending: true }));
    const timer = window.setTimeout(() => {
      api.edgePlan(spec).then(
        (plan) => !cancelled && setState({ plan, error: null, pending: false }),
        (err: Error) => !cancelled && setState({ plan: null, error: err.message, pending: false }),
      );
    }, 250);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
    // The key says when the spec changed in a way that matters.
  }, [key]);
  return key ? state : { plan: null, error: null, pending: false };
}
