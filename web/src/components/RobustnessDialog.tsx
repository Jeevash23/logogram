import { useEffect, useState } from "react";

import type { BaselineSpec, ExperimentSpec } from "../api/types";
import { usesFeatures } from "../lib/buildSpec";
import { DEFAULT_IG_STEPS } from "../lib/formState";
import { experimentText } from "../lib/spec";
import { useStore } from "../store/app";
import { Button, Choices, Dialog, Field, Input, Select } from "./ui";

type Option = "direction" | "zero" | "mean" | "resample" | "donors" | "seed" | "total" | "prompts" | "exact" | "estimator" | "split" | "other" | "mlps" | "pathdirection" | "whole" | "float32" | "edgedirection";

function suggest(e: ExperimentSpec): Option {
  if (e.kind === "activation_patching") return e.direction === "clean_to_corrupt" ? "direction" : "resample";
  if (e.kind === "direct_logit_attribution") return "total";
  if (e.kind === "attribution_patching") return "exact";
  if (e.kind === "steering" || e.kind === "probe" || e.kind === "das") return "split";
  if (e.kind === "path_patching") return "mlps";
  if (e.kind === "edge_attribution") return "estimator";
  if (e.kind === "subspace_patching") return "whole";
  if (e.baseline.kind === "scale" || e.baseline.kind === "projection") return "zero";
  if (e.baseline.kind === "resample") return "donors";
  if (e.baseline.kind === "zero") return "mean";
  return "resample";
}

function variant(e: ExperimentSpec, option: Option, params: { donors: number; seed: number; reference: "clean" | "corrupt"; pool: "clean" | "corrupt"; steps: number }): ExperimentSpec {
  const base = e.kind === "ablation" ? e.baseline : null;
  switch (option) {
    case "total":
      // The same components, patched: their total effect, through everything downstream.
      return { kind: "activation_patching", direction: e.kind === "direct_logit_attribution" && e.prompts === "corrupt" ? "clean_to_corrupt" : "corrupt_to_clean" };
    case "mlps":
      return e.kind === "path_patching" ? { ...e, freeze_mlps: !e.freeze_mlps } : e;
    case "pathdirection":
      return e.kind === "path_patching" ? { ...e, direction: e.direction === "clean_to_corrupt" ? "corrupt_to_clean" : "clean_to_corrupt" } : e;
    case "whole":
      // The senders' whole effect, through every path: how much of it the receivers carry.
      return { kind: "activation_patching", direction: e.kind === "path_patching" || e.kind === "subspace_patching" ? e.direction : "clean_to_corrupt" };
    case "split":
      // The same method with the pairs split differently: does what it found depend on which
      // pairs found it?
      return e.kind === "steering" || e.kind === "probe" || e.kind === "das" ? { ...e, seed: e.seed + 1 } : e;
    case "other":
      return e.kind === "steering" ? { ...e, apply_to: e.apply_to === "clean" ? "corrupt" : "clean" } : e;
    case "exact":
      // Every site of the sweep, patched for real.
      return { kind: "activation_patching", direction: e.kind === "attribution_patching" ? e.direction : "clean_to_corrupt" };
    case "estimator":
      // The same estimate made the other way: integrated gradients for one gradient, and back.
      if (e.kind !== "attribution_patching" && e.kind !== "edge_attribution") return e;
      return e.method === "gradient"
        ? { ...e, method: "integrated_gradients", steps: params.steps }
        : { ...e, method: "gradient", steps: null };
    case "edgedirection":
      // The same edges estimated the other way: the other prompt of each pair as the receiver.
      return e.kind === "edge_attribution" ? { ...e, direction: e.direction === "clean_to_corrupt" ? "corrupt_to_clean" : "clean_to_corrupt" } : e;
    case "prompts":
      return { kind: "direct_logit_attribution", prompts: e.kind === "direct_logit_attribution" && e.prompts === "clean" ? "corrupt" : "clean" };
    case "direction":
      return {
        kind: "activation_patching",
        direction: e.kind === "activation_patching" && e.direction === "clean_to_corrupt" ? "corrupt_to_clean" : "clean_to_corrupt",
      };
    case "zero":
      return { kind: "ablation", baseline: { kind: "zero" } };
    case "mean":
      return { kind: "ablation", baseline: { kind: "mean", reference: params.reference } };
    case "resample": {
      const b: BaselineSpec = { kind: "resample", pool: params.pool, donors: params.donors, seed: params.seed };
      return { kind: "ablation", baseline: b };
    }
    case "donors":
      // Only the donor count changes; the pool and seed stay as they were.
      return base?.kind === "resample"
        ? { kind: "ablation", baseline: { ...base, donors: params.donors } }
        : variant(e, "resample", params);
    case "seed":
      // Only the seed changes.
      return base?.kind === "resample"
        ? { kind: "ablation", baseline: { ...base, seed: params.seed } }
        : variant(e, "resample", params);
    case "float32":
      // The experiment stays; the model runs in float32 instead.
      return e;
  }
}

export function RobustnessDialog() {
  const open = useStore((st) => st.robustnessDialogOpen);
  const start = useStore((st) => st.startRobustness);
  const job = useStore((st) => st.job);
  // Only the active run's spec: the dialog doesn't follow a run's progress.
  const runId = useStore((st) => st.activeRunId);
  const spec = useStore((st) => (st.activeRunId ? st.runDetails[st.activeRunId]?.spec : undefined));
  const exp = spec?.experiment;
  const dtype = spec?.model.dtype;
  const [option, setOption] = useState<Option>("resample");
  const [donors, setDonors] = useState(10);
  const [seed, setSeed] = useState(1);
  const [reference, setReference] = useState<"clean" | "corrupt">("corrupt");
  const [pool, setPool] = useState<"clean" | "corrupt">("corrupt");

  useEffect(() => {
    if (!open || !exp) return;
    setOption(suggest(exp));
    if (exp.kind === "ablation" && exp.baseline.kind === "resample") {
      setDonors(exp.baseline.donors * 2);
      setSeed(exp.baseline.seed + 1);
      setPool(exp.baseline.pool);
    } else {
      setDonors(10);
      setSeed(0);
    }
    if (exp.kind === "ablation" && exp.baseline.kind === "mean") setReference(exp.baseline.reference === "clean" ? "corrupt" : "clean");
  }, [open, exp]);

  if (!exp || !runId) return null;
  const isResample = exp.kind === "ablation" && exp.baseline.kind === "resample";
  const options: { value: Option; title: string; detail: string }[] = [];
  if (exp.kind === "direct_logit_attribution") {
    options.push({
      value: "total",
      title: "Total effect by patching",
      detail:
        exp.prompts === "clean"
          ? "Patch corrupt into clean at the same components: what each does through everything downstream, not just directly."
          : "Patch clean into corrupt at the same components: what each does through everything downstream.",
    });
    options.push({
      value: "prompts",
      title: exp.prompts === "clean" ? "Split the corrupt prompts" : "Split the clean prompts",
      detail: "The same split on the other prompt of each pair: which direct effects the corruption changes.",
    });
  } else if (exp.kind === "path_patching") {
    options.push({
      value: "mlps",
      title: exp.freeze_mlps ? "Let the MLPs carry the path" : "Hold the MLPs too",
      detail: exp.freeze_mlps ? "Recompute the MLPs, so paths through them count." : "Keep only the direct path through the residual stream.",
    });
    options.push({
      value: "whole",
      title: "Whole effect by patching",
      detail: "Patch the same senders through every path, to see how much of their effect the receivers carry.",
    });
    options.push({ value: "pathdirection", title: "Opposite direction", detail: "The same paths, patched the other way." });
  } else if (exp.kind === "steering") {
    // Along an SAE feature, the split decides what strength 1 is: its typical activation.
    const feature = exp.direction?.kind === "sae_feature";
    options.push({
      value: "split",
      title: "Another split of the pairs",
      detail: feature
        ? `Seed ${exp.seed + 1}: other pairs measure the feature's typical activation, and others are measured.`
        : `Seed ${exp.seed + 1}: other pairs compute the directions, and others are measured.`,
    });
    options.push({
      value: "other",
      title: feature
        ? exp.apply_to === "clean" ? "Steer corrupt prompts" : "Steer clean prompts"
        : exp.apply_to === "clean" ? "Steer corrupt prompts toward clean" : "Steer clean prompts toward corrupt",
      detail: feature ? "The same feature, sites and strengths, added to the other prompts." : "The same sites and strengths in the other direction.",
    });
  } else if (exp.kind === "probe") {
    options.push({
      value: "split",
      title: "Another split of the pairs",
      detail: `Seed ${exp.seed + 1}: other pairs train the probes, and others measure them.`,
    });
  } else if (exp.kind === "das") {
    options.push({
      value: "split",
      title: "Another split of the pairs",
      detail: `Seed ${exp.seed + 1}: other pairs learn the subspaces, from another starting point, and others measure them.`,
    });
  } else if (exp.kind === "subspace_patching") {
    options.push({
      value: "whole",
      title: "Patch the whole site",
      detail: "The same sites patched as a whole: how much of their effect the direction carries.",
    });
  } else if (exp.kind === "attribution_patching") {
    options.push({
      value: "exact",
      title: "Patch every site for real",
      detail: "The same sweep with activation patching: one patched run per site and prompt, exact rather than estimated.",
    });
    // SAE features are estimated from one gradient only.
    if (!(spec && usesFeatures(spec.scope))) options.push(
      exp.method === "gradient"
        ? {
            value: "estimator",
            title: "Estimate with integrated gradients",
            detail: `Average the gradient over ${DEFAULT_IG_STEPS} steps between the prompts, which corrects much of the saturation one gradient misses, at ${DEFAULT_IG_STEPS} times the cost.`,
          }
        : { value: "estimator", title: "Estimate with one gradient", detail: "The gradient at the receiver run only: what the steps of integrated gradients changed." },
    );
  } else if (exp.kind === "edge_attribution") {
    options.push(
      exp.method === "gradient"
        ? {
            value: "estimator",
            title: "Estimate with integrated gradients",
            detail: `Average each receiver's gradient over ${DEFAULT_IG_STEPS} steps between the prompts, which corrects much of the saturation one gradient misses, at ${DEFAULT_IG_STEPS} times the cost of the estimates.`,
          }
        : { value: "estimator", title: "Estimate with one gradient", detail: "The gradient at the receiver run only: what the steps of integrated gradients changed." },
    );
    options.push({
      value: "edgedirection",
      title: "Opposite direction",
      detail:
        exp.direction === "clean_to_corrupt"
          ? "Estimate the edges on the clean prompts, carrying corrupt values: which edges the behavior needs."
          : "Estimate the edges on the corrupt prompts, carrying clean values: which edges bring the behavior back.",
    });
  } else if (exp.kind === "activation_patching") {
    options.push({
      value: "direction",
      title: "Opposite direction",
      detail:
        exp.direction === "clean_to_corrupt"
          ? "Patch corrupt into clean instead: necessity rather than sufficiency."
          : "Patch clean into corrupt instead: sufficiency rather than necessity.",
    });
  }
  if (isResample) {
    options.push({ value: "donors", title: "More donors", detail: "Same pool and seed rule, more donors per prompt." });
    options.push({ value: "seed", title: "Another donor seed", detail: "Same number of donors, drawn differently." });
  }
  if (exp.kind === "activation_patching" || exp.kind === "ablation") {
    if (!(exp.kind === "ablation" && exp.baseline.kind === "zero")) options.push({ value: "zero", title: "Zero ablation", detail: "Replace the activation with zeros." });
    if (!(exp.kind === "ablation" && exp.baseline.kind === "mean" && !isResample))
      options.push({ value: "mean", title: "Mean ablation", detail: "Replace it with its mean over a reference set." });
    if (!isResample) options.push({ value: "resample", title: "Resample ablation", detail: "Replace it with its value in other prompts." });
  }

  if (dtype && dtype !== "float32") {
    options.push({
      value: "float32",
      title: "The same run in float32",
      detail: `Reload the model in float32 and rerun: what rounding in ${dtype} changed. Needs about twice the memory.`,
    });
  }

  const next = variant(exp, option, { donors, seed, reference, pool, steps: DEFAULT_IG_STEPS });
  const same = option !== "float32" && JSON.stringify(next) === JSON.stringify(exp);

  return (
    <Dialog
      open={open}
      onOpenChange={(v) => useStore.setState({ robustnessDialogOpen: v })}
      title="Check robustness"
      description="Rerun this sweep with one methodological choice changed, then compare the two. Conclusions that change are flagged on the map."
      wide
      footer={
        <>
          <Button variant="ghost" onClick={() => useStore.setState({ robustnessDialogOpen: false })}>
            Cancel
          </Button>
          <Button
            variant="primary"
            disabled={same || job?.status === "running"}
            onClick={() => {
              useStore.setState({ robustnessDialogOpen: false });
              void start(runId, option === "float32" ? { dtype: "float32" } : { experiment: next });
            }}
          >
            Run the check
          </Button>
        </>
      }
    >
      <p style={{ fontSize: "var(--text-sm)" }}>
        Now: <strong>{experimentText(exp)}</strong>
      </p>
      <Choices label="Change" value={option} onChange={setOption} columns={2} options={options} />
      {option === "mean" && (
        <Field label="Reference set">
          <Select value={reference} onChange={(e) => setReference(e.target.value as "clean" | "corrupt")}>
            <option value="corrupt">Corrupt prompts</option>
            <option value="clean">Clean prompts</option>
          </Select>
        </Field>
      )}
      {(option === "resample" || option === "donors" || option === "seed") && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 12 }}>
          <Field label="Donor pool">
            <Select value={pool} disabled={option !== "resample"} onChange={(e) => setPool(e.target.value as "clean" | "corrupt")}>
              <option value="corrupt">Corrupt prompts</option>
              <option value="clean">Clean prompts</option>
            </Select>
          </Field>
          <Field label="Donors per prompt">
            <Input type="number" min={1} value={donors} disabled={option === "seed"} onChange={(e) => setDonors(Math.max(1, Number(e.target.value)))} />
          </Field>
          <Field label="Donor seed">
            <Input type="number" value={seed} disabled={option === "donors"} onChange={(e) => setSeed(Number(e.target.value))} />
          </Field>
        </div>
      )}
      <p style={{ fontSize: "var(--text-sm)" }}>
        Will run: <strong>{experimentText(next)}</strong>, with everything else in the spec unchanged.
      </p>
    </Dialog>
  );
}
