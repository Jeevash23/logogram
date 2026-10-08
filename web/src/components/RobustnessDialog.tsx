import { useEffect, useState } from "react";

import type { BaselineSpec, ExperimentSpec } from "../api/types";
import { useActiveRun } from "../lib/hooks";
import { experimentText } from "../lib/spec";
import { useStore } from "../store/app";
import { Button, Choices, Dialog, Field, Input, Select } from "./ui";

type Option = "direction" | "zero" | "mean" | "resample" | "donors" | "seed" | "total" | "prompts" | "exact" | "split" | "other" | "mlps" | "pathdirection" | "whole";

function suggest(e: ExperimentSpec): Option {
  if (e.kind === "activation_patching") return e.direction === "clean_to_corrupt" ? "direction" : "resample";
  if (e.kind === "direct_logit_attribution") return "total";
  if (e.kind === "attribution_patching") return "exact";
  if (e.kind === "steering") return "split";
  if (e.kind === "path_patching") return "mlps";
  if (e.baseline.kind === "resample") return "donors";
  if (e.baseline.kind === "zero") return "mean";
  return "resample";
}

function variant(e: ExperimentSpec, option: Option, params: { donors: number; seed: number; reference: "clean" | "corrupt"; pool: "clean" | "corrupt" }): ExperimentSpec {
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
      return { kind: "activation_patching", direction: e.kind === "path_patching" ? e.direction : "clean_to_corrupt" };
    case "split":
      // The same steering with the pairs split differently: does the direction depend on which
      // pairs computed it?
      return e.kind === "steering" ? { ...e, seed: e.seed + 1 } : e;
    case "other":
      return e.kind === "steering" ? { ...e, apply_to: e.apply_to === "clean" ? "corrupt" : "clean" } : e;
    case "exact":
      // Every site of the sweep, patched for real.
      return { kind: "activation_patching", direction: e.kind === "attribution_patching" ? e.direction : "clean_to_corrupt" };
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
  }
}

export function RobustnessDialog() {
  const open = useStore((st) => st.robustnessDialogOpen);
  const start = useStore((st) => st.startRobustness);
  const job = useStore((st) => st.job);
  const run = useActiveRun();
  const exp = run.detail?.spec.experiment;
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

  if (!exp || !run.id) return null;
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
    options.push({
      value: "split",
      title: "Another split of the pairs",
      detail: `Seed ${exp.seed + 1}: other pairs compute the directions, and others are measured.`,
    });
    options.push({
      value: "other",
      title: exp.apply_to === "clean" ? "Steer corrupt prompts toward clean" : "Steer clean prompts toward corrupt",
      detail: "The same sites and strengths in the other direction.",
    });
  } else if (exp.kind === "attribution_patching") {
    options.push({
      value: "exact",
      title: "Patch every site for real",
      detail: "The same sweep with activation patching: one patched run per site and prompt, exact rather than estimated.",
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
  if (exp.kind !== "direct_logit_attribution" && exp.kind !== "attribution_patching" && exp.kind !== "steering" && exp.kind !== "path_patching") {
    if (!(exp.kind === "ablation" && exp.baseline.kind === "zero")) options.push({ value: "zero", title: "Zero ablation", detail: "Replace the activation with zeros." });
    if (!(exp.kind === "ablation" && exp.baseline.kind === "mean" && !isResample))
      options.push({ value: "mean", title: "Mean ablation", detail: "Replace it with its mean over a reference set." });
    if (!isResample) options.push({ value: "resample", title: "Resample ablation", detail: "Replace it with its value in other prompts." });
  }

  const next = variant(exp, option, { donors, seed, reference, pool });
  const same = JSON.stringify(next) === JSON.stringify(exp);

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
              void start(run.id as string, next);
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
