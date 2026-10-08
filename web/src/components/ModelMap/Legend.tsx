import { legendStops } from "../../lib/color";
import { num } from "../../lib/format";
import { useActiveRun } from "../../lib/hooks";
import { useStore } from "../../store/app";
import { Segmented } from "../ui";
import s from "./ModelMap.module.css";

export function Legend({ bound, hasFlags, flagCount }: { bound: number; hasFlags: boolean; flagCount: number }) {
  const theme = useStore((st) => st.theme);
  const metric = useStore((st) => st.mapMetric);
  const scaleMode = useStore((st) => st.scaleMode);
  const run = useActiveRun();
  const stops = legendStops(theme);
  const gradient = `linear-gradient(to right, ${stops.join(", ")})`;
  const exp = run.detail?.spec.experiment;
  const direction =
    exp?.kind === "activation_patching" && exp.direction === "clean_to_corrupt"
      ? "restores the clean behavior"
      : "shifts the output as far as the corrupt prompt does";

  return (
    <div className={s.legend}>
      <div className={s.legendControls}>
        <Segmented
          label="Value shown on the map"
          value={metric}
          onChange={(v) => useStore.setState({ mapMetric: v })}
          options={[
            { value: "effect", label: "Normalized effect" },
            { value: "delta", label: "Δ logit diff" },
          ]}
        />
        <Segmented
          label="Color scale"
          value={metric === "effect" ? scaleMode : "auto"}
          onChange={(v) => useStore.setState({ scaleMode: v })}
          options={[
            { value: "auto", label: "Fit", title: "Scale to the largest value in this run" },
            { value: "unit", label: "±1", title: "Fixed scale from −1 to 1", disabled: metric !== "effect" },
          ]}
        />
      </div>
      <div className={s.legendBar} style={{ background: gradient }} aria-hidden="true" />
      <div className={s.legendTicks}>
        <span>{num(-bound, bound < 1 ? 2 : 1)}</span>
        <span>0</span>
        <span>{num(bound, bound < 1 ? 2 : 1)}</span>
      </div>
      <div className={s.legendText}>
        {metric === "effect" ? (
          <>
            Normalized effect: 1 means the site alone {direction}; 0 means no change.
          </>
        ) : (
          <>Change in logit difference (answer − distractor) caused by the intervention.</>
        )}
      </div>
      {hasFlags && (
        <div className={s.legendFlag}>
          <span className={s.flagSwatch} aria-hidden="true" />
          {flagCount} conclusion{flagCount === 1 ? "" : "s"} changed in the robustness check
        </div>
      )}
    </div>
  );
}
