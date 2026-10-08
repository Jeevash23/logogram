import { inkStops, legendStops, type ResolvedTheme } from "../../lib/color";
import { num } from "../../lib/format";

/** A compact legend for a heatmap: diverging (−max…max) or ink (0…max). */
export function ScaleBar({
  bound,
  theme,
  kind = "diverging",
  label,
}: {
  bound: number;
  theme: ResolvedTheme;
  kind?: "diverging" | "ink";
  label: string;
}) {
  const stops = kind === "diverging" ? legendStops(theme) : inkStops(theme);
  const digits = bound < 1 ? 2 : 1;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: "var(--text-xs)", color: "var(--muted)" }}>
      <span>{label}</span>
      <span>{kind === "diverging" ? num(-bound, digits) : "0"}</span>
      <span
        aria-hidden="true"
        style={{
          width: 120,
          height: 8,
          borderRadius: 2,
          background: `linear-gradient(to right, ${stops.join(", ")})`,
        }}
      />
      <span>{num(bound, digits)}</span>
    </div>
  );
}
