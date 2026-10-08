import { useId, useMemo } from "react";

import { divergingScale } from "../lib/color";
import { logogram } from "../lib/logogram";
import { useStore } from "../store/app";

interface Props {
  seed: string;
  /** Per-layer signed strengths (see lib/logogram). Without one the glyph comes from the seed. */
  profile?: readonly (number | null | undefined)[] | null;
  size?: number;
  /** An accessible description; without one the glyph is decorative. */
  title?: string;
  className?: string;
  /** Soft ink diffusing around the stroke, as in the film. On by default for large glyphs. */
  haze?: boolean;
  /** Tendrils and droplets, 0–1; by default from the size. */
  detail?: number;
  /** Stroke weight relative to a normal glyph. */
  weight?: number;
}

/** The seed of Logogram's own glyph (the mark in the shell, the favicon, the welcome screens). */
export const BRAND_SEED = "louise";

/** A run's logogram: an ink ring written by its results. */
export function Logogram({ seed, profile, size = 40, title, className, haze, detail: detailOverride, weight = 1 }: Props) {
  const theme = useStore((st) => st.theme);
  const raw = useId();
  const id = raw.replace(/[^a-zA-Z0-9_-]/g, "");
  const detail = detailOverride ?? (size >= 96 ? 1 : size >= 52 ? 0.6 : size >= 30 ? 0.35 : 0);
  const key = profile ? profile.map((v) => (v === null || v === undefined ? "" : v.toFixed(3))).join(",") : "";
  // `key` stands in for `profile`, which callers often rebuild on every render.
  const glyph = useMemo(() => logogram({ seed, profile, detail, weight }), [seed, key, detail, weight]);
  const color = useMemo(() => divergingScale(1, theme), [theme]);
  const showHaze = haze ?? size >= 64;
  const showBleed = glyph.bleeds.length > 0 && size >= 22;
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 100 100"
      overflow="visible"
      role={title ? "img" : undefined}
      aria-label={title}
      aria-hidden={title ? undefined : true}
      focusable="false"
    >
      {title && <title>{title}</title>}
      <defs>
        {showBleed && (
          <filter id={`${id}b`} filterUnits="userSpaceOnUse" x="-30" y="-30" width="160" height="160">
            <feGaussianBlur stdDeviation={size >= 64 ? 3.2 : 2.4} />
          </filter>
        )}
        {showHaze && (
          <filter id={`${id}h`} filterUnits="userSpaceOnUse" x="-30" y="-30" width="160" height="160">
            <feGaussianBlur stdDeviation="2.6" />
          </filter>
        )}
      </defs>
      {showBleed && (
        <g filter={`url(#${id}b)`} opacity={theme === "dark" ? 0.75 : 0.85}>
          {glyph.bleeds.map((b, i) => (
            <circle key={i} cx={b.x} cy={b.y} r={b.r} fill={color(b.value)} />
          ))}
        </g>
      )}
      {showHaze && <path d={glyph.d} fill="currentColor" opacity={0.18} filter={`url(#${id}h)`} />}
      <path d={glyph.d} fill="currentColor" fillRule="nonzero" />
    </svg>
  );
}
