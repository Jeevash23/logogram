// Data colors. Effects use a diverging cobalt–neutral–ochre scale, symmetric around zero and
// readable with the common forms of color blindness. Attention uses a single-hue ink ramp.
// Stops are interpolated in CIELAB so equal steps in value look like equal steps in color.

import { lab, rgb } from "d3-color";
import { scaleLinear } from "d3-scale";

export type ResolvedTheme = "light" | "dark";

interface Stops {
  neg: string; // strongest negative (ochre)
  negMid: string;
  zero: string;
  posMid: string;
  pos: string; // strongest positive (cobalt)
}

const DIVERGING: Record<ResolvedTheme, Stops> = {
  light: {
    neg: "#86560f",
    negMid: "#d29a3c",
    zero: "#eceeee",
    posMid: "#7aa0d8",
    pos: "#1f4795",
  },
  dark: {
    neg: "#f0bd62",
    negMid: "#9c7130",
    zero: "#232b36",
    posMid: "#3f679f",
    pos: "#a9c6f5",
  },
};

const INK: Record<ResolvedTheme, [string, string]> = {
  light: ["#f4f6f6", "#1c2226"],
  dark: ["#1d2530", "#e3e8ec"],
};

function labInterpolator(stops: string[], domain: number[]) {
  const labs = stops.map((c) => lab(c));
  return (t: number) => {
    let i = 0;
    while (i < domain.length - 2 && t > domain[i + 1]) i++;
    const span = domain[i + 1] - domain[i];
    const f = span === 0 ? 0 : Math.min(1, Math.max(0, (t - domain[i]) / span));
    const a = labs[i];
    const b = labs[i + 1];
    return lab(a.l + (b.l - a.l) * f, a.a + (b.a - a.a) * f, a.b + (b.b - a.b) * f);
  };
}

export interface ColorScale {
  (value: number | null | undefined): string;
  max: number;
}

/** Diverging scale over [-max, max]. Values beyond max are clamped. */
export function divergingScale(max: number, theme: ResolvedTheme): ColorScale {
  const s = DIVERGING[theme];
  const interp = labInterpolator([s.neg, s.negMid, s.zero, s.posMid, s.pos], [-1, -0.5, 0, 0.5, 1]);
  const cache = new Map<number, string>();
  const m = max > 0 ? max : 1;
  const fn = ((value: number | null | undefined) => {
    if (value === null || value === undefined || !Number.isFinite(value)) return s.zero;
    const t = Math.max(-1, Math.min(1, value / m));
    const key = Math.round(t * 512);
    let c = cache.get(key);
    if (!c) {
      c = rgb(interp(key / 512)).formatHex();
      cache.set(key, c);
    }
    return c;
  }) as ColorScale;
  fn.max = m;
  return fn;
}

/** Single-hue ink ramp over [0, max] for attention weights. */
export function inkScale(max: number, theme: ResolvedTheme): ColorScale {
  const [from, to] = INK[theme];
  const interp = labInterpolator([from, to], [0, 1]);
  const m = max > 0 ? max : 1;
  const fn = ((value: number | null | undefined) => {
    if (value === null || value === undefined || !Number.isFinite(value)) return from;
    return rgb(interp(Math.max(0, Math.min(1, value / m)))).formatHex();
  }) as ColorScale;
  fn.max = m;
  return fn;
}

/** A readable text color on top of a data color. */
export function textOn(color: string, theme: ResolvedTheme): string {
  const l = lab(color).l;
  if (theme === "light") return l < 58 ? "#f4f6f6" : "#1c2226";
  return l > 62 ? "#161c24" : "#e3e8ec";
}

/** Smallest scale bound per metric, so negligible effects never fill the color range. */
export const SCALE_FLOOR = { effect: 0.1, delta: 0.5 } as const;

/**
 * A tidy symmetric bound for a set of values (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8 × 10^k), never
 * below ``floor``.
 */
export function niceBound(values: (number | null | undefined)[], floor = 0): number {
  let m = floor;
  for (const v of values) if (v !== null && v !== undefined && Number.isFinite(v)) m = Math.max(m, Math.abs(v));
  if (m === 0) return 1;
  const exp = Math.floor(Math.log10(m));
  const base = 10 ** exp;
  for (const step of [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) {
    if (m <= step * base + 1e-12) return step * base;
  }
  return 10 * base;
}

export function ticks(max: number, count = 5): number[] {
  return scaleLinear().domain([-max, max]).ticks(count);
}

export function legendStops(theme: ResolvedTheme, steps = 32): string[] {
  const scale = divergingScale(1, theme);
  return Array.from({ length: steps + 1 }, (_, i) => scale(-1 + (2 * i) / steps));
}

export function inkStops(theme: ResolvedTheme, steps = 16): string[] {
  const scale = inkScale(1, theme);
  return Array.from({ length: steps + 1 }, (_, i) => scale(i / steps));
}
