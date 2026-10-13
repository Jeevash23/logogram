// The scatter of principal components: each class of points marked by an ink shape (no new hues),
// the axes' scales, which point a click or the keyboard reaches, and a point in words.

import type { ReadSite, RepresentationPoint } from "../api/types";
import { count } from "./format";

export type Shape = "circle" | "square" | "triangle" | "diamond";

/** How a class of points is drawn: an ink shape, filled or hollow. */
export interface ClassMark {
  shape: Shape;
  filled: boolean;
}

const SHAPES: Shape[] = ["circle", "square", "triangle", "diamond"];
/** The classes that get a shape of their own; the rest, and points without a class, are dots. */
export const MARKED_CLASSES = SHAPES.length * 2;

/** Each class's mark, in the order the classes first appear: the four shapes filled, then the four
 * hollow. Classes beyond the eighth have none (they are drawn as small dots, and the legend says
 * how many there are). */
export function classMarks(marks: string[]): Map<string, ClassMark> {
  const out = new Map<string, ClassMark>();
  marks.slice(0, MARKED_CLASSES).forEach((mark, i) => {
    out.set(mark, { shape: SHAPES[i % SHAPES.length], filled: i < SHAPES.length });
  });
  return out;
}

/** How many points each class has, in the order the classes are listed; null for points without a
 * class. */
export function classCounts(points: RepresentationPoint[]): Map<string | null, number> {
  const out = new Map<string | null, number>();
  for (const p of points) out.set(p.mark, (out.get(p.mark) ?? 0) + 1);
  return out;
}

/** A tidy domain for one axis: the coordinates' extent, zero included, padded a little. */
export function axisDomain(values: number[]): [number, number] {
  let lo = 0;
  let hi = 0;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  if (hi - lo < 1e-12) return [lo - 1, hi + 1];
  const pad = (hi - lo) * 0.06;
  return [lo - pad, hi + pad];
}

export interface ScatterScale {
  x: (v: number) => number;
  y: (v: number) => number;
  xDomain: [number, number];
  yDomain: [number, number];
}

/** Where points go in a plot of `width` × `height` (margins included): component `xc` across,
 * component `yc` up. */
export function scatterScale(
  points: RepresentationPoint[],
  xc: number,
  yc: number,
  width: number,
  height: number,
  margin: { left: number; right: number; top: number; bottom: number },
): ScatterScale {
  const xDomain = axisDomain(points.map((p) => p.coords[xc] ?? 0));
  const yDomain = axisDomain(points.map((p) => p.coords[yc] ?? 0));
  const w = Math.max(1, width - margin.left - margin.right);
  const h = Math.max(1, height - margin.top - margin.bottom);
  return {
    x: (v) => margin.left + ((v - xDomain[0]) / (xDomain[1] - xDomain[0])) * w,
    y: (v) => margin.top + h - ((v - yDomain[0]) / (yDomain[1] - yDomain[0])) * h,
    xDomain,
    yDomain,
  };
}

/** The order the arrow keys move through the points: from left to right along the horizontal
 * axis, ties from the bottom up, then by row. */
export function keyboardOrder(points: RepresentationPoint[], xc: number, yc: number): number[] {
  return points
    .map((_, i) => i)
    .sort((a, b) => {
      const pa = points[a].coords;
      const pb = points[b].coords;
      return (pa[xc] ?? 0) - (pb[xc] ?? 0) || (pa[yc] ?? 0) - (pb[yc] ?? 0) || points[a].row - points[b].row;
    });
}

/** The point nearest to (mx, my) within `radius` pixels, or null; ties to the earlier point. */
export function nearestPoint(points: RepresentationPoint[], scale: ScatterScale, xc: number, yc: number, mx: number, my: number, radius = 8): number | null {
  let best: number | null = null;
  let bestD = radius * radius;
  points.forEach((p, i) => {
    const dx = scale.x(p.coords[xc] ?? 0) - mx;
    const dy = scale.y(p.coords[yc] ?? 0) - my;
    const d = dx * dx + dy * dy;
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  });
  return best;
}

/** Which prompt or text a point is, and its class. */
export function pointLabel(point: RepresentationPoint): string {
  const which = point.side ? `Prompt ${point.index} (${point.side})` : `Text ${point.index}`;
  return point.mark !== null && point.mark !== point.side ? `${which} · ${point.mark}` : which;
}

/** A point in words: which prompt or text it is, its class, and the start of its text. */
export function pointText(point: RepresentationPoint, max = 140): string {
  const text = point.text.length > max ? `${point.text.slice(0, max)}…` : point.text;
  return `${pointLabel(point)}: ${text}`;
}

/** A share of variance as a percentage, with a decimal while it is small. */
export function explainedText(ratio: number): string {
  const p = 100 * ratio;
  return `${p >= 10 ? p.toFixed(0) : p.toFixed(1)}%`;
}

/** The sites an export writes: the site read, or every layer of its kind. */
export function exportSites(site: ReadSite, everyLayer: boolean, nLayers: number): ReadSite[] {
  if (!everyLayer) return [site];
  return Array.from({ length: nLayers }, (_, layer) => ({ kind: site.kind, layer }));
}

/** How large an export would be: rows × width × 4 bytes for each site. */
export function exportSize(rows: number, width: number, sites: number): string {
  const bytes = rows * width * 4 * sites;
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : bytes >= 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${count(Math.ceil(bytes / 1e3))} KB`;
}
