// Edges of the model's graph (src/logogram/edges.py): where each node is drawn, which edges are
// strongest, how near a point is to an edge as drawn, and the ticks of a faithfulness curve's
// axis. Pure, so the logic checks run it.

import type { EdgeSenderSpec, EdgeSpec, PathReceiverSpec } from "../api/types";

/** Where a node is drawn: its column (the embeddings, then each layer's heads and its MLP, then
 * the logits, left to right) and its row (a head's index, or null for a node drawn in the middle
 * of its column). */
export interface NodePlace {
  column: number;
  head: number | null;
}

/** Columns of a model's graph: the embeddings, two per layer and the logits. */
export function graphColumns(nLayers: number): number {
  return 2 * nLayers + 2;
}

export function senderPlace(s: EdgeSenderSpec): NodePlace {
  if (s.kind === "embed") return { column: 0, head: null };
  if (s.kind === "mlp") return { column: 2 + 2 * s.layer, head: null };
  return { column: 1 + 2 * s.layer, head: s.head };
}

/** Where a receiver is drawn. With grouped-query attention (`group` query heads to a key/value
 * head), a key or value is its key/value head's, read by every query head of its group: it is
 * drawn in the middle of the group. */
export function receiverPlace(r: PathReceiverSpec, nLayers: number, group = 1): NodePlace {
  if (r.kind === "logits") return { column: graphColumns(nLayers) - 1, head: null };
  if (r.kind === "mlp") return { column: 2 + 2 * r.layer, head: null };
  const shared = r.input !== "q" && group > 1;
  return { column: 1 + 2 * r.layer, head: shared ? r.head * group + (group - 1) / 2 : r.head };
}

/** Which input of a head an edge ends at, as an offset in rows: the query above the head, the
 * key at it and the value below it. */
export function inputOffset(r: PathReceiverSpec): number {
  return r.kind === "head" ? { q: -1, k: 0, v: 1 }[r.input] : 0;
}

interface Ranked {
  index: number;
  effect: { mean: number | null };
}

/** The strongest first: by the size of the mean effect, ties in the run's order. Edges without a
 * value are left out. */
export function rankEdges<T extends Ranked>(sites: T[]): T[] {
  return sites
    .filter((x) => x.effect.mean !== null && Number.isFinite(x.effect.mean))
    .sort((a, b) => Math.abs(b.effect.mean as number) - Math.abs(a.effect.mean as number) || a.index - b.index);
}

/** A curve from a sender to a receiver, leaving and arriving horizontally. */
export interface EdgeCurve {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** The point at t (0 to 1) along an edge drawn as a cubic curve whose handles are pulled 45% of
 * the way across. */
export function curvePoint(c: EdgeCurve, t: number): { x: number; y: number } {
  const bend = (c.x2 - c.x1) * 0.45;
  const p0 = [c.x1, c.y1];
  const p1 = [c.x1 + bend, c.y1];
  const p2 = [c.x2 - bend, c.y2];
  const p3 = [c.x2, c.y2];
  const u = 1 - t;
  const a = u * u * u;
  const b = 3 * u * u * t;
  const d = 3 * u * t * t;
  const e = t * t * t;
  return { x: a * p0[0] + b * p1[0] + d * p2[0] + e * p3[0], y: a * p0[1] + b * p1[1] + d * p2[1] + e * p3[1] };
}

/** The handles of the curve curvePoint follows, for drawing it. */
export function curveHandles(c: EdgeCurve): [number, number, number, number] {
  const bend = (c.x2 - c.x1) * 0.45;
  return [c.x1 + bend, c.y1, c.x2 - bend, c.y2];
}

/** How far a point is from an edge as drawn, measured at `samples` points along it. */
export function distanceToCurve(c: EdgeCurve, x: number, y: number, samples = 32): number {
  let best = Infinity;
  let prev = curvePoint(c, 0);
  for (let i = 1; i <= samples; i++) {
    const next = curvePoint(c, i / samples);
    best = Math.min(best, distanceToSegment(x, y, prev.x, prev.y, next.x, next.y));
    prev = next;
  }
  return best;
}

function distanceToSegment(x: number, y: number, x1: number, y1: number, x2: number, y2: number): number {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const length = dx * dx + dy * dy;
  const t = length === 0 ? 0 : Math.max(0, Math.min(1, ((x - x1) * dx + (y - y1) * dy) / length));
  return Math.hypot(x - (x1 + t * dx), y - (y1 + t * dy));
}

/** The edge of `curves` nearest a point, if one is within `within` pixels: its position in the
 * list. Where two are as near, the first (the stronger) wins. */
export function nearestEdge(curves: EdgeCurve[], x: number, y: number, within = 6): number | null {
  let best: number | null = null;
  let distance = within;
  curves.forEach((c, i) => {
    if (x < Math.min(c.x1, c.x2) - within || x > Math.max(c.x1, c.x2) + within) return;
    const d = distanceToCurve(c, x, y);
    if (d < distance) {
      best = i;
      distance = d;
    }
  });
  return best;
}

/** Ticks of a logarithmic axis from lo to hi (both from 1): 1, 3, 10, 30, … within it, or only
 * the powers of ten when there are more than four decades. */
export function logTicks(lo: number, hi: number): number[] {
  const out: number[] = [];
  const decades = Math.log10(hi) - Math.log10(lo);
  for (let power = Math.floor(Math.log10(lo)); 10 ** power <= hi; power++) {
    for (const m of decades > 4 ? [1] : [1, 3]) {
      const v = m * 10 ** power;
      if (v >= lo && v <= hi) out.push(v);
    }
  }
  return out;
}

/** An edge's identity, the same in any run that names it (src/logogram/edges.py edge_key). */
export function edgeKey(e: EdgeSpec): string {
  const s = e.sender;
  const r = e.receiver;
  return `${s.kind}:${s.layer ?? ""}:${s.head ?? ""}>${r.kind}:${"layer" in r ? r.layer : ""}:${r.kind === "head" ? `${r.head}:${r.input}` : ""}`;
}
