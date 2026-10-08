import { test, expect } from "@playwright/test";
import { layerAngle, layerProfile, logogram, RING_RADIUS } from "../src/lib/logogram";

/** The outer contour of a glyph (its first subpath) as points. */
function outerPoints(d: string): [number, number][] {
  const first = d.slice(1, d.indexOf("Z"));
  return first.split("L").map((pair) => pair.trim().split(" ").map(Number) as [number, number]);
}

function innerPoints(d: string): [number, number][] {
  const start = d.indexOf("Z") + 2;
  const second = d.slice(start, d.indexOf("Z", start));
  return second.split("L").map((pair) => pair.trim().split(" ").map(Number) as [number, number]);
}

/** Radius of the contour point closest in angle to ``angle``. */
function radiusAt(points: [number, number][], angle: number): number {
  let best = points[0];
  let bestGap = Infinity;
  for (const p of points) {
    const a = Math.atan2(p[1] - 50, p[0] - 50);
    const gap = Math.abs(Math.atan2(Math.sin(a - angle), Math.cos(a - angle)));
    if (gap < bestGap) {
      bestGap = gap;
      best = p;
    }
  }
  return Math.hypot(best[0] - 50, best[1] - 50);
}

test("a glyph is a pure function of its seed and profile", () => {
  const profile = [0.1, -0.4, 0.9, 0, 0.2];
  const a = logogram({ seed: "run-1", profile, detail: 1 });
  expect(logogram({ seed: "run-1", profile, detail: 1 })).toEqual(a);
  expect(logogram({ seed: "run-2", profile, detail: 1 }).d).not.toBe(a.d);
  expect(a.d).not.toMatch(/NaN|Infinity/);
  for (const seed of ["", "x", "a much longer seed with spaces"]) {
    expect(logogram({ seed, detail: 1 }).d).not.toMatch(/NaN|Infinity/);
  }
});

test("the results write the ring: outward for positive layers, inward for negative", () => {
  const n = 6;
  const flat = logogram({ seed: "same", profile: new Array(n).fill(0), detail: 0 });
  const positive = logogram({ seed: "same", profile: [0, 0, 1, 0, 0, 0], detail: 0 });
  const negative = logogram({ seed: "same", profile: [0, 0, 0, 0, -1, 0], detail: 0 });
  const at2 = layerAngle(2, n);
  const at4 = layerAngle(4, n);
  expect(radiusAt(outerPoints(positive.d), at2) - radiusAt(outerPoints(flat.d), at2)).toBeGreaterThan(8);
  expect(radiusAt(innerPoints(negative.d), at4)).toBeLessThan(radiusAt(innerPoints(flat.d), at4) - 6);
  // Far from the swell the ring is unchanged.
  const opposite = layerAngle(5, n);
  expect(Math.abs(radiusAt(outerPoints(positive.d), opposite) - radiusAt(outerPoints(flat.d), opposite))).toBeLessThan(0.5);
  // The data colour bleeds where the strong layer is, with its sign.
  expect(positive.bleeds).toHaveLength(1);
  expect(positive.bleeds[0].value).toBe(1);
  expect(Math.atan2(positive.bleeds[0].y - 50, positive.bleeds[0].x - 50)).toBeCloseTo(at2, 5);
  expect(negative.bleeds[0].value).toBe(-1);
  expect(flat.bleeds).toHaveLength(0);
  expect(radiusAt(outerPoints(flat.d), at2)).toBeGreaterThan(RING_RADIUS - 2);
});

test("a run's profile keeps each layer's strongest signed value", () => {
  const sites = [
    { index: 0, layer: 0 },
    { index: 1, layer: 0 },
    { index: 2, layer: 2 },
    { index: 3, layer: 9 },
  ];
  const values: Record<number, number | null> = { 0: 0.2, 1: -0.5, 2: null, 3: 1 };
  expect(layerProfile(sites, (i) => values[i], 3)).toEqual([-0.5, 0, 0]);
});
