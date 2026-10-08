// Logograms: circular ink glyphs, after the heptapod writing in Arrival.
//
// Every run has one. Its hand (wobble, brush texture, where the stroke starts) comes from a seed,
// usually the run id, so a run keeps its glyph. Once results exist, the ring is written by them:
// layers go clockwise from the top, and the ink swells outward where a layer's strongest effect is
// positive and inward where it is negative. A strong, robust result reads as a heavy, lopsided
// ring; a null result as a thin, even one. Glyphs are pure functions of their inputs.

export interface GlyphOptions {
  seed: string;
  /** Per-layer signed strength, any scale (normalized here). Omit for a glyph from the seed alone. */
  profile?: readonly (number | null | undefined)[] | null;
  /** 0–1: how many tendrils and droplets. Small glyphs read better plain. */
  detail?: number;
  /** Stroke weight relative to a normal glyph (a selection ring is thinner). */
  weight?: number;
}

export interface Bleed {
  x: number;
  y: number;
  r: number;
  /** Signed strength in [-1, 1], for the data color of the ink bleeding at this layer. */
  value: number;
}

export interface Glyph {
  /** Path data in a 100 × 100 box; fill with the nonzero rule. */
  d: string;
  bleeds: Bleed[];
}

const TAU = Math.PI * 2;
const N = 240;
/** Ring radius in the 100 × 100 box; swells and tendrils reach up to about 18 beyond it. */
export const RING_RADIUS = 30;
const R = RING_RADIUS;

export function hashString(text: string): number {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/** mulberry32: small, fast and the same in every browser. */
export function random(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function angularDistance(a: number, b: number): number {
  const d = Math.abs(a - b) % TAU;
  return d > Math.PI ? TAU - d : d;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

const fmt = (v: number) => (Math.round(v * 10) / 10).toString();

function polygon(points: [number, number][]): string {
  return `M${points.map(([x, y]) => `${fmt(x)} ${fmt(y)}`).join("L")}Z`;
}

/** Twice the signed area: positive for clockwise on screen (y grows downward). */
function signedArea(points: [number, number][]): number {
  let a = 0;
  for (let i = 0; i < points.length; i++) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[(i + 1) % points.length];
    a += x1 * y2 - x2 * y1;
  }
  return a;
}

function clockwise(points: [number, number][]): [number, number][] {
  return signedArea(points) >= 0 ? points : [...points].reverse();
}

/** The angle of layer ``i`` of ``n``: clockwise from the top. */
export function layerAngle(i: number, n: number): number {
  return -Math.PI / 2 + (TAU * (i + 0.5)) / n;
}

export function logogram({ seed, profile, detail = 0.5, weight = 1 }: GlyphOptions): Glyph {
  const rand = random(hashString(seed));
  const phase = () => rand() * TAU;
  const hasData = !!profile && profile.length > 0;

  // The hand: a gently uneven circle, a brush that thickens and thins, and where it starts.
  const wobble = [1, 2, 3].map((j) => ({ j, a: [0, 1.1, 0.75, 0.45][j] * (0.45 + rand()), p: phase() }));
  const texture = [2, 3, 5].map((j) => ({ j, a: 0.2 * (0.35 + rand()), p: phase() }));
  const start = phase();
  const blots = Array.from({ length: 2 + Math.floor(rand() * 3) }, () => ({
    at: phase(),
    width: 0.12 + rand() * 0.22,
    height: ((hasData ? 0.8 : 2.2) + rand() * (hasData ? 1.2 : 3)) * Math.min(1, weight * 1.5),
    outward: rand() > 0.3,
  }));
  const base = (hasData ? 4.4 : 6.2) * weight;

  // The data: per-layer strengths, normalized so the strongest layer fills the swell.
  const values = hasData ? profile!.map((v) => (v !== null && v !== undefined && Number.isFinite(v) ? v : 0)) : [];
  const scale = Math.max(0.25, ...values.map((v) => Math.abs(v)));
  const n = values.length;
  const sigma = n ? Math.max(0.1, (Math.PI / n) * 0.95) : 1;
  const angles = values.map((_, i) => layerAngle(i, n));

  const outer: [number, number][] = [];
  const inner: [number, number][] = [];
  const outerR: number[] = [];
  const innerR: number[] = [];
  for (let k = 0; k < N; k++) {
    const theta = -Math.PI / 2 + (TAU * k) / N;
    let w = 0;
    for (const o of wobble) w += o.a * Math.sin(o.j * theta + o.p);
    let t = base;
    for (const o of texture) t *= 1 + o.a * Math.sin(o.j * theta + o.p);
    t *= 0.38 + 0.62 * smoothstep(0, 0.55, angularDistance(theta, start));
    let out = R + w + t / 2;
    let inn = R + w - t / 2;
    for (const b of blots) {
      const k2 = Math.exp(-((angularDistance(theta, b.at) / b.width) ** 2));
      if (b.outward) out += b.height * k2;
      else inn -= b.height * 0.7 * k2;
    }
    let pos = 0;
    let neg = 0;
    for (let i = 0; i < n; i++) {
      const v = values[i] / scale;
      if (v === 0) continue;
      const kernel = Math.exp(-((angularDistance(theta, angles[i]) / sigma) ** 2));
      if (v > 0) pos += v * kernel;
      else neg -= v * kernel;
    }
    out += 12 * Math.min(1.15, pos);
    inn -= 9 * Math.min(1.15, neg);
    inn = Math.max(6, inn);
    outerR.push(out);
    innerR.push(inn);
    outer.push([50 + out * Math.cos(theta), 50 + out * Math.sin(theta)]);
    inner.push([50 + inn * Math.cos(theta), 50 + inn * Math.sin(theta)]);
  }
  const parts = [polygon(clockwise(outer)), polygon(clockwise(inner).reverse())];
  const radiusAt = (theta: number, radii: number[]) => {
    const k = Math.round(((((theta + Math.PI / 2) % TAU) + TAU) % TAU) / (TAU / N)) % N;
    return radii[k];
  };

  // Tendrils: at the strongest layers, or where the seed puts them.
  const tendrils: { at: number; outward: boolean; strength: number }[] = [];
  if (hasData) {
    const ranked = values
      .map((v, i) => ({ v: v / scale, i }))
      .filter((x) => Math.abs(x.v) > 0.6)
      .sort((a, b) => Math.abs(b.v) - Math.abs(a.v))
      .slice(0, detail >= 0.5 ? 2 : 1);
    for (const x of ranked) tendrils.push({ at: angles[x.i], outward: x.v > 0, strength: Math.abs(x.v) });
  } else {
    const count = detail >= 0.8 ? 2 : detail >= 0.4 ? 1 : 0;
    for (let i = 0; i < count; i++) tendrils.push({ at: phase(), outward: rand() > 0.25, strength: 0.6 + rand() * 0.4 });
  }
  for (const td of tendrils) {
    const curl = (rand() - 0.5) * 1.1;
    const length = (td.outward ? 9 : 6) + 8 * td.strength * Math.max(0.5, detail);
    const r0 = td.outward ? radiusAt(td.at, outerR) - 1.5 : radiusAt(td.at, innerR) + 1.5;
    const dir = td.outward ? 1 : -1;
    const ux = Math.cos(td.at);
    const uy = Math.sin(td.at);
    const sx = 50 + r0 * ux;
    const sy = 50 + r0 * uy;
    const spine: [number, number][] = [];
    for (let i = 0; i <= 14; i++) {
      const s = i / 14;
      const along = dir * length * s;
      const side = curl * length * s * s;
      spine.push([sx + ux * along - uy * side, sy + uy * along + ux * side]);
    }
    const left: [number, number][] = [];
    const right: [number, number][] = [];
    for (let i = 0; i < spine.length; i++) {
      const [x0, y0] = spine[Math.max(0, i - 1)];
      const [x1, y1] = spine[Math.min(spine.length - 1, i + 1)];
      const len = Math.hypot(x1 - x0, y1 - y0) || 1;
      const nx = -(y1 - y0) / len;
      const ny = (x1 - x0) / len;
      const half = (1.7 * (1 - i / (spine.length - 1)) ** 1.15 + 0.12) * (0.75 + 0.5 * td.strength);
      left.push([spine[i][0] + nx * half, spine[i][1] + ny * half]);
      right.push([spine[i][0] - nx * half, spine[i][1] - ny * half]);
    }
    parts.push(polygon(clockwise([...left, ...right.reverse()])));
  }

  // Droplets: spatter beside the ring on the larger glyphs.
  const droplets = detail >= 0.8 ? 3 : detail >= 0.5 ? 1 : 0;
  for (let i = 0; i < droplets; i++) {
    const at = phase();
    const dist = radiusAt(at, outerR) + 3 + rand() * 7;
    const size = 0.7 + rand() * 1.4;
    const cx = 50 + dist * Math.cos(at);
    const cy = 50 + dist * Math.sin(at);
    const dot: [number, number][] = Array.from({ length: 14 }, (_, j) => {
      const a = (TAU * j) / 14;
      return [cx + size * Math.cos(a), cy + size * (0.85 + 0.15 * Math.sin(a * 2)) * Math.sin(a)];
    });
    parts.push(polygon(clockwise(dot)));
  }

  // Where the data colors bleed: one soft stain per layer that matters.
  const bleeds: Bleed[] = [];
  for (let i = 0; i < n; i++) {
    const v = values[i] / scale;
    if (Math.abs(v) < 0.18) continue;
    const r = v > 0 ? radiusAt(angles[i], outerR) - 2 : radiusAt(angles[i], innerR) + 2;
    bleeds.push({
      x: 50 + r * Math.cos(angles[i]),
      y: 50 + r * Math.sin(angles[i]),
      r: 3 + 8 * Math.abs(v),
      value: Math.max(-1, Math.min(1, v)),
    });
  }
  return { d: parts.join(""), bleeds };
}

/** Per-layer profile of a run: the strongest (by |value|) measured value in each layer. */
export function layerProfile(
  sites: readonly { index: number; layer: number }[],
  value: (index: number) => number | null | undefined,
  nLayers: number,
): number[] {
  const out = new Array<number>(Math.max(0, nLayers)).fill(0);
  for (const site of sites) {
    if (site.layer < 0 || site.layer >= out.length) continue;
    const v = value(site.index);
    if (v === null || v === undefined || !Number.isFinite(v)) continue;
    if (Math.abs(v) > Math.abs(out[site.layer])) out[site.layer] = v;
  }
  return out;
}
