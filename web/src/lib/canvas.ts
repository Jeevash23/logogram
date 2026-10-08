import { useEffect, useLayoutEffect, useState, type RefObject } from "react";

import { logogram, RING_RADIUS } from "./logogram";
import { useStore } from "../store/app";

export interface ChromeColors {
  bg: string;
  surface: string;
  surface2: string;
  line: string;
  lineSoft: string;
  text: string;
  muted: string;
  faint: string;
  font: string;
}

function readColors(): ChromeColors {
  const css = getComputedStyle(document.documentElement);
  const v = (name: string) => css.getPropertyValue(name).trim();
  return {
    bg: v("--bg"),
    surface: v("--surface"),
    surface2: v("--surface-2"),
    line: v("--line"),
    lineSoft: v("--line-soft"),
    text: v("--text"),
    muted: v("--muted"),
    faint: v("--faint"),
    font: v("--font"),
  };
}

/** Chrome colors from the CSS tokens, re-read when the theme changes. */
export function useChromeColors(): ChromeColors {
  const theme = useStore((s) => s.theme);
  const [colors, setColors] = useState<ChromeColors>(() => readColors());
  useLayoutEffect(() => {
    setColors(readColors());
  }, [theme]);
  return colors;
}

export function useElementSize(ref: RefObject<HTMLElement | null>): { width: number; height: number } {
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const update = () => setSize({ width: el.clientWidth, height: el.clientHeight });
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, [ref]);
  return size;
}

/** Size a canvas for the device pixel ratio and return a context in CSS pixels. */
export function prepareCanvas(
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
): CanvasRenderingContext2D | null {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(width * dpr));
  const h = Math.max(1, Math.round(height * dpr));
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return ctx;
}

export function font(colors: ChromeColors, size: number, weight = 500): string {
  return `${weight} ${size}px ${colors.font || "sans-serif"}`;
}

/** Draw text truncated with an ellipsis to fit ``maxWidth``. */
export function fitText(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string {
  if (ctx.measureText(text).width <= maxWidth) return text;
  let lo = 0;
  let hi = text.length;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (ctx.measureText(text.slice(0, mid) + "…").width <= maxWidth) lo = mid;
    else hi = mid - 1;
  }
  return lo > 0 ? text.slice(0, lo) + "…" : "";
}

/** Outline ring around a rect, outside it, with a gap in the surface color. */
export function ring(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  color: string,
  gapColor: string,
  width = 2,
): void {
  ctx.save();
  ctx.lineWidth = 1;
  ctx.strokeStyle = gapColor;
  ctx.strokeRect(x - 0.5, y - 0.5, w + 1, h + 1);
  ctx.lineWidth = width;
  ctx.strokeStyle = color;
  ctx.strokeRect(x - 1 - width / 2, y - 1 - width / 2, w + 2 + width, h + 2 + width);
  ctx.restore();
}

export function cornerFlag(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  color: string,
  backing: string,
): void {
  const size = Math.max(4, Math.min(8, w * 0.38));
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(x + w - size - 1, y);
  ctx.lineTo(x + w, y);
  ctx.lineTo(x + w, y + size + 1);
  ctx.closePath();
  ctx.fillStyle = backing;
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(x + w - size, y);
  ctx.lineTo(x + w, y);
  ctx.lineTo(x + w, y + size);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  ctx.restore();
}

const inkRings = new Map<string, Path2D>();

/**
 * Circle a cell in ink, like marking a printout: a thin hand-drawn ring around the rectangle,
 * cut out of its surroundings by a band of ``gapColor`` so it reads over any data color.
 */
export function inkRing(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  color: string,
  gapColor: string,
  seed = "selection",
): void {
  let path = inkRings.get(seed);
  if (!path) {
    path = new Path2D(logogram({ seed, detail: 0, weight: 0.4 }).d);
    inkRings.set(seed, path);
  }
  // Clear the cell's corners: the ring's inner edge sits outside the half-diagonal.
  const rx = (w / 2) * 1.42 + 2.5;
  const ry = (h / 2) * 1.42 + 2.5;
  const sx = rx / RING_RADIUS;
  const sy = ry / RING_RADIUS;
  ctx.save();
  ctx.translate(x + w / 2, y + h / 2);
  ctx.scale(sx, sy);
  ctx.translate(-50, -50);
  ctx.lineJoin = "round";
  ctx.lineWidth = 2.6 / Math.min(sx, sy);
  ctx.strokeStyle = gapColor;
  ctx.stroke(path);
  ctx.fillStyle = color;
  ctx.fill(path, "nonzero");
  ctx.restore();
}
