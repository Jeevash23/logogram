import { memo, useEffect, useId, useMemo, useRef, useState, type KeyboardEvent, type MouseEvent } from "react";

import type { SweepSite } from "../../api/types";
import { fitText, font, inkRing, prepareCanvas, useChromeColors, useElementSize } from "../../lib/canvas";
import type { ColorScale } from "../../lib/color";
import s from "./SweepChart.module.css";

/** A value of the sweep on the chart's axis: its label, and what became of its run when it has no
 * results (failed, queued, not run …). */
export interface ChartValue {
  label: string;
  note: string | null;
}

export interface ChartCell {
  site: number;
  value: number;
}

interface Props {
  values: ChartValue[];
  sites: SweepSite[];
  /** Revisions over training: the means are joined in the order given, and no band is drawn. */
  ordered: boolean;
  /** The symmetric bound of the vertical scale and of the effect colors. */
  bound: number;
  color: ColorScale;
  selected: ChartCell | null;
  onSelect: (cell: ChartCell) => void;
  /** Enter, or a double click: open the value's run at the site. */
  onOpen: (cell: ChartCell) => void;
  /** What a cell holds, in words: read out as the keyboard moves, and shown on hover. */
  describe: (cell: ChartCell) => string;
  ariaLabel: string;
}

const TRACK = 64;
const VERDICT_W = 176;
const VERDICTS: Record<SweepSite["verdict"], string> = {
  holds: "keeps sign and size",
  size: "changes size",
  sign: "flips sign",
  unknown: "too few values",
};

/** Where a site's intervals exclude zero, under its verdict. */
function zeroText(site: SweepSite): string {
  const confident = site.above.length + site.below.length;
  if (site.measured === 0) return "";
  if (site.excludes_zero === "every") return "excludes zero at every value";
  if (site.excludes_zero === "none") return "includes zero at every value";
  return `excludes zero at ${confident} of ${site.measured}`;
}

/**
 * A sweep's strongest sites across its values, drawn on canvas: one track per site, the values in
 * the order given from left to right. Each run's mean is a dot in the effect colors with that
 * run's interval as a bar; behind them, the spread of the means across values (their mean ± 1 SD)
 * as a faint band, or for revisions a line joining the means. A value without results draws
 * nothing but a dashed mark: it is missing, never zero.
 */
export const SweepChart = memo(function SweepChart({ values, sites, ordered, bound, color, selected, onSelect, onOpen, describe, ariaLabel }: Props) {
  const colors = useChromeColors();
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const { width } = useElementSize(wrapRef);
  const [hover, setHover] = useState<(ChartCell & { x: number; y: number }) | null>(null);
  const [focused, setFocused] = useState(false);
  const helpId = useId();
  const valid = (cell: ChartCell | null): cell is ChartCell =>
    !!cell && cell.site >= 0 && cell.site < sites.length && cell.value >= 0 && cell.value < values.length;
  const active: ChartCell | null = valid(selected) ? selected : sites.length && values.length ? { site: 0, value: 0 } : null;

  const layout = useMemo(() => {
    const measure = document.createElement("canvas").getContext("2d");
    if (!measure || width === 0) return null;
    measure.font = font(colors, 10.5, 500);
    const labelW = Math.min(170, Math.max(48, ...sites.map((x) => measure.measureText(x.label).width)) + 16);
    const n = Math.max(1, values.length);
    const avail = width - labelW - VERDICT_W - 8;
    const col = Math.max(22, Math.min(120, Math.floor(avail / n)));
    const widest = Math.max(0, ...values.map((v) => measure.measureText(v.label).width));
    const rotate = widest > col - 6;
    const notes = values.some((v) => v.note);
    const header = rotate ? Math.min(110, (widest + (notes ? 60 : 0)) * 0.72 + 18) : notes ? 34 : 22;
    const top = header + 8;
    const plotW = n * col;
    return {
      labelW,
      col,
      rotate,
      top,
      left: labelW,
      plotW,
      width: Math.max(width, labelW + plotW + VERDICT_W + 8),
      height: top + sites.length * TRACK + 10,
    };
  }, [width, sites, values, colors]);

  /** The vertical position of a value in a track, clamped to the track. */
  const yOf = (row: number, v: number, top: number) =>
    top + row * TRACK + TRACK / 2 - Math.max(-1, Math.min(1, v / bound)) * (TRACK / 2 - 8);

  // The tracks: drawn again only when the data, the layout or the theme change.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !layout) return;
    const ctx = prepareCanvas(canvas, layout.width, layout.height);
    if (!ctx) return;
    const { col, left, top, plotW } = layout;
    const xOf = (i: number) => left + (i + 0.5) * col;

    // The values along the top, with what became of a run that has no results.
    values.forEach((v, i) => {
      const x = xOf(i);
      ctx.fillStyle = v.note ? colors.faint : colors.muted;
      ctx.font = font(colors, 10.5, v.note ? 500 : 560);
      if (layout.rotate) {
        ctx.save();
        ctx.translate(x, top - 6);
        ctx.rotate(-Math.PI / 4);
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(fitText(ctx, v.note ? `${v.label} · ${v.note}` : v.label, 140), 0, 0);
        ctx.restore();
      } else {
        ctx.textAlign = "center";
        ctx.textBaseline = "alphabetic";
        ctx.fillText(fitText(ctx, v.label, col - 4), x, v.note ? top - 20 : top - 8);
        if (v.note) {
          ctx.font = font(colors, 9.5, 500);
          ctx.fillText(fitText(ctx, v.note, col - 2), x, top - 7);
        }
      }
    });

    sites.forEach((site, r) => {
      const y0 = top + r * TRACK;
      const yZero = yOf(r, 0, top);
      if (r > 0) {
        ctx.strokeStyle = colors.lineSoft;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, y0 + 0.5);
        ctx.lineTo(layout.width, y0 + 0.5);
        ctx.stroke();
      }
      // The site, and whether it keeps its sign and size.
      ctx.textBaseline = "middle";
      ctx.textAlign = "right";
      ctx.font = font(colors, 10.5, 540);
      ctx.fillStyle = colors.text;
      ctx.fillText(fitText(ctx, site.label, layout.labelW - 12), left - 10, y0 + TRACK / 2);
      ctx.textAlign = "left";
      const vx = left + plotW + 14;
      const strong = site.verdict === "sign" || site.verdict === "size";
      ctx.font = font(colors, 10.5, strong ? 680 : 520);
      ctx.fillStyle = strong ? colors.text : site.verdict === "unknown" ? colors.faint : colors.muted;
      ctx.fillText(fitText(ctx, VERDICTS[site.verdict], VERDICT_W - 16), vx, y0 + TRACK / 2 - 7);
      ctx.font = font(colors, 9.5, 500);
      ctx.fillStyle = colors.faint;
      ctx.fillText(fitText(ctx, zeroText(site), VERDICT_W - 16), vx, y0 + TRACK / 2 + 8);

      // Zero, dashed.
      ctx.save();
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = colors.faint;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(left, Math.round(yZero) + 0.5);
      ctx.lineTo(left + plotW, Math.round(yZero) + 0.5);
      ctx.stroke();
      ctx.restore();

      // The spread across values: the mean of the means ± 1 SD, faintly, and their mean.
      if (!ordered && site.spread && site.measured >= 2) {
        const sd = site.spread.sd ?? 0;
        const hi = yOf(r, site.spread.mean + sd, top);
        const lo = yOf(r, site.spread.mean - sd, top);
        ctx.fillStyle = colors.band;
        ctx.fillRect(left, hi, plotW, Math.max(1, lo - hi));
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        const ym = Math.round(yOf(r, site.spread.mean, top)) + 0.5;
        ctx.moveTo(left, ym);
        ctx.lineTo(left + plotW, ym);
        ctx.stroke();
      }

      // Revisions: the means joined in the order given, broken where a value has no results.
      if (ordered) {
        ctx.strokeStyle = colors.faint;
        ctx.lineWidth = 1.25;
        ctx.beginPath();
        let drawing = false;
        site.points.forEach((p, i) => {
          if (!p || p.mean === null) {
            drawing = false;
            return;
          }
          const y = yOf(r, p.mean, top);
          if (drawing) ctx.lineTo(xOf(i), y);
          else ctx.moveTo(xOf(i), y);
          drawing = true;
        });
        ctx.stroke();
      }

      site.points.forEach((p, i) => {
        const x = xOf(i);
        if (!p || p.mean === null) {
          // Missing, never zero: a dashed mark where the value would be.
          ctx.save();
          ctx.setLineDash([2, 3]);
          ctx.strokeStyle = colors.faint;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(Math.round(x) + 0.5, y0 + 12);
          ctx.lineTo(Math.round(x) + 0.5, y0 + TRACK - 12);
          ctx.stroke();
          ctx.restore();
          return;
        }
        // The run's own interval, as a bar; an end beyond the scale is drawn open.
        if (p.lo !== null && p.hi !== null) {
          const yHi = yOf(r, p.hi, top);
          const yLo = yOf(r, p.lo, top);
          ctx.strokeStyle = colors.muted;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(x, yHi);
          ctx.lineTo(x, yLo);
          for (const [end, beyond] of [[yHi, p.hi > bound], [yLo, p.lo < -bound]] as const) {
            if (beyond) continue;
            ctx.moveTo(x - 3.5, end);
            ctx.lineTo(x + 3.5, end);
          }
          ctx.stroke();
        }
        const y = yOf(r, p.mean, top);
        const strength = Math.max(0.6, Math.min(1, Math.abs(p.mean) / bound));
        ctx.fillStyle = p.mean === 0 ? colors.muted : color(Math.sign(p.mean) * strength * bound);
        ctx.strokeStyle = colors.bg;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      });
    });
  }, [layout, values, sites, ordered, bound, color, colors]);

  // Hover and the selected cell, above the tracks.
  const hoverSite = hover ? hover.site : -1;
  const hoverValue = hover ? hover.value : -1;
  const ringSite = focused ? active?.site ?? -1 : valid(selected) ? selected.site : -1;
  const ringValue = focused ? active?.value ?? -1 : valid(selected) ? selected.value : -1;
  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas || !layout) return;
    const ctx = prepareCanvas(canvas, layout.width, layout.height);
    if (!ctx) return;
    const { col, left, top } = layout;
    if (hoverSite >= 0 && (hoverSite !== ringSite || hoverValue !== ringValue)) {
      ctx.strokeStyle = colors.muted;
      ctx.lineWidth = 1;
      ctx.strokeRect(left + hoverValue * col + 2.5, top + hoverSite * TRACK + 3.5, col - 5, TRACK - 7);
    }
    if (ringSite >= 0 && ringValue >= 0) {
      inkRing(ctx, left + ringValue * col + 4, top + ringSite * TRACK + 6, col - 8, TRACK - 12, colors.text, colors.surface);
    }
  }, [layout, colors, hoverSite, hoverValue, ringSite, ringValue]);

  const hit = (e: MouseEvent): ChartCell | null => {
    const canvas = canvasRef.current;
    if (!canvas || !layout) return null;
    const rect = canvas.getBoundingClientRect();
    const value = Math.floor((e.clientX - rect.left - layout.left) / layout.col);
    const site = Math.floor((e.clientY - rect.top - layout.top) / TRACK);
    const cell = { site, value };
    return valid(cell) ? cell : null;
  };

  const onKeyDown = (e: KeyboardEvent) => {
    if (!active) return;
    let next: ChartCell | null = null;
    if (e.key === "ArrowLeft") next = { ...active, value: Math.max(0, active.value - 1) };
    else if (e.key === "ArrowRight") next = { ...active, value: Math.min(values.length - 1, active.value + 1) };
    else if (e.key === "ArrowUp") next = { ...active, site: Math.max(0, active.site - 1) };
    else if (e.key === "ArrowDown") next = { ...active, site: Math.min(sites.length - 1, active.site + 1) };
    else if (e.key === "Home") next = { ...active, value: 0 };
    else if (e.key === "End") next = { ...active, value: values.length - 1 };
    else if (e.key === "Enter") {
      e.preventDefault();
      onOpen(active);
      return;
    }
    if (!next) return;
    e.preventDefault();
    onSelect(next);
    // Keep the cell in view when the chart scrolls.
    const scroll = canvasRef.current?.parentElement?.parentElement;
    if (scroll && layout) {
      const x = layout.left + next.value * layout.col;
      const left = Math.max(x + layout.col + 16 - scroll.clientWidth, Math.min(scroll.scrollLeft, x - layout.left));
      scroll.scrollTo({ left: Math.max(0, left) });
    }
  };

  return (
    <div className={s.wrap} ref={wrapRef}>
      <div className={s.scroll}>
        <div className={s.stage}>
          <canvas
            ref={canvasRef}
            className={s.canvas}
            tabIndex={active ? 0 : -1}
            role="group"
            aria-roledescription="interactive chart"
            aria-label={ariaLabel}
            aria-describedby={helpId}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            onMouseMove={(e) => {
              const cell = hit(e);
              setHover(cell ? { ...cell, x: e.clientX, y: e.clientY } : null);
            }}
            onMouseLeave={() => setHover(null)}
            onClick={(e) => {
              const cell = hit(e);
              if (cell) onSelect(cell);
            }}
            onDoubleClick={(e) => {
              const cell = hit(e);
              if (cell) onOpen(cell);
            }}
            onKeyDown={onKeyDown}
          />
          <canvas ref={overlayRef} className={s.overlay} aria-hidden="true" />
        </div>
      </div>
      <p id={helpId} className={s.help}>
        Arrow keys move between sites (up and down) and values (left and right). Enter opens that value's run at the site.
      </p>
      <div className={s.readout} aria-live="polite" aria-atomic="true">
        {focused && active ? describe(active) : ""}
      </div>
      {hover && (
        <div className={s.tooltip} style={{ left: hover.x + 14, top: hover.y + 14 }}>
          {describe(hover)}
        </div>
      )}
    </div>
  );
});
