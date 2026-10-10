import { memo, useEffect, useId, useMemo, useRef, useState, type KeyboardEvent, type MouseEvent, type ReactNode } from "react";

import { cornerFlag, fitText, font, inkRing, prepareCanvas, useChromeColors, useElementSize } from "../../lib/canvas";
import { textOn, type ResolvedTheme } from "../../lib/color";
import { visibleToken } from "../../lib/format";
import { moveCell, type Cell } from "../../lib/heatmapNavigation";
import { usePrefersReducedMotion } from "../../lib/motion";
import s from "./Heatmap.module.css";

export interface Axis {
  key: string;
  label: string;
  emphasis?: boolean; // e.g. tokens that differ between clean and corrupt
}

/**
 * Props that are functions should keep their identity between renders (useCallback in the
 * caller): the cells are drawn again whenever value, color, flagged or format change.
 */
interface Props {
  rows: Axis[];
  cols: Axis[];
  /** undefined: no cell here (masked); null: cell without a value yet. */
  value: (r: number, c: number) => number | null | undefined;
  color: (v: number) => string;
  theme: ResolvedTheme;
  selected?: { r: number; c: number } | null;
  flagged?: (r: number, c: number) => boolean;
  onSelect?: (r: number, c: number) => void;
  tooltip?: (r: number, c: number) => ReactNode;
  rowTitle?: string;
  colTitle?: string;
  cellMin?: number;
  cellMax?: number;
  tokens?: boolean; // labels are tokens: show whitespace, rotate column labels
  showValues?: boolean;
  format?: (v: number) => string;
  fade?: boolean;
  ariaLabel: string;
  aspect?: number; // cell height / width
}

const twoDecimals = (v: number) => v.toFixed(2);
/** How long a newly arrived value takes to fade in. */
const FADE_MS = 320;

export const Heatmap = memo(function Heatmap({
  rows,
  cols,
  value,
  color,
  theme,
  selected: externalSelected,
  flagged,
  onSelect,
  tooltip,
  rowTitle,
  colTitle,
  cellMin = 10,
  cellMax = 34,
  tokens = false,
  showValues = false,
  format = twoDecimals,
  fade = false,
  ariaLabel,
  aspect = 1,
}: Props) {
  const colors = useChromeColors();
  const reduceMotion = usePrefersReducedMotion();
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Hover and selection are drawn on a second canvas above the cells, so moving the pointer
  // never redraws the whole map.
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const { width } = useElementSize(wrapRef);
  const [hover, setHover] = useState<{ r: number; c: number; x: number; y: number } | null>(null);
  const arrivals = useRef(new Map<string, number>());
  const [tick, setTick] = useState(0);
  const [cursor, setCursor] = useState<Cell | null>(null);
  const [focused, setFocused] = useState(false);
  const [tableOpen, setTableOpen] = useState(false);
  const helpId = useId();
  const valid = (cell: Cell | null | undefined): cell is Cell => !!cell && cell.r < rows.length && cell.c < cols.length && value(cell.r, cell.c) !== undefined;
  // Where keyboard movement starts when nothing is selected: the first cell that exists.
  const first = useMemo(() => {
    for (let r = 0; r < rows.length; r++) {
      for (let c = 0; c < cols.length; c++) if (value(r, c) !== undefined) return { r, c };
    }
    return null;
  }, [rows.length, cols.length, value]);
  const active = valid(cursor) ? cursor : valid(externalSelected) ? externalSelected : first;
  const selected = focused ? active : externalSelected;
  const selectedR = selected ? selected.r : -1;
  const selectedC = selected ? selected.c : -1;
  const hoverR = hover ? hover.r : -1;
  const hoverC = hover ? hover.c : -1;

  const layout = useMemo(() => {
    const measure = document.createElement("canvas").getContext("2d");
    if (!measure || width === 0) return null;
    measure.font = font(colors, 10.5, 500);
    const labelOf = (a: Axis) => (tokens ? visibleToken(a.label) : a.label);
    const rowLabelW = Math.min(
      tokens ? 110 : 90,
      Math.max(16, ...rows.map((r) => measure.measureText(labelOf(r)).width)) + 10,
    );
    const titleW = rowTitle ? 16 : 0;
    const avail = width - rowLabelW - titleW - 8;
    const cell = Math.max(cellMin, Math.min(cellMax, Math.floor(avail / Math.max(1, cols.length))));
    const cellH = Math.max(cellMin, Math.round(cell * aspect));
    const maxColLabel = Math.max(0, ...cols.map((c) => measure.measureText(labelOf(c)).width));
    const rotate = tokens || maxColLabel > cell - 2;
    const headerH = rotate ? Math.min(84, maxColLabel * 0.72 + 14) : 18;
    const left = titleW + rowLabelW;
    const top = headerH + (colTitle ? 16 : 0);
    const gap = cell >= 20 ? 2 : cell >= 14 ? 1 : 0;
    // Room around the grid for the ink ring of a selected edge cell.
    const margin = Math.ceil(cell * 0.25 + 6);
    return {
      cell,
      cellH,
      gap,
      left,
      top,
      rotate,
      rowLabelW,
      titleW,
      width: Math.max(width, left + cols.length * cell + margin),
      height: top + rows.length * cellH + margin,
      labelOf,
    };
  }, [width, rows, cols, colors, tokens, cellMin, cellMax, rowTitle, colTitle, aspect]);

  // Fade in values as they arrive, unless the person asked for less motion.
  useEffect(() => {
    if (!fade || reduceMotion) return;
    const now = performance.now();
    for (let r = 0; r < rows.length; r++) {
      for (let c = 0; c < cols.length; c++) {
        const v = value(r, c);
        const key = `${r}:${c}`;
        if (v !== null && v !== undefined && !arrivals.current.has(key)) arrivals.current.set(key, now);
      }
    }
  }, [fade, reduceMotion, value, rows.length, cols.length]);

  // The cells, labels and titles: drawn again only when the data, layout or theme change.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !layout) return;
    const ctx = prepareCanvas(canvas, layout.width, layout.height);
    if (!ctx) return;
    const { cell, cellH, gap, left, top } = layout;
    const now = performance.now();
    const animate = fade && !reduceMotion;
    let fading = false;

    // Titles.
    ctx.fillStyle = colors.faint;
    ctx.font = font(colors, 10.5, 560);
    if (colTitle) {
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(colTitle, left, 0);
    }
    if (rowTitle) {
      ctx.save();
      ctx.translate(9, top + (rows.length * cellH) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(rowTitle, 0, 0);
      ctx.restore();
    }

    // Column labels.
    ctx.font = font(colors, 10.5, 500);
    cols.forEach((col, c) => {
      const x = left + c * cell + cell / 2;
      const label = layout.labelOf(col);
      ctx.fillStyle = col.emphasis ? colors.text : colors.muted;
      ctx.font = font(colors, 10.5, col.emphasis ? 680 : 500);
      if (layout.rotate) {
        ctx.save();
        ctx.translate(x, top - 6);
        ctx.rotate(-Math.PI / 4);
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(fitText(ctx, label, 100), 0, 0);
        ctx.restore();
      } else {
        ctx.textAlign = "center";
        ctx.textBaseline = "alphabetic";
        ctx.fillText(label, x, top - 6);
      }
    });

    // Row labels and cells.
    rows.forEach((row, r) => {
      const y = top + r * cellH;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.font = font(colors, 10.5, row.emphasis ? 680 : 500);
      ctx.fillStyle = row.emphasis ? colors.text : colors.muted;
      ctx.fillText(fitText(ctx, layout.labelOf(row), layout.rowLabelW - 8), left - 6, y + cellH / 2 + 0.5);
      for (let c = 0; c < cols.length; c++) {
        const v = value(r, c);
        if (v === undefined) continue;
        const x = left + c * cell;
        const w = cell - gap;
        const h = cellH - gap;
        ctx.fillStyle = colors.surface2;
        ctx.fillRect(x, y, w, h);
        if (v === null) {
          ctx.strokeStyle = colors.lineSoft;
          ctx.lineWidth = 1;
          ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
          continue;
        }
        const arrived = animate ? arrivals.current.get(`${r}:${c}`) ?? 0 : 0;
        const alpha = arrived === 0 ? 1 : Math.min(1, (now - arrived) / FADE_MS);
        if (alpha < 1) fading = true;
        ctx.globalAlpha = alpha;
        const fill = color(v);
        ctx.fillStyle = fill;
        ctx.fillRect(x, y, w, h);
        ctx.globalAlpha = 1;
        if (showValues && cell >= 34 && cellH >= 18) {
          ctx.fillStyle = textOn(fill, theme);
          ctx.font = font(colors, 9.5, 540);
          ctx.textAlign = "center";
          ctx.fillText(format(v), x + w / 2, y + h / 2 + 0.5);
        }
        if (flagged?.(r, c)) cornerFlag(ctx, x, y, w, colors.text, colors.surface);
      }
    });

    if (fading) {
      const frame = requestAnimationFrame(() => setTick((t) => t + 1));
      return () => cancelAnimationFrame(frame);
    }
  }, [layout, rows, cols, value, color, colors, flagged, showValues, format, theme, fade, reduceMotion, tick, colTitle, rowTitle]);

  // Hover and selection, above the cells.
  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas || !layout) return;
    const ctx = prepareCanvas(canvas, layout.width, layout.height);
    if (!ctx) return;
    const { cell, cellH, gap, left, top } = layout;
    if (hoverR >= 0 && (hoverR !== selectedR || hoverC !== selectedC)) {
      ctx.strokeStyle = colors.muted;
      ctx.lineWidth = 1;
      ctx.strokeRect(left + hoverC * cell - 1.5, top + hoverR * cellH - 1.5, cell - gap + 3, cellH - gap + 3);
    }
    if (selectedR >= 0 && selectedR < rows.length && selectedC >= 0 && selectedC < cols.length) {
      inkRing(ctx, left + selectedC * cell, top + selectedR * cellH, cell - gap, cellH - gap, colors.text, colors.surface);
    }
  }, [layout, colors, hoverR, hoverC, selectedR, selectedC, rows.length, cols.length]);

  const hit = (e: MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas || !layout) return null;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left - layout.left;
    const y = e.clientY - rect.top - layout.top;
    const c = Math.floor(x / layout.cell);
    const r = Math.floor(y / layout.cellH);
    if (r < 0 || c < 0 || r >= rows.length || c >= cols.length) return null;
    if (value(r, c) === undefined) return null;
    return { r, c };
  };

  const onKeyDown = (e: KeyboardEvent) => {
    if (!active || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(e.key)) return;
    e.preventDefault();
    const next = moveCell(active, e.key, rows.length, cols.length, (r, c) => value(r, c) !== undefined);
    setCursor(next);
    onSelect?.(next.r, next.c);
    const scroll = canvasRef.current?.parentElement?.parentElement;
    if (scroll && layout) {
      const x = layout.left + next.c * layout.cell, y = layout.top + next.r * layout.cellH;
      const left = Math.max(x + layout.cell - scroll.clientWidth, Math.min(scroll.scrollLeft, x));
      const top = Math.max(y + layout.cellH - scroll.clientHeight, Math.min(scroll.scrollTop, y));
      scroll.scrollTo({ left: Math.max(0, left), top: Math.max(0, top) });
    }
  };

  const activeValue = active ? value(active.r, active.c) : undefined;
  return (
    <div className={s.wrap} ref={wrapRef}>
      <div className={s.scroll}>
        <div className={s.stage}>
          <canvas
            ref={canvasRef}
            className={s.canvas}
            tabIndex={first ? 0 : -1}
            role="group"
            aria-roledescription="interactive heatmap"
            aria-label={ariaLabel}
            aria-describedby={helpId}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            onMouseMove={(e) => {
              const h = hit(e);
              setHover(h ? { ...h, x: e.clientX, y: e.clientY } : null);
            }}
            onMouseLeave={() => setHover(null)}
            onClick={(e) => {
              const h = hit(e);
              if (h) { setCursor(h); onSelect?.(h.r, h.c); }
            }}
            onKeyDown={onKeyDown}
            style={{ cursor: onSelect ? "pointer" : "default" }}
          />
          <canvas ref={overlayRef} className={s.overlay} aria-hidden="true" />
        </div>
      </div>
      <p id={helpId} className={s.help}>Arrow keys inspect cells. Home and End move to the first and last cell in a row.</p>
      <div className={s.keyboardReadout} aria-live="polite" aria-atomic="true">
        {focused && active && <>{rowTitle ?? "Row"} {active.r}: {rows[active.r]?.label}; {colTitle ?? "Column"} {active.c}: {cols[active.c]?.label}. {activeValue === null ? "Waiting for a value" : `Value ${activeValue?.toPrecision(6)}`}</>}
      </div>
      <details onToggle={(e) => setTableOpen(e.currentTarget.open)} className={s.table}>
        <summary>View values as a table</summary>
        {tableOpen && <div className={s.scroll}><table>
          <caption>{ariaLabel}</caption>
          <thead><tr><th scope="col">{rowTitle ?? "Row"} / {colTitle ?? "Column"}</th>{cols.map((c, i) => <th scope="col" key={c.key}>{i}: {c.label}</th>)}</tr></thead>
          <tbody>{rows.map((row, r) => <tr key={row.key}><th scope="row">{r}: {row.label}</th>{cols.map((col, c) => {
            const v = value(r, c);
            return <td key={col.key}>{v === undefined ? "Masked" : v === null ? "Pending" : v.toPrecision(6)}</td>;
          })}</tr>)}</tbody>
        </table></div>}
      </details>
      {hover && tooltip && (
        <div className={s.readout} style={{ left: hover.x + 14, top: hover.y + 14 }}>
          {tooltip(hover.r, hover.c)}
        </div>
      )}
    </div>
  );
});
