import { useRef, type KeyboardEvent, type PointerEvent } from "react";

import s from "./Splitter.module.css";

/** A draggable divider. ``onResize`` receives the pointer delta in pixels since the drag began.
 * ``value``, ``min`` and ``max`` are the size it sets, in pixels, which assistive technology reads. */
export function Splitter({
  orientation,
  onResize,
  onReset,
  label,
  value,
  min,
  max,
}: {
  orientation: "vertical" | "horizontal";
  onResize: (delta: number, phase: "start" | "move" | "end") => void;
  onReset?: () => void;
  label: string;
  value: number;
  min: number;
  max: number;
}) {
  const start = useRef<number | null>(null);

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    start.current = orientation === "vertical" ? e.clientX : e.clientY;
    onResize(0, "start");
    document.body.style.cursor = orientation === "vertical" ? "col-resize" : "row-resize";
  };

  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    if (start.current === null) return;
    const pos = orientation === "vertical" ? e.clientX : e.clientY;
    onResize(pos - start.current, "move");
  };

  const end = (e: PointerEvent<HTMLDivElement>) => {
    if (start.current === null) return;
    const pos = orientation === "vertical" ? e.clientX : e.clientY;
    onResize(pos - start.current, "end");
    start.current = null;
    document.body.style.cursor = "";
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? 40 : 12;
    const keys = orientation === "vertical" ? ["ArrowLeft", "ArrowRight"] : ["ArrowUp", "ArrowDown"];
    if (!keys.includes(e.key)) return;
    e.preventDefault();
    const delta = e.key === keys[0] ? -step : step;
    onResize(0, "start");
    onResize(delta, "end");
  };

  return (
    <div
      className={orientation === "vertical" ? s.vertical : s.horizontal}
      role="separator"
      aria-orientation={orientation}
      aria-label={label}
      aria-valuenow={Math.round(value)}
      aria-valuemin={min}
      aria-valuemax={max}
      tabIndex={0}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={end}
      onPointerCancel={end}
      onDoubleClick={onReset}
      onKeyDown={onKeyDown}
    />
  );
}
