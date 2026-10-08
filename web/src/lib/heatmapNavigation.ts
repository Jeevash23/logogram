export interface Cell { r: number; c: number }

/** Move in a straight line, skipping cells which are masked or absent. */
export function moveCell(current: Cell, key: string, rows: number, cols: number, exists: (r: number, c: number) => boolean): Cell {
  const delta: Record<string, Cell> = {
    ArrowLeft: { r: 0, c: -1 }, ArrowRight: { r: 0, c: 1 },
    ArrowUp: { r: -1, c: 0 }, ArrowDown: { r: 1, c: 0 },
  };
  if (key === "Home" || key === "End") {
    for (let i = 0; i < cols; i++) {
      const c = key === "Home" ? i : cols - 1 - i;
      if (exists(current.r, c)) return { r: current.r, c };
    }
  }
  const step = delta[key];
  if (!step) return current;
  let { r, c } = current;
  while (true) {
    r += step.r; c += step.c;
    if (r < 0 || c < 0 || r >= rows || c >= cols) return current;
    if (exists(r, c)) return { r, c };
  }
}
