import { test, expect } from "@playwright/test";
import type { SiteBase } from "../src/api/types";
import { gridKey, siteAt, siteGrid } from "../src/lib/sites";

function site(index: number, row: number, col: number, kind: SiteBase["kind"] = "head"): SiteBase {
  return { index, kind, layer: row, head: kind === "head" ? col : null, position: { kind: "all" }, position_key: "all", row, col, label: `L${row} ${kind} ${col}` };
}

test("the heatmap's cell map finds the site a search through every site finds", () => {
  // Two sites in one cell: the first wins, as with siteAt.
  const sites = [site(0, 0, 0), site(1, 0, 2), site(2, 1, 1), site(3, 2, 0), site(4, 1, 1, "attn_out")];
  const grid = siteGrid(sites);
  expect(grid.size).toBe(4);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) expect(grid.get(gridKey(r, c)) ?? null).toBe(siteAt(sites, r, c));
  }

  // A full sweep with masked cells, as a layer × head map with gaps.
  const sweep: SiteBase[] = [];
  for (let r = 0; r < 32; r++) for (let c = 0; c < 16; c++) if ((r * 7 + c * 3) % 5 !== 0) sweep.push(site(sweep.length, r, c));
  const big = siteGrid(sweep);
  expect(big.size).toBe(sweep.length);
  for (let r = 0; r < 33; r++) {
    for (let c = 0; c < 17; c++) expect(big.get(gridKey(r, c)) ?? null).toBe(siteAt(sweep, r, c));
  }
  expect(siteGrid([]).size).toBe(0);
});
