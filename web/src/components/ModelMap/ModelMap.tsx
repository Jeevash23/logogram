import * as ContextMenu from "@radix-ui/react-context-menu";
import { useCallback, useEffect, useId, useMemo, useRef, useState, type KeyboardEvent, type MouseEvent } from "react";

import type { SiteBase, SiteResult } from "../../api/types";
import { cornerFlag, font, inkRing, prepareCanvas, useChromeColors, useElementSize } from "../../lib/canvas";
import { divergingScale, niceBound, SCALE_FLOOR } from "../../lib/color";
import { ci, signed } from "../../lib/format";
import { siteValue, useActiveRun, useArchitecture, modelName } from "../../lib/hooks";
import { measureWords } from "../../lib/spec";
import { usePrefersReducedMotion } from "../../lib/motion";
import {
  componentLabel,
  isResidKind,
  nextSelection,
  partOfKind,
  type MapPart,
  type Selection,
} from "../../lib/sites";
import { useStore } from "../../store/app";
import { Button, menuClasses } from "../ui";
import { Legend } from "./Legend";
import s from "./ModelMap.module.css";

interface Column {
  part: MapPart;
  head?: number;
  x: number;
}

interface Geometry {
  cell: number;
  pitch: number;
  columns: Column[];
  labelX: number;
  top: number;
  headerH: number;
  width: number;
  height: number;
}

interface CellData {
  value: number | null;
  site: SiteResult | null;
  base: SiteBase | null;
  pending: boolean;
  note?: string;
}

const PAD_X = 12;
const LABEL_W = 22;
const GROUP_GAP = 7;

function computeGeometry(width: number, nLayers: number, nHeads: number, prominent: boolean): Geometry {
  const nCols = nHeads + 3;
  const avail = Math.max(0, width - PAD_X * 2 - LABEL_W - GROUP_GAP * 2);
  const pitch = Math.max(7, Math.min(prominent ? 44 : 30, Math.floor(avail / nCols)));
  const gap = pitch >= 14 ? 2 : 1;
  const cell = pitch - gap;
  const gridW = LABEL_W + nCols * pitch + GROUP_GAP * 2 - gap;
  const left = Math.max(PAD_X, Math.floor((width - gridW) / 2));
  const columns: Column[] = [];
  let x = left + LABEL_W;
  columns.push({ part: "resid", x });
  x += pitch + GROUP_GAP;
  for (let h = 0; h < nHeads; h++) {
    columns.push({ part: "head", head: h, x });
    x += pitch;
  }
  x += GROUP_GAP;
  columns.push({ part: "attn", x });
  x += pitch;
  columns.push({ part: "mlp", x });
  const headerH = 20;
  const top = 6;
  return {
    cell,
    pitch,
    columns,
    labelX: left + LABEL_W - 7,
    top,
    headerH,
    width: Math.max(width, left + gridW + PAD_X),
    height: top + headerH + nLayers * pitch + Math.ceil(pitch * 0.3 + 6),
  };
}

function cellKey(layer: number, part: MapPart, head?: number): string {
  return part === "head" ? `${layer}:h${head}` : `${layer}:${part}`;
}

function siteCellKey(site: SiteBase): string {
  return cellKey(site.layer, partOfKind(site.kind), site.head ?? undefined);
}

export function ModelMap({ prominent = false, structureOnly = false }: { prominent?: boolean; structureOnly?: boolean }) {
  const arch = useArchitecture();
  const run = useActiveRun();
  const selection = useStore((st) => st.selection);
  const select = useStore((st) => st.select);
  const metric = useStore((st) => st.mapMetric);
  const scaleMode = useStore((st) => st.scaleMode);
  const theme = useStore((st) => st.theme);
  const flags = useStore((st) => (run.id ? st.flags[run.id] : undefined));
  const colors = useChromeColors();
  const modelState = useStore((st) => st.model.state);
  const loadingId = useStore((st) => st.model.id);

  const scrollRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Hover and selection are drawn on a second canvas above the cells, so moving the pointer or
  // the selection never redraws the whole map.
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const reduceMotion = usePrefersReducedMotion();
  const helpId = useId();
  const { width } = useElementSize(scrollRef);
  const [hover, setHover] = useState<{ sel: Selection; x: number; y: number } | null>(null);
  const [menuSel, setMenuSel] = useState<Selection | null>(null);
  const arrivals = useRef(new Map<number, number>());
  const lastRun = useRef<string | null>(null);
  const [tick, setTick] = useState(0);

  const geo = useMemo(
    () => (arch && width > 0 ? computeGeometry(width, arch.nLayers, arch.nHeads, prominent) : null),
    [arch, width, prominent],
  );

  // Values per map cell. A cell can hold several sites: positions of a layer × position run, or
  // the residual stream before, between and after a layer's blocks. Show the ones the selection
  // names, else the strongest, and say which in the readout.
  const wantedKind = selection?.part === "resid" ? selection.kind : undefined;
  const wantedPosition = selection?.positionKey;
  const cells = useMemo(() => {
    const byCell = new Map<string, SiteBase[]>();
    for (const site of run.sites) {
      const key = siteCellKey(site);
      const list = byCell.get(key);
      if (list) list.push(site);
      else byCell.set(key, [site]);
    }
    const out = new Map<string, CellData>();
    for (const [key, sites] of byCell) {
      if (sites.length === 1) {
        const r = run.results[sites[0].index];
        out.set(key, { value: siteValue(r, metric), site: r ?? null, base: sites[0], pending: !r });
        continue;
      }
      let pool = sites;
      if (wantedKind && pool.some((x) => x.kind === wantedKind)) pool = pool.filter((x) => x.kind === wantedKind);
      if (wantedPosition !== undefined) {
        pool = pool.filter((x) => x.position_key === wantedPosition);
        if (pool.length === 0) {
          out.set(key, { value: null, site: null, base: null, pending: false, note: `not measured at position ${wantedPosition}` });
          continue;
        }
      }
      let best: SiteResult | null = null;
      let bestBase: SiteBase | null = null;
      for (const b of pool) {
        const r = run.results[b.index];
        const v = siteValue(r, metric);
        if (v !== null && (best === null || Math.abs(v) > Math.abs(siteValue(best, metric) ?? 0))) {
          best = r;
          bestBase = b;
        }
      }
      if (pool.length === 1) bestBase = pool[0];
      out.set(key, {
        value: siteValue(best ?? undefined, metric),
        site: best,
        base: bestBase,
        pending: pool.some((b) => !run.results[b.index]),
        note: pool.length > 1 && bestBase ? `strongest of ${pool.length} here` : undefined,
      });
    }
    return out;
  }, [run.sites, run.results, metric, wantedKind, wantedPosition]);

  const bound = useMemo(() => {
    if (scaleMode === "unit" && metric === "effect") return 1;
    return niceBound(Object.values(run.results).map((r) => siteValue(r, metric)), SCALE_FLOOR[metric]);
  }, [run.results, metric, scaleMode]);
  const color = useMemo(() => divergingScale(bound, theme), [bound, theme]);
  const flagged = useMemo(() => {
    const set = new Set<string>();
    if (!flags) return set;
    const byIndex = new Map(run.sites.map((x) => [x.index, x]));
    for (const idx of flags.sites) {
      const base = byIndex.get(idx);
      if (base) set.add(siteCellKey(base));
    }
    return set;
  }, [flags, run.sites]);

  // Track when streamed results arrive, so new cells fade in.
  useEffect(() => {
    if (lastRun.current !== run.id) {
      arrivals.current.clear();
      lastRun.current = run.id;
      for (const k of Object.keys(run.results)) arrivals.current.set(Number(k), 0);
      return;
    }
    const now = performance.now();
    for (const k of Object.keys(run.results)) {
      const idx = Number(k);
      // With reduced motion a new cell shows at once instead of fading in.
      if (!arrivals.current.has(idx)) arrivals.current.set(idx, run.running && !reduceMotion ? now : 0);
    }
  }, [run.id, run.results, run.running, reduceMotion]);

  const runningLayer = run.live?.status === "running" ? run.live.progress?.layer ?? null : null;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !geo || !arch) return;
    const ctx = prepareCanvas(canvas, geo.width, geo.height);
    if (!ctx) return;
    const { cell, pitch, columns, top, headerH } = geo;
    const now = performance.now();
    let fading = false;

    // Column headers.
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "center";
    ctx.fillStyle = colors.faint;
    const every = cell >= 16 ? 1 : cell >= 10 ? 2 : 4;
    for (const col of columns) {
      let label = "";
      if (col.part === "resid") label = cell >= 15 ? "res" : "r";
      else if (col.part === "attn") label = cell >= 15 ? "attn" : "a";
      else if (col.part === "mlp") label = cell >= 15 ? "mlp" : "m";
      else if (col.head !== undefined && col.head % every === 0) label = String(col.head);
      // Word labels may be wider than a cell; a smaller size keeps neighbours apart.
      ctx.font = font(colors, label.length > 2 ? 9 : 10, 520);
      if (label) ctx.fillText(label, col.x + cell / 2, top + headerH - 7);
    }

    // Rows.
    for (let layer = 0; layer < arch.nLayers; layer++) {
      const y = top + headerH + layer * pitch;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.font = font(colors, 10, layer === runningLayer ? 650 : 500);
      ctx.fillStyle = layer === runningLayer ? colors.text : colors.faint;
      ctx.fillText(String(layer), geo.labelX, y + cell / 2 + 0.5);
      if (layer === runningLayer) {
        ctx.fillStyle = colors.text;
        ctx.fillRect(geo.labelX - 18, y + cell / 2 - 1, 3, 3);
      }
      for (const col of columns) {
        const key = cellKey(layer, col.part, col.head);
        const data = cells.get(key);
        // Measured cells are filled (near zero is the scale's pale neutral); unmeasured ones are
        // empty outlines, so "nothing here" never looks like "no effect".
        if (structureOnly || !data) ctx.fillStyle = colors.surface;
        else ctx.fillStyle = colors.surface2;
        ctx.fillRect(col.x, y, cell, cell);
        if (!structureOnly && data && data.value !== null) {
          const arrived = data.site && !reduceMotion ? arrivals.current.get(data.site.index) ?? 0 : 0;
          const alpha = arrived === 0 ? 1 : Math.min(1, (now - arrived) / 320);
          if (alpha < 1) fading = true;
          ctx.globalAlpha = alpha;
          ctx.fillStyle = color(data.value);
          ctx.fillRect(col.x, y, cell, cell);
          ctx.globalAlpha = 1;
        } else {
          ctx.strokeStyle = data?.pending ? colors.line : colors.lineSoft;
          ctx.lineWidth = 1;
          ctx.strokeRect(col.x + 0.5, y + 0.5, cell - 1, cell - 1);
        }
        if (flagged.has(key)) cornerFlag(ctx, col.x, y, cell, colors.text, colors.surface);
      }
    }

    // Keep drawing until every newly arrived cell has fully faded in.
    if (fading) {
      const frame = requestAnimationFrame(() => setTick((t) => t + 1));
      return () => cancelAnimationFrame(frame);
    }
  }, [geo, arch, cells, color, colors, flagged, runningLayer, tick, structureOnly, reduceMotion]);

  // Hover and selection, above the cells. Only the component counts, not a position on it, so
  // the overlay is drawn again only when the pointer or the selection moves to another cell.
  const hoverLayer = hover?.sel.layer ?? -1;
  const hoverPart = hover?.sel.part ?? null;
  const hoverHead = hover?.sel.head ?? -1;
  const selectedLayer = selection?.layer ?? -1;
  const selectedPart = selection?.part ?? null;
  const selectedHead = selection?.head ?? -1;
  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas || !geo || !arch) return;
    const ctx = prepareCanvas(canvas, geo.width, geo.height);
    if (!ctx) return;
    const { cell, pitch, columns, top, headerH } = geo;
    const place = (layer: number, part: MapPart | null, head: number) => {
      const col = part === null ? undefined : columns.find((c) => c.part === part && (part !== "head" || c.head === head));
      return col && layer >= 0 && layer < arch.nLayers ? { x: col.x, y: top + headerH + layer * pitch } : null;
    };
    const hovered = place(hoverLayer, hoverPart, hoverHead);
    const chosen = place(selectedLayer, selectedPart, selectedHead);
    if (hovered && (hovered.x !== chosen?.x || hovered.y !== chosen?.y)) {
      ctx.strokeStyle = colors.muted;
      ctx.lineWidth = 1;
      ctx.strokeRect(hovered.x - 1.5, hovered.y - 1.5, cell + 3, cell + 3);
    }
    if (chosen) inkRing(ctx, chosen.x, chosen.y, cell, cell, colors.text, colors.surface);
  }, [geo, arch, colors, hoverLayer, hoverPart, hoverHead, selectedLayer, selectedPart, selectedHead]);

  const hit = useCallback(
    (clientX: number, clientY: number): Selection | null => {
      const canvas = canvasRef.current;
      if (!canvas || !geo || !arch) return null;
      const rect = canvas.getBoundingClientRect();
      const x = clientX - rect.left;
      const y = clientY - rect.top;
      const layer = Math.floor((y - geo.top - geo.headerH) / geo.pitch);
      if (layer < 0 || layer >= arch.nLayers) return null;
      if ((y - geo.top - geo.headerH) % geo.pitch > geo.cell + 1) return null;
      const col = geo.columns.find((c) => x >= c.x - 1 && x <= c.x + geo.cell + 1);
      if (!col) return null;
      return { layer, part: col.part, head: col.head };
    },
    [geo, arch],
  );

  const withContext = useCallback(
    (sel: Selection): Selection => {
      // Keep the selected position when moving between cells of a layer × position run.
      const positions = new Set(run.sites.filter((x) => x.position_key !== "all").map((x) => x.position_key));
      const out =
        selection?.positionKey !== undefined && positions.has(selection.positionKey)
          ? { ...sel, positionKey: selection.positionKey }
          : { ...sel, positionKey: undefined };
      // A residual cell selects the residual site it shows.
      if (out.part === "resid") {
        const shown = cells.get(cellKey(out.layer, "resid"))?.base?.kind;
        out.kind = shown && isResidKind(shown) ? shown : undefined;
      }
      return out;
    },
    [run.sites, selection?.positionKey, cells],
  );

  const onMove = (e: MouseEvent) => {
    const sel = hit(e.clientX, e.clientY);
    setHover(sel ? { sel, x: e.clientX, y: e.clientY } : null);
  };

  const onClick = (e: MouseEvent) => {
    const sel = hit(e.clientX, e.clientY);
    if (sel) select(withContext(sel));
    canvasRef.current?.focus();
  };

  const onContextMenu = (e: MouseEvent) => {
    const sel = hit(e.clientX, e.clientY);
    if (sel) {
      setMenuSel(withContext(sel));
      select(withContext(sel));
      return;
    }
    // Opened from the keyboard (Shift+F10 or the menu key), the pointer is elsewhere: the menu
    // acts on the selected component.
    setMenuSel(document.activeElement === canvasRef.current ? selection : null);
  };

  const onKeyDown = (e: KeyboardEvent) => {
    if (!arch) return;
    if (e.key === "Escape") {
      select(null);
      return;
    }
    const next = nextSelection(selection, e.key, arch.nLayers, arch.nHeads);
    if (!next) return;
    e.preventDefault();
    select(withContext(next));
  };

  const hoverData = hover ? cells.get(cellKey(hover.sel.layer, hover.sel.part, hover.sel.head)) : undefined;
  // Screen readers can't see the canvas: announce the selected cell and its value.
  const selectedData = selection ? cells.get(cellKey(selection.layer, selection.part, selection.head)) : undefined;
  const words = measureWords(run.detail?.spec.experiment, run.detail?.summary?.metric ?? run.detail?.spec.metric);
  const announcement = selection
    ? `${readoutLabel(selection, selectedData)}: ${
        selectedData?.site
          ? `${(metric === "effect" ? words.effect : words.delta).toLowerCase()} ${signed(siteValue(selectedData.site, metric), 3)}, confidence interval ${
              metric === "effect"
                ? ci(selectedData.site.effect.lo, selectedData.site.effect.hi)
                : ci(selectedData.site.delta.lo, selectedData.site.delta.hi)
            }`
          : selectedData?.pending
            ? "waiting for results"
            : "not measured in this run"
      }`
    : "";
  const statsTitle = metric === "effect" ? "effect" : words.delta;

  return (
    <div className={`${s.map} ${prominent ? s.prominent : ""}`}>
      {!prominent && <div className={s.header}>
        <div className={s.title}>
          <span>{arch ? modelName(arch.id) : "Model"}</span>
          {arch && (
            <span className={s.dims}>
              {arch.nLayers} layers × {arch.nHeads} heads
            </span>
          )}
        </div>
      </div>}
      <div className={s.scroll} ref={scrollRef}>
        {!arch ? (
          <div className={s.placeholder}>
            {modelState === "loading" ? (
              <p>Loading {modelName(loadingId)}. The map appears when it's ready.</p>
            ) : (
              <>
                <p>Load a model to see its map. Results paint onto it as they arrive.</p>
                <Button size="small" onClick={() => useStore.setState({ modelDialogOpen: true })}>
                  Load a model
                </Button>
              </>
            )}
          </div>
        ) : (
          <div className={s.stage}>
            <ContextMenu.Root
              onOpenChange={(open) => {
                if (!open) setMenuSel(null);
              }}
            >
              <ContextMenu.Trigger asChild>
                <canvas
                  ref={canvasRef}
                  className={s.canvas}
                  tabIndex={0}
                  role="group"
                  aria-roledescription="interactive model map"
                  aria-label="Model map"
                  aria-describedby={helpId}
                  onMouseMove={onMove}
                  onMouseLeave={() => setHover(null)}
                  onClick={onClick}
                  onDoubleClick={() => {
                    useStore.setState({ exploreMode: "layer" });
                    useStore.getState().setView("explore");
                  }}
                  onContextMenu={onContextMenu}
                  onKeyDown={onKeyDown}
                />
              </ContextMenu.Trigger>
              <ContextMenu.Portal>
                <MapMenu sel={menuSel} />
              </ContextMenu.Portal>
            </ContextMenu.Root>
            <canvas ref={overlayRef} className={s.overlay} aria-hidden="true" />
          </div>
        )}
      </div>
      {hover && (
        <div className={s.readout} style={{ left: hover.x + 14, top: hover.y + 14 }}>
          <div className={s.readoutTitle}>{readoutLabel(hover.sel, hoverData)}</div>
          {hoverData?.site ? (
            <>
              <div>
                {statsTitle} <strong>{signed(siteValue(hoverData.site, metric), 3)}</strong>
              </div>
              <div className={s.readoutMuted}>
                {run.ciLevel}% CI{" "}
                {metric === "effect"
                  ? ci(hoverData.site.effect.lo, hoverData.site.effect.hi)
                  : ci(hoverData.site.delta.lo, hoverData.site.delta.hi)}{" "}
                · n {hoverData.site.n}
              </div>
              {hoverData.note && <div className={s.readoutMuted}>{hoverData.note}</div>}
            </>
          ) : (
            <div className={s.readoutMuted}>
              {hoverData?.pending ? "Waiting for this layer" : run.id ? "Not part of this run" : "No run selected"}
            </div>
          )}
        </div>
      )}
      <p id={helpId} className="visually-hidden">
        Layers are rows; the residual stream, each attention head, and the attention and MLP outputs are cells. Arrow keys
        move between components and layers, Home and End go to the start and end of a layer, and Escape clears the
        selection. P patches the selected component, B ablates it, A opens a head's attention, and Shift+F10 lists these
        actions.
      </p>
      <div className="visually-hidden" aria-live="polite">
        {announcement}
      </div>
      {arch && !structureOnly && <Legend bound={bound} hasFlags={flagged.size > 0} flagCount={flagged.size} />}
      {!structureOnly && [...cells.values()].some(cell => cell.note?.startsWith("strongest")) && <p className={s.legendText}>Cells with several measured sites show the largest absolute value within the selected position and residual site. The tooltip names the displayed site.</p>}
      {structureOnly && <p className={s.legend}>Architecture only. Select a component to inspect its measurements.</p>}
    </div>
  );
}

function readoutLabel(sel: Selection, data: CellData | undefined): string {
  return data?.base ? data.base.label : componentLabel(sel);
}

function MapMenu({ sel }: { sel: Selection | null }) {
  const prefill = useStore((st) => st.prefillExperiment);
  const setView = useStore((st) => st.setView);
  const focusInspector = useStore((st) => st.focusInspector);
  const hasModel = useStore((st) => st.model.state === "ready");
  if (!sel) {
    return (
      <ContextMenu.Content className={menuClasses.menu}>
        <div className={menuClasses.label}>Right-click a cell</div>
      </ContextMenu.Content>
    );
  }
  return (
    <ContextMenu.Content className={menuClasses.menu}>
      <div className={menuClasses.label}>{componentLabel(sel)}</div>
      <ContextMenu.Item className={menuClasses.item} onSelect={() => prefill("activation_patching", sel)}>
        Patch here
        <span className={menuClasses.shortcut}>P</span>
      </ContextMenu.Item>
      <ContextMenu.Item className={menuClasses.item} onSelect={() => prefill("ablation", sel)}>
        Ablate here
        <span className={menuClasses.shortcut}>B</span>
      </ContextMenu.Item>
      <ContextMenu.Item
        className={menuClasses.item}
        disabled={sel.part !== "head" || !hasModel}
        onSelect={() => setView("attention")}
      >
        Open attention
        <span className={menuClasses.shortcut}>A</span>
      </ContextMenu.Item>
      <ContextMenu.Separator className={menuClasses.separator} />
      <ContextMenu.Item className={menuClasses.item} onSelect={() => focusInspector("runs")}>
        Compare across runs
        <span className={menuClasses.shortcut}>C</span>
      </ContextMenu.Item>
    </ContextMenu.Content>
  );
}
