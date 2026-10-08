// Global keyboard control. Shortcuts are ignored while typing in a field, while a dialog or menu
// is open, and for keys a focused control already handled.

import { useEffect } from "react";

import { useStore, VIEWS } from "../store/app";
import { nextSelection } from "./sites";

function typing(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

// Controls that use arrow keys themselves.
const ARROW_WIDGETS = "[role=menu], [role=menubar], [role=listbox], [role=separator], [role=slider], [role=radiogroup], [role=tablist], [role=grid]";
const OVERLAYS = "[role=dialog], [role=alertdialog], [role=menu], [role=listbox]";

/** Whether the arrow keys would scroll something around the focused element. */
function scrollsWithArrows(el: HTMLElement | null): boolean {
  for (let node = el; node && node !== document.body; node = node.parentElement) {
    const style = getComputedStyle(node);
    const scrollable = /(auto|scroll)/.test(`${style.overflowY} ${style.overflowX}`);
    if (scrollable && (node.scrollHeight > node.clientHeight + 1 || node.scrollWidth > node.clientWidth + 1)) return true;
  }
  return false;
}

export function architectureOf(): { nLayers: number; nHeads: number } | null {
  const st = useStore.getState();
  const id = st.activeRunId;
  const shape = (id && (st.runDetails[id]?.summary?.model ?? st.live[id]?.model)) || null;
  if (shape) return { nLayers: shape.n_layers, nHeads: shape.n_heads };
  const info = st.model.info;
  return info ? { nLayers: info.n_layers, nHeads: info.n_heads } : null;
}

export function useGlobalKeys(): void {
  useEffect(() => {
    // Seen before any menu reacts to the key: Escape that closes a menu mustn't also clear the
    // selection, and keys pressed in a menu mustn't switch views behind it.
    let overlayOpen = false;
    const onKeyCapture = () => {
      overlayOpen = document.querySelector(OVERLAYS) !== null;
    };
    const onKey = (e: KeyboardEvent) => {
      const st = useStore.getState();
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === "k") {
        e.preventDefault();
        useStore.setState({ paletteOpen: !st.paletteOpen });
        return;
      }
      if (st.screen !== "workbench" || st.paletteOpen || st.modelDialogOpen || st.robustnessDialogOpen) return;
      if (e.defaultPrevented || overlayOpen || typing(e.target) || mod || e.altKey) return;
      const target = e.target instanceof HTMLElement ? e.target : null;

      if (e.key.startsWith("Arrow") || e.key === "Home" || e.key === "End") {
        if (target?.tagName === "CANVAS" || target?.closest(ARROW_WIDGETS)) return; // it handles its own
        if (target && target !== document.body && scrollsWithArrows(target)) return; // let it scroll
        const arch = architectureOf();
        if (!arch) return;
        const next = nextSelection(st.selection, e.key, arch.nLayers, arch.nHeads);
        if (next) {
          e.preventDefault();
          st.select(next);
        }
        return;
      }
      const sel = st.selection;
      switch (e.key) {
        case "Escape":
          st.select(null);
          return;
        case "[":
          st.setPromptIndex(st.promptIndex - 1);
          return;
        case "]":
          st.setPromptIndex(st.promptIndex + 1);
          return;
        case "a":
          if (sel?.part === "head") st.setView("attention");
          return;
        case "p":
          if (sel) st.prefillExperiment("activation_patching", sel);
          return;
        case "b":
          if (sel) st.prefillExperiment("ablation", sel);
          return;
        case "c":
          if (sel) st.focusInspector("runs");
          return;
        default:
          if (/^[1-7]$/.test(e.key)) st.setView(VIEWS[Number(e.key) - 1].id);
      }
    };
    window.addEventListener("keydown", onKeyCapture, true);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKeyCapture, true);
      window.removeEventListener("keydown", onKey);
    };
  }, []);
}
