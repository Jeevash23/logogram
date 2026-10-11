import { useSyncExternalStore } from "react";

const QUERY = "(prefers-reduced-motion: reduce)";

function media(): MediaQueryList | null {
  return typeof window !== "undefined" && typeof window.matchMedia === "function" ? window.matchMedia(QUERY) : null;
}

/** Whether the person asked for less motion: a change then shows at once instead of animating. */
export function prefersReducedMotion(): boolean {
  return media()?.matches ?? false;
}

/** prefersReducedMotion, following changes to the setting while the page is open. */
export function usePrefersReducedMotion(): boolean {
  return useSyncExternalStore(
    (onChange) => {
      const list = media();
      list?.addEventListener("change", onChange);
      return () => list?.removeEventListener("change", onChange);
    },
    prefersReducedMotion,
    () => false,
  );
}
