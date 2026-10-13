import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { useLayoutEffect, useRef, useState, type ReactNode } from "react";

import type { View } from "../store/app";
import { Icon, menuClasses } from "./ui";
import s from "../screens/Workbench.module.css";

// The space between tabs (Workbench.module.css .tabs gap).
const GAP = 24;

/**
 * A workspace's views as tabs. Those that don't fit in the row are listed under More, whose label
 * is the open view's when it is one of them, so the open view is always named in the row.
 */
export function ViewTabs({
  label,
  views,
  view,
  onView,
  shortcuts,
  badge,
}: {
  label: string;
  views: { id: View; label: string }[];
  view: View;
  onView: (v: View) => void;
  /** Whether the number keys open the views (1 for the first, and so on). */
  shortcuts: boolean;
  /** What a tab shows beside its name, such as the heads pinned for comparison. */
  badge?: (id: View) => ReactNode;
}) {
  const nav = useRef<HTMLElement>(null);
  const ruler = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(views.length);
  // While the menu is open its items stay as they are, so a layout that moves under it (a font
  // loading, say) never changes what the keyboard is moving through.
  const [open, setOpen] = useState(false);
  const openRef = useRef(false);
  openRef.current = open;

  // How many tabs fit, measured on hidden copies of them all, again when the row changes size.
  useLayoutEffect(() => {
    const row = nav.current;
    const copies = ruler.current;
    if (!row || !copies) return;
    const fit = () => {
      if (openRef.current) return;
      const widths = Array.from(copies.children, (el) => (el as HTMLElement).offsetWidth);
      const more = widths.pop() ?? 80;
      const room = row.clientWidth;
      const all = widths.reduce((sum, w) => sum + w, 0) + GAP * (widths.length - 1);
      if (all <= room) {
        setShown(widths.length);
        return;
      }
      let used = more;
      let n = 0;
      while (n < widths.length && used + GAP + widths[n] <= room) used += GAP + widths[n++];
      setShown(Math.max(1, n));
    };
    fit();
    const observer = new ResizeObserver(fit);
    observer.observe(row);
    observer.observe(copies);
    // The tabs' widths change once the typeface has loaded.
    let live = true;
    document.fonts?.ready.then(() => live && fit());
    return () => {
      live = false;
      observer.disconnect();
    };
  }, [views, open]);

  const inRow = views.slice(0, shown);
  const rest = views.slice(shown);
  const openInRest = rest.find((v) => v.id === view);
  const tab = (v: { id: View; label: string }) => (
    <button
      key={v.id}
      type="button"
      className={s.tab}
      data-tour={`tab-${v.id}`}
      aria-current={view === v.id ? "page" : undefined}
      aria-keyshortcuts={shortcuts ? String(views.indexOf(v) + 1) : undefined}
      onClick={() => onView(v.id)}
    >
      {v.label}
      {badge?.(v.id)}
    </button>
  );
  return (
    <>
      <nav className={s.tabs} aria-label={label} ref={nav}>
        {inRow.map(tab)}
        {rest.length > 0 && (
          <DropdownMenu.Root modal={false} open={open} onOpenChange={setOpen}>
            <DropdownMenu.Trigger
              className={s.tab}
              aria-current={openInRest ? "page" : undefined}
              aria-label={openInRest ? `${openInRest.label}, and ${rest.length - 1} more views` : `${rest.length} more views`}
            >
              {openInRest ? openInRest.label : "More"}
              <Icon name="chevronDown" size={14} />
            </DropdownMenu.Trigger>
            <DropdownMenu.Portal>
              <DropdownMenu.Content className={menuClasses.menu} sideOffset={6} align="start">
                {rest.map((v) => (
                  <DropdownMenu.Item
                    key={v.id}
                    className={menuClasses.item}
                    data-tour={`tab-${v.id}`}
                    aria-current={view === v.id ? "page" : undefined}
                    onSelect={() => onView(v.id)}
                  >
                    {v.label}
                    {badge?.(v.id)}
                    {view === v.id && <Icon name="check" size={14} />}
                  </DropdownMenu.Item>
                ))}
              </DropdownMenu.Content>
            </DropdownMenu.Portal>
          </DropdownMenu.Root>
        )}
      </nav>
      {/* Hidden copies of every tab and of More, to measure what fits. */}
      <div className={s.tabRuler} ref={ruler} aria-hidden="true">
        {views.map((v) => (
          <span key={v.id} className={s.tab}>
            {v.label}
            {badge?.(v.id)}
          </span>
        ))}
        {/* More, as wide as it gets: named after the longest view. */}
        <span className={s.tab}>
          {views.reduce((a, b) => (b.label.length > a.length ? b.label : a), "More")}
          <Icon name="chevronDown" size={14} />
        </span>
      </div>
    </>
  );
}
