import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { useEffect, useId, useRef } from "react";

import { GUIDE, guideUrl } from "../lib/help";
import { useStore } from "../store/app";
import { Icon, menuClasses, Tip } from "./ui";
import s from "./HelpMenu.module.css";

type State = ReturnType<typeof useStore.getState>;

/**
 * Help, in the top bar: the keyboard shortcuts, the guide's sections (on GitHub, in a new tab),
 * a problem report, the system check, and About. A dialog opens once the menu has closed, and
 * when it closes the focus comes back to this button, where it was.
 */
export function HelpMenu({ className }: { className: string }) {
  const singleKeys = useStore((st) => st.singleKeys);
  const trigger = useRef<HTMLButtonElement>(null);
  const after = useRef<(() => void) | null>(null);
  // Whether the dialog this menu opened is still open.
  const opened = useRef<((st: State) => boolean) | null>(null);
  const guideId = useId();

  useEffect(
    () =>
      useStore.subscribe((st, prev) => {
        const isOpen = opened.current;
        if (!isOpen || !isOpen(prev) || isOpen(st)) return;
        opened.current = null;
        // Radix gives the focus back only to a dialog's own trigger; these open from the menu.
        window.setTimeout(() => trigger.current?.focus(), 0);
      }),
    [],
  );

  /** Open a dialog once the menu has closed and the focus is back on Help. */
  const later = (isOpen: (st: State) => boolean, open: Partial<State>) => () => {
    after.current = () => {
      opened.current = isOpen;
      useStore.setState(open);
    };
  };

  // Not modal, and inside the top bar rather than a portal: the page around the menu stays as it
  // is, for assistive technology too, and the menu belongs to the banner landmark.
  return (
    <DropdownMenu.Root modal={false}>
      <Tip text="Help" side="bottom">
        <DropdownMenu.Trigger ref={trigger} className={className} aria-label="Help">
          <Icon name="help" size={16} />
        </DropdownMenu.Trigger>
      </Tip>
      <DropdownMenu.Content
        className={menuClasses.menu}
        sideOffset={8}
        align="end"
        onCloseAutoFocus={() => {
          const action = after.current;
          after.current = null;
          if (action) window.setTimeout(action, 0);
        }}
      >
        <DropdownMenu.Item className={menuClasses.item} onSelect={later((st) => st.shortcutsOpen, { shortcutsOpen: true })}>
          Keyboard shortcuts
          {singleKeys && <span className={menuClasses.shortcut}>?</span>}
        </DropdownMenu.Item>
        <DropdownMenu.Separator className={menuClasses.separator} />
        <DropdownMenu.Group aria-labelledby={guideId}>
          <DropdownMenu.Label className={menuClasses.label} id={guideId}>
            The guide, on GitHub
          </DropdownMenu.Label>
          {GUIDE.map((section) => (
            <DropdownMenu.Item key={section.anchor} className={`${menuClasses.item} ${s.link}`} asChild>
              <a href={guideUrl(section.anchor)} target="_blank" rel="noopener noreferrer">
                {section.title}
                <span className="visually-hidden"> (opens in a new tab)</span>
                <span className={menuClasses.shortcut} aria-hidden="true">
                  ↗
                </span>
              </a>
            </DropdownMenu.Item>
          ))}
        </DropdownMenu.Group>
        <DropdownMenu.Separator className={menuClasses.separator} />
        <DropdownMenu.Item className={menuClasses.item} onSelect={later((st) => st.helpDialog === "report", { helpDialog: "report" })}>
          Report a problem…
        </DropdownMenu.Item>
        <DropdownMenu.Item className={menuClasses.item} onSelect={() => useStore.getState().goto("system")}>
          System check
        </DropdownMenu.Item>
        <DropdownMenu.Item className={menuClasses.item} onSelect={later((st) => st.helpDialog === "about", { helpDialog: "about" })}>
          About Logogram
        </DropdownMenu.Item>
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
}
