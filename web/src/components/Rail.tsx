import * as DropdownMenu from "@radix-ui/react-dropdown-menu";

import { useStore, type Workspace } from "../store/app";
import { MOD } from "./Header";
import { BRAND_SEED, Logogram } from "./Logogram";
import { Icon, menuClasses } from "./ui";
import type { IconName } from "./ui/Icon";
import s from "./Rail.module.css";

export const WORKSPACE_ITEMS: { id: Workspace; label: string; icon: IconName; hint: string }[] = [
  { id: "explore", label: "Explore", icon: "explore", hint: "Look inside the model" },
  { id: "experiment", label: "Experiment", icon: "experiment", hint: "Prompts, baseline and the intervention" },
  { id: "evidence", label: "Evidence", icon: "evidence", hint: "Results, comparisons and notes" },
];

/** The shell: the one dark, solid shape on the screen. It holds where you are and where to go. */
export function Rail({
  workspace,
  onWorkspace,
  onHistory,
  runs,
  staged,
}: {
  workspace: Workspace;
  onWorkspace: (w: Workspace) => void;
  onHistory: () => void;
  runs: number;
  staged: number;
}) {
  return (
    <nav className={s.rail} aria-label="Workspaces">
      <button
        type="button"
        className={s.home}
        onClick={() => useStore.getState().goto("projects")}
        title="All projects"
        aria-label="All projects"
      >
        <Logogram seed={BRAND_SEED} size={40} haze={false} detail={0.85} weight={1.3} />
      </button>
      <div className={s.group}>
        {WORKSPACE_ITEMS.map((w) => (
          <button
            key={w.id}
            type="button"
            className={s.item}
            aria-current={workspace === w.id ? "page" : undefined}
            title={w.hint}
            onClick={() => onWorkspace(w.id)}
          >
            <Icon name={w.icon} size={18} />
            <span className={s.label}>{w.label}</span>
            {w.id === "experiment" && staged > 0 && (
              <span className={s.badge} title="Staged sites">
                {staged}
              </span>
            )}
          </button>
        ))}
      </div>
      <div className={s.spacer} />
      <div className={s.group}>
        <button type="button" className={s.item} onClick={onHistory} title="Every run in this project">
          <Icon name="history" size={18} />
          <span className={s.label}>History</span>
          {runs > 0 && <span className={s.count}>{runs}</span>}
        </button>
        <button
          type="button"
          className={s.item}
          onClick={() => useStore.setState({ paletteOpen: true })}
          title={`Commands (${MOD}+K)`}
          aria-label="Open the command palette"
        >
          <Icon name="palette" size={18} />
          <span className={s.label}>Commands</span>
        </button>
        <ThemeMenu />
      </div>
    </nav>
  );
}

function ThemeMenu() {
  const setting = useStore((st) => st.themeSetting);
  const theme = useStore((st) => st.theme);
  const setTheme = useStore((st) => st.setTheme);
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger className={s.item} aria-label="Appearance" title="Appearance">
        <Icon name={theme === "dark" ? "moon" : "sun"} size={18} />
        <span className={s.label}>{theme === "dark" ? "Dark" : "Light"}</span>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className={menuClasses.menu} sideOffset={8} side="right" align="end">
          <div className={menuClasses.label}>Appearance</div>
          {(["light", "dark", "system"] as const).map((t) => (
            <DropdownMenu.Item key={t} className={menuClasses.item} onSelect={() => setTheme(t)}>
              {setting === t ? <Icon name="check" size={14} /> : <span style={{ width: 14 }} />}
              {t === "system" ? "Match the system" : t === "light" ? "Light: fog" : "Dark: inside the shell"}
            </DropdownMenu.Item>
          ))}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
