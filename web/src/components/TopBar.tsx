import * as DropdownMenu from "@radix-ui/react-dropdown-menu";

import { useStore, type Workspace } from "../store/app";
import { MOD, ProjectMenu } from "./Header";
import { HelpMenu } from "./HelpMenu";
import { BRAND_SEED, Logogram } from "./Logogram";
import { UpdateNotice } from "./UpdateNotice";
import { Icon, menuClasses, Tip } from "./ui";
import type { IconName } from "./ui/Icon";
import s from "./TopBar.module.css";

export const WORKSPACE_ITEMS: { id: Workspace; label: string; icon: IconName; hint: string }[] = [
  { id: "explore", label: "Explore", icon: "explore", hint: "Look inside the model" },
  { id: "experiment", label: "Experiment", icon: "experiment", hint: "Prompts, baseline and the intervention" },
  { id: "evidence", label: "Evidence", icon: "evidence", hint: "Results, comparisons and notes" },
];

/** One bar across the top: the project, the three workspaces, and the tools. */
export function TopBar({
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
    <header className={s.bar}>
      <div className={s.left}>
        <Tip text="All projects" side="bottom">
          <button
            type="button"
            className={s.home}
            onClick={() => useStore.getState().goto("projects")}
            aria-label="All projects"
          >
            <Logogram seed={BRAND_SEED} size={30} haze={false} detail={0.85} weight={1.3} />
          </button>
        </Tip>
        <ProjectMenu />
      </div>
      <nav className={s.workspaces} aria-label="Workspaces">
        {WORKSPACE_ITEMS.map((w) => {
          const stagedHere = w.id === "experiment" && staged > 0;
          const stagedText = `${staged} site${staged === 1 ? "" : "s"} staged on the model map`;
          return (
            <Tip key={w.id} text={stagedHere ? `${w.hint}. ${stagedText}.` : w.hint} side="bottom">
              <button
                type="button"
                className={s.workspace}
                data-workspace={w.id}
                aria-current={workspace === w.id ? "page" : undefined}
                onClick={() => onWorkspace(w.id)}
              >
                <Icon name={w.icon} size={16} />
                <span className={s.label}>{w.label}</span>
                {stagedHere && (
                  <span className={s.badge}>
                    {staged}
                    <span className="visually-hidden"> {staged === 1 ? "site" : "sites"} staged</span>
                  </span>
                )}
              </button>
            </Tip>
          );
        })}
      </nav>
      <div className={s.right}>
        <UpdateNotice />
        <Tip text="Every run in this project" side="bottom">
          <button type="button" className={s.tool} onClick={onHistory}>
            <Icon name="history" size={16} />
            <span className={s.label}>History</span>
            {runs > 0 && <span className={s.count}>{runs}</span>}
          </button>
        </Tip>
        <Tip text={`Commands (${MOD}+K)`} side="bottom">
          <button
            type="button"
            className={s.tool}
            onClick={() => useStore.setState({ paletteOpen: true })}
            aria-label="Open the command palette"
          >
            <Icon name="search" size={16} />
            <span className={s.label}>Commands</span>
            <kbd className={s.kbd}>{MOD} K</kbd>
          </button>
        </Tip>
        <HelpMenu className={s.iconTool} />
        <ThemeMenu />
      </div>
    </header>
  );
}

function ThemeMenu() {
  const setting = useStore((st) => st.themeSetting);
  const theme = useStore((st) => st.theme);
  const setTheme = useStore((st) => st.setTheme);
  return (
    <DropdownMenu.Root>
      <Tip text="Appearance" side="bottom">
        <DropdownMenu.Trigger className={s.iconTool} aria-label="Appearance">
          <Icon name={theme === "dark" ? "moon" : "sun"} size={16} />
        </DropdownMenu.Trigger>
      </Tip>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className={menuClasses.menu} sideOffset={8} align="end">
          <div className={menuClasses.label}>Appearance</div>
          {(["light", "dark", "system"] as const).map((t) => (
            <DropdownMenu.Item key={t} className={menuClasses.item} onSelect={() => setTheme(t)}>
              {setting === t ? <Icon name="check" size={14} /> : <span style={{ width: 14 }} />}
              {t === "system" ? "Match the system" : t === "light" ? "White" : "Black"}
            </DropdownMenu.Item>
          ))}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
