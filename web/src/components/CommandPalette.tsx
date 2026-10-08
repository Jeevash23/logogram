import * as RadixDialog from "@radix-ui/react-dialog";
import { useEffect, useMemo, useRef, useState } from "react";

import { architectureOf } from "../lib/keys";
import { parseHeadQuery } from "../lib/sites";
import { useStore, VIEWS } from "../store/app";
import { Icon, Kbd } from "./ui";
import s from "./CommandPalette.module.css";

interface Command {
  id: string;
  label: string;
  group: string;
  keywords?: string;
  shortcut?: string;
  disabled?: boolean;
  run: () => void;
}

function useCommands(query: string): Command[] {
  const st = useStore();
  return useMemo(() => {
    const cmds: Command[] = [];
    const inWorkbench = st.screen === "workbench";
    if (inWorkbench) {
      VIEWS.forEach((v, i) =>
        cmds.push({ id: `view-${v.id}`, label: `Go to ${v.label.toLowerCase()}`, group: "Views", shortcut: i < 7 ? String(i + 1) : undefined, run: () => st.setView(v.id) }),
      );
      cmds.push(
        { id: "model", label: "Load a model…", group: "Actions", keywords: "gpt2 hugging face download", run: () => useStore.setState({ modelDialogOpen: true }) },
        { id: "baseline", label: "Check the baseline", group: "Actions", run: () => st.setView("baseline") },
        { id: "experiment", label: "New experiment", group: "Actions", keywords: "patch ablate run sweep", run: () => st.setView("experiment") },
        {
          id: "robustness",
          label: "Check robustness of this run…",
          group: "Actions",
          disabled: !st.activeRunId || !st.runDetails[st.activeRunId]?.summary,
          run: () => useStore.setState({ robustnessDialogOpen: true }),
        },
        { id: "compare", label: "Compare two runs", group: "Actions", run: () => st.setView("compare") },
        { id: "spec", label: "Show the spec", group: "Actions", keywords: "json cli", run: () => useStore.setState({ view: "spec", specSource: "run" }) },
        {
          id: "cancel",
          label: "Cancel the running job",
          group: "Actions",
          disabled: st.job?.status !== "running" || st.job?.kind !== "run",
          run: () => void st.cancelJob(),
        },
      );
      if (st.selection) {
        const sel = st.selection;
        cmds.push(
          { id: "patch-here", label: "Patch here", group: "Selected component", shortcut: "P", run: () => st.prefillExperiment("activation_patching", sel) },
          { id: "ablate-here", label: "Ablate here", group: "Selected component", shortcut: "B", run: () => st.prefillExperiment("ablation", sel) },
          { id: "attention", label: "Open attention", group: "Selected component", shortcut: "A", disabled: sel.part !== "head", run: () => st.setView("attention") },
          { id: "across", label: "Compare across runs", group: "Selected component", shortcut: "C", run: () => st.focusInspector("runs") },
        );
      }
      for (const r of st.runs.slice(0, 12)) {
        cmds.push({
          id: `run-${r.id}`,
          label: `Open run: ${r.name}`,
          group: "Runs",
          keywords: r.id,
          run: () => void st.openRun(r.id, r.status === "finished" ? "results" : undefined),
        });
      }
    }
    cmds.push(
      { id: "projects", label: "All projects", group: "Project", run: () => st.goto("projects") },
      { id: "system", label: "System check", group: "Project", keywords: "gpu cuda doctor hardware", run: () => st.goto("system") },
      { id: "updates", label: "Check for updates", group: "Project", keywords: "version upgrade new release pypi", run: () => void st.checkUpdates() },
      { id: "light", label: "White appearance", group: "Appearance", keywords: "theme light", run: () => st.setTheme("light") },
      { id: "dark", label: "Black appearance", group: "Appearance", keywords: "theme dark", run: () => st.setTheme("dark") },
      { id: "system-theme", label: "Match the system appearance", group: "Appearance", keywords: "theme", run: () => st.setTheme("system") },
    );

    const head = parseHeadQuery(query);
    const arch = architectureOf();
    if (head && arch && head.layer < arch.nLayers && head.head < arch.nHeads && inWorkbench) {
      cmds.unshift({
        id: "goto-head",
        label: `Select L${head.layer} H${head.head}`,
        group: "Map",
        run: () => st.select({ layer: head.layer, part: "head", head: head.head }),
      });
    }
    return cmds;
  }, [st, query]);
}

export function CommandPalette() {
  const open = useStore((st) => st.paletteOpen);
  return (
    <RadixDialog.Root open={open} onOpenChange={(v) => useStore.setState({ paletteOpen: v })}>
      {open && <PaletteBody />}
    </RadixDialog.Root>
  );
}

function PaletteBody() {
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);
  const commands = useCommands(query);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return commands;
    return commands
      .map((c) => {
        const hay = `${c.label} ${c.keywords ?? ""} ${c.group}`.toLowerCase();
        const idx = hay.indexOf(q);
        const words = q.split(/\s+/).every((w) => hay.includes(w));
        return { c, score: c.id === "goto-head" ? -1 : idx >= 0 ? idx : words ? 100 : Infinity };
      })
      .filter((x) => x.score !== Infinity)
      .sort((a, b) => a.score - b.score)
      .map((x) => x.c);
  }, [commands, query]);

  useEffect(() => {
    setActive(0);
  }, [query]);

  useEffect(() => {
    listRef.current?.querySelector(`[data-index="${active}"]`)?.scrollIntoView({ block: "nearest" });
  }, [active]);

  const close = () => useStore.setState({ paletteOpen: false });
  const runAt = (i: number) => {
    const cmd = filtered[i];
    if (!cmd || cmd.disabled) return;
    close();
    window.setTimeout(cmd.run, 0);
  };

  let lastGroup = "";
  return (
    <>
      <RadixDialog.Portal>
        <RadixDialog.Overlay className={s.overlay} />
        <RadixDialog.Content className={s.palette} aria-describedby={undefined}>
          <RadixDialog.Title className="visually-hidden">Command palette</RadixDialog.Title>
          <div className={s.inputRow}>
            <Icon name="search" />
            <input
              className={s.input}
              autoFocus
              placeholder="Type a command, or a head like L9H6"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "ArrowDown") {
                  e.preventDefault();
                  setActive((a) => Math.min(filtered.length - 1, a + 1));
                } else if (e.key === "ArrowUp") {
                  e.preventDefault();
                  setActive((a) => Math.max(0, a - 1));
                } else if (e.key === "Enter") {
                  e.preventDefault();
                  runAt(active);
                }
              }}
              aria-label="Command"
              aria-controls="palette-list"
              aria-activedescendant={filtered[active] ? `cmd-${filtered[active].id}` : undefined}
            />
          </div>
          <div className={s.list} id="palette-list" role="listbox" ref={listRef}>
            {filtered.length === 0 && <div className={s.empty}>No matching commands.</div>}
            {filtered.map((cmd, i) => {
              const header = cmd.group !== lastGroup ? cmd.group : null;
              lastGroup = cmd.group;
              return (
                <div key={cmd.id}>
                  {header && <div className={s.group}>{header}</div>}
                  <div
                    id={`cmd-${cmd.id}`}
                    role="option"
                    aria-selected={i === active}
                    aria-disabled={cmd.disabled}
                    data-index={i}
                    className={s.item}
                    onMouseMove={() => setActive(i)}
                    onClick={() => runAt(i)}
                  >
                    <span>{cmd.label}</span>
                    {cmd.shortcut && <Kbd>{cmd.shortcut}</Kbd>}
                  </div>
                </div>
              );
            })}
          </div>
        </RadixDialog.Content>
      </RadixDialog.Portal>
    </>
  );
}
