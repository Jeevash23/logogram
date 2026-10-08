import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { useState, type ReactNode } from "react";

import { api } from "../api/client";
import { FolderPicker } from "./FolderPicker";
import { Icon, Spinner, menuClasses } from "./ui";
import { modelName, useAnalysisContext } from "../lib/hooks";
import { pct, plural } from "../lib/format";
import { useStore } from "../store/app";
import s from "./Header.module.css";

const isMac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);
export const MOD = isMac ? "⌘" : "Ctrl";

/** The context bar, on fog: which project, which model, which prompts, and what to do next. */
export function Header({ next }: { next?: ReactNode } = {}) {
  const project = useStore((st) => st.project);
  const [picker, setPicker] = useState(false);
  const guard = useStore((st) => st.guard);
  const enter = useStore((st) => st.enterProject);

  return (
    <header className={s.header}>
      <DropdownMenu.Root>
        <DropdownMenu.Trigger className={s.project}>
          <span className={s.projectName}>{project?.name ?? "Logogram"}</span>
          <Icon name="chevronDown" size={14} />
        </DropdownMenu.Trigger>
        <DropdownMenu.Portal>
          <DropdownMenu.Content className={menuClasses.menu} sideOffset={6} align="start">
            <div className={menuClasses.label} title={project?.path}>
              {project?.path}
            </div>
            <DropdownMenu.Item className={menuClasses.item} onSelect={() => useStore.getState().goto("projects")}>
              All projects
            </DropdownMenu.Item>
            <DropdownMenu.Item className={menuClasses.item} onSelect={() => setPicker(true)}>
              Open a project folder…
            </DropdownMenu.Item>
            <DropdownMenu.Item className={menuClasses.item} onSelect={() => useStore.getState().goto("system")}>
              System check
            </DropdownMenu.Item>
            <DropdownMenu.Separator className={menuClasses.separator} />
            <DropdownMenu.Item
              className={menuClasses.item}
              onSelect={async () => {
                const result = await guard(() => api.closeProject());
                if (result) useStore.getState().leaveProject();
              }}
            >
              Close project
            </DropdownMenu.Item>
          </DropdownMenu.Content>
        </DropdownMenu.Portal>
      </DropdownMenu.Root>

      <span className={s.divider} aria-hidden="true" />
      <ModelChip />
      <DatasetChip />

      <div className={s.spacer} />
      {next}
      <FolderPicker
        open={picker}
        onOpenChange={setPicker}
        title="Open a project"
        start={project?.path}
        chooseLabel="Open"
        requireProject
        onChoose={async (path) => {
          const next = await guard(() => api.openProject(path));
          if (next) await enter(next);
        }}
      />
    </header>
  );
}

function ModelChip() {
  const model = useStore((st) => st.model);
  const open = () => useStore.setState({ modelDialogOpen: true });
  if (model.state === "loading") {
    const progress =
      model.stage === "downloading" && model.total ? ` · downloading ${pct((model.done ?? 0) / model.total)}` : model.stage ? ` · ${model.stage}` : "";
    return (
      <button type="button" className={s.chip} onClick={open}>
        <Spinner />
        <span>
          {modelName(model.id)}
          <span className={s.chipMuted}>{progress}</span>
        </span>
      </button>
    );
  }
  if (model.state === "ready" && model.info) {
    const info = model.info;
    return (
      <button type="button" className={s.chip} onClick={open} title={`${info.id} @ ${info.revision ?? ""}`}>
        <span className={s.live} aria-hidden="true" />
        <span>
          {modelName(info.id)}
          <span className={s.chipMuted}>
            {" "}
            · {info.device} · {info.dtype}
          </span>
        </span>
      </button>
    );
  }
  return (
    <button type="button" className={`${s.chip} ${s.chipEmpty}`} onClick={open}>
      <span className={s.ring} aria-hidden="true" />
      <span>{model.state === "error" ? "Model failed to load" : "Load a model"}</span>
    </button>
  );
}

function DatasetChip() {
  const context = useAnalysisContext();
  const project = useStore((st) => st.project);
  const datasetPath = useStore((st) => st.datasetPath);
  const dataset = useStore((st) => st.dataset);
  const select = useStore((st) => st.selectDataset);
  const setView = useStore((st) => st.setView);
  const datasets = project?.datasets ?? [];
  const current = datasets.find((d) => d.path === datasetPath);
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger className={`${s.chip} ${current ? "" : s.chipEmpty}`}>
        <Icon name="file" size={14} />
        <span>
          {current ? current.name : datasetPath?.startsWith("datasets/snapshots/") ? "Saved prompts of this run" : "No prompts yet"}
          {dataset && <span className={s.chipMuted}> · {context.n < dataset.n ? `${context.n} of ${dataset.n} prompts` : plural(dataset.n, "prompt")}</span>}
        </span>
        <Icon name="chevronDown" size={14} />
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className={menuClasses.menu} sideOffset={6} align="start">
          <div className={menuClasses.label}>Datasets in this project</div>
          {datasets.length === 0 && <div className={menuClasses.label}>None yet</div>}
          {datasets.map((d) => (
            <DropdownMenu.Item
              key={d.path}
              className={menuClasses.item}
              disabled={!!d.error}
              onSelect={() => void select(d.path)}
            >
              {d.path === datasetPath ? <Icon name="check" size={14} /> : <span style={{ width: 14 }} />}
              {d.name}
              <span className={menuClasses.shortcut}>{d.error ? "unreadable" : d.n}</span>
            </DropdownMenu.Item>
          ))}
          <DropdownMenu.Separator className={menuClasses.separator} />
          <DropdownMenu.Item className={menuClasses.item} onSelect={() => setView("prompts")}>
            Set up prompts…
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
