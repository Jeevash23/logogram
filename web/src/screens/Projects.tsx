import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { useEffect, useRef, useState } from "react";

import { api } from "../api/client";
import type { RecentProject } from "../api/types";
import { ExampleGallery } from "../components/ExampleGallery";
import { FolderPicker } from "../components/FolderPicker";
import { Logogram } from "../components/Logogram";
import { DeleteProjectDialog } from "../components/ManageDialogs";
import { Button, Field, Icon, Input, menuClasses } from "../components/ui";
import { ago } from "../lib/format";
import { useStore } from "../store/app";
import s from "./Screens.module.css";
import { Welcome } from "./Welcome";

export function Projects() {
  const [recent, setRecent] = useState<RecentProject[] | null>(null);
  const [name, setName] = useState("");
  const parentDefault = useStore((st) => st.projectsParent);
  const [parent, setParent] = useState("");
  const [picker, setPicker] = useState<"parent" | "open" | null>(null);
  const [busy, setBusy] = useState(false);
  const guard = useStore((st) => st.guard);
  const enter = useStore((st) => st.enterProject);
  // The project still open behind this screen: going back to it keeps your place in it.
  const current = useStore((st) => st.project);
  const back = () => useStore.getState().goto("workbench");

  const [deleting, setDeleting] = useState<{ path: string; name: string } | null>(null);
  const refreshRecent = () => api.recent().then(setRecent, () => setRecent((cur) => cur ?? []));

  useEffect(() => {
    void refreshRecent();
  }, []);

  useEffect(() => {
    if (!parent) setParent(parentDefault);
  }, [parentDefault, parent]);

  const forget = async (path: string) => {
    const left = await guard(() => api.forgetRecent(path));
    if (left) setRecent(left);
  };

  const open = async (path: string) => {
    if (current?.path === path) return back();
    setBusy(true);
    let project = null;
    try {
      project = await api.openProject(path);
    } catch (err) {
      setBusy(false);
      // A folder that has gone since the list was read says so, and offers to take it out.
      const listed = await api.recent().catch(() => null);
      if (listed) setRecent(listed);
      const missing = listed?.find((p) => p.path === path)?.missing;
      useStore.getState().notify(err instanceof Error ? err.message : String(err), "error", {
        ...(missing ? { action: { label: "Remove from the list", run: () => void forget(path) } } : {}),
      });
      return;
    }
    setBusy(false);
    await enter(project);
  };

  const create = async () => {
    setBusy(true);
    const project = await guard(() => api.createProject(name.trim(), parent || null));
    setBusy(false);
    if (project) await enter(project);
  };

  return (
    <Welcome>
      <div className={s.wideColumn}>
        <span className="eyebrow">Your work</span>
        <h1 className={s.pageTitle}>Projects</h1>
        <p className={s.lead}>
          A project is a folder of datasets and experiments. Each experiment is a spec and its results, as
          plain files you can commit, share and rerun.
        </p>
        {current && (
          <div className={s.backRow}>
            <Button variant="primary" icon="arrowLeft" onClick={back}>
              Back to {current.name}
            </Button>
          </div>
        )}

        <div className={s.projectGrid}>
          <section>
            <h2 className={s.sectionTitle}>Recent</h2>
            {recent === null ? null : recent.length === 0 ? (
              <p className={s.muted}>Projects you open will be listed here.</p>
            ) : (
              <ul className={s.recent}>
                {recent.map((p) => (
                  <li key={p.path} className={s.recentRow}>
                    <button
                      type="button"
                      className={`${s.recentItem} ${p.missing ? s.recentMissing : ""}`}
                      onClick={() => void open(p.path)}
                      disabled={busy}
                    >
                      <Logogram seed={p.path} size={34} className={s.recentGlyph} />
                      <span className={s.recentText}>
                        {/* When the column is narrow, when it changed goes under its name. */}
                        <span className={s.recentHead}>
                          <span className={s.recentName}>{p.name}</span>
                          <span className={s.recentTime}>{p.missing ? (p.problem ?? "Can't be read") : `Modified ${ago(p.modified)}`}</span>
                        </span>
                        <span className={s.recentPath} title={p.path}>
                          {/* The left-to-right mark keeps the leading slash in place when truncating from the left. */}
                          {"\u200e" + p.path}
                        </span>
                      </span>
                    </button>
                    {p.missing ? (
                      <Button
                        size="small"
                        variant="ghost"
                        icon="close"
                        aria-label={`Remove \u201c${p.name}\u201d from the list`}
                        title="Remove from the list"
                        onClick={() => void forget(p.path)}
                      />
                    ) : (
                      <RecentMenu
                        name={p.name}
                        onForget={() => void forget(p.path)}
                        onDelete={() => setDeleting({ path: p.path, name: p.name })}
                      />
                    )}
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className={s.side}>
            <div className={s.block}>
              <h2 className={s.sectionTitle}>New project</h2>
              <form
                className={s.form}
                onSubmit={(e) => {
                  e.preventDefault();
                  if (name.trim()) void create();
                }}
              >
                <Field label="Name" htmlFor="project-name">
                  <Input
                    id="project-name"
                    value={name}
                    placeholder="Name mover heads"
                    onChange={(e) => setName(e.target.value)}
                  />
                </Field>
                <Field label="Location" htmlFor="project-parent" help="A new folder is created here.">
                  <div className={s.inline}>
                    <Input
                      id="project-parent"
                      value={parent}
                      onChange={(e) => setParent(e.target.value)}
                      spellCheck={false}
                    />
                    <Button onClick={() => setPicker("parent")}>Browse</Button>
                  </div>
                </Field>
                <div>
                  <Button variant="primary" type="submit" disabled={!name.trim() || busy}>
                    Create project
                  </Button>
                </div>
              </form>
            </div>

            <div className={s.block}>
              <h2 className={s.sectionTitle}>Open</h2>
              <div className={s.stack}>
                <Button icon="folder" onClick={() => setPicker("open")} disabled={busy}>
                  Open a project folder
                </Button>
              </div>
            </div>
            <button type="button" className={s.textLink} onClick={() => useStore.getState().goto("system")}>
              <Icon name="cpu" size={14} /> System check
            </button>
          </section>
        </div>

        <section className={s.examples} aria-labelledby="examples-title">
          {/* The guide's Start here can send the keyboard here. */}
          <h2 id="examples-title" className={s.sectionTitle} tabIndex={-1}>
            Examples
          </h2>
          <ExampleGallery disabled={busy} />
        </section>
      </div>
      <FolderPicker
        open={picker === "parent"}
        onOpenChange={(v) => !v && setPicker(null)}
        title="Choose a location"
        start={parent}
        chooseLabel="Use this folder"
        onChoose={setParent}
      />
      <FolderPicker
        open={picker === "open"}
        onOpenChange={(v) => !v && setPicker(null)}
        title="Open a project"
        start={parentDefault}
        chooseLabel="Open"
        requireProject
        onChoose={(path) => void open(path)}
      />
      <DeleteProjectDialog target={deleting} onOpenChange={(v) => !v && setDeleting(null)} onDeleted={() => void refreshRecent()} />
    </Welcome>
  );
}

/** A recent project's actions: take it out of the list, or delete it. Delete starts once the menu
 * has closed, so the keyboard lands in its dialog. */
function RecentMenu({ name, onForget, onDelete }: { name: string; onForget: () => void; onDelete: () => void }) {
  const pending = useRef<(() => void) | null>(null);
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <Button size="small" variant="ghost" icon="more" aria-label={`Actions for “${name}”`} />
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          className={menuClasses.menu}
          align="end"
          sideOffset={4}
          onCloseAutoFocus={(e) => {
            const action = pending.current;
            pending.current = null;
            if (!action) return;
            e.preventDefault();
            action();
          }}
        >
          <DropdownMenu.Item className={menuClasses.item} onSelect={onForget}>
            Remove from the list
          </DropdownMenu.Item>
          <DropdownMenu.Separator className={menuClasses.separator} />
          <DropdownMenu.Item
            className={menuClasses.item}
            onSelect={() => {
              pending.current = onDelete;
            }}
          >
            Delete project…
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
