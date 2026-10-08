import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { RecentProject } from "../api/types";
import { FolderPicker } from "../components/FolderPicker";
import { Logogram } from "../components/Logogram";
import { Button, Field, Icon, Input } from "../components/ui";
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

  useEffect(() => {
    api.recent().then(setRecent, () => setRecent([]));
  }, []);

  useEffect(() => {
    if (!parent) setParent(parentDefault);
  }, [parentDefault, parent]);

  const open = async (path: string) => {
    setBusy(true);
    const project = await guard(() => api.openProject(path));
    setBusy(false);
    if (project) await enter(project);
  };

  const create = async () => {
    setBusy(true);
    const project = await guard(() => api.createProject(name.trim(), parent || null));
    setBusy(false);
    if (project) await enter(project);
  };

  const example = async () => {
    setBusy(true);
    const project = await guard(() => api.openExample());
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

        <div className={s.projectGrid}>
          <section>
            <h3 className={s.sectionTitle}>Recent</h3>
            {recent === null ? null : recent.length === 0 ? (
              <p className={s.muted}>Projects you open will be listed here.</p>
            ) : (
              <ul className={s.recent}>
                {recent.map((p) => (
                  <li key={p.path}>
                    <button type="button" className={s.recentItem} onClick={() => void open(p.path)} disabled={busy}>
                      <Logogram seed={p.path} size={34} className={s.recentGlyph} />
                      <span className={s.recentName}>{p.name}</span>
                      <span className={s.recentPath} title={p.path}>
                        {/* The left-to-right mark keeps the leading slash in place when truncating from the left. */}
                        {"\u200e" + p.path}
                      </span>
                      <span className={s.recentTime}>Modified {ago(p.modified)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className={s.side}>
            <div className={s.block}>
              <h3 className={s.sectionTitle}>New project</h3>
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
              <h3 className={s.sectionTitle}>Open</h3>
              <div className={s.stack}>
                <Button icon="folder" onClick={() => setPicker("open")} disabled={busy}>
                  Open a project folder
                </Button>
                <Button icon="layers" onClick={() => void example()} disabled={busy}>
                  Open the example project
                </Button>
                <p className={s.muted}>
                  The example measures which GPT-2 heads carry the answer in indirect object identification. It
                  runs in about a minute on a CPU.
                </p>
              </div>
            </div>
            <button type="button" className={s.textLink} onClick={() => useStore.getState().goto("system")}>
              <Icon name="cpu" size={14} /> System check
            </button>
          </section>
        </div>
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
    </Welcome>
  );
}
