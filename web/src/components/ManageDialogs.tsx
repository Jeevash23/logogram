// Renaming and deleting projects and datasets. Every deletion says exactly what goes and what
// stays; a project's folder goes only once its name is typed, and the server refuses a folder that
// holds what Logogram didn't make, a dataset a draft or a queued run names, or anything in use.

import { useEffect, useId, useState } from "react";

import { api } from "../api/client";
import type { DeletionPlan } from "../api/types";
import { plural } from "../lib/format";
import { nameProblem } from "../lib/promptEdits";
import { useStore } from "../store/app";
import { Button, Callout, ConfirmDialog, Dialog, Field, Input, Spinner } from "./ui";

function messageOf(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** Give the open project another name: its project.json says it; its folder keeps its own. */
export function RenameProjectDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const project = useStore((st) => st.project);
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const field = useId();
  useEffect(() => {
    if (!open) return;
    setName(project?.name ?? "");
    setError(null);
  }, [open]);
  const text = name.trim().split(/\s+/).join(" ");

  const save = async () => {
    if (!text || saving) return;
    if (text === project?.name) return onOpenChange(false);
    setSaving(true);
    setError(null);
    try {
      const info = await api.renameProject(text);
      onOpenChange(false);
      await useStore.getState().refreshProject();
      useStore.getState().notify(`The project is now called “${info.name}”. Its folder keeps its name.`);
    } catch (err) {
      setError(messageOf(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title="Rename the project"
      description="Its project.json says the new name. Its folder keeps its own name, so its path stays the same."
      footer={
        <>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button variant="primary" disabled={!text || saving} onClick={() => void save()}>
            Rename
          </Button>
        </>
      }
    >
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void save();
        }}
      >
        <Field label="Name" htmlFor={field}>
          <Input id={field} autoFocus value={name} maxLength={200} spellCheck={false} onChange={(e) => setName(e.target.value)} />
        </Field>
      </form>
      {error && (
        <Callout tone="error" title="The project wasn't renamed">
          {error}
        </Callout>
      )}
    </Dialog>
  );
}

/** Delete a project's folder for good, once its name is typed: what goes is said first, and a
 * project that can't be deleted (a folder holding what Logogram didn't make, a run in progress)
 * says why instead. */
export function DeleteProjectDialog({
  target,
  onOpenChange,
  onDeleted,
}: {
  /** The project to delete: its folder, and its name as shown. */
  target: { path: string; name: string } | null;
  onOpenChange: (open: boolean) => void;
  onDeleted?: (plan: DeletionPlan) => void;
}) {
  const current = useStore((st) => st.project);
  const [plan, setPlan] = useState<DeletionPlan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [typed, setTyped] = useState("");
  const [deleting, setDeleting] = useState(false);
  const field = useId();
  const path = target?.path;
  useEffect(() => {
    setPlan(null);
    setTyped("");
    setError(null);
    if (!path) return;
    let cancelled = false;
    api.deletionPlan(path).then(
      (found) => {
        if (!cancelled) setPlan(found);
      },
      (err) => {
        if (!cancelled) setError(messageOf(err));
      },
    );
    return () => {
      cancelled = true;
    };
  }, [path]);
  const matches = plan !== null && plan.problem === null && typed.trim() === plan.name;

  const remove = async () => {
    if (!plan || !matches || deleting) return;
    setDeleting(true);
    setError(null);
    const open = current?.path === plan.path;
    try {
      await api.deleteProject(plan.path, typed.trim());
    } catch (err) {
      setError(messageOf(err));
      setDeleting(false);
      return;
    }
    setDeleting(false);
    onOpenChange(false);
    const st = useStore.getState();
    // The server closed it first, and tells every tab; this one leaves it now.
    if (open) st.leaveProject();
    onDeleted?.(plan);
    st.notify(`Deleted “${plan.name}” and its folder.`);
  };

  return (
    <Dialog
      open={target !== null}
      onOpenChange={onOpenChange}
      title={`Delete “${target?.name ?? ""}”?`}
      description="The project's folder and everything in it are deleted for good."
      footer={
        <>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            {plan?.problem ? "Close" : "Keep it"}
          </Button>
          {!plan?.problem && (
            <Button variant="primary" disabled={!matches || deleting} onClick={() => void remove()}>
              Delete the project
            </Button>
          )}
        </>
      }
    >
      {!plan && !error && (
        <p className="faint">
          <Spinner /> Reading the project…
        </p>
      )}
      {plan?.problem && (
        <Callout tone="error" title="This project can't be deleted">
          {plan.problem}
        </Callout>
      )}
      {plan && !plan.problem && (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void remove();
          }}
        >
          <p>{plan.text}</p>
          <Field label={`Type the project's name, “${plan.name}”, to delete it`} htmlFor={field}>
            <Input id={field} autoFocus value={typed} autoComplete="off" spellCheck={false} onChange={(e) => setTyped(e.target.value)} />
          </Field>
        </form>
      )}
      {error && (
        <Callout tone="error" title="Nothing was deleted">
          {error}
        </Callout>
      )}
    </Dialog>
  );
}

/** A dataset of the open project, as a menu or a list names it. */
export interface DatasetTarget {
  path: string;
  name: string;
  n?: number;
}

/** Give a dataset file another name. Its prompts, and so its hash, stay the same. */
export function RenameDatasetDialog({ target, onOpenChange }: { target: DatasetTarget | null; onOpenChange: (open: boolean) => void }) {
  const datasets = useStore((st) => st.project?.datasets);
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const field = useId();
  useEffect(() => {
    if (!target) return;
    setName(target.name.replace(/\.jsonl$/, ""));
    setError(null);
  }, [target?.path]);
  const stem = name.trim().replace(/\.jsonl$/, "");
  const same = target !== null && `${stem}.jsonl` === target.name;
  const taken = new Set((datasets ?? []).map((d) => d.name).filter((n) => n !== target?.name));
  const problem = stem ? nameProblem(stem, taken) : "Give the dataset a name.";

  const save = async () => {
    if (!target || problem || saving) return;
    if (same) return onOpenChange(false);
    setSaving(true);
    setError(null);
    try {
      const out = await api.renameDataset(target.name, stem);
      onOpenChange(false);
      const st = useStore.getState();
      await st.refreshProject();
      st.followDataset(target.path, out.path);
      st.notify(`Renamed ${target.name} to ${out.name}.`);
    } catch (err) {
      setError(messageOf(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog
      open={target !== null}
      onOpenChange={onOpenChange}
      title={`Rename ${target?.name ?? ""}`}
      description="The file gets the new name; its prompts stay the same. Runs that ran on it keep their own copy of its prompts."
      footer={
        <>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button variant="primary" disabled={problem !== null || saving} onClick={() => void save()}>
            Rename
          </Button>
        </>
      }
    >
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void save();
        }}
      >
        <Field label="Name" htmlFor={field} help={problem ?? `Saved as datasets/${stem}.jsonl`}>
          <Input
            id={field}
            autoFocus
            value={name}
            maxLength={106}
            spellCheck={false}
            aria-invalid={problem !== null}
            aria-describedby={`${field}-help`}
            onChange={(e) => setName(e.target.value)}
          />
        </Field>
      </form>
      {error && (
        <Callout tone="error" title="The dataset wasn't renamed">
          {error}
        </Callout>
      )}
    </Dialog>
  );
}

/** Delete a dataset file, saying that runs keep their own copy of its prompts. A draft or a queued
 * run that names it keeps it: the server says which. */
export function DeleteDatasetDialog({ target, onOpenChange }: { target: DatasetTarget | null; onOpenChange: (open: boolean) => void }) {
  const edited = useStore((st) => (target ? st.datasetEdits[target.path] : undefined));
  const remove = async () => {
    if (!target) return;
    onOpenChange(false);
    const st = useStore.getState();
    const out = await st.guard(() => api.deleteDataset(target.name));
    if (!out) return;
    st.followDataset(target.path, null);
    await st.refreshProject();
    st.notify(`Deleted ${out.path}. Runs that ran on it keep their own copy of its prompts.`);
  };
  const prompts = target?.n !== undefined ? ` and its ${plural(target.n, "prompt pair")}` : "";
  return (
    <ConfirmDialog
      open={target !== null}
      onOpenChange={onOpenChange}
      title={`Delete ${target?.name ?? ""}?`}
      keepLabel="Keep it"
      confirmLabel="Delete the dataset"
      onConfirm={() => void remove()}
    >
      {target
        ? `This deletes ${target.path}${prompts}, for good.${edited ? " Its unsaved edits go with it." : ""} Runs that ran on it keep their own copy of its prompts, in datasets/snapshots/, so their results and prompts stay as they are.`
        : ""}
    </ConfirmDialog>
  );
}
