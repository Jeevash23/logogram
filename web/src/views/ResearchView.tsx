import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { NoteInput, ResearchNote, SiteSpec } from "../api/types";
import { Button, Callout, Dialog, Empty, Field, Input, Spinner, TextArea } from "../components/ui";
import { ago, shortRevision } from "../lib/format";
import { modelName } from "../lib/hooks";
import { isResidKind, partOfKind, type Selection } from "../lib/sites";
import { positionKey, positionText, siteText } from "../lib/spec";
import { useStore } from "../store/app";
import s from "./views.module.css";
import r from "./ResearchView.module.css";

export function selectionFromNote(site: SiteSpec): Selection {
  return { layer: site.layer, part: partOfKind(site.kind), head: site.head ?? undefined,
    kind: isResidKind(site.kind) ? site.kind : undefined,
    positionKey: site.position.kind === "all" ? undefined : positionKey(site.position) };
}

export function ResearchView() {
  const project = useStore(st => st.project?.session_id);
  const version = useStore(st => st.researchVersion);
  const [notes, setNotes] = useState<ResearchNote[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [removing, setRemoving] = useState<ResearchNote | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true); setError(null);
    api.research().then(data => { if (!cancelled) setNotes(data.notes); }, (e: Error) => { if (!cancelled) setError(e.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [project, version]);

  const restore = async (note: ResearchNote) => {
    setError(null); setBusy(true);
    try {
      const before = useStore.getState().project?.session_id;
      if (note.run_id) {
        await useStore.getState().openRun(note.run_id);
        if (before !== useStore.getState().project?.session_id) return;
        if (!useStore.getState().runDetails[note.run_id]) throw new Error("This selection’s run is unavailable. Restore its run folder to this project to open it.");
      } else {
        const info = useStore.getState().model.info;
        const m = note.model;
        if (!info || info.id !== m.id || (m.revision && m.revision !== info.revision) || info.dtype !== m.dtype || info.process_weights !== m.process_weights || (m.device !== "auto" && m.device !== info.device)) {
          throw new Error("Load this note’s model with the recorded revision, dtype, device and weight processing, then open the selection again.");
        }
        await useStore.getState().openRun(null);
        useStore.setState({ analysisSource: "form" });
      }
      useStore.setState({ view: "explore", exploreMode: "layer", selection: note.sites.length ? selectionFromNote(note.sites[0]) : null, stagedSites: note.sites, tokenPosition: null });
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };

  const remove = async () => {
    if (!removing) return;
    setBusy(true); setError(null);
    try {
      await api.deleteNote(removing);
      setRemoving(null);
      useStore.setState(st => ({ researchVersion: st.researchVersion + 1 }));
    } catch (e) { setError((e as Error).message); setRemoving(null); }
    finally { setBusy(false); }
  };

  return <div className={s.view}>
    <div className={s.head}><div className={s.titleBlock}><h1 className={s.title}>Research notes</h1><p className={s.subtitle}>Keep a selection, its provenance, and the question you want to investigate next.</p></div><Button onClick={() => useStore.getState().setView("explore")}>Choose a component</Button></div>
    <p className={s.small}>Saved in this project’s research.json. Model, revision, sites, positions, and source run travel with each note.</p>
    {error && <Callout tone="error">{error}</Callout>}
    {loading && <Spinner label="Reading notes…" />}
    {!loading && !error && notes.length === 0 && <Empty title="Keep your first observation" action={<Button variant="primary" onClick={() => useStore.getState().setView("explore")}>Explore the model</Button>}>Select a component or several experiment sites, then choose Save selection to add a name and research notes.</Empty>}
    <div className={r.notebook}>{notes.map(note => <article key={note.id} className={r.note}>
      <div className={r.noteHeading}><div><h2>{note.title}</h2><p className={s.small}>{modelName(note.model.id)} · revision {shortRevision(note.model.revision)} · {note.model.dtype} · {note.model.device} · weights {note.model.process_weights ? "processed" : "original"}</p></div><time className={s.faint} dateTime={note.updated} title={note.updated}>{ago(note.updated)}</time></div>
      {note.body ? <p className={r.body}>{note.body}</p> : <p className={s.faint}>No written observation yet.</p>}
      <div className={r.sites}>{note.sites.map((site, i) => <span key={i} title={positionText(site.position)}>{siteText(site)}{site.position.kind === "all" ? " · all positions" : ""}</span>)}</div>
      <div className={r.footer}><span className={s.faint}>{note.run_id ? `Source run · ${note.run_id}` : "Saved from model exploration"}</span><div className={s.headActions}><Button size="small" variant="ghost" disabled={busy} onClick={() => setRemoving(note)}>Delete</Button><Button size="small" disabled={busy} onClick={() => useStore.setState({ noteEditor: { value: inputOf(note), edit: { id: note.id, revision: note.revision } } })}>Edit note</Button><Button size="small" disabled={busy} onClick={() => void restore(note)}>Open selection</Button></div></div>
    </article>)}</div>
    <Dialog open={!!removing} onOpenChange={open => { if (!open && !busy) setRemoving(null); }} title="Delete this research note?" description={removing?.title} footer={<><Button disabled={busy} onClick={() => setRemoving(null)}>Keep note</Button><Button variant="primary" disabled={busy} onClick={() => void remove()}>Delete note</Button></>}><p>The saved selection and written observation will be removed from this project. Its experiment run will remain available.</p></Dialog>
  </div>;
}

function inputOf(note: ResearchNote): NoteInput {
  const { title, body, model, sites, run_id } = note;
  return { title, body, model, sites, run_id };
}

export function NoteEditorDialog() {
  const editor = useStore(st => st.noteEditor);
  return editor ? <NoteEditor key={`${editor.edit?.id ?? "new"}-${editor.edit?.revision ?? 0}`} editor={editor} /> : null;
}

function NoteEditor({ editor }: { editor: { value: NoteInput; edit?: { id: string; revision: number } } }) {
  const [title, setTitle] = useState(editor.value.title);
  const [body, setBody] = useState(editor.value.body);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const save = async (asNew = false) => {
    setBusy(true); setError(null);
    try {
      await api.saveNote({ ...editor.value, title: title.trim(), body }, asNew ? undefined : editor.edit);
      useStore.setState(st => ({ noteEditor: null, researchVersion: st.researchVersion + 1 }));
      useStore.getState().notify("Selection and research note saved.");
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };
  return <Dialog open onOpenChange={open => { if (!open && !busy) useStore.setState({ noteEditor: null }); }} title={editor.edit ? "Edit research note" : "Save selection"} description={`${editor.value.sites.length} site${editor.value.sites.length === 1 ? "" : "s"} · ${modelName(editor.value.model.id)} · revision ${shortRevision(editor.value.model.revision)}`} footer={<>
    <Button disabled={busy} onClick={() => useStore.setState({ noteEditor: null })}>Cancel</Button>
    {error && editor.edit && <Button disabled={busy || !title.trim()} onClick={() => void save(true)}>Save as a new note</Button>}
    <Button variant="primary" disabled={busy || !title.trim()} onClick={() => void save()}>{busy ? "Saving…" : "Save note"}</Button>
  </>}><div className={r.editor}>
    <Field label="Title"><Input value={title} onChange={e => setTitle(e.target.value)} maxLength={200} autoFocus /></Field>
    <Field label="Observation or question"><TextArea value={body} onChange={e => setBody(e.target.value)} maxLength={50000} rows={8} placeholder="What did you observe? What would you test next?" /></Field>
    <div className={r.sites}>{editor.value.sites.map((site, i) => <span key={i}>{siteText(site)} · {positionText(site.position)}</span>)}</div>
    {error && <Callout tone="error">{error} Your text is still here. You can save a new note to keep both versions.</Callout>}
  </div></Dialog>;
}
