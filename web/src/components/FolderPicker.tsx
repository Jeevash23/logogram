import { useEffect, useRef, useState } from "react";

import { api } from "../api/client";
import type { FolderListing } from "../api/types";
import { Button, Dialog, Icon, Input } from "./ui";
import s from "./FolderPicker.module.css";

/** Browse folders on this machine (served by the local server; directories only). */
export function FolderPicker({
  open,
  onOpenChange,
  title,
  start,
  onChoose,
  chooseLabel,
  requireProject = false,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  title: string;
  start?: string;
  onChoose: (path: string) => void;
  chooseLabel: string;
  requireProject?: boolean;
}) {
  const [listing, setListing] = useState<FolderListing | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [typed, setTyped] = useState("");
  const [pending, setPending] = useState(false);

  // Only the folder asked for last is shown: the first listing, arriving after a typed path was
  // listed, doesn't take its place. Nor does a listing replace a path typed since it was asked
  // for: Go would then list the folder the picker opened in.
  const latest = useRef(0);
  const edited = useRef(false);
  const go = async (path?: string): Promise<FolderListing | null> => {
    const request = ++latest.current;
    edited.current = false;
    setPending(true);
    try {
      const next = await api.folders(path);
      if (request !== latest.current) return null;
      setListing(next);
      if (!edited.current) setTyped(next.path);
      setError(null);
      return next;
    } catch (e) {
      if (request !== latest.current) return null;
      setError(e instanceof Error ? e.message : String(e));
      return null;
    } finally {
      if (request === latest.current) setPending(false);
    }
  };

  useEffect(() => {
    if (open) void go(start || undefined);
  }, [open, start]);

  // What is chosen is the folder in the box. One typed but not listed yet, or still on its way,
  // is listed first, so the folder listed before it is never chosen in its place.
  const listed = !pending && listing !== null && typed === listing.path;
  const canChoose = listed ? !requireProject || listing.is_project : typed.trim() !== "";
  const choose = async () => {
    const chosen = listed ? listing : await go(typed);
    if (chosen && (!requireProject || chosen.is_project)) {
      onChoose(chosen.path);
      onOpenChange(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title={title}
      description={requireProject ? "Choose a folder that contains project.json." : "Choose a folder."}
      footer={
        <>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button variant="primary" disabled={!canChoose} onClick={() => void choose()}>
            {chooseLabel}
          </Button>
        </>
      }
    >
      <form
        className={s.pathRow}
        onSubmit={(e) => {
          e.preventDefault();
          void go(typed);
        }}
      >
        <Button
          size="small"
          icon="arrowLeft"
          aria-label="Parent folder"
          disabled={!listing?.parent}
          onClick={() => listing?.parent && void go(listing.parent)}
        />
        <Input
          value={typed}
          onChange={(e) => {
            edited.current = true;
            setTyped(e.target.value);
          }}
          spellCheck={false}
          aria-label="Folder path"
        />
        <Button size="small" type="submit">
          Go
        </Button>
      </form>
      {error && <div className={s.error}>{error}</div>}
      <div className={s.list} role="listbox" aria-label="Folders">
        {listing?.entries.length === 0 && <div className={s.empty}>No folders here.</div>}
        {listing?.entries.map((entry) => (
          <button
            key={entry.path}
            type="button"
            className={s.entry}
            onClick={() => void go(entry.path)}
            onDoubleClick={() => {
              if (!requireProject || entry.is_project) {
                onChoose(entry.path);
                onOpenChange(false);
              }
            }}
          >
            <Icon name="folder" size={14} />
            <span className={s.name}>{entry.name}</span>
            {entry.is_project && <span className={s.tag}>project</span>}
          </button>
        ))}
      </div>
      {listing?.is_project && <div className={s.hint}>This folder is a Logogram project.</div>}
    </Dialog>
  );
}
