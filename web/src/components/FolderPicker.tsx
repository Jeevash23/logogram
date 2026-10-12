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

  // Only the folder asked for last is shown: the first listing, arriving after a typed path was
  // listed, doesn't take its place.
  const latest = useRef(0);
  const go = async (path?: string) => {
    const request = ++latest.current;
    try {
      const next = await api.folders(path);
      if (request !== latest.current) return;
      setListing(next);
      setTyped(next.path);
      setError(null);
    } catch (e) {
      if (request !== latest.current) return;
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  useEffect(() => {
    if (open) void go(start || undefined);
  }, [open, start]);

  const canChoose = listing && (!requireProject || listing.is_project);

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
          <Button
            variant="primary"
            disabled={!canChoose}
            onClick={() => {
              if (listing) {
                onChoose(listing.path);
                onOpenChange(false);
              }
            }}
          >
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
        <Input value={typed} onChange={(e) => setTyped(e.target.value)} spellCheck={false} aria-label="Folder path" />
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
