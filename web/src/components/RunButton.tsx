import { useEffect } from "react";

import type { Spec } from "../api/types";
import { runWouldQueue, useStore } from "../store/app";
import { MOD } from "./Header";
import { Button } from "./ui";

/**
 * Runs the experiment. While a job runs, or runs wait their turn (or another process runs the
 * queue), it reads Queue instead and adds the experiment to the queue, to start when the ones
 * ahead of it end. A paused queue doesn't hold it back. Ctrl/⌘ Enter presses it from anywhere it
 * is shown.
 */
export function RunButton({
  spec,
  reason,
  onRun,
  onQueue,
}: {
  /** What to run, or null while it can't run. */
  spec: Spec | null;
  /** Why it can't run, said on hover and focus (the form also shows it beside the button). */
  reason?: string | null;
  /** Start the run now. */
  onRun: (spec: Spec) => void;
  /** Add it to the queue. */
  onQueue: (spec: Spec) => void;
}) {
  const queue = useStore(runWouldQueue);

  const press = () => {
    if (!spec) return;
    if (queue) onQueue(spec);
    else onRun(spec);
  };

  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") {
        ev.preventDefault();
        // Held down, the keys repeat: only the first press counts, or each would queue a run.
        if (!ev.repeat) press();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  return (
    <Button
      variant="primary"
      icon={queue ? "plus" : "play"}
      onClick={(ev) => {
        // A double click is one press: its second click would queue the run again.
        if (ev.detail < 2) press();
      }}
      disabled={!spec}
      tip={!spec ? (reason ?? undefined) : queue ? `Add to the queue: it starts when the jobs ahead of it end (${MOD}+Enter)` : `Run (${MOD}+Enter)`}
      aria-keyshortcuts="Control+Enter Meta+Enter"
    >
      {queue ? "Queue" : "Run"}
    </Button>
  );
}
