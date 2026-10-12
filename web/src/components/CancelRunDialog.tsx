import { useEffect } from "react";

import type { RunListing } from "../api/types";
import { pct } from "../lib/format";
import { useStore } from "../store/app";
import { ConfirmDialog } from "./ui";

const close = () => useStore.setState({ cancelConfirmOpen: false });

/** What a run keeps when it is cancelled, by how it measures (src/logogram/partial.py). */
function keptText(run: RunListing | undefined): string {
  const kept = "What it has finished is kept and marked as partial";
  if (!run) return `${kept}. What is in progress isn't kept.`;
  if (run.scope.kind === "features") return "A run of SAE features reports nothing before it ends, so it keeps no results.";
  if (run.experiment.kind === "attribution_patching" || run.experiment.kind === "direct_logit_attribution") {
    return `${kept}: every site, over the prompts that finished. The prompts in progress aren't kept.`;
  }
  if (run.scope.kind === "site_sets") return `${kept}: the sets that finished. The set in progress isn't kept.`;
  if (run.experiment.kind === "steering") {
    const control = run.experiment.control ? ", each with its random control" : "";
    return `${kept}: the sites that finished${control}. The site in progress isn't kept.`;
  }
  if (run.scope.kind === "sites") return `${kept}: the sites that finished. The site in progress isn't kept.`;
  return `${kept}: the layers that finished. The layer in progress isn't kept.`;
}

/** Cancelling stops a run before the end (it keeps what finished), so the status line and the
 * palette ask first. */
export function CancelRunDialog() {
  const asked = useStore((st) => st.cancelConfirmOpen);
  const job = useStore((st) => st.job);
  const progress = useStore((st) => (st.cancelConfirmOpen && st.job?.run_id ? st.live[st.job.run_id]?.progress ?? null : null));
  const queued = useStore((st) => st.queue.length);
  // Only a queue this server runs starts its next run when this one ends.
  const starts = useStore((st) => !st.queueRunner && (st.queueState === "waiting" || st.queueState === "running"));
  const paused = useStore((st) => st.queueState === "paused");
  const running = job?.status === "running" && job.kind === "run" && !job.cancelling;
  const run = useStore((st) => (st.job?.run_id ? st.runs.find((r) => r.id === st.job?.run_id) : undefined));

  // The run may end while the question is open: there is nothing left to cancel.
  useEffect(() => {
    if (asked && !running) close();
  }, [asked, running]);

  const done = progress && progress.total > 0 ? progress.done / progress.total : null;
  return (
    <ConfirmDialog
      open={asked && running}
      onOpenChange={(open) => {
        if (!open) close();
      }}
      title="Cancel this run?"
      keepLabel="Keep running"
      confirmLabel="Cancel the run"
      onConfirm={() => {
        close();
        void useStore.getState().cancelJob();
      }}
    >
      {job && <>“{job.title}” {done === null ? "has started" : `is ${pct(done)} done`}. </>}
      {keptText(run)} Its spec and any dataset snapshot stay in its folder, so you can run it again.
      {starts && queued === 1 && " The run waiting in the queue starts next."}
      {starts && queued > 1 && ` The ${queued} runs waiting in the queue start next, one at a time.`}
      {paused && queued > 0 && " The queue is paused, so nothing starts after it."}
    </ConfirmDialog>
  );
}
