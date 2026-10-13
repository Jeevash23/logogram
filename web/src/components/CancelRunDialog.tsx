import { useEffect } from "react";

import { pct } from "../lib/format";
import { keptText } from "../lib/runs";
import { useStore } from "../store/app";
import { ConfirmDialog } from "./ui";

const close = () => useStore.setState({ cancelConfirmOpen: false });

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
  // A training is cancelled with nothing kept: the SAE or lens isn't written.
  const training = job?.kind === "train_sae" || job?.kind === "train_lens";
  const running = job?.status === "running" && (job.kind === "run" || training) && !job.cancelling;
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
      title={training ? "Cancel this training?" : "Cancel this run?"}
      keepLabel={training ? "Keep training" : "Keep running"}
      confirmLabel={training ? "Cancel the training" : "Cancel the run"}
      onConfirm={() => {
        close();
        void useStore.getState().cancelJob();
      }}
    >
      {training && job ? (
        <>“{job.title}” is still going. Nothing it has done is kept: the {job.kind === "train_sae" ? "SAE" : "lens"} isn&rsquo;t written.</>
      ) : (
        <>
          {job && <>“{job.title}” {done === null ? "has started" : `is ${pct(done)} done`}. </>}
          {keptText(run)} Its spec and any dataset snapshot stay in its folder, so you can run it again.
        </>
      )}
      {starts && queued === 1 && " The run waiting in the queue starts next."}
      {starts && queued > 1 && ` The ${queued} runs waiting in the queue start next, one at a time.`}
      {paused && queued > 0 && " The queue is paused, so nothing starts after it."}
    </ConfirmDialog>
  );
}
