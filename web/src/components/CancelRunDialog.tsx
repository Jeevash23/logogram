import { useEffect } from "react";

import { pct } from "../lib/format";
import { useStore } from "../store/app";
import { ConfirmDialog } from "./ui";

const close = () => useStore.setState({ cancelConfirmOpen: false });

/** Cancelling discards what a run has measured, so the status line and the palette ask first. */
export function CancelRunDialog() {
  const asked = useStore((st) => st.cancelConfirmOpen);
  const job = useStore((st) => st.job);
  const progress = useStore((st) => (st.cancelConfirmOpen && st.job?.run_id ? st.live[st.job.run_id]?.progress ?? null : null));
  const running = job?.status === "running" && job.kind === "run" && !job.cancelling;

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
      The results it has measured so far are discarded, and none of them are saved. Its spec and any dataset snapshot
      stay in its folder, so you can run it again.
    </ConfirmDialog>
  );
}
