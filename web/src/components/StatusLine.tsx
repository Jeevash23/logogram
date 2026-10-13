import { useEffect, useRef, useState } from "react";

import { bytes, count, duration, pct } from "../lib/format";
import { modelName } from "../lib/hooks";
import { milestone } from "../lib/milestones";
import { progressText } from "../lib/training";
import type { TrainingProgress } from "../api/types";
import { useStore } from "../store/app";
import { Button, Progress, Spinner } from "./ui";
import s from "./StatusLine.module.css";

export function StatusLine() {
  const job = useStore((st) => st.job);
  const model = useStore((st) => st.model);
  const version = useStore((st) => st.version);
  const connection = useStore((st) => st.connection);
  const connectionError = useStore((st) => st.connectionError);
  const live = useStore((st) => (st.job?.run_id ? st.live[st.job.run_id] : undefined));
  const queued = useStore((st) => st.queue.length);
  const queueState = useStore((st) => st.queueState);
  const queueRunner = useStore((st) => st.queueRunner);
  const said = useMilestones();
  // The queue, and how it stands: shown while a job runs, and while it waits paused or another
  // process runs it.
  const queueWord = queueRunner ? "run by another process" : queueState === "paused" ? "paused" : queueState === "running" ? "running" : "waiting";
  const queueText = queued > 0 ? `${count(queued)} queued · ${queueWord}` : "";

  const running = job?.status === "running";
  let text = "Ready";
  let progress: number | null = null;
  let detail = "";
  if (running && job.kind === "run") {
    const p = live?.progress;
    if (p) {
      progress = p.done / p.total;
      const eta = progress > 0.02 ? (p.elapsed_s * (1 - progress)) / progress : null;
      text = `Running “${job.title}” · layer ${p.layer}`;
      detail = `${count(p.done)} of ${count(p.total)} rows · ${pct(progress)}${eta !== null ? ` · about ${duration(eta)} left` : ""}`;
    } else if (model.state === "loading") {
      text = `Loading ${modelName(model.id)} for “${job.title}”`;
      if (model.stage === "downloading" && model.total) {
        progress = (model.done ?? 0) / model.total;
        detail = `downloading ${bytes(model.done)} of ${bytes(model.total)}`;
      } else detail = model.stage ?? "";
    } else {
      text = `Starting “${job.title}”`;
    }
  } else if (running && (job.kind === "train_sae" || job.kind === "train_lens")) {
    const what = job.kind === "train_sae" ? "SAE" : "lens";
    const name = job.title.replace(/^Train the (SAE|lens) /, "");
    const where = progressText(job.progress as TrainingProgress, what, name);
    text = `Training the ${what} ${name}`;
    detail = where.text;
    progress = where.share;
  } else if (running && job.kind === "load_model") {
    text = `Loading ${modelName(model.id)}`;
    if (model.stage === "downloading" && model.total) {
      progress = (model.done ?? 0) / model.total;
      detail = `downloading ${bytes(model.done)} of ${bytes(model.total)}${model.file ? ` · ${model.file}` : ""}`;
    } else detail = model.stage === "resolving" ? "checking the revision" : model.stage ?? "";
  } else if (running) {
    // Another job, such as a model folder hashed or a corpus downloaded: its title, and how far it
    // has got (in bytes for the jobs that count bytes).
    text = job.title;
    const p = (job.progress ?? {}) as { done?: unknown; total?: unknown };
    if (typeof p.total === "number" && p.total > 0 && typeof p.done === "number") {
      progress = p.done / p.total;
      const inBytes = job.kind === "add_model" || job.kind === "import_corpus";
      detail = inBytes ? `${bytes(p.done)} of ${bytes(p.total)}` : `${count(p.done)} of ${count(p.total)}`;
    }
  } else if (job && job.status === "failed") {
    text = `“${job.title}” failed`;
    detail = job.error ?? "";
  } else if (job && job.status === "cancelled") {
    text = `“${job.title}” was cancelled`;
  }

  // While a run is going, its progress events carry the current reading.
  const memory = (running ? live?.progress?.memory : null) ?? model.memory ?? null;
  if (connection === "ended") {
    text = "Session ended";
    detail = connectionError ?? "Open the link printed in the terminal where Logogram is running.";
    progress = null;
  } else if (connection !== "connected") {
    text = connection === "connecting" ? "Connecting to the server…" : "Disconnected · reconnecting…";
    detail = connectionError ?? "Keep the Logogram terminal running. Displayed progress may be out of date.";
    progress = null;
  }
  return (
    // Not a live region itself: it changes several times a second during a run. Screen readers
    // hear the milestones instead.
    <footer className={s.status}>
      <div className="visually-hidden" role="status" aria-live="polite" aria-atomic="true">
        {said}
      </div>
      <div className={s.left}>
        {running && <Spinner />}
        <span className={s.text}>{text}</span>
        {detail && <span className={s.detail}>{detail}</span>}
        {queueText && (running || queueState === "paused" || queueRunner) && <span className={s.queued}>{queueText}</span>}
        {running && (
          <div className={s.bar}>
            <Progress value={progress} label={job.kind === "run" ? "Progress of the run" : "Progress of the job"} />
          </div>
        )}
        {running && (job.kind === "run" || job.kind === "train_sae" || job.kind === "train_lens") && (
          <Button size="small" onClick={() => useStore.setState({ cancelConfirmOpen: true })} disabled={job.cancelling}>
            {job.cancelling ? "Cancelling…" : "Cancel"}
          </Button>
        )}
      </div>
      <div className={s.right}>
        {model.info && (
          <span>
            {modelName(model.info.id)} · {model.info.device_name}
            {memory !== null && memory !== undefined &&
              ` · ${bytes(memory)} ${model.info.device === "cpu" ? "of RAM" : "of GPU memory"} in use`}
          </span>
        )}
        <span className={s.version}>Logogram {version}</span>
      </div>
    </footer>
  );
}

/** The latest milestone of the work in progress (see milestone), said once when it is reached.
 * What was already true when the page opened isn't news. */
function useMilestones(): string {
  const job = useStore((st) => st.job);
  const progress = useStore((st) => (st.job?.run_id ? st.live[st.job.run_id]?.progress ?? null : null));
  const connection = useStore((st) => st.connection);
  const now = milestone(job, progress, connection);
  const [said, setSaid] = useState("");
  const last = useRef<string | null>(null);
  useEffect(() => {
    if (last.current === null || last.current === now.key) {
      last.current = now.key;
      return;
    }
    const reconnected = last.current === "reconnecting";
    last.current = now.key;
    setSaid(reconnected ? `Connected to the server again. ${now.text}`.trim() : now.text);
  }, [now.key, now.text]);
  return said;
}
