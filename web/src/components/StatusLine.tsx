import { bytes, count, duration, pct } from "../lib/format";
import { modelName } from "../lib/hooks";
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
  const cancel = useStore((st) => st.cancelJob);

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
  } else if (running && job.kind === "load_model") {
    text = `Loading ${modelName(model.id)}`;
    if (model.stage === "downloading" && model.total) {
      progress = (model.done ?? 0) / model.total;
      detail = `downloading ${bytes(model.done)} of ${bytes(model.total)}${model.file ? ` · ${model.file}` : ""}`;
    } else detail = model.stage === "resolving" ? "checking the revision" : model.stage ?? "";
  } else if (job && job.status === "failed") {
    text = `“${job.title}” failed`;
    detail = job.error ?? "";
  } else if (job && job.status === "cancelled") {
    text = `“${job.title}” was cancelled`;
  }

  // While a run is going, its progress events carry the current reading.
  const memory = (running ? live?.progress?.memory : null) ?? model.memory ?? null;
  if (connection !== "connected") {
    text = connection === "connecting" ? "Connecting to the server…" : "Disconnected · reconnecting…";
    detail = connectionError ?? "Keep the Logogram terminal running. Displayed progress may be out of date.";
    progress = null;
  }
  return (
    <footer className={s.status} role="status" aria-live="polite">
      <div className={s.left}>
        {running && <Spinner />}
        <span className={s.text}>{text}</span>
        {detail && <span className={s.detail}>{detail}</span>}
        {running && (
          <div className={s.bar}>
            <Progress value={progress} />
          </div>
        )}
        {running && job.kind === "run" && (
          <Button size="small" onClick={() => void cancel()} disabled={job.cancelling}>
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
