import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { SystemReport } from "../api/types";
import { CopyCommand } from "../components/CopyCommand";
import { Button, Callout, Mark, Spinner } from "../components/ui";
import { bytes } from "../lib/format";
import { useStore } from "../store/app";
import s from "./Screens.module.css";

export function SystemCheck() {
  const [report, setReport] = useState<SystemReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [opening, setOpening] = useState(false);
  const version = useStore((st) => st.version);
  const project = useStore((st) => st.project);
  const firstRun = useStore((st) => st.firstRun);

  useEffect(() => {
    api.system().then(setReport, (e: Error) => setError(e.message));
  }, []);

  const markSeen = () => {
    if (firstRun) {
      useStore.setState({ firstRun: false });
      void api.settings({ system_check_seen: true }).catch(() => undefined);
    }
  };

  const openExample = async () => {
    setOpening(true);
    markSeen();
    const st = useStore.getState();
    const project = await st.guard(() => api.openExample());
    setOpening(false);
    if (project) await st.enterProject(project);
  };

  const gpu = report?.gpus[0];
  const backendName = report
    ? { cuda: "CUDA", mps: "Apple Metal (MPS)", cpu: "CPU" }[report.backend]
    : "";

  return (
    <div className={s.page}>
      <div className={s.column}>
        <div className={s.brand}>
          <Mark size={22} />
          <span className={s.wordmark}>Logogram</span>
          <span className={s.version}>{version}</span>
        </div>
        <h1>System check</h1>
        <p className={s.lead}>
          Logogram runs every experiment on this computer. This is what it found, and what it will use.
        </p>

        {!report && !error && (
          <div className={s.loadingRow}>
            <Spinner /> Checking the hardware and libraries…
          </div>
        )}
        {error && <Callout tone="error" title="The system check failed">{error}</Callout>}

        {report && (
          <>
            <dl className={s.facts}>
              <dt>Operating system</dt>
              <dd>
                {report.os} <span className="faint">({report.machine})</span>
              </dd>
              <dt>Processor</dt>
              <dd>
                {report.cpu}
                {report.cpu_cores ? (
                  <span className="faint">
                    {" "}
                    · {report.cpu_cores} cores, {report.cpu_threads} threads
                  </span>
                ) : null}
              </dd>
              <dt>Memory</dt>
              <dd>
                {bytes(report.memory_total)} <span className="faint">· {bytes(report.memory_available)} available</span>
              </dd>
              <dt>GPU</dt>
              <dd>{gpu ? gpu.name : <span className="faint">None found</span>}</dd>
              <dt>Video memory</dt>
              <dd>
                {gpu && report.backend === "cuda" ? (
                  <>
                    {bytes(gpu.memory_total)}
                    {gpu.memory_free !== null && <span className="faint"> · {bytes(gpu.memory_free)} free</span>}
                  </>
                ) : report.backend === "mps" ? (
                  <span className="faint">Shared with system memory</span>
                ) : (
                  <span className="faint">—</span>
                )}
              </dd>
              <dt>Compute backend</dt>
              <dd>
                <strong>{backendName}</strong>
                {report.torch_cuda && report.backend === "cuda" && (
                  <span className="faint"> · CUDA {report.torch_cuda}</span>
                )}
              </dd>
              <dt>Recommended precision</dt>
              <dd>
                <strong>{report.recommended_dtype}</strong>
              </dd>
              <dt>Libraries</dt>
              <dd className="faint">
                PyTorch {report.torch} · TransformerLens {report.transformer_lens ?? "missing"} · Python{" "}
                {report.python}
              </dd>
            </dl>
            <p className={s.note}>{report.precision_note}</p>

            {report.issues.length > 0 && (
              <div className={s.issues}>
                {report.issues.map((issue) => (
                  <Callout key={issue.title} tone={issue.severity === "error" ? "error" : "info"} title={issue.title}>
                    <p>{issue.detail}</p>
                    {issue.fix && <CopyCommand command={issue.fix} />}
                  </Callout>
                ))}
              </div>
            )}
            {report.issues.length === 0 && <p className={s.ok}>No problems found.</p>}
          </>
        )}

        <div className={s.actions}>
          <Button variant="primary" size="large" onClick={openExample} disabled={opening}>
            {opening ? "Opening…" : "Open the example project"}
          </Button>
          <Button
            variant="ghost"
            size="large"
            onClick={() => {
              markSeen();
              useStore.getState().goto(project ? "workbench" : "projects");
            }}
          >
            {project ? "Back to the project" : "Go to projects"}
          </Button>
        </div>
        <p className={s.privacy}>
          No account, no telemetry. The only network access is model downloads from Hugging Face that you start.
        </p>
      </div>
    </div>
  );
}
