import { Fragment, useState, type ReactNode } from "react";

import { api } from "../api/client";
import type { Spec } from "../api/types";
import { copyText, CopyCommand } from "../components/CopyCommand";
import { Button, Empty, Segmented } from "../components/ui";
import { duration } from "../lib/format";
import { useActiveRun } from "../lib/hooks";
import { useStore } from "../store/app";
import { buildSpec } from "./ExperimentView";
import s from "./views.module.css";
import v from "./SpecView.module.css";

/** Light typographic emphasis for JSON: keys in ink, punctuation faint. No color. */
function JsonBlock({ value }: { value: unknown }) {
  const text = JSON.stringify(value, null, 2);
  const lines = text.split("\n");
  return (
    <pre className={v.json} aria-label="Spec JSON">
      {lines.map((line, i) => {
        const m = line.match(/^(\s*)("(?:[^"\\]|\\.)*")(:\s?)(.*)$/);
        let content: ReactNode = line;
        if (m) {
          content = (
            <>
              {m[1]}
              <span className={v.key}>{m[2]}</span>
              <span className={v.punct}>{m[3]}</span>
              <span className={v.value}>{m[4]}</span>
            </>
          );
        }
        return (
          <Fragment key={i}>
            {content}
            {"\n"}
          </Fragment>
        );
      })}
    </pre>
  );
}

export function SpecView() {
  const source = useStore((st) => st.specSource);
  const run = useActiveRun();
  const form = useStore((st) => st.form);
  const model = useStore((st) => st.model);
  const dataset = useStore((st) => st.dataset);
  const datasetPath = useStore((st) => st.datasetPath);
  const [copied, setCopied] = useState(false);
  const guard = useStore((st) => st.guard);

  const draft = buildSpec(form, { model, datasetPath, datasetSha: dataset?.sha256 ?? null });
  const runSpec: Spec | undefined = run.detail?.spec;
  const showing: "run" | "draft" = source === "run" && runSpec ? "run" : "draft";
  const spec = showing === "run" ? runSpec : "spec" in draft ? draft.spec : undefined;
  const manifest = run.detail?.manifest;
  const folder = run.detail?.folder;

  return (
    <div className={s.view}>
      <div className={s.head}>
        <div className={s.titleBlock}>
          <h2 className={s.title}>Spec</h2>
          <p className={s.subtitle}>
            The complete description of an experiment. The app and <code>logogram run</code> execute the same spec,
            so their results are identical on the same machine.
          </p>
        </div>
        <div className={s.headActions}>
          <Segmented
            label="Which spec"
            value={showing}
            onChange={(src) => useStore.setState({ specSource: src })}
            options={[
              { value: "run", label: "Selected run", disabled: !runSpec },
              { value: "draft", label: "Experiment form" },
            ]}
          />
        </div>
      </div>

      {!spec ? (
        <Empty title="Nothing to show yet">
          {"error" in draft ? draft.error : "Select a run in the history, or set up an experiment."}
        </Empty>
      ) : (
        <div className={s.split}>
          <div className={v.main}>
            <div className={v.toolbar}>
              <span className={s.small}>
                {showing === "run" ? `${folder}/spec.json, as executed (revision and dataset hash pinned)` : "Draft from the experiment form"}
              </span>
              <div className={s.row}>
                {showing === "draft" && (
                  <Button
                    size="small"
                    onClick={async () => {
                      const st = useStore.getState();
                      const out = await guard(() => api.saveDraft(spec, st.form.draftId));
                      if (out && useStore.getState().project?.session_id === st.project?.session_id) {
                        // Running the form now fills this draft's folder.
                        st.setForm({ draftId: out.run_id });
                        st.notify(`Saved ${out.path}`);
                        await st.refreshRuns();
                      }
                    }}
                  >
                    Save without running
                  </Button>
                )}
                <Button
                  size="small"
                  icon={copied ? "check" : "copy"}
                  onClick={async () => {
                    if (await copyText(JSON.stringify(spec, null, 2) + "\n")) {
                      setCopied(true);
                      window.setTimeout(() => setCopied(false), 1400);
                    }
                  }}
                >
                  {copied ? "Copied" : "Copy"}
                </Button>
              </div>
            </div>
            <JsonBlock value={spec} />
          </div>
          <aside className={v.aside}>
            {showing === "run" && folder && (
              <>
                <h3 className={s.panelTitle}>Rerun from a terminal</h3>
                <p className={s.small}>From the project folder:</p>
                <CopyCommand command={`logogram run ${folder}/spec.json`} />
                <p className={s.faint}>
                  It writes a new experiment folder and reports whether every per-prompt value matches this run.
                </p>
                <h3 className={s.panelTitle} style={{ marginTop: 16 }}>
                  Files
                </h3>
                <ul className={v.files}>
                  <li>
                    <strong>spec.json</strong> the experiment
                  </li>
                  <li>
                    <strong>results.parquet</strong> per prompt and site
                  </li>
                  <li>
                    <strong>summary.json</strong> per-site statistics
                  </li>
                  <li>
                    <strong>manifest.json</strong> versions, device, timing
                  </li>
                </ul>
              </>
            )}
            {showing === "run" && manifest?.versions && (
              <>
                <h3 className={s.panelTitle} style={{ marginTop: 16 }}>
                  Manifest
                </h3>
                <dl className={s.kv}>
                  {Object.entries(manifest.versions).map(([k, val]) => (
                    <Fragment key={k}>
                      <dt>{k.replace("_", " ")}</dt>
                      <dd>{val ?? "—"}</dd>
                    </Fragment>
                  ))}
                  {manifest.device && (
                    <>
                      <dt>device</dt>
                      <dd>
                        {manifest.device.type} · {manifest.device.name}
                      </dd>
                    </>
                  )}
                  {manifest.dtype && (
                    <>
                      <dt>dtype</dt>
                      <dd>{manifest.dtype}</dd>
                    </>
                  )}
                  {manifest.model && (
                    <>
                      <dt>revision</dt>
                      <dd title={manifest.model.revision}>{manifest.model.revision?.slice(0, 12)}</dd>
                    </>
                  )}
                  {manifest.wall_time_s !== undefined && (
                    <>
                      <dt>wall time</dt>
                      <dd>{duration(manifest.wall_time_s)}</dd>
                    </>
                  )}
                </dl>
              </>
            )}
            {showing === "draft" && (
              <>
                <h3 className={s.panelTitle}>Run it</h3>
                <p className={s.small}>
                  Press Run in the experiment form, or save it and run <code>logogram run</code> on the saved file.
                </p>
              </>
            )}
          </aside>
        </div>
      )}
    </div>
  );
}
