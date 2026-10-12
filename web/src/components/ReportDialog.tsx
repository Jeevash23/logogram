import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { ProblemReport } from "../api/types";
import { errorForReport, NEW_ISSUE_URL } from "../lib/help";
import { useStore } from "../store/app";
import { copyText } from "./CopyCommand";
import { Button, Callout, Checkbox, Dialog, Spinner } from "./ui";
import s from "./HelpDialogs.module.css";

const close = () => useStore.setState({ helpDialog: null });

/**
 * Report a problem: a short report, built by the server and sent nowhere, to copy into a new
 * issue. It names versions, the operating system's family, the device, the loaded model and the
 * last error the app showed; hardware details only when asked for.
 */
export function ReportDialog() {
  const open = useStore((st) => st.helpDialog === "report");
  const lastError = useStore((st) => st.lastError);
  const [hardware, setHardware] = useState(false);
  const [report, setReport] = useState<ProblemReport | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // Each report starts afresh, without hardware details.
  useEffect(() => {
    if (open) return;
    setHardware(false);
    setReport(null);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    let current = true;
    setFailure(null);
    setCopied(false);
    api.report({ include_hardware: hardware, last_error: errorForReport(lastError) }).then(
      (built) => {
        if (current) setReport(built);
      },
      (error: Error) => {
        if (current) {
          setReport(null);
          setFailure(error.message);
        }
      },
    );
    return () => {
      current = false;
    };
  }, [open, hardware, lastError]);

  const shown = report && report.hardware === hardware ? report : null;
  const quotesError = shown?.rows.some((row) => row.label === "Last error");
  return (
    <Dialog
      open={open}
      onOpenChange={(value) => {
        if (!value) close();
      }}
      title="Report a problem"
      description="Nothing is sent from here. Read the report, copy it, and paste it into a new issue on GitHub."
      wide
      footer={
        <>
          {/* There even when the report can't be built: the problem may be that. */}
          <a className={s.linkButton} href={NEW_ISSUE_URL} target="_blank" rel="noopener noreferrer">
            Open a new issue
            <span className="visually-hidden"> (opens in a new tab)</span>
            <span aria-hidden="true"> ↗</span>
          </a>
          <Button
            variant="primary"
            icon={copied ? "check" : "copy"}
            disabled={!shown}
            onClick={async () => {
              if (shown && (await copyText(shown.text))) setCopied(true);
            }}
          >
            {copied ? "Copied" : "Copy the report"}
          </Button>
        </>
      }
    >
      <div className={s.option}>
        <Checkbox checked={hardware} onChange={setHardware}>
          Include hardware details
        </Checkbox>
        <p className={s.note}>
          Adds the processor, GPUs, memory and the operating system's version, which can tell computers apart. Names
          of people and computers, and folder paths, are never included.
        </p>
      </div>
      {failure ? (
        <Callout tone="error" title="The report couldn't be built">
          {failure} In a terminal, <code>logogram doctor --report</code> prints the same report without the app.
        </Callout>
      ) : shown ? (
        <section aria-label="The report">
          <dl className={s.rows}>
            {shown.rows.map((row) => (
              <div key={row.label} className={s.row}>
                <dt>{row.label}</dt>
                <dd>{row.value}</dd>
              </div>
            ))}
          </dl>
          {!quotesError && <p className={s.note}>No error has been shown since this page opened.</p>}
        </section>
      ) : (
        <p className={s.loading}>
          <Spinner label="Building the report" /> Building the report…
        </p>
      )}
      {/* Always in the page, so that a screen reader hears it fill; out of sight while empty. */}
      <p className={copied ? s.status : "visually-hidden"} role="status">
        {copied ? "Copied. Paste it into the new issue's Problem report field." : ""}
      </p>
    </Dialog>
  );
}
