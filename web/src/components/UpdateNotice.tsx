import * as Popover from "@radix-ui/react-popover";
import { useState } from "react";

import { ago } from "../lib/format";
import { useStore } from "../store/app";
import { CopyCommand } from "./CopyCommand";
import { Button, Checkbox, Spinner } from "./ui";
import s from "./UpdateNotice.module.css";

const DISMISSED = "logogram.dismissedUpdate";

function dismissed(): string | null {
  try {
    return localStorage.getItem(DISMISSED);
  } catch {
    return null;
  }
}

/**
 * A new version, or an old one: a small pill in the top bar that opens the details. Nothing
 * here contacts the network unless the user presses Check now or allows daily checks.
 */
export function UpdateNotice() {
  const update = useStore((st) => st.update);
  const [hidden, setHidden] = useState(dismissed);
  if (!update) return null;
  const fresh = update.available && update.latest !== hidden;
  if (!fresh && !update.old) return null;
  return (
    <Popover.Root>
      <Popover.Trigger className={fresh ? s.pillNew : s.pillOld}>
        <span className={s.dot} aria-hidden="true" />
        {fresh ? `Logogram ${update.latest}` : "Check for updates"}
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content className={s.panel} sideOffset={8} align="end" collisionPadding={12}>
          <UpdateDetails onDismiss={update.latest ? () => {
            try {
              localStorage.setItem(DISMISSED, update.latest ?? "");
            } catch {
              // the notice comes back next time; nothing else depends on it
            }
            setHidden(update.latest);
          } : undefined} />
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

/** What is known, how to update, and the user's choice about daily checks. */
export function UpdateDetails({ onDismiss }: { onDismiss?: () => void }) {
  const update = useStore((st) => st.update);
  const check = useStore((st) => st.checkUpdates);
  const consent = useStore((st) => st.setUpdateConsent);
  const [busy, setBusy] = useState(false);
  if (!update) return null;
  return (
    <div className={s.details}>
      {update.available ? (
        <>
          <p className={s.title}>Logogram {update.latest} is out</p>
          <p className={s.text}>
            You have {update.current}. Updating keeps your projects, results and settings as they are.
          </p>
          {update.notes_url && (
            <a className={s.link} href={update.notes_url} target="_blank" rel="noopener noreferrer">
              What's new in {update.latest} ↗
            </a>
          )}
          <span className={s.label}>Update with</span>
          <CopyCommand command={update.command} />
        </>
      ) : update.latest ? (
        <>
          <p className={s.title}>You have the newest Logogram</p>
          <p className={s.text}>Version {update.current} is the latest release.</p>
        </>
      ) : (
        <>
          <p className={s.title}>Is there a newer Logogram?</p>
          <p className={s.text}>
            You have {update.current}
            {update.released ? `, released ${update.released}` : ""}. Check now, or let Logogram look once a day.
          </p>
        </>
      )}
      <div className={s.row}>
        <Button
          size="small"
          onClick={async () => {
            setBusy(true);
            await check();
            setBusy(false);
          }}
          disabled={busy}
        >
          {busy ? <Spinner label="Checking" /> : null}
          Check now
        </Button>
        {update.checked_at && <span className={s.meta}>Checked {ago(update.checked_at)}</span>}
        {onDismiss && update.available && (
          <Button size="small" variant="ghost" onClick={onDismiss}>
            Not now
          </Button>
        )}
      </div>
      {update.error && <p className={s.error}>{update.error}</p>}
      <div className={s.consent}>
        <Checkbox checked={update.automatic === true} onChange={(v) => void consent(v)}>
          Check for new versions once a day
        </Checkbox>
        <p className={s.meta}>Asks pypi.org for the newest version number. Nothing about you, this computer or your work is sent.</p>
      </div>
    </div>
  );
}
