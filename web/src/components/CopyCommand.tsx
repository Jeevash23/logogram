import { useState } from "react";

import { Icon } from "./ui";
import s from "./CopyCommand.module.css";

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Clipboard API can be unavailable; fall back to a hidden textarea.
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand("copy");
    area.remove();
    return ok;
  }
}

export function CopyCommand({ command, label = "Copy" }: { command: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className={s.command}>
      {/* Focusable, so a command wider than its box can be scrolled from the keyboard. */}
      <code tabIndex={0}>{command}</code>
      <button
        type="button"
        className={s.copy}
        onClick={async () => {
          if (await copyText(command)) {
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1400);
          }
        }}
        aria-label={copied ? "Copied" : label}
      >
        <Icon name={copied ? "check" : "copy"} size={14} />
        <span>{copied ? "Copied" : label}</span>
      </button>
    </div>
  );
}
