import { SESSION_ENDED_TEXT } from "../api/client";
import type { Job } from "../api/types";

export interface Milestone {
  /** Equal keys mean there is nothing new to say. */
  key: string;
  text: string;
}

/**
 * What is said aloud about the work in progress: that it started, each quarter of a run, and how
 * it ended. Progress ticks arrive several times a second; a screen reader hears only these.
 */
export function milestone(
  job: Job | null,
  progress: { done: number; total: number } | null | undefined,
  connection: "connecting" | "connected" | "reconnecting" | "ended",
): Milestone {
  if (connection === "ended") {
    return { key: "ended", text: SESSION_ENDED_TEXT };
  }
  if (connection === "reconnecting") {
    return { key: "reconnecting", text: "Disconnected from the server. Reconnecting; the progress shown may be out of date." };
  }
  if (!job) return { key: "idle", text: "" };
  if (job.status === "running") {
    if (job.cancelling) return { key: `${job.id}:cancelling`, text: `Cancelling “${job.title}”.` };
    const share = job.kind === "run" && progress && progress.total > 0 ? progress.done / progress.total : 0;
    const quarter = Math.max(0, Math.min(3, Math.floor(share * 4)));
    return quarter === 0
      ? { key: `${job.id}:started`, text: `Started “${job.title}”.` }
      : { key: `${job.id}:${quarter}`, text: `“${job.title}” is ${quarter * 25}% done.` };
  }
  const ended = { finished: "finished", failed: "failed", cancelled: "was cancelled" }[job.status];
  return { key: `${job.id}:${job.status}`, text: `“${job.title}” ${ended}.` };
}
