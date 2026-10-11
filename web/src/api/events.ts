// The event stream: job progress, streamed results and model loading, pushed by the server.
// On connecting, the server first sends the current run so far, so a reloaded or reconnected tab
// repaints it; a tab that falls far behind is disconnected and catches up the same way.

import { api, SESSION_ENDED_TEXT } from "./client";
import { useStore } from "../store/app";

/** The close code of a stream whose session token the server no longer accepts. */
export const SESSION_ENDED = 4401;

export function connectEvents(): () => void {
  let stopped = false;
  let retry = 0;
  let socket: WebSocket | null = null;
  let timer: number | undefined;

  const open = () => {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${location.host}/ws`);
    socket = ws;
    let synced = false;
    const pending: (Record<string, unknown> & { type: string })[] = [];
    ws.onopen = () => {
      // Catch up on anything that changed before the stream opened: the job, the model and run
      // statuses. (The current run itself is replayed by the server.)
      const store = useStore.getState();
      void api.state().then(async (s) => {
        if (socket !== ws || stopped) return;
        if (s.project?.session_id !== store.project?.session_id) {
          if (s.project) await store.enterProject(s.project);
          else store.leaveProject();
        }
        store.applyServerState(s);
        await store.refreshRuns();
        if (socket !== ws || stopped) return;
        synced = true;
        useStore.setState({ connection: "connected", connectionError: null });
        for (const event of pending) useStore.getState().handleEvent(event);
        pending.length = 0;
        retry = 0;
      }).catch((error: Error) => {
        useStore.setState({ connectionError: error.message });
        ws.close();
      });
    };
    ws.onmessage = (msg) => {
      try {
        const event = JSON.parse(String(msg.data));
        if (event && typeof event.type === "string") {
          if (synced) useStore.getState().handleEvent(event);
          else pending.push(event);
        }
      } catch {
        // ignore malformed messages
      }
    };
    ws.onclose = (event) => {
      if (stopped || socket !== ws) return;
      if (event.code === SESSION_ENDED) {
        // The server no longer knows this tab's session (it restarted): trying again can't help.
        stopped = true;
        useStore.setState({ connection: "ended", connectionError: SESSION_ENDED_TEXT });
        useStore.getState().notify(SESSION_ENDED_TEXT, "error", { key: "session-ended" });
        return;
      }
      useStore.setState({ connection: "reconnecting" });
      retry += 1;
      const delay = Math.min(5000, 250 * 2 ** Math.min(retry, 5));
      timer = window.setTimeout(open, delay);
    };
  };

  open();
  return () => {
    stopped = true;
    window.clearTimeout(timer);
    socket?.close();
  };
}
