import { useEffect } from "react";

import { connectEvents } from "./api/events";
import { CommandPalette } from "./components/CommandPalette";
import { Notices } from "./components/Notices";
import { Mark, TooltipProvider } from "./components/ui";
import { useGlobalKeys } from "./lib/keys";
import { Projects } from "./screens/Projects";
import { SystemCheck } from "./screens/SystemCheck";
import { Workbench } from "./screens/Workbench";
import { useStore } from "./store/app";
import s from "./App.module.css";

export function App() {
  const screen = useStore((st) => st.screen);
  const bootError = useStore((st) => st.bootError);

  useEffect(() => {
    let disconnect: (() => void) | null = null;
    let unmounted = false;
    // Connect once booted: entering the project resets run state, which would otherwise wipe the
    // current run that the server replays to a new connection.
    void useStore
      .getState()
      .boot()
      .finally(() => {
        if (!unmounted) disconnect = connectEvents();
      });
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => useStore.getState().resolveTheme();
    media.addEventListener("change", onChange);
    return () => {
      unmounted = true;
      disconnect?.();
      media.removeEventListener("change", onChange);
    };
  }, []);

  useGlobalKeys();

  return (
    <TooltipProvider delayDuration={350} skipDelayDuration={200}>
      {screen === "loading" && (
        <div className={s.center}>
          <Mark size={28} />
        </div>
      )}
      {screen === "error" && (
        <div className={s.center}>
          <div className={s.error}>
            <Mark size={28} />
            <h2>Logogram can't start</h2>
            <p className="muted">{bootError}</p>
          </div>
        </div>
      )}
      {screen === "system" && <SystemCheck />}
      {screen === "projects" && <Projects />}
      {screen === "workbench" && <Workbench />}
      <CommandPalette />
      <Notices />
    </TooltipProvider>
  );
}
