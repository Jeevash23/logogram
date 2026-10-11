import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "./styles/global.css";
import { App } from "./App";
import { ErrorBoundary } from "./components/ErrorBoundary";

createRoot(document.getElementById("root") as HTMLElement).render(
  <StrictMode>
    <ErrorBoundary name="Logogram" page>
      <App />
    </ErrorBoundary>
  </StrictMode>,
);
