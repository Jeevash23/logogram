// Contain a render error to the part of the page it happened in, say what went wrong, and offer
// a way back. The store is untouched, so the experiment form and results survive a reload here.

import { Component, Fragment, createRef, type ErrorInfo, type ReactNode } from "react";

import { useStore } from "../store/app";
import { copyText } from "./CopyCommand";
import { Button } from "./ui";
import s from "./ErrorBoundary.module.css";

interface Props {
  /** What stopped, as the start of a sentence: "This view", "The inspector", "Logogram". */
  name: string;
  children: ReactNode;
  /** Fill the page: the boundary around the whole app. */
  page?: boolean;
  /** While showing an error, a change of this value (a new selection, say) tries again. */
  resetKey?: unknown;
}

interface State {
  error: Error | null;
  componentStack: string;
  attempt: number;
  copied: boolean;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, componentStack: "", attempt: 0, copied: false };
  private reloadButton = createRef<HTMLButtonElement>();

  static getDerivedStateFromError(error: unknown): Partial<State> {
    return { error: error instanceof Error ? error : new Error(String(error)), copied: false };
  }

  componentDidCatch(_error: unknown, info: ErrorInfo): void {
    this.setState({ componentStack: info.componentStack ?? "" });
  }

  componentDidUpdate(prev: Props, prevState: State): void {
    if (this.state.error && !prevState.error) {
      // Focus was inside what just disappeared: move it to the way back.
      const active = document.activeElement;
      if (!active || active === document.body) this.reloadButton.current?.focus();
    }
    if (this.state.error && !Object.is(prev.resetKey, this.props.resetKey)) this.reload();
  }

  /** Render the contents again from scratch. */
  reload = (): void => {
    this.setState((st) => ({ error: null, componentStack: "", copied: false, attempt: st.attempt + 1 }));
  };

  copy = async (): Promise<void> => {
    const { error, componentStack } = this.state;
    if (!error) return;
    const st = useStore.getState();
    const details = [
      `Logogram ${st.version || "(version unknown)"}`,
      `Where: ${this.props.name}${st.screen === "workbench" ? ` (view: ${st.view})` : ` (screen: ${st.screen})`}`,
      `${error.name}: ${error.message}`,
      error.stack ? `\nStack:\n${error.stack}` : "",
      componentStack ? `\nComponents:${componentStack}` : "",
    ]
      .filter(Boolean)
      .join("\n");
    if (await copyText(details)) this.setState({ copied: true });
  };

  render(): ReactNode {
    const { error, attempt, copied } = this.state;
    if (!error) return <Fragment key={attempt}>{this.props.children}</Fragment>;
    const { name, page } = this.props;
    return (
      <div className={page ? s.page : s.inline}>
        <div className={s.box} role="alert">
          <h2 className={s.title}>{name} stopped with an error</h2>
          <p className={s.message}>{error.message || `${error.name}, with no message.`}</p>
          <p className={s.help}>
            Reload this view to try again; your experiment form and results are kept. If the error comes back, copy its
            details and include them when you report it.
          </p>
          <div className={s.actions}>
            <Button ref={this.reloadButton} variant="primary" icon="refresh" onClick={this.reload}>
              Reload this view
            </Button>
            <Button icon={copied ? "check" : "copy"} onClick={() => void this.copy()}>
              {copied ? "Copied error details" : "Copy error details"}
            </Button>
          </div>
        </div>
      </div>
    );
  }
}
