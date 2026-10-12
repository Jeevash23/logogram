import { useId, useRef, type CSSProperties, type KeyboardEvent, type ReactNode } from "react";

import { GUIDE_PAGES, type GuidePageId } from "../lib/guide";
import { guideUrl } from "../lib/help";
import { useStore, type View } from "../store/app";
import { useHelp, walkThroughExample } from "../store/help";
import { MOD } from "./Header";
import { Button, Dialog, Kbd } from "./ui";
import s from "./GuideDialog.module.css";

/**
 * Help → Guide: short pages on what each workspace is for, how to read the map and the logogram,
 * what the numbers and intervals mean, and the keyboard, each with buttons that open the views it
 * talks about and links to the documentation on GitHub. The pages are tabs: the arrow keys move
 * between them.
 */
export function GuideDialog() {
  const open = useStore((st) => st.helpDialog === "guide");
  const page = useHelp((st) => st.guidePage);
  const inWorkbench = useStore((st) => st.screen === "workbench");
  const tabs = useRef<(HTMLButtonElement | null)[]>([]);
  const id = useId();
  const index = Math.max(0, GUIDE_PAGES.findIndex((p) => p.id === page));
  const current = GUIDE_PAGES[index];

  const choose = (i: number, focus: boolean) => {
    useHelp.setState({ guidePage: GUIDE_PAGES[i].id });
    if (focus) tabs.current[i]?.focus();
  };
  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>, i: number) => {
    const n = GUIDE_PAGES.length;
    const next =
      e.key === "ArrowRight" ? (i + 1) % n : e.key === "ArrowLeft" ? (i - 1 + n) % n : e.key === "Home" ? 0 : e.key === "End" ? n - 1 : null;
    if (next === null || e.altKey || e.ctrlKey || e.metaKey) return;
    e.preventDefault();
    choose(next, true);
  };
  // A button that opens a view closes the guide, and the keyboard goes to that view.
  const go = (view: View) => {
    useHelp.setState({ toView: true });
    useStore.setState({ helpDialog: null });
    useStore.getState().setView(view);
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(value) => {
        if (!value) useStore.setState({ helpDialog: null });
      }}
      title="Guide"
      description="How Logogram works, in short. The documentation on GitHub says more."
      wide
      onCloseAutoFocus={(e) => {
        if (!useHelp.getState().toView) return;
        e.preventDefault();
        useHelp.setState({ toView: false });
        document.getElementById("main")?.focus();
      }}
    >
      <div className={s.tabs} role="tablist" aria-label="Pages of the guide">
        {GUIDE_PAGES.map((p, i) => (
          <button
            key={p.id}
            ref={(el) => {
              tabs.current[i] = el;
            }}
            type="button"
            role="tab"
            id={`${id}-${p.id}`}
            className={s.tab}
            aria-selected={p.id === current.id}
            aria-controls={`${id}-panel`}
            tabIndex={p.id === current.id ? 0 : -1}
            onClick={() => choose(i, false)}
            onKeyDown={(e) => onKeyDown(e, i)}
          >
            {p.title}
          </button>
        ))}
      </div>
      <div className={s.panel} role="tabpanel" id={`${id}-panel`} aria-labelledby={`${id}-${current.id}`} tabIndex={0}>
        <PageText page={current.id} />
        {(inWorkbench && current.views.length > 0) || current.id === "keys" ? (
          <div className={s.actions}>
            {inWorkbench &&
              current.views.map((v) => (
                <Button key={v.label} size="small" onClick={() => go(v.view)}>
                  {v.label}
                </Button>
              ))}
            {current.id === "keys" && (
              <Button size="small" onClick={() => useStore.setState({ helpDialog: null, shortcutsOpen: true })}>
                Show every shortcut
              </Button>
            )}
          </div>
        ) : null}
        <div className={s.docs}>
          <h3 className={s.docsTitle} id={`${id}-docs`}>
            In the documentation
          </h3>
          <ul aria-labelledby={`${id}-docs`}>
            {current.docs.map((d) => (
              <li key={d.title}>
                <a className={s.link} href={guideUrl(d)} target="_blank" rel="noopener noreferrer">
                  {d.title}
                  <span className="visually-hidden"> (opens in a new tab)</span>
                  <span aria-hidden="true"> ↗</span>
                </a>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </Dialog>
  );
}

/** A workspace's word, underlined in its color: the color marks it, the word stays ink. */
function Workspace({ color, children }: { color: string; children: ReactNode }) {
  return (
    <span className={s.workspace} style={{ "--word": `var(${color})` } as CSSProperties}>
      {children}
    </span>
  );
}

/** Keys pressed together, or, with a joiner, either of two. */
function Keys({ keys, joiner }: { keys: string[]; joiner?: string }) {
  return (
    <span className={s.keys}>
      {keys.map((k, i) => (
        <span key={k} className={s.key}>
          {i > 0 && joiner && <span className={s.joiner}>{joiner}</span>}
          <Kbd>{k}</Kbd>
        </span>
      ))}
    </span>
  );
}

function Row({ term, children }: { term: ReactNode; children: ReactNode }) {
  return (
    <div className={s.row}>
      <dt>{term}</dt>
      <dd>{children}</dd>
    </div>
  );
}

function PageText({ page }: { page: GuidePageId }) {
  const singleKeys = useStore((st) => st.singleKeys);
  if (page === "workspaces") {
    return (
      <>
        <p>Three workspaces run across the top. Each has a color, which marks its tabs and its choices.</p>
        <dl className={s.rows}>
          <Row term={<Workspace color="--explore">Explore</Workspace>}>
            Look inside the model: the model map, where heads attend and which heads the head finder finds, what each
            layer would predict, an SAE&rsquo;s features, and any text of your own.
          </Row>
          <Row term={<Workspace color="--experiment">Experiment</Workspace>}>
            Set up what runs: the prompt pairs, the baseline check, the experiment form, and the complete spec it makes.
          </Row>
          <Row term={<Workspace color="--evidence">Evidence</Workspace>}>
            Read what happened: each run&rsquo;s results with their statistics, comparisons between runs, sweeps across
            seeds or checkpoints, and your research notes.
          </Row>
        </dl>
        <p>
          <strong>History</strong> lists every run in the project, and <strong>Commands</strong> runs any action by name.
          The walkthrough shows the IOI example step by step, at the controls themselves.
        </p>
        <div>
          <Button size="small" onClick={() => void walkThroughExample()}>
            Walk me through the example
          </Button>
        </div>
      </>
    );
  }
  if (page === "map") {
    return (
      <>
        <p>
          The map shows a run&rsquo;s results on the model: one row per layer, from layer 0 at the top, and a cell for each
          component, such as each attention head, the attention output, the MLP and the residual stream. The results
          view draws the sites a run measured as a grid of their own.
        </p>
        <p>
          Queries, keys, values, neurons and the edges between components have no cell on the map. The results view
          shows queries, keys and values as a grid of layers by heads, neurons layer by layer with the strongest
          listed, and edges drawn between the components they join.
        </p>
        <p>
          Color is the effect, on a scale symmetric around zero and always shown beside it: cobalt where the
          intervention moves the answer toward the other prompt, amber where it moves it away, and no color where it
          changes nothing. <strong>Show values</strong> prints each cell&rsquo;s mean, and <strong>±1</strong> fixes the
          scale, so that runs compare by eye.
        </p>
        <p>
          Choose a cell, with the mouse or the arrow keys, to circle it in ink and open it in the inspector, with the
          evidence behind its number.
        </p>
        <p>
          The <strong>logogram</strong> is a run&rsquo;s glyph, written by its results: layers run clockwise from the top,
          and the ink swells outward where a layer&rsquo;s strongest effect is positive and inward where it is negative. A
          few strong layers make a heavy, lopsided ring; a null result a thin, even one.
        </p>
      </>
    );
  }
  if (page === "numbers") {
    return (
      <>
        <dl className={s.rows}>
          <Row term="Effect">
            The change in the metric, divided by the gap between the two prompts of each pair: 1 is as far as switching
            to the other prompt, 0 is no change, and a negative effect goes the other way.
          </Row>
          <Row term="Interval">
            A percentile bootstrap interval over the prompts, at the run&rsquo;s confidence level: where the mean would fall
            with other prompts like these.
          </Row>
          <Row term="Band">
            A simultaneous band, which holds for every site of the sweep at once. A site whose band excludes zero survives
            the number of sites tested.
          </Row>
          <Row term="q-value">The false discovery rate at which the site counts as a finding.</Row>
          <Row term="Flips">
            The prompts whose preference for the answer changed sign. Opposite sign counts those whose effect goes against
            the mean.
          </Row>
          <Row term="n">The prompts measured. Each prompt&rsquo;s own value is kept, and every number comes from them.</Row>
        </dl>
        <p>
          An interval describes the prompts. It says nothing about another seed, another sample of prompts or another
          checkpoint: a sweep measures those, and <strong>Check robustness</strong> reruns with one choice changed.
        </p>
      </>
    );
  }
  return (
    <>
      <dl className={s.rows}>
        <Row term={<Keys keys={[MOD, "K"]} />}>Run any command by name, or jump to a head, such as L9H9.</Row>
        <Row term={<Keys keys={["?"]} />}>Show every keyboard shortcut.</Row>
        <Row term={<Keys keys={["1", "6"]} joiner="to" />}>Go to a tab of the open workspace.</Row>
        <Row term={<Keys keys={["[", "]"]} joiner="or" />}>Show the previous or next prompt.</Row>
        <Row term={<Keys keys={["←", "→", "↑", "↓"]} />}>Move across the model map and the heatmaps.</Row>
        <Row term={<Keys keys={["P", "B", "A", "C"]} />}>
          On a selected component: patch here, ablate here, show its attention, compare it across runs.
        </Row>
        <Row term={<Keys keys={[MOD, "Enter"]} />}>Run the experiment form, or queue it while another job runs.</Row>
        <Row term={<Keys keys={["Esc"]} />}>Close a dialog or a menu, or clear the selection.</Row>
      </dl>
      <p>
        {singleKeys
          ? "The single keys work when the focus isn't in a text field. Turn them off in the list of shortcuts, for speech input or keys pressed by accident."
          : "Single-key shortcuts are off. Turn them on in the list of shortcuts."}
      </p>
    </>
  );
}
