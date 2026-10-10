import { Fragment, useId, type ReactNode } from "react";

import { useStore } from "../store/app";
import { MOD, SHIFT } from "./Header";
import { Dialog, Kbd } from "./ui";
import s from "./ShortcutsDialog.module.css";

/** Keys pressed together, such as Ctrl K. */
function Keys({ keys }: { keys: string[] }) {
  return (
    <span className={s.keys}>
      {keys.map((k, i) => (
        <Kbd key={i}>{k}</Kbd>
      ))}
    </span>
  );
}

/** Alternatives: either of these does the same. */
function Either({ children }: { children: ReactNode[] }) {
  return (
    <>
      {children.map((child, i) => (
        <Fragment key={i}>
          {i > 0 && <span className={s.or}>or</span>}
          {child}
        </Fragment>
      ))}
    </>
  );
}

const SECTIONS: { title: string; rows: [ReactNode, string][] }[] = [
  {
    title: "Anywhere",
    rows: [
      [<Keys keys={[MOD, "K"]} />, "Open the command palette. Type a head, such as L9H6, to select it."],
      [<Keys keys={["?"]} />, "Show these shortcuts."],
      [<Keys keys={["Esc"]} />, "Close a dialog, menu or the palette."],
    ],
  },
  {
    title: "In the workbench, outside text fields",
    rows: [
      [
        <span className={s.keys}>
          <Kbd>1</Kbd>
          <span className={s.or}>to</span>
          <Kbd>7</Kbd>
        </span>,
        "Go to Prompts, Baseline, Configure, Results, Attention, Compare runs or Spec.",
      ],
      [<Either>{[<Keys key="a" keys={["["]} />, <Keys key="b" keys={["]"]} />]}</Either>, "Show the previous or next prompt."],
    ],
  },
  {
    title: "The selected component",
    rows: [
      [<Keys keys={["←", "→", "↑", "↓"]} />, "Move across a layer's components, or to the layer above or below."],
      [<Either>{[<Keys key="a" keys={["Home"]} />, <Keys key="b" keys={["End"]} />]}</Either>, "Go to the layer's residual stream or its MLP."],
      [<Keys keys={["Esc"]} />, "Clear the selection."],
      [<Keys keys={["P"]} />, "Patch here: set up activation patching at the component."],
      [<Keys keys={["B"]} />, "Ablate here: set up an ablation at the component."],
      [<Keys keys={["A"]} />, "Open the selected head's attention."],
      [<Keys keys={["C"]} />, "Compare the component across runs."],
      [<Keys keys={[SHIFT, "F10"]} />, "On the model map, list these actions in a menu."],
    ],
  },
  {
    title: "Heatmaps",
    rows: [
      [<Keys keys={["←", "→", "↑", "↓"]} />, "Move between cells, skipping empty ones."],
      [<Either>{[<Keys key="a" keys={["Home"]} />, <Keys key="b" keys={["End"]} />]}</Either>, "Go to the first or last cell of the row."],
    ],
  },
  {
    title: "Experiment form (Configure)",
    rows: [
      [<Keys keys={[MOD, "Enter"]} />, "Run the experiment."],
      [<Keys keys={[MOD, "Z"]} />, "Undo the last change to the form. In a text field, this undoes typing first."],
      [<Either>{[<Keys key="a" keys={[MOD, SHIFT, "Z"]} />, <Keys key="b" keys={["Ctrl", "Y"]} />]}</Either>, "Redo the change."],
    ],
  },
  {
    title: "Choices",
    rows: [
      [<Keys keys={["←", "→", "↑", "↓"]} />, "Choose the next or previous option in a group of choices."],
      [<Either>{[<Keys key="a" keys={["Home"]} />, <Keys key="b" keys={["End"]} />]}</Either>, "Choose the first or last option."],
    ],
  },
];

/** Every keyboard shortcut, from ? or the command palette. */
export function ShortcutsDialog() {
  const open = useStore((st) => st.shortcutsOpen);
  const id = useId();
  return (
    <Dialog
      open={open}
      onOpenChange={(value) => useStore.setState({ shortcutsOpen: value })}
      title="Keyboard shortcuts"
      description="Single keys work when the focus isn't in a text field."
      wide
    >
      <div className={s.sections}>
        {SECTIONS.map((section, i) => (
          <section key={section.title} className={s.section} aria-labelledby={`${id}-${i}`}>
            <h3 className={s.title} id={`${id}-${i}`}>
              {section.title}
            </h3>
            <dl className={s.list}>
              {section.rows.map(([keys, what], j) => (
                <div key={j} className={s.row}>
                  <dt>{keys}</dt>
                  <dd>{what}</dd>
                </div>
              ))}
            </dl>
          </section>
        ))}
      </div>
    </Dialog>
  );
}
