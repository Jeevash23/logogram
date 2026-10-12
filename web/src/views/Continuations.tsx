import { useEffect, useId, useState } from "react";

import type { ContinuationsFile, GeneratedTokens } from "../api/types";
import { Field, Select } from "../components/ui";
import { plural, visibleToken } from "../lib/format";
import s from "./views.module.css";
import c from "./Continuations.module.css";

/** Tokens as the token strip shows them: one cell each, spaces as ·, line breaks as ⏎. A reader
 * hears the text once, as written. */
export function TokenLine({ tokens }: { tokens: string[] }) {
  return (
    <>
      <span className="visually-hidden">{tokens.join("")}</span>
      <span className={c.tokens} aria-hidden="true">
        {tokens.map((t, i) => (
          <span key={i} className={c.token}>
            {visibleToken(t) || "∅"}
          </span>
        ))}
      </span>
    </>
  );
}

/** What a steered model says: each continued prompt as written, then what the model says without
 * steering and at each strength, the random control's lines marked. With several steered sites,
 * one site's lines at a time: the selected one's, or as chosen here. */
export function Continuations({ data, selectedRow }: { data: ContinuationsFile; selectedRow: number | null }) {
  const first = data.rows[0]?.row ?? 0;
  const [chosen, setChosen] = useState(selectedRow ?? first);
  // Another run's continuations start again at its selected site, or at its first.
  const [shownData, setShownData] = useState(data);
  if (shownData !== data) {
    setShownData(data);
    setChosen(selectedRow ?? first);
  }
  useEffect(() => {
    if (selectedRow !== null) setChosen(selectedRow);
  }, [selectedRow]);
  // A site this run didn't continue (or another run's) shows its first steered site instead.
  const row = data.rows.some((r) => r.row === chosen) ? chosen : first;
  const titleId = useId();
  const several = data.rows.length > 1;
  return (
    <section className={c.section} aria-labelledby={titleId}>
      <div className={c.head}>
        <h2 id={titleId} className={s.panelTitle}>
          What the steered model says
        </h2>
        {several && (
          <Field label="Steered site" className={c.site}>
            <Select value={String(row)} onChange={(ev) => setChosen(Number(ev.target.value))}>
              {data.rows.map((r) => (
                <option key={r.row} value={r.row}>
                  {r.label}
                </option>
              ))}
            </Select>
          </Field>
        )}
      </div>
      <p className={s.small}>
        {plural(data.tokens, "new token")} after each of the first held-out {data.receiver} prompts, the most likely token each
        time (ties to the lowest token id). Every forward pass adds the vector at the steered token of the prompt, as when it
        was measured; the tokens the model generates aren't steered themselves. · is a space, ⏎ a line break.
      </p>
      {data.prompts.map((prompt) => (
        <div key={prompt.index} className={c.prompt}>
          <p className={c.text}>
            <span className={c.index}>Prompt {prompt.index}</span> {prompt.text}
          </p>
          <SaidLines
            lines={[
              { label: "Without steering", said: prompt.unsteered },
              ...prompt.steered
                .filter((x) => x.row === row)
                .map((x) => ({ label: x.key, said: x, control: x.control })),
            ]}
          />
        </div>
      ))}
    </section>
  );
}

/** Continuations one above the other, each after its label. */
export function SaidLines({ lines }: { lines: { label: string; said: GeneratedTokens; control?: boolean }[] }) {
  return (
    <table className={c.lines}>
      <tbody>
        {lines.map((line) => (
          <Line key={line.label} {...line} />
        ))}
      </tbody>
    </table>
  );
}

function Line({ label, said, control = false }: { label: string; said: GeneratedTokens; control?: boolean }) {
  return (
    <tr className={control ? c.control : undefined}>
      <th scope="row" className={c.label}>
        {label}
      </th>
      <td>
        <TokenLine tokens={said.tokens} />
      </td>
    </tr>
  );
}
