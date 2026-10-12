import { useEffect, useRef, useState } from "react";

import { api, type AnalysisOptions } from "../api/client";
import type { FeatureWrites as Writes, PositionSpec } from "../api/types";
import { num, signed, visibleToken } from "../lib/format";
import { Spinner } from "./ui";
import s from "./FeatureWrites.module.css";

/** How many tokens each list shows. */
const TOP = 8;

/**
 * The tokens a feature promotes and suppresses directly: one unit of its decoder direction read
 * straight into the logits at one token of one prompt, with exactly what was computed. Magnitudes
 * are drawn in ink; the sign is the list a token is in.
 */
export function FeatureWrites({
  dataset,
  index,
  which,
  feature,
  position,
  options,
  optionsKey,
  sae,
  standardized,
  compact = false,
}: {
  dataset: string;
  index: number;
  which: "clean" | "corrupt";
  feature: number;
  /** The token to read it at, as a spec names positions; null for where it fires most. */
  position: PositionSpec | null;
  options: Partial<AnalysisOptions>;
  /** Changes when the options do, so the request is made again. */
  optionsKey: string;
  /** Changes when another SAE is loaded. */
  sae: string;
  /** Whether the SAE standardizes its inputs (a unit then scales with each input's spread). */
  standardized: boolean;
  /** One list under the other, for a narrow panel. */
  compact?: boolean;
}) {
  const [data, setData] = useState<Writes | null>(null);
  const [error, setError] = useState<string | null>(null);
  const seq = useRef(0);
  const positionKey = JSON.stringify(position);

  useEffect(() => {
    const mine = ++seq.current;
    setData(null);
    setError(null);
    api.saeWrites({ dataset, index, which, feature, position, top: TOP, ...options }).then(
      (d) => mine === seq.current && setData(d),
      (e: Error) => mine === seq.current && setError(e.message),
    );
    // The position and options are compared by their keys.
  }, [dataset, index, which, feature, positionKey, optionsKey, sae]);

  if (error) return <p className={s.note}>{error}</p>;
  if (!data) {
    return (
      <p className={s.note}>
        <Spinner /> Reading its direction into the logits
      </p>
    );
  }
  const bound = Math.max(1e-12, ...[...data.promoted, ...data.suppressed].map((r) => Math.abs(r.logit)));
  return (
    <div className={s.writes}>
      <div className={compact ? s.stacked : s.lists}>
        <List title="Promotes" rows={data.promoted} bound={bound} />
        <List title="Suppresses" rows={data.suppressed} bound={bound} />
      </div>
      <p className={s.fine}>{explanation(data, standardized)}</p>
    </div>
  );
}

function List({ title, rows, bound }: { title: string; rows: Writes["promoted"]; bound: number }) {
  return (
    <div className={s.list}>
      <div className={s.listTitle}>{title}</div>
      <ol className={s.rows} aria-label={`${title}, logit change per unit of the feature`}>
        {rows.map((r) => (
          <li key={r.id} className={s.row}>
            <span className={s.token} title={`Token id ${r.id}`}>
              {visibleToken(r.token)}
            </span>
            <span className={s.bar} aria-hidden="true">
              <span style={{ width: `${(Math.abs(r.logit) / bound) * 100}%` }} />
            </span>
            <span className={s.value}>{signed(r.logit, 3)}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

/** Exactly what was computed, in words. */
function explanation(d: Writes, standardized: boolean): string {
  const token = `token ${d.position} (‘${visibleToken(d.token)}’)`;
  const where =
    d.chosen === "fires_most"
      ? `${token}, the feature's strongest in this prompt (${num(d.activation, 2)})`
      : d.chosen === "last"
        ? `${token}, the last one, since it fires nowhere in this prompt`
        : `${token}${d.activation > 0 ? `, where it fires at ${num(d.activation, 2)}` : ", where it doesn't fire"}`;
  const unit = standardized ? "its decoder direction times the input's spread at that token" : "its decoder direction";
  const path = d.site === "head" ? `${unit}, through each head's slice of the attention's output projection,` : unit;
  return (
    `One unit of the feature: ${path} added to the residual stream entering the final normalization at ${where}, ` +
    "skipping every later layer, and read through that normalization, with its scale held at its value there, " +
    `and the unembedding${d.soft_cap ? `, before the model's soft cap on its logits (${num(d.soft_cap, 0)})` : ""}. ` +
    "Each value is a token's logit change per unit of the feature, relative to the mean change over the vocabulary. " +
    "Tokens rank the same at every token; only the sizes change."
  );
}
