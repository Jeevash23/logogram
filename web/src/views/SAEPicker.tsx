import { type RefObject, useEffect, useRef, useState } from "react";

import { api } from "../api/client";
import type { CatalogueSAE, SAECatalogue } from "../api/types";
import { Button, Callout, Field, Input, Spinner } from "../components/ui";
import { count, pct, shortRevision } from "../lib/format";
import { KIND_NAMES } from "../lib/sites";
import s from "./views.module.css";
import p from "./SAEPicker.module.css";

/** Where a catalogued SAE reads (and a transcoder writes), in words. */
export function catalogueWhere(x: CatalogueSAE): string {
  const layer = x.layer ?? "?";
  if (x.kind !== "sae") return `Layer ${layer}'s MLP input → output${x.kind === "skip_transcoder" ? ", with a skip connection" : ""}`;
  switch (x.site) {
    case "resid_pre":
      return `Residual stream entering layer ${layer}`;
    case "resid_mid":
      return `Residual stream between layer ${layer}'s attention and MLP`;
    case "resid_post":
      return `Residual stream leaving layer ${layer}`;
    case "attn_out":
      return `Layer ${layer}'s attention output`;
    case "mlp_out":
      return `Layer ${layer}'s MLP output`;
    case "head":
      return `Layer ${layer}'s attention heads' outputs (z)`;
    case null:
      return x.hook ?? "A site Logogram can't name";
    default:
      return `${KIND_NAMES[x.site as keyof typeof KIND_NAMES] ?? x.site} of layer ${layer}`;
  }
}

const STATUS: Record<CatalogueSAE["reads"], string> = {
  exact: "Reads exactly",
  refused: "Not read",
  unknown: "Checked on loading",
};

function optionId(x: CatalogueSAE): string {
  return `sae-option-${x.release}-${x.id}`.replace(/[^\w-]/g, "_");
}

/**
 * The published SAEs for the loaded model, from the catalogue of SAELens's registry that ships
 * with Logogram: searchable by release, hook, layer and width, each with whether Logogram reads it
 * exactly or why not. The search field keeps focus; the arrow keys move through the list (the
 * listbox pattern with aria-activedescendant) and Enter goes to the chosen SAE's Load button.
 */
export function SAEPicker({ modelId, busy, onLoad }: { modelId: string; busy: boolean; onLoad: (sae: CatalogueSAE) => void }) {
  const [query, setQuery] = useState("");
  const [data, setData] = useState<SAECatalogue | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState(0);
  const seq = useRef(0);
  const listRef = useRef<HTMLUListElement>(null);
  const loadRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const mine = ++seq.current;
    const timer = window.setTimeout(() => {
      api.saeCatalogue(modelId, query).then(
        (d) => {
          if (mine !== seq.current) return;
          setData(d);
          setError(null);
          setActive(0);
        },
        (e: Error) => mine === seq.current && setError(e.message),
      );
    }, query ? 120 : 0);
    return () => window.clearTimeout(timer);
  }, [modelId, query]);

  const saes = data?.saes ?? [];
  const chosen = saes[Math.min(active, saes.length - 1)] ?? null;

  useEffect(() => {
    if (chosen) listRef.current?.querySelector(`#${CSS.escape(optionId(chosen))}`)?.scrollIntoView({ block: "nearest" });
  }, [chosen]);

  if (error) return <Callout tone="error" title="The catalogue of published SAEs didn't open">{error}</Callout>;
  if (!data) {
    return (
      <p className={s.small}>
        <Spinner /> Reading the catalogue of published SAEs
      </p>
    );
  }
  if (data.total === 0) {
    return (
      <p className={s.small}>
        {data.source}'s registry has no SAEs for {modelId}. It names models by their Hugging Face id; name a repository and folder below instead.
      </p>
    );
  }

  const move = (by: number) => setActive((a) => Math.max(0, Math.min(saes.length - 1, a + by)));
  return (
    <section className={p.picker} aria-labelledby="published-saes-title">
      <div>
        <h2 id="published-saes-title" className={s.panelTitle}>
          Published SAEs for {modelId}
        </h2>
        <p className={s.small}>
          {count(data.total)} in {data.source}'s registry, as the catalogue that ships with Logogram records them. Logogram reads {count(data.exact)}{" "}
          of them exactly. Nothing is downloaded until you load one.
        </p>
      </div>
      <Field label="Search" help="A release, a hook, a site, layer 8, or a width such as 16k.">
        <Input
          type="search"
          role="combobox"
          aria-expanded={saes.length > 0}
          aria-autocomplete="list"
          aria-controls={saes.length > 0 ? "sae-catalogue-list" : undefined}
          aria-activedescendant={chosen ? optionId(chosen) : undefined}
          value={query}
          placeholder="Release, hook, layer 8, 16k"
          spellCheck={false}
          autoComplete="off"
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown" || e.key === "ArrowUp" || e.key === "PageDown" || e.key === "PageUp") {
              e.preventDefault();
              move({ ArrowDown: 1, ArrowUp: -1, PageDown: 10, PageUp: -10 }[e.key]);
            } else if (e.key === "Enter") {
              e.preventDefault();
              loadRef.current?.focus();
            }
          }}
          style={{ width: "min(420px, 100%)" }}
        />
      </Field>
      {saes.length === 0 ? (
        <p className={s.small}>No SAE for {modelId} matches “{query}”.</p>
      ) : (
        <ul id="sae-catalogue-list" role="listbox" aria-label={`Published SAEs for ${modelId}`} className={p.list} ref={listRef}>
          {saes.map((x, i) => (
            <li
              key={`${x.release}/${x.id}`}
              id={optionId(x)}
              role="option"
              aria-selected={x === chosen}
              className={p.option}
              data-reads={x.reads}
              onClick={() => setActive(i)}
            >
              <span className={p.dot} aria-hidden="true" />
              <span className={p.name}>
                <span className={p.release}>{x.release}</span>
                <span className={p.id}>{x.id}</span>
              </span>
              <span className={p.where}>{catalogueWhere(x)}</span>
              <span className={p.width}>{x.width ? `${count(x.width)} features` : "Width not stated"}</span>
              <span className={p.status}>{STATUS[x.reads]}</span>
            </li>
          ))}
        </ul>
      )}
      {data.matches > saes.length && (
        <p className={s.small}>
          Showing {count(saes.length)} of {count(data.matches)}. Search to narrow them.
        </p>
      )}
      {chosen && <Details sae={chosen} busy={busy} onLoad={onLoad} loadRef={loadRef} />}
    </section>
  );
}

function Details({ sae, busy, onLoad, loadRef }: { sae: CatalogueSAE; busy: boolean; onLoad: (sae: CatalogueSAE) => void; loadRef: RefObject<HTMLButtonElement | null> }) {
  const fit = sae.verified?.fit;
  return (
    <div className={p.details} aria-label={`The chosen SAE: ${sae.release}, ${sae.id}`} role="group">
      <div className={p.detailsHead}>
        <div>
          <p className={s.small}>{sae.release}</p>
          <h3 className={p.title}>{sae.id}</h3>
        </div>
        <Button ref={loadRef} variant="primary" disabled={busy || sae.reads === "refused"} onClick={() => onLoad(sae)}>
          Load this SAE
        </Button>
      </div>
      <dl className={p.facts}>
        <div>
          <dt>Published in</dt>
          <dd>
            {sae.repo}
            <span className={p.path}>/{sae.folder}</span>
          </dd>
        </div>
        <div>
          <dt>Reads</dt>
          <dd>
            {catalogueWhere(sae)}
            {sae.hook && <span className={p.path}> · {sae.hook}</span>}
          </dd>
        </div>
        <div>
          <dt>Features</dt>
          <dd>
            {sae.width ? count(sae.width) : "not stated"}
            {sae.architecture && ` · ${sae.architecture}`}
            {sae.l0 != null && ` · about ${count(Math.round(sae.l0))} fire per token, its publisher says`}
          </dd>
        </div>
        <div>
          <dt>Licence</dt>
          <dd>{sae.licence ?? "Its repository doesn't state one"}{sae.gated && " · gated: accept its terms on Hugging Face first"}</dd>
        </div>
      </dl>
      {sae.reads === "exact" ? (
        <p className={s.small}>
          Logogram reads it exactly, as its configuration states it
          {sae.verified
            ? sae.verified.read === "weights"
              ? `; its files at ${shortRevision(sae.verified.revision)} were read whole${fit ? ` and explain ${pct(fit.variance_explained)} of the variance of ${fit.model}'s activations over ${count(fit.tokens)} tokens of plain text, with weight processing ${fit.process_weights ? "on" : "off"}` : ""}`
              : `; its weights file's header at ${shortRevision(sae.verified.revision)} agrees`
            : ""}
          . Loading downloads its configuration and weights at the repository's current commit, which specs record, and checks them again.
        </p>
      ) : (
        <Callout title={sae.reads === "refused" ? "Logogram doesn't read this SAE" : "Logogram checks this SAE when it loads"}>{sae.reason}</Callout>
      )}
    </div>
  );
}
