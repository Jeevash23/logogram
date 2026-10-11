import { useState } from "react";

import type { PositionSpec, SiteSetSpec, SiteSpec, UniverseKind } from "../api/types";
import { Button, Checkbox, Field, Input, Segmented, Select } from "../components/ui";
import {
  freeLabel,
  MAX_LABEL,
  MAX_SETS,
  parseSiteQuery,
  SET_SITE_KINDS,
  siteOf,
  siteSetsError,
  UNIVERSE_KINDS,
  UNIVERSE_NAMES,
  universeText,
} from "../lib/circuits";
import { plural } from "../lib/format";
import { siteText } from "../lib/spec";
import { useStore, type SiteSetsScope } from "../store/app";
import s from "./views.module.css";
import c from "./SiteSets.module.css";

/**
 * What a set that keeps its sites replaces: every head, attention output or MLP output of the
 * model except those sites. A layer's attention output is the sum of its heads, so the rest of
 * the model holds one or the other.
 */
export function UniverseChooser({
  value,
  onChange,
}: {
  value: UniverseKind[] | null;
  onChange: (universe: UniverseKind[] | null) => void;
}) {
  const chosen = value ?? [];
  return (
    <div className={c.universe} role="group" aria-label="The rest of the model">
      {UNIVERSE_KINDS.map((kind) => {
        const conflict = (kind === "head" && chosen.includes("attn_out")) || (kind === "attn_out" && chosen.includes("head"));
        return (
          <Checkbox
            key={kind}
            checked={chosen.includes(kind)}
            disabled={conflict && !chosen.includes(kind)}
            onChange={(on) => {
              const next = on ? UNIVERSE_KINDS.filter((k) => k === kind || chosen.includes(k)) : chosen.filter((k) => k !== kind);
              onChange(next.length ? next : null);
            }}
          >
            {UNIVERSE_NAMES[kind].many[0].toUpperCase() + UNIVERSE_NAMES[kind].many.slice(1)}
          </Checkbox>
        );
      })}
    </div>
  );
}

/** Positions a site typed into a set can take: all, the last token, or a named position. */
function positionOptions(labels: string[]): { key: string; label: string; position: PositionSpec }[] {
  return [
    { key: "all", label: "All positions", position: { kind: "all" } },
    { key: "last", label: "Last token", position: { kind: "last" } },
    ...labels.map((label) => ({ key: `label:${label}`, label: `Position ${label}`, position: { kind: "label" as const, label } })),
  ];
}

/**
 * Sets of sites, each intervened on at once: their labels, their sites, and whether each replaces
 * its sites or keeps them and replaces the rest of the model.
 */
export function SiteSetsEditor({
  scope,
  onChange,
  labels,
}: {
  scope: SiteSetsScope;
  onChange: (scope: SiteSetsScope) => void;
  labels: string[];
}) {
  const model = useStore((st) => st.model.info);
  const staged = useStore((st) => st.stagedSites);
  const error = siteSetsError(scope.universe, scope.sets, model ?? null);
  const full = scope.sets.length >= MAX_SETS;
  const keeps = scope.sets.some((set) => set.complement);
  const update = (i: number, set: SiteSetSpec) => onChange({ ...scope, sets: scope.sets.map((x, j) => (j === i ? set : x)) });
  const usable = staged.filter((x) => SET_SITE_KINDS.includes(x.kind));
  return (
    <div className={c.results}>
      <Field
        label="The rest of the model"
        help={
          keeps
            ? `A set that keeps its sites replaces ${universeText(scope.universe)} except them, at every position. A layer's attention output is the sum of its heads, so choose one or the other.`
            : "What a set that keeps its sites replaces. No set keeps its sites yet, so this can stay empty."
        }
      >
        <UniverseChooser value={scope.universe} onChange={(universe) => onChange({ ...scope, universe })} />
      </Field>
      <div>
        <div className={c.sets}>
          {scope.sets.map((set, i) => (
            <SetEditor
              key={i}
              index={i}
              set={set}
              universe={scope.universe}
              labels={labels}
              onChange={(next) => update(i, next)}
              onRemove={scope.sets.length > 1 ? () => onChange({ ...scope, sets: scope.sets.filter((_, j) => j !== i) }) : undefined}
            />
          ))}
        </div>
      </div>
      <div className={c.footer}>
        <Button size="small" icon="plus" disabled={full} onClick={() => onChange({ ...scope, sets: [...scope.sets, { label: freeLabel(scope.sets, `Set ${scope.sets.length + 1}`), sites: [], complement: false }] })}>
          Add a set
        </Button>
        {usable.length > 0 && (
          <Button
            size="small"
            disabled={full}
            title="A set that replaces the sites staged in the model explorer"
            onClick={() => onChange({ ...scope, sets: [...scope.sets, { label: freeLabel(scope.sets, "Staged sites"), sites: usable, complement: false }] })}
          >
            Add the staged sites as a set
          </Button>
        )}
        <span className={c.count}>
          {full ? `${MAX_SETS} sets: a run holds no more. Remove a set to add another.` : `${plural(scope.sets.length, "set")} of at most ${MAX_SETS}`}
        </span>
      </div>
      {error && <p className={s.small} role="note">{error}</p>}
    </div>
  );
}

function SetEditor({
  index,
  set,
  universe,
  labels,
  onChange,
  onRemove,
}: {
  index: number;
  set: SiteSetSpec;
  universe: UniverseKind[] | null;
  labels: string[];
  onChange: (set: SiteSetSpec) => void;
  onRemove?: () => void;
}) {
  const [query, setQuery] = useState("");
  const [at, setAt] = useState("all");
  const options = positionOptions(labels);
  const parsed = parseSiteQuery(query);
  const position = (options.find((o) => o.key === at) ?? options[0]).position;
  const add = () => {
    if (!parsed) return;
    const site: SiteSpec = siteOf(parsed, position);
    if (!set.sites.some((x) => siteText(x) === siteText(site))) onChange({ ...set, sites: [...set.sites, site] });
    setQuery("");
  };
  const name = set.label.trim() || `set ${index + 1}`;
  return (
    <div className={c.set} role="group" aria-label={`Set ${index + 1}: ${name}`}>
      <div className={c.setHead}>
        <Input
          className={c.label}
          value={set.label}
          maxLength={MAX_LABEL}
          onChange={(ev) => onChange({ ...set, label: ev.target.value })}
          aria-label={`Label of set ${index + 1}`}
          placeholder="Label, such as Circuit kept"
        />
        <Segmented
          label={`What set ${index + 1} does`}
          value={set.complement ? "keep" : "replace"}
          onChange={(v) => onChange({ ...set, complement: v === "keep" })}
          options={[
            { value: "replace", label: "Replace these sites" },
            { value: "keep", label: "Keep these, replace the rest" },
          ]}
        />
        {onRemove && (
          <Button size="small" variant="ghost" icon="close" onClick={onRemove} aria-label={`Remove set ${index + 1}, ${name}`} />
        )}
      </div>
      <div className={c.sites}>
        {set.sites.map((site, j) => (
          <span key={`${siteText(site)}-${j}`} className={c.site}>
            {siteText(site)}
            <button type="button" aria-label={`Remove ${siteText(site)} from ${name}`} onClick={() => onChange({ ...set, sites: set.sites.filter((_, k) => k !== j) })}>
              ×
            </button>
          </span>
        ))}
        {set.sites.length === 0 && (
          <span className={c.empty}>
            {set.complement ? `Keeps nothing: replaces ${universeText(universe)}.` : "No sites yet: add one below."}
          </span>
        )}
      </div>
      <div className={c.add}>
        <Input
          className={c.addInput}
          value={query}
          onChange={(ev) => setQuery(ev.target.value)}
          onKeyDown={(ev) => {
            if (ev.key === "Enter") {
              ev.preventDefault();
              add();
            }
          }}
          placeholder="L9 H6 or L3 mlp"
          aria-label={`Site to add to ${name}, like L9 H6, L3 attn or L3 mlp`}
          aria-invalid={query.trim() !== "" && !parsed}
          spellCheck={false}
        />
        <Select value={at} onChange={(ev) => setAt(ev.target.value)} aria-label={`Position of the site to add to ${name}`}>
          {options.map((o) => (
            <option key={o.key} value={o.key}>
              {o.label}
            </option>
          ))}
        </Select>
        <Button size="small" disabled={!parsed} onClick={add}>
          Add site
        </Button>
      </div>
    </div>
  );
}
