"""Sets of sites intervened on at once: circuits, and how much of the behavior they carry.

A set's sites are all patched (or ablated) in the same forward pass. A *complement* set intervenes
on every component of the scope's universe (every head, or every attention and MLP output, at
every position) except its sites, so "keep the circuit, replace the rest" is one set. A site at
one position keeps only that position of its component; its other positions are replaced with the
rest of the model.

Values come from the source prompt (patching), a donor (resample ablation), the mean over prompts
of the same token length at each position (mean ablation), or zeros. Every set is one row of the
results, with the usual per-prompt effects and intervals.

When the scope has a set that replaces the whole universe (a complement set with no sites), every
row is also reported as a share of that set's effect, from the same resamples: for a set that
keeps a circuit, 1 - share is its faithfulness, the part of the behavior the circuit carries
alone (1 when keeping it alone loses nothing, 0 when it does no better than replacing
everything).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from logogram.backends.base import Cancelled, ModelBackend, ModelInfo, Patch, float64
from logogram.engine import (
    EngineError,
    EngineResult,
    LayerFn,
    ProgressFn,
    _chunks,
    check_finite,
    check_gap,
    check_values,
    compute_baselines,
    draw_donors,
    empty_values,
    make_scorer,
)
from logogram.prompts import LengthGroup, PreparedPrompt, group_by_length
from logogram.sites import ResolvedSite, ScopeError, resolve_position, site_label
from logogram.spec import (
    ALL_POSITIONS,
    Ablation,
    ActivationPatching,
    MeanBaseline,
    ResampleBaseline,
    Site,
    SiteSet,
    SiteSetsScope,
    Spec,
    ZeroBaseline,
)

Key = tuple[str, int]


def check_sets(scope: SiteSetsScope, info: ModelInfo) -> None:
    """Refuse sites and universes the loaded model doesn't have."""
    for kind in scope.universe or []:
        if kind not in info.site_kinds:
            raise ScopeError(f"This model has no {kind} site in TransformerLens.")
    for s in scope.sets:
        for site in s.sites:
            if site.layer >= info.n_layers:
                raise ScopeError(
                    f"The set {s.label!r} has a site in layer {site.layer}; this model has "
                    f"{info.n_layers} layers."
                )
            if site.kind not in info.site_kinds:
                raise ScopeError(f"This model has no {site.kind} site in TransformerLens.")
            if site.kind == "head" and site.head is not None and site.head >= info.n_heads:
                raise ScopeError(
                    f"The set {s.label!r} has head {site.head}; this model has {info.n_heads} "
                    "heads."
                )
        kinds = {(site.kind, site.layer) for site in s.sites}
        for site in s.sites:
            if site.kind == "head" and ("attn_out", site.layer) in kinds:
                raise ScopeError(
                    f"The set {s.label!r} has both heads and the attention output of layer "
                    f"{site.layer}; the attention output is the sum of the heads. Use one or the "
                    "other."
                )


def set_rows(scope: SiteSetsScope) -> tuple[list[ResolvedSite], dict[str, Any]]:
    """One result row per set, and the layout that lists them."""
    rows = []
    for i, s in enumerate(scope.sets):
        placeholder = (
            s.sites[0] if s.sites else Site(kind="resid_pre", layer=0, position=ALL_POSITIONS)
        )
        rows.append(
            ResolvedSite(
                index=i,
                site=placeholder,
                row=i,
                col=0,
                label=s.label,
                variant={"complement": s.complement, "size": len(s.sites)},
                variant_key=s.label,
                site_set=s,
            )
        )
    layout = {
        "kind": "site_sets",
        "row_title": "Set",
        "col_title": "",
        "rows": [
            {
                "key": str(i),
                "label": s.label,
                "complement": s.complement,
                "size": len(s.sites),
            }
            for i, s in enumerate(scope.sets)
        ],
        "cols": [{"key": "effect", "label": "effect"}],
        "universe": scope.universe,
    }
    return rows, layout


def replaced(
    s: SiteSet, universe: Sequence[str] | None, prompt: PreparedPrompt, info: ModelInfo
) -> dict[Key, torch.Tensor]:
    """For one prompt, which entries of each activation the set replaces: per (kind, layer), a
    boolean mask over positions ([L]) or positions and heads ([L, H])."""
    L, H = prompt.length, info.n_heads

    def blank(kind: str, value: bool) -> torch.Tensor:
        shape = (L, H) if kind == "head" else (L,)
        return torch.full(shape, value, dtype=torch.bool)

    masks: dict[Key, torch.Tensor] = {}
    if s.complement:
        for kind in universe or []:
            for layer in range(info.n_layers):
                masks[(kind, layer)] = blank(kind, True)
    for site in s.sites:
        key = (site.kind, site.layer)
        if key not in masks:
            masks[key] = blank(site.kind, False)
        mask = masks[key]
        pos = resolve_position(site.position, prompt)
        where: Any = slice(None) if pos is None else pos
        value = not s.complement  # complement sets keep their sites
        if site.kind == "head":
            mask[where, site.head] = value
        else:
            mask[where] = value
    return {k: m for k, m in masks.items() if bool(m.any())}


def tail_masks(
    s: SiteSet, universe: Sequence[str] | None, info: ModelInfo
) -> dict[Key, torch.Tensor]:
    """What the set replaces at the positions of tokens appended to read a continuation, per
    (kind, layer): a component (or head) at every position covers them too; a site at one prompt
    position doesn't. ``[H]`` for heads, a single value otherwise."""
    H = info.n_heads
    masks: dict[Key, torch.Tensor] = {}

    def blank(kind: str, value: bool) -> torch.Tensor:
        return torch.full((H,) if kind == "head" else (), value, dtype=torch.bool)

    if s.complement:
        for kind in universe or []:
            for layer in range(info.n_layers):
                masks[(kind, layer)] = blank(kind, True)
    for site in s.sites:
        if site.position.kind != "all":
            continue
        key = (site.kind, site.layer)
        if key not in masks:
            masks[key] = blank(site.kind, False)
        value = not s.complement
        if site.kind == "head":
            masks[key][site.head] = value
        else:
            masks[key] = torch.tensor(value)
    return masks


class _Means:
    """Mean activations per position over the reference prompts of each length group."""

    def __init__(
        self,
        backend: ModelBackend,
        groups: list[LengthGroup],
        which: str,
        batch_size: int,
        cancel: threading.Event | None,
    ) -> None:
        self.backend = backend
        self.groups = groups
        self.which = which
        self.batch_size = batch_size
        self.cancel = cancel
        self._cache: dict[tuple[int, Key], torch.Tensor] = {}

    def get(self, group_index: int, keys: list[Key]) -> dict[Key, torch.Tensor]:
        missing = [k for k in keys if (group_index, k) not in self._cache]
        if missing:
            group = self.groups[group_index]
            tokens = group.clean if self.which == "clean" else group.corrupt
            sums: dict[Key, torch.Tensor] = {}
            for sl in _chunks(len(group.members), self.batch_size):
                if self.cancel is not None and self.cancel.is_set():
                    raise Cancelled()
                acts = self.backend.capture(tokens[sl], missing)
                for k in missing:
                    part = float64(acts[k]).sum(0)
                    sums[k] = part if k not in sums else sums[k] + part
            for k in missing:
                self._cache[(group_index, k)] = (sums[k] / len(group.members)).float()
        return {k: self._cache[(group_index, k)] for k in keys}


def run_site_sets(
    spec: Spec,
    backend: ModelBackend,
    prompts: list[PreparedPrompt],
    *,
    on_progress: ProgressFn | None = None,
    on_layer: LayerFn | None = None,
    cancel: threading.Event | None = None,
    on_start: Callable[[list[ResolvedSite], dict[str, Any]], None] | None = None,
    receiver_override: str | None = None,
    source_override: str | None = None,
) -> EngineResult:
    scope = spec.scope
    exp = spec.experiment
    assert isinstance(scope, SiteSetsScope)
    if not isinstance(exp, ActivationPatching | Ablation):
        raise ScopeError(
            "Sets of sites are intervened on together by activation patching or ablation. Choose "
            "one of those."
        )
    info = backend.info
    batch_size = spec.execution.batch_size
    check_sets(scope, info)
    sites, layout = set_rows(scope)
    for s in scope.sets:  # every position must resolve in every prompt
        for site in s.sites:
            for prompt in prompts:
                resolve_position(site.position, prompt)
    if on_start is not None:
        on_start(sites, layout)
    groups = group_by_length(prompts)
    group_of = {i: gi for gi, g in enumerate(groups) for i in g.members}
    local_of = {i: li for g in groups for li, i in enumerate(g.members)}
    scorer = make_scorer(spec, prompts)
    baselines = compute_baselines(backend, prompts, groups, batch_size, cancel, scorer)
    check_finite(baselines, info.dtype)

    if isinstance(exp, ActivationPatching):
        receiver, source = (
            ("corrupt", "clean") if exp.direction == "clean_to_corrupt" else ("clean", "corrupt")
        )
        reference, kind = source, "patch"
    else:
        receiver, reference = "clean", "corrupt"
        b = exp.baseline
        if isinstance(b, ZeroBaseline):
            source, kind = "", "zero"
        elif isinstance(b, MeanBaseline):
            source, kind = b.reference, "mean"
        elif isinstance(b, ResampleBaseline):
            source, kind = b.pool, "resample"
        else:  # pragma: no cover
            raise EngineError(f"Unknown baseline {b!r}")
    receiver = receiver_override or receiver
    source = source_override or source
    warnings = check_gap(spec, baselines, prompts, receiver, reference)

    n = len(prompts)
    donors: list[list[int]] | None = None
    k = 1
    if kind == "resample":
        assert isinstance(exp, Ablation) and isinstance(exp.baseline, ResampleBaseline)
        k = exp.baseline.donors
        donors = draw_donors(prompts, groups, k, exp.baseline.seed, same_length=True)
    means = _Means(backend, groups, source, batch_size, cancel) if kind == "mean" else None
    if kind == "mean" and not scorer.single_position():
        raise EngineError(
            "Mean ablation of a set replaces positions with their mean over prompts of the same "
            "length, and the tokens appended to read answers of several tokens have no such "
            "mean. Use zero or resample ablation, or single-token answers."
        )
    if kind == "mean":
        alone = sum(1 for g in groups if len(g.members) == 1)
        if alone:
            warnings.append(
                f"{alone} prompt(s) are the only prompt of their length, so their per-position "
                "mean is their own activation."
            )

    patched, patched_prob, patched_pref = empty_values(len(sites), n)
    result = EngineResult(
        sites=sites,
        layout=layout,
        prompts=prompts,
        baselines=baselines,
        receiver=receiver,
        reference=reference,
        patched=patched,
        patched_prob=patched_prob,
        patched_pref=patched_pref,
        donors=donors,
        warnings=warnings,
        extra={"circuit": {"universe": scope.universe}},
    )
    total = len(sites) * n * k
    done = 0
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        info.dtype
    ]
    for rs in sites:
        s = rs.site_set
        assert s is not None
        for gi, group in enumerate(groups):
            receiver_tokens = group.clean if receiver == "clean" else group.corrupt
            source_tokens = group.clean if source == "clean" else group.corrupt
            masks = {p: replaced(s, scope.universe, prompts[p], info) for p in group.members}
            tails = tail_masks(s, scope.universe, info)
            keys = sorted({key for m in masks.values() for key in m})
            if not keys:
                raise ScopeError(f"The set {s.label!r} replaces nothing in these prompts.")
            rows = [
                (p, d) for p in group.members for d in (donors[p] if donors is not None else [None])
            ]
            for sl in _chunks(len(rows), batch_size):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                chunk = rows[sl]
                values = _values(
                    backend,
                    kind,
                    keys,
                    chunk,
                    gi,
                    group,
                    source_tokens,
                    local_of,
                    group_of,
                    means,
                    info,
                    dtype,
                )
                patches = [
                    Patch(
                        kind=key[0],
                        layer=key[1],
                        values=values[key],
                        mask=torch.stack(
                            [masks[p].get(key, _blank(key, group.length, info)) for p, _ in chunk]
                        ),
                    )
                    for key in keys
                ]
                local = torch.tensor([local_of[p] for p, _ in chunk], dtype=torch.long)
                donor = torch.tensor(
                    [local_of[p if d is None else d] for p, d in chunk], dtype=torch.long
                )
                forward = _continued(
                    backend,
                    patches,
                    tails,
                    None if kind == "zero" else source_tokens[donor],
                    group.length,
                )
                scores = scorer.score(forward, receiver_tokens[local], [p for p, _ in chunk])
                for (p, _), value, prob, pref in zip(
                    chunk, scores.metric, scores.prob, scores.pref, strict=True
                ):
                    patched[rs.index, p] += value / k
                    patched_prob[rs.index, p] += prob / k
                    patched_pref[rs.index, p] += pref / k
                done += len(chunk)
                if on_progress is not None:
                    on_progress(done, total, rs.index)
        if on_layer is not None:
            on_layer(rs.index, [rs.index], result)
    warnings.extend(check_values(patched, "patched values", info.dtype))
    return result


def _continued(
    backend: ModelBackend,
    patches: list[Patch],
    tails: dict[Key, torch.Tensor],
    source: torch.Tensor | None,
    length: int,
) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """A forward pass with the set's patches, extended over tokens appended to read a
    continuation: there the set replaces what it replaces at every position, with the source's
    own values with the same continuation appended (zeros for zero ablation)."""

    def forward(tokens: torch.Tensor, keep: int) -> torch.Tensor:
        if tokens.shape[1] == length:
            return backend.logits(tokens, patches, keep)
        tail = tokens[:, length:]
        extra: dict[Key, torch.Tensor] = {}
        if source is not None:
            full = backend.capture(
                torch.cat([source, tail.to(source.device)], dim=1),
                [(p.kind, p.layer) for p in patches],
            )
            extra = {k: v[:, length:] for k, v in full.items()}
        whole = []
        for p in patches:
            key = (p.kind, p.layer)
            B, T = p.values.shape[0], tail.shape[1]
            values = extra.get(key)
            if values is None:
                values = torch.zeros(
                    (B, T, *p.values.shape[2:]), dtype=p.values.dtype, device=p.values.device
                )
            rule = tails.get(key)
            if rule is None:
                rule = torch.zeros(p.mask.shape[2:], dtype=torch.bool)  # type: ignore[union-attr]
            mask = rule.expand(B, T, *rule.shape).to(p.mask.device)  # type: ignore[union-attr]
            whole.append(
                Patch(
                    kind=p.kind,
                    layer=p.layer,
                    values=torch.cat([p.values, values.to(p.values.device, p.values.dtype)], dim=1),
                    mask=torch.cat([p.mask, mask], dim=1),  # type: ignore[list-item]
                )
            )
        return backend.logits(tokens, whole, keep)

    return forward


def _blank(key: Key, length: int, info: ModelInfo) -> torch.Tensor:
    shape = (length, info.n_heads) if key[0] == "head" else (length,)
    return torch.zeros(shape, dtype=torch.bool)


def _values(
    backend: ModelBackend,
    kind: str,
    keys: list[Key],
    chunk: list[tuple[int, int | None]],
    group_index: int,
    group: LengthGroup,
    source_tokens: torch.Tensor,
    local_of: dict[int, int],
    group_of: dict[int, int],
    means: _Means | None,
    info: ModelInfo,
    dtype: torch.dtype,
) -> dict[Key, torch.Tensor]:
    """The replacement activations for a batch of rows, whole: [B, L, d] or [B, L, H, d_head]."""
    B = len(chunk)
    if kind == "zero":
        out = {}
        for key in keys:
            shape = (
                (B, group.length, info.n_heads, info.d_head)
                if key[0] == "head"
                else (B, group.length, info.d_model)
            )
            out[key] = torch.zeros(shape, dtype=dtype, device=backend.device)
        return out
    if kind == "mean":
        assert means is not None
        mean = means.get(group_index, keys)
        return {key: mean[key].unsqueeze(0).expand(B, *mean[key].shape) for key in keys}
    # Patching reads the other prompt of each pair; resampling reads each row's donor (drawn from
    # the same length group, so positions line up).
    sources = [p if d is None else d for p, d in chunk]
    assert all(group_of[p] == group_index for p in sources)
    local = torch.tensor([local_of[p] for p in sources], dtype=torch.long)
    return backend.capture(source_tokens[local], keys)


def circuit_summary(result: EngineResult, stats: Any, ci: float) -> dict[str, Any]:
    """How much of the behavior each set carries, from the same resamples as the effects.

    * ``share``: the set's mean change in the metric as a share of the set that replaces the
      whole universe; for a set that keeps sites, ``faithfulness`` = 1 - share. For a set that
      removes sites, a share near 1 says removing them does what removing everything does (the
      circuit is complete).
    * ``without``: for a set that keeps one site fewer than another keeping set, the faithfulness
      that site adds (the circuit's minimality, one site at a time).
    * ``interaction``: for a set of two sites that are also intervened on alone, the effect of
      both beyond the sum of the two.
    """
    sites = result.sites
    sets = {rs.index: rs.site_set for rs in sites}
    assert all(s is not None for s in sets.values())
    keys = {i: frozenset(site.model_dump_json() for site in s.sites) for i, s in sets.items()}  # type: ignore[union-attr]
    named = {site.model_dump_json(): site for s in sets.values() for site in s.sites}  # type: ignore[union-attr]
    everything = next(
        (i for i, s in sets.items() if s.complement and not s.sites),  # type: ignore[union-attr]
        None,
    )
    alone = {
        next(iter(keys[i])): i
        for i, s in sets.items()
        if not s.complement and len(s.sites) == 1  # type: ignore[union-attr]
    }
    boot = getattr(stats, "delta_boot", None)
    effect_boot = getattr(stats, "effect_boot", None)
    alpha = (1.0 - ci) / 2.0

    def interval(values: np.ndarray, mean: float) -> dict[str, float] | None:
        if not (np.isfinite(values).all() and np.isfinite(mean) and stats.n >= 2):
            return None
        lo, hi = (float(x) for x in np.quantile(values, [alpha, 1.0 - alpha]))
        return {"mean": float(mean), "sd": float(values.std(ddof=1)), "lo": lo, "hi": hi}

    rows = []
    for i, s in sets.items():
        assert s is not None
        row: dict[str, Any] = {
            "index": i,
            "label": s.label,
            "complement": s.complement,
            "size": len(s.sites),
            "share": None,
            "faithfulness": None,
            "without": None,
            "interaction": None,
        }
        if everything is not None and boot is not None and i != everything:
            with np.errstate(divide="ignore", invalid="ignore"):
                share = interval(
                    boot[i] / boot[everything],
                    stats.delta_mean[i] / stats.delta_mean[everything],
                )
            row["share"] = share
            if s.complement and share is not None:
                row["faithfulness"] = {
                    "mean": 1.0 - share["mean"],
                    "sd": share["sd"],
                    "lo": 1.0 - share["hi"],
                    "hi": 1.0 - share["lo"],
                }
        if everything is not None and boot is not None and s.complement:
            larger = next(
                (
                    j
                    for j, o in sets.items()
                    if o.complement and keys[i] < keys[j] and len(keys[j] - keys[i]) == 1  # type: ignore[union-attr]
                ),
                None,
            )
            if larger is not None:
                (missing,) = keys[larger] - keys[i]
                with np.errstate(divide="ignore", invalid="ignore"):
                    drop = interval(
                        (boot[i] - boot[larger]) / boot[everything],
                        (stats.delta_mean[i] - stats.delta_mean[larger])
                        / stats.delta_mean[everything],
                    )
                if drop is not None:
                    row["without"] = {
                        "of": sets[larger].label,  # type: ignore[union-attr]
                        "site": site_label(named[missing]),
                        "drop": drop,
                    }
        if not s.complement and len(s.sites) == 2 and effect_boot is not None:
            a, b = (alone.get(k) for k in sorted(keys[i]))
            if a is not None and b is not None:
                effect = interval(
                    effect_boot[i] - effect_boot[a] - effect_boot[b],
                    stats.effect_mean[i] - stats.effect_mean[a] - stats.effect_mean[b],
                )
                if effect is not None:
                    row["interaction"] = {
                        "a": sets[a].label,  # type: ignore[union-attr]
                        "b": sets[b].label,  # type: ignore[union-attr]
                        "effect": effect,
                    }
        rows.append(row)
    return {"everything": everything, "rows": rows}
