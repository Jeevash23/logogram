"""Steering: add a direction to the residual stream at several strengths, and measure the effect.

At each steered site (a residual stream state at one token), the direction is the mean of
(other prompt - receiver prompt) over a training split of the pairs. Adding it at strength 1 to a
held-out receiver prompt moves that prompt's activation by the average difference between the two
prompts; the effect is normalized like patching, so 1 means the output moved as far as switching
to the other prompt. Directions are never applied to the prompts that computed them. A random
direction of the same length, at the same strengths, is the control: a direction that matters
should beat it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
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
    _LayerSources,
    check_finite,
    check_gap,
    check_values,
    compute_baselines,
    empty_values,
    make_scorer,
    patched_forward,
)
from logogram.prompts import PreparedPrompt, group_by_length, seeded_split
from logogram.sites import (
    ResolvedSite,
    ScopeError,
    _check_site,
    resolve_position,
    site_label,
)
from logogram.spec import (
    AllPositions,
    LayerComponentsScope,
    Site,
    SitesScope,
    Spec,
    Steering,
    strength_text,
)

RESIDUAL = ("resid_pre", "resid_mid", "resid_post")


def split_pairs(n: int, train_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Positions of the pairs that train the direction and of the held-out pairs it is measured
    on: a seeded shuffle from the raw PCG64 stream, so the split is the same on every machine."""
    train, test = seeded_split(n, train_fraction, seed)
    if len(train) < 1 or len(test) < 2:
        raise EngineError(
            f"Steering needs at least one prompt pair to compute the direction and two to measure "
            f"it on, but {n} pairs with a training share of {train_fraction:.0%} leaves "
            f"{len(train)} and {len(test)}. Use more prompts or another training share."
        )
    return train, test


def steered_sites(spec: Spec, info: ModelInfo, prompts: list[PreparedPrompt]) -> list[Site]:
    """The residual-stream sites a steering spec adds its direction at, one token per prompt."""
    scope = spec.scope
    if isinstance(scope, LayerComponentsScope):
        if len(scope.components) != 1 or scope.components[0] not in RESIDUAL:
            raise ScopeError(
                "Steering adds a direction to the residual stream: choose exactly one residual "
                "site (before, between or after the layer's blocks)."
            )
        sites = [
            Site(kind=scope.components[0], layer=layer, position=scope.position)
            for layer in range(info.n_layers)
        ]
    elif isinstance(scope, SitesScope):
        sites = list(scope.sites)
        if any(s.kind not in RESIDUAL for s in sites):
            raise ScopeError(
                "Steering adds a direction to the residual stream; heads, attention outputs and "
                "MLP outputs can't be steered here. Choose residual sites."
            )
    else:
        raise ScopeError(
            "Steering runs at one residual site in every layer, or at chosen residual sites. "
            "Choose “Attention and MLP per layer” with a single residual component, or pick sites."
        )
    for site in sites:
        if isinstance(site.position, AllPositions):
            raise ScopeError(
                "Steering adds the direction at one token of each prompt. Choose the last token, a "
                "named position or a token index."
            )
        _check_site(site, info)
        for prompt in prompts:
            resolve_position(site.position, prompt)
    return sites


def run_steering(
    spec: Spec,
    backend: ModelBackend,
    prompts: list[PreparedPrompt],
    *,
    on_progress: ProgressFn | None = None,
    on_layer: LayerFn | None = None,
    cancel: threading.Event | None = None,
    on_start: Callable[[list[ResolvedSite], dict[str, Any]], None] | None = None,
) -> EngineResult:
    exp = spec.experiment
    assert isinstance(exp, Steering)
    info = backend.info
    batch_size = spec.execution.batch_size
    base = steered_sites(spec, info, prompts)
    train_at, test_at = split_pairs(len(prompts), exp.train_fraction, exp.seed)
    train = [prompts[i] for i in train_at]
    test = [prompts[i] for i in test_at]
    receiver = exp.apply_to
    reference = "corrupt" if receiver == "clean" else "clean"

    # The variants of each site: every strength along the direction, then along the control.
    variants = [(c, False) for c in exp.coefficients]
    if exp.control:
        variants += [(c, True) for c in exp.coefficients]
    keys = [
        f"random {strength_text(c)}" if control else strength_text(c) for c, control in variants
    ]
    uniform = len({(s.kind, s.position.model_dump_json()) for s in base}) == 1
    sites: list[ResolvedSite] = []
    for b, site in enumerate(base):
        for v, ((coefficient, control), key) in enumerate(zip(variants, keys, strict=True)):
            sites.append(
                ResolvedSite(
                    index=len(sites),
                    site=site,
                    row=b,
                    col=v,
                    label=f"{site_label(site)} {key}",
                    variant={"coefficient": coefficient, "control": control},
                    variant_key=key,
                )
            )
    layout = {
        "kind": "steering",
        "site": base[0].kind if uniform else None,
        "row_title": "Layer" if uniform else "Site",
        "col_title": "Strength",
        "rows": [
            {"key": str(b), "label": str(s.layer) if uniform else site_label(s)}
            for b, s in enumerate(base)
        ],
        "cols": [
            {"key": key, "label": key, "coefficient": c, "control": control}
            for (c, control), key in zip(variants, keys, strict=True)
        ],
    }
    if on_start is not None:
        on_start(sites, layout)

    test_groups = group_by_length(test)
    train_groups = group_by_length(train)
    scorer = make_scorer(spec, test)
    baselines = compute_baselines(backend, test, test_groups, batch_size, cancel, scorer)
    check_finite(baselines, info.dtype)
    warnings = check_gap(spec, baselines, test, receiver, reference)
    n = len(test)
    patched, patched_prob, patched_pref = empty_values(len(sites), n)
    result = EngineResult(
        sites=sites,
        layout=layout,
        prompts=test,
        baselines=baselines,
        receiver=receiver,
        reference=reference,
        patched=patched,
        patched_prob=patched_prob,
        patched_pref=patched_pref,
        warnings=warnings,
        extra={
            "steering": {
                "train": [p.index for p in train],
                "test": [p.index for p in test],
                "norms": [],
            }
        },
    )
    model_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = model_dtype[info.dtype]
    local_of = {i: li for g in test_groups for li, i in enumerate(g.members)}
    total = len(sites) * n
    done = 0
    for b, site in enumerate(base):
        probe = ResolvedSite(0, site, 0, 0, "")
        kinds = [site.kind]
        # The direction: mean (reference - receiver) over the training pairs, at the site's token.
        toward = _LayerSources(
            backend, train, train_groups, site.layer, kinds, reference, batch_size, cancel
        )
        away = _LayerSources(
            backend, train, train_groups, site.layer, kinds, receiver, batch_size, cancel
        )
        direction = (
            float64(toward.table(site.kind, probe)) - float64(away.table(site.kind, probe))
        ).mean(0)
        norm = float(direction.norm())
        result.extra["steering"]["norms"].append(norm)
        control_direction = None
        if exp.control:
            generator = torch.Generator().manual_seed(exp.seed * 1_000_003 + b)
            # float64 on the CPU: a seeded CPU generator draws it, whatever the model's device.
            noise = torch.randn(direction.shape, generator=generator, dtype=torch.float64)
            control_direction = noise / noise.norm() * norm
        # The held-out receivers' own activations, to which the direction is added.
        own = float64(
            _LayerSources(
                backend, test, test_groups, site.layer, kinds, receiver, batch_size, cancel
            ).table(site.kind, probe)
        )
        positions = [resolve_position(site.position, p) for p in test]
        first = b * len(variants)
        for group in test_groups:
            tokens = group.clean if receiver == "clean" else group.corrupt
            rows = [(v, p) for v in range(len(variants)) for p in group.members]
            for sl in _chunks(len(rows), batch_size):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                chunk = rows[sl]
                values = torch.stack(
                    [
                        own[p]
                        + variants[v][0]
                        * _along(variants[v][1], direction, control_direction).to(own.device)
                        for v, p in chunk
                    ]
                ).to(dtype)
                patch = Patch(
                    kind=site.kind,
                    layer=site.layer,
                    values=values,
                    positions=torch.tensor([positions[p] for _, p in chunk], dtype=torch.long),
                )
                local = torch.tensor([local_of[p] for _, p in chunk], dtype=torch.long)
                scores = scorer.score(
                    patched_forward(backend, patch), tokens[local], [p for _, p in chunk]
                )
                for (v, p), value, prob, pref in zip(
                    chunk, scores.metric, scores.prob, scores.pref, strict=True
                ):
                    patched[first + v, p] = value
                    patched_prob[first + v, p] = prob
                    patched_pref[first + v, p] = pref
                done += len(chunk)
                if on_progress is not None:
                    on_progress(done, total, site.layer)
        if on_layer is not None:
            on_layer(site.layer, list(range(first, first + len(variants))), result)
    warnings.extend(check_values(patched, "steered values", info.dtype))
    return result


def control_comparison(result: EngineResult, stats: Any, ci: float) -> list[dict[str, Any]]:
    """Each steered site and strength against its random control, paired: the same held-out
    prompts and the same resamples measure both, so the difference of their effects' magnitudes,
    |direction| - |control|, gets its own interval. A direction beats its control where that
    interval lies above zero (pushing harder in either direction counts; a direction that does
    significantly less than chance doesn't)."""
    index = {
        (rs.row, rs.variant["coefficient"], rs.variant["control"]): rs.index
        for rs in result.sites
        if rs.variant is not None
    }
    boot = stats.effect_boot
    out: list[dict[str, Any]] = []
    if boot is None:
        return out
    alpha = (1.0 - ci) / 2.0
    for (row, coefficient, control), i in sorted(index.items(), key=lambda kv: kv[1]):
        j = index.get((row, coefficient, True))
        if control or j is None:
            continue
        difference = np.abs(boot[i]) - np.abs(boot[j])
        if not np.isfinite(difference).all() or stats.n < 2:
            lo = hi = None
            beats = False
        else:
            lo, hi = (float(x) for x in np.quantile(difference, [alpha, 1.0 - alpha]))
            beats = lo > 0
        mean = abs(float(stats.effect_mean[i])) - abs(float(stats.effect_mean[j]))
        out.append(
            {
                "row": row,
                "coefficient": float(coefficient),
                "index": i,
                "control_index": j,
                "difference": mean if np.isfinite(mean) else None,
                "lo": lo,
                "hi": hi,
                "beats_control": beats,
            }
        )
    return out


def control_warnings(comparison: list[dict[str, Any]], ci: float) -> list[str]:
    """Warn when the direction beats its random control at no site and strength."""
    compared = [c for c in comparison if c["lo"] is not None]
    if not compared or any(c["beats_control"] for c in compared):
        return []
    return [
        f"At no site and strength does the direction move the prompts further than its random "
        f"control (the {ci:.0%} interval of the paired difference |direction| − |control| never "
        "lies above zero), so these results show no effect of the direction. Steering by a mean "
        "difference needs pairs that differ the same way: in IOI prompts that mix the ABBA and "
        "BABA orders, the differences cancel out."
    ]


def _along(
    control: bool, direction: torch.Tensor, control_direction: torch.Tensor | None
) -> torch.Tensor:
    """The direction a variant adds: the steering direction, or its random control."""
    if not control:
        return direction
    assert control_direction is not None  # a control variant exists only with a control
    return control_direction
