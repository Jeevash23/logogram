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

from logogram.backends.base import Cancelled, ModelBackend, ModelInfo, Patch
from logogram.engine import (
    EngineError,
    EngineResult,
    LayerFn,
    ProgressFn,
    _answer_tensors,
    _chunks,
    _LayerSources,
    _metric,
    check_gap,
    compute_baselines,
)
from logogram.prompts import PreparedPrompt, group_by_length
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
    n_train = int(round(n * train_fraction))
    if n_train < 1 or n - n_train < 2:
        raise EngineError(
            f"Steering needs at least one prompt pair to compute the direction and two to measure "
            f"it on, but {n} pairs with a training share of {train_fraction:.0%} leaves "
            f"{n_train} and {n - n_train}. Use more prompts or another training share."
        )
    order = list(range(n))
    raw = np.random.PCG64(seed).random_raw(n)
    for t in range(n - 1, 0, -1):
        j = int(raw[t] % np.uint64(t + 1))
        order[t], order[j] = order[j], order[t]
    return sorted(order[:n_train]), sorted(order[n_train:])


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
    baselines = compute_baselines(backend, test, test_groups, batch_size, cancel)
    warnings = check_gap(spec, baselines, test, receiver, reference)
    n = len(test)
    patched_ld = np.zeros((len(sites), n))
    patched_prob = np.zeros((len(sites), n))
    result = EngineResult(
        sites=sites,
        layout=layout,
        prompts=test,
        baselines=baselines,
        receiver=receiver,
        reference=reference,
        patched_ld=patched_ld,
        patched_prob=patched_prob,
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
            toward.table(site.kind, probe).double() - away.table(site.kind, probe).double()
        ).mean(0)
        norm = float(direction.norm())
        result.extra["steering"]["norms"].append(norm)
        control_direction = None
        if exp.control:
            generator = torch.Generator().manual_seed(exp.seed * 1_000_003 + b)
            noise = torch.randn(direction.shape, generator=generator, dtype=torch.float64)
            control_direction = noise / noise.norm() * norm
        # The held-out receivers' own activations, to which the direction is added.
        own = _LayerSources(
            backend, test, test_groups, site.layer, kinds, receiver, batch_size, cancel
        ).table(site.kind, probe)
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
                        own[p].double()
                        + variants[v][0]
                        * (control_direction if variants[v][1] else direction).to(own.device)
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
                logits = backend.final_logits(tokens[local], patch)
                ld, prob = _metric(logits, *_answer_tensors(test, [p for _, p in chunk]))
                for (v, p), value_ld, value_prob in zip(chunk, ld, prob, strict=True):
                    patched_ld[first + v, p] = value_ld
                    patched_prob[first + v, p] = value_prob
                done += len(chunk)
                if on_progress is not None:
                    on_progress(done, total, site.layer)
        if on_layer is not None:
            on_layer(site.layer, list(range(first, first + len(variants))), result)
    return result
