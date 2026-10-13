"""Steering: add a direction to the model at several strengths, and measure what it does.

The direction is one of two:

* The mean difference. At each steered site (a residual stream state at one token), the mean of
  (other prompt - receiver prompt) over a training split of the pairs. Adding it at strength 1 to
  a held-out receiver prompt moves that prompt's activation by the average difference between the
  two prompts. Directions are never applied to the prompts that computed them.
* An SAE feature. Its decoder direction W_dec[f], added where the SAE's output is read into the
  model: an SAE on a residual stream site, an attention output or an MLP output adds it to that
  site of its layer, a transcoder to the MLP output it writes, and an SAE on the attention heads'
  outputs (z) to those outputs, before the output projection carries them into the attention
  output. Adding W_dec[f] x a is what the feature writes when its activation is a, so strength 1
  is the feature's typical activation: the mean of its positive activations at the steered token
  of the training split's prompts, clean and corrupt. An SAE that standardizes its inputs scales
  what a feature writes by each input's standard deviation, so there the vector is also scaled by
  the spread of each held-out prompt's own activation at that token.

The effect is normalized like patching, so 1 means the output moved as far as switching to the
other prompt. A random direction of the same length as the steering vector at strength 1, at the
same strengths, is the control: a direction that matters should beat it.

Each prompt is read at its own token: between prompts of different lengths, the mean difference
takes the other prompt's token at the same label (or the last token), never a token index.

With continuations, the run also records what the model goes on to say: greedy continuations of
the first held-out prompts (the most likely token each time, ties to the lowest token id),
without steering, at each strength and with the control. Each forward pass adds the vector at the
steered site and token of the prompt, as when it was measured; the generated tokens aren't steered
themselves.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from logogram.backends.base import Cancelled, ModelBackend, ModelInfo, Patch, Patches, float64
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
from logogram.prompts import LengthGroup, PreparedPrompt, group_by_length, seeded_split
from logogram.sites import (
    ResolvedSite,
    ScopeError,
    _check_site,
    check_lined_up,
    resolve_position,
    site_label,
)
from logogram.spec import (
    AllPositions,
    FeatureDirection,
    LayerComponentsScope,
    SavedDirection,
    Site,
    SitesScope,
    Spec,
    Steering,
    describe_position,
    describe_sae_site,
    strength_text,
)

RESIDUAL = ("resid_pre", "resid_mid", "resid_post")

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


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


# -- where an SAE feature is added ---------------------------------------------------------------


def feature_site(sae: Any) -> str:
    """The site a steering scope names to steer along one of the SAE's features: the site the SAE
    writes. An SAE on the attention heads' outputs (z) writes them before the output projection,
    which turns them into the layer's attention output, so its scope names that."""
    return "attn_out" if sae.site == "head" else str(sae.site)


def writes_where(sae: Any) -> str:
    """Where the SAE's output is read into the model, in words: where a feature is added."""
    return describe_sae_site(str(sae.site), int(sae.layer), bool(sae.transcoder))


def _feature_sites(spec: Spec, info: ModelInfo, sae: Any) -> list[Site]:
    """The sites a steering spec along an SAE feature names: where the SAE writes, and nothing
    else."""
    from logogram.features import check_sae_fits

    exp = spec.experiment
    assert isinstance(exp, Steering) and isinstance(exp.direction, FeatureDirection)
    feature = exp.direction.feature
    if sae is None:
        raise EngineError("This spec steers along an SAE feature, but no SAE is loaded.")
    check_sae_fits(sae, info)
    if feature >= sae.d_sae:
        raise ScopeError(
            f"The SAE has {sae.d_sae} features (0 to {sae.d_sae - 1}), so there is no feature "
            f"{feature} to steer along."
        )
    kind = feature_site(sae)
    name = site_label(Site(kind=kind, layer=sae.layer, position=AllPositions()))  # type: ignore[arg-type]
    if sae.site == "head":
        name += ", the attention output those heads' outputs make"
    needed = (
        f"Steering along SAE feature {feature} adds its decoder direction where the SAE writes: "
        f"{writes_where(sae)}. Choose that site alone as the scope ({name}), at one token of "
        "each prompt"
    )
    scope = spec.scope
    if not isinstance(scope, SitesScope):
        raise ScopeError(f"{needed}, not a site in every layer.")
    for site in scope.sites:
        if site.kind != kind or site.layer != sae.layer:
            raise ScopeError(f"{needed}; {site_label(site)} isn't it.")
    return list(scope.sites)


def steered_sites(
    spec: Spec, info: ModelInfo, prompts: list[PreparedPrompt], sae: Any = None
) -> list[Site]:
    """The sites a steering spec adds its direction at, one token per prompt: residual stream
    sites for the mean difference, the SAE's own site for an SAE feature."""
    exp = spec.experiment
    scope = spec.scope
    if isinstance(exp, Steering) and isinstance(exp.direction, FeatureDirection):
        sites = _feature_sites(spec, info, sae)
    elif isinstance(scope, LayerComponentsScope):
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
                "Steering by the mean difference adds it to the residual stream; heads, their "
                "queries, keys and values, attention outputs, MLP outputs and neurons can't be "
                "steered by it. Choose residual sites, or steer along an SAE feature where its SAE "
                "writes."
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
            for side in ("clean", "corrupt"):
                resolve_position(site.position, prompt, side)
    if not (isinstance(exp, Steering) and isinstance(exp.direction, FeatureDirection)):
        # The mean difference compares each prompt's token with the other prompt's: between
        # prompts of different lengths, at a label or the last token.
        check_lined_up([site.position for site in sites], prompts, "steer")
    return sites


def _flat(sae: Any, x: torch.Tensor) -> torch.Tensor:
    """Activations at one token of each prompt (``[n, ...]``) as the SAE reads them: heads'
    outputs side by side."""
    return x.reshape(x.shape[0], -1) if sae.site_in == "head" else x


def typical_activation(
    backend: ModelBackend,
    sae: Any,
    feature: int,
    prompts: list[PreparedPrompt],
    groups: list[LengthGroup],
    site: Site,
    batch_size: int,
    cancel: threading.Event | None = None,
) -> tuple[float, int, int]:
    """The feature's typical activation where it is steered: the mean of its positive activations
    at the site's token of each of ``prompts``, clean and corrupt. Returns it, how many of those
    activations were positive and how many there were. Refused when none is positive: the
    feature never fires there, so it has no typical activation."""
    probe = ResolvedSite(0, site, 0, 0, "")
    key = str(sae.site_in)
    column: list[torch.Tensor] = []
    for which in ("clean", "corrupt"):
        acts = _LayerSources(
            backend, prompts, groups, sae.layer, [key], which, batch_size, cancel
        ).table(key, probe)
        x = _flat(sae, acts)
        # In batches: an SAE has tens of thousands of features per token.
        for sl in _chunks(x.shape[0], batch_size):
            f, _ = sae.encode(x[sl].to(sae.device))
            column.append(float64(f[:, feature]).cpu())
    values = torch.cat(column)
    fired = values[values > 0]
    if fired.numel() == 0:
        raise ScopeError(
            f"SAE feature {feature} never fires at {describe_position(site.position)} of the "
            f"{len(prompts)} training pairs' prompts (clean or corrupt), so it has no typical "
            "activation to set strength 1 by. Choose a feature that fires there (Explore → "
            "Features shows where each one fires), another position, or more prompts."
        )
    return float(fired.mean()), int(fired.numel()), int(values.numel())


# -- the steered forward passes ----------------------------------------------------------------


@dataclass
class _Steered:
    """One steered site: where its vector is added, the held-out receivers' own activations
    there (``[n, ...]``, float64) at their tokens, and each receiver's vector at strength 1 along
    the direction and along its random control."""

    kind: str  # the site the vector is added to: the heads' outputs for an SAE on z
    layer: int
    own: torch.Tensor
    positions: list[int]
    along: torch.Tensor
    control: torch.Tensor | None
    # The vector's length at strength 1 (the mean over the held-out prompts, when it is scaled
    # by each prompt's spread).
    norm: float

    def patch(
        self, chunk: list[tuple[int, int]], variants: list[tuple[float, bool]], dtype: torch.dtype
    ) -> Patches:
        """The patch that steers each (variant, held-out prompt) row of ``chunk``: its own
        activation plus the variant's strength times its vector, at its token."""
        values = torch.stack(
            [
                self.own[p] + variants[v][0] * self._vector(variants[v][1], p).to(self.own.device)
                for v, p in chunk
            ]
        ).to(dtype)
        positions = torch.tensor([self.positions[p] for _, p in chunk], dtype=torch.long)
        if self.kind != "head":
            return Patch(kind=self.kind, layer=self.layer, values=values, positions=positions)
        # Every head's output at one token of each row (values: [B, heads, d_head]).
        rows = values.shape[0]
        return [
            Patch(
                kind="head",
                layer=self.layer,
                values=values[:, h],
                heads=torch.full((rows,), h, dtype=torch.long),
                positions=positions,
            )
            for h in range(values.shape[1])
        ]

    def _vector(self, control: bool, p: int) -> torch.Tensor:
        if not control:
            return self.along[p]
        assert self.control is not None  # a control variant exists only with a control
        return self.control[p]


def random_direction(shape: torch.Size, seed: int, length: float) -> torch.Tensor:
    """A random direction of ``length``, the control of a steered site: float64 on the CPU, where
    a seeded CPU generator draws it, whatever the model's device. Each site draws its own, with
    the split's seed times 1,000,003 plus its row as the seed."""
    generator = torch.Generator().manual_seed(seed)
    # float64 on the CPU: a seeded CPU generator draws it, whatever the model's device.
    noise = torch.randn(shape, generator=generator, dtype=torch.float64)
    return noise / noise.norm() * length


def run_steering(
    spec: Spec,
    backend: ModelBackend,
    prompts: list[PreparedPrompt],
    *,
    on_progress: ProgressFn | None = None,
    on_layer: LayerFn | None = None,
    cancel: threading.Event | None = None,
    on_start: Callable[[list[ResolvedSite], dict[str, Any]], None] | None = None,
    sae: Any = None,
    directions: dict[str, Any] | None = None,
) -> EngineResult:
    exp = spec.experiment
    assert isinstance(exp, Steering)
    saved = None
    if isinstance(exp.direction, SavedDirection):
        name = exp.direction.direction.name
        saved = (directions or {}).get(name)
        if saved is None:
            raise EngineError(
                f"Steering along {name} needs that direction, which this run didn't load."
            )
        if saved.dimensions != 1:
            raise EngineError(
                f"Steering adds one direction, and {name} has {saved.dimensions} dimensions. "
                "Save one of its vectors as a direction of its own."
            )
    info = backend.info
    batch_size = spec.execution.batch_size
    feature = exp.direction.feature if isinstance(exp.direction, FeatureDirection) else None
    base = steered_sites(spec, info, prompts, sae)
    train_at, test_at = split_pairs(len(prompts), exp.train_fraction, exp.seed)
    train = [prompts[i] for i in train_at]
    test = [prompts[i] for i in test_at]
    receiver = exp.apply_to
    reference = "corrupt" if receiver == "clean" else "clean"
    # The held-out prompts continued, in split order.
    shown = test[: exp.continuations.prompts] if exp.continuations is not None else []
    if exp.continuations is not None:
        check_room(shown, exp.continuations.tokens, info.n_ctx, receiver)

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

    # The steered prompts run in batches of their own length.
    test_groups = group_by_length(test, receiver)
    train_groups = group_by_length(train, receiver)
    scorer = make_scorer(spec, test)
    baselines = compute_baselines(backend, test, test_groups, batch_size, cancel, scorer)
    check_finite(baselines, info.dtype)
    warnings = check_gap(spec, baselines, test, receiver, reference)
    n = len(test)
    patched, patched_prob, patched_pref = empty_values(len(sites), n)
    steering: dict[str, Any] = {
        "train": [p.index for p in train],
        "test": [p.index for p in test],
        "norms": [],
    }
    if saved is not None:
        steering["direction"] = {"name": saved.name, "sha256": saved.sha256, "along": []}
    if feature is not None:
        steering["feature"] = {
            "feature": feature,
            "sae": {"repo": sae.repo, "path": sae.path, "revision": sae.revision},
            "site": str(sae.site),
            "layer": int(sae.layer),
            "transcoder": bool(sae.transcoder),
            "standardized": bool(sae.standardizes),
            "activations": [],
            "fired": [],
            "measured": [],
        }
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
        extra={"steering": steering},
    )
    dtype = DTYPES[info.dtype]
    local_of = {i: li for g in test_groups for li, i in enumerate(g.members)}
    continued = len(shown) * (1 + len(base) * len(variants))
    total = len(sites) * n + continued * (exp.continuations.tokens if exp.continuations else 0)
    done = 0

    def advance(rows: int, layer: int) -> None:
        nonlocal done
        done += rows
        if on_progress is not None:
            on_progress(done, total, layer)

    generated: dict[tuple[int, int, int], list[int]] = {}
    if saved is not None:
        from logogram.directions import check_fits

        warnings += check_fits(saved, info, "the residual stream")
    for b, site in enumerate(base):
        if saved is not None:
            steered, along, spread = _saved_direction(
                backend,
                spec,
                saved.vectors[0],
                site,
                b,
                train,
                train_groups,
                test,
                test_groups,
                cancel,
            )
            steering["direction"]["along"].append(along)
            if spread < 1e-9:
                warnings.append(
                    f"At {site_label(site)}, the training pairs' prompts don't differ along "
                    f"{saved.name} (their mean difference along it is {along:.3g}), so steering "
                    "along it at any strength adds next to nothing there."
                )
            elif abs(along) < 0.25 * spread:
                warnings.append(
                    f"At {site_label(site)}, the training pairs differ along {saved.name} in both "
                    f"directions: their mean difference along it is {along:.3g}, though each "
                    f"pair's is {spread:.3g} in size on average. Steering adds the mean to every "
                    "prompt, so it adds little, and moves some prompts away from their pair. "
                    "Steer prompts whose pairs differ the same way along it, or patch along it "
                    "(subspace patching), which takes each prompt's own pair's value."
                )
        elif feature is None:
            steered = _mean_difference(
                backend, spec, site, b, train, train_groups, test, test_groups, cancel
            )
        else:
            steered, typical = _feature_direction(
                backend, spec, sae, feature, site, b, train, train_groups, test, test_groups, cancel
            )
            for name, value in zip(("activations", "fired", "measured"), typical, strict=True):
                steering["feature"][name].append(value)
        steering["norms"].append(steered.norm)
        first = b * len(variants)
        for group in test_groups:
            tokens = group.tokens(receiver)
            rows = [(v, p) for v in range(len(variants)) for p in group.members]
            for sl in _chunks(len(rows), batch_size):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                chunk = rows[sl]
                patch = steered.patch(chunk, variants, dtype)
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
                advance(len(chunk), site.layer)
        if on_layer is not None:
            on_layer(site.layer, list(range(first, first + len(variants))), result)
        if shown:
            # What the steered model goes on to say: the first held-out prompts, which come first
            # in ``test``, so their positions there index ``steered``.
            assert exp.continuations is not None
            for (p, v), ids in _continue(
                backend,
                shown,
                receiver,
                exp.continuations.tokens,
                batch_size,
                cancel,
                lambda rows, layer=site.layer: advance(rows, layer),
                steered,
                variants,
                dtype,
            ).items():
                generated[(p, b, v)] = ids
    warnings.extend(check_values(patched, "steered values", info.dtype))
    if shown:
        assert exp.continuations is not None
        plain = _continue(
            backend,
            shown,
            receiver,
            exp.continuations.tokens,
            batch_size,
            cancel,
            lambda rows: advance(rows, base[-1].layer),
        )
        result.extra["continuations"] = continuations_record(
            backend,
            shown,
            receiver,
            exp.continuations.tokens,
            base,
            variants,
            keys,
            plain,
            generated,
        )
    return result


def _mean_difference(
    backend: ModelBackend,
    spec: Spec,
    site: Site,
    b: int,
    train: list[PreparedPrompt],
    train_groups: list[LengthGroup],
    test: list[PreparedPrompt],
    test_groups: list[LengthGroup],
    cancel: threading.Event | None,
) -> _Steered:
    """The mean of (reference - receiver) over the training pairs, at the site's token, the same
    for every held-out prompt; its control, a random direction of the same length."""
    exp = spec.experiment
    assert isinstance(exp, Steering)
    receiver = exp.apply_to
    reference = "corrupt" if receiver == "clean" else "clean"
    batch_size = spec.execution.batch_size
    probe = ResolvedSite(0, site, 0, 0, "")
    kinds = [site.kind]
    toward = _LayerSources(
        backend, train, train_groups, site.layer, kinds, reference, batch_size, cancel
    )
    away = _LayerSources(
        backend, train, train_groups, site.layer, kinds, receiver, batch_size, cancel
    )
    direction = (
        float64(toward.table(site.kind, probe)) - float64(away.table(site.kind, probe))
    ).mean(0)
    control = None
    if exp.control:
        control = random_direction(
            direction.shape, exp.seed * 1_000_003 + b, float(direction.norm())
        )
    # The held-out receivers' own activations, to which the direction is added.
    own = float64(
        _LayerSources(
            backend, test, test_groups, site.layer, kinds, receiver, batch_size, cancel
        ).table(site.kind, probe)
    )
    n = len(test)
    return _Steered(
        kind=site.kind,
        layer=site.layer,
        own=own,
        positions=[_position(site, p, exp.apply_to) for p in test],
        along=direction.expand(n, *direction.shape),
        control=None if control is None else control.expand(n, *control.shape),
        norm=float(direction.norm()),
    )


def _saved_direction(
    backend: ModelBackend,
    spec: Spec,
    unit: torch.Tensor,
    site: Site,
    b: int,
    train: list[PreparedPrompt],
    train_groups: list[LengthGroup],
    test: list[PreparedPrompt],
    test_groups: list[LengthGroup],
    cancel: threading.Event | None,
) -> tuple[_Steered, float, float]:
    """A saved direction's unit vector, scaled by the mean difference between the two prompts of
    each training pair along it (reference - receiver, at the site's token): the same for every
    held-out prompt. Its control: a random direction of the same length. Also the mean size of
    each pair's difference along it, to tell pairs that differ both ways along it."""
    exp = spec.experiment
    assert isinstance(exp, Steering)
    receiver = exp.apply_to
    reference = "corrupt" if receiver == "clean" else "clean"
    batch_size = spec.execution.batch_size
    probe = ResolvedSite(0, site, 0, 0, "")
    kinds = [site.kind]
    toward = _LayerSources(
        backend, train, train_groups, site.layer, kinds, reference, batch_size, cancel
    )
    away = _LayerSources(
        backend, train, train_groups, site.layer, kinds, receiver, batch_size, cancel
    )
    difference = float64(toward.table(site.kind, probe)) - float64(away.table(site.kind, probe))
    u = float64(unit).to(difference.device)
    projected = difference @ u
    along = float(projected.mean())
    spread = float(projected.abs().mean())
    direction = along * u
    control = None
    if exp.control:
        control = random_direction(
            direction.shape, exp.seed * 1_000_003 + b, float(direction.norm())
        )
    own = float64(
        _LayerSources(
            backend, test, test_groups, site.layer, kinds, receiver, batch_size, cancel
        ).table(site.kind, probe)
    )
    n = len(test)
    steered = _Steered(
        kind=site.kind,
        layer=site.layer,
        own=own,
        positions=[_position(site, p, exp.apply_to) for p in test],
        along=direction.expand(n, *direction.shape),
        control=None if control is None else control.expand(n, *control.shape),
        norm=float(direction.norm()),
    )
    return steered, along, spread


def _feature_direction(
    backend: ModelBackend,
    spec: Spec,
    sae: Any,
    feature: int,
    site: Site,
    b: int,
    train: list[PreparedPrompt],
    train_groups: list[LengthGroup],
    test: list[PreparedPrompt],
    test_groups: list[LengthGroup],
    cancel: threading.Event | None,
) -> tuple[_Steered, tuple[float, int, int]]:
    """The feature's decoder direction times its typical activation, added where the SAE writes
    (the heads' outputs, for an SAE on z), and scaled for each held-out prompt by the spread of
    its own input for an SAE that standardizes its inputs; its control, a random direction of
    the same length. Also the typical activation, how often it fired and on how many."""
    exp = spec.experiment
    assert isinstance(exp, Steering)
    batch_size = spec.execution.batch_size
    typical = typical_activation(
        backend, sae, feature, train, train_groups, site, batch_size, cancel
    )
    probe = ResolvedSite(0, site, 0, 0, "")
    kind = str(sae.site)
    # The input the SAE reads matters only when it scales what a feature writes.
    read = str(sae.site_in)
    kinds = list(dict.fromkeys([kind, read])) if sae.standardizes else [kind]
    sources = _LayerSources(
        backend, test, test_groups, sae.layer, kinds, exp.apply_to, batch_size, cancel
    )
    own = float64(sources.table(kind, probe))  # [n, d], or [n, heads, d_head] for z
    n = len(test)
    shape = own.shape[1:]
    unit = float64(sae.params.W_dec[feature]).reshape(shape)
    vector = typical[0] * unit
    # 1 for each prompt, or for an SAE that standardizes its inputs, the spread of each held-out
    # prompt's own input at the token: what decoding scales a feature's write by.
    spread = (
        sae.spread(_flat(sae, sources.table(read, probe)).to(sae.device))
        if sae.standardizes
        else None
    )
    scale = float64(torch.ones(n) if spread is None else spread.reshape(n)).cpu()
    scale = scale.reshape(n, *([1] * len(shape)))
    along = scale.to(vector.device) * vector
    length = float(vector.norm())
    control = None
    if exp.control:
        control = scale * random_direction(shape, exp.seed * 1_000_003 + b, length)
    return (
        _Steered(
            kind=kind,
            layer=sae.layer,
            own=own,
            positions=[_position(site, p, exp.apply_to) for p in test],
            along=along,
            control=control,
            norm=length if spread is None else length * float(scale.mean()),
        ),
        typical,
    )


def _position(site: Site, prompt: PreparedPrompt, side: str) -> int:
    at = resolve_position(site.position, prompt, side)
    assert at is not None  # steering adds at one token, never at every position
    return at


# -- what the steered model says -----------------------------------------------------------------


def check_room(prompts: list[PreparedPrompt], tokens: int, n_ctx: int, side: str = "clean") -> None:
    """Refuse continuations of ``side``'s prompts that would run past the tokens the model reads
    at once."""
    if not prompts:
        return
    longest = max(p.length_of(side) for p in prompts)
    if longest + tokens > n_ctx:
        room = n_ctx - longest
        fix = (
            f"Ask for at most {room} new token{'s' if room != 1 else ''}"
            if room > 0
            else "Use shorter prompts"
        )
        raise EngineError(
            f"Continuing a prompt of {longest} tokens by {tokens} would pass the {n_ctx} tokens "
            f"this model reads at once. {fix}."
        )


def greedy_next(logits: torch.Tensor, size: int) -> torch.Tensor:
    """The most likely next token of each row of ``logits`` (``[B, vocab]``) among the first
    ``size`` token ids (those the tokenizer has), ties to the lowest token id."""
    scores = logits[:, :size]
    if not torch.isfinite(scores).all():
        raise EngineError(
            "The model produced values that aren't finite numbers while continuing the text, so "
            "there is no most likely token. Run it in float32."
        )
    best = scores.max(dim=-1, keepdim=True).values
    ids = torch.arange(size, device=scores.device).expand_as(scores)
    return torch.where(scores == best, ids, size).min(dim=-1).values


def greedy_continue(
    backend: ModelBackend,
    tokens: torch.Tensor,
    new: int,
    forward: Callable[[torch.Tensor], torch.Tensor] | None = None,
    cancel: threading.Event | None = None,
    on_step: Callable[[int], None] | None = None,
) -> torch.Tensor:
    """``new`` greedy tokens after each row of ``tokens`` (``[B, L]``): the whole sequence runs
    again for each one, through ``forward`` (the next-token logits ``[B, vocab]`` of a sequence;
    the model unchanged by default). ``[B, new]`` on the CPU."""
    size, _ = backend.token_space()
    run = forward or (lambda seq: backend.logits(seq, None, 1)[:, -1])
    seq = tokens.cpu()
    for _ in range(new):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        nxt = greedy_next(run(seq), size).cpu()
        seq = torch.cat([seq, nxt[:, None].to(seq.dtype)], dim=1)
        if on_step is not None:
            on_step(seq.shape[0])
    return seq[:, tokens.shape[1] :]


def _continue(
    backend: ModelBackend,
    shown: list[PreparedPrompt],
    receiver: str,
    new: int,
    batch_size: int,
    cancel: threading.Event | None,
    on_step: Callable[[int], None],
    steered: _Steered | None = None,
    variants: list[tuple[float, bool]] | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[tuple[int, int], list[int]]:
    """Greedy continuations of ``shown`` (the first held-out prompts): unsteered, keyed by
    ``(prompt, 0)``, or with ``steered`` at each variant, keyed by ``(prompt, variant)``. Each
    forward pass adds the variant's vector at the steered token of the prompt, as when it was
    measured; the tokens generated aren't steered themselves."""
    out: dict[tuple[int, int], list[int]] = {}
    for group in group_by_length(shown, receiver):
        tokens = group.tokens(receiver)
        local_of = {p: li for li, p in enumerate(group.members)}
        rows = (
            [(0, p) for p in group.members]
            if steered is None
            else [(v, p) for v in range(len(variants or [])) for p in group.members]
        )
        for sl in _chunks(len(rows), batch_size):
            chunk = rows[sl]
            forward: Callable[[torch.Tensor], torch.Tensor] | None = None
            if steered is not None:
                assert variants is not None
                patch = steered.patch(chunk, variants, dtype)

                def steer(seq: torch.Tensor, patch: Patches = patch) -> torch.Tensor:
                    return backend.logits(seq, patch, 1)[:, -1]

                forward = steer
            local = torch.tensor([local_of[p] for _, p in chunk], dtype=torch.long)
            new_ids = greedy_continue(backend, tokens[local], new, forward, cancel, on_step)
            for (v, p), ids in zip(chunk, new_ids.tolist(), strict=True):
                out[(p, v)] = [int(i) for i in ids]
    return out


def continuations_record(
    backend: ModelBackend,
    shown: list[PreparedPrompt],
    receiver: str,
    new: int,
    base: list[Site],
    variants: list[tuple[float, bool]],
    keys: list[str],
    plain: dict[tuple[int, int], list[int]],
    generated: dict[tuple[int, int, int], list[int]],
) -> dict[str, Any]:
    """What continuations.json holds (without the run's id): each continued prompt as written and
    as tokens, its unsteered continuation, and one per steered site and variant."""

    def written(ids: list[int]) -> dict[str, Any]:
        return {"ids": ids, "tokens": [backend.token_str(i) for i in ids]}

    prompts = []
    for p, prompt in enumerate(shown):
        seq = prompt.clean if receiver == "clean" else prompt.corrupt
        prompts.append(
            {
                "index": prompt.index,
                "text": prompt.record.clean if receiver == "clean" else prompt.record.corrupt,
                "prompt": list(seq.tokens),
                "unsteered": written(plain[(p, 0)]),
                "steered": [
                    {
                        "row": b,
                        "coefficient": float(c),
                        "control": control,
                        "key": keys[v],
                        **written(generated[(p, b, v)]),
                    }
                    for b in range(len(base))
                    for v, (c, control) in enumerate(variants)
                ],
            }
        )
    return {
        "receiver": receiver,
        "tokens": new,
        "decoding": "greedy",
        "rows": [{"row": b, "label": site_label(s)} for b, s in enumerate(base)],
        "prompts": prompts,
    }


# -- against the random control ----------------------------------------------------------------


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


def control_warnings(
    comparison: list[dict[str, Any]], ci: float, feature: bool = False
) -> list[str]:
    """Warn when the direction beats its random control at no site and strength."""
    compared = [c for c in comparison if c["lo"] is not None]
    if not compared or any(c["beats_control"] for c in compared):
        return []
    why = (
        "A feature's direction moves these prompts only as far as a random one of the same "
        "length: what it writes may not matter for this behavior, or may need other strengths."
        if feature
        else "Steering by a mean difference needs pairs that differ the same way: in IOI prompts "
        "that mix the ABBA and BABA orders, the differences cancel out."
    )
    return [
        f"At no site and strength does the direction move the prompts further than its random "
        f"control (the {ci:.0%} interval of the paired difference |direction| − |control| never "
        f"lies above zero), so these results show no effect of the direction. {why}"
    ]
