"""Distributed alignment search: a learned subspace whose patching carries an effect
(``spec.DistributedAlignmentSearch``, after Geiger et al. 2023).

At each site of the scope (one stream site at one position of each prompt), an orthonormal basis Q
of ``dimensions`` vectors is learned so that patching only the component within it, from the
source prompt into the receiver run,

    receiver activation + ((source activation − receiver activation) · Q) · Qᵀ,

moves the metric as far toward the source prompt's as it can: Adam maximizes the mean normalized
effect over the training pairs, each step over all of them in their batches. Q is the orthonormal
part of a free matrix (its QR decomposition, taken in float64 on the CPU), which starts from a
seeded random draw.

The held-out pairs then measure three patches at each site: within the learned subspace, of the
whole site (activation patching, the most any subspace can carry), and, with ``control``, within a
random subspace of the same size. A learned subspace can carry an effect the model doesn't use as
a variable, by finding a direction that steers the computation from outside it (Makelov et al.
2023): compare it with the whole site, check it on held-out prompts (as here) and on other
prompts, and test what it means with other methods. Each site's learned basis is kept in the run's
vectors file, to save as a direction.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from logogram.backends.base import STREAM_KINDS, Cancelled, ModelBackend, Patch, float64
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
from logogram.metrics import Scorer, extend_tokens
from logogram.prompts import PreparedPrompt, group_by_length, seeded_split
from logogram.sites import (
    ResolvedSite,
    ScopeError,
    check_lined_up,
    expand_scope,
    resolve_position,
    site_label,
)
from logogram.spec import AllPositions, DistributedAlignmentSearch, Spec
from logogram.steering import DTYPES

VARIANTS = (("learned", "Learned subspace"), ("full", "Whole site"), ("random", "Random subspace"))


def orthonormal_basis(free: torch.Tensor) -> torch.Tensor:
    """The orthonormal basis ``[d, k]`` spanned by ``free``'s columns: its QR decomposition, in
    float64 on the CPU (the same on every device), with each column's sign fixed by R's diagonal
    so the basis changes smoothly with ``free``."""
    q, r = torch.linalg.qr(free.to(device="cpu", dtype=torch.float64))  # float64 on the CPU
    return q * torch.sign(torch.diagonal(r))[None, :]


def _draw(d: int, k: int, seed: int) -> torch.Tensor:
    """A seeded standard normal ``[d, k]`` in float64, from the raw PCG64 stream (the same on every
    machine and NumPy version) by the Box-Muller transform."""
    count = d * k
    raw = np.random.PCG64(seed).random_raw(2 * count)
    # 53 random bits each, in (0, 1]: the logarithm never meets zero.
    u = ((raw >> np.uint64(11)).astype(np.float64) + 1.0) / 2.0**53
    normal = np.sqrt(-2.0 * np.log(u[:count])) * np.cos(2.0 * np.pi * u[count:])
    return torch.from_numpy(normal.reshape(d, k))


def run_das(
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
    assert isinstance(exp, DistributedAlignmentSearch)
    info = backend.info
    batch_size = spec.execution.batch_size
    base_sites, _grid = expand_scope(spec, info, prompts)
    for rs in base_sites:
        if isinstance(rs.site.position, AllPositions):
            raise ScopeError(
                "Distributed alignment search learns a subspace at one position of each prompt: "
                "choose the last token, a token index or a labelled position."
            )
        if rs.kind not in STREAM_KINDS:
            raise ScopeError(
                f"Distributed alignment search learns a subspace of activations as wide as the "
                f"residual stream (stream sites, attention and MLP outputs); {rs.label} isn't one."
            )
        if exp.dimensions >= info.d_model:
            raise EngineError(
                f"A subspace of {exp.dimensions} dimensions is the whole {info.d_model}-wide "
                "activation: choose fewer dimensions."
            )
    receiver, source = (
        ("corrupt", "clean") if exp.direction == "clean_to_corrupt" else ("clean", "corrupt")
    )
    reference = source
    check_lined_up([rs.site.position for rs in base_sites], prompts, "das")
    train_at, test_at = seeded_split(len(prompts), exp.train_fraction, exp.seed)
    if len(train_at) < 1 or len(test_at) < 2:
        raise EngineError(
            f"Distributed alignment search needs at least one prompt pair to learn from and two "
            f"to measure on, but {len(prompts)} pairs with a training share of "
            f"{exp.train_fraction:.0%} leaves {len(train_at)} and {len(test_at)}. Use more "
            "prompts or another training share."
        )
    train = [prompts[i] for i in train_at]
    test = [prompts[i] for i in test_at]

    variants = [(key, label) for key, label in VARIANTS if key != "random" or exp.control]
    base = [rs.site for rs in base_sites]
    uniform = len({s.layer for s in base}) == len(base) and (
        len({(s.kind, s.position.model_dump_json()) for s in base}) == 1
    )
    sites: list[ResolvedSite] = []
    for b, site in enumerate(base):
        for v, (key, _label) in enumerate(variants):
            sites.append(
                ResolvedSite(
                    index=len(sites),
                    site=site,
                    row=b,
                    col=v,
                    label=f"{site_label(site)} {key}",
                    variant={"subspace": key},
                    variant_key=key,
                )
            )
    layout = {
        "kind": "das",
        "site": base[0].kind if uniform else None,
        "row_title": "Layer" if uniform else "Site",
        "col_title": "Patched",
        "rows": [
            {"key": str(b), "label": str(s.layer) if uniform else site_label(s)}
            for b, s in enumerate(base)
        ],
        "cols": [{"key": key, "label": label} for key, label in variants],
    }
    if on_start is not None:
        on_start(sites, layout)

    train_scorer = make_scorer(spec, train)
    test_scorer = make_scorer(spec, test)
    train_groups = group_by_length(train, receiver)
    test_groups = group_by_length(test, receiver)
    train_baselines = compute_baselines(
        backend, train, train_groups, batch_size, cancel, train_scorer
    )
    baselines = compute_baselines(backend, test, test_groups, batch_size, cancel, test_scorer)
    check_finite(baselines, info.dtype)
    warnings = check_gap(spec, baselines, test, receiver, reference)
    train_gap = train_baselines.metric(reference) - train_baselines.metric(receiver)
    if spec.normalization == "dataset_gap":
        denominator = np.full(len(train), float(train_gap.mean()))
    else:
        denominator = train_gap
    if not np.isfinite(denominator).all() or (np.abs(denominator) < 1e-6).any():
        raise EngineError(
            "The training pairs' clean–corrupt gap is zero or not a finite number, so the effect "
            "the subspace is trained to carry is undefined. Use more prompts, or check the "
            "baseline."
        )

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
    )
    dtype = DTYPES[info.dtype]
    groups = group_by_length(prompts, receiver)
    learned: list[dict[str, Any]] = []
    vectors: dict[str, torch.Tensor] = {}
    total = len(base) * (exp.steps * len(train) + len(variants) * n)
    done = 0

    def advance(rows: int, layer: int) -> None:
        nonlocal done
        done += rows
        if on_progress is not None:
            on_progress(done, total, layer)

    for layer in sorted({s.layer for s in base}):
        at_layer = [b for b, s in enumerate(base) if s.layer == layer]
        kinds = list(dict.fromkeys(base[b].kind for b in at_layer))
        sources = _LayerSources(backend, prompts, groups, layer, kinds, source, batch_size, cancel)
        owns = _LayerSources(backend, prompts, groups, layer, kinds, receiver, batch_size, cancel)
        finished: list[int] = []
        for b in at_layer:
            rs = base_sites[b]
            src = sources.table(rs.kind, rs)  # [n, d] by position in ``prompts``
            own = owns.table(rs.kind, rs)
            free = _draw(info.d_model, exp.dimensions, _site_seed(exp.seed, b, 0)).requires_grad_(
                True
            )
            optimizer = torch.optim.Adam([free], lr=exp.learning_rate)
            curve: list[float] = []
            for _step in range(exp.steps):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                optimizer.zero_grad()
                gradient = torch.zeros_like(free)
                total_effect = 0.0
                for group in train_groups:
                    tokens = group.tokens(receiver)
                    for sl in _chunks(len(group.members), batch_size):
                        rows = group.members[sl]
                        at = [train_at[r] for r in rows]
                        basis = orthonormal_basis(free).to(
                            device=backend.device, dtype=torch.float32
                        )
                        effect = _train_effect(
                            backend,
                            train_scorer,
                            tokens[sl],
                            rows,
                            rs,
                            [train[r] for r in rows],
                            receiver,
                            own[at].float(),
                            src[at].float(),
                            basis,
                            dtype,
                            train_baselines.metric(receiver)[rows],
                            denominator[rows],
                        )
                        loss = -effect.sum() / len(train)
                        (grad,) = torch.autograd.grad(loss, [free])
                        gradient += grad
                        total_effect += float(effect.detach().sum())
                        advance(len(rows), layer)
                free.grad = gradient
                optimizer.step()
                curve.append(total_effect / len(train))
            bases = {
                "learned": orthonormal_basis(free.detach()),
                "random": orthonormal_basis(
                    _draw(info.d_model, exp.dimensions, _site_seed(exp.seed, b, 1))
                ),
            }
            for v, (key, _label) in enumerate(variants):
                index = b * len(variants) + v
                basis = (
                    None
                    if key == "full"
                    else bases[key].to(device=backend.device, dtype=torch.float32)
                )
                for group in test_groups:
                    tokens = group.tokens(receiver)
                    for sl in _chunks(len(group.members), batch_size):
                        if cancel is not None and cancel.is_set():
                            raise Cancelled()
                        rows = group.members[sl]
                        at = [test_at[r] for r in rows]
                        values = _patched_values(own[at].float(), src[at].float(), basis)
                        patch = _patch(rs, [test[r] for r in rows], receiver, values, dtype)
                        scores = test_scorer.score(
                            patched_forward(backend, patch), tokens[sl], rows, receiver
                        )
                        patched[index, rows] = scores.metric
                        patched_prob[index, rows] = scores.prob
                        patched_pref[index, rows] = scores.pref
                        advance(len(rows), layer)
                finished.append(index)
            index = b * len(variants)  # the learned subspace's site
            vectors[f"site{index}"] = bases["learned"].T.to(torch.float32)
            learned.append(
                {
                    "index": index,
                    "row": b,
                    "curve": curve,
                    "train_effect": curve[-1] if curve else None,
                }
            )
        if on_layer is not None:
            on_layer(layer, finished, result)
    warnings.extend(check_values(patched, "patched values", info.dtype))
    result.extra["vectors"] = vectors
    result.extra["das"] = {
        "dimensions": exp.dimensions,
        "steps": exp.steps,
        "learning_rate": exp.learning_rate,
        "train": [p.index for p in train],
        "test": [p.index for p in test],
        "sites": learned,
    }
    return result


def _site_seed(seed: int, row: int, stream: int) -> int:
    """A seed for one site's draw (``stream`` 0 starts the learned basis, 1 is the control), as
    a PCG64 seed: a different stream for each site, the same on every machine."""
    return (seed * 1_000_003 + row * 2 + stream) % 2**63


def _patched_values(
    own: torch.Tensor, source: torch.Tensor, basis: torch.Tensor | None
) -> torch.Tensor:
    """``own`` with its component within ``basis``'s span (its columns, orthonormal) taken from
    ``source``; with no basis, ``source`` itself (patching the whole site)."""
    if basis is None:
        return source
    return own + ((source - own) @ basis) @ basis.T


def _patch(
    rs: ResolvedSite,
    rows: list[PreparedPrompt],
    receiver: str,
    values: torch.Tensor,
    dtype: torch.dtype,
) -> Patch:
    positions = torch.tensor(
        [resolve_position(rs.site.position, p, receiver) for p in rows], dtype=torch.long
    )
    return Patch(kind=rs.kind, layer=rs.layer, values=values.to(dtype), positions=positions)


def _train_effect(
    backend: ModelBackend,
    scorer: Scorer,
    tokens: torch.Tensor,
    rows: list[int],
    rs: ResolvedSite,
    prompts: list[PreparedPrompt],
    receiver: str,
    own: torch.Tensor,
    source: torch.Tensor,
    basis: torch.Tensor,
    dtype: torch.dtype,
    receiver_metric: np.ndarray,
    denominator: np.ndarray,
) -> torch.Tensor:
    """Each training row's normalized effect of patching within ``basis``, with gradients."""
    values = _patched_values(own, source, basis)
    patch = _patch(rs, prompts, receiver, values, dtype)
    metric = None
    for part, conts, keep in scorer.passes(rows):
        logits = backend.logits(extend_tokens(tokens, conts, keep), patch, keep, grad=True)
        term = float64(scorer.differentiable(logits, rows, part))
        metric = term if metric is None else metric + term
    assert metric is not None
    base = torch.from_numpy(np.asarray(receiver_metric, dtype=np.float64)).to(metric.device)
    gap = torch.from_numpy(np.asarray(denominator, dtype=np.float64)).to(metric.device)
    return (metric - base) / gap
