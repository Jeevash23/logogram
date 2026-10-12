"""Attribution patching: estimate activation patching at every site from gradients.

Patching a site changes the metric by f(receiver with the source activation) - f(receiver). To
first order that is (source activation - receiver activation) · the gradient of the metric with
respect to the activation, at the receiver run. One forward pass on the source prompts and one
forward and backward pass on the receiver prompts give the estimate for every site at once, so a
sweep costs a few passes per batch instead of one patched run per site and prompt.

The estimate is only first order. It misses saturation (in attention patterns, normalization and
the final softmax) and can miss or even invert an effect, which is why the results say "estimated"
and offer to verify the strongest sites with real patching.

Integrated gradients (as in EAP-IG) average the gradient over runs whose input embeddings lie
evenly between the receiver's and the source's, at the midpoints of ``steps`` equal intervals. The
average follows the metric along the way from one prompt to the other instead of only at its
start, which corrects much of what a single gradient misses, at ``steps`` times the cost.

Pairs whose prompts have different token lengths are compared at a label or the last token: the
source prompt's activation at its own token there against the receiver's at its own. Integrated
gradients interpolate the two prompts position by position, so they need pairs of equal length.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from logogram.backends.base import Cancelled, ModelBackend, float64
from logogram.engine import (
    Baselines,
    EngineResult,
    LayerFn,
    ProgressFn,
    PromptsFn,
    _chunks,
    _LayerSources,
    check_finite,
    check_gap,
    check_values,
    compute_baselines,
    make_scorer,
)
from logogram.metrics import Scorer, extend_tokens
from logogram.prompts import LengthGroup, PreparedPrompt, group_by_length, uneven_text
from logogram.sites import (
    ResolvedSite,
    ScopeError,
    check_lined_up,
    expand_scope,
    resolve_position,
    site_unit,
)
from logogram.spec import AllPositions, AttributionPatching, Spec

RESIDUAL = ("resid_pre", "resid_mid", "resid_post")

Grads = dict[tuple[str, int], torch.Tensor]


def metric_gradients(
    backend: ModelBackend,
    scorer: Scorer,
    tokens: torch.Tensor,
    rows: list[int],
    sites: list[tuple[str, int]],
    embeddings: torch.Tensor | None = None,
) -> tuple[Grads, Grads]:
    """Activations and the metric's gradients at ``sites``, over the prompts' positions only.

    Answers or distractors of several tokens need a forward pass with each continuation
    appended; the metric is a sum of what each pass contributes, so their gradients add up. The
    prompt's activations are the same in every pass (a causal model never reads ahead)."""
    length = tokens.shape[1]
    acts: Grads | None = None
    total: Grads | None = None
    for part, conts, keep in scorer.passes(rows):

        def score(logits: torch.Tensor, part: str = part) -> torch.Tensor:
            return scorer.differentiable(logits, rows, part)

        kept, grads = backend.gradients(
            extend_tokens(tokens, conts, keep), sites, score, keep, embeddings
        )
        if acts is None:
            acts = {k: v[:, :length] for k, v in kept.items()}
        sliced = {k: v[:, :length] for k, v in grads.items()}
        total = sliced if total is None else {k: total[k] + sliced[k] for k in total}
    assert acts is not None and total is not None
    return acts, total


def pass_products(
    backend: ModelBackend,
    scorer: Scorer,
    receiver_tokens: torch.Tensor,
    source_tokens: torch.Tensor,
    rows: list[int],
    sites: list[tuple[str, int]],
    steps: int | None,
) -> tuple[dict[tuple[str, int], torch.Tensor], dict[tuple[str, int], torch.Tensor] | None]:
    """(source - receiver activation) · gradient at every position of ``sites``: at the prompt's
    positions ``[B, L]`` (``[B, L, heads]`` for heads' outputs, queries, keys and values, and
    ``[B, L, d_mlp]`` for neurons, each of which is one number), and summed over the positions
    of tokens appended to read continuations (None when nothing was appended).

    Each forward pass the metric needs (one per continuation) contributes its own products: the
    source prompt runs with the same continuation appended, so the appended positions are compared
    like for like. With ``steps``, the gradient is averaged over runs whose input embeddings lie
    between the receiver's and the source's (integrated gradients)."""
    length = receiver_tokens.shape[1]
    prompt: dict[tuple[str, int], torch.Tensor] = {}
    tail: dict[tuple[str, int], torch.Tensor] | None = None
    for part, conts, keep in scorer.passes(rows):

        def score(logits: torch.Tensor, part: str = part) -> torch.Tensor:
            return scorer.differentiable(logits, rows, part)

        receiver_ext = extend_tokens(receiver_tokens, conts, keep)
        source_ext = extend_tokens(source_tokens, conts, keep)
        source_acts = backend.capture(source_ext, sites)
        if steps is None:
            receiver_acts, grads = backend.gradients(receiver_ext, sites, score, keep)
        else:
            receiver_acts = backend.capture(receiver_ext, sites)
            grads = _integrated(backend, receiver_ext, source_ext, sites, score, keep, steps)
        for key in sites:
            product = (float64(source_acts[key]) - float64(receiver_acts[key])) * float64(
                grads[key]
            )
            if key[0] != "neuron":  # a neuron is one number: nothing to sum over
                product = product.sum(-1)  # [B, pos], or [B, pos, heads]
            here = product[:, :length]
            prompt[key] = here if key not in prompt else prompt[key] + here
            if product.shape[1] > length:
                rest = product[:, length:].sum(1)
                tail = {} if tail is None else tail
                tail[key] = rest if key not in tail else tail[key] + rest
    return prompt, tail


def _integrated(
    backend: ModelBackend,
    receiver_tokens: torch.Tensor,
    source_tokens: torch.Tensor,
    sites: list[tuple[str, int]],
    score: Callable[[torch.Tensor], torch.Tensor],
    keep: int,
    steps: int,
) -> dict[tuple[str, int], torch.Tensor]:
    """The gradients at ``sites``, averaged over ``steps`` runs whose input embeddings lie between
    the receiver's and the source's, at the midpoints of equal intervals."""
    key = ("resid_pre", 0)
    start = backend.capture(receiver_tokens, [key])[key]
    end = backend.capture(source_tokens, [key])[key]
    total: dict[tuple[str, int], torch.Tensor] | None = None
    for k in range(steps):
        alpha = (k + 0.5) / steps
        _, grads = backend.gradients(
            receiver_tokens, sites, score, keep, start + alpha * (end - start)
        )
        grads = {s: float64(g) for s, g in grads.items()}
        total = grads if total is None else {s: total[s] + grads[s] for s in total}
    assert total is not None
    return {s: g / steps for s, g in total.items()}


def check_integrated(exp: AttributionPatching, prompts: list[PreparedPrompt]) -> None:
    """Refuse integrated gradients on pairs of different lengths: they run on inputs that lie
    between the two prompts' embeddings position by position, which needs the positions to line
    up."""
    if exp.method != "integrated_gradients":
        return
    uneven = [p.index for p in prompts if not p.even]
    if uneven:
        raise ScopeError(
            f"{uneven_text(uneven, len(prompts))}. Integrated gradients run on inputs between the "
            "two prompts' embeddings, position by position, which needs their positions to line "
            "up. Estimate with one gradient, which compares the prompts at a label or the last "
            "token, or use pairs of equal length."
        )


def source_tables(
    backend: ModelBackend,
    prompts: list[PreparedPrompt],
    groups: list[LengthGroup],
    sites: list[ResolvedSite],
    source: str,
    batch_size: int,
    cancel: threading.Event | None,
) -> dict[tuple[tuple[str, int], str], torch.Tensor]:
    """Each source prompt's activation at each site's token, ``[n, ...]`` by (kind, layer) and
    position, captured a layer at a time in batches of the source prompts' own length."""
    tables: dict[tuple[tuple[str, int], str], torch.Tensor] = {}
    for layer in sorted({rs.layer for rs in sites}):
        here = [rs for rs in sites if rs.layer == layer]
        kinds = list(dict.fromkeys(rs.kind for rs in here))
        sources = _LayerSources(backend, prompts, groups, layer, kinds, source, batch_size, cancel)
        for rs in here:
            key = ((rs.kind, layer), rs.site.position.model_dump_json())
            if key not in tables:
                tables[key] = sources.table(rs.kind, rs)
    return tables


def _batch_products(
    backend: ModelBackend,
    scorer: Scorer,
    receiver_tokens: torch.Tensor,
    source_tokens: torch.Tensor,
    idx: list[int],
    prompts: list[PreparedPrompt],
    sites: list[ResolvedSite],
    needed: list[tuple[str, int]],
    steps: int | None,
    receiver: str,
) -> np.ndarray:
    """Every site's estimate ``[S, B]`` for a batch of pairs whose positions line up: the products
    at every position of each layer once, then each site reads its own part of them."""
    products, tail = pass_products(
        backend, scorer, receiver_tokens, source_tokens, idx, needed, steps
    )
    rows = torch.arange(len(idx))
    out = np.zeros((len(sites), len(idx)))
    for rs in sites:
        key = (rs.kind, rs.layer)
        per_position = products[key]
        unit = site_unit(rs.site)
        if unit is not None:  # one head (key/value head) or neuron of the activation
            per_position = per_position[:, :, unit]
        if isinstance(rs.site.position, AllPositions):
            values = per_position.sum(1)
            if tail is not None:  # every position includes the appended ones
                values = values + (tail[key][:, unit] if unit is not None else tail[key])
        else:
            positions = torch.tensor(
                [resolve_position(rs.site.position, prompts[p], receiver) for p in idx]
            )
            values = per_position[rows, positions]
        out[rs.index] = values.cpu().numpy()
    return out


def _batch_at_tokens(
    backend: ModelBackend,
    scorer: Scorer,
    receiver_tokens: torch.Tensor,
    idx: list[int],
    prompts: list[PreparedPrompt],
    sites: list[ResolvedSite],
    needed: list[tuple[str, int]],
    tables: dict[tuple[tuple[str, int], str], torch.Tensor],
    receiver: str,
) -> np.ndarray:
    """Every site's estimate ``[S, B]`` for a batch of receivers whose pairs may differ in length:
    (the source's activation at the site's token - the receiver's at its own) · the gradient
    there, with ``tables`` holding the sources'."""
    acts, grads = metric_gradients(backend, scorer, receiver_tokens, idx, needed)
    rows = torch.arange(len(idx))
    out = np.zeros((len(sites), len(idx)))
    for rs in sites:
        key = (rs.kind, rs.layer)
        at = torch.tensor([resolve_position(rs.site.position, prompts[p], receiver) for p in idx])
        src = tables[(key, rs.site.position.model_dump_json())][idx]
        own, grad = acts[key][rows, at], grads[key][rows, at]
        unit = site_unit(rs.site)
        if unit is not None:  # one head (key/value head) or neuron of the activation
            src, own, grad = src[:, unit], own[:, unit], grad[:, unit]
        own, grad = own.to(src.device), grad.to(src.device)
        product = (float64(src) - float64(own)) * float64(grad)
        if product.dim() > 1:  # a neuron is one number: nothing to sum over
            product = product.sum(-1)
        out[rs.index] = product.cpu().numpy()
    return out


def run_attribution_patching(
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
    on_prompts: PromptsFn | None = None,
) -> EngineResult:
    """Estimate every site at once, batch by batch of prompts. Each finished batch is reported
    through ``on_prompts``, so a run that stops early keeps every site's estimate over the prompts
    that finished."""
    exp = spec.experiment
    assert isinstance(exp, AttributionPatching)
    info = backend.info
    batch_size = spec.execution.batch_size
    sites, layout = expand_scope(spec, info, prompts)
    receiver, source = (
        ("corrupt", "clean") if exp.direction == "clean_to_corrupt" else ("clean", "corrupt")
    )
    reference = source
    receiver = receiver_override or receiver
    source = source_override or source
    # Pairs of different lengths are compared at a label or the last token, one token each.
    lined_up = source == receiver or all(p.even for p in prompts)
    if not lined_up:
        check_integrated(exp, prompts)
        check_lined_up([rs.site.position for rs in sites], prompts, "estimate")
    if on_start is not None:
        on_start(sites, layout)
    groups = group_by_length(prompts, receiver)
    scorer = make_scorer(spec, prompts)
    baselines = compute_baselines(backend, prompts, groups, batch_size, cancel, scorer)
    check_finite(baselines, info.dtype)
    warnings = check_gap(spec, baselines, prompts, receiver, reference)

    n = len(prompts)
    needed = list(dict.fromkeys((rs.kind, rs.layer) for rs in sites))
    estimate = np.zeros((len(sites), n))
    done = 0
    finished: list[int] = []

    def over(at: list[int]) -> EngineResult:
        """Every site's estimates over the prompts at positions ``at``, as a run of only those
        prompts gives them: their own gap normalizes them, and is checked as a run's is."""
        chosen = [prompts[i] for i in at]
        base = baselines.subset(at)
        notes = check_gap(spec, base, chosen, receiver, reference)
        return _estimates(
            exp,
            sites,
            layout,
            chosen,
            base,
            receiver,
            reference,
            estimate[:, at],
            notes,
            info.dtype,
        )

    # Pairs of different lengths: the source prompts' activations at each site's token, captured
    # in batches of their own length, to compare with the receivers' at theirs.
    tables = (
        None
        if lined_up
        else source_tables(backend, prompts, groups, sites, source, batch_size, cancel)
    )
    steps = exp.steps if exp.method == "integrated_gradients" else None
    for group in groups:
        receiver_tokens = group.tokens(receiver)
        for sl in _chunks(len(group.members), batch_size):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            idx = group.members[sl]
            if tables is None:
                source_tokens = group.tokens(source)
                values = _batch_products(
                    backend, scorer, receiver_tokens[sl], source_tokens[sl], idx, prompts, sites,
                    needed, steps, receiver,
                )  # fmt: skip
            else:
                values = _batch_at_tokens(
                    backend, scorer, receiver_tokens[sl], idx, prompts, sites, needed, tables,
                    receiver,
                )  # fmt: skip
            estimate[:, idx] = values
            done += len(idx)
            finished.extend(idx)
            if on_progress is not None:
                on_progress(done, n, info.n_layers - 1)
            if on_prompts is not None:
                at = sorted(finished)
                on_prompts(at, lambda at=at: over(at))

    result = _estimates(
        exp, sites, layout, prompts, baselines, receiver, reference, estimate, warnings, info.dtype
    )
    if on_layer is not None:
        for layer in sorted({rs.layer for rs in sites}):
            on_layer(layer, [rs.index for rs in sites if rs.layer == layer], result)
    return result


def _estimates(
    exp: AttributionPatching,
    sites: list[ResolvedSite],
    layout: dict[str, Any],
    prompts: list[PreparedPrompt],
    baselines: Baselines,
    receiver: str,
    reference: str,
    estimate: np.ndarray,
    warnings: list[str],
    dtype: str,
) -> EngineResult:
    """The result of estimating ``sites`` on ``prompts``: ``estimate`` is [sites, prompts]."""
    if any(rs.kind in RESIDUAL for rs in sites):
        warnings.append(
            "Estimates at residual stream sites can miss effects where patching replaces a whole "
            "token's representation: in GPT-2 small's IOI example, patching the residual stream at "
            "the changed name in the first layer restores the whole answer, while its estimate is "
            "slightly negative. Verifying the top estimates can't catch such a miss; Check "
            "robustness patches every site."
        )
    warnings.extend(check_values(estimate, "estimates", dtype))
    nan = np.full(estimate.shape, np.nan)
    return EngineResult(
        sites=sites,
        layout=layout,
        prompts=prompts,
        baselines=baselines,
        receiver=receiver,
        reference=reference,
        # The metric patching would give, to first order. Sign flips and probabilities aren't
        # estimated: they are left undefined rather than read off an estimate.
        patched=baselines.metric(receiver)[None, :] + estimate,
        patched_prob=nan,
        patched_pref=nan.copy(),
        warnings=warnings,
        measure="estimate",
        delta=estimate,
        extra={"attribution": {"method": exp.method, "steps": exp.steps}},
    )
