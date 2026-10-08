"""SAE features as sites: patch chosen features for real, or estimate every feature at once.

Patching a feature runs the receiver prompt, encodes the activation the SAE reads, and changes
only that feature, to its value in the source prompt (or to zero, for zero ablation): the
activation moves by (new - old) · the feature's decoder direction, and the SAE's error is kept as
it was. Attribution patching estimates, to first order, what patching each feature would do:
(source - receiver feature activation) · (decoder direction · the gradient of the logit
difference). It covers every feature from one gradient and keeps the strongest as sites.

Each run also records how well the SAE fits these prompts (variance explained), because features
of an SAE that doesn't fit the loaded model describe little.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from logogram.backends.base import Cancelled, ModelBackend
from logogram.engine import (
    EngineResult,
    LayerFn,
    ProgressFn,
    _answer_tensors,
    _chunks,
    _metric,
    check_gap,
    compute_baselines,
)
from logogram.prompts import PreparedPrompt, group_by_length
from logogram.sae import SAE, fit_on
from logogram.sites import ResolvedSite, ScopeError, expand_scope, resolve_position, site_label
from logogram.spec import (
    Ablation,
    ActivationPatching,
    AllPositions,
    AttributionPatching,
    FeaturesScope,
    Site,
    Spec,
    ZeroBaseline,
)


def run_features(
    spec: Spec, backend: ModelBackend, prompts: list[PreparedPrompt], sae: SAE, **kwargs: Any
) -> EngineResult:
    exp = spec.experiment
    if sae.params.d_in != backend.info.d_model:
        raise ScopeError(
            f"This SAE reads {sae.params.d_in}-dimensional activations, but the loaded model's "
            f"are {backend.info.d_model}-dimensional: it was made for another model."
        )
    if isinstance(exp, AttributionPatching):
        return _attribution(spec, backend, prompts, sae, **kwargs)
    if isinstance(exp, ActivationPatching) or (
        isinstance(exp, Ablation) and isinstance(exp.baseline, ZeroBaseline)
    ):
        if isinstance(spec.scope, FeaturesScope):
            raise ScopeError(
                "Patching every SAE feature for real would take a run per feature. Estimate them "
                "all with attribution patching, then verify the strongest."
            )
        return _patching(spec, backend, prompts, sae, **kwargs)
    raise ScopeError(
        "SAE features can be patched, zero-ablated, or estimated by attribution patching."
    )


def _directions(
    exp: ActivationPatching | Ablation | AttributionPatching,
) -> tuple[str, str, str]:
    """(receiver, source, reference) prompts for a method."""
    if isinstance(exp, ActivationPatching | AttributionPatching):
        if exp.direction == "clean_to_corrupt":
            return "corrupt", "clean", "clean"
        return "clean", "corrupt", "corrupt"
    return "clean", "", "corrupt"  # zero ablation: no source prompt


def _check_sites(sites: list[ResolvedSite], sae: SAE) -> None:
    for rs in sites:
        if rs.kind != "sae_feature":
            raise ScopeError(
                "A run measures either SAE features or model components. Put "
                f"{rs.label} in a run of its own."
            )
        if rs.layer != sae.layer:
            raise ScopeError(
                f"{rs.label} is in layer {rs.layer}, but the SAE reads layer {sae.layer}."
            )
        if (rs.site.feature or 0) >= sae.d_sae:
            raise ScopeError(f"The SAE has {sae.d_sae} features, so {rs.label} doesn't exist.")


def _fit(sae: SAE, acts: list[torch.Tensor], skip_first: bool) -> dict[str, Any]:
    """The SAE's fit on the receiver prompts' activations (without the first token when it is
    the beginning-of-sequence token, whose activations SAEs usually aren't trained on)."""
    rows = [a[:, 1:] if skip_first and a.shape[1] > 1 else a for a in acts]
    flat = torch.cat([r.reshape(-1, r.shape[-1]) for r in rows])
    return fit_on(sae, flat)


def _patching(
    spec: Spec,
    backend: ModelBackend,
    prompts: list[PreparedPrompt],
    sae: SAE,
    *,
    on_progress: ProgressFn | None = None,
    on_layer: LayerFn | None = None,
    cancel: threading.Event | None = None,
    on_start: Callable[[list[ResolvedSite], dict[str, Any]], None] | None = None,
    receiver_override: str | None = None,
    source_override: str | None = None,
) -> EngineResult:
    exp = spec.experiment
    assert isinstance(exp, ActivationPatching | Ablation)
    batch_size = spec.execution.batch_size
    sites, layout = expand_scope(spec, backend.info, prompts)
    _check_sites(sites, sae)
    if on_start is not None:
        on_start(sites, layout)
    groups = group_by_length(prompts)
    baselines = compute_baselines(backend, prompts, groups, batch_size, cancel)
    receiver, source, reference = _directions(exp)
    receiver = receiver_override or receiver
    source = source_override or source
    warnings = check_gap(spec, baselines, prompts, receiver, reference)
    key = (sae.site, sae.layer)
    features = sorted({rs.site.feature for rs in sites if rs.site.feature is not None})
    column = {f: j for j, f in enumerate(features)}
    n = len(prompts)
    patched_ld = np.zeros((len(sites), n))
    patched_prob = np.zeros((len(sites), n))
    result = EngineResult(
        sites=sites,
        layout=layout,
        prompts=prompts,
        baselines=baselines,
        receiver=receiver,
        reference=reference,
        patched_ld=patched_ld,
        patched_prob=patched_prob,
        warnings=warnings,
    )
    receiver_acts: list[torch.Tensor] = []
    total = len(sites) * n
    done = 0
    local_of = {i: li for g in groups for li, i in enumerate(g.members)}
    for group in groups:
        tokens = group.clean if receiver == "clean" else group.corrupt
        # The chosen features' activations in the source prompts, at every position.
        targets: dict[int, torch.Tensor] = {}
        if source:
            source_tokens = group.clean if source == "clean" else group.corrupt
            for sl in _chunks(len(group.members), batch_size):
                x = backend.capture(source_tokens[sl], [key])[key]
                f, _ = sae.encode(x)
                for row, p in enumerate(group.members[sl]):
                    targets[p] = f[row][:, features]  # [pos, n_features]
        for sl in _chunks(len(group.members), batch_size):
            receiver_acts.append(backend.capture(tokens[sl], [key])[key].float())
        rows = [(rs, p) for rs in sites for p in group.members]
        for sl in _chunks(len(rows), batch_size):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            chunk = rows[sl]
            local = torch.tensor([local_of[p] for _, p in chunk], dtype=torch.long)

            def edit(
                x: torch.Tensor,
                chunk: list[tuple[ResolvedSite, int]] = chunk,
                targets: dict[int, torch.Tensor] = targets,
            ) -> torch.Tensor:
                f, stats = sae.encode(x)
                out = x.float().clone()
                for b, (rs, p) in enumerate(chunk):
                    i = int(rs.site.feature or 0)
                    now = f[b, :, i]  # [pos]
                    if source:
                        new = targets[p][:, column[i]].to(now.device)
                    else:
                        new = torch.zeros_like(now)
                    change = new - now
                    if not isinstance(rs.site.position, AllPositions):
                        at = resolve_position(rs.site.position, prompts[p])
                        keep = torch.zeros_like(change)
                        keep[at] = change[at]
                        change = keep
                    row_stats = None if stats is None else (stats[0][b], stats[1][b])
                    out[b] = out[b] + change[:, None] * sae.feature_direction(i, row_stats)
                return out

            logits = backend.edit_logits(tokens[local], sae.site, sae.layer, edit)
            ld, prob = _metric(logits, *_answer_tensors(prompts, [p for _, p in chunk]))
            for (rs, p), value_ld, value_prob in zip(chunk, ld, prob, strict=True):
                patched_ld[rs.index, p] = value_ld
                patched_prob[rs.index, p] = value_prob
            done += len(chunk)
            if on_progress is not None:
                on_progress(done, total, sae.layer)
    result.extra["features"] = {
        "sae": sae.describe() | {"fit": None},
        "fit": _fit(sae, receiver_acts, spec.tokenization.prepend_bos),
    }
    if on_layer is not None:
        on_layer(sae.layer, [rs.index for rs in sites], result)
    return result


def _attribution(
    spec: Spec,
    backend: ModelBackend,
    prompts: list[PreparedPrompt],
    sae: SAE,
    *,
    on_progress: ProgressFn | None = None,
    on_layer: LayerFn | None = None,
    cancel: threading.Event | None = None,
    on_start: Callable[[list[ResolvedSite], dict[str, Any]], None] | None = None,
    receiver_override: str | None = None,
    source_override: str | None = None,
) -> EngineResult:
    exp = spec.experiment
    assert isinstance(exp, AttributionPatching)
    batch_size = spec.execution.batch_size
    scope = spec.scope
    every = isinstance(scope, FeaturesScope)
    chosen: list[ResolvedSite] = []
    layout: dict[str, Any] = {}
    if not every:
        chosen, layout = expand_scope(spec, backend.info, prompts)
        _check_sites(chosen, sae)
        if on_start is not None:
            on_start(chosen, layout)
    position = scope.position if isinstance(scope, FeaturesScope) else None
    if position is not None and not isinstance(position, AllPositions):
        for prompt in prompts:
            resolve_position(position, prompt)
    groups = group_by_length(prompts)
    baselines = compute_baselines(backend, prompts, groups, batch_size, cancel)
    receiver, source, reference = _directions(exp)
    receiver = receiver_override or receiver
    source = source_override or source
    warnings = check_gap(spec, baselines, prompts, receiver, reference)
    key = (sae.site, sae.layer)
    n, d_sae = len(prompts), sae.d_sae
    W_dec = sae.params.W_dec
    # Every feature's estimate for every prompt (or, for chosen sites, those features' at every
    # position), and the whole site's estimate, to see how much of it the features account for.
    estimates = np.zeros((n, d_sae)) if every else np.zeros((len(chosen), n))
    site_total = np.zeros(n)
    receiver_acts: list[torch.Tensor] = []
    done = 0
    for group in groups:
        receiver_tokens = group.clean if receiver == "clean" else group.corrupt
        source_tokens = group.clean if source == "clean" else group.corrupt
        for sl in _chunks(len(group.members), batch_size):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            idx = group.members[sl]
            x_src = backend.capture(source_tokens[sl], [key])[key].float()
            acts, grads = backend.gradients(
                receiver_tokens[sl], *_answer_tensors(prompts, idx), [key]
            )
            x_rec, grad = acts[key].float(), grads[key].float()
            receiver_acts.append(x_rec)
            positions = range(x_rec.shape[1])
            per_position = ((x_src - x_rec) * grad).sum(-1)  # [B, pos]
            if every:
                rows = torch.zeros(len(idx), d_sae, dtype=torch.float64, device=x_rec.device)
                for pos in positions:
                    if position is not None and not isinstance(position, AllPositions):
                        at = torch.tensor([resolve_position(position, prompts[p]) for p in idx])
                        mask = (at == pos).to(x_rec.device)
                        if not bool(mask.any()):
                            continue
                    else:
                        mask = None
                    f_src, _ = sae.encode(x_src[:, pos])
                    f_rec, stats = sae.encode(x_rec[:, pos])
                    g = grad[:, pos]
                    if stats is not None:
                        g = g * stats[1]
                    term = ((f_src - f_rec) * (g @ W_dec.T)).double()  # [B, d_sae]
                    if mask is not None:
                        term = term * mask[:, None]
                    rows += term
                estimates[idx] = rows.cpu().numpy()
                if position is None or isinstance(position, AllPositions):
                    site_total[idx] = per_position.sum(1).double().cpu().numpy()
                else:
                    at = [resolve_position(position, prompts[p]) for p in idx]
                    site_total[idx] = (
                        per_position[torch.arange(len(idx)), at].double().cpu().numpy()
                    )
            else:
                f_src, _ = sae.encode(x_src)
                f_rec, stats = sae.encode(x_rec)
                g = grad if stats is None else grad * stats[1]
                for rs in chosen:
                    i = int(rs.site.feature or 0)
                    term = (f_src[..., i] - f_rec[..., i]) * (g @ W_dec[i])  # [B, pos]
                    if isinstance(rs.site.position, AllPositions):
                        values = term.sum(1)
                    else:
                        at = [resolve_position(rs.site.position, prompts[p]) for p in idx]
                        values = term[torch.arange(len(idx)), at]
                    estimates[rs.index, idx] = values.double().cpu().numpy()
                site_total[idx] = per_position.sum(1).double().cpu().numpy()
            done += len(idx)
            if on_progress is not None:
                on_progress(done, n, sae.layer)

    if every:
        assert position is not None
        means = estimates.mean(0)
        order = sorted(range(d_sae), key=lambda i: (-abs(means[i]), i))[: scope.top]  # type: ignore[union-attr]
        sites = []
        for j, feature in enumerate(order):
            site = Site(kind="sae_feature", layer=sae.layer, feature=feature, position=position)
            sites.append(ResolvedSite(j, site, j, 0, site_label(site)))
        layout = {
            "kind": "sites",
            "row_title": "Feature",
            "col_title": "",
            "rows": [{"key": str(j), "label": rs.label} for j, rs in enumerate(sites)],
            "cols": [{"key": "effect", "label": "effect"}],
        }
        if on_start is not None:
            on_start(sites, layout)
        delta = estimates[:, order].T.copy()
        features_sum = estimates.sum(1)
    else:
        sites = chosen
        delta = estimates
        features_sum = None
    receiver_ld = baselines.ld(receiver)
    result = EngineResult(
        sites=sites,
        layout=layout,
        prompts=prompts,
        baselines=baselines,
        receiver=receiver,
        reference=reference,
        patched_ld=receiver_ld[None, :] + delta,
        patched_prob=np.full(delta.shape, np.nan),
        warnings=warnings,
        measure="estimate",
        delta=delta,
        extra={
            "features": {
                "sae": sae.describe() | {"fit": None},
                "fit": _fit(sae, receiver_acts, spec.tokenization.prepend_bos),
                "site_estimate": float(site_total.mean()),
                "features_estimate": None if features_sum is None else float(features_sum.mean()),
                "evaluated": d_sae if every else len(chosen),
            }
        },
    )
    if on_layer is not None:
        on_layer(sae.layer, [rs.index for rs in sites], result)
    return result
