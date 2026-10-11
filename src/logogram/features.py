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

from logogram.atp import metric_gradients
from logogram.backends.base import Cancelled, ModelBackend, float64
from logogram.engine import (
    EngineResult,
    LayerFn,
    ProgressFn,
    _chunks,
    check_finite,
    check_gap,
    check_values,
    compute_baselines,
    empty_values,
    make_scorer,
)
from logogram.prompts import PreparedPrompt, group_by_length, seeded_split
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
    check_dimensions(sae, backend.info)
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


def check_dimensions(sae: SAE, info: Any) -> None:
    """Refuse an SAE made for another model: its sizes must match the sites it reads and writes."""
    heads = info.n_heads * info.d_head

    def size(site: str) -> int:
        return heads if site == "head" else info.d_model

    for what, site, d in (
        ("reads", sae.site_in, sae.params.d_in),
        ("writes", sae.site, sae.params.d_out),
    ):
        if d != size(site):
            raise ScopeError(
                f"This SAE {what} {d}-dimensional activations, but the loaded model's are "
                f"{size(site)}-dimensional there: it was made for another model."
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


def _feature_split(scope: FeaturesScope, n: int) -> tuple[list[int], list[int]]:
    """Positions of the prompts that choose the strongest features, and of those that report
    them: every prompt for both, unless the scope holds a share out."""
    if scope.choose_on is None:
        return list(range(n)), list(range(n))
    choose, report = seeded_split(n, scope.choose_on, scope.seed or 0)
    if len(choose) < 1 or len(report) < 2:
        raise ScopeError(
            f"Choosing features on {scope.choose_on:.0%} of {n} prompts leaves {len(choose)} to "
            f"choose them and {len(report)} to report them, but it needs at least one and two. "
            "Use more prompts, or another share."
        )
    return choose, report


def _check_continuations(scorer: Any, positions: list[Any]) -> None:
    """Features are read from the prompt: a feature at every position would also need its value
    on tokens appended to read an answer of several tokens, which the source prompt lacks."""
    if not scorer.single_position() and any(isinstance(p, AllPositions) for p in positions):
        raise ScopeError(
            "With answers of several tokens, SAE features are patched or estimated at one "
            "position of the prompt (the last token, or a named position), not at every position."
        )


def _fit(
    sae: SAE,
    acts: list[torch.Tensor],
    skip_first: bool,
    targets: list[torch.Tensor] | None = None,
) -> dict[str, Any]:
    """The SAE's fit on the receiver prompts' activations (without the first token when it is
    the beginning-of-sequence token, whose activations SAEs usually aren't trained on). A
    transcoder's fit is how well it predicts its MLP's outputs (``targets``) from the inputs."""

    def stack(parts: list[torch.Tensor]) -> torch.Tensor:
        rows = [sae.reads(a) if a.dim() == 4 else a for a in parts]
        rows = [r[:, 1:] if skip_first and r.shape[1] > 1 else r for r in rows]
        return torch.cat([r.reshape(-1, r.shape[-1]) for r in rows])

    return fit_on(sae, stack(acts), stack(targets) if targets is not None else None)


def _write_direction(sae: SAE, feature: int, stats: Any, like: torch.Tensor) -> torch.Tensor:
    """What one unit of a feature adds to the activation it writes, shaped like ``like``'s last
    dimensions (heads' outputs are written side by side)."""
    direction = sae.feature_direction(feature, stats)
    if like.dim() >= 2 and sae.site == "head":
        return direction.reshape(like.shape[-2], like.shape[-1])
    return direction


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
    scorer = make_scorer(spec, prompts)
    baselines = compute_baselines(backend, prompts, groups, batch_size, cancel, scorer)
    check_finite(baselines, backend.info.dtype)
    receiver, source, reference = _directions(exp)
    receiver = receiver_override or receiver
    source = source_override or source
    warnings = check_gap(spec, baselines, prompts, receiver, reference)
    _check_continuations(scorer, [rs.site.position for rs in sites])
    key = (sae.site_in, sae.layer)  # what the features are read from
    out_key = (sae.site, sae.layer)  # what they write (the same site, but for a transcoder)
    features = sorted({rs.site.feature for rs in sites if rs.site.feature is not None})
    column = {f: j for j, f in enumerate(features)}
    n = len(prompts)
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
        warnings=warnings,
    )
    receiver_acts: list[torch.Tensor] = []
    receiver_outs: list[torch.Tensor] = []
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
                f, _ = sae.encode(sae.reads(x))
                for row, p in enumerate(group.members[sl]):
                    targets[p] = f[row][:, features]  # [pos, n_features]
        for sl in _chunks(len(group.members), batch_size):
            captured = backend.capture(tokens[sl], list(dict.fromkeys([key, out_key])))
            receiver_acts.append(captured[key].float().cpu())  # for the fit, later
            if sae.transcoder:
                receiver_outs.append(captured[out_key].float().cpu())
        rows = [(rs, p) for rs in sites for p in group.members]
        for sl in _chunks(len(rows), batch_size):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            chunk = rows[sl]
            local = torch.tensor([local_of[p] for _, p in chunk], dtype=torch.long)

            def edit(
                x: torch.Tensor,
                read: torch.Tensor | None = None,
                chunk: list[tuple[ResolvedSite, int]] = chunk,
                targets: dict[int, torch.Tensor] = targets,
                length: int = group.length,
            ) -> torch.Tensor:
                # Only the prompt's positions: tokens appended to read a continuation stay as
                # they are. A transcoder reads the MLP's input and changes its output; an SAE
                # reads and changes the same activation.
                source_x = x if read is None else read
                f, stats = sae.encode(sae.reads(source_x[:, :length]))
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
                    direction = _write_direction(sae, i, row_stats, out[b, 0])
                    shape = (-1,) + (1,) * direction.dim()
                    out[b, :length] = out[b, :length] + change.reshape(shape) * direction
                return out

            def forward(toks: torch.Tensor, keep: int, edit: Any = edit) -> torch.Tensor:
                read = sae.site_in if sae.transcoder else None
                return backend.edit_logits(toks, sae.site, sae.layer, edit, keep, read)

            scores = scorer.score(forward, tokens[local], [p for _, p in chunk])
            for (rs, p), value, prob, pref in zip(
                chunk, scores.metric, scores.prob, scores.pref, strict=True
            ):
                patched[rs.index, p] = value
                patched_prob[rs.index, p] = prob
                patched_pref[rs.index, p] = pref
            done += len(chunk)
            if on_progress is not None:
                on_progress(done, total, sae.layer)
    result.extra["features"] = {
        "sae": sae.describe() | {"fit": None},
        "fit": _fit(
            sae,
            receiver_acts,
            spec.tokenization.prepend_bos,
            receiver_outs if sae.transcoder else None,
        ),
    }
    warnings.extend(check_values(patched, "patched values", backend.info.dtype))
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
    # Every feature is chosen on some prompts and reported on others (all of them, unless the
    # scope holds some out): the run measures the reported ones.
    choose_at, report_at = _feature_split(scope, len(prompts)) if every else ([], [])
    reported = [prompts[i] for i in report_at] if every else prompts
    scorer = make_scorer(spec, prompts)
    baselines = compute_baselines(backend, prompts, groups, batch_size, cancel, scorer)
    check_finite(baselines, backend.info.dtype)
    receiver, source, reference = _directions(exp)
    receiver = receiver_override or receiver
    source = source_override or source
    if every:
        baselines = baselines.subset(report_at)
    warnings = check_gap(spec, baselines, reported, receiver, reference)
    _check_continuations(
        scorer,
        [position] if every else [rs.site.position for rs in chosen],  # type: ignore[list-item]
    )
    if exp.method != "gradient":
        raise ScopeError(
            "SAE features are estimated from a single gradient. Choose that method, or estimate "
            "the model's components with integrated gradients."
        )
    key = (sae.site_in, sae.layer)  # the features are read here
    out_key = (sae.site, sae.layer)  # and written here (the same site, but for a transcoder)
    keys = list(dict.fromkeys([key, out_key]))
    n, d_sae = len(prompts), sae.d_sae
    W_dec = sae.params.W_dec
    # Every feature's estimate for every prompt (or, for chosen sites, those features' at every
    # position), and the whole site's estimate, to see how much of it the features account for.
    estimates = np.zeros((n, d_sae)) if every else np.zeros((len(chosen), n))
    site_total = np.zeros(n)
    receiver_acts: list[torch.Tensor] = []
    receiver_outs: list[torch.Tensor] = []
    done = 0
    for group in groups:
        receiver_tokens = group.clean if receiver == "clean" else group.corrupt
        source_tokens = group.clean if source == "clean" else group.corrupt
        for sl in _chunks(len(group.members), batch_size):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            idx = group.members[sl]
            src = backend.capture(source_tokens[sl], keys)
            acts, grads = metric_gradients(backend, scorer, receiver_tokens[sl], idx, keys)
            x_src = sae.reads(src[key].float())
            x_rec = sae.reads(acts[key].float())
            grad = (
                sae.flat(grads[out_key].float()) if sae.site == "head" else grads[out_key].float()
            )
            receiver_acts.append(acts[key].float().cpu())  # for the fit, later
            if sae.transcoder:
                receiver_outs.append(acts[out_key].float().cpu())
            out_src = sae.flat(src[out_key].float()) if sae.site == "head" else src[out_key].float()
            out_rec = (
                sae.flat(acts[out_key].float()) if sae.site == "head" else acts[out_key].float()
            )
            positions = range(x_rec.shape[1])
            # The whole site's estimate, to see how much of it the features account for.
            per_position = ((out_src - out_rec) * grad).sum(-1)  # [B, pos]
            if every:
                rows = float64(torch.zeros(len(idx), d_sae, device=x_rec.device))
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
                    term = float64((f_src - f_rec) * (g @ W_dec.T))  # [B, d_sae]
                    if mask is not None:
                        term = term * mask[:, None].to(term.device)
                    rows += term
                estimates[idx] = rows.cpu().numpy()
                if position is None or isinstance(position, AllPositions):
                    site_total[idx] = float64(per_position.sum(1)).cpu().numpy()
                else:
                    at = [resolve_position(position, prompts[p]) for p in idx]
                    site_total[idx] = (
                        float64(per_position[torch.arange(len(idx)), at]).cpu().numpy()
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
                    estimates[rs.index, idx] = float64(values).cpu().numpy()
                # The whole site at the chosen sites' position, when they share one.
                shared = {rs.site.position.model_dump_json() for rs in chosen}
                if len(shared) == 1 and not isinstance(chosen[0].site.position, AllPositions):
                    at = [resolve_position(chosen[0].site.position, prompts[p]) for p in idx]
                    site_total[idx] = (
                        float64(per_position[torch.arange(len(idx)), at]).cpu().numpy()
                    )
                else:
                    site_total[idx] = float64(per_position.sum(1)).cpu().numpy()
            done += len(idx)
            if on_progress is not None:
                on_progress(done, n, sae.layer)

    if every:
        assert position is not None and isinstance(scope, FeaturesScope)
        means = estimates[choose_at].mean(0)
        order = sorted(range(d_sae), key=lambda i: (-abs(means[i]), i))[: scope.top]
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
        delta = estimates[report_at][:, order].T.copy()
        features_sum = estimates[report_at].sum(1)
        site_total = site_total[report_at]
    else:
        sites = chosen
        delta = estimates
        features_sum = None
    warnings.extend(check_values(delta, "estimates", backend.info.dtype))
    held_out = every and len(report_at) < len(prompts)
    result = EngineResult(
        sites=sites,
        layout=layout,
        prompts=reported,
        baselines=baselines,
        receiver=receiver,
        reference=reference,
        patched=baselines.metric(receiver)[None, :] + delta,
        patched_prob=np.full(delta.shape, np.nan),
        patched_pref=np.full(delta.shape, np.nan),
        warnings=warnings,
        measure="estimate",
        delta=delta,
        extra={
            "features": {
                "sae": sae.describe() | {"fit": None},
                "fit": _fit(
                    sae,
                    receiver_acts,
                    spec.tokenization.prepend_bos,
                    receiver_outs if sae.transcoder else None,
                ),
                "site_estimate": float(site_total.mean()),
                "features_estimate": None if features_sum is None else float(features_sum.mean()),
                "evaluated": d_sae if every else len(chosen),
                # With a held-out split: the prompts that chose the features, and those that
                # report them (the run's prompts).
                "chosen_on": [prompts[i].index for i in choose_at] if held_out else None,
                "reported_on": [p.index for p in reported] if held_out else None,
            }
        },
    )
    if on_layer is not None:
        on_layer(sae.layer, [rs.index for rs in sites], result)
    return result
