"""Interactive analyses: the token strip, the baseline check and attention patterns."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from logogram.backends.base import ModelBackend, float64
from logogram.datasets import PromptRecord
from logogram.engine import compute_baselines, patched_forward
from logogram.metrics import Scorer
from logogram.prompts import PreparedPrompt, PromptIssue, group_by_length, prepare_prompt
from logogram.spec import (
    METRIC_LABELS,
    SINGLE_POSITION_METRICS,
    IndexPosition,
    LogProbDiffMetric,
    PredictionSettings,
    describe_metric,
)


def _f(x: float) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


def answer_text(value: str | list[str]) -> str:
    """An answer or distractor for display: a set reads as its members."""
    if isinstance(value, list):
        shown = ", ".join(repr(v) for v in value[:6])
        return f"any of {shown}{f' and {len(value) - 6} more' if len(value) > 6 else ''}"
    return value


def tokenize_pair(
    backend: ModelBackend, record: PromptRecord, prepend_bos: bool, continuations: bool = True
) -> dict[str, Any]:
    """Everything the token strip shows for one prompt pair, including what's wrong with it."""
    prepared, issues, clean, corrupt = prepare_prompt(
        backend, record, 0, prepend_bos, continuations
    )
    n = min(len(clean.ids), len(corrupt.ids))
    differs = [i for i in range(n) if clean.ids[i] != corrupt.ids[i]]
    differs += list(range(n, max(len(clean.ids), len(corrupt.ids))))

    def answer_info(value: str | list[str]) -> dict[str, Any]:
        if isinstance(value, list):
            ids = [backend.single_token_id(v) for v in value]
            return {
                "text": answer_text(value),
                "tokens": [
                    backend.token_str(i) if i is not None else v
                    for v, i in zip(value, ids, strict=True)
                ],
                "id": None,
                "alternatives": True,
            }
        pieces = backend.tokenize(value, prepend_bos=False)
        return {
            "text": value,
            "tokens": pieces.tokens,
            "id": pieces.ids[0] if len(pieces.ids) == 1 else None,
            "alternatives": False,
        }

    return {
        "clean": {"tokens": clean.tokens, "ids": clean.ids},
        "corrupt": {"tokens": corrupt.tokens, "ids": corrupt.ids},
        "aligned": len(clean.ids) == len(corrupt.ids),
        "differs": differs,
        "labels": prepared.labels if prepared else {},
        "answer": answer_info(record.answer),
        "distractor": answer_info(record.distractor),
        "issues": [i.to_dict() for i in issues],
    }


def prepare_with_issues(
    backend: ModelBackend,
    records: list[PromptRecord],
    prepend_bos: bool,
    continuations: bool = True,
) -> tuple[list[PreparedPrompt], list[PromptIssue]]:
    prepared, issues = [], []
    for i, record in enumerate(records):
        p, prompt_issues, _, _ = prepare_prompt(backend, record, i, prepend_bos, continuations)
        issues.extend(prompt_issues)
        if p is not None:
            prepared.append(p)
    return prepared, issues


def baseline_report(
    backend: ModelBackend,
    records: list[PromptRecord],
    *,
    prepend_bos: bool,
    batch_size: int,
    metric: Any = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run clean and corrupt prompts unpatched: the metric, the preference (log P(answer) −
    log P(distractor), the logit difference for single tokens), probabilities and top tokens."""
    metric = metric or LogProbDiffMetric(kind="logprob_diff", normalization="dataset_gap")
    continuations = metric.kind not in SINGLE_POSITION_METRICS
    prepared, issues = prepare_with_issues(backend, records, prepend_bos, continuations)
    if not prepared:
        return {"n": 0, "prompts": [], "issues": [i.to_dict() for i in issues], "summary": None}
    groups = group_by_length(prepared)
    scorer = Scorer(metric, prepared)
    base = compute_baselines(backend, prepared, groups, batch_size, scorer=scorer)
    top: dict[str, list[list[dict[str, Any]]]] = {
        "clean": [[] for _ in prepared],
        "corrupt": [[] for _ in prepared],
    }
    for which in ("clean", "corrupt"):
        for group in groups:
            tokens = group.clean if which == "clean" else group.corrupt
            for start in range(0, len(group.members), batch_size):
                idx = group.members[start : start + batch_size]
                logits = backend.final_logits(tokens[start : start + batch_size])
                probs = torch.softmax(logits, dim=-1)
                values, ids = probs.topk(top_k, dim=-1)
                for row, p in enumerate(idx):
                    top[which][p] = [
                        {"token": backend.token_str(int(t)), "id": int(t), "prob": float(v)}
                        for v, t in zip(values[row].tolist(), ids[row].tolist(), strict=True)
                    ]
    rows = []
    for i, p in enumerate(prepared):
        rows.append(
            {
                "index": p.index,
                "clean": p.record.clean,
                "corrupt": p.record.corrupt,
                "answer": answer_text(p.record.answer),
                "distractor": answer_text(p.record.distractor),
                "clean_logit_diff": _f(base.clean_pref[i]),
                "corrupt_logit_diff": _f(base.corrupt_pref[i]),
                "clean_answer_prob": _f(base.clean_prob[i]),
                "corrupt_answer_prob": _f(base.corrupt_prob[i]),
                "clean_metric": _f(base.clean[i]),
                "corrupt_metric": _f(base.corrupt[i]),
                "clean_top": top["clean"][i],
                "corrupt_top": top["corrupt"][i],
            }
        )
    gap = base.clean_pref - base.corrupt_pref
    summary = {
        "clean_logit_diff": _f(base.clean_pref.mean()),
        "corrupt_logit_diff": _f(base.corrupt_pref.mean()),
        "gap": _f(gap.mean()),
        "clean_prefers_answer": int((base.clean_pref > 0).sum()),
        "corrupt_prefers_answer": int((base.corrupt_pref > 0).sum()),
        "clean_answer_prob": _f(base.clean_prob.mean()),
        "corrupt_answer_prob": _f(base.corrupt_prob.mean()),
        "metric": {
            "kind": metric.kind,
            "target": getattr(metric, "target", None),
            "label": METRIC_LABELS[metric.kind],
            "description": describe_metric(metric),
            "clean": _f(base.clean.mean()),
            "corrupt": _f(base.corrupt.mean()),
            "gap": _f((base.clean - base.corrupt).mean()),
        },
    }
    return {
        "n": len(prepared),
        "prompts": rows,
        "issues": [i.to_dict() for i in issues],
        "summary": summary,
    }


def attention_report(
    backend: ModelBackend,
    records: list[PromptRecord],
    *,
    index: int,
    layer: int,
    head: int,
    which: str = "clean",
    prepend_bos: bool = True,
    batch_size: int = 64,
) -> dict[str, Any]:
    """The head's attention on one prompt, and averaged over prompts of the same token length."""
    info = backend.info
    if not 0 <= layer < info.n_layers or not 0 <= head < info.n_heads:
        raise ValueError(f"L{layer} H{head} doesn't exist in this model.")
    if not 0 <= index < len(records):
        raise ValueError(f"There is no prompt {index}.")
    prepared, _ = prepare_with_issues(backend, records, prepend_bos)
    target = next((p for p in prepared if p.index == index), None)
    if target is None:
        raise ValueError(f"Prompt {index} can't be used; fix it in the dataset first.")
    same = [p for p in prepared if p.length == target.length]
    tokens = torch.tensor(
        [(p.clean if which == "clean" else p.corrupt).ids for p in same], dtype=torch.long
    )
    total = None
    chosen = None
    for start in range(0, len(same), batch_size):
        pattern = backend.attention_pattern(tokens[start : start + batch_size], layer)[:, head]
        pattern = float64(pattern).cpu()
        total = pattern.sum(0) if total is None else total + pattern.sum(0)
        for row, p in enumerate(same[start : start + batch_size]):
            if p.index == index:
                chosen = pattern[row]
    assert total is not None and chosen is not None
    average = total / len(same)
    tok = (target.clean if which == "clean" else target.corrupt).tokens
    # The average mixes prompts, so only labels at the same index in all of them, and tokens
    # identical in all of them, describe its positions.
    seqs = [(p.clean if which == "clean" else p.corrupt).tokens for p in same]
    common_tokens = [t if all(s[i] == t for s in seqs) else None for i, t in enumerate(tok)]
    common_labels = {
        label: pos
        for label, pos in target.labels.items()
        if all(p.labels.get(label) == pos for p in same)
    }
    return {
        "layer": layer,
        "head": head,
        "which": which,
        "index": index,
        "tokens": tok,
        "labels": target.labels,
        "pattern": chosen.tolist(),
        "average": average.tolist(),
        "average_tokens": common_tokens,
        "average_labels": common_labels,
        "n_average": len(same),
        "n_total": len(prepared),
        "length": target.length,
    }


def prediction_report(
    backend: ModelBackend,
    records: list[PromptRecord],
    *,
    index: int,
    settings: PredictionSettings,
    prepend_bos: bool,
    batch_size: int,
) -> dict[str, Any]:
    """A descriptive logit lens, preserving the baseline's length groups and batch shape."""
    if index != settings.prompt_index:
        raise ValueError("The prompt index must match the prediction settings in the spec.")
    if not 0 <= index < len(records):
        raise ValueError(f"There is no prompt {index}.")
    prepared, issues = prepare_with_issues(backend, records, prepend_bos)
    target_idx = next((i for i, p in enumerate(prepared) if p.index == index), None)
    if target_idx is None:
        raise ValueError(f"Prompt {index} can't be used; fix it in the dataset first.")
    target = prepared[target_idx]
    pos = settings.position.index if isinstance(settings.position, IndexPosition) else -1
    pos = target.length + pos if pos < 0 else pos
    if not 0 <= pos < target.length:
        raise ValueError(f"Token position {pos} is outside this prompt ({target.length} tokens).")
    group = next(g for g in group_by_length(prepared) if target_idx in g.members)
    within = group.members.index(target_idx)
    start = (within // batch_size) * batch_size
    tokens = group.clean if settings.which == "clean" else group.corrupt
    batch = tokens[start : start + batch_size]
    logits = backend.layer_logits(batch, pos, within - start)
    if not torch.isfinite(logits).all():
        raise ValueError("The prediction diagnostic produced non-finite logits. Try float32.")
    probs = torch.softmax(logits, dim=-1)
    # Stable ordering makes ties deterministic, including toy or degenerate models.
    top = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, : settings.top_k]
    # The lens predicts one token: a set reads as the sum over its members, a continuation as
    # its first token.
    answer = list(target.answer.ids if target.answer.alternatives else target.answer.ids[:1])
    distractor = list(
        target.distractor.ids if target.distractor.alternatives else target.distractor.ids[:1]
    )
    rows = []
    for layer in range(backend.info.n_layers):
        answer_logit = torch.logsumexp(logits[layer, answer], dim=-1)
        distractor_logit = torch.logsumexp(logits[layer, distractor], dim=-1)
        rows.append(
            {
                "layer": layer,
                "top": [
                    {
                        "id": int(t),
                        "token": backend.token_str(int(t)),
                        "prob": float(probs[layer, t]),
                    }
                    for t in top[layer]
                ],
                "answer_prob": float(probs[layer, answer].sum()),
                "distractor_prob": float(probs[layer, distractor].sum()),
                "logit_diff": float(answer_logit - distractor_logit),
            }
        )
    return {
        "index": index,
        "position": pos,
        "tokens": (target.clean if settings.which == "clean" else target.corrupt).tokens,
        "settings": settings.model_dump(mode="json"),
        "layers": rows,
        "batch_members": [prepared[i].index for i in group.members[start : start + batch_size]],
        "answer": answer_text(target.record.answer),
        "distractor": answer_text(target.record.distractor),
        "answer_reading": _reading(target.answer),
        "issues": [issue.to_dict() for issue in issues],
    }


def _reading(target: Any) -> str:
    if target.alternatives:
        return "set"
    return "token" if target.single else "first_token"


# -- sparse autoencoder features -------------------------------------------------------------


def _first_real_token(prepend_bos: bool) -> int:
    """Where real tokens start: SAEs aren't trained on the beginning-of-sequence token, whose
    activations are unlike any other, so fits and splices leave it alone."""
    return 1 if prepend_bos else 0


def sae_fit_report(
    backend: ModelBackend,
    sae: Any,
    records: list[PromptRecord],
    *,
    prepend_bos: bool,
    batch_size: int,
) -> dict[str, Any]:
    """How well the SAE fits these prompts: the variance of the activations it explains, how many
    features fire per token, and what the logit difference becomes when the model runs on the
    SAE's reconstruction instead of the activation (its error removed)."""
    from logogram.sae import fit_on

    prepared, issues = prepare_with_issues(backend, records, prepend_bos)
    if not prepared:
        raise ValueError("None of these prompts can be used with the loaded model.")
    start = _first_real_token(prepend_bos)
    key = (sae.site_in, sae.layer)
    out_key = (sae.site, sae.layer)
    acts, outs, clean, spliced = [], [], [], []
    scorer = Scorer(LogProbDiffMetric(kind="logprob_diff", normalization="dataset_gap"), prepared)

    def written(x: torch.Tensor) -> torch.Tensor:
        return sae.flat(x) if sae.site == "head" else x

    for group in group_by_length(prepared):
        length = group.length

        def splice(
            x: torch.Tensor, read: torch.Tensor | None = None, length: int = length
        ) -> torch.Tensor:
            # The prompt's real tokens only: not the beginning-of-sequence token, nor tokens
            # appended to read a continuation. A transcoder's prediction replaces its MLP's output.
            out = x.float().clone()
            source = out if read is None else read.float()
            f, stats = sae.encode(sae.reads(source[:, start:length]))
            out[:, start:length] = sae.decode(f, stats).reshape(out[:, start:length].shape)
            return out

        def edited(toks: torch.Tensor, keep: int, splice: Any = splice) -> torch.Tensor:
            read = sae.site_in if sae.transcoder else None
            return backend.edit_logits(toks, sae.site, sae.layer, splice, keep, read)

        for begin in range(0, len(group.members), batch_size):
            idx = group.members[begin : begin + batch_size]
            tokens = group.clean[begin : begin + batch_size]
            captured = backend.capture(tokens, list(dict.fromkeys([key, out_key])))
            # Kept in CPU memory until the fit, which encodes them a chunk at a time.
            acts.append(sae.reads(captured[key][:, start:].float()).cpu())
            outs.append(written(captured[out_key][:, start:].float()).cpu())
            clean.append(scorer.score(patched_forward(backend, None), tokens, idx).pref)
            spliced.append(scorer.score(edited, tokens, idx).pref)
    flat = torch.cat([a.reshape(-1, a.shape[-1]) for a in acts])
    targets = torch.cat([a.reshape(-1, a.shape[-1]) for a in outs]) if sae.transcoder else None
    fit = fit_on(sae, flat, targets)
    ld, ld_spliced = np.concatenate(clean), np.concatenate(spliced)
    fit.update(
        {
            "logit_diff": _f(ld.mean()),
            "spliced_logit_diff": _f(ld_spliced.mean()),
            "n": len(prepared),
            "skipped": len(issues),
        }
    )
    sae.fit = fit
    return fit


def _single(backend: ModelBackend, records: list[PromptRecord], index: int, prepend_bos: bool):  # type: ignore[no-untyped-def]
    prepared, _ = prepare_with_issues(backend, records, prepend_bos)
    target = next((p for p in prepared if p.index == index), None)
    if target is None:
        raise ValueError(f"Prompt {index} can't be used; fix it in the dataset first.")
    return prepared, target


def token_features_report(
    backend: ModelBackend,
    sae: Any,
    records: list[PromptRecord],
    *,
    index: int,
    which: str,
    prepend_bos: bool,
    top_k: int = 8,
) -> dict[str, Any]:
    """The features that fire most on each token of one prompt."""
    _, target = _single(backend, records, index, prepend_bos)
    tokenized = target.clean if which == "clean" else target.corrupt
    tokens = torch.tensor([tokenized.ids], dtype=torch.long)
    key = (sae.site_in, sae.layer)
    f, _ = sae.encode(sae.reads(backend.capture(tokens, [key])[key][0].float()))
    values, ids = f.topk(min(top_k, f.shape[-1]), dim=-1)
    per_token = [
        [
            {"feature": int(i), "activation": float(v)}
            for v, i in zip(values[p].tolist(), ids[p].tolist(), strict=True)
            if v > 0
        ]
        for p in range(f.shape[0])
    ]
    return {
        "index": index,
        "which": which,
        "tokens": tokenized.tokens,
        "labels": target.labels,
        "features": per_token,
        "active": [int((f[p] > 0).sum()) for p in range(f.shape[0])],
        "first_real_token": _first_real_token(prepend_bos),
    }


def feature_report(
    backend: ModelBackend,
    sae: Any,
    records: list[PromptRecord],
    *,
    feature: int,
    index: int,
    which: str,
    prepend_bos: bool,
    batch_size: int,
    top: int = 10,
) -> dict[str, Any]:
    """Where one feature fires: along one prompt's tokens, and on which prompts of the dataset
    most strongly (all computed here, from the project's own prompts)."""
    if not 0 <= feature < sae.d_sae:
        raise ValueError(f"The SAE has {sae.d_sae} features; there is no feature {feature}.")
    prepared, target = _single(backend, records, index, prepend_bos)
    key = (sae.site_in, sae.layer)
    start = _first_real_token(prepend_bos)
    strongest: list[dict[str, Any]] = []
    along: list[float] = []
    for group in group_by_length(prepared):
        tokens = group.clean if which == "clean" else group.corrupt
        for begin in range(0, len(group.members), batch_size):
            idx = group.members[begin : begin + batch_size]
            f, _ = sae.encode(
                sae.reads(backend.capture(tokens[begin : begin + batch_size], [key])[key].float())
            )
            column = float64(f[..., feature]).cpu()  # [B, pos]
            for row, p in enumerate(idx):
                prompt = prepared[p]
                values = column[row]
                if prompt.index == index:
                    along = [float(v) for v in values]
                best = int(values[start:].argmax()) + start if values.shape[0] > start else 0
                seq = prompt.clean if which == "clean" else prompt.corrupt
                strongest.append(
                    {
                        "index": prompt.index,
                        "max": float(values[best]),
                        "position": best,
                        "token": seq.tokens[best],
                        "text": prompt.record.clean if which == "clean" else prompt.record.corrupt,
                    }
                )
    strongest.sort(key=lambda r: (-r["max"], r["index"]))
    tokenized = target.clean if which == "clean" else target.corrupt
    return {
        "feature": feature,
        "which": which,
        "index": index,
        "tokens": tokenized.tokens,
        "activations": along,
        "top": [r for r in strongest[:top] if r["max"] > 0],
        "active_prompts": sum(1 for r in strongest if r["max"] > 0),
        "n": len(strongest),
    }
