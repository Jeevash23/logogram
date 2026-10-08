"""Interactive analyses: the token strip, the baseline check and attention patterns."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from logogram.backends.base import ModelBackend
from logogram.datasets import PromptRecord
from logogram.engine import compute_baselines
from logogram.prompts import PreparedPrompt, PromptIssue, group_by_length, prepare_prompt
from logogram.spec import IndexPosition, PredictionSettings


def _f(x: float) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


def tokenize_pair(backend: ModelBackend, record: PromptRecord, prepend_bos: bool) -> dict[str, Any]:
    """Everything the token strip shows for one prompt pair, including what's wrong with it."""
    prepared, issues, clean, corrupt = prepare_prompt(backend, record, 0, prepend_bos)
    n = min(len(clean.ids), len(corrupt.ids))
    differs = [i for i in range(n) if clean.ids[i] != corrupt.ids[i]]
    differs += list(range(n, max(len(clean.ids), len(corrupt.ids))))

    def answer_info(text: str) -> dict[str, Any]:
        pieces = backend.tokenize(text, prepend_bos=False)
        return {
            "text": text,
            "tokens": pieces.tokens,
            "id": pieces.ids[0] if len(pieces.ids) == 1 else None,
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
    backend: ModelBackend, records: list[PromptRecord], prepend_bos: bool
) -> tuple[list[PreparedPrompt], list[PromptIssue]]:
    prepared, issues = [], []
    for i, record in enumerate(records):
        p, prompt_issues, _, _ = prepare_prompt(backend, record, i, prepend_bos)
        issues.extend(prompt_issues)
        if p is not None:
            prepared.append(p)
    return prepared, issues


def baseline_report(
    backend: ModelBackend,
    records: list[PromptRecord],
    *,
    prepend_bos: bool = True,
    batch_size: int = 64,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run clean and corrupt prompts unpatched: logit differences, probabilities, top tokens."""
    prepared, issues = prepare_with_issues(backend, records, prepend_bos)
    if not prepared:
        return {"n": 0, "prompts": [], "issues": [i.to_dict() for i in issues], "summary": None}
    groups = group_by_length(prepared)
    base = compute_baselines(backend, prepared, groups, batch_size)
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
                "answer": p.record.answer,
                "distractor": p.record.distractor,
                "clean_logit_diff": _f(base.clean_ld[i]),
                "corrupt_logit_diff": _f(base.corrupt_ld[i]),
                "clean_answer_prob": _f(base.clean_prob[i]),
                "corrupt_answer_prob": _f(base.corrupt_prob[i]),
                "clean_top": top["clean"][i],
                "corrupt_top": top["corrupt"][i],
            }
        )
    gap = base.clean_ld - base.corrupt_ld
    summary = {
        "clean_logit_diff": _f(base.clean_ld.mean()),
        "corrupt_logit_diff": _f(base.corrupt_ld.mean()),
        "gap": _f(gap.mean()),
        "clean_prefers_answer": int((base.clean_ld > 0).sum()),
        "corrupt_prefers_answer": int((base.corrupt_ld > 0).sum()),
        "clean_answer_prob": _f(base.clean_prob.mean()),
        "corrupt_answer_prob": _f(base.corrupt_prob.mean()),
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
        pattern = pattern.double().cpu()
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
    answer = target.answer_id
    distractor = target.distractor_id
    rows = []
    for layer in range(backend.info.n_layers):
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
                "answer_prob": float(probs[layer, answer]),
                "distractor_prob": float(probs[layer, distractor]),
                "logit_diff": float(logits[layer, answer] - logits[layer, distractor]),
            }
        )
    return {
        "index": index,
        "position": pos,
        "tokens": (target.clean if settings.which == "clean" else target.corrupt).tokens,
        "settings": settings.model_dump(mode="json"),
        "layers": rows,
        "batch_members": [prepared[i].index for i in group.members[start : start + batch_size]],
        "answer": target.record.answer,
        "distractor": target.record.distractor,
        "issues": [issue.to_dict() for issue in issues],
    }
