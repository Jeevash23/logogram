"""Look inside any text, without making a dataset: what the model predicts after each token, what
each layer would predict at one of them (the logit lens), where a head attends, and which SAE
features fire. Nothing is saved, and nothing here is an intervention."""

from __future__ import annotations

from typing import Any

import torch

from logogram.backends.base import ModelBackend

# A few paragraphs: enough to look at, small enough to answer at once.
TEXT_MAX = 4000


def _tokens(backend: ModelBackend, text: str, prepend_bos: bool) -> tuple[Any, torch.Tensor]:
    if not text.strip():
        raise ValueError("Type some text to read.")
    if len(text) > TEXT_MAX:
        raise ValueError(
            f"The text is {len(text):,} characters; read up to {TEXT_MAX:,} at a time."
        )
    tokenized = backend.tokenize(text, prepend_bos)
    n_ctx = backend.info.n_ctx
    if len(tokenized.ids) > n_ctx:
        raise ValueError(
            f"The text is {len(tokenized.ids)} tokens, and this model reads at most {n_ctx}. "
            "Shorten it."
        )
    return tokenized, torch.tensor([tokenized.ids], dtype=torch.long)


def _top(backend: ModelBackend, logits: torch.Tensor, k: int) -> list[list[dict[str, Any]]]:
    """The k most likely next tokens for each row of ``logits`` ``[rows, vocab]``, ties in token
    order so the list never changes between runs."""
    if not torch.isfinite(logits).all():
        raise ValueError("The model produced values that aren't finite numbers here. Try float32.")
    probs = torch.softmax(logits.double(), dim=-1)
    order = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :k]
    return [
        [
            {"id": int(t), "token": backend.token_str(int(t)), "prob": float(probs[r, t])}
            for t in row
        ]
        for r, row in enumerate(order)
    ]


def look_report(
    backend: ModelBackend,
    text: str,
    *,
    prepend_bos: bool,
    top_k: int,
    position: int | None,
) -> dict[str, Any]:
    """The text's tokens, the model's top ``top_k`` next tokens after each of them, and at
    ``position`` (the last token by default) what each layer's residual stream would predict
    through the final normalization and unembedding, when this model passed that check."""
    tokenized, tokens = _tokens(backend, text, prepend_bos)
    n = tokens.shape[1]
    pos = n - 1 if position is None else position
    if not 0 <= pos < n:
        raise ValueError(f"Token {pos} isn't in this text, which has {n} tokens (0 to {n - 1}).")
    predictions = _top(backend, backend.logits(tokens, keep=n)[0], top_k)
    lens: list[dict[str, Any]] | None = None
    lens_note = None
    if backend.info.extra.get("prediction_method") == "final_norm_logit_lens":
        layers = backend.layer_logits(tokens, pos, 0)
        lens = [
            {"layer": layer, "top": top} for layer, top in enumerate(_top(backend, layers, top_k))
        ]
    else:
        lens_note = (
            "What each layer would predict is shown only when the model's final normalization and "
            "unembedding reproduce its output, and for this model they didn't when it loaded."
        )
    return {
        "tokens": tokenized.tokens,
        "ids": tokenized.ids,
        "prepend_bos": prepend_bos,
        "top_k": top_k,
        "position": pos,
        "predictions": predictions,
        "lens": lens,
        "lens_note": lens_note,
    }


def look_attention(
    backend: ModelBackend, text: str, *, prepend_bos: bool, layer: int, head: int
) -> dict[str, Any]:
    """Where one head attends in the text: its attention probabilities ``[query][key]``."""
    info = backend.info
    if not 0 <= layer < info.n_layers or not 0 <= head < info.n_heads:
        raise ValueError(f"L{layer} H{head} doesn't exist in this model.")
    tokenized, tokens = _tokens(backend, text, prepend_bos)
    pattern = backend.attention_pattern(tokens, layer)[0, head].double().cpu()
    return {"tokens": tokenized.tokens, "layer": layer, "head": head, "pattern": pattern.tolist()}


def look_features(
    backend: ModelBackend, sae: Any, text: str, *, prepend_bos: bool, top_k: int
) -> dict[str, Any]:
    """The SAE features that fire most on each token of the text."""
    tokenized, tokens = _tokens(backend, text, prepend_bos)
    key = (sae.site_in, sae.layer)
    f, _ = sae.encode(sae.reads(backend.capture(tokens, [key])[key][0].float()))
    values, ids = f.topk(min(top_k, f.shape[-1]), dim=-1)
    return {
        "tokens": tokenized.tokens,
        "features": [
            [
                {"feature": int(i), "activation": float(v)}
                for v, i in zip(values[p].tolist(), ids[p].tolist(), strict=True)
                if v > 0
            ]
            for p in range(f.shape[0])
        ],
        "active": [int((f[p] > 0).sum()) for p in range(f.shape[0])],
        # SAEs aren't trained on the beginning-of-sequence token.
        "first_real_token": 1 if prepend_bos else 0,
    }
