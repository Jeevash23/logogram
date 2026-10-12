"""Look inside any text, without making a dataset: what the model predicts after each token, what
each layer would predict at one of them (the logit lens), where a head attends, and which SAE
features fire. Nothing is saved, and nothing here is an intervention.

With a chat template (logogram.chat), the text is wrapped in it as a prompt of a dataset would be:
as the person's message, or as the start of the model's reply; the tokens read are the wrapped
text's, and ``text_tokens`` says which of them hold the text itself."""

from __future__ import annotations

from typing import Any

import torch

from logogram.backends.base import ModelBackend, float64
from logogram.chat import WrapError, chat_wrapper
from logogram.prompts import first_real_token, prompt_span, tokenize_prompt
from logogram.spec import AssistantPrefix, UserTurn

# A few paragraphs: enough to look at, small enough to answer at once.
TEXT_MAX = 4000

ChatTemplate = UserTurn | AssistantPrefix | None


def _tokens(
    backend: ModelBackend, text: str, prepend_bos: bool, chat_template: ChatTemplate = None
) -> tuple[Any, torch.Tensor]:
    if not text.strip():
        raise ValueError("Type some text to read.")
    if len(text) > TEXT_MAX:
        raise ValueError(
            f"The text is {len(text):,} characters; read up to {TEXT_MAX:,} at a time."
        )
    try:
        tokenized = tokenize_prompt(
            backend, text, prepend_bos, chat_wrapper(backend, chat_template)
        )
    except WrapError as exc:
        raise ValueError(f"In the text, {exc}") from exc
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
    probs = torch.softmax(float64(logits), dim=-1)
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
    chat_template: ChatTemplate = None,
) -> dict[str, Any]:
    """The text's tokens, the model's top ``top_k`` next tokens after each of them, and at
    ``position`` (the last token by default) what each layer's residual stream would predict
    through the final normalization and unembedding, when this model passed that check."""
    tokenized, tokens = _tokens(backend, text, prepend_bos, chat_template)
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
        "chat_template": None if chat_template is None else chat_template.model_dump(mode="json"),
        # The tokens that hold the text itself: the first and one past the last.
        "text_tokens": list(_text_tokens(tokenized, text, prepend_bos, chat_template)),
        "top_k": top_k,
        "position": pos,
        "predictions": predictions,
        "lens": lens,
        "lens_note": lens_note,
    }


def _text_tokens(
    tokenized: Any, text: str, prepend_bos: bool, chat_template: ChatTemplate
) -> tuple[int, int]:
    if chat_template is None:
        return (1 if prepend_bos else 0), len(tokenized.ids)
    return prompt_span(tokenized, len(text))


def look_attention(
    backend: ModelBackend,
    text: str,
    *,
    prepend_bos: bool,
    layer: int,
    head: int,
    chat_template: ChatTemplate = None,
) -> dict[str, Any]:
    """Where one head attends in the text: its attention probabilities ``[query][key]``."""
    info = backend.info
    if not 0 <= layer < info.n_layers or not 0 <= head < info.n_heads:
        raise ValueError(f"L{layer} H{head} doesn't exist in this model.")
    tokenized, tokens = _tokens(backend, text, prepend_bos, chat_template)
    pattern = float64(backend.attention_pattern(tokens, layer)[0, head]).cpu()
    return {"tokens": tokenized.tokens, "layer": layer, "head": head, "pattern": pattern.tolist()}


def look_features(
    backend: ModelBackend,
    sae: Any,
    text: str,
    *,
    prepend_bos: bool,
    top_k: int,
    chat_template: ChatTemplate = None,
) -> dict[str, Any]:
    """The SAE features that fire most on each token of the text."""
    tokenized, tokens = _tokens(backend, text, prepend_bos, chat_template)
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
        "first_real_token": first_real_token(
            backend, tokenized.ids, prepend_bos, chat_template is not None
        ),
    }
