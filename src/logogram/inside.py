"""Look inside any text, without making a dataset: what the model predicts after each token, what
each layer would predict at one of them (the logit lens), where a head attends, which SAE features
fire, and what the model writes next, with or without an SAE feature added. Nothing is saved.

With a chat template (logogram.chat), the text is wrapped in it as a prompt of a dataset would be:
as the person's message, or as the start of the model's reply; the tokens read are the wrapped
text's, and ``text_tokens`` says which of them hold the text itself."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from logogram.backends.base import ModelBackend, float64
from logogram.chat import WrapError, chat_wrapper
from logogram.prompts import first_real_token, prompt_span, tokenize_prompt
from logogram.spec import AssistantPrefix, UserTurn

# A few paragraphs: enough to look at, small enough to answer at once.
TEXT_MAX = 4000
# The most tokens a continuation adds to the text.
CONTINUE_MAX = 64

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
    f = _encoded(backend, sae, tokens)
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


def _encoded(backend: ModelBackend, sae: Any, tokens: torch.Tensor) -> torch.Tensor:
    """The SAE's features at each token of one text (``tokens`` ``[1, pos]``): ``[pos, d_sae]``.
    Heads' outputs are read side by side while the batch dimension is still there."""
    key = (sae.site_in, sae.layer)
    f, _ = sae.encode(sae.reads(backend.capture(tokens, [key])[key].float())[0])
    return f


def _check_feature(sae: Any, feature: int) -> None:
    if not 0 <= feature < sae.d_sae:
        raise ValueError(
            f"The SAE has {sae.d_sae:,} features (0 to {sae.d_sae - 1:,}); there is no feature "
            f"{feature}."
        )


def look_feature(
    backend: ModelBackend,
    sae: Any,
    text: str,
    *,
    prepend_bos: bool,
    feature: int,
    chat_template: ChatTemplate = None,
) -> dict[str, Any]:
    """One SAE feature's activation at each token of the text, and its largest from the first
    real token on (SAEs aren't trained on the beginning-of-sequence token): how active it is
    here, in its own units, before choosing how strongly to add it."""
    _check_feature(sae, feature)
    tokenized, tokens = _tokens(backend, text, prepend_bos, chat_template)
    column = float64(_encoded(backend, sae, tokens)[:, feature]).cpu()
    first = first_real_token(backend, tokenized.ids, prepend_bos, chat_template is not None)
    at = first + int(torch.argmax(column[first:]))  # the first token where it is largest
    return {
        "tokens": tokenized.tokens,
        "feature": feature,
        "activations": [float(v) for v in column],
        "largest": float(column[at]),
        "position": at,
        "first_real_token": first,
    }


def feature_edit(
    sae: Any, feature: int, activation: float, start: int
) -> Callable[..., torch.Tensor]:
    """An edit of the activation the SAE writes (``[B, pos, d]``, or ``[B, pos, heads, d_head]``
    for the heads' outputs) that adds ``activation`` units of ``feature`` at every position from
    ``start`` on: its decoder row times ``activation``, which is what the feature writes when
    that active (times each input's standard deviation, for an SAE that standardizes its
    inputs; a transcoder's input arrives as ``read``)."""

    def edit(act: torch.Tensor, read: torch.Tensor | None = None) -> torch.Tensor:
        out = act.float().clone()
        here = out[:, start:]
        inputs = None
        if sae.standardizes:
            inputs = sae.reads((read if read is not None else act)[:, start:].float())
        add = activation * sae.unit(feature, inputs).to(out.device)
        if sae.site == "head":
            add = add.reshape(*add.shape[:-1], out.shape[-2], out.shape[-1])
        out[:, start:] = here + add
        return out

    return edit


def continue_text(
    backend: ModelBackend,
    text: str,
    *,
    prepend_bos: bool,
    tokens: int,
    sae: Any = None,
    feature: int | None = None,
    activation: float | None = None,
    start: int | None = None,
    chat_template: ChatTemplate = None,
) -> dict[str, Any]:
    """What the model writes after the text (wrapped in ``chat_template``, if given): ``tokens`` greedy tokens (the most likely each time,
    ties to the lowest token id), the whole text running again for each. With ``feature``, a
    second continuation adds that feature of the SAE where the SAE writes, ``activation`` units
    of it (its decoder row times ``activation``, in the feature's own units), at every position
    from token ``start`` on (the first real token by default), the new tokens included."""
    from logogram.features import check_sae_fits
    from logogram.spec import describe_sae_site
    from logogram.steering import greedy_continue

    if not 1 <= tokens <= CONTINUE_MAX:
        raise ValueError(f"Continue the text by 1 to {CONTINUE_MAX} tokens.")
    tokenized, ids = _tokens(backend, text, prepend_bos, chat_template)
    n = len(tokenized.ids)
    n_ctx = backend.info.n_ctx
    if n + tokens > n_ctx:
        room = n_ctx - n
        fix = f"continue it by at most {room}" if room > 0 else "shorten it"
        raise ValueError(
            f"The text is {n} tokens, and this model reads at most {n_ctx} at once: {fix}."
        )

    def written(new: torch.Tensor) -> dict[str, Any]:
        out = [int(i) for i in new[0].tolist()]
        return {"ids": out, "tokens": [backend.token_str(i) for i in out]}

    report: dict[str, Any] = {
        "tokens": tokenized.tokens,
        "ids": tokenized.ids,
        "prepend_bos": prepend_bos,
        "new": tokens,
        "plain": written(greedy_continue(backend, ids, tokens)),
        "steer": None,
        "steered": None,
    }
    if feature is None:
        return report
    if sae is None:
        raise ValueError("Load an SAE to add one of its features.")
    check_sae_fits(sae, backend.info)
    _check_feature(sae, feature)
    if activation is None or not math.isfinite(activation):
        raise ValueError("Give the activation to add the feature at: a number, in its own units.")
    real = first_real_token(backend, tokenized.ids, prepend_bos, chat_template is not None)
    first = real if start is None else start
    if not 0 <= first < n:
        raise ValueError(f"Token {first} isn't in this text, which has {n} tokens (0 to {n - 1}).")
    reference = look_feature(
        backend, sae, text, prepend_bos=prepend_bos, feature=feature, chat_template=chat_template
    )
    edit = feature_edit(sae, feature, activation, first)
    # A transcoder that standardized its input would scale what it writes by the input's spread.
    read = sae.site_in if sae.standardizes and sae.transcoder else None

    def forward(seq: torch.Tensor) -> torch.Tensor:
        return backend.edit_logits(seq, sae.site, sae.layer, edit, 1, read)[:, -1]

    report["steered"] = written(greedy_continue(backend, ids, tokens, forward))
    report["steer"] = {
        "feature": feature,
        "activation": float(activation),
        "start": first,
        "site": str(sae.site),
        "layer": int(sae.layer),
        "where": describe_sae_site(str(sae.site), int(sae.layer), bool(sae.transcoder)),
        "largest": reference["largest"],
        "largest_at": reference["position"],
    }
    return report
