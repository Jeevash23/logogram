"""Attention heads by what they attend to: previous-token, duplicate-token and induction heads.

Each head is read on sequences of random tokens repeated twice, as in Olsson et al. (2022) and
TransformerLens's head detector. A random sequence carries no meaning, so a head that attends to
the earlier copy of the current token (duplicate-token), or to the token after it (induction), does
so by position and identity alone. Scores are attention probabilities averaged over the second
copy's queries (over every query after the first, for previous-token heads), kept per sequence,
and reported as their mean with a seeded percentile bootstrap interval over sequences.

With a chat template (logogram.chat), each sequence sits where a prompt would: after the tokens of
the template's text before the prompt and before those of its text after it, each tokenized on its
own (the random tokens are no text to tokenize with them).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from logogram.backends.base import ModelBackend, float64
from logogram.stats import resample_counts

if TYPE_CHECKING:
    from logogram.chat import ChatWrapper

# What each kind of head does, as the app and the Python API say it.
HEAD_KINDS: dict[str, str] = {
    "previous_token": "attends to the token just before the current one",
    "duplicate_token": "attends to the earlier copy of the current token",
    "induction": "attends to the token that followed the earlier copy of the current token",
}

TOP = 10


@dataclass(frozen=True)
class HeadKindSettings:
    """Every choice that changes the scores."""

    seed: int
    sequences: int
    length: int
    prepend_bos: bool
    batch_size: int
    bootstrap: int = 1000
    ci: float = 0.95
    # A spec's tokenization.chat_template, as JSON: the template the sequences are wrapped in.
    chat_template: dict[str, Any] | None = None


def _wrapper(backend: ModelBackend, settings: HeadKindSettings) -> ChatWrapper | None:
    if settings.chat_template is None:
        return None
    from logogram.chat import chat_wrapper
    from logogram.spec import read_chat_template

    return chat_wrapper(backend, read_chat_template(settings.chat_template))


def _around(backend: ModelBackend, settings: HeadKindSettings) -> tuple[list[int], list[int]]:
    """The tokens before and after each sequence: the beginning-of-sequence token if asked, or
    a chat template's text before and after a prompt."""
    chat = _wrapper(backend, settings)
    if chat is not None:
        before = backend.tokenize(chat.before, prepend_bos=False).ids if chat.before else []
        after = backend.tokenize(chat.after, prepend_bos=False).ids if chat.after else []
        return before, after
    if settings.prepend_bos:
        return backend.tokenize("", prepend_bos=True).ids, []
    return [], []


def repeated_random_tokens(
    backend: ModelBackend,
    settings: HeadKindSettings,
    around: tuple[list[int], list[int]] | None = None,
) -> torch.Tensor:
    """``[sequences, (1 +) 2 × length]`` token ids: ``length`` distinct, non-special tokens drawn
    with the seed, then the same tokens again, after the beginning-of-sequence token if asked (or
    between a chat template's tokens, ``around``)."""
    count, special = backend.token_space()
    allowed = torch.tensor([i for i in range(count) if i not in special], dtype=torch.long)
    if len(allowed) < settings.length:
        raise ValueError(
            f"The tokenizer has only {len(allowed)} ordinary tokens, fewer than the {settings.length} "
            "a sequence needs. Choose a shorter length."
        )
    prefix, suffix = around if around is not None else _around(backend, settings)
    generator = torch.Generator().manual_seed(settings.seed)
    rows = []
    for _ in range(settings.sequences):
        drawn = allowed[torch.randperm(len(allowed), generator=generator)[: settings.length]]
        rows.append(
            torch.cat(
                [
                    torch.tensor(prefix, dtype=torch.long),
                    drawn,
                    drawn,
                    torch.tensor(suffix, dtype=torch.long),
                ]
            )
        )
    return torch.stack(rows)


def kind_scores_of_pattern(
    pattern: torch.Tensor, length: int, offset: int
) -> dict[str, torch.Tensor]:
    """Each kind's score per sequence and head, ``[B, H]``, from one layer's attention
    probabilities ``[B, H, query, key]`` on repeated sequences: ``offset`` tokens (the beginning
    of sequence) and then two copies of ``length`` tokens."""
    second = torch.arange(offset + length, offset + 2 * length)
    after_first = torch.arange(offset + 1, offset + 2 * length)
    return {
        "previous_token": pattern[:, :, after_first, after_first - 1].mean(-1),
        "duplicate_token": pattern[:, :, second, second - length].mean(-1),
        "induction": pattern[:, :, second, second - length + 1].mean(-1),
    }


def head_kind_scores(backend: ModelBackend, settings: HeadKindSettings) -> dict[str, Any]:
    """Every head's previous-token, duplicate-token and induction score, per sequence and
    summarized with a seeded bootstrap interval over sequences."""
    info = backend.info
    if settings.sequences < 2:
        raise ValueError("Use at least 2 sequences, so the scores have an interval.")
    if settings.length < 2:
        raise ValueError("Use sequences of at least 2 tokens.")
    prefix, suffix = _around(backend, settings)
    offset = len(prefix)
    if offset + 2 * settings.length + len(suffix) > info.n_ctx:
        raise ValueError(
            f"Two copies of {settings.length} tokens don't fit in this model's context of "
            f"{info.n_ctx} tokens. Choose a length up to "
            f"{(info.n_ctx - offset - len(suffix)) // 2}."
        )
    tokens = repeated_random_tokens(backend, settings, (prefix, suffix))
    layers = list(range(info.n_layers))
    per = {kind: np.zeros((settings.sequences, info.n_layers, info.n_heads)) for kind in HEAD_KINDS}
    for start in range(0, settings.sequences, settings.batch_size):
        batch = tokens[start : start + settings.batch_size]
        patterns = backend.attention_patterns(batch, layers)
        for layer, pattern in patterns.items():
            scores = kind_scores_of_pattern(float64(pattern).cpu(), settings.length, offset)
            for kind, value in scores.items():
                per[kind][start : start + len(batch), layer] = value.numpy()
    counts = resample_counts(settings.sequences, settings.bootstrap, settings.seed)
    alpha = (1.0 - settings.ci) / 2.0
    kinds: dict[str, Any] = {}
    top: dict[str, Any] = {}
    for kind, values in per.items():
        flat = values.reshape(settings.sequences, -1)
        mean = flat.mean(0)
        boot = counts @ flat / settings.sequences  # [bootstrap, heads]
        lo, hi = np.quantile(boot, [alpha, 1.0 - alpha], axis=0)
        shape = (info.n_layers, info.n_heads)
        kinds[kind] = {
            "description": HEAD_KINDS[kind],
            "mean": mean.reshape(shape).tolist(),
            "lo": lo.reshape(shape).tolist(),
            "hi": hi.reshape(shape).tolist(),
            "per_sequence": values.tolist(),
        }
        order = np.argsort(-mean, kind="stable")[:TOP]
        top[kind] = [
            {
                "layer": int(i // info.n_heads),
                "head": int(i % info.n_heads),
                "mean": float(mean[i]),
                "lo": float(lo[i]),
                "hi": float(hi[i]),
            }
            for i in order
        ]
    return {
        "settings": asdict(settings),
        "n_layers": info.n_layers,
        "n_heads": info.n_heads,
        "kinds": kinds,
        "top": top,
    }
