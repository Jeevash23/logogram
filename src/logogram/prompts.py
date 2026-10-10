"""Tokenize prompt pairs for a model and check that they can be used in an experiment.

A usable pair has clean and corrupt prompts of the same token length (so positions line up), and
an answer and a distractor the metric can read: single tokens, sets of single tokens, or (for the
log-probability and probability metrics) continuations of several tokens.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch

from logogram.backends.base import ModelBackend, Tokenized
from logogram.datasets import PromptRecord
from logogram.metrics import Target


@dataclass
class PromptIssue:
    index: int
    kind: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {"index": self.index, "kind": self.kind, "message": self.message}


@dataclass
class PreparedPrompt:
    index: int
    record: PromptRecord
    clean: Tokenized
    corrupt: Tokenized
    answer: Target
    distractor: Target
    labels: dict[str, int] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.clean.ids)

    @property
    def answer_id(self) -> int:
        """The answer's token, for methods that read one token (it must be a single token)."""
        return _single_id(self.answer, "answer")

    @property
    def distractor_id(self) -> int:
        return _single_id(self.distractor, "distractor")

    def differing_positions(self) -> list[int]:
        return [
            i
            for i, (a, b) in enumerate(zip(self.clean.ids, self.corrupt.ids, strict=True))
            if a != b
        ]


def _single_id(target: Target, what: str) -> int:
    if not target.single:
        raise ValueError(
            f"This analysis reads the {what} as one token, and this prompt's {what} is "
            f"{'a set of tokens' if target.alternatives else 'several tokens'}."
        )
    return target.ids[0]


class PromptError(ValueError):
    def __init__(self, issues: list[PromptIssue]):
        self.issues = issues
        shown = "; ".join(f"prompt {i.index}: {i.message}" for i in issues[:3])
        more = f" (and {len(issues) - 3} more)" if len(issues) > 3 else ""
        super().__init__(f"{len(issues)} prompt(s) can't be used. {shown}{more}")


def _show_tokens(backend: ModelBackend, text: str) -> str:
    pieces = backend.tokenize(text, prepend_bos=False).tokens
    return " + ".join(repr(p) for p in pieces)


def label_position(tokenized: Tokenized, span: tuple[int, int]) -> int | None:
    """The last token whose character span overlaps ``span``."""
    start, end = span
    found = None
    for i, (a, b) in enumerate(tokenized.offsets):
        if b > a and a < end and b > start:
            found = i
    return found


def _target(
    backend: ModelBackend,
    value: str | list[str],
    what: str,
    index: int,
    continuations: bool,
    issues: list[PromptIssue],
) -> Target | None:
    """The answer or distractor as tokens, or None (with an issue) if the metric can't read it."""
    if isinstance(value, list):
        ids = []
        for text in value:
            token = backend.single_token_id(text)
            if token is None:
                issues.append(
                    PromptIssue(
                        index,
                        f"{what}_tokens",
                        f"the {what} set's {text!r} is several tokens "
                        f"({_show_tokens(backend, text)}). Every member of a set must be a single "
                        "token.",
                    )
                )
                return None
            ids.append(token)
        if len(set(ids)) != len(ids):
            issues.append(
                PromptIssue(index, f"{what}_tokens", f"the {what} set names the same token twice.")
            )
            return None
        return Target(ids=tuple(ids), alternatives=True)
    token = backend.single_token_id(value)
    if token is not None:
        return Target(ids=(token,))
    pieces = backend.tokenize(value, prepend_bos=False).ids
    if not continuations or not pieces:
        issues.append(
            PromptIssue(
                index,
                f"{what}_tokens",
                f"the {what} {value!r} is several tokens ({_show_tokens(backend, value)}). The "
                f"logit difference reads one token: use a single-token {what} (often with a "
                "leading space), or a log-probability or probability metric, which read "
                "continuations of several tokens.",
            )
        )
        return None
    return Target(ids=tuple(int(i) for i in pieces))


def prepare_prompt(
    backend: ModelBackend,
    record: PromptRecord,
    index: int,
    prepend_bos: bool,
    continuations: bool = False,
) -> tuple[PreparedPrompt | None, list[PromptIssue], Tokenized, Tokenized]:
    """Tokenize one pair and check it. ``continuations``: whether the metric can read answers and
    distractors of several tokens."""
    issues: list[PromptIssue] = []
    clean = backend.tokenize(record.clean, prepend_bos)
    corrupt = backend.tokenize(record.corrupt, prepend_bos)
    if len(clean.ids) != len(corrupt.ids):
        issues.append(
            PromptIssue(
                index,
                "length_mismatch",
                f"clean has {len(clean.ids)} tokens and corrupt has {len(corrupt.ids)}, so "
                "positions can't be aligned. Make both prompts tokenize to the same length.",
            )
        )
    answer = _target(backend, record.answer, "answer", index, continuations, issues)
    distractor = _target(backend, record.distractor, "distractor", index, continuations, issues)
    if clean.ids == corrupt.ids:
        issues.append(
            PromptIssue(
                index,
                "identical",
                "the clean and corrupt prompts are the same, so there is nothing to patch.",
            )
        )
    if answer is not None and distractor is not None:
        if answer == distractor:
            issues.append(
                PromptIssue(index, "same_answer", "the answer and distractor are the same token.")
            )
        elif answer.alternatives or distractor.alternatives:
            shared = set(answer.ids) & set(distractor.ids)
            if shared and answer.positions == 1 and distractor.positions == 1:
                names = ", ".join(repr(backend.token_str(t)) for t in sorted(shared)[:5])
                issues.append(
                    PromptIssue(
                        index,
                        "same_answer",
                        f"the answer and the distractor share {names}, so a token would count "
                        "for both.",
                    )
                )
    n_ctx = backend.info.n_ctx
    longest = (
        max(len(clean.ids), len(corrupt.ids))
        + max(answer.positions if answer else 1, distractor.positions if distractor else 1)
        - 1
    )
    if longest > n_ctx:
        issues.append(
            PromptIssue(index, "too_long", f"the prompt is longer than the model's {n_ctx} tokens.")
        )
    labels: dict[str, int] = {}
    for label, span in (record.positions or {}).items():
        pos = label_position(clean, span)
        if pos is None:
            issues.append(
                PromptIssue(
                    index,
                    "label",
                    f"position {label!r} (characters {span[0]}–{span[1]}) doesn't cover any token.",
                )
            )
        else:
            labels[label] = pos
    if issues:
        return None, issues, clean, corrupt
    assert answer is not None and distractor is not None
    prepared = PreparedPrompt(
        index=index,
        record=record,
        clean=clean,
        corrupt=corrupt,
        answer=answer,
        distractor=distractor,
        labels=labels,
    )
    return prepared, [], clean, corrupt


def prepare_prompts(
    backend: ModelBackend,
    records: list[PromptRecord],
    prepend_bos: bool,
    continuations: bool = False,
) -> list[PreparedPrompt]:
    prepared: list[PreparedPrompt] = []
    issues: list[PromptIssue] = []
    for i, record in enumerate(records):
        p, prompt_issues, _, _ = prepare_prompt(backend, record, i, prepend_bos, continuations)
        issues.extend(prompt_issues)
        if p is not None:
            prepared.append(p)
    if issues:
        raise PromptError(issues)
    return prepared


@dataclass
class LengthGroup:
    """Prompts of one token length. Batches never mix lengths, so no padding is needed."""

    length: int
    members: list[int]  # indices into the prepared prompt list
    clean: torch.Tensor  # [n_g, length]
    corrupt: torch.Tensor


def group_by_length(prompts: list[PreparedPrompt]) -> list[LengthGroup]:
    by_len: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(prompts):
        by_len[p.length].append(i)
    groups = []
    for length in sorted(by_len):
        members = by_len[length]
        groups.append(
            LengthGroup(
                length=length,
                members=members,
                clean=torch.tensor([prompts[i].clean.ids for i in members], dtype=torch.long),
                corrupt=torch.tensor([prompts[i].corrupt.ids for i in members], dtype=torch.long),
            )
        )
    return groups


def common_labels(prompts: list[PreparedPrompt]) -> list[str]:
    """Labels present in every prompt, ordered by their mean token position."""
    if not prompts:
        return []
    shared = set(prompts[0].labels)
    for p in prompts[1:]:
        shared &= set(p.labels)
    return sorted(shared, key=lambda lab: (sum(p.labels[lab] for p in prompts) / len(prompts), lab))
