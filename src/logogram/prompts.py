"""Tokenize prompt pairs for a model and check that they can be used in an experiment.

A usable pair has clean and corrupt prompts of the same token length (so positions line up), and
an answer and distractor that are each a single token.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch

from logogram.backends.base import ModelBackend, Tokenized
from logogram.datasets import PromptRecord


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
    answer_id: int
    distractor_id: int
    labels: dict[str, int] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.clean.ids)

    def differing_positions(self) -> list[int]:
        return [
            i
            for i, (a, b) in enumerate(zip(self.clean.ids, self.corrupt.ids, strict=True))
            if a != b
        ]


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


def prepare_prompt(
    backend: ModelBackend, record: PromptRecord, index: int, prepend_bos: bool
) -> tuple[PreparedPrompt | None, list[PromptIssue], Tokenized, Tokenized]:
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
    answer_id = backend.single_token_id(record.answer)
    if answer_id is None:
        issues.append(
            PromptIssue(
                index,
                "answer_tokens",
                f"the answer {record.answer!r} is several tokens "
                f"({_show_tokens(backend, record.answer)}). Use a single-token answer, often "
                "with a leading space.",
            )
        )
    distractor_id = backend.single_token_id(record.distractor)
    if distractor_id is None:
        issues.append(
            PromptIssue(
                index,
                "distractor_tokens",
                f"the distractor {record.distractor!r} is several tokens "
                f"({_show_tokens(backend, record.distractor)}). Use a single-token distractor.",
            )
        )
    if clean.ids == corrupt.ids:
        issues.append(
            PromptIssue(
                index,
                "identical",
                "the clean and corrupt prompts are the same, so there is nothing to patch.",
            )
        )
    if answer_id is not None and answer_id == distractor_id:
        issues.append(
            PromptIssue(index, "same_answer", "the answer and distractor are the same token.")
        )
    n_ctx = backend.info.n_ctx
    if max(len(clean.ids), len(corrupt.ids)) > n_ctx:
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
    assert answer_id is not None and distractor_id is not None
    prepared = PreparedPrompt(
        index=index,
        record=record,
        clean=clean,
        corrupt=corrupt,
        answer_id=answer_id,
        distractor_id=distractor_id,
        labels=labels,
    )
    return prepared, [], clean, corrupt


def prepare_prompts(
    backend: ModelBackend, records: list[PromptRecord], prepend_bos: bool
) -> list[PreparedPrompt]:
    prepared: list[PreparedPrompt] = []
    issues: list[PromptIssue] = []
    for i, record in enumerate(records):
        p, prompt_issues, _, _ = prepare_prompt(backend, record, i, prepend_bos)
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
