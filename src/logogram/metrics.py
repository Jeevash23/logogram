"""What a forward pass is measured by.

Every forward pass of a run gives three numbers per prompt:

* the spec's **metric**: the logit difference, a log-probability (difference), a probability
  (difference) or the KL divergence from a target prompt's next-token distribution;
* the **answer probability**, P(answer);
* the **preference**, log P(answer) − log P(distractor): positive when the model prefers the
  answer. For single tokens it is the logit difference, computed exactly as Logogram 0.1 did, and
  sign flips are counted from it.

An answer (or distractor) is one of:

* a single token;
* a set of single tokens (any of them counts): its probability is the sum of theirs;
* a continuation of several tokens: its log-probability is the sum over its tokens, each
  predicted from the prompt and the tokens before it (teacher forcing). The continuation is
  appended to the prompt for the forward pass. Interventions touch the prompt's positions only;
  the appended tokens are read, never patched. Batches pad continuations of different lengths at
  the end, which a causal model never reads back.

A distractor of several tokens needs a second forward pass with its own continuation appended.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from logogram.backends.base import float64

# (tokens [B, L + T - 1], keep) -> logits at the last ``keep`` positions, [B, keep, vocab], float32
Forward = Callable[[torch.Tensor, int], torch.Tensor]


@dataclass(frozen=True)
class Target:
    """An answer or a distractor as token ids: a continuation read in order, or a set of single
    tokens, any of which counts."""

    ids: tuple[int, ...]
    alternatives: bool = False

    @property
    def single(self) -> bool:
        return not self.alternatives and len(self.ids) == 1

    @property
    def positions(self) -> int:
        """How many positions reading it takes: one for a token or a set."""
        return 1 if self.alternatives else len(self.ids)


@dataclass
class Scores:
    """Per row of a batch: the metric, P(answer) and log P(answer) − log P(distractor)."""

    metric: np.ndarray
    prob: np.ndarray
    pref: np.ndarray


def _extend(tokens: torch.Tensor, conts: list[tuple[int, ...]], width: int) -> torch.Tensor:
    """Append each row's continuation (all but its last token, which is only predicted) and pad
    every row to the same width at the end, which a causal model never reads back."""
    if width <= 0:
        return tokens
    extra = torch.zeros((tokens.shape[0], width), dtype=tokens.dtype)
    for b, ids in enumerate(conts):
        head = list(ids[:-1])
        if head:
            extra[b, : len(head)] = torch.tensor(head, dtype=tokens.dtype)
    return torch.cat([tokens, extra.to(tokens.device)], dim=1)


def _conts(targets: list[Target]) -> list[tuple[int, ...]]:
    """Each row's continuation to append (a set is read at one position, so nothing)."""
    return [() if t.alternatives else t.ids for t in targets]


def _logp_of(lp: torch.Tensor, b: int, target: Target) -> torch.Tensor:
    """log P(target) for row ``b`` from log-probabilities ``lp`` [B, keep, vocab]."""
    if target.alternatives:
        return torch.logsumexp(lp[b, 0, list(target.ids)], dim=-1)
    steps = torch.arange(len(target.ids), device=lp.device)
    ids = torch.tensor(target.ids, dtype=torch.long, device=lp.device)
    return lp[b, steps, ids].sum()


class Scorer:
    """Scores forward passes with a spec's metric. ``prompts`` are the run's prepared prompts;
    rows of a batch are given as indices into them."""

    def __init__(self, metric: Any, prompts: list[Any]) -> None:
        self.metric = metric
        self.kind: str = metric.kind
        self.prompts = prompts
        # KL: the target prompts' next-token logits, by prompt index (float32, on the CPU).
        self.target_logits: dict[int, torch.Tensor] = {}

    # -- what the rows need -------------------------------------------------------------------

    @property
    def target(self) -> str | None:
        return getattr(self.metric, "target", None)

    def answer(self, row: int) -> Target:
        return self.prompts[row].answer

    def distractor(self, row: int) -> Target:
        return self.prompts[row].distractor

    def single_position(self, rows: list[int] | None = None) -> bool:
        """Whether every answer and distractor is read at the last prompt position."""
        rows = range(len(self.prompts)) if rows is None else rows
        return all(
            self.answer(r).positions == 1 and self.distractor(r).positions == 1 for r in rows
        )

    # -- scoring ------------------------------------------------------------------------------

    def score(self, forward: Forward, tokens: torch.Tensor, rows: list[int]) -> Scores:
        """Run the batch (``tokens``: the prompts, ``rows``: their prompt indices) and score it."""
        answers = [self.answer(r) for r in rows]
        distractors = [self.distractor(r) for r in rows]
        width_a = max(t.positions for t in answers)
        logits_a = forward(_extend(tokens, _conts(answers), width_a - 1), width_a)
        first = logits_a[:, 0]  # the next-token logits at the last prompt position
        B = len(rows)
        simple = all(t.single for t in answers) and all(t.single for t in distractors)
        pref = prob = None
        if simple:
            # Logogram 0.1's arithmetic, so the runs it made reproduce bit for bit.
            idx = torch.arange(B, device=first.device)
            a = torch.tensor([t.ids[0] for t in answers], dtype=torch.long, device=first.device)
            d = torch.tensor([t.ids[0] for t in distractors], dtype=torch.long, device=first.device)
            pref = float64(first[idx, a] - first[idx, d]).cpu().numpy()
            prob = float64(torch.log_softmax(first, dim=-1)[idx, a].exp()).cpu().numpy()
            if self.kind == "logit_diff":
                return Scores(metric=pref.copy(), prob=prob, pref=pref)
            if self.kind == "prob":
                return Scores(metric=prob.copy(), prob=prob, pref=pref)
        lp_a = torch.log_softmax(float64(logits_a), dim=-1)
        logp_answer = torch.stack([_logp_of(lp_a, b, t) for b, t in enumerate(answers)])
        logp_distractor = torch.zeros_like(lp_a[:, 0, 0])  # float64, where lp_a is
        later = [b for b, t in enumerate(distractors) if t.positions > 1]
        for b, t in enumerate(distractors):
            if t.positions == 1:
                logp_distractor[b] = _logp_of(lp_a, b, t)
        if later:
            width_d = max(t.positions for t in distractors)
            logits_d = forward(_extend(tokens, _conts(distractors), width_d - 1), width_d)
            lp_d = torch.log_softmax(float64(logits_d), dim=-1)
            for b in later:
                logp_distractor[b] = _logp_of(lp_d, b, distractors[b])
        logp_answer, logp_distractor = logp_answer.cpu(), logp_distractor.cpu()
        if pref is None or prob is None:
            pref = (logp_answer - logp_distractor).numpy()
            prob = logp_answer.exp().numpy()
        metric = self._metric(first, rows, logp_answer, logp_distractor)
        return Scores(metric=metric, prob=prob, pref=pref)

    def _metric(
        self,
        first: torch.Tensor,
        rows: list[int],
        logp_answer: torch.Tensor,
        logp_distractor: torch.Tensor,
    ) -> np.ndarray:
        kind = self.kind
        if kind in ("logit_diff", "logprob_diff"):
            return (logp_answer - logp_distractor).numpy()
        if kind == "logprob":
            return logp_answer.numpy().copy()
        if kind == "prob":
            return logp_answer.exp().numpy()
        if kind == "prob_diff":
            return (logp_answer.exp() - logp_distractor.exp()).numpy()
        if kind == "kl":
            return self.kl(first, rows).cpu().numpy()
        raise ValueError(f"Unknown metric {kind!r}")

    def kl(self, first: torch.Tensor, rows: list[int]) -> torch.Tensor:
        """KL(P_target || P) at the last prompt position, in float64."""
        lx = torch.log_softmax(float64(first), dim=-1)
        lt = torch.log_softmax(
            float64(torch.stack([self.target_logits[r] for r in rows]).to(first.device)), dim=-1
        ).to(lx.device)
        return (lt.exp() * (lt - lx)).sum(-1)

    # -- gradients ----------------------------------------------------------------------------

    def differentiable(self, logits: torch.Tensor, rows: list[int], part: str) -> torch.Tensor:
        """The metric's terms that one forward pass contributes, per row, with gradients: ``part``
        "answer" is the pass with the answers' continuations appended (and every single-position
        reading), "distractor" the pass with the distractors' continuations. Each metric is a sum
        of the two parts' terms."""
        B = logits.shape[0]
        first = logits[:, 0].float()
        answers = [self.answer(r) for r in rows]
        distractors = [self.distractor(r) for r in rows]
        if self.kind == "kl":
            if part != "answer":
                return torch.zeros(B, device=logits.device)
            lx = torch.log_softmax(first, dim=-1)
            lt = torch.log_softmax(
                torch.stack([self.target_logits[r] for r in rows]).to(first.device).float(), dim=-1
            )
            return (lt.exp() * (lt - lx)).sum(-1)
        if (
            self.kind == "logit_diff"
            and part == "answer"
            and all(t.single for t in answers)
            and all(t.single for t in distractors)
        ):
            idx = torch.arange(B, device=first.device)
            a = torch.tensor([t.ids[0] for t in answers], device=first.device)
            d = torch.tensor([t.ids[0] for t in distractors], device=first.device)
            return first[idx, a] - first[idx, d]
        lp = torch.log_softmax(logits.float(), dim=-1)
        out = []
        for b in range(B):
            if part == "answer":
                value = self._answer_term(lp, b, answers[b], distractors[b])
            else:
                value = self._distractor_term(lp, b, distractors[b])
            out.append(value)
        return torch.stack(out)

    def _answer_term(
        self, lp: torch.Tensor, b: int, answer: Target, distractor: Target
    ) -> torch.Tensor:
        la = _logp_of(lp, b, answer)
        ld = _logp_of(lp, b, distractor) if distractor.positions == 1 else None
        zero = lp.new_zeros(())
        if self.kind in ("logit_diff", "logprob_diff"):
            return la - (ld if ld is not None else zero)
        if self.kind == "logprob":
            return la
        if self.kind == "prob":
            return la.exp()
        if self.kind == "prob_diff":
            return la.exp() - (ld.exp() if ld is not None else zero)
        raise ValueError(f"Unknown metric {self.kind!r}")

    def _distractor_term(self, lp: torch.Tensor, b: int, distractor: Target) -> torch.Tensor:
        if distractor.positions == 1 or self.kind in ("logprob", "prob", "kl"):
            return lp.new_zeros(())
        ld = _logp_of(lp, b, distractor)
        if self.kind in ("logit_diff", "logprob_diff"):
            return -ld
        if self.kind == "prob_diff":
            return -ld.exp()
        raise ValueError(f"Unknown metric {self.kind!r}")

    def passes(self, rows: list[int]) -> list[tuple[str, list[tuple[int, ...]], int]]:
        """The forward passes a gradient of the metric needs for these rows: (part, each row's
        continuation, positions to keep)."""
        answers = [self.answer(r) for r in rows]
        distractors = [self.distractor(r) for r in rows]
        width_a = max(t.positions for t in answers)
        out = [("answer", _conts(answers), width_a)]
        if self.kind in ("logit_diff", "logprob_diff", "prob_diff") and any(
            t.positions > 1 for t in distractors
        ):
            width_d = max(t.positions for t in distractors)
            out.append(("distractor", _conts(distractors), width_d))
        return out


def extend_tokens(tokens: torch.Tensor, conts: list[tuple[int, ...]], keep: int) -> torch.Tensor:
    """``tokens`` with each row's continuation appended for a pass that keeps ``keep`` positions."""
    return _extend(tokens, conts, keep - 1)
