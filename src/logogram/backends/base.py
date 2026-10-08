"""The interface between experiments and a model library.

Experiments speak only in abstract sites (``resid_pre``, ``head`` ...) and tensors. A backend
maps those sites onto its own hooks. TransformerLens is the only backend in v0.1; others (for
example remote execution) can be added by implementing :class:`ModelBackend`.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

STREAM_KINDS = ("resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out")
ALL_KINDS = (*STREAM_KINDS, "head")


@dataclass(frozen=True)
class ModelInfo:
    id: str
    revision: str | None
    architecture: str
    n_layers: int
    n_heads: int
    d_model: int
    d_head: int
    d_mlp: int | None
    d_vocab: int
    n_ctx: int
    n_params: int | None
    dtype: str
    device: str  # "cpu", "cuda" or "mps"
    device_name: str
    process_weights: bool
    site_kinds: tuple[str, ...]
    backend: str
    backend_version: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["site_kinds"] = list(self.site_kinds)
        return data


@dataclass
class Tokenized:
    ids: list[int]
    tokens: list[str]
    offsets: list[tuple[int, int]]  # character span of each token; (0, 0) for added BOS


@dataclass
class Patch:
    """Replace one site's activation, row by row, during a forward pass.

    ``heads`` gives each row's head for head sites. ``positions`` gives each row's single
    position, or is ``None`` to replace every position. ``values`` is ``[B, pos, d]`` when every
    position is replaced and ``[B, d]`` otherwise (``d`` is ``d_head`` for heads).
    """

    kind: str
    layer: int
    values: torch.Tensor
    heads: torch.Tensor | None = None
    positions: torch.Tensor | None = None


class BackendError(RuntimeError):
    """A model can't be loaded or run. The message says what went wrong and how to fix it."""


class Cancelled(Exception):
    """The user cancelled the job (a run between batches, or a model load between steps)."""


class ModelBackend(ABC):
    info: ModelInfo

    def __init__(self) -> None:
        # All model use goes through this lock, one forward pass at a time, so interactive
        # requests can interleave with a running sweep between batches.
        self.lock = threading.RLock()

    @abstractmethod
    def tokenize(self, text: str, prepend_bos: bool) -> Tokenized: ...

    @abstractmethod
    def single_token_id(self, text: str) -> int | None:
        """The token id if ``text`` is exactly one token, else None."""

    @abstractmethod
    def token_str(self, token_id: int) -> str: ...

    @abstractmethod
    def final_logits(self, tokens: torch.Tensor, patch: Patch | None = None) -> torch.Tensor:
        """Logits at the last position, ``[B, vocab]`` in float32."""

    @abstractmethod
    def capture(
        self, tokens: torch.Tensor, sites: list[tuple[str, int]]
    ) -> dict[tuple[str, int], torch.Tensor]:
        """Activations for ``(kind, layer)`` sites: ``[B, pos, d]``, or ``[B, pos, H, d_head]``."""

    @abstractmethod
    def attention_pattern(self, tokens: torch.Tensor, layer: int) -> torch.Tensor:
        """Attention probabilities ``[B, H, query, key]`` in float32."""

    def edit_logits(
        self,
        tokens: torch.Tensor,
        kind: str,
        layer: int,
        edit: Any,
    ) -> torch.Tensor:
        """Logits at the last position, ``[B, vocab]``, with the activation at ``(kind, layer)``
        replaced by ``edit(activation)``: ``[B, pos, d]`` in, the same shape out. For
        residual-stream sites the edit changes the stream itself, as patching does."""
        raise BackendError("Editing activations isn't supported by this model backend.")

    def path_patch(
        self,
        tokens: torch.Tensor,
        sender: Patch,
        frozen_heads: dict[int, torch.Tensor],
        frozen_mlps: dict[int, torch.Tensor] | None,
        receivers: list[tuple[str, int, int, str]],
    ) -> torch.Tensor:
        """Logits at the last position, ``[B, vocab]``, after patching the sender's effect into
        the receivers' inputs only.

        First pass: the sender is patched, every attention head's output ``z`` is held at
        ``frozen_heads`` (the receiver run's own values, ``[B, pos, H, d_head]`` per layer) except
        the sender's, and so are MLP outputs when ``frozen_mlps`` is given; the receivers' inputs
        are recorded. Second pass: only those inputs are patched in. Receivers are
        ``("head", layer, head, "q" | "k" | "v")`` or ``("logits", -1, -1, "")``.
        """
        raise BackendError("Path patching isn't supported by this model backend.")

    def gradients(
        self,
        tokens: torch.Tensor,
        answers: torch.Tensor,
        distractors: torch.Tensor,
        sites: list[tuple[str, int]],
    ) -> tuple[dict[tuple[str, int], torch.Tensor], dict[tuple[str, int], torch.Tensor]]:
        """Activations at ``(kind, layer)`` sites and the gradient of logit(answer) -
        logit(distractor) at the last position with respect to each, in one forward and backward
        pass. Shapes as in :meth:`capture`. The gradient of a residual-stream site is with respect
        to the residual stream itself, not only to what the next component reads."""
        raise BackendError("Gradients aren't supported by this model backend.")

    def direct_effects(
        self,
        tokens: torch.Tensor,
        answers: torch.Tensor,
        distractors: torch.Tensor,
        heads: bool,
    ) -> dict[str, torch.Tensor]:
        """Direct contributions to logit(answer) - logit(distractor) at the last position.

        Returns float64 tensors on the CPU, one row per prompt: ``embed`` ``[B]``, ``attn_out``
        and ``mlp_out`` ``[B, layers]``, ``head`` ``[B, layers, heads]`` when ``heads`` is true,
        ``logit_diff`` ``[B]`` and ``remainder`` ``[B]`` (what biases add: the logit difference
        minus every component's term).
        """
        raise BackendError("Direct logit attribution isn't supported by this model backend.")

    def layer_logits(self, tokens: torch.Tensor, position: int, row: int) -> torch.Tensor:
        """Final-norm logit lens at resid_post, ``[layer, vocab]`` for one row, on CPU.

        Preserve the supplied batch composition. Backends must opt in rather than assuming
        that every model has the same normalization and vocabulary projection.
        """
        raise BackendError("Per-layer predictions aren't supported by this model backend.")

    @property
    def device(self) -> torch.device:
        return torch.device(self.info.device)

    def memory_in_use(self) -> int | None:
        """Bytes in use: accelerator memory on a GPU, this process's resident memory on a CPU."""
        dev = self.info.device
        if dev == "cuda" and torch.cuda.is_available():
            return int(torch.cuda.memory_allocated())
        if dev == "mps" and hasattr(torch, "mps"):
            try:
                return int(torch.mps.current_allocated_memory())
            except Exception:
                return None
        try:
            import psutil

            return int(psutil.Process().memory_info().rss)
        except Exception:
            return None

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release memory held by the model."""
