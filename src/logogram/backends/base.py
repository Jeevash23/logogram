"""The interface between experiments and a model library.

Experiments speak only in abstract sites (``resid_pre``, ``head`` ...) and tensors. A backend
maps those sites onto its own hooks. TransformerLens is the only backend in v0.1; others (for
example remote execution) can be added by implementing :class:`ModelBackend`.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

STREAM_KINDS = ("resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out")
ALL_KINDS = (*STREAM_KINDS, "head")
# A head's query, key and value: offered only where the backend measured, when the model loaded,
# that they are all the attention reads and that each head reads its own (see the backend's
# checks). Keys and values belong to key/value heads, fewer than query heads with grouped-query
# attention.
HEAD_INPUT_KINDS = ("q", "k", "v")
# Sites that are one unit along an axis of their layer's activation: a head of the heads' outputs
# (z), queries, keys or values (``[B, pos, heads, d_head]``), or a neuron of the MLP's hidden
# activation (``[B, pos, d_mlp]``, one number per position).
UNIT_KINDS = ("head", *HEAD_INPUT_KINDS, "neuron")

# Devices with no float64 arithmetic: Apple's Metal has no doubles.
NO_FLOAT64 = frozenset({"mps"})


def float64(t: torch.Tensor) -> torch.Tensor:
    """``t`` in float64, for statistics and decompositions: on its own device, or on the CPU when
    that device has no float64 (Apple's MPS), where converting it would fail. Every conversion to
    float64 goes through here."""
    device = torch.device("cpu") if t.device.type in NO_FLOAT64 else t.device
    return t.to(device=device, dtype=torch.float64)


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
    device: str  # "cpu", "mps", or "cuda:N": the one GPU it is on (see device_type)
    device_name: str
    process_weights: bool
    site_kinds: tuple[str, ...]
    backend: str
    backend_version: str
    extra: dict[str, Any] = field(default_factory=dict)
    # Where the weights came from: Hugging Face ("hub", pinned by the revision), or a folder on
    # this machine added to Logogram's local models ("local", pinned by sha256, the hash of its
    # files; see logogram.local_models).
    source: str = "hub"
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["site_kinds"] = list(self.site_kinds)
        return data


def device_type(device: str) -> str:
    """The kind of a device as torch names it: "cuda" for "cuda:1", "cpu", "mps"."""
    return device.split(":", 1)[0]


@dataclass
class Tokenized:
    ids: list[int]
    tokens: list[str]
    offsets: list[tuple[int, int]]  # character span of each token; (0, 0) for added BOS
    # Where the prompt begins in the text the offsets count in: after what a chat template wrote
    # before it (logogram.chat), 0 for a prompt read as it is.
    start: int = 0


@dataclass
class Patch:
    """Replace one site's activation, row by row, during a forward pass.

    ``heads`` gives each row's head for head sites (its query head for q sites, its key/value
    head for k and v sites); ``neurons`` gives each row's neuron for neuron sites. ``positions``
    gives each row's single position, or is ``None`` to replace every position. ``values`` is
    ``[B, pos, d]`` when every position is replaced and ``[B, d]`` otherwise (``d`` is ``d_head``
    for heads, queries, keys and values; a neuron is one number, so its values are ``[B, pos]``
    or ``[B]``).

    With ``mask``, the patch replaces any entries at once: ``values`` is the whole activation,
    ``[B, pos, d]`` (``[B, pos, H, d_head]`` for heads, queries, keys and values, ``[B, pos,
    d_mlp]`` for neurons), and ``mask`` (``[B, pos]``, ``[B, pos, H]`` for heads, queries, keys
    and values, ``[B, pos, d_mlp]`` for neurons) says which entries take it; ``heads``,
    ``neurons`` and ``positions`` are unused.

    A forward pass can be longer than the patch when tokens are appended to the prompt (an
    answer's continuation): every patch covers the prompt's positions only.
    """

    kind: str
    layer: int
    values: torch.Tensor
    heads: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    neurons: torch.Tensor | None = None

    def units(self) -> torch.Tensor | None:
        """Each row's unit of the activation: its head (or key/value head) for head, q, k and v
        sites, its neuron for neuron sites; None for a site that is a whole vector."""
        return self.neurons if self.kind == "neuron" else self.heads


Patches = Patch | list[Patch] | None

# score(logits [B, keep, vocab]) -> [B]: what a gradient is taken of, row by row.
ScoreFn = Callable[[torch.Tensor], torch.Tensor]

# The inputs of the model's graph (see logogram.edges): a head's query, key or value input, an
# MLP's input, and the logits.
EDGE_INPUTS = ("q", "k", "v", "mlp", "logits")


@dataclass
class InputGradients:
    """The gradient of a score with respect to each receiver's own input: the copy of the residual
    stream that only that receiver reads, before its normalization, which recomputes its scale
    from the copy as the model does.

    ``q``, ``k`` and ``v`` hold one tensor per layer, ``[B, pos, H, d_model]`` (keys and values
    per key-value head), ``mlp`` one per layer ``[B, pos, d_model]``, and ``logits`` the final
    residual stream's ``[B, pos, d_model]``. Layers that weren't asked for are None."""

    q: list[torch.Tensor | None]
    k: list[torch.Tensor | None]
    v: list[torch.Tensor | None]
    mlp: list[torch.Tensor | None]
    logits: torch.Tensor | None


class EdgePatcher:
    """What an edge-patched forward pass asks as it goes (see :meth:`ModelBackend.edge_patch`).

    ``sent`` gets each node's output as the pass computes it, in the order of the pass: the
    embeddings (``"embed"``, layer -1, ``[B, pos, d_model]``), a layer's heads (``"head"``, their
    outputs z, ``[B, pos, H, d_head]``) and a layer's MLP (``"mlp"``, ``[B, pos, d_model]``).
    ``update`` returns what to add to a receiver's own input before it is read (``"q"`` of a
    layer, ``[B, pos, H, d_model]``; ``"k"`` or ``"v"``, ``[B, pos, H_kv, d_model]``, one copy for
    each key/value head; ``"mlp"`` of a layer or ``"logits"``, ``[B, pos, d_model]``), or None to
    leave it as it is."""

    def sent(self, node: str, layer: int, output: torch.Tensor) -> None:  # pragma: no cover
        raise NotImplementedError

    def update(self, receiver: str, layer: int) -> torch.Tensor | None:  # pragma: no cover
        raise NotImplementedError


class BackendError(RuntimeError):
    """A model can't be loaded or run. The message says what went wrong and how to fix it."""


class Cancelled(Exception):
    """The user cancelled the job (a run between batches, or a model load between steps)."""


class ModelBackend(ABC):
    info: ModelInfo
    # The model's own beginning-of-sequence token, if it has one.
    bos_token_id: int | None = None

    def __init__(self) -> None:
        # All model use goes through this lock, one forward pass at a time, so interactive
        # requests can interleave with a running sweep between batches.
        self.lock = threading.RLock()

    @abstractmethod
    def tokenize(self, text: str, prepend_bos: bool) -> Tokenized: ...

    @property
    def chat_template(self) -> str | None:
        """The chat template the model's tokenizer ships (at the model's revision), as its
        tokenizer chooses it by default; None when it has none."""
        return None

    def render_chat(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        continue_final_message: bool,
    ) -> str:
        """``messages`` as text, in the model's own chat template (see :mod:`logogram.chat`)."""
        raise BackendError(
            f"{self.info.id} has no chat template. Set tokenization.chat_template to null."
        )

    @abstractmethod
    def single_token_id(self, text: str) -> int | None:
        """The token id if ``text`` is exactly one token, else None."""

    @abstractmethod
    def token_str(self, token_id: int) -> str: ...

    @abstractmethod
    def logits(
        self, tokens: torch.Tensor, patch: Patches = None, keep: int = 1, grad: bool = False
    ) -> torch.Tensor:
        """Logits at the last ``keep`` positions, ``[B, keep, vocab]`` in float32, with the patches
        applied (all at once). With ``grad``, the pass keeps its graph, so a gradient can be taken
        through the patches' values to whatever they were computed from (a learned subspace)."""

    def logits_and_reads(
        self,
        tokens: torch.Tensor,
        patch: Patches,
        keep: int,
        reads: list[tuple[str, int]],
    ) -> tuple[torch.Tensor, dict[tuple[str, int], torch.Tensor]]:
        """Logits as :meth:`logits` gives them, and the activations at ``reads`` (``(kind,
        layer)``, shaped as :meth:`capture` gives them, and ``"mlp_in"`` for the MLP's input as a
        transcoder reads it) from the same forward pass, after any patch at the same place: what a
        readout metric reads."""
        raise BackendError("Reading activations in a patched pass isn't supported by this backend.")

    def final_logits(self, tokens: torch.Tensor, patch: Patches = None) -> torch.Tensor:
        """Logits at the last position, ``[B, vocab]`` in float32."""
        return self.logits(tokens, patch, 1)[:, -1]

    @abstractmethod
    def capture(
        self, tokens: torch.Tensor, sites: list[tuple[str, int]]
    ) -> dict[tuple[str, int], torch.Tensor]:
        """Activations for ``(kind, layer)`` sites: ``[B, pos, d]``, ``[B, pos, H, d_head]`` for
        heads' outputs and queries, ``[B, pos, H_kv, d_head]`` for keys and values (``H_kv`` key/
        value heads), ``[B, pos, d_mlp]`` for neurons."""

    @abstractmethod
    def attention_pattern(self, tokens: torch.Tensor, layer: int) -> torch.Tensor:
        """Attention probabilities ``[B, H, query, key]`` in float32."""

    def attention_patterns(
        self, tokens: torch.Tensor, layers: list[int]
    ) -> dict[int, torch.Tensor]:
        """Attention probabilities ``[B, H, query, key]`` in float32 for each of ``layers``: one
        forward pass where the backend can (this default runs one per layer)."""
        return {layer: self.attention_pattern(tokens, layer) for layer in layers}

    def token_space(self) -> tuple[int, frozenset[int]]:
        """How many token ids the tokenizer has, and which of them are special (beginning or end
        of sequence, padding): what random token sequences may be drawn from."""
        return self.info.d_vocab, frozenset()

    def edit_logits(
        self,
        tokens: torch.Tensor,
        kind: str,
        layer: int,
        edit: Any,
        keep: int = 1,
        read: str | None = None,
    ) -> torch.Tensor:
        """Logits at the last ``keep`` positions, ``[B, keep, vocab]``, with the activation at
        ``(kind, layer)`` replaced by ``edit(activation)``: ``[B, pos, d]`` in (``[B, pos, H,
        d_head]`` for heads), the same shape out. For residual-stream sites the edit changes the
        stream itself, as patching does. With ``read``, the edit also gets that site's activation
        in the same layer and pass: ``edit(activation, read_activation)`` (a transcoder reads an
        MLP's input and edits its output)."""
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
        ``("head", layer, head, "q" | "k" | "v")``, ``("mlp", layer, -1, "")`` (the MLP's own
        input, before its normalization) or ``("logits", -1, -1, "")``. A sender of kind
        ``resid_pre`` at layer 0 is the embeddings.
        """
        raise BackendError("Path patching isn't supported by this model backend.")

    def edge_checks(self) -> dict[str, float]:
        """Measure, once, what edge attribution relies on, and raise BackendError when the model
        doesn't have it: that each head reads its own copy of the residual stream for its query,
        key and value (with grouped-query attention, each key/value head its own for its key and
        value), before its normalization, without changing what the model computes; that MLPs and
        the logits read theirs before their normalization; and that heads and MLPs add into the
        residual stream (measured when the model loaded). Returns the measured errors."""
        raise BackendError("Edge attribution isn't supported by this model backend.")

    def head_output_weights(self, layer: int) -> torch.Tensor:
        """``[H, d_head, d_model]`` in float64: each head's slice of the attention's output
        projection at ``layer``, which writes its output z into the residual stream."""
        raise BackendError("Reading heads' writes isn't supported by this model backend.")

    def input_gradients(
        self,
        tokens: torch.Tensor,
        score: ScoreFn,
        layers: list[int],
        logits: bool,
        embeddings: torch.Tensor | None = None,
    ) -> InputGradients:
        """The gradient of ``score`` (summed over rows, read at the last position) with respect
        to each receiver's own input in ``layers`` (and the logits', with ``logits``), in one
        forward and backward pass. ``embeddings`` replaces the residual stream entering the first
        layer, as in :meth:`gradients`."""
        raise BackendError("Edge attribution isn't supported by this model backend.")

    def edge_patch(self, tokens: torch.Tensor, patcher: EdgePatcher) -> torch.Tensor:
        """Logits at the last position, ``[B, vocab]`` in float32, from a forward pass in which
        each receiver's own input (before its normalization) gets what ``patcher`` adds, while the
        residual stream itself runs on unchanged."""
        raise BackendError("Edge patching isn't supported by this model backend.")

    def gradients(
        self,
        tokens: torch.Tensor,
        sites: list[tuple[str, int]],
        score: ScoreFn,
        keep: int = 1,
        embeddings: torch.Tensor | None = None,
    ) -> tuple[dict[tuple[str, int], torch.Tensor], dict[tuple[str, int], torch.Tensor]]:
        """Activations at ``(kind, layer)`` sites and the gradient of ``score`` (summed over rows)
        with respect to each, in one forward and backward pass. ``score`` gets the logits at the
        last ``keep`` positions. Shapes as in :meth:`capture`, over the whole (possibly extended)
        sequence. The gradient of a residual-stream site is with respect to the residual stream
        itself, not only to what the next component reads.

        ``embeddings`` (``[B, L, d_model]``) replaces the residual stream entering the first layer
        at the first ``L`` positions: integrated gradients run on interpolated inputs."""
        raise BackendError("Gradients aren't supported by this model backend.")

    def direct_effects(
        self,
        tokens: torch.Tensor,
        answers: torch.Tensor,
        distractors: torch.Tensor,
        heads: bool,
        *,
        neurons: Sequence[int] = (),
        values: Sequence[int] = (),
    ) -> dict[str, torch.Tensor]:
        """Direct contributions to logit(answer) - logit(distractor) at the last position.

        Returns float64 tensors on the CPU, one row per prompt: ``embed`` ``[B]``, ``attn_out``
        and ``mlp_out`` ``[B, layers]``, ``head`` ``[B, layers, heads]`` when ``heads`` is true,
        ``logit_diff`` ``[B]``, ``remainder`` ``[B]`` (what biases add: the logit difference
        minus every component's term) and ``logit_scale`` ``[B]`` (the larger magnitude of the
        answer's and the distractor's logit, which sets how finely the model's dtype resolves
        them).

        For the layers in ``neurons``, ``neuron`` ``[B, len(neurons), d_mlp]``: each neuron's
        activation at the last position times its row of the MLP's output projection. For the
        layers in ``values``, ``value`` ``[B, len(values), H_kv, pos]``: what each key/value
        head's value at each position adds to the last position's attention output, through
        every query head that reads it (its attention to that position times the value through
        the head's share of the output projection; the pattern doesn't depend on the values).
        Both are read like the heads' terms, and only where the backend measured that they add up.
        """
        raise BackendError("Direct logit attribution isn't supported by this model backend.")

    def unembed_directions(self, residual: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        """How much each of ``vectors`` (``[k, d_model]``) changes every logit when added to
        ``residual`` (``[d_model]``, the residual stream entering the final normalization at one
        token): read through the final normalization, with its scale held at its value for
        ``residual``, and the unembedding, before any soft cap on the logits. With the scale held
        both are affine, so this is exactly linear in each vector, as in direct logit
        attribution. ``[k, vocab]`` in float32, on the CPU.

        Backends must opt in: the final normalization and unembedding have to reproduce the
        model's output (checked when it loads)."""
        raise BackendError("Reading directions into the logits isn't supported by this backend.")

    def head_writes(self, layer: int, z: torch.Tensor) -> torch.Tensor:
        """What attention heads' outputs ``z`` (``[k, H, d_head]``) at ``layer`` write into the
        residual stream: each head's through its slice of the attention's output projection,
        summed, without the projection's bias. ``[k, d_model]`` in float64, on the CPU. This is
        what the attention adds only where its output is the sum of its heads' writes, which
        check_model measures when a model loads (``checks.heads``)."""
        raise BackendError("Reading heads' writes isn't supported by this model backend.")

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
        if device_type(dev) == "cuda" and torch.cuda.is_available():
            return int(torch.cuda.memory_allocated(torch.device(dev)))
        if dev == "mps" and hasattr(torch, "mps"):
            try:
                return int(torch.mps.current_allocated_memory())
            except Exception:  # noqa: BLE001 - a memory readout is best effort
                return None
        try:
            import psutil

            return int(psutil.Process().memory_info().rss)
        except Exception:  # noqa: BLE001 - a memory readout is best effort
            return None

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release memory held by the model."""
