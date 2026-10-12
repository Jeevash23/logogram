"""Toy models whose circuits are known: small transformers whose weights are set by hand, so that
a planted mechanism does the task. Every method can be checked against them, and people can learn
the methods on a model whose answer is known.

The toys
--------

``induction``: two layers of four heads, attention only. The prompts are random words in which
two pairs appear, "A B" and "F C", and the last word is A again::

    clean:   … A B … F C … A   → " B"
    corrupt: … F B … A C … A   → " C"   (A and F swap places)

* Layer 0, head 2 is a **previous-token head**: every token attends to the token before it and
  copies that token's identity into a "previous token" subspace.
* Layer 1, head 1 is an **induction head**: the last token's query (its own identity) matches the
  key of the token whose previous token is the same word (K-composition with the previous-token
  head), and its value copies that token's identity into the logits.
* The other six heads attend to the first token, to themselves or evenly to every token, and write
  into dimensions nothing reads: they do nothing that matters.

``ioi``: four layers of four heads, attention only. It completes indirect-object prompts made by
Logogram's own generator (:mod:`logogram.ioi`, templates "reached" and "arrived")::

    clean:   Once Mary and John reached the store, John threw a drink to   → " Mary"
    corrupt: Once Mary and John reached the store, Mary threw a drink to   → " John"

with the circuit GPT-2 small uses (Wang et al. 2022), cut down to one head per step:

* L0 H2, a **previous-token head**, copies each token's identity, and which name slot it is in
  (the first or second name), into the next token.
* L1 H1, an **induction head**: at the second mention of the subject (S2) it attends to the token
  after the first mention (S1 + 1), whose previous token matches, and copies which slot S1 was in
  (V-composition with the previous-token head's slot), with a flag that it found a repeat.
* L2 H3, an **S-inhibition head**: at the last token it attends to the flagged token (S2) and
  copies the slot into an "inhibit" subspace (V-composition with the induction head).
* L3 H0, a **name mover**: at the last token it attends to names; the query reads "inhibit"
  (Q-composition with the S-inhibition head) and turns away from the subject's slot, so it attends
  to the indirect object and copies its name into the logits.
* The other twelve heads do nothing that matters, as in the induction toy.

Because S-inhibition carries the subject's *position*, not its name, the difference between a
clean and a flipped prompt is one direction: on prompts of one order (all ABBA) the mean
difference steers the name mover, while on prompts that mix ABBA and BABA the differences cancel,
as in GPT-2 small.

How the weights make it so
--------------------------

Both toys are GPT-2 checkpoints (256 dimensions, heads of 64, an MLP of 8 neurons whose weights
are zero, so its output is zero). Every direction of the residual stream is one named feature:
one dimension each, the token's identity (one per word), the previous token's, the output's,
the position p/128 and its square, flags. Dimension 0 holds minus the sum of the others, so every
vector written has mean zero and the layer norms' centering changes nothing.

The layer norms' epsilon is 2^20, far above the variance of any residual stream (a few units), so
each normalization divides by 2^10 to within a part in a million, and their weights multiply by
2^10: they leave the residual stream as it is. The circuit's arithmetic is then exact, up to the
saturation of the attention softmax, which is set by the scores below.

An attention score is a sum of products of a query feature (or a constant) and a key feature,
each pair on its own dimension of the head. Matching scores are 40, the first token (a rest
position) scores 20, so a head that finds no match rests there; positional heads score
-20·(p' - target)², so the target position wins by 20. Every planted head's attention to its
target is 1 to within 10^-8. A copied token writes 10 into its own logit.

What each head reads from which other is recorded as data (:class:`Toy`'s ``heads`` and
``compositions``), for tests of methods that find heads and the edges between them.

The files
---------

Each toy's files (config.json, tokenizer.json, tokenizer_config.json, model.safetensors) are
written byte for byte by this module, without randomness or a library's serialization: every
weight is a small integer or a binary fraction, exact in float32. A toy's files therefore hash the
same on every machine (:mod:`logogram.local_models`), and specs name a toy by that hash
(``model.source: "local"``). Loading a model whose files hash as a toy's, wherever it loads (the
app, ``logogram run``, the Python API), also checks that it solves its prompts and that its
planted heads attend where they were built to (:func:`check_loaded_toy`).

``logogram toy NAME [FOLDER]`` and ``logogram.toy_model(name)`` write a toy's project: the model's
files, its datasets and experiments whose notes say what each method should find
(:func:`write_toy`).
"""

from __future__ import annotations

import json
import random
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from logogram.datasets import PromptRecord

if TYPE_CHECKING:
    from logogram.backends.base import ModelBackend
    from logogram.project import Project
    from logogram.spec import Spec


class ToyError(ValueError):
    """A toy can't be made, or doesn't behave as designed. The message says why."""


# -- the shapes and scores shared by every toy -------------------------------------------------

N_CTX = 128
D_MODEL = 256
N_HEADS = 4
D_HEAD = 64
D_MLP = 8
# The layer norms' epsilon, and their weights (its square root): with the variance of every
# residual stream a few units, each normalization divides by about 2^10, and its weight undoes it.
LN_EPS = 2.0**20
LN_WEIGHT = 2.0**10
SCALE = 8.0  # GPT-2 divides attention scores by the square root of the head's size

# Attention scores, in nats.
MATCH = 40.0  # a key that matches the query
REST = 20.0  # the first token, where a head rests when no key matches
SHARP = 20.0  # how far a positional head's target beats the positions next to it
NAMES = 10.0  # a name, for the name mover
TURN = 20.0  # how far the S-inhibition signal turns the name mover from a name slot
OUT = 10.0  # the logit a copied token writes

BOS = "<|endoftext|>"
UNK = "<unk>"
JUNK = 4  # dimensions the other heads write into, which nothing reads

ToyName = Literal["induction", "ioi"]
Corruption = Literal["key", "value", "query"]


def _token(word: str) -> str:
    """A word as the toys' tokenizer spells it (byte-level, a leading space marked Ġ)."""
    return word if not word[0].isalnum() else "Ġ" + word


# -- the planted circuit, as data ---------------------------------------------------------------


@dataclass(frozen=True)
class PlantedHead:
    """A head that does part of the task: ``kind`` is what it is (as the head finder names
    previous-token and induction heads), ``does`` what its weights make it do."""

    layer: int
    head: int
    kind: Literal["previous_token", "induction", "s_inhibition", "name_mover"]
    does: str

    @property
    def label(self) -> str:
        return f"L{self.layer} H{self.head}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "head": self.head,
            "kind": self.kind,
            "label": self.label,
            "does": self.does,
        }


@dataclass(frozen=True)
class Composition:
    """An edge of the planted circuit: what ``sender`` writes (a head as (layer, head), or None
    for the embeddings of tokens and positions) is read by ``receiver``'s ``input`` ("q", "k" or
    "v"), or by the logits (``receiver`` None). ``changes`` says whether the clean and corrupt
    prompts of the toy's dataset differ along it where the receiver reads it: patching such an
    edge alone moves the answer all the way (normalized effect 1), any other edge not at all."""

    sender: tuple[int, int] | None
    receiver: tuple[int, int] | None
    input: Literal["q", "k", "v"] | None
    carries: str
    changes: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender": None if self.sender is None else list(self.sender),
            "receiver": None if self.receiver is None else list(self.receiver),
            "input": self.input,
            "carries": self.carries,
            "changes": self.changes,
        }


@dataclass(frozen=True)
class Toy:
    """A toy model: its planted heads and compositions, the words it knows, and the hash of its
    files (``sha256``), which a spec names it by."""

    name: ToyName
    title: str
    summary: str
    n_layers: int
    words: tuple[str, ...]
    names: tuple[str, ...]
    heads: tuple[PlantedHead, ...]
    compositions: tuple[Composition, ...]
    # Every other head, as (layer, head, what it does): "rest" on the first token, attend to
    # "self", or attend "even"ly to every token; each writes what it reads where nothing reads.
    idle: tuple[tuple[int, int, Literal["rest", "self", "even"]], ...]
    sha256: str
    # The dataset `logogram toy` writes: its file name and size.
    dataset: str
    n_prompts: int

    @property
    def model_name(self) -> str:
        """The name the toy is added under in the local models."""
        return f"toy-{self.name}"

    def head(self, kind: str) -> PlantedHead:
        return next(h for h in self.heads if h.kind == kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "summary": self.summary,
            "model_name": self.model_name,
            "n_layers": self.n_layers,
            "n_heads": N_HEADS,
            "heads": [h.to_dict() for h in self.heads],
            "compositions": [c.to_dict() for c in self.compositions],
            "others": OTHER_HEADS,
            "sha256": self.sha256,
        }


INDUCTION_WORDS: tuple[str, ...] = (
    "apple", "river", "stone", "cloud", "bread", "chair", "lamp", "tiger", "ocean", "pencil",
    "garden", "window", "candle", "forest", "rabbit", "silver", "button", "castle", "dragon",
    "feather", "guitar", "harbor", "island", "jacket", "kettle", "lemon", "marble", "needle",
    "orange", "pepper", "rocket", "saddle", "tunnel", "velvet", "wagon", "yellow", "zebra",
    "anchor", "basket", "copper", "desert", "engine", "falcon", "glacier", "hammer", "insect",
    "jungle", "ladder",
)  # fmt: skip

# Names, places and objects of logogram.ioi, and the words of its templates "reached" and
# "arrived", whose names sit at the same two token positions (2 and 4, after the first token).
IOI_NAMES: tuple[str, ...] = (
    "Mary", "John", "Alice", "Bob", "Tom", "James", "Sarah", "Emma", "David", "Michael",
    "Daniel", "Laura", "Anna", "Kate", "Paul", "Lisa", "Peter", "Jack", "Rachel", "Eric",
)  # fmt: skip
IOI_TEMPLATES = ("reached", "arrived")
IOI_WORDS: tuple[str, ...] = (
    "Once", "After", "and", "reached", "arrived", "at", "the", ",", "threw", "handed", "a", "to",
    *IOI_NAMES,
    "store", "park", "school", "station", "beach", "office", "market", "library",
    "drink", "book", "ring", "bag", "letter", "gift", "card", "key",
)  # fmt: skip
# Where the two names of an IOI prompt sit: after the first token and its first word.
NAME_SLOTS = {2: 1.0, 4: -1.0}

OTHER_HEADS = (
    "The other heads attend to the first token, to the token itself, or evenly to every token, "
    "and write what they read into four dimensions that no head and no logit reads."
)

TOYS: dict[str, Toy] = {
    "induction": Toy(
        name="induction",
        title="Induction toy",
        summary=(
            "Two layers of four heads, attention only. L0 H2 is a previous-token head and L1 H1 "
            "an induction head: in “… A B … F C … A” the last A attends to B, the token after "
            "the earlier A, and predicts it. The corrupt prompt swaps A and F, so it predicts C."
        ),
        n_layers=2,
        words=INDUCTION_WORDS,
        names=(),
        heads=(
            PlantedHead(
                0,
                2,
                "previous_token",
                "Attends to the token before each token and copies its identity into the "
                "previous-token subspace.",
            ),
            PlantedHead(
                1,
                1,
                "induction",
                "Its query (the token's identity) matches the key of the token whose previous "
                "token is the same word; its value copies that token into the logits (+10).",
            ),
        ),
        compositions=(
            Composition(None, (0, 2), "q", "each token's position", False),
            Composition(None, (0, 2), "k", "each token's position", False),
            Composition(None, (0, 2), "v", "each token's identity", True),
            Composition(None, (1, 1), "q", "the last token's identity", False),
            Composition((0, 2), (1, 1), "k", "which word came before each token", True),
            Composition(None, (1, 1), "k", "the first token, where it rests", False),
            Composition(None, (1, 1), "v", "the identity of the token attended to", False),
            Composition((1, 1), None, None, "the predicted token", True),
        ),
        idle=(
            (0, 0, "even"),
            (0, 1, "rest"),
            (0, 3, "self"),
            (1, 0, "rest"),
            (1, 2, "even"),
            (1, 3, "self"),
        ),
        sha256="9bf8b10f6ccadcc6484cb708fc5bfeba25eaa90b23e0744bbf1ea23fe93a832f",
        dataset="induction.jsonl",
        n_prompts=64,
    ),
    "ioi": Toy(
        name="ioi",
        title="IOI toy",
        summary=(
            "Four layers of four heads, attention only, with GPT-2 small's indirect-object "
            "circuit cut down to one head per step: a previous-token head (L0 H2), an induction "
            "head (L1 H1) that finds the repeated name, an S-inhibition head (L2 H3) that moves "
            "which name slot repeats to the last token, and a name mover (L3 H0) that copies the "
            "other name."
        ),
        n_layers=4,
        words=IOI_WORDS,
        names=IOI_NAMES,
        heads=(
            PlantedHead(
                0,
                2,
                "previous_token",
                "Attends to the token before each token and copies its identity, and its name "
                "slot (first +1, second −1), into the previous-token subspace.",
            ),
            PlantedHead(
                1,
                1,
                "induction",
                "At the repeated name (S2) it attends to the token after the name's first "
                "mention and copies that name's slot, with a flag that it found a repeat.",
            ),
            PlantedHead(
                2,
                3,
                "s_inhibition",
                "At the last token it attends to the flagged token (S2) and copies the repeated "
                "name's slot into the inhibit subspace.",
            ),
            PlantedHead(
                3,
                0,
                "name_mover",
                "At the last token it attends to names, turned away from the slot the inhibit "
                "subspace names, and copies the name it attends to into the logits (+10).",
            ),
        ),
        compositions=(
            Composition(None, (0, 2), "q", "each token's position", False),
            Composition(None, (0, 2), "k", "each token's position", False),
            Composition(None, (0, 2), "v", "each token's identity and name slot", False),
            Composition(None, (1, 1), "q", "the repeated name's identity", True),
            Composition((0, 2), (1, 1), "k", "which word came before each token", False),
            Composition(None, (1, 1), "k", "the first token, where it rests", False),
            Composition((0, 2), (1, 1), "v", "the name slot of the word before a token", False),
            Composition(None, (1, 1), "v", "that a token is a word: a repeat was found", False),
            Composition((1, 1), (2, 3), "k", "the flag: a repeated name is here", False),
            Composition(None, (2, 3), "k", "the first token, where it rests", False),
            Composition((1, 1), (2, 3), "v", "which slot the repeated name is in", True),
            Composition((2, 3), (3, 0), "q", "which name slot to turn away from", True),
            Composition(None, (3, 0), "k", "which tokens are names, and their slots", False),
            Composition(None, (3, 0), "v", "the names' identities", False),
            Composition((3, 0), None, None, "the indirect object's name", True),
        ),
        idle=(
            (0, 0, "even"),
            (0, 1, "rest"),
            (0, 3, "self"),
            (1, 0, "rest"),
            (1, 2, "even"),
            (1, 3, "self"),
            (2, 0, "rest"),
            (2, 1, "self"),
            (2, 2, "even"),
            (3, 1, "rest"),
            (3, 2, "self"),
            (3, 3, "even"),
        ),
        sha256="534bbfb40fc1fbd680d3b5de253e427b629df210ff16c6bfa10e43a502ac98f5",
        dataset="ioi.jsonl",
        n_prompts=64,
    ),
}


def get_toy(name: str) -> Toy:
    """The toy called ``name``, or a ToyError naming the toys there are."""
    toy = TOYS.get(name) if isinstance(name, str) else None
    if toy is None:
        raise ToyError(f"There is no toy model called {name!r}. Choose one of: {', '.join(TOYS)}.")
    return toy


# -- the weights ---------------------------------------------------------------------------------


class _Builder:
    """A GPT-2 checkpoint built feature by feature. Each feature is one dimension of the residual
    stream, in the order :meth:`layout` gives them; dimension 0 takes minus everything written,
    so every vector written has mean zero. Each head's query/key dimensions and value/output
    dimensions are handed out in turn: a score term, or a feature moved, uses one of its 64."""

    def __init__(self, toy: Toy) -> None:
        self.toy = toy
        self.vocab = {BOS: 0, UNK: 1} | {_token(w): i + 2 for i, w in enumerate(toy.words)}
        self.features = {name: i + 1 for i, name in enumerate(self.layout())}
        if len(self.features) >= D_MODEL:
            raise ToyError(f"The {toy.name} toy needs more than {D_MODEL - 1} features.")
        n, vocab = toy.n_layers, len(self.vocab)
        self.wte = np.zeros((vocab, D_MODEL))
        self.wpe = np.zeros((N_CTX, D_MODEL))
        self.qkv = np.zeros((n, D_MODEL, 3 * D_MODEL))
        self.qkv_bias = np.zeros((n, 3 * D_MODEL))
        self.proj = np.zeros((n, D_MODEL, D_MODEL))
        self.unembed = np.zeros((vocab, D_MODEL))
        self.used: dict[tuple[int, int, str], int] = {}

    def layout(self) -> list[str]:
        """The features, in the order of their dimensions (from 1)."""
        words = self.toy.words
        return [
            "bos",  # the first token: where heads rest
            "content",  # every word
            "u",  # the position p, as p/128
            "u2",  # its square
            *([f"tok:{w}" for w in words]),  # the token's identity
            *([f"prev:{w}" for w in words]),  # the identity of the token before it
            *([f"out:{w}" for w in words]),  # the token to predict: what the logits read
            *[f"junk{k}" for k in range(JUNK)],  # what the other heads write
            *(  # the IOI circuit's own features
                ["name", "slot", "prev_slot", "repeat", "repeat_slot", "inhibit"]
                if self.toy.names
                else []
            ),
        ]

    def write(self, row: np.ndarray, name: str, value: float) -> None:
        row[self.features[name]] += value
        row[0] -= value

    def _dim(self, layer: int, head: int, part: str) -> int:
        j = self.used.get((layer, head, part), 0)
        if j >= D_HEAD:
            raise ToyError(f"L{layer} H{head} needs more than {D_HEAD} dimensions.")
        self.used[(layer, head, part)] = j + 1
        return head * D_HEAD + j

    def attend(
        self, layer: int, head: int, query: dict[str | None, float], key: dict[str, float]
    ) -> None:
        """Add (Σ query features × coefficients, None a constant) × (Σ key features ×
        coefficients) to the head's attention scores."""
        col = self._dim(layer, head, "qk")
        for name, c in query.items():
            if name is None:
                self.qkv_bias[layer, col] += c * SCALE
            else:
                self.qkv[layer, self.features[name], col] += c * SCALE
        for name, c in key.items():
            self.qkv[layer, self.features[name], D_MODEL + col] += c

    def move(self, layer: int, head: int, source: str, targets: dict[str, float]) -> None:
        """Write the ``source`` feature of the tokens the head attends to into ``targets``, each
        times its weight."""
        col = self._dim(layer, head, "ov")
        self.qkv[layer, self.features[source], 2 * D_MODEL + col] = 1.0
        for target, weight in targets.items():
            self.write(self.proj[layer, col], target, weight)

    # -- what every toy has ------------------------------------------------------------------

    def embed(self) -> None:
        """Tokens write their identity (the first token its rest flag, names a name flag);
        positions write p/128 and its square, and an IOI prompt's two name slots their sign; the
        logits read the out features."""
        self.write(self.wte[0], "bos", 1.0)
        for word in self.toy.words:
            index = self.vocab[_token(word)]
            self.write(self.wte[index], "content", 1.0)
            self.write(self.wte[index], f"tok:{word}", 1.0)
            if word in self.toy.names:
                self.write(self.wte[index], "name", 1.0)
            self.unembed[index, self.features[f"out:{word}"]] = 1.0
        for p in range(N_CTX):
            u = p / N_CTX
            self.write(self.wpe[p], "u", u)
            self.write(self.wpe[p], "u2", u * u)
            if self.toy.names and p in NAME_SLOTS:
                self.write(self.wpe[p], "slot", NAME_SLOTS[p])

    def positional(self, layer: int, head: int, back: int) -> None:
        """Attend to the token ``back`` positions before (0: itself): the score is
        -SHARP·(p' - p + back)² plus a constant for each query, so the target wins by SHARP."""
        c = SHARP * N_CTX**2
        self.attend(layer, head, {"u": 2 * c, None: -2 * c * back / N_CTX}, {"u": 1.0})
        self.attend(layer, head, {None: -c}, {"u2": 1.0})

    def rest(self, layer: int, head: int) -> None:
        """Score the first token REST, where the head rests when nothing scores more."""
        self.attend(layer, head, {None: REST}, {"bos": 1.0})

    def idle(self, layer: int, head: int, how: str) -> None:
        """A head that does nothing that matters: it rests on the first token, attends to
        itself, or attends evenly to every token, and writes the identity of what it attends to
        into the junk dimensions, which nothing reads."""
        if how == "rest":
            self.rest(layer, head)
        elif how == "self":
            self.positional(layer, head, 0)
        for i, word in enumerate(self.toy.words):
            signs = {f"junk{k}": 0.25 if (i >> k) & 1 else -0.25 for k in range(JUNK)}
            self.move(layer, head, f"tok:{word}", signs)

    # -- the planted heads -----------------------------------------------------------------

    def previous_token(self, layer: int, head: int) -> None:
        """Attend to the token before; copy its identity (and its name slot) forward."""
        self.positional(layer, head, 1)
        for word in self.toy.words:
            self.move(layer, head, f"tok:{word}", {f"prev:{word}": 1.0})
        if self.toy.names:
            self.move(layer, head, "slot", {"prev_slot": 1.0})

    def induction(self, layer: int, head: int) -> None:
        """Match the query's identity against the key's previous token, resting on the first
        token when nothing matches; copy the identity of the token found into the logits. In the
        IOI toy, also copy the name slot of the token before it (the first mention's), and that
        it found a word (the first token has no content feature)."""
        for word in self.toy.words:
            self.attend(layer, head, {f"tok:{word}": MATCH}, {f"prev:{word}": 1.0})
        self.rest(layer, head)
        for word in self.toy.words:
            self.move(layer, head, f"tok:{word}", {f"out:{word}": OUT})
        if self.toy.names:
            self.move(layer, head, "prev_slot", {"repeat_slot": 1.0})
            self.move(layer, head, "content", {"repeat": 1.0})

    def s_inhibition(self, layer: int, head: int) -> None:
        """Attend to where a repeat was found (resting on the first token if none was); copy
        which name slot repeats into inhibit."""
        self.attend(layer, head, {None: MATCH}, {"repeat": 1.0})
        self.rest(layer, head)
        self.move(layer, head, "repeat_slot", {"inhibit": 1.0})

    def name_mover(self, layer: int, head: int) -> None:
        """Attend to names (NAMES), away from the name slot inhibit says repeats (TURN); copy the
        name attended to into the logits."""
        self.attend(layer, head, {None: NAMES}, {"name": 1.0})
        self.attend(layer, head, {"inhibit": -TURN}, {"slot": 1.0})
        for word in self.toy.names:
            self.move(layer, head, f"tok:{word}", {f"out:{word}": OUT})

    # -- the files --------------------------------------------------------------------------

    def tensors(self) -> dict[str, np.ndarray]:
        ln = np.full(D_MODEL, LN_WEIGHT)
        zeros = np.zeros(D_MODEL)
        out: dict[str, np.ndarray] = {
            "transformer.wte.weight": self.wte,
            "transformer.wpe.weight": self.wpe,
            "transformer.ln_f.weight": ln,
            "transformer.ln_f.bias": zeros,
            "lm_head.weight": self.unembed,
        }
        for i in range(self.toy.n_layers):
            p = f"transformer.h.{i}."
            out |= {
                p + "ln_1.weight": ln,
                p + "ln_1.bias": zeros,
                p + "attn.c_attn.weight": self.qkv[i],
                p + "attn.c_attn.bias": self.qkv_bias[i],
                p + "attn.c_proj.weight": self.proj[i],
                p + "attn.c_proj.bias": zeros,
                p + "ln_2.weight": ln,
                p + "ln_2.bias": zeros,
                # The MLP writes nothing: the toys are attention only.
                p + "mlp.c_fc.weight": np.zeros((D_MODEL, D_MLP)),
                p + "mlp.c_fc.bias": np.zeros(D_MLP),
                p + "mlp.c_proj.weight": np.zeros((D_MLP, D_MODEL)),
                p + "mlp.c_proj.bias": zeros,
            }
        return out


def _build(toy: Toy) -> _Builder:
    """A toy's weights: the embeddings, each planted head as its kind says, every other head
    idle as the toy says. Every head of every layer is one or the other."""
    b = _Builder(toy)
    b.embed()
    planted = {(h.layer, h.head): h for h in toy.heads}
    idle = {(layer, head): how for layer, head, how in toy.idle}
    every = {(layer, head) for layer in range(toy.n_layers) for head in range(N_HEADS)}
    if set(planted) | set(idle) != every or set(planted) & set(idle):
        raise ToyError(f"The {toy.name} toy must give every head one job.")
    for layer, head in sorted(every):
        if (layer, head) in planted:
            getattr(b, planted[(layer, head)].kind)(layer, head)
        else:
            b.idle(layer, head, idle[(layer, head)])
    return b


def _json(data: Any) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _safetensors(tensors: dict[str, np.ndarray]) -> bytes:
    """The safetensors file of ``tensors`` in float32, laid out by name: an 8-byte header size,
    the JSON header (padded with spaces to 8 bytes) and the little-endian data."""
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    blobs: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        array = tensors[name]
        data = np.ascontiguousarray(array, dtype="<f4").tobytes()
        if not np.array_equal(np.frombuffer(data, dtype="<f4").reshape(array.shape), array):
            raise ToyError(f"{name} isn't exact in float32, so the file wouldn't be either.")
        header[name] = {
            "dtype": "F32",
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    text = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    text += b" " * (-len(text) % 8)
    return struct.pack("<Q", len(text)) + text + b"".join(blobs)


def _config(toy: Toy, vocab: int) -> dict[str, Any]:
    return {
        "activation_function": "gelu_new",
        "architectures": ["GPT2LMHeadModel"],
        "attn_pdrop": 0.0,
        "bos_token_id": 0,
        "embd_pdrop": 0.0,
        "eos_token_id": 0,
        "initializer_range": 0.02,
        "layer_norm_epsilon": LN_EPS,
        "model_type": "gpt2",
        "n_ctx": N_CTX,
        "n_embd": D_MODEL,
        "n_head": N_HEADS,
        "n_inner": D_MLP,
        "n_layer": toy.n_layers,
        "n_positions": N_CTX,
        "pad_token_id": 0,
        "reorder_and_upcast_attn": False,
        "resid_pdrop": 0.0,
        "scale_attn_by_inverse_layer_idx": False,
        "scale_attn_weights": True,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": vocab,
    }


def _tokenizer(vocab: dict[str, int]) -> dict[str, Any]:
    """A word-level tokenizer over the toy's words, byte-level like GPT-2's: every word, with a
    space before it or at the start of the text, is one token."""
    byte_level = {
        "type": "ByteLevel",
        "add_prefix_space": True,
        "trim_offsets": True,
        "use_regex": True,
    }
    special = [
        {
            "id": index,
            "content": text,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for text, index in ((BOS, 0), (UNK, 1))
    ]
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": special,
        "normalizer": None,
        "pre_tokenizer": byte_level,
        "post_processor": None,
        "decoder": byte_level,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": UNK},
    }


def model_files(name: str) -> dict[str, bytes]:
    """A toy's files, by name: the same bytes on every machine."""
    toy = get_toy(name)
    builder = _build(toy)
    tokenizer_config = {
        "bos_token": BOS,
        "clean_up_tokenization_spaces": False,
        "eos_token": BOS,
        "model_max_length": N_CTX,
        "pad_token": BOS,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": UNK,
    }
    return {
        "config.json": _json(_config(toy, len(builder.vocab))),
        "model.safetensors": _safetensors(builder.tensors()),
        "tokenizer.json": _json(_tokenizer(builder.vocab)),
        "tokenizer_config.json": _json(tokenizer_config),
    }


def files_sha256(files: dict[str, bytes]) -> str:
    """The hash :mod:`logogram.local_models` gives a folder holding exactly ``files``."""
    import hashlib

    lines = "".join(
        f"{hashlib.sha256(files[name]).hexdigest()}  {name}\n" for name in sorted(files)
    )
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def write_model(name: str, folder: str | Path) -> Path:
    """Write a toy's files into ``folder`` (created if needed), replacing any of the same name
    whose bytes differ, and check that they hash to the toy's recorded hash. Returns the folder."""
    from logogram.fileio import atomic_output

    toy = get_toy(name)
    files = model_files(toy.name)
    sha = files_sha256(files)
    if toy.sha256 and sha != toy.sha256:
        raise ToyError(
            f"The {toy.name} toy's files hash to {sha[:12]}… here, not to {toy.sha256[:12]}… "
            "as recorded: building them isn't exact on this machine. Please report it."
        )
    target = Path(folder).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    for file_name, data in files.items():
        path = target / file_name
        if path.is_file() and not path.is_symlink() and path.read_bytes() == data:
            continue
        with atomic_output(path) as fh:
            fh.write(data)
    return target


# -- the datasets ----------------------------------------------------------------------------


def _pick(rng: random.Random, n: int) -> int:
    return int(rng.random() * n)


def _sample(rng: random.Random, items: Sequence[str], k: int) -> list[str]:
    """``k`` distinct items in the order drawn (a partial Fisher–Yates shuffle)."""
    pool = list(items)
    for i in range(k):
        j = i + _pick(rng, len(pool) - i)
        pool[i], pool[j] = pool[j], pool[i]
    return pool[:k]


def induction_prompts(
    n: int, seed: int = 0, *, length: int = 16, corruption: Corruption = "key"
) -> list[PromptRecord]:
    """``n`` prompt pairs of ``length`` distinct random words, in which "A B" and "F C" appear
    in a random order and the last word is A again, so the answer is " B" and the distractor
    " C". The corrupt prompt changes what the induction head reads through one of its inputs:

    * ``key``: A and F swap places (… F B … A C … A), so the token after the earlier A is C;
    * ``value``: B and C swap places (… A C … F B … A);
    * ``query``: the last word is F instead of A (… A B … F C … F).

    Labels (character spans in the clean prompt): ``A`` and ``B`` (the first pair), ``F`` and
    ``C`` (the second) and ``end``. Randomness uses only ``random.Random.random()``."""
    if not 6 <= length <= len(INDUCTION_WORDS) + 1:
        raise ToyError(f"Choose a length from 6 to {len(INDUCTION_WORDS) + 1} words.")
    if corruption not in ("key", "value", "query"):
        raise ToyError(f"Unknown corruption {corruption!r}: use key, value or query.")
    rng = random.Random(seed)
    records: list[PromptRecord] = []
    seen: set[str] = set()
    attempts = 0
    while len(records) < n:
        attempts += 1
        if attempts > 50 * n + 1000:
            raise ToyError(f"Could only make {len(records)} distinct prompts. Choose fewer.")
        a, b, f, c, *rest = _sample(rng, INDUCTION_WORDS, length - 1)
        units: list[list[str]] = [["A", "B"], ["F", "C"], *[[w] for w in rest]]
        for i in range(len(units) - 1, 0, -1):  # shuffle the pairs among the other words
            j = _pick(rng, i + 1)
            units[i], units[j] = units[j], units[i]
        order = [w for unit in units for w in unit] + ["end"]
        clean_words = {"A": a, "B": b, "F": f, "C": c, "end": a}
        corrupt_words = dict(clean_words)
        if corruption == "key":
            corrupt_words |= {"A": f, "F": a}
        elif corruption == "value":
            corrupt_words |= {"B": c, "C": b}
        else:
            corrupt_words["end"] = f
        clean, spans = _words(order, clean_words)
        if clean in seen:
            continue
        seen.add(clean)
        corrupt, _ = _words(order, corrupt_words)
        first = "AB" if order.index("A") < order.index("F") else "FC"
        records.append(
            PromptRecord(
                clean=clean,
                corrupt=corrupt,
                answer=" " + b,
                distractor=" " + c,
                positions=spans,
                id=f"induction-{len(records):05d}",
                meta={"first": first, "corruption": corruption},
            )
        )
    return records


def _words(order: list[str], words: dict[str, str]) -> tuple[str, dict[str, tuple[int, int]]]:
    """The prompt that writes ``order`` (role names, or words) with ``words`` for the roles, and
    each role's character span."""
    text = ""
    spans: dict[str, tuple[int, int]] = {}
    for item in order:
        word = words.get(item, item)
        if text:
            text += " "
        if item in words:
            spans[item] = (len(text), len(text) + len(word))
        text += word
    return text, spans


def ioi_prompts(
    n: int,
    seed: int = 0,
    *,
    patterns: Sequence[Literal["ABBA", "BABA"]] = ("ABBA", "BABA"),
    corruption: Literal["flip", "abc"] = "flip",
) -> list[PromptRecord]:
    """``n`` IOI prompt pairs made by :func:`logogram.ioi.generate_ioi` with the templates whose
    names sit at the toy's name slots, from the names, places and objects the toy knows."""
    from logogram.ioi import IOIError, generate_ioi

    known = set(IOI_WORDS)
    try:
        return generate_ioi(
            n,
            seed=seed,
            templates=IOI_TEMPLATES,
            patterns=patterns,
            corruption=corruption,
            single_token=lambda text: text.strip() in known,
        )
    except IOIError as exc:
        raise ToyError(str(exc)) from exc


def dataset(name: str, n: int | None = None, seed: int = 0) -> list[PromptRecord]:
    """The dataset `logogram toy` writes for a toy (``n`` prompts, by default the toy's own
    number, from ``seed``)."""
    toy = get_toy(name)
    count = toy.n_prompts if n is None else n
    if toy.name == "induction":
        return induction_prompts(count, seed)
    return ioi_prompts(count, seed)


# -- what the planted heads should do, checked on the loaded model ------------------------------


@dataclass
class HeadCheck:
    """How much a planted head attends where it should: its attention from each prompt's query
    position to the target position (the least over prompts, and the mean)."""

    head: PlantedHead
    where: str
    least: float
    mean: float

    def to_dict(self) -> dict[str, Any]:
        return {**self.head.to_dict(), "where": self.where, "least": self.least, "mean": self.mean}


@dataclass
class ToyCheck:
    """What a toy did on its prompts, loaded as any model is: how many prompts it solves as built
    (a clean prompt prefers the answer, and a corrupt one the distractor, by at least half the
    designed logit difference of 10), the mean logit differences, and each planted head's
    attention to its target."""

    toy: Toy
    n_prompts: int
    clean_correct: int
    corrupt_correct: int
    clean_logit_diff: float
    corrupt_logit_diff: float
    heads: list[HeadCheck] = field(default_factory=list)

    # A planted head attends to its target with all but 10^-8 of its attention; anything below
    # this means the model isn't the one designed.
    THRESHOLD = 0.99
    # How far a solved prompt prefers its answer: half the logit difference it was built to give.
    MARGIN = OUT / 2

    @property
    def problems(self) -> list[str]:
        out = []
        if self.clean_correct < self.n_prompts:
            out.append(
                f"it prefers the answer on {self.clean_correct} of {self.n_prompts} clean prompts "
                f"by {self.MARGIN:g} or more"
            )
        if self.corrupt_correct < self.n_prompts:
            out.append(
                f"it prefers the distractor on {self.corrupt_correct} of {self.n_prompts} corrupt "
                f"prompts by {self.MARGIN:g} or more"
            )
        for h in self.heads:
            if not h.least >= self.THRESHOLD:
                out.append(
                    f"{h.head.label} ({h.head.kind.replace('_', '-')}) attends {h.least:.3f} "
                    f"{h.where}"
                )
        return out

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "toy": self.toy.name,
            "n_prompts": self.n_prompts,
            "margin": self.MARGIN,
            "solved": {"clean": self.clean_correct, "corrupt": self.corrupt_correct},
            "accuracy": {
                "clean": self.clean_correct / self.n_prompts,
                "corrupt": self.corrupt_correct / self.n_prompts,
            },
            "logit_diff": {"clean": self.clean_logit_diff, "corrupt": self.corrupt_logit_diff},
            "heads": [h.to_dict() for h in self.heads],
            "ok": self.ok,
            "problems": self.problems,
        }


def _targets(toy: Toy, ids: list[int], labels: dict[str, int]) -> dict[str, tuple[int, int]]:
    """Each planted head's (query position, the key it should attend to) in one prompt (its
    token ids and labelled positions), worked out from the tokens: where a head has no target in
    this prompt, it is left out."""
    out: dict[str, tuple[int, int]] = {}
    if toy.name == "induction":
        end = len(ids) - 1
        out["induction"] = (end, _after_copy(ids, end))
        return out
    s2 = labels["S2"]
    if _after_copy(ids, s2):
        out["induction"] = (s2, _after_copy(ids, s2))
        end = len(ids) - 1
        out["s_inhibition"] = (end, s2)
        other = [p for p in NAME_SLOTS if ids[p] != ids[s2]]
        if len(other) == 1:
            out["name_mover"] = (end, other[0])
    return out


def _after_copy(ids: list[int], q: int) -> int:
    """The token after the latest earlier copy of the token at ``q`` (0, the first token, if
    there is none): where an induction head at ``q`` attends."""
    for p in range(q, 0, -1):
        if ids[p - 1] == ids[q] and p - 1 < q:
            return p
    return 0


def check_toy(
    backend: ModelBackend, name: str, records: list[PromptRecord] | None = None
) -> ToyCheck:
    """Check, on the loaded model, that a toy does its task and that its planted heads attend
    where they were built to: run each prompt of ``records`` (the toy's dataset by default) clean
    and corrupt, count those it solves, and read each planted head's attention to its target
    (the previous token at every position, and the target its role gives at its query)."""
    from logogram.backends.base import float64
    from logogram.prompts import prepare_prompts

    toy = get_toy(name)
    if backend.info.n_layers != toy.n_layers or backend.info.n_heads != N_HEADS:
        raise ToyError(
            f"The model loaded has {backend.info.n_layers} layers of {backend.info.n_heads} "
            f"heads, so it isn't the {toy.name} toy."
        )
    prompts = prepare_prompts(backend, records or dataset(toy.name), prepend_bos=True)
    import torch

    diffs = {"clean": [], "corrupt": []}
    seen: dict[str, list[float]] = {h.kind: [] for h in toy.heads}
    layers = sorted({h.layer for h in toy.heads})
    with backend.lock:
        for prompt in prompts:
            for side in ("clean", "corrupt"):
                ids = list(prompt.tokens_of(side).ids)
                tokens = torch.tensor([ids], dtype=torch.long)
                logits = backend.logits(tokens, None, 1)[0, -1]
                diff = float(logits[prompt.answer_id] - logits[prompt.distractor_id])
                diffs[side].append(diff)
                patterns = backend.attention_patterns(tokens, layers)
                targets = _targets(toy, ids, prompt.labels_of(side))
                for h in toy.heads:
                    pattern = float64(patterns[h.layer][0, h.head]).cpu()
                    if h.kind == "previous_token":
                        at = torch.arange(1, len(ids))
                        seen[h.kind] += pattern[at, at - 1].tolist()
                    elif h.kind in targets:
                        q, k = targets[h.kind]
                        seen[h.kind].append(float(pattern[q, k]))
    where = {
        "previous_token": "to the token before each token",
        "induction": "to the token after the earlier copy of its token",
        "s_inhibition": "from the last token to the repeated name",
        "name_mover": "from the last token to the name that isn't repeated",
    }
    clean, corrupt = np.array(diffs["clean"]), np.array(diffs["corrupt"])
    return ToyCheck(
        toy=toy,
        n_prompts=len(prompts),
        clean_correct=int((clean >= ToyCheck.MARGIN).sum()),
        corrupt_correct=int((corrupt <= -ToyCheck.MARGIN).sum()),
        clean_logit_diff=float(clean.mean()),
        corrupt_logit_diff=float(corrupt.mean()),
        heads=[
            HeadCheck(
                head=h,
                where=where[h.kind],
                least=float(min(seen[h.kind])) if seen[h.kind] else float("nan"),
                mean=float(np.mean(seen[h.kind])) if seen[h.kind] else float("nan"),
            )
            for h in toy.heads
        ],
    )


# -- the experiments a toy comes with: what each method should find ----------------------------


@dataclass(frozen=True)
class ToyExperiment:
    """An experiment a toy's project comes with: a spec's question (``name``), what it should find
    and why (``notes``), and the designed result: each site's normalized effect (``expect``, by
    the site's label; every other site 0) to within ``tolerance``, and for sets of sites their
    faithfulness."""

    id: str
    name: str
    notes: str
    dataset: str
    experiment: dict[str, Any]
    scope: dict[str, Any]
    expect: dict[str, float]
    tolerance: float = 1e-4
    faithfulness: dict[str, float] = field(default_factory=dict)


_ALL: dict[str, Any] = {"kind": "all"}
_LAST: dict[str, Any] = {"kind": "last"}
_PATCH = {"kind": "activation_patching", "direction": "clean_to_corrupt"}
_HEADS = {"kind": "heads", "position": _ALL}


def _site(kind: str, layer: int, head: int | None = None, position: Any = None) -> dict[str, Any]:
    site: dict[str, Any] = {"kind": kind, "layer": layer, "position": position or _ALL}
    if head is not None:
        site["head"] = head
    return site


def _label(site: dict[str, Any]) -> str:
    from logogram.sites import site_label
    from logogram.spec import Site

    return site_label(Site.model_validate(site))


def _inputs(heads: Sequence[PlantedHead]) -> list[dict[str, Any]]:
    return [_site(part, h.layer, h.head) for h in heads for part in "qkv"]


def _circuit(toy: Toy) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    """Sets that keep the planted heads, or keep every other head, replacing the rest with the
    corrupt prompt's values: the circuit should be faithful (1) and the rest not (0)."""
    planted = [_site("head", h.layer, h.head) for h in toy.heads]
    taken = {(h.layer, h.head) for h in toy.heads}
    others = [
        _site("head", layer, head)
        for layer in range(toy.n_layers)
        for head in range(N_HEADS)
        if (layer, head) not in taken
    ]
    scope = {
        "kind": "site_sets",
        "universe": ["head"],
        "sets": [
            {"label": "every head replaced", "sites": [], "complement": True},
            {"label": "only the planted heads kept", "sites": planted, "complement": True},
            {"label": "only the other heads kept", "sites": others, "complement": True},
        ],
    }
    expect = {"every head replaced": 1.0, "only the other heads kept": 1.0}
    faithful = {"only the planted heads kept": 1.0, "only the other heads kept": 0.0}
    return scope, expect, faithful


def _induction_experiments(toy: Toy) -> list[ToyExperiment]:
    prev, ind = toy.head("previous_token"), toy.head("induction")
    p, i = prev.label, ind.label
    circuit, circuit_expect, faithful = _circuit(toy)
    data = toy.dataset
    return [
        ToyExperiment(
            id="which-heads",
            name="Which heads restore the answer?",
            notes=(
                f"Patch one head's output from the clean prompt into the corrupt one, at every "
                f"position. Two heads were built to do the task: {p}, the previous-token head, "
                f"and {i}, the induction head. Expect each of them to restore all of it (1.0) "
                "and every other head nothing (0.0). The corrupt prompt swaps A and F, so what "
                f"{p} writes at B and C differs, and {i} reads it."
            ),
            dataset=data,
            experiment=_PATCH,
            scope=_HEADS,
            expect={p: 1.0, i: 1.0},
        ),
        ToyExperiment(
            id="which-input",
            name="Which input of each head carries the difference?",
            notes=(
                f"Patch each query, key and value of {p} and {i} at every position. The swap "
                f"changes which word comes before B and C: it enters through {p}'s value (the "
                f"words it copies) and reaches {i} through its key. Expect those two at 1.0 and "
                "the other four at 0. Prompts made with corruption 'value' (B and C swap) move "
                f"it to {i}'s value, and 'query' (the last word changes) to its query."
            ),
            dataset=data,
            experiment=_PATCH,
            scope={"kind": "sites", "sites": _inputs((prev, ind))},
            expect={
                _label(_site("v", prev.layer, prev.head)): 1.0,
                _label(_site("k", ind.layer, ind.head)): 1.0,
            },
        ),
        ToyExperiment(
            id="previous-token-into-key",
            name=f"Does the previous-token head reach {i} through its key?",
            notes=(
                f"Path patching: each head's output from the clean prompt reaches {i}'s key and "
                "nothing else (every other head is held at its corrupt value). Expect "
                f"{p} at 1.0 and the other heads of layer 0 at 0. Into {i}'s query or value "
                "the same senders give 0: K-composition is the only path."
            ),
            dataset=data,
            experiment={
                "kind": "path_patching",
                "direction": "clean_to_corrupt",
                "receivers": [{"kind": "head", "layer": ind.layer, "head": ind.head, "input": "k"}],
                "freeze_mlps": False,
            },
            scope=_HEADS,
            expect={p: 1.0},
        ),
        ToyExperiment(
            id="is-the-circuit-all",
            name="Is the planted circuit all there is?",
            notes=(
                "Run the clean prompt with heads replaced by their values from the corrupt one: "
                "every head, every head but the planted ones, or only the planted ones. Expect "
                "the planted heads kept alone to keep the answer (faithfulness 1.0), and the "
                "other heads kept alone to lose all of it (faithfulness 0)."
            ),
            dataset=data,
            experiment={"kind": "activation_patching", "direction": "corrupt_to_clean"},
            scope=circuit,
            expect=circuit_expect,
            faithfulness=faithful,
        ),
        ToyExperiment(
            id="gradient-estimates",
            name="What does one gradient estimate?",
            notes=(
                f"Attribution patching estimates patching from one gradient. Expect {i} at 1.0, "
                "exact: it writes the answer straight into the logits, a linear path. Expect "
                f"{p} at 0, though patching it restores everything: it acts through {i}'s "
                "attention, which is saturated (its target wins by 20 nats), so the gradient "
                "there is about e^-20. Verify estimates by patching."
            ),
            dataset=data,
            experiment={
                "kind": "attribution_patching",
                "direction": "clean_to_corrupt",
                "method": "gradient",
                "steps": None,
            },
            scope=_HEADS,
            expect={i: 1.0},
        ),
        ToyExperiment(
            id="integrated-gradients",
            name="Do integrated gradients find the previous-token head?",
            notes=(
                "Integrated gradients average the gradient over inputs between the corrupt and "
                f"the clean prompt. On the way, {i}'s attention moves from C to B within about a "
                "fortieth of the way, and the average sees it when its steps are finer than "
                f"that. Expect {p} at 0.999 with these 32 steps (0.47 with 8, 0.92 with 16) and "
                f"{i} at 1.0."
            ),
            dataset=data,
            experiment={
                "kind": "attribution_patching",
                "direction": "clean_to_corrupt",
                "method": "integrated_gradients",
                "steps": 32,
            },
            scope=_HEADS,
            expect={p: 1.0, i: 1.0},
            tolerance=0.01,
        ),
        ToyExperiment(
            id="direct-effects",
            name="Which head writes the answer?",
            notes=(
                "Split each clean prompt's logit difference into what each head writes into the "
                f"logits at the last token. Expect {i} to write all of it (1.0) and every other "
                f"head none (0): {p} writes only into the previous-token subspace, which no logit "
                "reads."
            ),
            dataset=data,
            experiment={"kind": "direct_logit_attribution", "prompts": "clean"},
            scope={"kind": "heads", "position": _LAST},
            expect={_label(_site("head", ind.layer, ind.head, _LAST)): 1.0},
        ),
    ]


def _ioi_experiments(toy: Toy) -> list[ToyExperiment]:
    prev, ind = toy.head("previous_token"), toy.head("induction")
    inh, mover = toy.head("s_inhibition"), toy.head("name_mover")
    i, s, m = ind.label, inh.label, mover.label
    circuit, circuit_expect, faithful = _circuit(toy)
    data = toy.dataset
    return [
        ToyExperiment(
            id="which-heads",
            name="Which heads restore the answer?",
            notes=(
                f"Patch one head's output from the clean prompt into the flipped one, at every "
                f"position. Expect the induction head {i}, the S-inhibition head {s} and the "
                f"name mover {m} at 1.0 each, and every other head at 0. The previous-token "
                f"head {prev.label} is part of the circuit but scores 0: both prompts name the "
                "same two people in the same slots, so what it writes is the same in both. "
                "Patching shows only what differs between the prompts."
            ),
            dataset=data,
            experiment=_PATCH,
            scope=_HEADS,
            expect={i: 1.0, s: 1.0, m: 1.0},
        ),
        ToyExperiment(
            id="which-input",
            name="Which input of each head carries the difference?",
            notes=(
                "Patch each query, key and value of the four planted heads at every position. "
                f"The repeated name enters {i} through its query; {s} reads which slot repeats "
                f"through its value; {m} turns away from that slot through its query. Expect "
                "those three at 1.0 and the other nine at 0."
            ),
            dataset=data,
            experiment=_PATCH,
            scope={"kind": "sites", "sites": _inputs((prev, ind, inh, mover))},
            expect={
                _label(_site("q", ind.layer, ind.head)): 1.0,
                _label(_site("v", inh.layer, inh.head)): 1.0,
                _label(_site("q", mover.layer, mover.head)): 1.0,
            },
        ),
        ToyExperiment(
            id="s-inhibition-into-query",
            name="Does S-inhibition reach the name mover through its query?",
            notes=(
                f"Path patching into {m}'s query: each earlier head's output from the clean "
                f"prompt reaches it and nothing else. Expect {s} at 1.0 and every other head at "
                f"0, {i} too: its effect reaches {m} only through {s}, which is held."
            ),
            dataset=data,
            experiment={
                "kind": "path_patching",
                "direction": "clean_to_corrupt",
                "receivers": [
                    {"kind": "head", "layer": mover.layer, "head": mover.head, "input": "q"}
                ],
                "freeze_mlps": False,
            },
            scope=_HEADS,
            expect={s: 1.0},
        ),
        ToyExperiment(
            id="induction-into-value",
            name="Does the induction head reach S-inhibition through its value?",
            notes=(
                f"Path patching into {s}'s value. Expect {i} at 1.0 and every other head of "
                f"layers 0 and 1 at 0. Into {s}'s key, {i} gives 0: the flag it writes (a "
                "repeat is here) is the same in both prompts; only the slot differs."
            ),
            dataset=data,
            experiment={
                "kind": "path_patching",
                "direction": "clean_to_corrupt",
                "receivers": [{"kind": "head", "layer": inh.layer, "head": inh.head, "input": "v"}],
                "freeze_mlps": False,
            },
            scope=_HEADS,
            expect={i: 1.0},
        ),
        ToyExperiment(
            id="is-the-circuit-all",
            name="Is the planted circuit all there is?",
            notes=(
                "Run the clean prompt with heads replaced by their values from the flipped one: "
                "every head, every head but the planted ones, or only the planted ones. Expect "
                "the planted heads kept alone to keep the answer (faithfulness 1.0), and the "
                "other heads kept alone to lose all of it (faithfulness 0)."
            ),
            dataset=data,
            experiment={"kind": "activation_patching", "direction": "corrupt_to_clean"},
            scope=circuit,
            expect=circuit_expect,
            faithfulness=faithful,
        ),
        ToyExperiment(
            id="gradient-estimates",
            name="What does one gradient estimate?",
            notes=(
                f"Attribution patching estimates patching from one gradient. Expect {m} at 1.0, "
                "exact: it writes the answer straight into the logits. Expect "
                f"{i} and {s} at 0, though patching gives 1.0: they act through saturated "
                "attention (S-inhibition's and the name mover's). Integrated gradients miss them "
                "too, even with 64 steps: two softmaxes in a row flip within a hundredth of the "
                "way from one prompt to the other, between two steps."
            ),
            dataset=data,
            experiment={
                "kind": "attribution_patching",
                "direction": "clean_to_corrupt",
                "method": "gradient",
                "steps": None,
            },
            scope=_HEADS,
            expect={m: 1.0},
        ),
        ToyExperiment(
            id="direct-effects",
            name="Which head writes the answer?",
            notes=(
                "Split each clean prompt's logit difference into what each head writes into the "
                f"logits at the last token. Expect the name mover {m} to write all of it (1.0) "
                "and every other head none (0)."
            ),
            dataset=data,
            experiment={"kind": "direct_logit_attribution", "prompts": "clean"},
            scope={"kind": "heads", "position": _LAST},
            expect={_label(_site("head", mover.layer, mover.head, _LAST)): 1.0},
        ),
        ToyExperiment(
            id="steer-the-slot",
            name="Can one direction flip which name is said?",
            notes=(
                "On prompts of one order (ABBA), the clean and flipped prompts differ after "
                f"{s} in one direction only: which name slot repeats. Add their mean difference "
                "at the last token after layer 2. Expect it to flip the answer at strengths 1 "
                f"and 2 (1.0), and at 0.5 to leave {m} split over the three names (2/3); a "
                "random direction of the same length does nothing. On prompts that mix ABBA and "
                "BABA (ioi.jsonl) the differences cancel, and the direction does nothing either."
            ),
            dataset="ioi-abba.jsonl",
            experiment={
                "kind": "steering",
                "apply_to": "clean",
                "direction": {"kind": "mean_difference"},
                "coefficients": [0.5, 1.0, 2.0],
                "train_fraction": 0.5,
                "seed": 0,
                "control": True,
                "continuations": None,
            },
            scope={"kind": "sites", "sites": [_site("resid_post", inh.layer, None, _LAST)]},
            expect={
                f"{_label(_site('resid_post', inh.layer, None, _LAST))} ×{c}": value
                for c, value in (("0.5", 2 / 3), ("1", 1.0), ("2", 1.0))
            },
            tolerance=0.02,
        ),
    ]


def experiments(name: str) -> list[ToyExperiment]:
    """The experiments a toy's project comes with, each saying what it should find."""
    toy = get_toy(name)
    return _induction_experiments(toy) if toy.name == "induction" else _ioi_experiments(toy)


def datasets(name: str) -> dict[str, list[PromptRecord]]:
    """The datasets a toy's project comes with, by file name."""
    toy = get_toy(name)
    out = {toy.dataset: dataset(toy.name)}
    if toy.name == "ioi":
        out["ioi-abba.jsonl"] = ioi_prompts(toy.n_prompts, 0, patterns=["ABBA"])
    return out


def model_ref(name: str) -> dict[str, Any]:
    """How a spec names a toy: as a local model, by the hash of its files."""
    toy = get_toy(name)
    return {
        "id": toy.model_name,
        "source": "local",
        "revision": None,
        "sha256": toy.sha256,
        "dtype": "float32",
        "device": "cpu",
        "process_weights": True,
    }


def spec_of(name: str, experiment: ToyExperiment, dataset_sha256: str | None = None) -> Spec:
    """The spec of one of a toy's experiments."""
    from logogram.spec import Spec

    return Spec.model_validate(
        {
            "logogram_spec": 3,
            "name": experiment.name,
            "notes": experiment.notes,
            "model": model_ref(name),
            "dataset": {
                "path": f"datasets/{experiment.dataset}",
                "sha256": dataset_sha256,
                "limit": None,
            },
            "tokenization": {"prepend_bos": True, "chat_template": None},
            "experiment": experiment.experiment,
            "scope": experiment.scope,
            "metric": {"kind": "logit_diff", "normalization": "dataset_gap"},
            "statistics": {"bootstrap": 1000, "ci": 0.95, "seed": 0, "cluster": None},
            "execution": {"batch_size": 32},
        }
    )


# -- a toy's project: its model, datasets and experiments, and its local model ----------------


@dataclass
class ToyFolder:
    """What :func:`write_toy` made: the project, the model's folder in it, the datasets and the
    experiments (project-relative paths), the local model the files are added as, and whether
    they were added now (``registered``: "added", or "already" when they were before)."""

    toy: Toy
    project: Project
    model_folder: Path
    datasets: list[str]
    experiments: list[str]
    local_name: str
    registered: str


def default_folder(name: str) -> Path:
    """Where a toy's project goes by default: beside the examples, in the projects folder."""
    from logogram.project import default_projects_parent

    return default_projects_parent() / get_toy(name).model_name


def write_toy(name: str, folder: str | Path | None = None) -> ToyFolder:
    """Write a toy's project into ``folder`` (by default :func:`default_folder`): its model's
    files in ``model/``, its datasets and its experiments (drafts whose notes say what each should
    find), and add the model's folder to this machine's local models under ``toy-NAME``, so
    specs find it by the hash of its files.

    A folder this made before is brought up to date: the model's files are rewritten if they
    changed, and datasets and experiments that are missing are written (those there are kept as
    they are). Any other folder that isn't empty is refused."""
    from logogram.datasets import file_sha256, write_dataset
    from logogram.fileio import write_text_atomic
    from logogram.project import PROJECT_FILE, Project, ProjectError, ProjectMeta, now_iso

    toy = get_toy(name)
    root = Path(folder).expanduser() if folder is not None else default_folder(toy.name)
    root = root.resolve()
    if (root / PROJECT_FILE).is_file():
        project = Project.open(root)
        made_by = getattr(project.meta, "toy", None)
        if made_by != toy.name:
            what = f"the {made_by} toy's" if made_by in TOYS else "another"
            raise ToyError(
                f"{root.name} is {what} project, not the {toy.name} toy's. Choose another folder."
            )
    else:
        try:
            project = Project.create(root.parent, toy.title, folder=root.name)
        except ProjectError as exc:
            raise ToyError(f"{exc} Choose an empty folder for the toy.") from exc
        meta = ProjectMeta.model_validate(
            {
                **project.meta.model_dump(),
                "description": toy.summary,
                "created": now_iso(),
                "toy": toy.name,
            }
        )
        write_text_atomic(
            project.writable(root / PROJECT_FILE), json.dumps(meta.model_dump(), indent=2) + "\n"
        )
        project = Project.open(root)
    model_folder = root / "model"
    try:
        model_folder.mkdir(exist_ok=True)
        write_model(toy.name, project.writable(model_folder / "config.json").parent)
    except ProjectError as exc:
        raise ToyError(f"{exc} Make the toy in a folder of its own.") from exc
    written: list[str] = []
    hashes: dict[str, str] = {}
    for file_name, records in datasets(toy.name).items():
        path = project.dataset_file(file_name)
        if not path.exists():
            write_dataset(path, records)
        hashes[file_name] = file_sha256(project.readable(path))
        written.append(path.relative_to(root).as_posix())
    drafts: list[str] = []
    for experiment in experiments(toy.name):
        if not (project.experiments_dir / experiment.id).exists():
            run_folder = project.prepare_run_dir(experiment.id)
            spec = spec_of(toy.name, experiment, hashes[experiment.dataset])
            write_text_atomic(project.writable(run_folder / "spec.json"), spec.to_json())
        drafts.append(f"experiments/{experiment.id}/spec.json")
    local_name, registered = register(toy.name, model_folder)
    return ToyFolder(
        toy=toy,
        project=project,
        model_folder=model_folder,
        datasets=written,
        experiments=drafts,
        local_name=local_name,
        registered=registered,
    )


def register(name: str, folder: Path) -> tuple[str, str]:
    """Add a toy's model folder to the local models under ``toy-NAME``, unless its files are
    added already: returns the name they are added under and "added" or "already". An entry for
    the same files whose folder is gone or changed is replaced by this folder."""
    from logogram import local_models
    from logogram.backends.base import BackendError

    toy = get_toy(name)
    same = local_models.find_model(sha256=toy.sha256)
    if same is not None:
        try:
            local_models.checked_folder(same.name, same.sha256)
            return same.name, "already"
        except BackendError:
            local_models.remove_model(same.name)  # its folder is gone or changed: this one is it
    if local_models.find_model(name=toy.model_name) is not None:
        raise ToyError(
            f"Another local model is called {toy.model_name}, with other files. Remove it "
            f"(logogram models remove {toy.model_name}), then make the toy again."
        )
    try:
        model = local_models.add_model(folder, toy.model_name)
    except BackendError as exc:
        raise ToyError(str(exc)) from exc
    if model.sha256 != toy.sha256:
        local_models.remove_model(model.name)
        raise ToyError(
            f"The files in {folder.name} hash to {model.sha256[:12]}…, not the {toy.name} toy's "
            f"{toy.sha256[:12]}…. Make the toy again in an empty folder."
        )
    return model.name, "added"


def load_toy(
    name: str, local_name: str | None = None, *, device: str = "cpu", dtype: str = "float32"
) -> tuple[ModelBackend, dict[str, Any]]:
    """Load a toy added to the local models, as any local model is loaded: its checks include
    that it solves its prompts and that its planted heads attend where they were built to (see
    :func:`check_loaded_toy`). Returns the model and what that check found."""
    from logogram.backends.transformer_lens import load_model
    from logogram.runner import configure_determinism

    toy = get_toy(name)
    configure_determinism()
    backend = load_model(
        local_name or toy.model_name,
        dtype=dtype,
        device=device,
        process_weights=True,
        source="local",
        sha256=toy.sha256,
    )
    return backend, backend.info.extra["toy"]


def check_loaded_toy(backend: ModelBackend) -> ToyCheck | None:
    """If the model just loaded is a toy (its files hash as one does), check that it solves its
    prompts and that its planted heads attend where they were built to, and record what the
    check found in its information (``extra["toy"]``). A toy that doesn't is refused: the model
    in front of us isn't the one designed. Other models are left alone (None)."""
    from logogram.backends.base import BackendError

    info = backend.info
    toy = next((t for t in TOYS.values() if t.sha256 and t.sha256 == info.sha256), None)
    if toy is None:
        return None
    check = check_toy(backend, toy.name)
    if not check.ok:
        fix = (
            " Load it in float32: its attention scores are large, and need float32's precision."
            if info.dtype != "float32"
            else " Its files are the toy's, so this is a bug: please report it."
        )
        raise BackendError(
            f"The {toy.name} toy didn't behave as built when it loaded in {info.dtype}: "
            f"{'; '.join(check.problems)}.{fix}"
        )
    info.extra["toy"] = check.to_dict()
    return check
