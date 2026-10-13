"""Where a unit is most active in plain text: the texts and tokens of a corpus where an MLP neuron,
a feature of the loaded SAE, a saved direction or an attention head responds most.

A unit gives one number at every token:

* an MLP neuron: its activation after the nonlinearity (for a gated MLP, the gate's activation
  times the up projection), offered where the model was measured to read its neurons so;
* a feature of an SAE: its activation, from the SAE's encoder on the activation the SAE reads;
* a direction (logogram.directions): the projection onto it of the activation at its site, signed;
  for a subspace of several dimensions, the length of the activation's component in it;
* an attention head: the length of what it writes into the residual stream at that token (its
  output z through its slice of the attention's output projection).

Each text is read as it is (after the beginning-of-sequence token when it is chosen), cut to its
first ``max_tokens`` tokens. The beginning-of-sequence token is read but never counted: its
activations are unlike any other token's. Over every token counted, the answer gives the ``top_k``
tokens where the unit is largest, each with a window of the text around it and the unit's value at
every token shown; for a signed unit, the ``top_k`` where it is most negative; a histogram of its
values; and the share of tokens where it is above zero. Ties are broken by the text, then the
token, so the same settings always give the same examples, and texts run in batches of one length
(shortest first), so they always give the same numbers.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from logogram.backends.base import Cancelled, ModelBackend, float64
from logogram.corpora import Corpus, label_text
from logogram.representations import (
    TOKENS_LIMIT,
    ReadSite,
    check_max_tokens,
    length_batches,
    site_words,
)

# The most examples asked for, the most tokens of context on each side, and the bins of the
# histogram of a unit's values.
TOP_MAX = 50
CONTEXT_MAX = 32
BINS = 40
# Your text reads up to this many characters (logogram.inside.TEXT_MAX): an example's text up to
# its window is offered there when it is no longer.
OPEN_MAX = 4000

ProgressFn = Callable[[int, int], None]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NeuronUnit(_Strict):
    """An MLP neuron: its activation after the nonlinearity."""

    kind: Literal["neuron"] = "neuron"
    layer: int = Field(ge=0)
    neuron: int = Field(ge=0)


class FeatureUnit(_Strict):
    """A feature of the loaded SAE: its activation."""

    kind: Literal["sae_feature"] = "sae_feature"
    feature: int = Field(ge=0)


class DirectionUnit(_Strict):
    """A direction saved in the project: the projection onto it at its site (or at ``site``)."""

    kind: Literal["direction"] = "direction"
    name: str = Field(min_length=1)
    sha256: str | None = None
    site: ReadSite | None = None


class HeadUnit(_Strict):
    """An attention head: the length of what it writes into the residual stream."""

    kind: Literal["head"] = "head"
    layer: int = Field(ge=0)
    head: int = Field(ge=0)


Unit = Annotated[NeuronUnit | FeatureUnit | DirectionUnit | HeadUnit, Field(discriminator="kind")]


@dataclass
class Reader:
    """How to read a unit at every token of a batch."""

    sites: list[tuple[str, int]]
    read: Callable[[dict[tuple[str, int], torch.Tensor]], torch.Tensor]
    """From the batch's captured activations to the unit's value ``[batch, tokens]`` in float64."""
    description: str
    signed: bool
    unit: dict[str, Any]
    """The unit as the answer records it, with what was resolved: a direction's hash and site,
    the SAE's repository and folder."""
    notes: list[str]


def _layer(backend: ModelBackend, layer: int) -> None:
    n = backend.info.n_layers
    if not 0 <= layer < n:
        raise ValueError(f"Layer {layer} isn't in this model, which has layers 0 to {n - 1}.")


def unit_reader(
    backend: ModelBackend, unit: Any, *, sae: Any = None, direction: Any = None
) -> Reader:
    """How to read ``unit``; ``sae`` is the loaded SAE (for a feature) and ``direction`` the saved
    direction a direction unit names, read from the project."""
    info = backend.info
    if isinstance(unit, NeuronUnit):
        _layer(backend, unit.layer)
        if "neuron" not in info.site_kinds:
            note = (info.extra.get("site_notes") or {}).get("neuron")
            why = note or (
                f"When {info.id} loaded, its neurons weren't measured to be what its MLPs read."
            )
            raise ValueError(f"{why} A neuron's activation isn't offered: choose another unit.")
        if info.d_mlp is None or not 0 <= unit.neuron < info.d_mlp:
            raise ValueError(
                f"Layer {unit.layer}'s MLP has {info.d_mlp or 0:,} neurons (0 to "
                f"{(info.d_mlp or 1) - 1:,}); there is no neuron {unit.neuron}."
            )
        key = ("neuron", unit.layer)
        neuron = unit.neuron
        return Reader(
            [key],
            lambda c: float64(c[key][..., neuron]).cpu(),
            f"MLP neuron {neuron} of layer {unit.layer}: its activation after the nonlinearity",
            False,
            unit.model_dump(mode="json"),
            [],
        )
    if isinstance(unit, HeadUnit):
        _layer(backend, unit.layer)
        if not 0 <= unit.head < info.n_heads:
            raise ValueError(
                f"Layer {unit.layer} has {info.n_heads} heads (0 to {info.n_heads - 1}); there is "
                f"no head {unit.head}."
            )
        key = ("head", unit.layer)
        head = unit.head
        weights = float64(backend.head_output_weights(unit.layer)[head]).cpu()  # [d_head, d_model]

        def head_norm(c: dict[tuple[str, int], torch.Tensor]) -> torch.Tensor:
            z = float64(c[key][:, :, head, :]).cpu()  # [batch, tokens, d_head]
            return (z @ weights).norm(dim=-1)

        return Reader(
            [key],
            head_norm,
            f"L{unit.layer} H{head}: the length of what the head writes into the residual stream "
            "(its output through its slice of the attention's output projection)",
            False,
            unit.model_dump(mode="json"),
            [],
        )
    if isinstance(unit, FeatureUnit):
        if sae is None:
            raise ValueError("Load an SAE to read one of its features.")
        if not 0 <= unit.feature < sae.d_sae:
            raise ValueError(
                f"The SAE has {sae.d_sae:,} features (0 to {sae.d_sae - 1:,}); there is no "
                f"feature {unit.feature}."
            )
        key = (sae.site_in, sae.layer)
        feature = unit.feature

        def feature_value(c: dict[tuple[str, int], torch.Tensor]) -> torch.Tensor:
            f, _ = sae.encode(sae.reads(c[key].float()))
            return float64(f[..., feature]).cpu()

        described = sae.describe() if hasattr(sae, "describe") else {}
        return Reader(
            [key],
            feature_value,
            f"feature {feature} of the SAE {described.get('repo', '')}"
            + (f" ({described['path']})" if described.get("path") else "")
            + ": its activation",
            False,
            {
                **unit.model_dump(mode="json"),
                "sae": {k: described.get(k) for k in ("repo", "path", "revision")},
            },
            [],
        )
    if isinstance(unit, DirectionUnit):
        from logogram.directions import check_fits

        if direction is None:
            raise ValueError(f"The direction {unit.name} wasn't read.")
        site = unit.site
        if site is None:
            found = direction.meta.site
            if found is None:
                raise ValueError(
                    f"The direction {unit.name} doesn't say where it was found: choose the site "
                    "to read it at."
                )
            site = ReadSite(kind=found.kind, layer=found.layer)
        _layer(backend, site.layer)
        if site.kind not in info.site_kinds:
            raise ValueError(f"{site_words(site.kind, site.layer).capitalize()} isn't a site here.")
        notes = check_fits(direction, info, site_words(site.kind, site.layer))
        key = (site.kind, site.layer)
        vectors = float64(direction.vectors)  # [k, width], read from its file on the CPU
        signed = direction.dimensions == 1

        def projection(c: dict[tuple[str, int], torch.Tensor]) -> torch.Tensor:
            x = float64(c[key]).cpu() @ vectors.T  # [batch, tokens, k]
            return x[..., 0] if signed else x.norm(dim=-1)

        what = (
            f"the projection onto the direction {unit.name}"
            if signed
            else f"the length of the component in the {direction.dimensions}-dimensional "
            f"subspace {unit.name}"
        )
        return Reader(
            [key],
            projection,
            f"{what}, at {site_words(site.kind, site.layer)}",
            signed,
            {
                "kind": "direction",
                "name": unit.name,
                "sha256": direction.sha256,
                "site": site.model_dump(mode="json"),
                "dimensions": direction.dimensions,
            },
            notes,
        )
    raise ValueError("Choose a unit: a neuron, an SAE feature, a direction or a head.")


def _ranked(values: np.ndarray, texts: np.ndarray, tokens: np.ndarray, k: int) -> np.ndarray:
    """The positions of the ``k`` largest values, ties broken by text and then token: exactly the
    first ``k`` of every value sorted that way, found without sorting them all."""
    if values.size <= k:
        return np.lexsort((tokens, texts, -values))
    threshold = -np.partition(-values, k - 1)[k - 1]
    candidates = np.flatnonzero(values >= threshold)
    order = np.lexsort((tokens[candidates], texts[candidates], -values[candidates]))
    return candidates[order[:k]]


def top_examples(
    backend: ModelBackend,
    corpus: Corpus,
    unit: Any,
    *,
    sae: Any = None,
    direction: Any = None,
    texts: int | None,
    max_tokens: int,
    prepend_bos: bool,
    batch_size: int,
    top_k: int,
    context: int,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> dict[str, Any]:
    """Where ``unit`` is most active over the first ``texts`` texts of ``corpus`` (all of them
    when None), each cut to its first ``max_tokens`` tokens: see the module's description."""
    from logogram.prompts import first_real_token
    from logogram.representations import _cut, _tokenize

    if not 1 <= top_k <= TOP_MAX:
        raise ValueError(f"Show 1 to {TOP_MAX} examples.")
    if not 0 <= context <= CONTEXT_MAX:
        raise ValueError(f"Show 0 to {CONTEXT_MAX} tokens of context on each side.")
    if batch_size < 1:
        raise ValueError("The batch size must be at least 1.")
    check_max_tokens(backend, max_tokens, prepend_bos)
    if texts is not None and texts < 1:
        raise ValueError("Read at least one text.")
    reader = unit_reader(backend, unit, sae=sae, direction=direction)
    records = corpus.records[:texts] if texts is not None else corpus.records
    tokenized = []
    truncated = 0
    for record in records:
        tokens = _tokenize(backend, record.text, prepend_bos)
        if len(tokens.ids) > max_tokens:
            tokens = _cut(tokens, max_tokens)
            truncated += 1
        tokenized.append(tokens)
    total_tokens = sum(len(t.ids) for t in tokenized)
    if total_tokens > TOKENS_LIMIT:
        raise ValueError(
            f"These settings read {total_tokens:,} tokens; an analysis reads at most "
            f"{TOKENS_LIMIT:,} at once. Read fewer texts, or fewer tokens of each."
        )
    first = first_real_token(backend, tokenized[0].ids, prepend_bos, False) if tokenized else 0
    values: list[np.ndarray] = [np.zeros(0)] * len(tokenized)
    done = 0
    for batch in length_batches([len(t.ids) for t in tokenized], batch_size):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        ids = torch.tensor([tokenized[i].ids for i in batch], dtype=torch.long)
        read = reader.read(backend.capture(ids, reader.sites)).numpy()  # [batch, tokens]
        if not np.isfinite(read).all():
            raise ValueError(
                "The model produced values that aren't finite numbers. Load it in float32 and "
                "try again."
            )
        for row, i in enumerate(batch):
            values[i] = read[row].copy()
        done += len(batch)
        if progress is not None:
            progress(done, len(tokenized))

    counted = [v[first:] for v in values]
    flat = np.concatenate(counted) if counted else np.zeros(0)
    if flat.size == 0:
        raise ValueError(
            "No token was counted: the texts have no tokens after the beginning-of-sequence "
            "token. Read more of each text."
        )
    which = np.concatenate([np.full(len(c), i, dtype=np.int64) for i, c in enumerate(counted)])
    where = np.concatenate([np.arange(first, first + len(c), dtype=np.int64) for c in counted])

    def example(at: int) -> dict[str, Any]:
        i, pos = int(which[at]), int(where[at])
        record, tokens = records[i], tokenized[i]
        start, end = max(0, pos - context), min(len(tokens.ids), pos + context + 1)
        # The text up to the window's last token, to read in Your text when it fits there.
        upto = tokens.offsets[end - 1][1]
        opened = record.text[:upto] if 0 < upto <= OPEN_MAX else None
        return {
            "text": i,
            "id": record.id,
            "label": label_text(record.label),
            "position": pos,
            "value": float(values[i][pos]),
            "start": start,
            "tokens": tokens.tokens[start:end],
            "values": [None if j < first else float(values[i][j]) for j in range(start, end)],
            "open": opened,
        }

    k = min(top_k, flat.size)
    top = [example(int(a)) for a in _ranked(flat, which, where, k)]
    bottom = [example(int(a)) for a in _ranked(-flat, which, where, k)] if reader.signed else None
    low, high = float(flat.min()), float(flat.max())
    edges = np.linspace(low, high, BINS + 1) if high > low else np.array([low - 0.5, low + 0.5])
    counts, _ = np.histogram(flat, bins=edges)
    active = int((flat > 0).sum())
    return {
        "corpus": corpus.describe(),
        "unit": reader.unit,
        "description": reader.description,
        "signed": reader.signed,
        "settings": {
            "texts": texts,
            "max_tokens": max_tokens,
            "prepend_bos": prepend_bos,
            "batch_size": batch_size,
            "top_k": top_k,
            "context": context,
        },
        "texts_read": len(tokenized),
        "truncated": truncated,
        "tokens": int(flat.size),
        "first_counted": first,
        "active": active,
        "active_share": active / flat.size,
        "mean": float(flat.mean()),
        "min": low,
        "max": high,
        "histogram": {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]},
        "top": top,
        "bottom": bottom,
        "notes": reader.notes,
    }
