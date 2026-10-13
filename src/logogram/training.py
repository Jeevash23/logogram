"""What training an SAE (logogram.sae_training) and a tuned lens (logogram.lenses) in a project
share: the corpus, pinned by its hash and split into training and held-out texts with a seed; the
texts tokenized and read by the model in fixed batches; the model they were trained on; and the
folder they are written to, whole or not at all.

Both are deterministic: the same training spec on the same machine gives bit-identical weights.
The split comes from the raw PCG64 stream (as runs split prompts, logogram.prompts.seeded_split),
the order of the training texts and every other random number from seeded generators on the CPU,
whatever device the model is on, and the model reads the texts in batches whose composition the
spec fixes.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from logogram.backends.base import Cancelled, ModelBackend
from logogram.corpora import read_corpus
from logogram.fileio import JSON_LIMIT, atomic_output, read_limited
from logogram.project import Project, ProjectError
from logogram.prompts import seeded_split
from logogram.spec import SHA256, TRAINED_NAME, ModelRef

# The most tokens of one text a trainer reads; longer texts are cut there.
MAX_TOKENS = 4096
# The most texts one forward pass reads.
MODEL_BATCH_MAX = 1024

Progress = Callable[[dict[str, Any]], None]


class TrainingError(ValueError):
    """A training spec can't run as written. The message says why and how to fix it."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)


class CorpusRef(_Strict):
    """The texts trained on: a corpus in the project's corpora folder."""

    name: str = Field(min_length=1, max_length=80)
    """The corpus's name, as corpora/NAME.jsonl."""
    sha256: str | None
    """The hash of its file; null uses the corpus as it is when training starts, and the
    training.json written beside the result pins it."""

    @model_validator(mode="after")
    def _hash(self) -> CorpusRef:
        if self.sha256 is not None and not SHA256.match(self.sha256):
            raise ValueError("corpus.sha256 is 64 lowercase hex digits")
        return self


class TextTokenization(_Strict):
    """How each text becomes the tokens the model reads."""

    prepend_bos: bool
    """Put the beginning-of-sequence token before each text. Its own position, the same in every
    text and unlike any other, is never trained on or measured."""
    max_tokens: int = Field(ge=2, le=MAX_TOKENS)
    """The most tokens read of each text (the beginning-of-sequence token included); the rest of
    a longer text is left out."""


def sub_seed(seed: int, purpose: int) -> int:
    """The seed of one use of a training spec's seed (the order of the texts, the initial
    weights, the order of the tokens), so that each draws its own numbers."""
    return seed * 1_000_003 + purpose


def seeded_order(n: int, seed: int) -> list[int]:
    """``range(n)`` shuffled from the raw PCG64 stream, as logogram.prompts.seeded_split shuffles:
    the same order on every machine and NumPy version."""
    order = list(range(n))
    raw = np.random.PCG64(seed).random_raw(n)
    for t in range(n - 1, 0, -1):
        j = int(raw[t] % np.uint64(t + 1))
        order[t], order[j] = order[j], order[t]
    return order


def check_name(name: str, what: str) -> str:
    if not TRAINED_NAME.match(name or ""):
        raise TrainingError(
            f"{name!r} can't name {what}: use letters, digits, dots, dashes and underscores, "
            "starting with a letter or digit, at most 80 characters."
        )
    return name


def check_new(project: Project, folder: str, name: str, what: str) -> None:
    """Refuse a name already taken in ``folder`` (saes or lenses): a trained SAE or lens is never
    replaced, so a spec pinned to its hash keeps finding it."""
    check_name(name, what)
    if os.path.lexists(project.root / folder / name):
        raise TrainingError(
            f"{what.capitalize()} called {name} already exists in {folder}/. Choose another name: "
            f"{what}s are never replaced, so the runs that used it can still find it."
        )


# -- the model ----------------------------------------------------------------------------------


def resolve_model(
    ref: ModelRef,
    backend: ModelBackend | None,
    provider: Callable[[ModelRef], ModelBackend] | None,
) -> ModelBackend:
    """The loaded ``backend`` when it is the spec's model; otherwise the spec's model, loaded."""
    from logogram.runner import default_provider, model_matches

    if model_matches(backend, ref):
        assert backend is not None
        return backend
    return (provider or default_provider())(ref)


def pinned_model(ref: ModelRef, info: Any) -> ModelRef:
    """The spec's model with the revision that loaded, as a run pins it."""
    return ref.model_copy(update={"revision": info.revision if info.source == "hub" else None})


def model_of(info: Any) -> dict[str, Any]:
    """The model something was trained on, as its files record it: a Hugging Face model by id and
    revision, a local model by the hash of its files, and whether TransformerLens processed its
    weights (which centers the residual stream)."""
    return {
        "id": info.id,
        "source": info.source,
        "revision": info.revision if info.source == "hub" else None,
        "sha256": info.sha256 if info.source == "local" else None,
        "process_weights": bool(info.process_weights),
    }


def same_model(trained: dict[str, Any], info: Any) -> bool:
    """Whether the loaded model is the one ``trained`` (from :func:`model_of`) names: the same
    files, by the hash of a local model's or a Hugging Face model's id and revision."""
    if trained.get("source") != info.source:
        return False
    if info.source == "local":
        return trained.get("sha256") is not None and trained.get("sha256") == info.sha256
    return trained.get("id") == info.id and trained.get("revision") == info.revision


def model_text(trained: dict[str, Any]) -> str:
    """A trained-on model in words: its id, at its revision, or a local model."""
    if trained.get("source") == "local":
        return f"the local model {trained.get('id')}"
    revision = trained.get("revision")
    return f"{trained.get('id')}" + (f" at {str(revision)[:7]}" if revision else "")


# -- the texts ----------------------------------------------------------------------------------


@dataclass
class Texts:
    """A corpus's texts as a trainer reads them: the token ids of the training texts, in the
    seeded order they are read in, and of the held-out texts, in the corpus's order (each cut to
    ``max_tokens``; texts with no token after ``first`` are left out), and where each text's own
    tokens start: 1 after a beginning-of-sequence token, else 0."""

    corpus: str
    sha256: str
    train: list[list[int]]
    held_out: list[list[int]]
    first: int
    n_texts: int

    def tokens(self, texts: Sequence[list[int]]) -> int:
        """How many tokens a trainer reads in ``texts``: each one's own, after ``first``."""
        return sum(max(0, len(ids) - self.first) for ids in texts)


def read_texts(
    project: Project,
    backend: ModelBackend,
    corpus: CorpusRef,
    tokenization: TextTokenization,
    held_out: float,
    seed: int,
) -> Texts:
    """The corpus pinned (refused if it has changed since ``corpus.sha256``), split into training
    and held-out texts by ``seed``, and tokenized as ``tokenization`` says."""
    if tokenization.max_tokens > backend.info.n_ctx:
        raise TrainingError(
            f"{backend.info.id} reads at most {backend.info.n_ctx} tokens at once: set "
            f"max_tokens to {backend.info.n_ctx} or fewer."
        )
    read = read_corpus(project, corpus.name, corpus.sha256)
    n = len(read.records)
    held, train = seeded_split(n, held_out, seed)
    if not held or not train:
        raise TrainingError(
            f"The corpus {read.name} has {n} text{'s' if n != 1 else ''}: with a held-out share "
            f"of {held_out:g}, {'none is held out' if not held else 'none is left to train on'}. "
            "Use more texts, or another held-out share."
        )
    order = [train[i] for i in seeded_order(len(train), sub_seed(seed, 1))]
    texts = [r.text for r in read.records]
    first = 1 if tokenization.prepend_bos else 0
    with backend.lock:
        ids = backend.tokenize_many(texts, tokenization.prepend_bos)
    cut = [row[: tokenization.max_tokens] for row in ids]
    keep = [i for i in range(n) if len(cut[i]) > first]
    kept = set(keep)
    return Texts(
        corpus=read.name,
        sha256=read.sha256,
        train=[cut[i] for i in order if i in kept],
        held_out=[cut[i] for i in held if i in kept],
        first=first,
        n_texts=n,
    )


def pad_id(backend: ModelBackend) -> int:
    """What fills a batch after a text's last token: nothing reads it (attention is causal)."""
    return int(backend.bos_token_id) if backend.bos_token_id is not None else 0


def batches(
    texts: Sequence[list[int]], size: int, pad: int
) -> Iterator[tuple[torch.Tensor, list[int], range]]:
    """``texts`` read ``size`` at a time, in order: each batch's tokens ``[n, longest]`` (shorter
    texts filled after their end with ``pad``), each text's length, and which texts it holds. A
    text's tokens never see what fills the batch after it, since attention looks back only."""
    for start in range(0, len(texts), size):
        group = texts[start : start + size]
        longest = max(len(ids) for ids in group)
        tokens = torch.full((len(group), longest), pad, dtype=torch.long)
        for row, ids in enumerate(group):
            tokens[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        yield tokens, [len(ids) for ids in group], range(start, start + len(group))


def check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


# -- the folder ---------------------------------------------------------------------------------


def write_folder(project: Project, folder: str, name: str, files: dict[str, bytes]) -> Path:
    """Write ``files`` (by name) as ``folder/name`` in the project, all of them or none: into a
    hidden folder beside it first, which then takes its name. A folder of that name is never
    replaced."""
    parent = project.root / folder
    parent.mkdir(exist_ok=True)
    final = project.writable(parent / name)  # refused when the folder links outside the project
    if os.path.lexists(final):
        raise TrainingError(f"{folder}/{name} already exists. Choose another name.")
    staging = parent / f".{name}.{secrets.token_hex(6)}.tmp"
    staging.mkdir()
    try:
        for file, data in files.items():
            with atomic_output(project.writable(staging / file), replace=False) as out:
                out.write(data)
        if os.path.lexists(final):
            raise TrainingError(f"{folder}/{name} was created meanwhile. Choose another name.")
        os.rename(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def json_bytes(data: Any) -> bytes:
    return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def read_json(project: Project, path: Path, what: str) -> dict[str, Any]:
    """A JSON object from a regular file inside the project."""
    try:
        raw = read_limited(project.readable(path), JSON_LIMIT, f"Train the {what} again.")
        data = json.loads(raw.decode("utf-8"))
    except (ProjectError, OSError, ValueError, RecursionError) as exc:
        raise TrainingError(f"{path.name} of the {what} can't be read: {first_line(exc)}") from exc
    if not isinstance(data, dict):
        raise TrainingError(f"{path.name} of the {what} isn't a JSON object.")
    return data


def read_bytes(project: Project, path: Path, limit: int, what: str) -> bytes:
    """A regular file inside the project, whole, refused when larger than ``limit`` bytes."""
    fix = f"Train the {what} again."
    try:
        resolved = project.readable(path)
        size = resolved.stat().st_size
        if size > limit:
            raise ValueError(f"{path.name} is larger than Logogram reads. {fix}")
        # read_limited makes room for as many bytes as it may read: the file's own size.
        return read_limited(resolved, size, fix)
    except (ProjectError, OSError, ValueError) as exc:
        raise TrainingError(f"{path.name} of the {what} can't be read: {first_line(exc)}") from exc


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def first_line(exc: BaseException) -> str:
    return (str(exc).strip().splitlines() or [type(exc).__name__])[0]


def environment(backend: ModelBackend) -> dict[str, Any]:
    """Where the training ran, as a run's manifest records it: the versions of the libraries and
    the device, with whether deterministic algorithms were on."""
    from logogram.runner import device_provenance, environment_versions

    return {"versions": environment_versions(), "device": device_provenance(backend.info)}
