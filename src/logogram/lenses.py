"""Tuned lenses trained in the project (Belrose et al. 2023): for each layer, an affine translator
that maps the residual stream after the layer to what the final layer would make of it, read
through the model's final normalization and unembedding.

The logit lens reads each layer's ``resid_post`` as if it were the last layer's. Layers don't all
speak in the final layer's basis, so early layers read poorly that way, and some directions are
read too strongly. A tuned lens learns, per layer ``l``, ``T_l(h) = h @ W_l + b_l`` (a
``d_model × d_model`` weight and a bias), initialized to the identity (``W_l = I``, ``b_l = 0``:
the logit lens), trained to minimize the KL divergence from the model's own next-token
distribution at every token of a corpus's texts, ``KL(P_final ‖ softmax(readout(T_l(h_l))))``,
summed over layers (each layer's translator learns from its own term only).

A training spec (:class:`LensTrainingSpec`, ``logogram_lens_training: 1``) states every choice
that can change the translators, with no default for any of them. Training:

1. reads the corpus (refused if it has changed since its hash was pinned), splits its texts into
   training and held-out texts with the seed, and tokenizes them;
2. takes ``steps`` steps of AdamW (Adam without weight decay when ``weight_decay`` is null). Each
   reads the next ``batch_tokens`` tokens of the training texts, in a seeded order (another once
   every text has been read), ``model_batch`` texts per forward pass, skipping each text's
   beginning-of-sequence token. The translators are parameterized as ``h + h @ A_l + b_l`` with
   ``A_l`` and ``b_l`` starting at zero, so weight decay pulls them towards the logit lens;
3. measures on the held-out texts, for every layer, the mean KL divergence from the model's own
   distribution of the logit lens and, for the layers it trained, of the tuned lens, as saved.

The model's own distribution is the last layer's ``resid_post`` read through the final
normalization and unembedding, which reproduces its output (Logogram checks that when a model
loads, and offers lenses only where it does).

The lens is ``lenses/NAME/`` in the project: ``lens.safetensors`` (``layers.L.weight``, ``W_l``,
read as ``h @ W_l``, and ``layers.L.bias`` for each layer ``L`` it trained, float32),
``training.json`` (the spec as it ran: the corpus's hash and the model's revision pinned) and
``metrics.json`` (the loss curve, each layer's held-out KL divergence before and after, and where
it ran). A cancelled or failed training writes nothing. A spec's prediction settings name it as
``{"name": NAME, "sha256": ...}``, the hash of lens.safetensors, pinned when a run starts. A lens
reads only the model it was trained on (the same files, with TransformerLens's weight processing
the same way, since processing centers the residual stream), and is refused on any other.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

import torch
from pydantic import Field, model_validator

from logogram.backends.base import ModelBackend, float64
from logogram.project import Project, now_iso
from logogram.spec import TRAINED_LENSES, LensRef, ModelRef, Seed
from logogram.training import (
    MODEL_BATCH_MAX,
    CorpusRef,
    Progress,
    TextTokenization,
    TrainingError,
    _Strict,
    batches,
    check_cancel,
    check_name,
    check_new,
    environment,
    first_line,
    json_bytes,
    model_of,
    model_text,
    pad_id,
    pinned_model,
    read_bytes,
    read_json,
    read_texts,
    resolve_model,
    same_model,
    sha256_of,
    sub_seed,
    write_folder,
)

LENS_FILE = "lens.safetensors"
TRAINING_FILE = "training.json"
METRICS_FILE = "metrics.json"
# The largest lens file read: a translator for every layer of a 128-layer, 16,384-wide model.
LENS_LIMIT = 160 * 1024**3
# Tokens whose distributions over the vocabulary are computed at once.
CHUNK = 256
BATCH_MAX = 2**20
STEPS_MAX = 10_000_000
_TENSOR = re.compile(r"layers\.(\d+)\.(weight|bias)")

Beta = Annotated[float, Field(ge=0.0, lt=1.0)]


class LensTrainingSpec(_Strict):
    """A tuned lens trained in the project: on which model and texts, which layers, and every
    choice of its training. Nothing that can change its translators has a default."""

    logogram_lens_training: Literal[1]
    """The training spec's version."""
    name: str = Field(min_length=1, max_length=80)
    """The lens's name: it is saved as lenses/NAME, never replacing another."""
    notes: str = ""
    """What it is for, kept in its training.json."""
    model: ModelRef
    """The model it reads, exactly."""
    corpus: CorpusRef
    """The texts it is trained on, and held out from them, measured on."""
    tokenization: TextTokenization
    """How each text becomes tokens."""
    layers: list[Annotated[int, Field(ge=0)]] | None = Field(min_length=1)
    """The layers that get a translator, in order; null for every layer but the last (whose
    readout is the model's own prediction)."""
    steps: int = Field(ge=1, le=STEPS_MAX)
    """How many steps of the optimizer it takes."""
    batch_tokens: int = Field(ge=1, le=BATCH_MAX)
    """How many tokens each step reads."""
    lr: float = Field(gt=0.0, le=1.0)
    """The optimizer's learning rate, the same at every step."""
    betas: tuple[Beta, Beta]
    """Adam's coefficients for its running averages of the gradient and of its square."""
    eps: float = Field(gt=0.0, le=1.0)
    """Adam's term added to the denominator, for numerical stability."""
    weight_decay: float | None = Field(gt=0.0, le=1.0)
    """Decoupled weight decay (AdamW) on each translator's change from the identity, pulling it
    towards the logit lens; null for none."""
    seed: Seed
    """Seeds the split into training and held-out texts and the order the texts are read in."""
    held_out: float = Field(gt=0.0, lt=1.0)
    """The share of the corpus's texts, drawn with the seed, held out from training and measured
    on."""
    log_every: int = Field(ge=1)
    """How often (in steps) the loss curve records the mean of the steps since its last point."""
    model_batch: int = Field(ge=1, le=MODEL_BATCH_MAX)
    """Texts per forward pass of the model. Part of the spec because batch shape can change
    floating-point results in the last digits."""

    @model_validator(mode="after")
    def _layers(self) -> LensTrainingSpec:
        check_name(self.name, "a lens")
        if self.layers is not None:
            if len(set(self.layers)) != len(self.layers):
                raise ValueError("layers must not repeat")
            if self.layers != sorted(self.layers):
                raise ValueError("list the layers in order")
        return self

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2) + "\n"


def load_training_spec(data: dict[str, Any] | str | bytes) -> LensTrainingSpec:
    """A lens training spec from a dict or its JSON."""
    if isinstance(data, str | bytes):
        return LensTrainingSpec.model_validate_json(data)
    return LensTrainingSpec.model_validate(data)


@dataclass
class Lens:
    """A tuned lens read from the project: its translators (layer to ``(weight [d_model,
    d_model], bias [d_model])``, float32 on the CPU), the hash of its file, the model it was
    trained on (:func:`logogram.training.model_of`) and what its training measured."""

    name: str
    sha256: str
    translators: dict[int, tuple[torch.Tensor, torch.Tensor]]
    model: dict[str, Any]
    metrics: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def width(self) -> int:
        weight, _ = next(iter(self.translators.values()))
        return int(weight.shape[0])

    @property
    def layers(self) -> list[int]:
        return sorted(self.translators)

    def ref(self) -> LensRef:
        return LensRef(name=self.name, sha256=self.sha256)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sha256": self.sha256,
            "layers": self.layers,
            "width": self.width,
            "model": self.model,
        }


@dataclass
class TrainedLens:
    """A lens training wrote: its name and folder, the hash a spec pins it by, the spec as it
    ran and what was measured."""

    name: str
    path: str
    sha256: str
    spec: LensTrainingSpec
    metrics: dict[str, Any]

    def ref(self) -> LensRef:
        return LensRef(name=self.name, sha256=self.sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "sha256": self.sha256,
            "spec": self.spec.model_dump(mode="json"),
            "metrics": self.metrics,
        }


# -- training -----------------------------------------------------------------------------------


def trained_layers(spec: LensTrainingSpec, info: Any) -> list[int]:
    """The layers ``spec`` trains on the loaded model, checked: every layer but the last, or the
    spec's own."""
    if info.extra.get("prediction_method") != "final_norm_logit_lens":
        raise TrainingError(
            f"A lens reads the residual stream through {info.id}'s final normalization and "
            "unembedding, which didn't reproduce its output when it loaded, so no lens can be "
            "trained on it."
        )
    last = info.n_layers - 1
    if last < 1:
        raise TrainingError(f"{info.id} has one layer: its readout is the model's own prediction.")
    layers = list(range(last)) if spec.layers is None else list(spec.layers)
    beyond = [layer for layer in layers if layer >= last]
    if beyond:
        raise TrainingError(
            f"{info.id}'s layers are 0 to {last}; its last layer's readout is the model's own "
            f"prediction, so translators go on layers 0 to {last - 1}, not {beyond[0]}."
        )
    return layers


def text_stream(texts: list[list[int]], seed: int) -> Iterator[list[int]]:
    """The training texts, read in a seeded order, then in another once each has been read."""
    generator = torch.Generator().manual_seed(sub_seed(seed, 3))
    while True:
        for i in torch.randperm(len(texts), generator=generator).tolist():
            yield texts[i]


def _sites(layers: list[int], last: int) -> list[tuple[str, int]]:
    return [("resid_post", layer) for layer in layers] + [("resid_post", last)]


def _rows(
    backend: ModelBackend,
    texts: list[list[int]],
    layers: list[int],
    last: int,
    first: int,
    model_batch: int,
) -> Iterator[torch.Tensor]:
    """Each batch's tokens as rows ``[tokens, len(layers) + 1, d_model]`` (float32, on the CPU):
    the layers' ``resid_post`` and, last, the final layer's, at every token after ``first``."""
    sites = _sites(layers, last)
    for tokens, lengths, _ in batches(texts, model_batch, pad_id(backend)):
        captured = backend.capture(tokens, sites)
        stacked = torch.stack([captured[site] for site in sites], dim=2)  # [B, L, n + 1, d]
        yield torch.cat([stacked[row, first:n] for row, n in enumerate(lengths)]).float().cpu()


def _kl(target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """KL(P ‖ Q) at each row, from P's log-probabilities and Q's logits."""
    logq = torch.log_softmax(logits, dim=-1)
    return (target.exp() * (target - logq)).sum(-1)


def train_lens(
    spec: LensTrainingSpec,
    project: Project,
    *,
    backend: ModelBackend | None = None,
    provider: Callable[[ModelRef], ModelBackend] | None = None,
    on_progress: Progress | None = None,
    cancel: threading.Event | None = None,
) -> TrainedLens:
    """Train the tuned lens ``spec`` describes and write it to ``lenses/NAME`` in ``project``
    (see the module's description). ``backend`` is used when it is the spec's model; otherwise
    ``provider`` (by default, the runner's) loads it. ``on_progress`` hears each stage: reading
    the corpus, training (with the loss curve so far), measuring on the held-out texts, writing.
    Raises Cancelled when ``cancel`` is set, having written nothing."""
    from logogram.runner import configure_determinism

    report = on_progress or (lambda data: None)
    started = time.perf_counter()
    check_new(project, TRAINED_LENSES, spec.name, "a lens")
    configure_determinism(spec.seed)
    report({"stage": "loading"})
    backend = resolve_model(spec.model, backend, provider)
    info = backend.info
    layers = trained_layers(spec, info)
    last = info.n_layers - 1
    report({"stage": "reading"})
    texts = read_texts(project, backend, spec.corpus, spec.tokenization, spec.held_out, spec.seed)
    ran = spec.model_copy(
        update={
            "model": pinned_model(spec.model, info),
            "corpus": spec.corpus.model_copy(update={"sha256": texts.sha256}),
        }
    )
    if texts.tokens(texts.train) < spec.batch_tokens:
        raise TrainingError(
            f"The training texts of {texts.corpus} hold {texts.tokens(texts.train):,} tokens, "
            f"fewer than one batch of {spec.batch_tokens:,}. Use a smaller batch, or a larger "
            "corpus."
        )
    device = backend.device
    d = info.d_model
    A = torch.zeros(len(layers), d, d, device=device, requires_grad=True)
    b = torch.zeros(len(layers), d, device=device, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [A, b], lr=spec.lr, betas=spec.betas, eps=spec.eps, weight_decay=spec.weight_decay or 0.0
    )
    stream = text_stream(texts.train, spec.seed)
    waiting: list[torch.Tensor] = []
    held = 0
    curve: list[dict[str, Any]] = []
    # float64 on the CPU
    sums = torch.zeros(len(layers), dtype=torch.float64)
    since = 0
    for step in range(1, spec.steps + 1):
        # The next batch_tokens tokens, read model_batch texts at a time.
        while held < spec.batch_tokens:
            check_cancel(cancel)
            group = [next(stream) for _ in range(spec.model_batch)]
            for rows in _rows(backend, group, layers, last, texts.first, spec.model_batch):
                waiting.append(rows)
                held += int(rows.shape[0])
        rows = torch.cat(waiting)
        batch, rest = rows[: spec.batch_tokens], rows[spec.batch_tokens :]
        waiting, held = ([rest], int(rest.shape[0])) if rest.shape[0] else ([], 0)
        check_cancel(cancel)
        optimizer.zero_grad(set_to_none=True)
        # float64 on the CPU
        kl_sums = torch.zeros(len(layers), dtype=torch.float64)
        for start in range(0, spec.batch_tokens, CHUNK):
            chunk = batch[start : start + CHUNK].to(device)
            with torch.no_grad():
                target = torch.log_softmax(backend.read_out(chunk[:, -1]), dim=-1)
            for j in range(len(layers)):
                h = chunk[:, j]
                kl = _kl(target, backend.read_out(h + h @ A[j] + b[j]))
                (kl.sum() / spec.batch_tokens).backward()
                kl_sums[j] += float64(kl.detach()).sum().cpu()
        if not bool(torch.isfinite(kl_sums).all()):
            raise TrainingError(
                f"The KL divergence stopped being a finite number at step {step}. Lower the "
                "learning rate."
            )
        optimizer.step()
        sums += kl_sums / spec.batch_tokens
        since += 1
        if step % spec.log_every == 0 or step == spec.steps:
            per_layer = (sums / since).tolist()
            curve.append(
                {
                    "step": step,
                    "tokens": step * spec.batch_tokens,
                    "kl": sum(per_layer) / len(per_layer),
                    "layers": per_layer,
                }
            )
            sums.zero_()
            since = 0
            report({"stage": "training", "done": step, "total": spec.steps, "curve": curve})

    with torch.no_grad():
        eye = torch.eye(d, device=device)
        tensors: dict[str, torch.Tensor] = {}
        for j, layer in enumerate(layers):
            tensors[f"layers.{layer}.weight"] = (eye + A[j]).detach().float().cpu().contiguous()
            tensors[f"layers.{layer}.bias"] = b[j].detach().float().cpu().contiguous()
    data = _safetensors(tensors)
    sha = sha256_of(data)
    lens = Lens(spec.name, sha, _translators(data, spec.name), model_of(info))
    report({"stage": "evaluating", "done": 0, "total": len(texts.held_out)})
    measured = evaluate_lens(
        backend, lens, texts.held_out, texts.first, spec.model_batch, report, cancel
    )
    check_cancel(cancel)
    if not torch.are_deterministic_algorithms_enabled():
        raise TrainingError(
            "Deterministic algorithms were turned off while the lens trained, so its translators "
            "may not be reproducible. Train it again."
        )
    metrics = {
        "logogram_lens_metrics": 1,
        "name": spec.name,
        "sha256": sha,
        "created": now_iso(),
        "model": model_of(info),
        "width": d,
        "layers": layers,
        "data": {
            "corpus": texts.corpus,
            "sha256": texts.sha256,
            "texts": texts.n_texts,
            "train_texts": len(texts.train),
            "held_out_texts": len(texts.held_out),
            "tokens": spec.steps * spec.batch_tokens,
        },
        "curve": curve,
        "held_out": measured,
        "wall_time_s": round(time.perf_counter() - started, 3),
        **environment(backend),
    }
    report({"stage": "writing"})
    files = {
        LENS_FILE: data,
        TRAINING_FILE: json_bytes(ran.model_dump(mode="json")),
        METRICS_FILE: json_bytes(metrics),
    }
    write_folder(project, TRAINED_LENSES, spec.name, files)
    return TrainedLens(spec.name, f"{TRAINED_LENSES}/{spec.name}", sha, ran, metrics)


def _safetensors(tensors: dict[str, torch.Tensor]) -> bytes:
    from safetensors.torch import save

    return save(tensors, metadata={"format": "pt"})


def _translators(data: bytes, name: str) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """The translators a lens file holds, checked: a square weight and a bias for each layer."""
    from safetensors.torch import load

    try:
        tensors = load(data)
    except Exception as exc:  # any malformed file
        raise TrainingError(f"The lens {name}'s {LENS_FILE} isn't safetensors.") from exc
    parts: dict[int, dict[str, torch.Tensor]] = {}
    for key, value in tensors.items():
        match = _TENSOR.fullmatch(key)
        if match is None:
            raise TrainingError(
                f"The lens {name}'s {LENS_FILE} holds {key}, which isn't a translator's weight or "
                "bias."
            )
        parts.setdefault(int(match.group(1)), {})[match.group(2)] = value
    if not parts:
        raise TrainingError(f"The lens {name}'s {LENS_FILE} holds no translator.")
    out: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    widths = set()
    for layer, pair in sorted(parts.items()):
        weight, bias = pair.get("weight"), pair.get("bias")
        if weight is None or bias is None:
            raise TrainingError(
                f"The lens {name}'s translator for layer {layer} needs both a weight and a bias."
            )
        d = int(bias.shape[0]) if bias.dim() == 1 else -1
        if tuple(weight.shape) != (d, d) or not weight.is_floating_point():
            raise TrainingError(
                f"The lens {name}'s translator for layer {layer} isn't a square weight and a bias "
                "of its width."
            )
        weight, bias = weight.to(torch.float32), bias.to(torch.float32)
        if not (torch.isfinite(weight).all() and torch.isfinite(bias).all()):
            raise TrainingError(f"The lens {name} holds values that aren't finite numbers.")
        widths.add(d)
        out[layer] = (weight, bias)
    if len(widths) != 1:
        raise TrainingError(f"The lens {name}'s translators aren't all of one width.")
    return out


# -- measuring ----------------------------------------------------------------------------------


def evaluate_lens(
    backend: ModelBackend,
    lens: Lens | None,
    texts: list[list[int]],
    first: int,
    model_batch: int,
    report: Progress | None = None,
    cancel: threading.Event | None = None,
) -> dict[str, Any]:
    """The mean KL divergence from the model's own next-token distribution, at every token of
    ``texts`` after ``first`` (read ``model_batch`` at a time), of every layer's logit lens and,
    for the layers ``lens`` has a translator for, of its tuned lens."""
    info = backend.info
    last = info.n_layers - 1
    every = list(range(last))
    translators = lens.translators if lens is not None else {}
    # float64 on the CPU
    logit = torch.zeros(len(every), dtype=torch.float64)
    # float64 on the CPU
    tuned = torch.zeros(len(every), dtype=torch.float64)
    n = 0
    device = backend.device
    with torch.no_grad():
        for done, rows in enumerate(_rows(backend, texts, every, last, first, model_batch)):
            check_cancel(cancel)
            for start in range(0, rows.shape[0], CHUNK):
                chunk = rows[start : start + CHUNK].to(device)
                target = torch.log_softmax(backend.read_out(chunk[:, -1]), dim=-1)
                for j, layer in enumerate(every):
                    h = chunk[:, j]
                    logit[j] += float64(_kl(target, backend.read_out(h))).sum().cpu()
                    if layer in translators:
                        weight, bias = translators[layer]
                        moved = h @ weight.to(device) + bias.to(device)
                        tuned[j] += float64(_kl(target, backend.read_out(moved))).sum().cpu()
            n += int(rows.shape[0])
            if report is not None:
                read = min(len(texts), (done + 1) * model_batch)
                report({"stage": "evaluating", "done": read, "total": len(texts)})
    if n == 0:
        raise TrainingError("The held-out texts hold no token to measure the lens on.")
    return {
        "tokens": n,
        "layers": [
            {
                "layer": layer,
                "logit_lens": float(logit[j] / n),
                "tuned_lens": float(tuned[j] / n) if layer in translators else None,
            }
            for j, layer in enumerate(every)
        ],
    }


# -- reading ------------------------------------------------------------------------------------


def read_lens(project: Project, ref: LensRef) -> Lens:
    """The lens ``ref`` names, from regular files inside ``project``: refused if its file's hash
    isn't ``ref.sha256`` (when given), or if its training.json doesn't say which model it was
    trained on."""
    folder = project.root / TRAINED_LENSES / ref.name
    if not os.path.lexists(folder / LENS_FILE):
        raise TrainingError(
            f"This project has no lens called {ref.name} in {TRAINED_LENSES}/. Train one first "
            "(Explore, Layer predictions, Train a tuned lens…, or `logogram lens train`)."
        )
    data = read_bytes(project, folder / LENS_FILE, LENS_LIMIT, "lens")
    sha = sha256_of(data)
    if ref.sha256 is not None and sha != ref.sha256:
        raise TrainingError(
            f"The lens {ref.name} has changed since this spec was written (its file's hash no "
            "longer matches). Restore its original file, or remove the lens's sha256 to use it as "
            "it is now."
        )
    translators = _translators(data, ref.name)
    try:
        training = LensTrainingSpec.model_validate(
            read_json(project, folder / TRAINING_FILE, "lens")
        )
    except (TrainingError, ValueError) as exc:
        raise TrainingError(
            f"The lens {ref.name}'s {TRAINING_FILE} can't be read, so Logogram can't tell which "
            f"model it reads: {first_line(exc)}"
        ) from exc
    model = training.model
    trained_on = {
        "id": model.id,
        "source": model.source,
        "revision": model.revision,
        "sha256": model.sha256,
        "process_weights": model.process_weights,
    }
    try:
        metrics = read_json(project, folder / METRICS_FILE, "lens")
    except TrainingError:
        metrics = None  # what its training measured is a description: the file is the lens
    return Lens(ref.name, sha, translators, trained_on, metrics)


def check_lens_fits(lens: Lens, info: Any) -> None:
    """Refuse a lens trained on another model (other files, by a local model's hash or a Hugging
    Face model's id and revision), with TransformerLens's weight processing the other way (which
    changes the residual stream it reads), of another width, or for layers the model lacks."""
    if lens.width != info.d_model:
        raise TrainingError(
            f"The lens {lens.name} reads a {lens.width}-wide residual stream, and {info.id}'s is "
            f"{info.d_model} wide. Train a lens for this model."
        )
    check_trained_on(lens.name, lens.model, info)
    beyond = [layer for layer in lens.layers if layer >= info.n_layers]
    if beyond:
        raise TrainingError(
            f"The lens {lens.name} has a translator for layer {beyond[0]}, which {info.id} "
            "doesn't have."
        )


def check_trained_on(name: str, model: dict[str, Any], info: Any) -> None:
    """Refuse a lens trained on another model than the loaded one, or with TransformerLens's
    weight processing the other way."""
    if not same_model(model, info):
        raise TrainingError(
            f"The lens {name} was trained on {model_text(model)}, not on the loaded model: a lens "
            "reads one model's residual stream. Train a lens for this model."
        )
    if bool(model.get("process_weights")) != bool(info.process_weights):
        state = "on" if model.get("process_weights") else "off"
        raise TrainingError(
            f"The lens {name} was trained with TransformerLens's weight processing {state}, which "
            "changes the residual stream it reads. Load the model that way, or train a lens with "
            "it as it is loaded."
        )


def pin_lens(project: Project, ref: LensRef, info: Any) -> Lens:
    """The lens a prediction diagnostic reads: read (refused if it changed), and checked against
    the loaded model."""
    lens = read_lens(project, ref)
    check_lens_fits(lens, info)
    return lens


def list_lenses(project: Project, info: Any = None) -> list[dict[str, Any]]:
    """The lenses trained in the project, by name, as their training recorded them (the hash is
    the one their training wrote; reading checks it against their file). With ``info`` (the
    loaded model's), each says whether it fits (``fits``) and if not why (``reason``). One that
    can't be read says why (``error``)."""
    folder = project.root / TRAINED_LENSES
    if not project.inside(folder) or not folder.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        name = path.name
        if name.startswith(".") or not path.is_dir():
            continue
        entry: dict[str, Any] = {"name": name, "path": f"{TRAINED_LENSES}/{name}", "error": None}
        try:
            check_name(name, "a lens")
            training = read_json(project, path / TRAINING_FILE, "lens")
            metrics = read_json(project, path / METRICS_FILE, "lens")
            spec = LensTrainingSpec.model_validate(training)
        except (TrainingError, ValueError) as exc:
            out.append({**entry, "error": first_line(exc)})
            continue
        model = metrics.get("model") if isinstance(metrics.get("model"), dict) else None
        fits, reason = None, None
        if info is not None and model is not None:
            fits, reason = _fits(name, model, metrics.get("width"), info)
        out.append(
            {
                **entry,
                "sha256": metrics.get("sha256"),
                "layers": metrics.get("layers"),
                "width": metrics.get("width"),
                "model": model,
                "corpus": spec.corpus.model_dump(mode="json"),
                "steps": spec.steps,
                "created": metrics.get("created"),
                "held_out": metrics.get("held_out"),
                "spec": training,
                "fits": fits,
                "reason": reason,
            }
        )
    return out


def _fits(name: str, model: dict[str, Any], width: Any, info: Any) -> tuple[bool, str | None]:
    try:
        if width != info.d_model:
            raise TrainingError(
                f"The lens {name} reads a {width}-wide residual stream, and {info.id}'s is "
                f"{info.d_model} wide."
            )
        check_trained_on(name, model, info)
    except TrainingError as exc:
        return False, str(exc)
    return True, None


# -- what training would take -------------------------------------------------------------------


def plan_lens(
    spec: LensTrainingSpec, project: Project, backend: ModelBackend, measure: bool = True
) -> dict[str, Any]:
    """What training ``spec`` on the loaded model would take: the corpus's texts and tokens, the
    memory it holds (the translators with their gradients and Adam's averages, a step's
    activations at every layer it reads, the distributions over the vocabulary of a chunk of
    tokens), and with ``measure`` how long, from one forward pass of the model and one chunk of a
    step timed here. ``refusal`` says why it can't train, from the same checks training makes."""
    from logogram.system import _verdict, available_memory

    info = backend.info
    try:
        layers = trained_layers(spec, info)
        texts = read_texts(
            project, backend, spec.corpus, spec.tokenization, spec.held_out, spec.seed
        )
    except (TrainingError, ValueError) as exc:
        return {"refusal": first_line(exc)}
    train, held = texts.tokens(texts.train), texts.tokens(texts.held_out)
    refusal = None
    if train < spec.batch_tokens:
        refusal = (
            f"The training texts hold {train:,} tokens, fewer than one batch of "
            f"{spec.batch_tokens:,}. Use a smaller batch, or a larger corpus."
        )
    d, n = info.d_model, len(layers)
    longest = spec.model_batch * spec.tokenization.max_tokens
    memory: dict[str, Any] = {
        "translators": n * (d * d + d) * 4 * 4,
        "activations": (spec.batch_tokens + longest) * (n + 1) * d * 4,
        "distributions": CHUNK * info.d_vocab * 4 * 4,
        "evaluation": longest * info.n_layers * d * 4,
    }
    memory["total"] = sum(memory.values())
    memory["available"] = available_memory(info.device)
    memory["verdict"] = _verdict(memory["total"], memory["available"])
    seconds = None
    if measure and refusal is None:
        seconds = _timing(backend, spec, texts.train, texts.first, layers, held)
    return {
        "refusal": refusal,
        "layers": layers,
        "texts": {
            "corpus": texts.corpus,
            "sha256": texts.sha256,
            "total": texts.n_texts,
            "train": len(texts.train),
            "held_out": len(texts.held_out),
        },
        "tokens": {
            "train": train,
            "used": spec.steps * spec.batch_tokens,
            "held_out": held,
            "per_step": spec.batch_tokens,
            "epochs": spec.steps * spec.batch_tokens / train if train else None,
        },
        "memory": memory,
        "seconds": seconds,
    }


def _timing(
    backend: ModelBackend,
    spec: LensTrainingSpec,
    train: list[list[int]],
    first: int,
    layers: list[int],
    held: int,
) -> dict[str, float]:
    """Seconds each part would take, from one forward pass over the first batch of texts and one
    chunk of tokens through every translator, timed here and scaled."""
    last = backend.info.n_layers - 1
    group = train[: spec.model_batch]
    began = time.perf_counter()
    rows = next(_rows(backend, group, layers, last, first, spec.model_batch))
    per_token = (time.perf_counter() - began) / max(1, int(rows.shape[0]))
    chunk = rows[:CHUNK].to(backend.device)
    d = backend.info.d_model
    A = torch.zeros(len(layers), d, d, device=backend.device, requires_grad=True)
    began = time.perf_counter()
    with torch.no_grad():
        target = torch.log_softmax(backend.read_out(chunk[:, -1]), dim=-1)
    for j in range(len(layers)):
        h = chunk[:, j]
        _kl(target, backend.read_out(h + h @ A[j])).sum().backward()
    per_chunk_token = (time.perf_counter() - began) / max(1, int(chunk.shape[0]))
    out = {
        "train": spec.steps * spec.batch_tokens * (per_token + per_chunk_token),
        # Every layer read through the logit lens, and the trained ones through the tuned lens.
        "evaluate": held * (per_token + 2 * per_chunk_token * last / max(1, len(layers))),
    }
    out["total"] = sum(out.values())
    return out
