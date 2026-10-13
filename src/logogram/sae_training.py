"""SAEs trained in the project: a TopK sparse autoencoder (Gao et al. 2024) on one site of one
layer, trained on a corpus's texts, deterministically, and saved in SAELens's format, so it loads
and is used everywhere a published SAE is.

A training spec (:class:`SAETrainingSpec`, ``logogram_sae_training: 1``) states every choice that
can change the weights, with no default for any of them. Training:

1. reads the corpus (refused if it has changed since its hash was pinned), splits its texts into
   training and held-out texts with the seed, and tokenizes them;
2. runs the model over the training texts, in the spec's order and fixed batches, until it has the
   activations of ``steps × batch_tokens`` tokens (or every training token, when there are fewer:
   they are then read several times over), skipping each text's beginning-of-sequence token. They
   are held in memory, in float32;
3. with ``normalize`` "mean_squared_norm", multiplies every activation by one factor, chosen so
   that their mean squared norm is ``d_in`` (as SAELens's "expected_average_only_in" does);
4. initializes the decoder's rows as random directions of unit length and the encoder as the
   decoder's transpose (Gao et al.), the encoder's bias at zero and the decoder's bias at the mean
   activation (where Bricken et al. and Gao et al. take the geometric median);
5. takes ``steps`` steps of Adam, each on ``batch_tokens`` tokens drawn in a seeded order (a new
   order each time every token has been read). Each step minimizes the batch's fraction of
   variance unexplained, ``Σ‖x − x̂‖² / Σ‖x − x̄‖²``, plus, with ``auxk``, ``coefficient`` times the
   AuxK loss of Gao et al.: the ``auxk.k`` dead latents (those that haven't fired in
   ``auxk.dead_after`` tokens) with the largest pre-activations reconstruct what the live ones
   missed, ``Σ‖e − ê‖² / Σ‖x − x̄‖²`` with ``ê`` their decoder rows' sum (no bias). The gradient's
   component along each decoder row is projected out before the step and the rows are made unit
   length again after it;
6. folds the normalization factor into the weights (the encoder times it, the decoder and its bias
   divided by it), so the saved SAE reads the activations as the model computes them;
7. measures, on the held-out texts and with the SAE exactly as saved: the fraction of variance
   explained, how many features fire per token (L0), the share of features that never fire, and
   the model's next-token loss with the SAE's reconstruction spliced in at the site, against the
   clean model and against the site set to zero: the loss recovered,
   ``(zero − spliced) / (zero − clean)``.

The SAE is ``saes/NAME/`` in the project: ``cfg.json`` and ``sae_weights.safetensors`` as SAELens
writes them (read back by :func:`logogram.backends.saes.read_sae`), ``training.json`` (the spec as
it ran, with the corpus's hash and the model's revision pinned) and ``metrics.json`` (the loss
curve, the held-out measurements, the normalization factor, and where it ran). A cancelled or
failed training writes nothing. A spec names it as ``{"source": "project", "repo": NAME, "path":
"saes/NAME", "revision": null, "sha256": ...}``; the hash is that of its two files (see
:func:`files_sha256`), pinned when a run starts.

The same spec on the same machine gives bit-identical weights: the random numbers come from
seeded CPU generators whatever the device, the batches' composition is fixed, the SAE trains in
float32, and deterministic algorithms stay on throughout (a training during which they were turned
off is refused).
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import torch
from pydantic import Field, model_validator

from logogram.backends.base import ModelBackend, float64
from logogram.fileio import JSON_LIMIT
from logogram.project import Project, now_iso
from logogram.sae import SAE
from logogram.spec import TRAINED_SAES, ModelRef, SAERef, Seed
from logogram.training import (
    MODEL_BATCH_MAX,
    CorpusRef,
    Progress,
    Texts,
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

CFG_FILE = "cfg.json"
WEIGHTS_FILE = "sae_weights.safetensors"
TRAINING_FILE = "training.json"
METRICS_FILE = "metrics.json"
# The largest weights file read: an SAE of a million features on a 4,096-wide model.
WEIGHTS_LIMIT = 64 * 1024**3
D_SAE_MAX = 2**21
BATCH_MAX = 2**20
STEPS_MAX = 10_000_000
# Tokens a timing step of the estimate reads; the time of a step scales with its batch.
PLAN_STEP_TOKENS = 256

TrainSite = Literal["resid_pre", "resid_post", "attn_out", "mlp_out"]
Beta = Annotated[float, Field(ge=0.0, lt=1.0)]


class TopK(_Strict):
    """A TopK SAE (Gao et al. 2024): each token keeps its ``k`` largest pre-activations, each
    through a ReLU, and every other feature is zero."""

    kind: Literal["topk"]
    """The architecture: TopK is the one Logogram trains."""
    k: int = Field(ge=1)
    """How many features each token keeps."""


class AuxK(_Strict):
    """The auxiliary loss for dead latents of Gao et al. (2024): the ``k`` dead latents with the
    largest pre-activations reconstruct what the live ones missed, so they get gradient again."""

    k: int = Field(ge=1)
    """How many dead latents reconstruct the error (at most as many as are dead)."""
    coefficient: float = Field(gt=0.0, le=10.0)
    """What the auxiliary loss is multiplied by before it is added (Gao et al. use 1/32)."""
    dead_after: int = Field(ge=1)
    """A latent is dead once it hasn't fired on this many training tokens."""


class SAETrainingSpec(_Strict):
    """An SAE trained in the project: what it reads, from which texts, and every choice of its
    training. Nothing that can change its weights has a default."""

    logogram_sae_training: Literal[1]
    """The training spec's version."""
    name: str = Field(min_length=1, max_length=80)
    """The SAE's name: it is saved as saes/NAME, never replacing another."""
    notes: str = ""
    """What it is for, kept in its training.json."""
    model: ModelRef
    """The model whose activations it is trained on, exactly."""
    corpus: CorpusRef
    """The texts it is trained on, and held out from them, measured on."""
    tokenization: TextTokenization
    """How each text becomes tokens."""
    site: TrainSite
    """The activation it reads and writes: the residual stream before or after the layer, or the
    layer's attention or MLP output."""
    layer: int = Field(ge=0)
    """The layer, from 0."""
    architecture: TopK
    """How it chooses the features that fire."""
    d_sae: int = Field(ge=1, le=D_SAE_MAX)
    """How many features it has."""
    steps: int = Field(ge=1, le=STEPS_MAX)
    """How many steps of the optimizer it takes."""
    batch_tokens: int = Field(ge=1, le=BATCH_MAX)
    """How many tokens' activations each step reads."""
    lr: float = Field(gt=0.0, le=1.0)
    """Adam's learning rate, the same at every step."""
    betas: tuple[Beta, Beta]
    """Adam's coefficients for its running averages of the gradient and of its square."""
    eps: float = Field(gt=0.0, le=1.0)
    """Adam's term added to the denominator, for numerical stability."""
    seed: Seed
    """Seeds the split into training and held-out texts, the order the texts are read in, the
    initial weights and the order of the tokens."""
    held_out: float = Field(gt=0.0, lt=1.0)
    """The share of the corpus's texts, drawn with the seed, held out from training and measured
    on."""
    normalize: Literal["mean_squared_norm", "none"]
    """mean_squared_norm: train on the activations times one factor, measured on the training
    tokens, that makes their mean squared norm d_in (it is folded into the saved weights and
    recorded); none: train on the activations as they are."""
    auxk: AuxK | None
    """The auxiliary loss for dead latents; null trains without it."""
    log_every: int = Field(ge=1)
    """How often (in steps) the loss curve records the mean of the steps since its last point."""
    model_batch: int = Field(ge=1, le=MODEL_BATCH_MAX)
    """Texts per forward pass of the model. Part of the spec because batch shape can change
    floating-point results in the last digits."""

    @model_validator(mode="after")
    def _sizes(self) -> SAETrainingSpec:
        check_name(self.name, "an SAE")
        if self.architecture.k > self.d_sae:
            raise ValueError(
                "architecture.k can't be more than d_sae: no token keeps more features"
            )
        if self.auxk is not None and self.auxk.k > self.d_sae:
            raise ValueError("auxk.k can't be more than d_sae")
        return self

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2) + "\n"


def load_training_spec(data: dict[str, Any] | str | bytes) -> SAETrainingSpec:
    """A training spec from a dict or its JSON."""
    if isinstance(data, str | bytes):
        return SAETrainingSpec.model_validate_json(data)
    return SAETrainingSpec.model_validate(data)


def files_sha256(cfg: bytes, weights: bytes) -> str:
    """The hash a spec pins a trained SAE by: the SHA-256 of the hex SHA-256 of its cfg.json
    followed by that of its sae_weights.safetensors."""
    return sha256_of((sha256_of(cfg) + sha256_of(weights)).encode("ascii"))


def ref_of(name: str, sha256: str | None) -> SAERef:
    """How a spec names the SAE ``name`` trained in the project."""
    return SAERef(
        repo=name, path=f"{TRAINED_SAES}/{name}", revision=None, source="project", sha256=sha256
    )


@dataclass
class TrainedSAE:
    """An SAE training wrote: its name and folder, the hash a spec pins it by, the spec as it ran
    and what was measured."""

    name: str
    path: str
    sha256: str
    spec: SAETrainingSpec
    metrics: dict[str, Any]

    def ref(self) -> SAERef:
        return ref_of(self.name, self.sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "sha256": self.sha256,
            "spec": self.spec.model_dump(mode="json"),
            "metrics": self.metrics,
        }


# -- training -----------------------------------------------------------------------------------


def check_site(spec: SAETrainingSpec, info: Any) -> None:
    if spec.layer >= info.n_layers:
        raise TrainingError(
            f"{info.id} has {info.n_layers} layers (0 to {info.n_layers - 1}): there is no layer "
            f"{spec.layer}."
        )
    if spec.site not in info.site_kinds:
        raise TrainingError(f"{info.id} has no {spec.site} site to train an SAE on.")


def chosen_texts(texts: Texts, needed: int) -> list[list[int]]:
    """The training texts, in their order, until they hold ``needed`` tokens (or all of them)."""
    out: list[list[int]] = []
    total = 0
    for ids in texts.train:
        if total >= needed:
            break
        out.append(ids)
        total += len(ids) - texts.first
    return out


def collect(
    backend: ModelBackend,
    spec: SAETrainingSpec,
    texts: Texts,
    chosen: list[list[int]],
    report: Progress,
    cancel: threading.Event | None,
) -> torch.Tensor:
    """The site's activations ``[tokens, d_in]`` (float32, on the CPU) at every token of the
    chosen texts after the first, read ``model_batch`` texts at a time."""
    total = texts.tokens(chosen)
    width = backend.info.d_model
    pool = torch.empty(total, width, dtype=torch.float32)
    at = 0
    key = (spec.site, spec.layer)
    for tokens, lengths, _ in batches(chosen, spec.model_batch, pad_id(backend)):
        check_cancel(cancel)
        acts = backend.capture(tokens, [key])[key]
        if acts.shape[-1] != width:
            raise TrainingError(f"{spec.site} of layer {spec.layer} isn't {width} wide here.")
        for row, length in enumerate(lengths):
            piece = acts[row, texts.first : length]
            pool[at : at + piece.shape[0]] = piece.float().cpu()
            at += int(piece.shape[0])
        report({"stage": "collecting", "done": at, "total": total})
    return pool


def _chunked_sums(pool: torch.Tensor, chunk: int = 65536) -> tuple[torch.Tensor, float]:
    """The sum of the rows of ``pool`` and the sum of their squared norms, in float64."""
    # float64 on the CPU
    total = torch.zeros(pool.shape[1], dtype=torch.float64)
    squares = 0.0
    for start in range(0, pool.shape[0], chunk):
        block = float64(pool[start : start + chunk])
        total += block.sum(0)
        squares += float(block.pow(2).sum())
    return total, squares


def token_stream(n: int, batch: int, seed: int) -> Iterator[torch.Tensor]:
    """Batches of ``batch`` indices into ``n`` tokens: the tokens in a seeded order, then in
    another once every one has been read, and so on, without gaps."""
    generator = torch.Generator().manual_seed(sub_seed(seed, 3))
    order = torch.randperm(n, generator=generator)
    at = 0
    while True:
        parts = []
        need = batch
        while need:
            if at == n:
                order = torch.randperm(n, generator=generator)
                at = 0
            take = min(need, n - at)
            parts.append(order[at : at + take])
            at += take
            need -= take
        yield torch.cat(parts)


@dataclass
class _Weights:
    W_enc: torch.Tensor
    b_enc: torch.Tensor
    W_dec: torch.Tensor
    b_dec: torch.Tensor

    def all(self) -> list[torch.Tensor]:
        return [self.W_enc, self.b_enc, self.W_dec, self.b_dec]


def initial_weights(d_in: int, d_sae: int, mean: torch.Tensor, seed: int) -> _Weights:
    """Decoder rows: random directions of unit length; the encoder: their transpose (Gao et
    al. 2024); the encoder's bias zero; the decoder's bias the activations' mean. Drawn on the
    CPU, in float32."""
    generator = torch.Generator().manual_seed(sub_seed(seed, 2))
    W_dec = torch.randn(d_sae, d_in, generator=generator, dtype=torch.float32)
    W_dec /= W_dec.norm(dim=-1, keepdim=True)
    return _Weights(
        W_enc=W_dec.T.clone(),
        b_enc=torch.zeros(d_sae, dtype=torch.float32),
        W_dec=W_dec,
        b_dec=mean.to(torch.float32).clone(),
    )


def _step(
    w: _Weights,
    x: torch.Tensor,
    k: int,
    auxk: AuxK | None,
    dead: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """One forward pass of training: the loss, the batch's fraction of variance unexplained, the
    features ``[batch, d_sae]``, and the AuxK loss (None without it, or no dead latent)."""
    pre = (x - w.b_dec) @ w.W_enc + w.b_enc
    top = pre.topk(k, dim=-1)
    f = torch.zeros_like(pre).scatter(-1, top.indices, torch.relu(top.values))
    err = x - (f @ w.W_dec + w.b_dec)
    total = (x - x.mean(0)).pow(2).sum().clamp_min(1e-12)
    fvu = err.pow(2).sum() / total
    loss = fvu
    aux = None
    if auxk is not None and dead is not None:
        n_dead = int(dead.sum())
        if n_dead:
            masked = pre.masked_fill(~dead[None, :], float("-inf"))
            top_aux = masked.topk(min(auxk.k, n_dead), dim=-1)
            f_aux = torch.zeros_like(pre).scatter(-1, top_aux.indices, torch.relu(top_aux.values))
            aux = (err.detach() - f_aux @ w.W_dec).pow(2).sum() / total
            loss = loss + auxk.coefficient * aux
    return loss, fvu, f, aux


def train_sae(
    spec: SAETrainingSpec,
    project: Project,
    *,
    backend: ModelBackend | None = None,
    provider: Callable[[ModelRef], ModelBackend] | None = None,
    on_progress: Progress | None = None,
    cancel: threading.Event | None = None,
) -> TrainedSAE:
    """Train the SAE ``spec`` describes and write it to ``saes/NAME`` in ``project`` (see the
    module's description). ``backend`` is used when it is the spec's model; otherwise ``provider``
    (by default, the runner's) loads it. ``on_progress`` hears each stage: reading the corpus,
    collecting activations, training (with the loss curve so far), measuring on the held-out
    texts, writing. Raises Cancelled when ``cancel`` is set, having written nothing."""
    from logogram.runner import configure_determinism

    report = on_progress or (lambda data: None)
    started = time.perf_counter()
    check_new(project, TRAINED_SAES, spec.name, "an SAE")
    configure_determinism(spec.seed)
    report({"stage": "loading"})
    backend = resolve_model(spec.model, backend, provider)
    info = backend.info
    check_site(spec, info)
    report({"stage": "reading"})
    texts = read_texts(project, backend, spec.corpus, spec.tokenization, spec.held_out, spec.seed)
    ran = spec.model_copy(
        update={
            "model": pinned_model(spec.model, info),
            "corpus": spec.corpus.model_copy(update={"sha256": texts.sha256}),
        }
    )
    needed = spec.steps * spec.batch_tokens
    chosen = chosen_texts(texts, needed)
    available = texts.tokens(chosen)
    if available < spec.batch_tokens:
        raise TrainingError(
            f"The training texts of {texts.corpus} hold {available:,} tokens, fewer than one "
            f"batch of {spec.batch_tokens:,}. Use a smaller batch, or a larger corpus."
        )
    pool = collect(backend, spec, texts, chosen, report, cancel)
    n, d_in = int(pool.shape[0]), int(pool.shape[1])
    total, squares = _chunked_sums(pool)
    mean_squared_norm = squares / n
    scale = 1.0
    if spec.normalize == "mean_squared_norm":
        if not mean_squared_norm > 0:
            raise TrainingError(
                f"{spec.site} of layer {spec.layer} is zero on every training token, so there is "
                "nothing to normalize or reconstruct."
            )
        scale = math.sqrt(d_in / mean_squared_norm)
        pool.mul_(scale)
    mean = (total / n) * scale

    device = backend.device
    w = initial_weights(d_in, spec.d_sae, mean, spec.seed)
    params = [p.to(device).requires_grad_(True) for p in w.all()]
    w = _Weights(*params)
    optimizer = torch.optim.Adam(params, lr=spec.lr, betas=spec.betas, eps=spec.eps)
    k = spec.architecture.k
    since = torch.zeros(spec.d_sae, dtype=torch.long)  # tokens since each latent last fired
    curve: list[dict[str, Any]] = []
    sums = {"fvu": 0.0, "l0": 0.0, "aux": 0.0, "aux_steps": 0, "steps": 0}
    stream = token_stream(n, spec.batch_tokens, spec.seed)
    for step in range(1, spec.steps + 1):
        check_cancel(cancel)
        x = pool[next(stream)].to(device)
        dead = None
        if spec.auxk is not None:
            dead = (since >= spec.auxk.dead_after).to(device)
        loss, fvu, f, aux = _step(w, x, k, spec.auxk, dead)
        if not torch.isfinite(loss):
            raise TrainingError(
                f"The loss stopped being a finite number at step {step}. Lower the learning rate."
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            grad = w.W_dec.grad
            assert grad is not None
            # Gao et al.: no step along the decoder rows themselves, which stay of unit length.
            grad -= (grad * w.W_dec).sum(-1, keepdim=True) * w.W_dec
        optimizer.step()
        with torch.no_grad():
            w.W_dec /= w.W_dec.norm(dim=-1, keepdim=True)
            fired = (f > 0).any(0).cpu()
            since += spec.batch_tokens
            since[fired] = 0
            sums["fvu"] += float(fvu)
            sums["l0"] += float((f > 0).sum(-1).float().mean())
            sums["steps"] += 1
            if aux is not None:
                sums["aux"] += float(aux)
                sums["aux_steps"] += 1
        if step % spec.log_every == 0 or step == spec.steps:
            dead_share = (
                float((since >= spec.auxk.dead_after).float().mean()) if spec.auxk else None
            )
            steps = sums["steps"]
            curve.append(
                {
                    "step": step,
                    "tokens": step * spec.batch_tokens,
                    "fvu": sums["fvu"] / steps,
                    "l0": sums["l0"] / steps,
                    "aux": sums["aux"] / sums["aux_steps"] if sums["aux_steps"] else None,
                    "dead": dead_share,
                }
            )
            sums = {"fvu": 0.0, "l0": 0.0, "aux": 0.0, "aux_steps": 0, "steps": 0}
            report({"stage": "training", "done": step, "total": spec.steps, "curve": curve})
    del pool

    # The normalization folded into the weights: the saved SAE reads activations as they are.
    with torch.no_grad():
        cfg, tensors = _saved(spec, info, w, scale)
    weights = _safetensors(tensors)
    cfg_bytes = json_bytes(cfg)
    sha = files_sha256(cfg_bytes, weights)
    sae = _sae_of(spec.name, cfg_bytes, weights, sha).to(str(device))
    report({"stage": "evaluating", "done": 0, "total": len(texts.held_out)})
    held = evaluate_sae(backend, sae, texts.held_out, texts.first, spec.model_batch, report, cancel)
    check_cancel(cancel)
    if not torch.are_deterministic_algorithms_enabled():
        raise TrainingError(
            "Deterministic algorithms were turned off while the SAE trained, so its weights may "
            "not be reproducible. Train it again."
        )
    metrics = {
        "logogram_sae_metrics": 1,
        "name": spec.name,
        "sha256": sha,
        "created": now_iso(),
        "model": model_of(info),
        "site": spec.site,
        "layer": spec.layer,
        "d_in": d_in,
        "d_sae": spec.d_sae,
        "k": k,
        "normalization": {
            "method": spec.normalize,
            "factor": scale,
            "mean_squared_norm": mean_squared_norm,
        },
        "data": {
            "corpus": texts.corpus,
            "sha256": texts.sha256,
            "texts": texts.n_texts,
            "train_texts": len(texts.train),
            "held_out_texts": len(texts.held_out),
            "texts_read": len(chosen),
            "tokens": n,
            "epochs": spec.steps * spec.batch_tokens / n,
        },
        "curve": curve,
        "held_out": held,
        "wall_time_s": round(time.perf_counter() - started, 3),
        **environment(backend),
    }
    report({"stage": "writing"})
    files = {
        CFG_FILE: cfg_bytes,
        WEIGHTS_FILE: weights,
        TRAINING_FILE: json_bytes(ran.model_dump(mode="json")),
        METRICS_FILE: json_bytes(metrics),
    }
    write_folder(project, TRAINED_SAES, spec.name, files)
    return TrainedSAE(spec.name, f"{TRAINED_SAES}/{spec.name}", sha, ran, metrics)


def _saved(
    spec: SAETrainingSpec, info: Any, w: _Weights, scale: float
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    from logogram.backends.saes import saelens_topk

    return saelens_topk(
        site=spec.site,
        layer=spec.layer,
        k=spec.architecture.k,
        W_enc=w.W_enc.detach() * scale,
        b_enc=w.b_enc.detach(),
        W_dec=w.W_dec.detach() / scale,
        b_dec=w.b_dec.detach() / scale,
        model_name=info.id,
        prepend_bos=spec.tokenization.prepend_bos,
        context_size=spec.tokenization.max_tokens,
    )


def _safetensors(tensors: dict[str, torch.Tensor]) -> bytes:
    from safetensors.torch import save

    return save(tensors, metadata={"format": "pt"})


def _sae_of(name: str, cfg: bytes, weights: bytes, sha: str) -> SAE:
    """The SAE these files hold, read as any SAELens SAE is read."""
    from safetensors.torch import load

    from logogram.backends.saes import read_config

    try:
        data = json.loads(cfg.decode("utf-8"))
        tensors = load(weights)
    except Exception as exc:  # any malformed file
        raise TrainingError(f"The SAE {name}'s files can't be read: {first_line(exc)}") from exc
    if not isinstance(data, dict):
        raise TrainingError(f"The SAE {name}'s {CFG_FILE} isn't a JSON object.")
    params = read_config(WEIGHTS_FILE, data, path=f"{TRAINED_SAES}/{name}", tensors=tensors)
    return SAE(
        repo=name,
        path=f"{TRAINED_SAES}/{name}",
        revision=None,
        params=params,
        source="project",
        sha256=sha,
    )


# -- measuring ----------------------------------------------------------------------------------


def evaluate_sae(
    backend: ModelBackend,
    sae: SAE,
    texts: list[list[int]],
    first: int,
    model_batch: int,
    report: Progress | None = None,
    cancel: threading.Event | None = None,
) -> dict[str, Any]:
    """How well ``sae`` does on ``texts`` (token ids, each read after ``first``), ``model_batch``
    at a time: the fraction of variance of the activations it explains (and unexplained), the
    features that fire per token (L0), the share of its features that fire on none of them, and
    the model's mean next-token loss over the texts' tokens with the SAE's reconstruction spliced
    in at its site (at every token it reads), clean, and with the site set to zero there; with
    the loss recovered, ``(zero − spliced) / (zero − clean)``."""
    rss = 0.0
    # float64 on the CPU
    total = torch.zeros(sae.params.d_in, dtype=torch.float64)
    squares = 0.0
    n = 0
    active = 0
    fired = torch.zeros(sae.d_sae, dtype=torch.bool)
    losses = {"clean": 0.0, "spliced": 0.0, "zero": 0.0}
    predicted = 0
    done = 0
    for tokens, lengths, which in batches(texts, model_batch, pad_id(backend)):
        if cancel is not None:
            check_cancel(cancel)
        length = int(tokens.shape[1])
        positions = torch.arange(length)
        lens = torch.tensor(lengths)
        own = (positions[None, :] >= first) & (positions[None, :] < lens[:, None])  # [B, L]

        def splice(act: torch.Tensor, own: torch.Tensor = own) -> torch.Tensor:
            nonlocal rss, total, squares, n, active, fired
            out = act.float().clone()
            mask = own.to(act.device)
            x = out[mask]
            f, stats = sae.encode(x)
            rebuilt = sae.decode(f, stats, x)
            out[mask] = rebuilt
            rss += float(float64(x - rebuilt).pow(2).sum())
            total += float64(x).sum(0).cpu()
            squares += float(float64(x).pow(2).sum())
            n += int(x.shape[0])
            active += int((f > 0).sum())
            fired |= (f > 0).any(0).cpu()
            return out

        def zero(act: torch.Tensor, own: torch.Tensor = own) -> torch.Tensor:
            out = act.clone()
            out[own.to(act.device)] = 0
            return out

        # The loss at each of the texts' own tokens that has a next token in its text.
        scored = own[:, :-1] & (positions[None, 1:] < lens[:, None])
        losses["clean"] += _loss(backend.logits(tokens, None, length), tokens, scored)
        spliced = backend.edit_logits(tokens, sae.site, sae.layer, splice, length)
        losses["spliced"] += _loss(spliced, tokens, scored)
        del spliced
        zeroed = backend.edit_logits(tokens, sae.site, sae.layer, zero, length)
        losses["zero"] += _loss(zeroed, tokens, scored)
        del zeroed
        predicted += int(scored.sum())
        done = which.stop
        if report is not None:
            report({"stage": "evaluating", "done": done, "total": len(texts)})
    if n == 0:
        raise TrainingError("The held-out texts hold no token to measure the SAE on.")
    centered = squares - float((total.pow(2)).sum()) / n
    fvu = rss / centered if centered > 0 else None
    loss = None
    if predicted:
        clean, spliced, zeroed = (losses[k] / predicted for k in ("clean", "spliced", "zero"))
        gap = zeroed - clean
        loss = {
            "clean": clean,
            "spliced": spliced,
            "zero": zeroed,
            "recovered": (zeroed - spliced) / gap if gap != 0 else None,
            "tokens": predicted,
        }
    return {
        "tokens": n,
        "fvu": fvu,
        "variance_explained": None if fvu is None else 1.0 - fvu,
        "l0": active / n,
        "dead_share": 1.0 - float(fired.float().mean()),
        "loss": loss,
    }


def _loss(logits: torch.Tensor, tokens: torch.Tensor, scored: torch.Tensor) -> float:
    """The summed next-token loss (negative log-probability of each next token) at the
    ``scored`` positions ``[B, L - 1]``, from logits ``[B, L, vocab]``."""
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    picked = logp.gather(-1, tokens[:, 1:].to(logp.device)[..., None])[..., 0].cpu()
    return -float(float64(picked[scored]).sum())


# -- reading ------------------------------------------------------------------------------------


def load_project_sae(project: Project, ref: SAERef, device: str, info: Any = None) -> SAE:
    """The SAE ``ref`` names, trained in ``project``: read from regular files inside the project,
    refused if its files' hash isn't ``ref.sha256`` (when given), on ``device``. With ``info``
    (the loaded model's), its warnings say if it was trained on another model or with other weight
    processing."""
    if ref.source != "project":
        raise TrainingError(f"{ref.repo} is an SAE on Hugging Face, not one trained here.")
    folder = project.root / TRAINED_SAES / ref.repo
    if not os.path.lexists(folder / CFG_FILE):
        raise TrainingError(
            f"This project has no SAE called {ref.repo} in {TRAINED_SAES}/. Train one first "
            "(Explore, Features, Train an SAE…, or `logogram sae train`)."
        )
    cfg = read_bytes(project, folder / CFG_FILE, JSON_LIMIT, "SAE")
    weights = read_bytes(project, folder / WEIGHTS_FILE, WEIGHTS_LIMIT, "SAE")
    sha = files_sha256(cfg, weights)
    if ref.sha256 is not None and sha != ref.sha256:
        raise TrainingError(
            f"The SAE {ref.repo} has changed since this spec was written (its files' hash no "
            "longer matches). Restore its original files, or remove sae.sha256 to use it as it "
            "is now."
        )
    sae = _sae_of(ref.repo, cfg, weights, sha)
    sae.project = str(project.root)
    try:
        metrics = read_json(project, folder / METRICS_FILE, "SAE")
    except TrainingError:
        metrics = None  # what its training measured is a description: the files are the SAE
    if metrics is not None:
        sae.trained = {
            "model": metrics.get("model"),
            "corpus": (metrics.get("data") or {}).get("corpus"),
            "held_out": metrics.get("held_out"),
        }
    if info is not None:
        sae.warnings = trained_warnings(sae, info)
    sae.params.note = " · ".join(
        part for part in ("trained in this project", _trained_on(sae)) if part
    )
    return sae.to(device)


def _trained_on(sae: SAE) -> str:
    model = (sae.trained or {}).get("model")
    return f"on {model_text(model)}" if isinstance(model, dict) else ""


def trained_warnings(sae: SAE, info: Any) -> list[str]:
    """What to know before reading an SAE trained in the project on the loaded model: that it was
    trained on another model, or with TransformerLens's weight processing the other way (which
    changes the residual stream it reads). Its fit on the prompts says how much that matters."""
    model = (sae.trained or {}).get("model")
    if not isinstance(model, dict):
        return [
            f"The SAE {sae.repo} has no {METRICS_FILE} saying which model it was trained on. "
            "Measure its fit before reading its features."
        ]
    out = []
    if not same_model(model, info):
        out.append(
            f"The SAE {sae.repo} was trained on {model_text(model)}, not on the loaded model. "
            "Measure its fit on these prompts before trusting its features."
        )
    elif bool(model.get("process_weights")) != bool(info.process_weights):
        state = "on" if model.get("process_weights") else "off"
        out.append(
            f"The SAE {sae.repo} was trained with TransformerLens's weight processing {state}, "
            "which changes the activations it reads. Load the model as it was trained."
        )
    return out


def list_project_saes(project: Project) -> list[dict[str, Any]]:
    """The SAEs trained in the project, by name, as their training recorded them (the hash is
    the one their training wrote; loading checks it against their files). One that can't be read
    says why (``error``)."""
    folder = project.root / TRAINED_SAES
    if not project.inside(folder) or not folder.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        name = path.name
        if name.startswith(".") or not path.is_dir():
            continue
        entry: dict[str, Any] = {"name": name, "path": f"{TRAINED_SAES}/{name}", "error": None}
        try:
            check_name(name, "an SAE")
            training = read_json(project, path / TRAINING_FILE, "SAE")
            metrics = read_json(project, path / METRICS_FILE, "SAE")
            spec = SAETrainingSpec.model_validate(training)
        except (TrainingError, ValueError) as exc:
            out.append({**entry, "error": first_line(exc)})
            continue
        out.append(
            {
                **entry,
                "sha256": metrics.get("sha256"),
                "site": spec.site,
                "layer": spec.layer,
                "d_in": metrics.get("d_in"),
                "d_sae": spec.d_sae,
                "k": spec.architecture.k,
                "model": metrics.get("model"),
                "corpus": spec.corpus.model_dump(mode="json"),
                "steps": spec.steps,
                "created": metrics.get("created"),
                "held_out": metrics.get("held_out"),
                "spec": training,
            }
        )
    return out


# -- what training would take -------------------------------------------------------------------


def plan_sae(
    spec: SAETrainingSpec, project: Project, backend: ModelBackend, measure: bool = True
) -> dict[str, Any]:
    """What training ``spec`` on the loaded model would take: the corpus's texts and tokens, the
    memory it holds (the activations of every token it trains on, the SAE with its gradients and
    Adam's averages, a step's features, the logits measured on the held-out texts), and with
    ``measure`` how long, from one forward pass of the model and one step of the SAE timed here.
    ``refusal`` says why it can't train, from the same checks training makes."""
    from logogram.system import _verdict, available_memory

    info = backend.info
    try:
        check_site(spec, info)
        texts = read_texts(
            project, backend, spec.corpus, spec.tokenization, spec.held_out, spec.seed
        )
    except (TrainingError, ValueError) as exc:
        return {"refusal": first_line(exc)}
    chosen = chosen_texts(texts, spec.steps * spec.batch_tokens)
    used = texts.tokens(chosen)
    held = texts.tokens(texts.held_out)
    refusal = None
    if used < spec.batch_tokens:
        refusal = (
            f"The training texts hold {used:,} tokens, fewer than one batch of "
            f"{spec.batch_tokens:,}. Use a smaller batch, or a larger corpus."
        )
    d_in, d_sae = info.d_model, spec.d_sae
    longest = max((len(t) for t in texts.held_out), default=1)
    memory: dict[str, Any] = {
        "activations": used * d_in * 4,
        "sae": (2 * d_in * d_sae + d_sae + d_in) * 4 * 4,
        "step": spec.batch_tokens * d_sae * 4 * (6 if spec.auxk else 4),
        "evaluation": min(spec.model_batch, len(texts.held_out)) * longest * info.d_vocab * 4 * 2,
    }
    memory["total"] = sum(memory.values())
    device = info.device
    memory["available"] = available_memory(device)
    memory["verdict"] = _verdict(memory["total"], memory["available"])
    seconds = None
    if measure and refusal is None:
        seconds = _timing(backend, spec, chosen, used, held)
    return {
        "refusal": refusal,
        "texts": {
            "corpus": texts.corpus,
            "sha256": texts.sha256,
            "total": texts.n_texts,
            "train": len(texts.train),
            "held_out": len(texts.held_out),
            "read": len(chosen),
        },
        "tokens": {
            "train": texts.tokens(texts.train),
            "used": used,
            "held_out": held,
            "per_step": spec.batch_tokens,
            "epochs": spec.steps * spec.batch_tokens / used if used else None,
        },
        "memory": memory,
        "seconds": seconds,
    }


def _timing(
    backend: ModelBackend,
    spec: SAETrainingSpec,
    chosen: list[list[int]],
    used: int,
    held: int,
) -> dict[str, float]:
    """Seconds each part would take, from one forward pass over the first batch of texts and one
    step of the SAE on a smaller batch, timed here and scaled."""
    first = 1 if spec.tokenization.prepend_bos else 0
    tokens, lengths, _ = next(batches(chosen, spec.model_batch, pad_id(backend)))
    began = time.perf_counter()
    backend.logits(tokens, None, 1)
    per_token = (time.perf_counter() - began) / max(1, sum(n - first for n in lengths))
    d_in = backend.info.d_model
    small = min(spec.batch_tokens, PLAN_STEP_TOKENS)
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(small, d_in, generator=generator).to(backend.device)
    w = initial_weights(d_in, spec.d_sae, torch.zeros(d_in), 0)
    params = [p.to(backend.device).requires_grad_(True) for p in w.all()]
    timed = 0.0
    for attempt in range(2):  # the first warms up
        began = time.perf_counter()
        loss, *_ = _step(_Weights(*params), x, min(spec.architecture.k, spec.d_sae), None, None)
        loss.backward()
        if attempt:
            timed = time.perf_counter() - began
    step = timed * spec.batch_tokens / small
    out = {
        "collect": used * per_token,
        "train": spec.steps * step,
        # Three passes over each held-out text: clean, with the SAE spliced in, with zeros.
        "evaluate": 3 * held * per_token,
    }
    out["total"] = sum(out.values())
    return out
