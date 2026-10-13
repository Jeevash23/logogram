"""Published sparse autoencoders and transcoders: find, download and read their files.

Three safetensors formats are read:

* SAELens (``cfg.json`` + ``sae_weights.safetensors``), as in the GPT-2 small residual SAEs. The
  config names the TransformerLens hook the SAE reads, such as ``blocks.8.hook_resid_pre``, or
  ``blocks.5.attn.hook_z`` for the outputs ``z`` of all the layer's attention heads, read side
  by side as one vector (head after head) per position.
* EleutherAI's sparsify (``cfg.json`` + ``sae.safetensors``), as for Pythia, SmolLM2 and Llama.
  The folder names the module whose output the SAE reads, such as ``layers.3`` (the residual
  stream after layer 3) or ``layers.3.mlp`` (the MLP output); a transcoder reads that MLP's input
  and predicts its output. Skip transcoders also add ``x @ W_skip.T`` of the input ``x``.
* Gemma Scope 2 (``config.json`` + ``params.safetensors``), for Gemma 3: JumpReLU SAEs on each
  layer's output, its attention heads' outputs (the input of ``o_proj``) and its MLP output, and
  transcoders from each MLP's input to its output, some with an affine skip connection that adds
  ``x @ affine_skip_connection``. The config names Hugging Face modules, such as
  ``model.layers.12.output``.

Hook and module names are library details, so they are mapped to abstract sites here. Parameters
are read only from safetensors; Gemma Scope's NumPy archives and PyTorch pickles are refused rather
than loaded with a different reader. Anything a file does that Logogram can't reproduce exactly (a
learned scaling, a gated encoder, a skip connection on an SAE, a transcoder across layers, settings
or tensors that sparsify's releases don't define) is refused with the reason.

:func:`read_config` runs the same checks on an SAE's configuration alone, on parameters of the
shapes it states, so the bundled catalogue of published SAEs (:mod:`logogram.backends.catalogue`)
says what Logogram makes of each one without downloading its weights.

SAEs trained in a project (:mod:`logogram.sae_training`) are written in SAELens's format by
:func:`saelens_topk`, as SAELens writes a TopK SAE, and read back through the same reader as any
other.
"""

from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from logogram.backends.base import BackendError

WEIGHT_FILES = ("sae_weights.safetensors", "sae.safetensors", "params.safetensors")
CONFIG_FILES = {
    "sae_weights.safetensors": "cfg.json",
    "sae.safetensors": "cfg.json",
    "params.safetensors": "config.json",
}
# sparsify writes the configuration of the whole training run here, at the top of its output.
TRAINING_CONFIG = "config.json"

_TL_HOOK = re.compile(
    r"blocks\.(\d+)\.(hook_resid_pre|hook_resid_post|hook_attn_out|hook_mlp_out|attn\.hook_z)"
)
_TL_SITES = {
    "hook_resid_pre": "resid_pre",
    "hook_resid_post": "resid_post",
    "hook_attn_out": "attn_out",
    "hook_mlp_out": "mlp_out",
    "attn.hook_z": "head",
}
_MODULE = re.compile(r"(?:.*\.)?layers\.(\d+)(?:\.(mlp|attention|self_attn|attn))?")
# Hugging Face module points, as Gemma Scope 2 names them (model.layers.N...), and the same points
# in GPT-2 (transformer.h.N...) and Pythia (gpt_neox.layers.N...) naming.
_HF_LAYER = r"(?:model\.layers|transformer\.h|gpt_neox\.layers|layers)\.(\d+)"
_HF_POINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(_HF_LAYER + r"\.output"), "resid_post"),
    (
        re.compile(_HF_LAYER + r"\.(?:self_attn\.o_proj|attn\.c_proj|attention\.dense)\.input"),
        "head",
    ),
    (re.compile(_HF_LAYER + r"\.(?:self_attn|attn|attention)\.output"), "attn_out"),
    (re.compile(_HF_LAYER + r"\.post_feedforward_layernorm\.output"), "mlp_out"),
    (re.compile(_HF_LAYER + r"\.mlp\.output"), "mlp_out"),
    (
        re.compile(
            _HF_LAYER + r"\.(?:pre_feedforward_layernorm|post_attention_layernorm|ln_2)\.output"
        ),
        "mlp_in",
    ),
    (re.compile(_HF_LAYER + r"\.mlp\.input"), "mlp_in"),
]

# What sparsify's released versions write in cfg.json and sae.safetensors. Forks of it add other
# settings and tensors, whose meaning Logogram can't check, so a file that uses any is refused.
_SPARSIFY_KEYS = frozenset(
    {
        "activation",
        "expansion_factor",
        "normalize_decoder",
        "num_latents",
        "k",
        "multi_topk",
        "skip_connection",
        "transcode",
        "d_in",
    }
)
_SPARSIFY_TENSORS = frozenset({"encoder.weight", "encoder.bias", "W_dec", "b_dec", "W_skip"})
_SAELENS_TENSORS = frozenset({"W_enc", "b_enc", "W_dec", "b_dec", "threshold"})
_GEMMA_SCOPE_TENSORS = frozenset(
    {"w_enc", "b_enc", "w_dec", "b_dec", "threshold", "affine_skip_connection"}
)


@dataclass
class SAEParams:
    """An SAE (or transcoder) as read from its files: what it reads and writes, and how it encodes
    and decodes. An SAE reads and writes the same site; a transcoder reads an MLP's input
    (``mlp_in``) and writes that MLP's output.

    Encoding: ``f = act((x - b_dec if subtract_b_dec else x) @ W_enc + b_enc)``, with ``x``
    standardized first when ``normalize`` is ``layer_norm``. Decoding: ``f @ W_dec + b_dec``
    (restored to the input's mean and spread when standardized), plus ``x @ W_skip`` for a
    transcoder with a skip connection, where ``x`` is its input as read."""

    site: str  # abstract site it writes: resid_pre, resid_post, attn_out, mlp_out or head (z)
    layer: int
    d_in: int
    d_sae: int
    W_enc: torch.Tensor  # [d_in, d_sae]
    b_enc: torch.Tensor  # [d_sae]
    W_dec: torch.Tensor  # [d_sae, d_out]
    b_dec: torch.Tensor  # [d_out]
    activation: str  # relu, topk or jumprelu
    k: int | None
    threshold: torch.Tensor | None  # [d_sae], JumpReLU only
    subtract_b_dec: bool  # subtract b_dec from the input before encoding
    normalize: str  # "none", or "layer_norm": each input standardized, each output restored
    format: str  # saelens, eleuther or gemma_scope_2
    note: str  # what the SAE says about itself, for the app (model name, training hook)
    # The site it reads: the same as ``site`` for an SAE, the MLP's input for a transcoder.
    site_in: str | None = None
    # A transcoder's skip connection, input by output: it adds x @ W_skip to what it decodes.
    W_skip: torch.Tensor | None = None  # [d_in, d_out]

    def __post_init__(self) -> None:
        if self.site_in is None:
            self.site_in = self.site
        self._check_shapes()

    @property
    def transcoder(self) -> bool:
        return self.site_in != self.site

    @property
    def d_out(self) -> int:
        return int(self.W_dec.shape[1])

    def _check_shapes(self) -> None:
        d_in, d_sae, d_out = self.d_in, self.d_sae, self.d_out
        expected = {
            "W_enc": (d_in, d_sae),
            "b_enc": (d_sae,),
            "W_dec": (d_sae, d_out),
            "b_dec": (d_out,),
            "threshold": (d_sae,),
            "W_skip": (d_in, d_out),
        }
        wrong = [
            name
            for name, shape in expected.items()
            if getattr(self, name) is not None and tuple(getattr(self, name).shape) != shape
        ]
        if self.subtract_b_dec and d_in != d_out:
            wrong.append("b_dec")
        if wrong:
            names = sorted(set(wrong))
            verb = "has" if len(names) == 1 else "have"
            raise BackendError(
                f"The SAE's weights don't fit together: {_names(names)} {verb} "
                "another shape than its other weights say. The file may be damaged or of another "
                "format."
            )


def site_of_hook(name: str) -> tuple[str, int]:
    match = _TL_HOOK.fullmatch(name)
    if match is None:
        raise BackendError(
            f"This SAE reads {name}, which Logogram can't map to a residual stream, an attention "
            "output, the attention heads' outputs (z) or an MLP output. SAEs on the stream "
            "between attention and MLP aren't supported yet."
        )
    return _TL_SITES[match.group(2)], int(match.group(1))


def hook_of_site(site: str, layer: int) -> str:
    """The TransformerLens hook SAELens names for a site Logogram trains SAEs on (the inverse of
    :func:`site_of_hook`), such as ``blocks.6.hook_resid_post``."""
    hooks = {v: k for k, v in _TL_SITES.items() if v != "head"}
    if site not in hooks:
        raise BackendError(
            f"SAEs are trained on the residual stream before or after a layer, or on its "
            f"attention or MLP output, not on {site}."
        )
    return f"blocks.{layer}.{hooks[site]}"


def saelens_topk(
    *,
    site: str,
    layer: int,
    k: int,
    W_enc: torch.Tensor,
    b_enc: torch.Tensor,
    W_dec: torch.Tensor,
    b_dec: torch.Tensor,
    model_name: str,
    prepend_bos: bool,
    context_size: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """A TopK SAE (Gao et al. 2024) as SAELens 6 writes one: ``cfg.json``'s contents and the
    tensors of ``sae_weights.safetensors``, in float32. It encodes ``x`` as the k largest of
    ``(x - b_dec) @ W_enc + b_enc``, each through a ReLU, and decodes ``f @ W_dec + b_dec``, on
    activations as the model computes them (``normalize_activations`` none: a trainer that scaled
    them folds its factor into these weights first). :func:`read_sae` reads it back exactly, as it
    reads SAELens's own."""
    d_in, d_sae = int(W_enc.shape[0]), int(W_enc.shape[1])
    hook = hook_of_site(site, layer)
    cfg: dict[str, Any] = {
        "architecture": "topk",
        "d_in": d_in,
        "d_sae": d_sae,
        "k": int(k),
        "dtype": "float32",
        "device": "cpu",
        "apply_b_dec_to_input": True,
        "normalize_activations": "none",
        "reshape_activations": "none",
        "metadata": {
            "model_name": model_name,
            "hook_name": hook,
            "hook_head_index": None,
            "prepend_bos": bool(prepend_bos),
            "context_size": int(context_size),
        },
    }
    tensors = {
        name: t.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for name, t in (("W_enc", W_enc), ("b_enc", b_enc), ("W_dec", W_dec), ("b_dec", b_dec))
    }
    return cfg, tensors


def site_of_module(folder: str) -> tuple[str, int]:
    match = _MODULE.fullmatch(folder.strip("/").split("/")[-1])
    if match is None:
        raise BackendError(
            f"This SAE reads the module {folder!r}, which Logogram can't map to a layer's "
            "output, attention output or MLP output."
        )
    part = match.group(2)
    kind = "resid_post" if part is None else "mlp_out" if part == "mlp" else "attn_out"
    return kind, int(match.group(1))


def site_of_hf_point(name: str) -> tuple[str, int]:
    """The abstract site and layer of a Hugging Face module point, such as
    ``model.layers.12.output`` (the residual stream after layer 12)."""
    for pattern, kind in _HF_POINTS:
        match = pattern.fullmatch(name)
        if match:
            return kind, int(match.group(1))
    raise BackendError(
        f"This SAE reads {name}, which Logogram can't map to a layer's output, its attention "
        "heads' outputs, its MLP's input or its MLP's output."
    )


_NPZ = (
    "This SAE is published as a NumPy archive (params.npz), as Gemma Scope's first release is. "
    "Logogram reads SAE parameters only from safetensors files; Gemma Scope 2 is published as "
    "safetensors."
)
_CROSS_LAYER = (
    "This is a cross-layer transcoder or crosscoder (its parameters are split by layer): it reads "
    "or writes several layers at once, which Logogram doesn't support yet. Choose an SAE or a "
    "single-layer transcoder."
)
_NO_FORMAT = (
    "This folder has no SAE in a format Logogram reads: it needs cfg.json with "
    "sae_weights.safetensors (SAELens) or sae.safetensors (EleutherAI), or config.json with "
    "params.safetensors (Gemma Scope 2)."
)
# PyTorch's own file types, all pickles: loading one can run any code it holds.
_PICKLES = (".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".bin")
_LAYER_FILE = re.compile(r"(?:params_layer|W_enc|W_dec)_\d+\.safetensors")


def weights_file(names: Iterable[str]) -> str | None:
    """The weights file of the format these files (names in an SAE's folder) hold, when it is one
    Logogram reads: sae_weights.safetensors, sae.safetensors or params.safetensors, each beside
    its configuration."""
    present = {n.strip("/") for n in names}
    return next((w for w in WEIGHT_FILES if w in present and CONFIG_FILES[w] in present), None)


def unreadable_files(names: Iterable[str]) -> str | None:
    """Why an SAE published as these files (their names in its folder, or the one file its
    publisher names) can't be read exactly, or None when they hold a format Logogram reads."""
    present = {n.strip("/") for n in names}
    if weights_file(present) is not None:
        return None
    leaves = {n.rsplit("/", 1)[-1] for n in present}
    if "params.npz" in leaves:
        return _NPZ
    if any(_LAYER_FILE.fullmatch(n) for n in leaves):
        return _CROSS_LAYER
    pickles = sorted(n for n in leaves if n.lower().endswith(_PICKLES))
    if pickles:
        kind = pickles[0].rsplit(".", 1)[-1].lower()
        return (
            f"Its weights are a PyTorch pickle (a .{kind} file), which Logogram never opens: "
            "loading a pickle can run code hidden in it. Logogram reads SAEs only from "
            "safetensors files."
        )
    if "hyperparams.json" in leaves:
        return (
            "This SAE is in Llama Scope's format (hyperparams.json and "
            "checkpoints/final.safetensors), which Logogram doesn't read yet."
        )
    if "conf.yaml" in leaves:
        return (
            "This is a temporal SAE (conf.yaml): its features also read earlier tokens through "
            "attention, so it isn't an SAE of one activation, which is what Logogram reproduces."
        )
    if "sae_weights.safetensors" in leaves and "config.json" in leaves:
        return (
            "This SAE is in OpenMOSS's format (config.json beside sae_weights.safetensors, as in "
            "Llama Scope R1), which Logogram doesn't read yet."
        )
    return _NO_FORMAT


def read_sae(folder: Path, path: str = "") -> SAEParams:
    """Read an SAE (or transcoder) from a local folder holding its config and safetensors
    weights. ``path`` is the folder's path in its repository, whose top may hold the training
    run's configuration (EleutherAI's sparsify writes it there)."""
    from safetensors.torch import load_file

    names = [p.name for p in folder.iterdir() if p.is_file()] if folder.is_dir() else []
    weights = weights_file(names)
    if weights is None:
        raise BackendError(unreadable_files(names) or _NO_FORMAT)
    cfg = _read_json(folder / CONFIG_FILES[weights], "The SAE's")
    tensors = load_file(str(folder / weights), device="cpu")
    # Only files from before sparsify recorded transcoders need the training run's configuration.
    training = (
        _training_config(folder, path)
        if weights == "sae.safetensors" and "transcode" not in cfg
        else None
    )
    return _read(weights, cfg, tensors, path or folder.name, training)


def _read(
    weights: str,
    cfg: dict[str, Any],
    tensors: Mapping[str, torch.Tensor],
    path: str,
    training: dict[str, Any] | None,
) -> SAEParams:
    if weights == "sae_weights.safetensors":
        return _saelens(cfg, tensors)
    if weights == "params.safetensors":
        return _gemma_scope_2(cfg, tensors)
    return _eleuther(cfg, tensors, path, training)


def read_config(
    weights: str,
    cfg: dict[str, Any],
    path: str = "",
    training: dict[str, Any] | None = None,
    tensors: Mapping[str, Sequence[int] | torch.Tensor] | None = None,
) -> SAEParams:
    """What :func:`read_sae` makes of an SAE from its configuration alone, without its weights:
    the same checks, site and sizes, run on parameters that hold no values, of the shapes its
    configuration states (or, given ``tensors``, of the names and shapes its weights file's header
    lists, and the values of any given as tensors). ``weights`` names the weights file beside the
    configuration, which says the format; ``path`` and ``training`` are as for :func:`read_sae`.
    Raises :class:`BackendError` with the reason it can't be read exactly, as reading its files
    would, or saying that the values it wasn't given decide it.

    A Gemma Scope 2 configuration doesn't state the model's width, so the parameters it reads and
    writes are given width 0: ``d_in`` and ``d_out`` mean nothing then."""
    if weights not in CONFIG_FILES:
        raise BackendError(_NO_FORMAT)
    given = dict(tensors) if tensors is not None else dict(_config_shapes(weights, cfg))
    empty = {
        name: value if isinstance(value, torch.Tensor) else torch.empty(tuple(value), device="meta")
        for name, value in given.items()
    }
    return _read(weights, cfg, empty, path, training)


def values_decide(reason: str) -> bool:
    """Whether a refusal from :func:`read_config` only says that values it wasn't given decide."""
    return reason == _SCALING_UNREAD


def _size(cfg: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = cfg.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise BackendError(f"The SAE's configuration doesn't state its size ({keys[0]}).")


def _config_shapes(weights: str, cfg: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """The parameters a configuration says its weights file holds, by name and shape."""
    if weights == "sae_weights.safetensors":
        d_in, d_sae = _size(cfg, "d_in"), _size(cfg, "d_sae")
        shapes = {"W_enc": (d_in, d_sae), "b_enc": (d_sae,), "W_dec": (d_sae, d_in)}
        shapes["b_dec"] = (d_in,)
        if cfg.get("architecture") == "jumprelu":
            shapes["threshold"] = (d_sae,)
        return shapes
    if weights == "params.safetensors":
        width = _size(cfg, "width")
        shapes = {"w_enc": (0, width), "b_enc": (width,), "w_dec": (width, 0), "b_dec": (0,)}
        shapes["threshold"] = (width,)
        if cfg.get("affine_connection"):
            shapes["affine_skip_connection"] = (0, 0)
        return shapes
    d_in = _size(cfg, "d_in")
    latents = cfg.get("num_latents") or d_in * _size(cfg, "expansion_factor")
    shapes = {"encoder.weight": (latents, d_in), "encoder.bias": (latents,)}
    shapes |= {"W_dec": (latents, d_in), "b_dec": (d_in,)}
    if _on(cfg.get("skip_connection")):
        shapes["W_skip"] = (d_in, d_in)
    return shapes


def _read_json(path: Path, whose: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BackendError(f"{whose} {path.name} can't be read: {exc}") from exc
    if not isinstance(data, dict):
        raise BackendError(f"{whose} {path.name} isn't a JSON object.")
    return data


def _training_config(folder: Path, path: str) -> dict[str, Any] | None:
    """The configuration of the training run that made the SAE in ``folder`` (``path`` in its
    repository), which sparsify writes at the top of the run's output, if it is there."""
    depth = len(Path(path.strip("/")).parts) if path.strip("/") else 0
    if depth > len(folder.parents):
        return None
    root = folder.parents[depth - 1] if depth else folder
    file = root / TRAINING_CONFIG
    return _read_json(file, "The training run's") if file.is_file() else None


def _unknown(names: set[str] | frozenset[str], known: frozenset[str]) -> list[str]:
    return sorted(set(names) - known)


def _names(names: list[str]) -> str:
    """Names in a sentence: "a", "a and b", "a, b and c"."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _extra_tensors(tensors: Mapping[str, torch.Tensor], known: frozenset[str], what: str) -> None:
    extra = _unknown(set(tensors), known)
    if extra:
        raise BackendError(
            f"This {what} file holds {_names(extra)}, which Logogram doesn't know how to use, "
            "so it can't reproduce the SAE exactly."
        )


def _gemma_scope_2(cfg: dict[str, Any], tensors: Mapping[str, torch.Tensor]) -> SAEParams:
    kind = cfg.get("type", "sae")
    if kind in ("clt", "crosscoder"):
        raise BackendError(
            f"This is a {'cross-layer transcoder' if kind == 'clt' else 'crosscoder'}: it reads or "
            "writes several layers at once, which Logogram doesn't support yet. Choose an SAE or "
            "a single-layer transcoder."
        )
    if kind not in ("sae", "transcoder"):
        raise BackendError(f"Gemma Scope 2 files of type {kind!r} aren't supported.")
    if cfg.get("architecture", "jump_relu") != "jump_relu":
        raise BackendError(f"Gemma Scope 2 {cfg.get('architecture')} SAEs aren't supported yet.")
    _extra_tensors(tensors, _GEMMA_SCOPE_TENSORS, "Gemma Scope 2")
    point_in, point_out = cfg.get("hf_hook_point_in"), cfg.get("hf_hook_point_out")
    if not isinstance(point_in, str) or not isinstance(point_out, str):
        raise BackendError("The SAE's config.json doesn't say which modules it reads and writes.")
    site_in, layer_in = site_of_hf_point(point_in)
    site, layer = site_of_hf_point(point_out)
    if kind == "sae" and (site_in, layer_in) != (site, layer):
        raise BackendError("This SAE reads one module and writes another; only transcoders do.")
    if kind == "transcoder" and (site_in, site, layer_in) != ("mlp_in", "mlp_out", layer):
        raise BackendError(
            "Logogram reads transcoders from an MLP's input to the same MLP's output; this one "
            f"maps {point_in} to {point_out}."
        )
    W_enc, b_enc, W_dec, b_dec, threshold = _need(
        tensors, "w_enc", "b_enc", "w_dec", "b_dec", "threshold"
    )
    W_skip = None
    affine = bool(cfg.get("affine_connection"))
    if affine != ("affine_skip_connection" in tensors):
        raise BackendError(
            "The SAE's config.json and its parameters disagree about an affine skip connection, "
            "so Logogram can't tell what it computes."
        )
    if affine:
        if kind != "transcoder":
            raise BackendError(_SKIP_ON_SAE)
        # Stored input by output, like w_enc and w_dec: the transcoder adds
        # x @ affine_skip_connection. (On Gemma 3 270M this reproduces the MLP's output better
        # than the same transcoder without one; the transpose fits worse than none at all.)
        (W_skip,) = _need(tensors, "affine_skip_connection")
    note = " · ".join(str(v) for v in (cfg.get("model_name"), point_in) if v)
    if W_skip is not None:
        note += " · affine skip connection"
    return SAEParams(
        site=site,
        layer=layer,
        d_in=int(W_enc.shape[0]),
        d_sae=int(W_enc.shape[1]),
        W_enc=W_enc,
        b_enc=b_enc,
        W_dec=W_dec,
        b_dec=b_dec,
        activation="jumprelu",
        k=None,
        threshold=threshold,
        # Gemma Scope encodes the activation itself, without subtracting the decoder bias.
        subtract_b_dec=False,
        normalize="none",
        format="gemma_scope_2",
        note=note,
        site_in=site_in,
        W_skip=W_skip,
    )


_SCALING_UNREAD = (
    "Whether Logogram reads this SAE exactly depends on the values of the scaling_factor its "
    "weights file holds (it does if they are all one), which only its weights say: Logogram "
    "checks them when it loads it."
)

_SKIP_ON_SAE = (
    "This SAE has a skip connection: it rebuilds the activation partly from the activation "
    "itself, so its features don't add up to it and one can't be changed alone. Logogram reads "
    "skip connections on transcoders only, whose input isn't the activation they write."
)


def _need(tensors: Mapping[str, torch.Tensor], *names: str) -> list[torch.Tensor]:
    missing = [n for n in names if n not in tensors]
    if missing:
        raise BackendError(f"The SAE's weights are missing {_names(missing)}.")
    return [tensors[n].float() for n in names]


def _saelens(cfg: dict[str, Any], tensors: Mapping[str, torch.Tensor]) -> SAEParams:
    # SAELens 6 keeps where the SAE reads under "metadata"; earlier versions at the top.
    meta: dict[str, Any] = cfg["metadata"] if isinstance(cfg.get("metadata"), dict) else {}
    hook = cfg.get("hook_name") or cfg.get("hook_point") or meta.get("hook_name")
    if not isinstance(hook, str):
        raise BackendError("The SAE's cfg.json doesn't say which hook it reads.")
    head = cfg.get("hook_head_index", cfg.get("hook_point_head_index", meta.get("hook_head_index")))
    if head is not None:
        raise BackendError(
            "SAEs on a single attention head aren't supported yet; SAEs on all of a layer's "
            "heads (hook_z) are."
        )
    site, layer = site_of_hook(hook)
    reshape = cfg.get("reshape_activations") or "none"
    if reshape not in ("none", "hook_z") or (reshape == "hook_z" and site != "head"):
        raise BackendError(
            f"This SAE reshapes its input ({reshape}) in a way Logogram doesn't reproduce for "
            f"{hook}."
        )
    architecture = cfg.get("architecture", "standard")
    if architecture not in ("standard", "jumprelu", "topk"):
        raise BackendError(f"SAELens {architecture} SAEs aren't supported yet.")
    normalize = cfg.get("normalize_activations") or "none"
    if normalize is True:
        # SAELens's configurations before version 6 said only whether: this kind.
        normalize = "expected_average_only_in"
    if normalize == "expected_average_only_in":
        raise BackendError(
            "This SAE rescales its inputs by a factor measured on its training data "
            "(normalize_activations: expected_average_only_in), which its files don't hold, so "
            "Logogram can't reproduce it exactly."
        )
    if normalize not in ("none", "layer_norm"):
        raise BackendError(
            f"This SAE rescales its inputs (normalize_activations: {normalize}), which Logogram "
            "doesn't reproduce yet."
        )
    if cfg.get("finetuning_scaling_factor") or "finetuning_scaling_factor" in tensors:
        raise BackendError(
            "This SAE scales its features by a factor fitted after training "
            "(finetuning_scaling_factor), which Logogram doesn't reproduce yet."
        )
    # SAELens's earlier files hold a scaling_factor of ones whatever their configuration says, and
    # SAELens drops it then: multiplying the features by one changes nothing. Any other value
    # would scale them, which their configuration says they don't.
    scaling = tensors.get("scaling_factor")
    if scaling is not None:
        if scaling.is_meta:
            raise BackendError(_SCALING_UNREAD)
        if not bool((scaling == 1).all()):
            raise BackendError(
                "This SAE's file holds a scaling_factor other than one, which would scale its "
                "features although its configuration says they aren't scaled, so Logogram can't "
                "tell what it computes."
            )
    _extra_tensors(tensors, _SAELENS_TENSORS | {"scaling_factor"}, "SAELens")
    W_enc, b_enc, W_dec, b_dec = _need(tensors, "W_enc", "b_enc", "W_dec", "b_dec")
    activation, k, threshold = "relu", None, None
    fn = cfg.get("activation_fn_str") or cfg.get("activation_fn") or "relu"
    if architecture == "jumprelu":
        activation = "jumprelu"
        (threshold,) = _need(tensors, "threshold")
    elif fn == "topk" or architecture == "topk":
        activation = "topk"
        kwargs = cfg.get("activation_fn_kwargs") or {}
        k = int(kwargs.get("k") or cfg.get("k") or 0)
        if k <= 0:
            raise BackendError("This TopK SAE doesn't say how many features it keeps (k).")
    elif fn != "relu":
        raise BackendError(f"SAEs with a {fn} activation aren't supported yet.")
    if "threshold" in tensors and activation != "jumprelu":
        raise BackendError(
            f"This SAE holds JumpReLU thresholds but says its architecture is {architecture}, so "
            "Logogram can't tell how it encodes."
        )
    note = " · ".join(str(v) for v in (cfg.get("model_name") or meta.get("model_name"), hook) if v)
    return SAEParams(
        site=site,
        layer=layer,
        d_in=int(W_enc.shape[0]),
        d_sae=int(W_enc.shape[1]),
        W_enc=W_enc,
        b_enc=b_enc,
        W_dec=W_dec,
        b_dec=b_dec,
        activation=activation,
        k=k,
        threshold=threshold,
        subtract_b_dec=bool(cfg.get("apply_b_dec_to_input", True)),
        normalize=normalize,
        format="saelens",
        note=note,
    )


def _on(value: Any) -> bool:
    """Whether a setting in a config changes anything: not off, zero, empty or "none"."""
    return value not in (None, False, 0, "", "none", [], {})


def _eleuther(
    cfg: dict[str, Any],
    tensors: Mapping[str, torch.Tensor],
    folder: str,
    training: dict[str, Any] | None,
) -> SAEParams:
    """sparsify's format, in both of its versions.

    Since February 2025 cfg.json says whether the file is a transcoder (``transcode``), and a
    transcoder encodes its input as it is. Files written before then don't say: only the training
    run's config.json does, and every input had ``b_dec`` subtracted before encoding (the
    published Llama 3.2 1B and DeepSeek-R1-Distill-Qwen 1.5B skip transcoders are of this kind).
    Either way, a skip connection adds ``x @ W_skip.T`` of the raw input to the decoded output.
    """
    if _on(cfg.get("signed")):
        raise BackendError("Signed SAEs (whose features can be negative) aren't supported yet.")
    unknown = [key for key in _unknown(set(cfg), _SPARSIFY_KEYS) if _on(cfg[key])]
    if unknown:
        raise BackendError(
            f"This EleutherAI file sets {_names(unknown)}, which sparsify's released versions "
            "don't define, so Logogram can't tell what it computes. Choose a file written by "
            "sparsify itself."
        )
    _extra_tensors(tensors, _SPARSIFY_TENSORS, "EleutherAI")
    activation = cfg.get("activation", "topk")
    if activation != "topk":
        raise BackendError(
            f"This SAE keeps features with sparsify's {activation} activation, which Logogram "
            "doesn't reproduce yet; TopK SAEs are read."
        )
    site, layer = site_of_module(folder)
    if "transcode" in cfg:
        transcoder = bool(cfg["transcode"])
        subtract_b_dec = not transcoder
    else:
        transcoder = _trained_as_transcoder(training, folder)
        subtract_b_dec = True
    site_in = site
    if transcoder:
        if site != "mlp_out":
            raise BackendError(
                "Logogram reads transcoders from an MLP's input to its output; this one hooks "
                f"{folder}."
            )
        site_in = "mlp_in"  # the MLP module's input (its normalized residual stream)
    encoder, b_enc, W_dec, b_dec = _need(
        tensors, "encoder.weight", "encoder.bias", "W_dec", "b_dec"
    )
    W_skip = None
    if _on(cfg.get("skip_connection")) != ("W_skip" in tensors):
        raise BackendError(
            "The SAE's cfg.json and its weights disagree about a skip connection, so Logogram "
            "can't tell what it computes."
        )
    if "W_skip" in tensors:
        if not transcoder:
            raise BackendError(_SKIP_ON_SAE)
        # sparsify stores it output by input and adds x @ W_skip.T.
        W_skip = _need(tensors, "W_skip")[0].T.contiguous()
    k = int(cfg.get("k") or 0)
    if k <= 0:
        raise BackendError("This TopK SAE doesn't say how many features it keeps (k).")
    d_in = cfg.get("d_in")
    if d_in is not None and int(d_in) != int(encoder.shape[1]):
        raise BackendError(
            f"The SAE's cfg.json says it reads {d_in} dimensions, but its encoder reads "
            f"{int(encoder.shape[1])}."
        )
    note = f"EleutherAI sparsify · {folder}"
    if transcoder:
        note += " · skip transcoder" if W_skip is not None else " · transcoder"
    return SAEParams(
        site=site,
        layer=layer,
        d_in=int(encoder.shape[1]),
        d_sae=int(encoder.shape[0]),
        W_enc=encoder.T.contiguous(),
        b_enc=b_enc,
        W_dec=W_dec,
        b_dec=b_dec,
        activation="topk",
        k=k,
        threshold=None,
        subtract_b_dec=subtract_b_dec,
        normalize="none",
        format="eleuther",
        note=note,
        site_in=site_in,
        W_skip=W_skip,
    )


def _trained_as_transcoder(training: dict[str, Any] | None, folder: str) -> bool:
    """Whether a file from before sparsify recorded transcoders is one: its training run's
    configuration says so, for the hookpoint it was trained on."""
    if training is None:
        return False
    # Hookpoints may be given as patterns, such as layers.*.mlp.
    hookpoints = [str(p) for p in training.get("hookpoints") or []]
    name = folder.strip("/").split("/")[-1]
    if hookpoints and not any(fnmatch.fnmatchcase(name, p) for p in hookpoints):
        raise BackendError(
            f"The training run's config.json beside this SAE doesn't list {name} among the "
            "modules it trained on, so it may describe other files. Logogram can't tell whether "
            "this one is a transcoder."
        )
    sae: dict[str, Any] = training["sae"] if isinstance(training.get("sae"), dict) else {}
    return bool(training.get("transcode") or sae.get("transcode"))


# -- the Hub -----------------------------------------------------------------------------------


def list_saes(repo: str, revision: str | None = None) -> tuple[str, list[str]]:
    """The exact revision of an SAE repository and the folders in it that hold an SAE."""
    from huggingface_hub import HfApi

    from logogram.backends.hub import _friendly_hub_error

    try:
        info = HfApi().model_info(repo, revision=revision)
    except Exception as exc:
        raise _friendly_hub_error(repo, exc) from exc
    names = {s.rfilename for s in info.siblings or []}
    folders = sorted(
        {
            n.rsplit("/", 1)[0] if "/" in n else ""
            for n in names
            if n.rsplit("/", 1)[-1] in WEIGHT_FILES
            and _sibling(n, CONFIG_FILES[n.rsplit("/", 1)[-1]]) in names
        },
        key=_natural,
    )
    return str(info.sha), folders


def _sibling(name: str, other: str) -> str:
    return f"{name.rsplit('/', 1)[0]}/{other}" if "/" in name else other


def _natural(text: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def download_sae(
    repo: str,
    path: str,
    revision: str | None,
    progress: Any = None,
    cancel: Any = None,
) -> tuple[str, Path]:
    """Download one SAE (its config and safetensors weights) at an exact revision."""
    from huggingface_hub import HfApi

    from logogram.backends import hub

    prefix = f"{path.strip('/')}/" if path.strip("/") else ""
    try:
        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    except Exception as exc:
        if hub._is_offline_error(exc):
            cached = _cached(repo, prefix, revision)
            if cached is not None:
                return cached
        raise hub._friendly_hub_error(repo, exc) from exc
    sizes = {s.rfilename: int(s.size or 0) for s in info.siblings or []}
    # The files in the SAE's folder (at the top level, only the top level's own files), or the one
    # file the path names, as some publishers name an SAE.
    inside = [
        n[len(prefix) :] for n in sizes if n.startswith(prefix) and (prefix or "/" not in n)
    ] or ([path.strip("/")] if path.strip("/") in sizes else [])
    weights = weights_file(inside)
    if weights is None:
        reason = unreadable_files(inside) if inside else None
        if reason is None or reason == _NO_FORMAT:
            reason = (
                f"{repo} has no SAE at {path or 'its top level'}: Logogram needs cfg.json with "
                "sae_weights.safetensors or sae.safetensors, or config.json with "
                "params.safetensors, there."
            )
        raise BackendError(reason)
    config = CONFIG_FILES[weights]
    wanted = [
        (f"{prefix}{config}", sizes[f"{prefix}{config}"]),
        (f"{prefix}{weights}", sizes[f"{prefix}{weights}"]),
    ]
    if weights == "sae.safetensors" and TRAINING_CONFIG in sizes:
        # sparsify's training run configuration: files written before it recorded transcoders in
        # cfg.json say only there whether they are one. First, so the SAE's folder comes back.
        wanted.insert(0, (TRAINING_CONFIG, sizes[TRAINING_CONFIG]))
    files = hub.RepoFiles(
        revision=str(info.sha), files=wanted, n_params=None, gated=bool(info.gated)
    )
    folder = hub.download(repo, files, progress, cancel=cancel)
    return str(info.sha), folder


def _cached(repo: str, prefix: str, revision: str | None) -> tuple[str, Path] | None:
    from logogram.backends.hub import cached_file

    for config in ("cfg.json", "config.json"):
        found = cached_file(repo, f"{prefix}{config}", revision)
        if found is None:
            continue
        commit, path = found
        if any((path.parent / w).is_file() for w in WEIGHT_FILES if CONFIG_FILES[w] == config):
            return commit, path.parent
    return None
