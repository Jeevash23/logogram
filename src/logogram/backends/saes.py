"""Published sparse autoencoders and transcoders: find, download and read their files.

Three safetensors formats are read:

* SAELens (``cfg.json`` + ``sae_weights.safetensors``), as in the GPT-2 small residual SAEs. The
  config names the TransformerLens hook the SAE reads, such as ``blocks.8.hook_resid_pre``.
* EleutherAI's sparsify (``cfg.json`` + ``sae.safetensors``), as for Pythia, SmolLM2 and Llama.
  The folder names the module whose output the SAE reads, such as ``layers.3`` (the residual
  stream after layer 3) or ``layers.3.mlp`` (the MLP output); a transcoder (``transcode``) reads
  that MLP's input and predicts its output.
* Gemma Scope 2 (``config.json`` + ``params.safetensors``), for Gemma 3: JumpReLU SAEs on each
  layer's output, its attention heads' outputs (the input of ``o_proj``) and its MLP output, and
  transcoders from each MLP's input to its output. The config names Hugging Face modules, such
  as ``model.layers.12.output``.

Hook and module names are library details, so they are mapped to abstract sites here. Parameters
are read only from safetensors; Gemma Scope's NumPy archives are refused rather than loaded with a
different reader. Anything a format does that Logogram can't reproduce exactly (a learned scaling,
a gated encoder, a skip connection, a transcoder across layers) is refused with the reason.
"""

from __future__ import annotations

import json
import re
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

_TL_HOOK = re.compile(r"blocks\.(\d+)\.hook_(resid_pre|resid_post|attn_out|mlp_out)")
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


@dataclass
class SAEParams:
    """An SAE (or transcoder) as read from its files: what it reads and writes, and how it encodes
    and decodes. An SAE reads and writes the same site; a transcoder reads an MLP's input
    (``mlp_in``) and writes that MLP's output."""

    site: str  # abstract site it writes: resid_pre, resid_post, attn_out, mlp_out or head (z)
    layer: int
    d_in: int
    d_sae: int
    W_enc: torch.Tensor  # [d_in, d_sae]
    b_enc: torch.Tensor  # [d_sae]
    W_dec: torch.Tensor  # [d_sae, d_in]
    b_dec: torch.Tensor  # [d_in]
    activation: str  # relu, topk or jumprelu
    k: int | None
    threshold: torch.Tensor | None  # [d_sae], JumpReLU only
    subtract_b_dec: bool  # subtract b_dec from the input before encoding
    normalize: str  # "none", or "layer_norm": each input standardized, each output restored
    format: str  # saelens, eleuther or gemma_scope_2
    note: str  # what the SAE says about itself, for the app (model name, training hook)
    # The site it reads: the same as ``site`` for an SAE, the MLP's input for a transcoder.
    site_in: str | None = None

    def __post_init__(self) -> None:
        if self.site_in is None:
            self.site_in = self.site

    @property
    def transcoder(self) -> bool:
        return self.site_in != self.site

    @property
    def d_out(self) -> int:
        return int(self.W_dec.shape[1])


def site_of_hook(name: str) -> tuple[str, int]:
    match = _TL_HOOK.fullmatch(name)
    if match is None:
        raise BackendError(
            f"This SAE reads {name}, which Logogram can't map to a residual stream, attention "
            "output or MLP output. SAEs on single heads or on the stream between attention and "
            "MLP aren't supported yet."
        )
    return match.group(2), int(match.group(1))


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


def read_sae(folder: Path, path: str = "") -> SAEParams:
    """Read an SAE (or transcoder) from a local folder holding its config and safetensors
    weights."""
    from safetensors.torch import load_file

    if (folder / "params.npz").exists():
        raise BackendError(
            "This SAE is published as a NumPy archive (params.npz), as Gemma Scope's first release "
            "is. Logogram reads SAE parameters only from safetensors files; Gemma Scope 2 is "
            "published as safetensors."
        )
    weights = next((folder / name for name in WEIGHT_FILES if (folder / name).is_file()), None)
    cfg_path = folder / CONFIG_FILES[weights.name] if weights is not None else folder / "cfg.json"
    if not cfg_path.is_file() or weights is None:
        if (folder / "params_layer_0.safetensors").is_file():
            raise BackendError(
                "This is a cross-layer transcoder or crosscoder (its parameters are split by "
                "layer): it reads or writes several layers at once, which Logogram doesn't "
                "support yet. Choose an SAE or a single-layer transcoder."
            )
        raise BackendError(
            "This folder has no SAE in a format Logogram reads: it needs cfg.json with "
            "sae_weights.safetensors (SAELens) or sae.safetensors (EleutherAI), or config.json "
            "with params.safetensors (Gemma Scope 2)."
        )
    try:
        cfg: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BackendError(f"The SAE's {cfg_path.name} can't be read: {exc}") from exc
    tensors = load_file(str(weights), device="cpu")
    if weights.name == "sae_weights.safetensors":
        return _saelens(cfg, tensors)
    if weights.name == "params.safetensors":
        return _gemma_scope_2(cfg, tensors)
    return _eleuther(cfg, tensors, path or folder.name)


def _gemma_scope_2(cfg: dict[str, Any], tensors: dict[str, torch.Tensor]) -> SAEParams:
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
    if cfg.get("affine_connection"):
        raise BackendError(
            "This transcoder has an affine skip connection, which Logogram doesn't support yet."
        )
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
    note = " · ".join(str(v) for v in (cfg.get("model_name"), point_in) if v)
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
    )


def _need(tensors: dict[str, torch.Tensor], *names: str) -> list[torch.Tensor]:
    missing = [n for n in names if n not in tensors]
    if missing:
        raise BackendError(f"The SAE's weights are missing {', '.join(missing)}.")
    return [tensors[n].float() for n in names]


def _saelens(cfg: dict[str, Any], tensors: dict[str, torch.Tensor]) -> SAEParams:
    hook = cfg.get("hook_name") or cfg.get("hook_point")
    if not isinstance(hook, str):
        raise BackendError("The SAE's cfg.json doesn't say which hook it reads.")
    if cfg.get("hook_head_index", cfg.get("hook_point_head_index")) is not None:
        raise BackendError("SAEs on a single attention head aren't supported yet.")
    site, layer = site_of_hook(hook)
    architecture = cfg.get("architecture", "standard")
    if architecture not in ("standard", "jumprelu", "topk"):
        raise BackendError(f"SAELens {architecture} SAEs aren't supported yet.")
    normalize = cfg.get("normalize_activations") or "none"
    if normalize not in ("none", "layer_norm"):
        raise BackendError(
            f"This SAE rescales its inputs ({normalize}) with a factor measured on its training "
            "data, which Logogram can't reproduce exactly."
        )
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
    note = " · ".join(str(v) for v in (cfg.get("model_name"), hook) if v)
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


def _eleuther(cfg: dict[str, Any], tensors: dict[str, torch.Tensor], folder: str) -> SAEParams:
    for flag, what in (
        ("skip_connection", "SAEs and transcoders with a skip connection"),
        ("signed", "Signed SAEs"),
    ):
        if cfg.get(flag):
            raise BackendError(f"{what} aren't supported yet.")
    site, layer = site_of_module(folder)
    site_in = site
    if cfg.get("transcode"):
        if site != "mlp_out":
            raise BackendError(
                "Logogram reads transcoders from an MLP's input to its output; this one hooks "
                f"{folder}."
            )
        site_in = "mlp_in"  # the MLP module's input (its normalized residual stream)
    encoder, b_enc, W_dec, b_dec = _need(
        tensors, "encoder.weight", "encoder.bias", "W_dec", "b_dec"
    )
    k = int(cfg.get("k") or 0)
    if k <= 0:
        raise BackendError("This TopK SAE doesn't say how many features it keeps (k).")
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
        subtract_b_dec=True,
        normalize="none",
        format="eleuther",
        note=f"EleutherAI sparsify · {folder}" + (" · transcoder" if site_in != site else ""),
        site_in=site_in,
    )


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
    if f"{prefix}params.npz" in sizes:
        raise BackendError(
            "This SAE is published as a NumPy archive (params.npz), as Gemma Scope's first release "
            "is. Logogram reads SAE parameters only from safetensors files; Gemma Scope 2 is "
            "published as safetensors."
        )
    if f"{prefix}params_layer_0.safetensors" in sizes:
        raise BackendError(
            "This is a cross-layer transcoder or crosscoder: it reads or writes several layers at "
            "once, which Logogram doesn't support yet. Choose an SAE or a single-layer transcoder."
        )
    weights = next((w for w in WEIGHT_FILES if f"{prefix}{w}" in sizes), None)
    config = CONFIG_FILES[weights] if weights is not None else "cfg.json"
    if weights is None or f"{prefix}{config}" not in sizes:
        raise BackendError(
            f"{repo} has no SAE at {path or 'its top level'}: Logogram needs cfg.json with "
            "sae_weights.safetensors or sae.safetensors, or config.json with params.safetensors, "
            "there."
        )
    files = hub.RepoFiles(
        revision=str(info.sha),
        files=[
            (f"{prefix}{config}", sizes[f"{prefix}{config}"]),
            (f"{prefix}{weights}", sizes[f"{prefix}{weights}"]),
        ],
        n_params=None,
        gated=bool(info.gated),
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
