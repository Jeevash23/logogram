"""Published sparse autoencoders: find, download and read their files.

Two safetensors formats are read:

* SAELens (``cfg.json`` + ``sae_weights.safetensors``), as in the GPT-2 small residual SAEs. The
  config names the TransformerLens hook the SAE reads, such as ``blocks.8.hook_resid_pre``.
* EleutherAI's sparsify (``cfg.json`` + ``sae.safetensors``), as for Pythia, SmolLM2 and Llama.
  The folder names the module whose output the SAE reads, such as ``layers.3`` (the residual
  stream after layer 3) or ``layers.3.mlp`` (the MLP output).

Hook and module names are library details, so they are mapped to abstract sites here. Parameters
are read only from safetensors; Gemma Scope's NumPy archives are refused rather than loaded with a
different reader. Anything a format does that Logogram can't reproduce exactly (a learned scaling,
a gated encoder, a transcoder) is refused with the reason.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from logogram.backends.base import BackendError

WEIGHT_FILES = ("sae_weights.safetensors", "sae.safetensors")

_TL_HOOK = re.compile(r"blocks\.(\d+)\.hook_(resid_pre|resid_post|attn_out|mlp_out)")
_MODULE = re.compile(r"(?:.*\.)?layers\.(\d+)(?:\.(mlp|attention|self_attn|attn))?")


@dataclass
class SAEParams:
    """An SAE as read from its files: what it reads, and how it encodes and decodes."""

    site: str  # abstract site: resid_pre, resid_post, attn_out or mlp_out
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
    format: str  # saelens or eleuther
    note: str  # what the SAE says about itself, for the app (model name, training hook)


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


def read_sae(folder: Path, path: str = "") -> SAEParams:
    """Read an SAE from a local folder holding its config and safetensors weights."""
    from safetensors.torch import load_file

    if (folder / "params.npz").exists():
        raise BackendError(
            "This SAE is published as a NumPy archive (params.npz), as Gemma Scope is. Logogram "
            "reads SAE parameters only from safetensors files."
        )
    cfg_path = folder / "cfg.json"
    weights = next((folder / name for name in WEIGHT_FILES if (folder / name).is_file()), None)
    if not cfg_path.is_file() or weights is None:
        raise BackendError(
            "This folder has no SAE in a format Logogram reads: it needs cfg.json with "
            "sae_weights.safetensors (SAELens) or sae.safetensors (EleutherAI)."
        )
    try:
        cfg: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BackendError(f"The SAE's cfg.json can't be read: {exc}") from exc
    tensors = load_file(str(weights), device="cpu")
    if weights.name == "sae_weights.safetensors":
        return _saelens(cfg, tensors)
    return _eleuther(cfg, tensors, path or folder.name)


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
        ("transcode", "Transcoders"),
        ("skip_connection", "SAEs with a skip connection"),
        ("signed", "Signed SAEs"),
    ):
        if cfg.get(flag):
            raise BackendError(f"{what} aren't supported yet.")
    site, layer = site_of_module(folder)
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
        note=f"EleutherAI sparsify · {folder}",
    )


# -- the Hub -----------------------------------------------------------------------------------


def list_saes(repo: str, revision: str | None = None) -> tuple[str, list[str]]:
    """The exact revision of an SAE repository and the folders in it that hold an SAE."""
    from huggingface_hub import HfApi

    from logogram.backends.hub import _friendly_hub_error

    try:
        info = HfApi().model_info(repo, revision=revision)
    except Exception as exc:  # noqa: BLE001
        raise _friendly_hub_error(repo, exc) from exc
    names = [s.rfilename for s in info.siblings or []]
    folders = sorted(
        {
            n.rsplit("/", 1)[0] if "/" in n else ""
            for n in names
            if n.rsplit("/", 1)[-1] in WEIGHT_FILES
        },
        key=_natural,
    )
    return str(info.sha), folders


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
    except Exception as exc:  # noqa: BLE001
        if hub._is_offline_error(exc):
            cached = _cached(repo, prefix, revision)
            if cached is not None:
                return cached
        raise hub._friendly_hub_error(repo, exc) from exc
    sizes = {s.rfilename: int(s.size or 0) for s in info.siblings or []}
    if f"{prefix}params.npz" in sizes:
        raise BackendError(
            "This SAE is published as a NumPy archive (params.npz), as Gemma Scope is. Logogram "
            "reads SAE parameters only from safetensors files."
        )
    weights = next((f"{prefix}{w}" for w in WEIGHT_FILES if f"{prefix}{w}" in sizes), None)
    if weights is None or f"{prefix}cfg.json" not in sizes:
        raise BackendError(
            f"{repo} has no SAE at {path or 'its top level'}: Logogram needs cfg.json with "
            "sae_weights.safetensors or sae.safetensors there."
        )
    files = hub.RepoFiles(
        revision=str(info.sha),
        files=[(f"{prefix}cfg.json", sizes[f"{prefix}cfg.json"]), (weights, sizes[weights])],
        n_params=None,
        gated=bool(info.gated),
    )
    folder = hub.download(repo, files, progress, cancel=cancel)
    return str(info.sha), folder


def _cached(repo: str, prefix: str, revision: str | None) -> tuple[str, Path] | None:
    from logogram.backends.hub import cached_file

    found = cached_file(repo, f"{prefix}cfg.json", revision)
    if found is None:
        return None
    commit, path = found
    if not any((path.parent / w).is_file() for w in WEIGHT_FILES):
        return None
    return commit, path.parent
