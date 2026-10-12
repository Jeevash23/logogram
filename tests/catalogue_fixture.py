"""A small catalogue of published SAEs for a tiny model, built in test code: the shape of the one
Logogram ships (see logogram.backends.catalogue), with SAEs it reads exactly, one it refuses and
one it reads only once loaded. The browser checks serve it in place of the real one, and serve
the SAE ``local/exact`` (folder ``exact``) from a folder in place of Hugging Face."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from logogram.backends.catalogue import Catalogue, parse_catalogue

REVISION = "0123456789abcdef0123456789abcdef01234567"
PICKLE = (
    "Its weights are a PyTorch pickle (a .pt file), which Logogram never opens: loading a pickle "
    "can run code hidden in it. Logogram reads SAEs only from safetensors files."
)
UNKNOWN = (
    "Its configuration couldn't be read when the catalogue was made; Logogram checks it when you "
    "load it."
)


def catalogue_data(model_id: str, d_model: int) -> dict[str, Any]:
    def sae(sae_id: str, folder: str, site: str, layer: int, width: int, **more: Any) -> dict:
        return {
            "id": sae_id,
            "folder": folder,
            "hook": more.pop("hook", f"blocks.{layer}.hook_{site}"),
            "site": site,
            "layer": layer,
            "width": width,
            "architecture": more.pop("architecture", "standard"),
            "kind": more.pop("kind", "sae"),
            "reads": more.pop("reads", "exact"),
            "why": more.pop("why", None),
            **more,
        }

    def release(name: str, repo: str, saes: list[dict], **more: Any) -> dict:
        return {
            "release": name,
            "model": "tiny",
            "hf_models": [model_id, "tiny-alias"],
            "repo": repo,
            "revision": REVISION,
            "licence": more.pop("licence", "mit"),
            "gated": more.pop("gated", False),
            "saes": saes,
        }

    return {
        "catalogue": 1,
        "source": {
            "package": "sae-lens",
            "version": "0.0.test",
            "file": "sae_lens/pretrained_saes.yaml",
            "sha256": "0" * 64,
            "licence": "MIT",
            "notice": "MIT License\n\nCopyright (c) a test\n\nPermission is hereby granted.\n",
        },
        "reasons": [PICKLE, UNKNOWN],
        "releases": [
            release(
                "tiny-res",
                "local/exact",
                [
                    sae(
                        "blocks.1.hook_resid_pre",
                        "exact",
                        "resid_pre",
                        1,
                        2 * d_model,
                        l0=12.0,
                        verified={"revision": REVISION, "read": "weights", "fit": None},
                    ),
                    sae("blocks.0.hook_resid_pre", "layer0", "resid_pre", 0, 16384),
                ],
            ),
            release(
                "tiny-attn-pickled",
                "local/pickled",
                [
                    sae(
                        "blocks.1.hook_z",
                        "blocks.1.hook_z.pt",
                        "head",
                        1,
                        24576,
                        hook="blocks.1.attn.hook_z",
                        reads="refused",
                        why=0,
                    )
                ],
                licence=None,
            ),
            release(
                "tiny-transcoders",
                "local/transcoders",
                [
                    sae(
                        "layer_0_width_64k",
                        "transcoder/layer_0_width_64k",
                        "mlp_out",
                        0,
                        65536,
                        hook="model.layers.0.pre_feedforward_layernorm.output",
                        architecture="jump_relu",
                        kind="skip_transcoder",
                    ),
                    sae(
                        "layer_1_width_64k",
                        "transcoder/layer_1_width_64k",
                        "mlp_out",
                        1,
                        65536,
                        hook="model.layers.1.pre_feedforward_layernorm.output",
                        architecture="jump_relu",
                        kind="transcoder",
                        reads="unknown",
                        why=1,
                    ),
                ],
                gated=True,
            ),
        ],
    }


def fixture_catalogue(model_id: str, d_model: int) -> Catalogue:
    return parse_catalogue(catalogue_data(model_id, d_model))


def serve_published(monkeypatch_setattr: Any, root: Path, d_model: int) -> None:
    """Serve ``local/exact`` (folder ``exact``, an SAE that reconstructs layer 1's residual
    stream exactly) from ``root`` in place of Hugging Face, at the catalogue's revision."""
    from logogram.backends import saes
    from test_sae import exact_sae

    exact_sae(root / "exact", d_model)
    real = saes.download_sae

    def download(repo: str, path: str, revision: str | None, progress=None, cancel=None):
        if repo == "local/exact" and path.strip("/") == "exact":
            return REVISION, root / "exact"
        return real(repo, path, revision, progress, cancel)

    monkeypatch_setattr(saes, "download_sae", download)
