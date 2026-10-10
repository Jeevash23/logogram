"""Sparse autoencoders (and transcoders) on a model's activations: encode, decode, edit one
feature, and check fit.

An SAE writes an activation x as a sparse sum of decoder directions: x = f · W_dec + b_dec + e,
where f are the feature activations and e is what the SAE misses (its error). Editing a feature
changes x by (new - old) · W_dec[i] and keeps the error as it was, so only that feature changes.
Whether an SAE fits a loaded model at all depends on how it was trained (on which model, and with
or without TransformerLens's weight processing), so Logogram measures the fit on the project's
prompts instead of assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from logogram.backends.saes import SAEParams
from logogram.spec import SAERef

LN_EPS = 1e-5


@dataclass
class SAE:
    """A loaded SAE, its parameters on the model's device, and where it came from."""

    repo: str
    path: str
    revision: str
    params: SAEParams
    device: str = "cpu"
    fit: dict[str, Any] = field(default_factory=dict)

    @property
    def site(self) -> str:
        """The site it writes (and, for an SAE, reads)."""
        return self.params.site

    @property
    def site_in(self) -> str:
        """The site it reads: the MLP's input for a transcoder."""
        return self.params.site_in or self.params.site

    @property
    def transcoder(self) -> bool:
        return self.params.transcoder

    @staticmethod
    def flat(x: torch.Tensor) -> torch.Tensor:
        """Activations as the SAE reads them: heads' outputs ``[..., H, d_head]`` side by side."""
        return x.reshape(*x.shape[:-2], -1) if x.dim() >= 4 else x

    def reads(self, x: torch.Tensor) -> torch.Tensor:
        return self.flat(x) if self.site_in == "head" else x

    @property
    def layer(self) -> int:
        return self.params.layer

    @property
    def d_sae(self) -> int:
        return self.params.d_sae

    def to(self, device: str) -> SAE:
        p = self.params
        for name in ("W_enc", "b_enc", "W_dec", "b_dec", "threshold"):
            value = getattr(p, name)
            if value is not None:
                setattr(p, name, value.to(device))
        self.device = device
        return self

    def describe(self) -> dict[str, Any]:
        p = self.params
        return {
            "repo": self.repo,
            "path": self.path,
            "revision": self.revision,
            "site": p.site,
            "layer": p.layer,
            "d_in": p.d_in,
            "d_sae": p.d_sae,
            "activation": p.activation,
            "k": p.k,
            "normalize": p.normalize,
            "format": p.format,
            "note": p.note,
            "site_in": self.site_in,
            "transcoder": self.transcoder,
            "fit": self.fit or None,
        }

    # -- the map --------------------------------------------------------------------------------

    def _standardize(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        if self.params.normalize != "layer_norm":
            return x, None
        mu = x.mean(dim=-1, keepdim=True)
        centered = x - mu
        std = centered.std(dim=-1, keepdim=True)
        return centered / (std + LN_EPS), (mu, std)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, Any]:
        """Feature activations ``[..., d_sae]`` of activations ``[..., d_in]``, and what
        decoding needs to restore a standardized input (or None)."""
        p = self.params
        x = x.float()
        sae_in, stats = self._standardize(x)
        if p.subtract_b_dec:
            sae_in = sae_in - p.b_dec
        pre = sae_in @ p.W_enc + p.b_enc
        if p.activation == "relu":
            f = torch.relu(pre)
        elif p.activation == "jumprelu":
            assert p.threshold is not None
            f = torch.relu(pre) * (pre > p.threshold)
        else:
            assert p.k is not None
            top = pre.topk(min(p.k, pre.shape[-1]), dim=-1, sorted=False)
            f = torch.zeros_like(pre).scatter(-1, top.indices, torch.relu(top.values))
        return f, stats

    def decode(self, f: torch.Tensor, stats: Any) -> torch.Tensor:
        p = self.params
        out = f @ p.W_dec + p.b_dec
        if stats is not None:
            mu, std = stats
            out = out * std + mu
        return out

    def feature_direction(self, feature: int, stats: Any = None) -> torch.Tensor:
        """What one unit of the feature adds to the activation: its decoder row, scaled back by
        each input's standard deviation for SAEs that standardize their inputs."""
        row = self.params.W_dec[feature]
        if stats is None:
            return row
        _, std = stats
        return row * std

    # -- fit ------------------------------------------------------------------------------------


def fit_on(
    sae: SAE, activations: torch.Tensor, targets: torch.Tensor | None = None
) -> dict[str, float]:
    """How well the SAE reconstructs ``activations`` ``[N, d_in]`` (a transcoder: predicts
    ``targets`` ``[N, d_out]``, its MLP's outputs, from its inputs): the fraction of variance it
    explains (1 is perfect, 0 is no better than the mean), and how many features fire per token."""
    x = activations.float()
    y = x if targets is None else targets.float()
    f, stats = sae.encode(x)
    y_hat = sae.decode(f, stats)
    residual = ((y - y_hat) ** 2).sum()
    total = ((y - y.mean(dim=0, keepdim=True)) ** 2).sum().clamp_min(1e-12)
    return {
        "variance_explained": float(1.0 - residual / total),
        "l0": float((f > 0).float().sum(-1).mean()),
        "tokens": int(x.shape[0]),
    }


def load_sae(
    ref: SAERef,
    device: str,
    progress: Any = None,
    cancel: Any = None,
) -> SAE:
    """Download (once) and read a published SAE at an exact revision, on ``device``."""
    from logogram.backends.saes import download_sae, read_sae

    revision, folder = download_sae(ref.repo, ref.path, ref.revision, progress, cancel)
    params = read_sae(folder, ref.path)
    return SAE(repo=ref.repo, path=ref.path, revision=revision, params=params).to(device)


def sae_matches(sae: SAE | None, ref: SAERef) -> bool:
    return (
        sae is not None
        and sae.repo == ref.repo
        and sae.path.strip("/") == ref.path.strip("/")
        and (ref.revision is None or ref.revision == sae.revision)
    )
