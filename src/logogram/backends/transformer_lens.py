"""TransformerLens backend, built on ``TransformerBridge`` (TransformerLens 4).

Abstract sites map to TransformerLens hook points here and nowhere else.
"""

from __future__ import annotations

import logging
import threading
import warnings
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from logogram.backends.base import (
    ALL_KINDS,
    BackendError,
    Cancelled,
    ModelBackend,
    ModelInfo,
    Patch,
    Tokenized,
)

log = logging.getLogger(__name__)

HOOKS: dict[str, str] = {
    "resid_pre": "blocks.{layer}.hook_resid_pre",
    "resid_mid": "blocks.{layer}.hook_resid_mid",
    "resid_post": "blocks.{layer}.hook_resid_post",
    "attn_out": "blocks.{layer}.hook_attn_out",
    "mlp_out": "blocks.{layer}.hook_mlp_out",
    "head": "blocks.{layer}.attn.hook_z",
}
PATTERN_HOOK = "blocks.{layer}.attn.hook_pattern"

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def hook_name(kind: str, layer: int) -> str:
    return HOOKS[kind].format(layer=layer)


def _resid_mid_hooks(patch: Patch) -> list[tuple[str, Callable[..., torch.Tensor]]]:
    """Patch the residual stream between attention and MLP.

    TransformerLens exposes ``hook_resid_mid`` as the input of the MLP's LayerNorm, so editing it
    would change only what the MLP reads while the residual stream kept its old value. Instead,
    since resid_mid = resid_pre + attn_out, set attn_out to (target - resid_pre) at the patched
    positions, using this run's own resid_pre.
    """
    seen: dict[str, torch.Tensor] = {}

    def keep_resid_pre(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
        seen["resid_pre"] = act
        return act

    def set_attn_out(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
        pre = seen["resid_pre"]
        act = act.clone()
        values = patch.values.to(dtype=act.dtype, device=act.device)
        if patch.positions is None:
            act[...] = values - pre.to(act.dtype)
        else:
            rows = torch.arange(act.shape[0], device=act.device)
            pos = patch.positions.to(act.device)
            act[rows, pos] = values - pre[rows, pos].to(act.dtype)
        return act

    return [
        (hook_name("resid_pre", patch.layer), keep_resid_pre),
        (hook_name("attn_out", patch.layer), set_attn_out),
    ]


def _patch_hook(patch: Patch) -> Callable[..., torch.Tensor]:
    def fn(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
        act = act.clone()
        values = patch.values.to(dtype=act.dtype, device=act.device)
        rows = torch.arange(act.shape[0], device=act.device)
        if patch.kind == "head":
            heads = patch.heads.to(act.device)  # type: ignore[union-attr]
            if patch.positions is None:
                act[rows, :, heads] = values  # [B, pos, d_head]
            else:
                act[rows, patch.positions.to(act.device), heads] = values  # [B, d_head]
        elif patch.positions is None:
            act[...] = values  # [B, pos, d_model]
        else:
            act[rows, patch.positions.to(act.device)] = values  # [B, d_model]
        return act

    return fn


class TransformerLensBackend(ModelBackend):
    def __init__(self, bridge: Any, info: ModelInfo) -> None:
        super().__init__()
        self.bridge = bridge
        self.info = info
        self.tokenizer = bridge.tokenizer
        self._logits_to_keep = self._supports_logits_to_keep()

    # -- construction ------------------------------------------------------------------------

    @classmethod
    def from_bridge(
        cls,
        bridge: Any,
        *,
        model_id: str,
        revision: str | None,
        dtype: str,
        process_weights: bool,
        n_params: int | None = None,
    ) -> TransformerLensBackend:
        bridge.eval()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bridge.enable_compatibility_mode(
                disable_warnings=True, no_processing=not process_weights
            )
        cfg = bridge.cfg
        device = torch.device(str(cfg.device)).type
        kinds = tuple(k for k in ALL_KINDS if hook_name(k, 0) in bridge.hook_dict)
        if "resid_mid" in kinds and not {"resid_pre", "attn_out"} <= set(kinds):
            # resid_mid is patched through resid_pre and attn_out (see _resid_mid_hooks).
            kinds = tuple(k for k in kinds if k != "resid_mid")
        if "head" not in kinds or PATTERN_HOOK.format(layer=0) not in bridge.hook_dict:
            raise BackendError(
                f"{model_id} doesn't expose per-head attention hooks in TransformerLens, so "
                "Logogram can't run head experiments on it."
            )
        if n_params is None:
            try:
                n_params = int(bridge.n_params_total)
            except Exception:  # noqa: BLE001
                n_params = None
        info = ModelInfo(
            id=model_id,
            revision=revision,
            architecture=str(getattr(cfg, "architecture", None) or "unknown"),
            n_layers=int(cfg.n_layers),
            n_heads=int(cfg.n_heads),
            d_model=int(cfg.d_model),
            d_head=int(cfg.d_head),
            d_mlp=int(cfg.d_mlp) if getattr(cfg, "d_mlp", None) else None,
            d_vocab=int(cfg.d_vocab),
            n_ctx=int(cfg.n_ctx),
            n_params=n_params,
            dtype=dtype,
            device=device,
            device_name=_device_name(device),
            process_weights=process_weights,
            site_kinds=kinds,
            backend="transformer_lens",
            backend_version=version("transformer-lens"),
            extra={
                "block_structure": "sequential_pre_norm"
                if getattr(bridge.original_model.config, "model_type", None) == "gpt2"
                else "components",
                "normalization": str(getattr(cfg, "normalization_type", "unknown")),
                "activation": str(getattr(cfg, "act_fn", "unknown")),
                "prediction_method": "final_norm_logit_lens"
                if getattr(bridge.original_model.config, "model_type", None) == "gpt2"
                else None,
            },
        )
        return cls(bridge, info)

    def _supports_logits_to_keep(self) -> bool:
        probe = torch.tensor([[self._bos_or_zero(), self._bos_or_zero()]], device=self.device)
        try:
            with torch.no_grad():
                out = self.bridge(probe, return_type="logits", logits_to_keep=1)
            return tuple(out.shape[:2]) == (1, 1)
        except Exception:  # noqa: BLE001 - architecture without logits_to_keep
            return False

    def _bos_or_zero(self) -> int:
        bos = getattr(self.tokenizer, "bos_token_id", None)
        return int(bos) if bos is not None else 0

    # -- tokens ------------------------------------------------------------------------------

    def tokenize(self, text: str, prepend_bos: bool) -> Tokenized:
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(enc["input_ids"])
        offsets = [tuple(o) for o in enc["offset_mapping"]]
        if prepend_bos:
            bos = getattr(self.tokenizer, "bos_token_id", None)
            if bos is None:
                raise BackendError(
                    "This model's tokenizer has no beginning-of-sequence token. Set "
                    "tokenization.prepend_bos to false in the spec."
                )
            ids = [int(bos), *ids]
            offsets = [(0, 0), *offsets]
        tokens = [self.token_str(i) for i in ids]
        return Tokenized(ids=ids, tokens=tokens, offsets=offsets)  # type: ignore[arg-type]

    def single_token_id(self, text: str) -> int | None:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return int(ids[0]) if len(ids) == 1 else None

    def token_str(self, token_id: int) -> str:
        return self.tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)

    # -- forward passes ----------------------------------------------------------------------

    def final_logits(self, tokens: torch.Tensor, patch: Patch | None = None) -> torch.Tensor:
        kwargs: dict[str, Any] = {"return_type": "logits"}
        if self._logits_to_keep:
            kwargs["logits_to_keep"] = 1
        with self.lock, torch.no_grad():
            bridge = self._bridge()
            tokens = tokens.to(self.device)
            if patch is None:
                logits = bridge(tokens, **kwargs)
            else:
                if patch.kind == "resid_mid":
                    hooks = _resid_mid_hooks(patch)
                else:
                    hooks = [(hook_name(patch.kind, patch.layer), _patch_hook(patch))]
                logits = bridge.run_with_hooks(tokens, fwd_hooks=hooks, **kwargs)
            return logits[:, -1, :].float()

    def capture(
        self, tokens: torch.Tensor, sites: list[tuple[str, int]]
    ) -> dict[tuple[str, int], torch.Tensor]:
        names = {hook_name(kind, layer): (kind, layer) for kind, layer in sites}
        last = max(layer for _, layer in sites)
        kwargs: dict[str, Any] = {}
        if last + 1 < self.info.n_layers:
            kwargs["stop_at_layer"] = last + 1
        with self.lock, torch.no_grad():
            _, cache = self._bridge().run_with_cache(
                tokens.to(self.device), names_filter=list(names), **kwargs
            )
            return {site: cache[name].detach() for name, site in names.items()}

    def attention_pattern(self, tokens: torch.Tensor, layer: int) -> torch.Tensor:
        name = PATTERN_HOOK.format(layer=layer)
        kwargs: dict[str, Any] = {}
        if layer + 1 < self.info.n_layers:
            kwargs["stop_at_layer"] = layer + 1
        with self.lock, torch.no_grad():
            _, cache = self._bridge().run_with_cache(
                tokens.to(self.device), names_filter=[name], **kwargs
            )
            return cache[name].detach().float()

    def _bridge(self) -> Any:
        if self.bridge is None:
            raise BackendError("The model was unloaded. Load it again to continue.")
        return self.bridge

    def layer_logits(self, tokens: torch.Tensor, position: int, row: int) -> torch.Tensor:
        if self.info.extra.get("prediction_method") != "final_norm_logit_lens":
            raise BackendError(
                "Per-layer predictions are currently validated for GPT-2. Use a GPT-2 model "
                "for this diagnostic; the other analyses remain available."
            )
        if not 0 <= position < tokens.shape[1] or not 0 <= row < tokens.shape[0]:
            raise ValueError("The prediction token position or batch row is out of range.")
        with self.lock, torch.no_grad():
            bridge = self._bridge()
            captured: dict[int, torch.Tensor] = {}

            def project(layer: int) -> Callable[..., torch.Tensor]:
                def keep(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
                    # Recompute final normalization at each layer. Cached final-layer scales
                    # would implement attribution, not the logit lens. Keep the original
                    # batch shape and dtype through both modules, including learned biases.
                    residual = act[:, position : position + 1, :]
                    logits = bridge.unembed(bridge.ln_final(residual))
                    captured[layer] = logits[row, 0].detach().float().cpu()
                    return act

                return keep

            hooks = [
                (hook_name("resid_post", layer), project(layer))
                for layer in range(self.info.n_layers)
            ]
            bridge.run_with_hooks(tokens.to(self.device), fwd_hooks=hooks, return_type=None)
            if len(captured) != self.info.n_layers:
                raise BackendError("This model didn't expose every layer's residual output.")
            return torch.stack([captured[layer] for layer in range(self.info.n_layers)])

    def close(self) -> None:
        # Waits for a forward pass in progress; callers hold the lock across multi-step work.
        with self.lock:
            self.bridge = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _device_name(device: str) -> str:
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.cuda.current_device())
    if device == "mps":
        return "Apple GPU (Metal)"
    from logogram.system import cpu_name

    return cpu_name()


def resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise BackendError(
            "The spec asks for CUDA, but PyTorch can't see a CUDA GPU here. Run `logogram doctor` "
            "for the fix, or set model.device to auto or cpu."
        )
    if device == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise BackendError(
            "The spec asks for Apple's MPS, which isn't available here. Set model.device to auto "
            "or cpu."
        )
    return device


def load_model(
    model_id: str,
    *,
    revision: str | None = None,
    dtype: str = "float32",
    device: str = "auto",
    process_weights: bool = True,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    cancel: threading.Event | None = None,
) -> TransformerLensBackend:
    """Resolve, download (with progress) and load a model from the Hugging Face Hub.

    ``cancel`` stops a download promptly and a load at the next step; it raises ``Cancelled``.
    """
    from logogram.backends import hub

    def emit(stage: str, **data: Any) -> None:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        if on_progress:
            on_progress({"stage": stage, **data})

    if dtype not in DTYPES:
        raise BackendError(f"Unknown dtype {dtype!r}; use float32, float16 or bfloat16.")
    resolved_device = resolve_device(device)
    emit("resolving", model_id=model_id)
    repo = hub.resolve(model_id, revision)
    emit("downloading", done=0, total=sum(s for _, s in repo.files), file="")

    def on_download(done: int, total: int, file: str) -> None:
        emit("downloading", done=done, total=total, file=file)

    folder = hub.download(model_id, repo, on_download, cancel=cancel)
    emit("loading", revision=repo.revision)
    bridge = boot_local(folder, device=resolved_device, dtype=dtype)
    emit("processing")
    backend = TransformerLensBackend.from_bridge(
        bridge,
        model_id=model_id,
        revision=repo.revision,
        dtype=dtype,
        process_weights=process_weights,
    )
    if cancel is not None and cancel.is_set():
        backend.close()
        raise Cancelled()
    emit("ready")
    return backend


def boot_local(folder: Path, *, device: str, dtype: str) -> Any:
    from transformer_lens.model_bridge import TransformerBridge
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from logogram.backends.hub import validate_local_weights

    _quiet_transformers()
    validate_local_weights(folder)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # The bridge otherwise lets Transformers fall back to pickle weights. Load both
            # objects explicitly and locally so neither a cache nor an index can change that.
            model = AutoModelForCausalLM.from_pretrained(
                str(folder),
                use_safetensors=True,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=DTYPES[dtype],
                attn_implementation="eager",
            )
            model = model.to(device)
            tokenizer = AutoTokenizer.from_pretrained(
                str(folder),
                local_files_only=True,
                trust_remote_code=False,
            )
            return TransformerBridge.boot_transformers(
                str(folder),
                device=device,
                dtype=DTYPES[dtype],
                hf_model=model,
                tokenizer=tokenizer,
                trust_remote_code=False,
            )
    except torch.OutOfMemoryError as exc:
        raise BackendError(
            "The model doesn't fit in GPU memory. Choose a smaller model, a 16-bit dtype, or "
            "the CPU."
        ) from exc
    except BackendError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise BackendError(
            f"TransformerLens couldn't load this model ({type(exc).__name__}: {exc}). Models "
            "TransformerLens supports are listed in its documentation."
        ) from exc


def _quiet_transformers() -> None:
    try:
        import transformers

        transformers.utils.logging.set_verbosity_error()
        transformers.utils.logging.disable_progress_bar()
    except Exception:  # noqa: BLE001
        pass
