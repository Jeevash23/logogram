"""TransformerLens backend, built on ``TransformerBridge`` (TransformerLens 4).

Abstract sites map to TransformerLens hook points here and nowhere else.
"""

from __future__ import annotations

import logging
import threading
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
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

# Load-time checks compare numbers, so a short fixed input is enough. The tolerance is relative to
# the largest value compared, and allows for the rounding of each precision.
CHECK_TOLERANCE = {"float32": 1e-4, "float16": 2e-2, "bfloat16": 5e-2}


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


@dataclass
class ModelChecks:
    """What Logogram measured about a model when it loaded (see :func:`check_model`)."""

    tolerance: float
    # Largest change in the log-probabilities against the original model (relative to their range).
    function: float
    # "sequential" (attention, then MLP, each added to the residual stream), "parallel" (both read
    # the same residual and are added together) or "components" (neither could be verified).
    structure: str
    # Largest error in the residual additions of that structure, if one was verified.
    residual: float | None
    # Error of the final-norm logit lens at the last layer against the model's output, if defined.
    lens: float | None
    # Error of each head's output (z through its slice of W_O) summed with b_O against the attention
    # output: small when heads add up to what attention writes, large when the model normalizes
    # after combining them (Gemma 2, OLMo 2). None if it couldn't be measured.
    heads: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _probe_tokens(d_vocab: int, device: str) -> torch.Tensor:
    rows = [[(7 * i + 3) % d_vocab for i in range(8)], [(5 * i + 11) % d_vocab for i in range(8)]]
    return torch.tensor(rows, dtype=torch.long, device=device)


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-6))


def _log_prob_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return _relative(torch.log_softmax(a.float(), dim=-1), torch.log_softmax(b.float(), dim=-1))


def _soft_cap(cfg: Any) -> float | None:
    cap = float(getattr(cfg, "output_logits_soft_cap", 0) or 0)
    return cap if cap > 0 else None


def final_projection(bridge: Any, residual: torch.Tensor) -> torch.Tensor:
    """What the model does after its last layer: final normalization, unembedding, and the logit
    soft-capping some models (Gemma 2) apply outside both."""
    logits = bridge.unembed(bridge.ln_final(residual))
    cap = _soft_cap(bridge.cfg)
    if cap is not None:
        logits = cap * torch.tanh(logits / cap)
    return logits


def check_model(
    bridge: Any,
    probe: torch.Tensor,
    reference: torch.Tensor,
    n_layers: int,
    kinds: tuple[str, ...],
    dtype: str,
) -> ModelChecks:
    """Check, on a short input, what Logogram's measurements assume about a loaded model.

    TransformerLens supports many architectures, and processes some weights; rather than trusting
    a list, this verifies the model in front of it: that TransformerLens's version predicts what the
    original model predicts, how each layer adds attention and MLP into the residual stream, and
    whether the final normalization and unembedding reproduce the output (the logit lens).
    """
    tolerance = CHECK_TOLERANCE[dtype]
    stream = [
        k for k in ("resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out") if k in kinds
    ]
    names = [hook_name(k, layer) for layer in range(n_layers) for k in stream]
    if "head" in kinds:
        names += [hook_name("head", layer) for layer in range(n_layers)]
    with torch.no_grad():
        logits, cache = bridge.run_with_cache(probe, names_filter=names)

    def act(kind: str, layer: int) -> torch.Tensor:
        return cache[hook_name(kind, layer)].float()

    structure, residual = "components", None
    if {"resid_pre", "resid_post", "attn_out", "mlp_out"} <= set(stream):
        chain = max(
            (_relative(act("resid_pre", i + 1), act("resid_post", i)) for i in range(n_layers - 1)),
            default=0.0,
        )
        parallel = max(
            _relative(
                act("resid_pre", i) + act("attn_out", i) + act("mlp_out", i), act("resid_post", i)
            )
            for i in range(n_layers)
        )
        sequential = None
        if "resid_mid" in stream:
            sequential = max(
                max(
                    _relative(act("resid_pre", i) + act("attn_out", i), act("resid_mid", i)),
                    _relative(act("resid_mid", i) + act("mlp_out", i), act("resid_post", i)),
                )
                for i in range(n_layers)
            )
        if sequential is not None and max(sequential, chain) <= tolerance:
            structure, residual = "sequential", max(sequential, chain)
        elif max(parallel, chain) <= tolerance:
            structure, residual = "parallel", max(parallel, chain)

    heads = None
    if "head" in kinds and "attn_out" in stream:
        try:
            errors = []
            for i in range(n_layers):
                attention = bridge.blocks[i].attn
                z = cache[hook_name("head", i)].float()
                with torch.no_grad():
                    out = torch.einsum("bphd,hdm->bpm", z, attention.W_O.float())
                    if getattr(attention, "b_O", None) is not None:
                        out = out + attention.b_O.float()
                errors.append(_relative(out, act("attn_out", i)))
            heads = max(errors)
        except Exception:  # noqa: BLE001 - no per-head output weights to check with
            heads = None

    lens = None
    if "resid_post" in stream:
        try:
            with torch.no_grad():
                final = cache[hook_name("resid_post", n_layers - 1)]
                lens = _log_prob_error(final_projection(bridge, final), logits)
        except Exception:  # noqa: BLE001 - no usable final norm or unembedding
            lens = None
    return ModelChecks(
        tolerance=tolerance,
        function=_log_prob_error(logits, reference),
        structure=structure,
        residual=residual,
        lens=lens,
        heads=heads,
    )


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
        cfg = bridge.cfg
        probe = _probe_tokens(int(cfg.d_vocab), str(cfg.device))
        with torch.no_grad():
            # The model's own predictions, before TransformerLens touches its weights.
            reference = bridge.original_model(probe).logits.float()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bridge.enable_compatibility_mode(
                disable_warnings=True, no_processing=not process_weights
            )
        device = torch.device(str(cfg.device)).type
        kinds = tuple(k for k in ALL_KINDS if hook_name(k, 0) in bridge.hook_dict)
        if "head" not in kinds or PATTERN_HOOK.format(layer=0) not in bridge.hook_dict:
            raise BackendError(
                f"{model_id} doesn't expose per-head attention hooks in TransformerLens, so "
                "Logogram can't run head experiments on it."
            )
        checks = check_model(bridge, probe, reference, int(cfg.n_layers), kinds, dtype)
        if checks.function > checks.tolerance:
            if process_weights:
                raise BackendError(
                    f"Processing the weights of {model_id} changed its predictions "
                    f"(log-probabilities moved by up to {checks.function:.1%} of their range), so "
                    "results wouldn't describe the original model. Load it with weight "
                    "processing off."
                )
            raise BackendError(
                f"TransformerLens's version of {model_id} doesn't reproduce the model's own "
                f"predictions (log-probabilities differ by up to {checks.function:.1%} of their "
                "range), so Logogram can't measure it reliably."
            )
        if not checks.structure.startswith("sequential"):
            # resid_mid is patched as resid_pre + attn_out (see _resid_mid_hooks), which is only
            # the residual stream between attention and MLP when the layer adds them in turn.
            kinds = tuple(k for k in kinds if k != "resid_mid")
        if n_params is None:
            try:
                n_params = int(bridge.n_params_total)
            except Exception:  # noqa: BLE001
                n_params = None
        model_type = getattr(bridge.original_model.config, "model_type", None)
        structure = checks.structure
        if structure == "sequential" and model_type == "gpt2":
            structure = "sequential_pre_norm"  # GPT-2's layout, normalization included, is known
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
                "block_structure": structure,
                "normalization": str(getattr(cfg, "normalization_type", "unknown")),
                "activation": str(getattr(cfg, "act_fn", "unknown")),
                "prediction_method": "final_norm_logit_lens"
                if checks.lens is not None and checks.lens <= checks.tolerance
                else None,
                "model_type": str(model_type or "unknown"),
                "n_key_value_heads": int(getattr(cfg, "n_key_value_heads", None) or cfg.n_heads),
                "bos": getattr(bridge.tokenizer, "bos_token_id", None) is not None,
                # TransformerLens marks "no soft-cap" with a value of zero or below.
                "logit_soft_cap": _soft_cap(cfg),
                "checks": checks.to_dict(),
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

    def gradients(
        self,
        tokens: torch.Tensor,
        answers: torch.Tensor,
        distractors: torch.Tensor,
        sites: list[tuple[str, int]],
    ) -> tuple[dict[tuple[str, int], torch.Tensor], dict[tuple[str, int], torch.Tensor]]:
        # A zero tensor added at each hook point: the gradient with respect to it is the gradient
        # with respect to the activation there, through every later use, without cutting the graph.
        # hook_resid_mid only feeds the MLP's normalization (see _resid_mid_hooks); the residual
        # stream between attention and MLP is resid_pre + attn_out, so its gradient is taken at
        # attn_out, which is added to it.
        kept: dict[tuple[str, int], torch.Tensor] = {}
        zeros: dict[tuple[str, int], torch.Tensor] = {}

        def value(site: tuple[str, int]) -> Callable[..., torch.Tensor]:
            def fn(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
                kept[site] = act.detach()
                return act

            return fn

        def probe(site: tuple[str, int]) -> Callable[..., torch.Tensor]:
            def fn(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
                zero = torch.zeros_like(act, requires_grad=True)
                zeros[site] = zero
                if site[0] != "resid_mid":
                    kept[site] = act.detach()
                return act + zero

            return fn

        hooks: list[tuple[str, Callable[..., torch.Tensor]]] = []
        for kind, layer in dict.fromkeys(sites):
            if kind == "resid_mid":
                hooks.append((hook_name("resid_mid", layer), value((kind, layer))))
                hooks.append((hook_name("attn_out", layer), probe((kind, layer))))
            else:
                hooks.append((hook_name(kind, layer), probe((kind, layer))))
        kwargs: dict[str, Any] = {"return_type": "logits"}
        if self._logits_to_keep:
            kwargs["logits_to_keep"] = 1
        with self.lock, torch.enable_grad():
            bridge = self._bridge()
            logits = bridge.run_with_hooks(tokens.to(self.device), fwd_hooks=hooks, **kwargs)
            last = logits[:, -1, :].float()
            rows = torch.arange(last.shape[0], device=last.device)
            ld = last[rows, answers.to(last.device)] - last[rows, distractors.to(last.device)]
            order = list(zeros)
            grads = torch.autograd.grad(ld.sum(), [zeros[s] for s in order])
        return kept, {site: g.detach() for site, g in zip(order, grads, strict=True)}

    def direct_effects(
        self,
        tokens: torch.Tensor,
        answers: torch.Tensor,
        distractors: torch.Tensor,
        heads: bool,
    ) -> dict[str, torch.Tensor]:
        n = self.info.n_layers
        kept: dict[str, torch.Tensor] = {}

        def keep(name: str) -> Callable[..., torch.Tensor]:
            def fn(act: torch.Tensor, hook: Any = None) -> torch.Tensor:
                kept[name] = act[:, -1].detach()  # only the last position is read out
                return act

            return fn

        names = [hook_name("resid_pre", 0), hook_name("resid_post", n - 1)]
        names += [hook_name(k, layer) for layer in range(n) for k in ("attn_out", "mlp_out")]
        if heads:
            names += [hook_name("head", layer) for layer in range(n)]
        with self.lock:
            bridge = self._bridge()
            with torch.no_grad():
                bridge.run_with_hooks(
                    tokens.to(self.device),
                    fwd_hooks=[(name, keep(name)) for name in names],
                    return_type=None,
                )

            # The direction in the residual stream that the logit difference reads, with the final
            # normalization's scale held at its value for each prompt: the logit difference is then
            # an affine function of the residual stream, and splits over its components.
            def hold(scale: torch.Tensor, hook: Any = None) -> torch.Tensor:
                return scale.detach()

            with torch.enable_grad(), warnings.catch_warnings():
                # TransformerLens warns that an edited scale is recomputed from the hooked values;
                # that is the point, and the result is checked against the measured logit difference.
                warnings.simplefilter("ignore")
                final = kept[hook_name("resid_post", n - 1)].clone().requires_grad_(True)
                with bridge.hooks(fwd_hooks=[("ln_final.hook_scale", hold)]):
                    logits = final_projection(bridge, final[:, None, :])[:, 0]
                rows = torch.arange(final.shape[0], device=final.device)
                ld = (
                    logits[rows, answers.to(final.device)]
                    - logits[rows, distractors.to(final.device)]
                )
                (direction,) = torch.autograd.grad(ld.sum(), final)
            g = direction.double()

            def term(vector: torch.Tensor) -> torch.Tensor:
                return (vector.double() * g).sum(-1)

            out: dict[str, torch.Tensor] = {
                "embed": term(kept[hook_name("resid_pre", 0)]),
                "attn_out": torch.stack(
                    [term(kept[hook_name("attn_out", layer)]) for layer in range(n)], dim=1
                ),
                "mlp_out": torch.stack(
                    [term(kept[hook_name("mlp_out", layer)]) for layer in range(n)], dim=1
                ),
                "logit_diff": ld.detach().double(),
            }
            if heads:
                per_layer = []
                for layer in range(n):
                    z = kept[hook_name("head", layer)].double()  # [B, H, d_head]
                    w_o = bridge.blocks[layer].attn.W_O.detach().double()  # [H, d_head, d_model]
                    written = torch.einsum("bhd,hdm->bhm", z, w_o)
                    per_layer.append((written * g[:, None, :]).sum(-1))
                out["head"] = torch.stack(per_layer, dim=1)
        total = out["embed"] + out["attn_out"].sum(1) + out["mlp_out"].sum(1)
        out["remainder"] = out["logit_diff"] - total
        return {k: v.cpu() for k, v in out.items()}

    def layer_logits(self, tokens: torch.Tensor, position: int, row: int) -> torch.Tensor:
        if self.info.extra.get("prediction_method") != "final_norm_logit_lens":
            raise BackendError(
                "Per-layer predictions need the model's final normalization and unembedding to "
                "reproduce its output, and for this model they didn't when it loaded. The other "
                "analyses remain available."
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
                    logits = final_projection(bridge, residual)
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


def architecture_support(architecture: str) -> str | None:
    """None if TransformerLens can load this Hugging Face architecture; otherwise, why not."""
    try:
        import transformer_lens.model_bridge  # noqa: F401 - loads before the factory (a cycle)
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )
    except Exception:  # noqa: BLE001 - can't tell; loading the model will say
        return None
    if architecture == "unknown" or architecture in SUPPORTED_ARCHITECTURES:
        return None
    return (
        f"TransformerLens {version('transformer-lens')} can't load {architecture} models, so "
        "Logogram can't either. Choose a model of a supported family, such as GPT-2, Llama, "
        "Qwen, Gemma, Pythia or OLMo."
    )


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
