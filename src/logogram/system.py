"""Hardware and environment report, with concrete fixes for common problems.

Used by the first-run system check, by ``logogram doctor`` and by the memory estimate. Nothing
here is written to disk or sent anywhere.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import psutil

GB = 1024**3


@dataclass
class Issue:
    severity: str  # "error" | "warning" | "info"
    title: str
    detail: str
    fix: str | None = None  # a command or a short instruction


@dataclass
class GPU:
    name: str
    memory_total: int
    memory_free: int | None
    bf16: bool
    # Its index among the GPUs PyTorch sees (a spec's model.device "cuda:N"); None for Apple's.
    index: int | None = None


@dataclass
class SystemReport:
    os: str
    machine: str
    python: str
    cpu: str
    cpu_cores: int | None
    cpu_threads: int | None
    memory_total: int
    memory_available: int
    gpus: list[GPU]
    backend: str  # what "auto" uses: cuda or cpu (Apple's MPS only when chosen; see MPS_NOTE)
    torch: str
    torch_cuda: str | None
    transformer_lens: str | None
    transformers: str | None
    recommended_dtype: str
    precision_note: str
    hf_cache_free: int | None
    mps_available: bool = False
    issues: list[Issue] = field(default_factory=list)
    # The GPU a spec's model.device "cuda" (and "auto") runs on: torch's current device.
    default_gpu: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pkg(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


@functools.cache
def cpu_name() -> str:
    system = platform.system()
    try:
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if out.stdout.strip():
                return out.stdout.strip()
        elif system == "Windows":
            name = platform.processor()
            if name:
                return name
    except Exception:  # noqa: BLE001, S110 - the generic name below will do
        pass
    return platform.processor() or platform.machine() or "Unknown CPU"


def _os_label() -> str:
    system = platform.system()
    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]}"
    if system == "Windows":
        return f"Windows {platform.release()}"
    if system == "Linux":
        try:
            info = platform.freedesktop_os_release()
            return info.get("PRETTY_NAME") or info.get("NAME") or "Linux"
        except OSError:
            return "Linux"
    return system or "Unknown"


def nvidia_driver_present() -> bool:
    if Path("/proc/driver/nvidia/version").exists():
        return True
    return shutil.which("nvidia-smi") is not None


def _rosetta() -> bool:
    if platform.system() != "Darwin" or platform.machine() != "x86_64":
        return False
    try:
        out = subprocess.run(
            ["sysctl", "-n", "sysctl.proc_translated"], capture_output=True, text=True, timeout=2
        )
        return out.stdout.strip() == "1"
    except Exception:  # noqa: BLE001
        return False


def installed_as_uv_tool() -> bool:
    return f"uv{os.sep}tools{os.sep}logogram" in sys.prefix.replace("/", os.sep)


def reinstall_command(backend: str) -> str:
    if installed_as_uv_tool():
        return f"uv tool install --reinstall --torch-backend={backend} logogram"
    return f"uv pip install --reinstall torch --torch-backend={backend}"


def hf_cache_dir() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


# Why "auto" doesn't choose Apple's MPS.
MPS_NOTE = (
    "TransformerLens reports that Apple's MPS can give silently wrong results, and Logogram hasn't "
    "been checked on it yet, so experiments run on the CPU unless you choose MPS. MPS is faster; "
    "check results that matter on the CPU."
)


def system_report() -> SystemReport:
    import torch

    vm = psutil.virtual_memory()
    gpus: list[GPU] = []
    backend = "cpu"
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    default_gpu = None
    if torch.cuda.is_available():
        backend = "cuda"
        with contextlib.suppress(Exception):
            default_gpu = int(torch.cuda.current_device())
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free = None
            with contextlib.suppress(Exception):
                free, _ = torch.cuda.mem_get_info(i)
            gpus.append(
                GPU(
                    name=props.name,
                    memory_total=int(props.total_memory),
                    memory_free=int(free) if free is not None else None,
                    bf16=bool(torch.cuda.is_bf16_supported()),
                    index=i,
                )
            )
    elif mps:
        # Listed, but used only when chosen (see MPS_NOTE).
        gpus.append(
            GPU(name="Apple GPU (Metal)", memory_total=int(vm.total), memory_free=None, bf16=True)
        )

    issues: list[Issue] = []
    torch_cuda = torch.version.cuda
    if backend != "cuda" and nvidia_driver_present():
        if torch_cuda is None:
            issues.append(
                Issue(
                    "warning",
                    "Your NVIDIA GPU isn't being used",
                    "This PyTorch build is CPU-only, so experiments run on the CPU. Reinstall "
                    "PyTorch with CUDA support to use the GPU.",
                    reinstall_command("auto"),
                )
            )
        else:
            issues.append(
                Issue(
                    "warning",
                    "PyTorch can't reach your NVIDIA GPU",
                    f"PyTorch was built for CUDA {torch_cuda}, but the NVIDIA driver can't run it. "
                    "Update the NVIDIA driver, or install a PyTorch build for an older CUDA "
                    "version (for example cu126).",
                    reinstall_command("cu126"),
                )
            )
    if mps and backend == "cpu":
        issues.append(
            Issue(
                "info",
                "The Apple GPU (MPS) is used only when you choose it",
                MPS_NOTE,
                "Choose MPS under Device when you load a model.",
            )
        )
    if platform.system() == "Darwin" and not mps:
        if _rosetta():
            issues.append(
                Issue(
                    "warning",
                    "Python is running under Rosetta",
                    "This is an Intel (x86_64) Python on Apple Silicon, so the GPU (MPS) can't be "
                    "used. Install a native arm64 Python and reinstall Logogram with it.",
                    "uv python install 3.12 && uv tool install --reinstall --python 3.12 logogram",
                )
            )
        elif platform.machine() == "arm64":
            issues.append(
                Issue(
                    "warning",
                    "Apple GPU (MPS) unavailable",
                    "PyTorch can't use Metal on this Mac. MPS needs macOS 12.3 or later.",
                    "Update macOS, then reinstall PyTorch.",
                )
            )
    if vm.total < 8 * GB:
        issues.append(
            Issue(
                "warning",
                "Less than 8 GB of memory",
                "GPT-2 small fits, but larger models may not. Close other applications before "
                "running sweeps.",
            )
        )
    cache_free = None
    try:
        cache = hf_cache_dir()
        probe = cache if cache.exists() else Path.home()
        cache_free = shutil.disk_usage(probe).free
        if cache_free < 2 * GB:
            issues.append(
                Issue(
                    "warning",
                    "Low disk space for model downloads",
                    f"Only {cache_free / GB:.1f} GB is free where Hugging Face stores models. "
                    "GPT-2 small needs about 0.6 GB.",
                    "Free some space, or set HF_HOME to a disk with more room.",
                )
            )
    except Exception:  # noqa: BLE001, S110 - the disk check is only advice
        pass

    bf16 = any(g.bf16 for g in gpus)
    note = "float32 keeps results exact; use it whenever the model fits."
    if backend == "cuda" and bf16:
        note += " bfloat16 is available on this GPU for models that don't fit in float32."
    elif backend == "cpu":
        note += " 16-bit types are slow on most CPUs."

    try:
        cores = psutil.cpu_count(logical=False)
    except Exception:  # noqa: BLE001
        cores = None
    return SystemReport(
        os=_os_label(),
        machine=platform.machine(),
        python=platform.python_version(),
        cpu=cpu_name(),
        cpu_cores=cores,
        cpu_threads=psutil.cpu_count(logical=True),
        memory_total=int(vm.total),
        memory_available=int(vm.available),
        gpus=gpus,
        backend=backend,
        torch=torch.__version__,
        torch_cuda=torch_cuda,
        transformer_lens=_pkg("transformer-lens"),
        transformers=_pkg("transformers"),
        recommended_dtype="float32",
        precision_note=note,
        hf_cache_free=cache_free,
        mps_available=mps,
        issues=issues,
        default_gpu=default_gpu,
    )


# -- the environment, for reproducing runs -----------------------------------------------------

# What a run's numbers can depend on beyond its spec, as their distributions are named on PyPI:
# the same spec, on the same machine with the same versions, gives bit-identical results.
LIBRARIES = (
    "torch",
    "transformer-lens",
    "transformers",
    "tokenizers",
    "safetensors",
    "huggingface-hub",
    "numpy",
    "pyarrow",
)


def os_family() -> str:
    """linux, macos or windows (or the system's own name, in lower case)."""
    system = platform.system()
    return {"Darwin": "macos"}.get(system, system.lower() or "unknown")


def device_types() -> list[str]:
    """The devices PyTorch can run a model on here: cpu, and cuda and mps where available."""
    import torch

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@functools.cache
def _environment(hardware: bool) -> dict[str, Any]:
    import torch

    from logogram import __version__

    devices = device_types()
    machine = platform.machine().lower()
    # Named and read as run manifests record them (runner.environment_versions), so the two
    # compare key by key: transformer_lens, and PyTorch's own version, with its build (+cu124).
    versions = {name.replace("-", "_"): _pkg(name) for name in LIBRARIES}
    versions["torch"] = torch.__version__
    out: dict[str, Any] = {
        "logogram": __version__,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "os": os_family(),
        "architecture": {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine),
        "versions": versions,
        "torch_cuda": torch.version.cuda,
        "devices": devices,
        # What a spec's "auto" device runs on here (see backends.transformer_lens.resolve_device).
        "default_device": "cuda" if "cuda" in devices else "cpu",
    }
    if hardware:
        vm = psutil.virtual_memory()
        gpus = [
            {
                "name": torch.cuda.get_device_name(i),
                "memory": int(torch.cuda.get_device_properties(i).total_memory),
            }
            for i in range(torch.cuda.device_count() if "cuda" in devices else 0)
        ]
        if "mps" in devices:
            gpus.append({"name": "Apple GPU (Metal)", "memory": int(vm.total)})
        cores = None
        with contextlib.suppress(Exception):
            cores = psutil.cpu_count(logical=False)
        out["hardware"] = {
            "os": _os_label(),
            "cpu": cpu_name(),
            "cpu_cores": cores,
            "cpu_threads": psutil.cpu_count(logical=True),
            "memory": int(vm.total),
            "gpus": gpus,
        }
    return out


def environment(hardware: bool = False) -> dict[str, Any]:
    """What a run's numbers can depend on, beyond its spec: Logogram's, Python's and the
    libraries' versions, the operating system's family, the processor architecture and the
    devices PyTorch can use. Nothing that names the person or the machine: no host or user names
    and no paths, and the hardware's models only with ``hardware``."""
    return copy.deepcopy(_environment(hardware))


def requirements() -> str:
    """Pinned requirements that install the versions :func:`environment` reports, with Python's
    version and PyTorch's build as comments (an index, not a version, chooses the build)."""
    env = environment()
    lines = [
        f"# Logogram's environment: Python {env['python']} ({env['implementation']}) on "
        f"{env['os']} ({env['architecture']}).",
        f"logogram=={env['logogram']}",
    ]
    for name in LIBRARIES:
        installed = env["versions"][name.replace("-", "_")]
        if installed is None:
            lines.append(f"# {name} isn't installed.")
            continue
        public, _, build = installed.partition("+")
        if build:
            # A local version label (2.6.0+cu124) names a build that PyPI doesn't serve.
            how = f"with uv: uv pip install --torch-backend={build} -r this file"
            lines.append(
                f"# {name} {installed} is the {build} build, from PyTorch's index ({how})."
            )
        lines.append(f"{name}=={public}")
    return "\n".join(lines) + "\n"


# -- memory estimate ---------------------------------------------------------------------------

DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


@dataclass
class MemoryEstimate:
    device: str
    dtype: str
    n_params: int
    weights: int
    activations: int
    # Extra memory for a moment while the model loads, when its weights are processed.
    processing: int
    margin: int
    total: int
    available: int
    verdict: str  # fits | tight | wont_fit
    explanation: str
    # Gradient methods and path patching hold every layer's activations at once.
    all_layers: int = 0
    all_layers_verdict: str = "fits"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def available_memory(device: str) -> int:
    """Memory free on ``device``: on one GPU (cuda:N, or the default GPU for cuda), Apple's
    unified memory, or RAM."""
    import torch

    if device.split(":")[0] == "cuda" and torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info(torch.device(device))
        return int(free)
    if device == "mps":
        try:
            return int(torch.mps.recommended_max_memory()) - int(
                torch.mps.current_allocated_memory()
            )
        except Exception:  # noqa: BLE001, S110 - fall back to the RAM that is free
            pass
    return int(psutil.virtual_memory().available)


def estimate_memory(
    *,
    n_params: int,
    n_layers: int,
    n_heads: int,
    d_model: int,
    d_mlp: int,
    d_vocab: int,
    dtype: str,
    device: str,
    batch_size: int = 64,
    seq_len: int = 32,
    loaded_bytes: int = 0,
    process_weights: bool = False,
) -> MemoryEstimate:
    """Weights + the larger of the activations of one batch and, while the model loads, weight
    processing + a safety margin, against free memory.

    ``loaded_bytes`` is memory held by a model that loading this one would replace.
    """
    b = DTYPE_BYTES[dtype]
    weights = n_params * b
    activations, all_layers = activation_bytes(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_mlp=d_mlp,
        d_vocab=d_vocab,
        dtype=dtype,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    # TransformerLens processes the weights on float32 copies, on the model's device: while the
    # model loads, that takes about three more float32 copies of the weights, or four for a 16-bit
    # model (measured on GPT-2 and Pythia).
    processing = n_params * 4 * (3 if dtype == "float32" else 4) if process_weights else 0
    need = weights + max(activations, processing)
    overhead = _overhead(device)
    margin = int(0.15 * need) + overhead
    total = need + margin
    available = available_memory(device) + loaded_bytes
    verdict = _verdict(total, available)
    deep = weights + all_layers
    all_layers_verdict = _verdict(deep + int(0.15 * deep) + overhead, available)
    where = _where(device)
    explanation = (
        f"{n_params / 1e6:,.0f}M stored values × {b} bytes ({dtype}) for weights; activations for a "
        f"batch of {batch_size} prompts of {seq_len} tokens"
    )
    if processing:
        explanation += (
            "; for a moment while the model loads, float32 copies of the weights to process them "
            "(loading with weight processing off avoids this)"
        )
    explanation += f"; a margin of 15% plus runtime overhead. Compared with free {where}."
    if all_layers_verdict != "fits" and verdict == "fits":
        explanation += (
            f" Attribution patching, direct logit attribution and path patching hold every "
            f"layer's activations at once (about {format_bytes(all_layers)} for that batch): "
            "use a smaller execution.batch_size for them."
        )
    return MemoryEstimate(
        device=device,
        dtype=dtype,
        n_params=n_params,
        weights=weights,
        activations=activations,
        processing=processing,
        margin=margin,
        total=total,
        available=available,
        verdict=verdict,
        explanation=explanation,
        all_layers=all_layers,
        all_layers_verdict=all_layers_verdict,
    )


def activation_bytes(
    *,
    n_layers: int,
    n_heads: int,
    d_model: int,
    d_mlp: int,
    d_vocab: int,
    dtype: str,
    batch_size: int,
    seq_len: int,
) -> tuple[int, int]:
    """One batch's activations: for a forward pass, and with every layer's held at once.

    A forward pass holds residual-sized tensors per block, MLP hidden states, attention scores
    and patterns, final logits for the last position, plus one layer of cached source
    activations. Attribution patching and direct logit attribution keep every layer's
    activations for the backward pass, and path patching holds every head's (and MLP's) output:
    about one layer's tensors for each layer, plus the gradients they produce.
    """
    b = DTYPE_BYTES[dtype]
    tokens = batch_size * seq_len
    per_layer = tokens * (8 * d_model + 2 * d_mlp) * b + 3 * batch_size * n_heads * seq_len**2 * b
    shared = batch_size * d_vocab * 4 + 3 * tokens * d_model * b
    return per_layer * 2 + shared, per_layer * n_layers + shared


@dataclass
class RunMemory:
    """What a run needs beyond the loaded model's weights, against the memory free now."""

    device: str
    batch_size: int
    seq_len: int
    all_layers: bool
    activations: int
    total: int
    available: int
    verdict: str  # fits | tight | wont_fit
    explanation: str
    # What the run's sites add: their statistics, a layer of cached source activations, and the
    # products attribution patching keeps for every neuron. Included in ``total``.
    sites: int = 0
    site_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScopeFootprint:
    """How much a scope's sites hold, for the memory estimate: how many sites it has, the width
    of a layer's activations at one token for the kinds it caches, and how many layers of
    neurons it has (attribution patching keeps a number per neuron, token and prompt for each)."""

    sites: int
    width: int
    neuron_layers: int


def scope_footprint(
    scope: Any,
    *,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    d_model: int,
    d_head: int,
    d_mlp: int,
    seq_len: int,
) -> ScopeFootprint:
    """The footprint of a spec's scope on a model of this shape (``seq_len`` positions stand in
    for a layer x position sweep's columns)."""
    widths = {
        "head": n_heads * d_head,
        "q": n_heads * d_head,
        "k": n_kv_heads * d_head,
        "v": n_kv_heads * d_head,
        "neuron": d_mlp,
    }
    kind = getattr(scope, "kind", None)
    if kind == "neurons":
        layers = len(scope.layers)
        return ScopeFootprint(layers * d_mlp, d_mlp, layers)
    if kind == "head_inputs":
        units = n_heads if scope.input == "q" else n_kv_heads
        return ScopeFootprint(n_layers * units, widths[scope.input], 0)
    if kind == "heads":
        return ScopeFootprint(n_layers * n_heads, widths["head"], 0)
    if kind == "layer_position":
        return ScopeFootprint(n_layers * seq_len, d_model, 0)
    if kind == "layer_components":
        return ScopeFootprint(n_layers * len(scope.components), d_model * len(scope.components), 0)
    if kind == "sites":
        by_layer: dict[int, set[str]] = {}
        for site in scope.sites:
            by_layer.setdefault(site.layer, set()).add(site.kind)
        width = max(
            (sum(widths.get(k, d_model) for k in kinds) for kinds in by_layer.values()), default=0
        )
        neurons = sum(1 for kinds in by_layer.values() if "neuron" in kinds)
        return ScopeFootprint(len(scope.sites), width, neurons)
    if kind == "site_sets":
        return ScopeFootprint(len(scope.sets), 0, 0)
    if kind == "features":
        return ScopeFootprint(int(scope.top), 0, 0)
    return ScopeFootprint(0, 0, 0)


def site_bytes(
    footprint: ScopeFootprint,
    *,
    method: str,
    prompts: int,
    bootstrap: int,
    batch_size: int,
    seq_len: int,
    dtype: str,
) -> int:
    """What a run's sites hold beyond a batch's forward pass: each site's per-prompt values and
    bootstrap resamples (with the ranks the simultaneous bands sort), the layer of source
    activations patching and ablation cache for every prompt, and the per-neuron products
    attribution patching keeps for each layer of neurons (in float64, beside the activations and
    gradients it multiplies)."""
    b = DTYPE_BYTES[dtype]
    stats = 8 * footprint.sites * (5 * prompts + 6 * bootstrap)
    cache = (
        prompts * seq_len * footprint.width * b
        if method in ("activation_patching", "ablation")
        else 0
    )
    products = (
        footprint.neuron_layers * batch_size * seq_len * footprint.width * (8 + 3 * b)
        if method == "attribution_patching"
        else 0
    )
    return stats + cache + products


# Methods that hold every layer's activations at once: the gradient methods keep them for the
# backward pass, and path patching keeps every head's and MLP's output.
ALL_LAYER_METHODS = frozenset({"attribution_patching", "direct_logit_attribution", "path_patching"})


def estimate_run_memory(
    *,
    n_layers: int,
    n_heads: int,
    d_model: int,
    d_mlp: int,
    d_vocab: int,
    dtype: str,
    device: str,
    batch_size: int,
    seq_len: int,
    all_layers: bool,
    footprint: ScopeFootprint | None = None,
    method: str = "",
    prompts: int = 0,
    bootstrap: int = 0,
) -> RunMemory:
    """A batch's activations for a run on the loaded model (every layer's at once for the methods
    that hold them), plus the same margin as a model's estimate, against the memory free now
    (the weights are already in it). With the scope's ``footprint``, the number of ``prompts``
    and of ``bootstrap`` resamples, what its sites hold too (see :func:`site_bytes`): thousands
    of neurons hold far more than a batch."""
    one, every = activation_bytes(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_mlp=d_mlp,
        d_vocab=d_vocab,
        dtype=dtype,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    activations = every if all_layers else one
    sites = (
        site_bytes(
            footprint,
            method=method,
            prompts=prompts,
            bootstrap=bootstrap,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=dtype,
        )
        if footprint is not None
        else 0
    )
    held = activations + sites
    total = held + int(0.15 * held) + _overhead(device)
    available = available_memory(device)
    every_layer = ", every layer's at once," if all_layers else ""
    explanation = (
        f"Activations for a batch of {batch_size} prompts of up to {seq_len} tokens{every_layer} "
        "plus a margin, against the "
        f"{_where(device)} free with the model loaded."
    )
    if footprint is not None and sites >= 64 * 1024**2:
        explanation += (
            f" The {footprint.sites:,} sites hold about {format_bytes(sites)} more: each one's "
            f"values for {prompts:,} prompts and {bootstrap:,} bootstrap resamples"
            + (
                ", a layer of source activations for every prompt"
                if method in ("activation_patching", "ablation")
                else ""
            )
            + (
                ", and attribution patching's products for every neuron"
                if method == "attribution_patching" and footprint.neuron_layers
                else ""
            )
            + "."
        )
    return RunMemory(
        device=device,
        batch_size=batch_size,
        seq_len=seq_len,
        all_layers=all_layers,
        activations=activations,
        total=total,
        available=available,
        verdict=_verdict(total, available),
        explanation=explanation,
        sites=footprint.sites if footprint is not None else 0,
        site_bytes=sites,
    )


def _overhead(device: str) -> int:
    return 512 * 1024**2 if device.split(":")[0] == "cuda" else 256 * 1024**2


def _verdict(total: int, available: int) -> str:
    if total <= 0.8 * available:
        return "fits"
    return "tight" if total <= available else "wont_fit"


def _where(device: str) -> str:
    if device.startswith("cuda:"):
        return f"memory on GPU {device.split(':')[1]}"
    return {"cuda": "GPU memory", "mps": "unified memory", "cpu": "RAM"}.get(device, "memory")


def format_bytes(n: int | float | None) -> str:
    if n is None:
        return "—"
    if n >= GB:
        return f"{n / GB:.1f} GB"
    return f"{n / 1024**2:.0f} MB"
