"""Hardware and environment report, with concrete fixes for common problems.

Used by the first-run system check, by ``logogram doctor`` and by the memory estimate. Nothing
here is written to disk or sent anywhere.
"""

from __future__ import annotations

import contextlib
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
    backend: str  # cuda | mps | cpu
    torch: str
    torch_cuda: str | None
    transformer_lens: str | None
    transformers: str | None
    recommended_dtype: str
    precision_note: str
    hf_cache_free: int | None
    issues: list[Issue] = field(default_factory=list)

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
    except Exception:  # noqa: BLE001
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


def system_report() -> SystemReport:
    import torch

    vm = psutil.virtual_memory()
    gpus: list[GPU] = []
    backend = "cpu"
    if torch.cuda.is_available():
        backend = "cuda"
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
                )
            )
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        backend = "mps"
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
    if platform.system() == "Darwin" and backend == "cpu":
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
    except Exception:  # noqa: BLE001
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
        issues=issues,
    )


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def available_memory(device: str) -> int:
    import torch

    if device == "cuda" and torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        return int(free)
    if device == "mps":
        try:
            return int(torch.mps.recommended_max_memory()) - int(
                torch.mps.current_allocated_memory()
            )
        except Exception:  # noqa: BLE001
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
    tokens = batch_size * seq_len
    # One forward pass: residual-sized tensors per block, MLP hidden states, attention scores and
    # patterns, final logits for the last position, plus one layer of cached source activations.
    per_layer = tokens * (8 * d_model + 2 * d_mlp) * b + 3 * batch_size * n_heads * seq_len**2 * b
    activations = per_layer * 2 + batch_size * d_vocab * 4 + 3 * tokens * d_model * b
    # TransformerLens processes the weights on float32 copies, on the model's device: while the
    # model loads, that takes about three more float32 copies of the weights, or four for a 16-bit
    # model (measured on GPT-2 and Pythia).
    processing = n_params * 4 * (3 if dtype == "float32" else 4) if process_weights else 0
    need = weights + max(activations, processing)
    margin = int(0.15 * need) + (512 * 1024**2 if device == "cuda" else 256 * 1024**2)
    total = need + margin
    available = available_memory(device) + loaded_bytes
    if total <= 0.8 * available:
        verdict = "fits"
    elif total <= available:
        verdict = "tight"
    else:
        verdict = "wont_fit"
    where = {"cuda": "GPU memory", "mps": "unified memory", "cpu": "RAM"}.get(device, "memory")
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
    )


def format_bytes(n: int | float | None) -> str:
    if n is None:
        return "—"
    if n >= GB:
        return f"{n / GB:.1f} GB"
    return f"{n / 1024**2:.0f} MB"
