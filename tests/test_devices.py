"""What depends on the compute device.

Apple's MPS has no float64: converting a tensor there to float64 fails, so a single `.double()` on
a model's activations would stop every run on a Mac. Every conversion goes through
`backends.base.float64`, which moves the tensor to the CPU first on such a device. CI has no Mac,
so this checks the rule in the source, and the helper where it can run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import logogram
from logogram.backends import base
from logogram.backends.base import float64

MARK = "# float64 on the CPU"


def test_every_conversion_to_float64_goes_through_one_helper():
    root = Path(logogram.__file__).parent
    helper = root / "backends" / "base.py"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path == helper:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for n, line in enumerate(lines, 1):
            marked = MARK in line or (n > 1 and MARK in lines[n - 2])
            if ".double()" in line or ("torch.float64" in line and not marked):
                offenders.append(f"{path.relative_to(root)}:{n}: {line.strip()}")
    assert offenders == []


def test_the_helper_converts_where_the_tensor_is():
    t = torch.tensor([1.5, -2.25, 3e-8], dtype=torch.float32)
    out = float64(t)
    assert out.dtype == torch.float64 and out.device == t.device
    assert torch.equal(out, t.double())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a second device")
def test_a_device_without_float64_converts_on_the_cpu(monkeypatch):
    monkeypatch.setattr(base, "NO_FLOAT64", frozenset({"cuda"}))
    t = torch.tensor([1.5, -2.25], device="cuda")
    out = float64(t)
    assert out.device.type == "cpu" and torch.equal(out, t.cpu().double())
