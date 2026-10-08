"""The real-weight validation script runs every method, and its identities hold on any model.

The script is meant for real models (the release check, and Apple's MPS); here it runs on the tiny
random model, where the identities must hold but agreement expected of a trained model needn't.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "validate_real_weights", ROOT / "scripts" / "validate_real_weights.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_real_weights"] = module
    spec.loader.exec_module(module)
    return module


def test_every_method_runs_and_the_identities_hold(tiny_backend, project):
    lines: list[str] = []
    checks = _load().validate(
        tiny_backend, project, "datasets/ioi.jsonl", prepend_bos=True, log=lines.append
    )
    names = {c.name for c in checks}
    for expected in (
        "Patching the token that differs restores everything",
        "Patching the final residual stream restores everything",
        "The same spec twice gives bit-identical results",
        "Direct effects add up to the logit difference",
        "From the last layer, a path to the logits is the whole effect",
        "The logit lens at the last layer is the model's own prediction",
        "Steering runs",
        "Path patching into heads runs",
    ):
        assert expected in names, "\n".join(lines)
    failed = [c for c in checks if c.exact and c.passed is False]
    assert not failed, "\n".join(lines)
