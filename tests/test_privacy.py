"""The privacy check passes on this repository, and catches what it should."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_privacy", ROOT / "scripts" / "check_privacy.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_privacy"] = module
    spec.loader.exec_module(module)
    return module


def test_repository_is_clean():
    findings = _load().check(ROOT)
    assert not findings, "\n".join(str(f) for f in findings)


def test_detects_paths_addresses_tokens_and_names(tmp_path):
    check = _load()
    home = "/" + "home" + "/someone"
    token = "hf_" + "x" * 34
    (tmp_path / "leak.txt").write_text(
        f"see {home}/data\nwrite to person@" + "university.edu\n"
        f"token {token}\nfine: dev@example.com and ~/cache\n",
        encoding="utf-8",
    )
    kinds = sorted(f.kind for f in check.check(tmp_path))
    assert kinds == ["Hugging Face token", "email address", "home directory path"]


def test_generic_account_names_are_not_flagged(tmp_path, monkeypatch):
    check = _load()
    monkeypatch.setattr(check.getpass, "getuser", lambda: "runner")
    (tmp_path / "code.py").write_text("from logogram.runner import run_spec\n", encoding="utf-8")
    assert not [f for f in check.check(tmp_path) if f.kind == "username"]
