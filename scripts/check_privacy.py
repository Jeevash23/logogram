#!/usr/bin/env python3
"""Fail if tracked files contain anything that identifies a person or the machine they used.

Checks every tracked file (``git ls-files``; outside a git checkout, every file that isn't in a
build or cache folder) for:

* home-directory paths (``/home/<name>``, ``/Users/<name>``, ``C:\\Users\\<name>``),
* the current username, hostname and CPU model, read at runtime,
* email addresses,
* Hugging Face access tokens.

Usage: ``python scripts/check_privacy.py [ROOT]``. Exits with status 1 and lists each finding.
Standard library only, so it runs before any dependency is installed.
"""

from __future__ import annotations

import getpass
import os
import platform
import re
import socket
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "dist",
    "build",
    ".claude",
}

# Account and machine names that identify nobody (CI runners, containers, generic setups).
GENERIC_NAMES = {
    "root",
    "user",
    "users",
    "admin",
    "runner",
    "ubuntu",
    "debian",
    "vagrant",
    "docker",
    "ci",
    "build",
    "builder",
    "jenkins",
    "travis",
    "circleci",
    "appveyor",
    "vscode",
    "codespace",
    "codespaces",
    "gitpod",
    "localhost",
    "home",
    "runneradmin",
}

HOME_PATH = re.compile(
    r"(?<![A-Za-z0-9_])(?:/home/(?!runner\b)[A-Za-z0-9._-]+|/Users/(?!Shared\b)[A-Za-z0-9._-]+"
    r"|[A-Za-z]:\\\\?Users\\\\?(?!Public\b)[A-Za-z0-9._ -]+)"
)
EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"
)
HF_TOKEN = re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")
# Placeholder addresses that are fine in docs and tests.
ALLOWED_EMAIL_DOMAINS = ("example.com", "example.org", "example.net")


@dataclass
class Finding:
    path: str
    line: int
    kind: str
    text: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.kind}: {self.text}"


def tracked_files(root: Path) -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            capture_output=True,
            check=True,
        )
        names = [n for n in out.stdout.decode("utf-8", "replace").split("\0") if n]
        if names:
            return [root / n for n in names]
    except (OSError, subprocess.CalledProcessError):
        pass
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        files.extend(Path(dirpath) / f for f in filenames)
    return files


def cpu_model() -> str | None:
    try:
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            )
            return out.stdout.strip() or None
    except OSError:
        return None
    return platform.processor() or None


def identity_patterns() -> list[tuple[str, re.Pattern[str]]]:
    """Names of this person and machine, as whole words."""
    patterns: list[tuple[str, re.Pattern[str]]] = []
    candidates = {
        "username": {getpass.getuser(), Path.home().name},
        "hostname": {socket.gethostname().split(".")[0], platform.node().split(".")[0]},
    }
    for kind, names in candidates.items():
        for name in sorted(n for n in names if n):
            if len(name) < 4 or name.lower() in GENERIC_NAMES:
                continue
            patterns.append(
                (kind, re.compile(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", re.I))
            )
    cpu = cpu_model()
    if cpu and len(cpu) >= 8:
        patterns.append(("CPU model", re.compile(re.escape(cpu))))
    return patterns


def read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:8192]:
        return None  # binary (fonts, images, parquet)
    return data.decode("utf-8", "replace")


def scan_text(
    rel: str, text: str, identities: list[tuple[str, re.Pattern[str]]]
) -> Iterator[Finding]:
    for i, line in enumerate(text.splitlines(), start=1):
        for m in HOME_PATH.finditer(line):
            yield Finding(rel, i, "home directory path", m.group(0))
        for m in EMAIL.finditer(line):
            domain = m.group(0).rsplit("@", 1)[1].lower()
            if not domain.endswith(ALLOWED_EMAIL_DOMAINS):
                yield Finding(rel, i, "email address", m.group(0))
        for m in HF_TOKEN.finditer(line):
            yield Finding(rel, i, "Hugging Face token", m.group(0)[:6] + "…")
        for kind, pattern in identities:
            if pattern.search(line):
                yield Finding(rel, i, kind, "(this machine's " + kind + ")")


def check(root: Path) -> list[Finding]:
    identities = identity_patterns()
    findings: list[Finding] = []
    for path in tracked_files(root):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if identities and any(p.search(rel) for _, p in identities):
            findings.append(Finding(rel, 0, "file name", "contains this machine's name"))
        text = read_text(path)
        if text is not None:
            findings.extend(scan_text(rel, text, identities))
    return findings


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else Path(__file__).resolve().parents[1]).resolve()
    findings = check(root)
    if findings:
        print(f"Privacy check failed: {len(findings)} finding(s).\n")
        for f in findings:
            print(f"  {f}")
        print(
            "\nRemove these before committing. Use project-relative paths, placeholder names "
            "and addresses at example.com."
        )
        return 1
    print("Privacy check passed: no personal paths, names, addresses or tokens found.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
