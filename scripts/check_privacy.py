#!/usr/bin/env python3
"""Fail if tracked files contain anything that identifies a person or the machine they used.

Checks every tracked file (``git ls-files``; outside a git checkout, every file that isn't in a
build or cache folder) for:

* home-directory paths: ``/home/<name>`` (and ``/var/home/<name>`` and ``/usr/home/<name>``,
  where Fedora's atomic editions and FreeBSD keep homes), ``/root/<path>``, ``/Users/<name>``, and
  Windows' ``C:\\Users\\<name>`` with either slash, also as WSL (``/mnt/c/Users/<name>``) and Git
  Bash (``/c/Users/<name>``) show it,
* the current username, hostname and CPU model, read at runtime,
* email addresses,
* Hugging Face access tokens.

Binary files are searched too (a stray ``.coverage`` database or parquet table holds absolute
paths): their bytes read as Latin-1, and the UTF-16 strings in them. Text saved as UTF-16 (as
Windows PowerShell writes it) is read as text.

Usernames and hostnames shorter than four characters aren't searched as words: "max", "li" or
"dev" would match ordinary words and code. A short username is still found in a home path (by
the path patterns above) and in ``name@host``, as a shell prompt or an ssh target shows it.

Usage: ``python scripts/check_privacy.py [ROOT]``. Exits with status 1 and lists each finding.
Standard library only, so it runs before any dependency is installed.
"""

from __future__ import annotations

import codecs
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

# Home folders that belong to nobody: CI runners', and shared or template ones.
_NOBODY = r"(?!(?:runner|runneradmin|Shared|Public|Default)\b)"
HOME_PATH = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    rf"(?:/var|/usr)?/home/{_NOBODY}[\w.-]+"
    r"|/root/[\w.-]+"
    rf"|(?:/mnt/[A-Za-z]/[Uu]sers|/[A-Za-z]/Users|/Users)/{_NOBODY}[\w.-]+"
    # Backslashes single or escaped (as in JSON), or forward slashes.
    rf"|[A-Za-z]:(?:\\\\?|/)[Uu]sers(?:\\\\?|/){_NOBODY}[\w. -]+"
    r")"
)
EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"
)
HF_TOKEN = re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")
# Placeholder addresses that are fine in docs and tests.
ALLOWED_EMAIL_DOMAINS = ("example.com", "example.org", "example.net")
# Runs of at least four printable ASCII characters stored as UTF-16, little- or big-endian.
UTF16_RUN = re.compile(rb"((?:[\x20-\x7e]\x00){4,})|((?:\x00[\x20-\x7e]){4,})")


@dataclass
class Finding:
    path: str
    line: int  # 0 for a file name or binary data
    kind: str
    text: str
    offset: int | None = None  # the byte, in binary data

    def __str__(self) -> str:
        where = f"byte {self.offset}" if self.offset is not None else str(self.line)
        return f"{self.path}:{where}: {self.kind}: {self.text}"


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
    """Names of this person and machine: as whole words, or a short username as ``name@host``."""
    patterns: list[tuple[str, re.Pattern[str]]] = []
    candidates = {
        "username": {getpass.getuser(), Path.home().name},
        "hostname": {socket.gethostname().split(".")[0], platform.node().split(".")[0]},
    }
    for kind, names in candidates.items():
        for name in sorted(n for n in names if n):
            if name.lower() in GENERIC_NAMES:
                continue
            if len(name) >= 4:
                word = rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])"
                patterns.append((kind, re.compile(word, re.I)))
            elif kind == "username" and len(name) >= 2:
                # A host with no dot after it, so an email address doesn't count.
                prompt = rf"(?<![\w.%+-]){re.escape(name)}@[A-Za-z0-9-]+(?![\w.@-])"
                patterns.append((kind, re.compile(prompt)))
    cpu = cpu_model()
    if cpu and len(cpu) >= 8:
        patterns.append(("CPU model", re.compile(re.escape(cpu))))
    return patterns


def matches(
    text: str, identities: list[tuple[str, re.Pattern[str]]]
) -> Iterator[tuple[int, str, str]]:
    """(index, kind, what to show) for everything in ``text`` that identifies someone."""
    for m in HOME_PATH.finditer(text):
        yield m.start(), "home directory path", m.group(0)
    for m in EMAIL.finditer(text):
        domain = m.group(0).rsplit("@", 1)[1].lower()
        if not domain.endswith(ALLOWED_EMAIL_DOMAINS):
            yield m.start(), "email address", m.group(0)
    for m in HF_TOKEN.finditer(text):
        yield m.start(), "Hugging Face token", m.group(0)[:6] + "…"
    for kind, pattern in identities:
        for m in pattern.finditer(text):
            yield m.start(), kind, "(this machine's " + kind + ")"


def scan_text(
    rel: str, text: str, identities: list[tuple[str, re.Pattern[str]]]
) -> Iterator[Finding]:
    for i, line in enumerate(text.splitlines(), start=1):
        for kind, shown in dict.fromkeys((k, s) for _, k, s in matches(line, identities)):
            yield Finding(rel, i, kind, shown)


def scan_binary(
    rel: str, data: bytes, identities: list[tuple[str, re.Pattern[str]]]
) -> Iterator[Finding]:
    """Each byte as one character, so ASCII and UTF-8 text show through; then each UTF-16
    string (Windows stores text that way)."""
    for start, kind, shown in matches(data.decode("latin-1"), identities):
        yield Finding(rel, 0, kind, shown, offset=start)
    for run in UTF16_RUN.finditer(data):
        text = run.group(0).decode("utf-16-le" if run.group(1) else "utf-16-be")
        for start, kind, shown in matches(text, identities):
            yield Finding(rel, 0, kind, shown, offset=run.start() + 2 * start)


def check(root: Path) -> list[Finding]:
    identities = identity_patterns()
    findings: list[Finding] = []
    for path in tracked_files(root):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if identities and any(p.search(rel) for _, p in identities):
            findings.append(Finding(rel, 0, "file name", "contains this machine's name"))
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            findings.extend(scan_text(rel, data.decode("utf-16", "replace"), identities))
        elif b"\0" in data[:8192]:
            findings.extend(scan_binary(rel, data, identities))
        else:
            findings.extend(scan_text(rel, data.decode("utf-8", "replace"), identities))
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
