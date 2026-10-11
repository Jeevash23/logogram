"""Asking for help: a short problem report that identifies nobody, and where to send it.

The report names Logogram's version and the versions it runs on, the operating system's family,
the kind of device, what the system check found, the loaded model and the last error the app
showed. User names, host names and paths never appear in it: they are scrubbed from error text.
The processor, GPUs, memory and the operating system's version, which tell machines apart, are
added only when the person asks.

Nothing here is sent anywhere. The app shows the report for the person to copy, and
``logogram doctor --report`` prints it; the person pastes it into an issue themselves.
"""

from __future__ import annotations

import contextlib
import getpass
import platform
import re
import socket
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from logogram import __version__

# The repository, as pyproject.toml's [project.urls] names it.
REPOSITORY = "https://github.com/Jeevash23/logogram"
# The issue form for problems. The report is never put in the address: the person pastes it.
NEW_ISSUE_URL = f"{REPOSITORY}/issues/new?template=bug_report.yml"
# How much of an error message a report keeps.
ERROR_LIMIT = 2000

# Account and machine names that identify nobody, and words too ordinary to scrub from a message
# ("server" is a word in error messages, not only a machine's name).
_GENERIC_NAMES = {
    "admin",
    "build",
    "ci",
    "computer",
    "desktop",
    "docker",
    "gpu",
    "home",
    "host",
    "laptop",
    "linux",
    "localhost",
    "logogram",
    "machine",
    "root",
    "runner",
    "server",
    "ubuntu",
    "user",
    "users",
    "workstation",
}

# Where a path may start: the start of the text, a space, or punctuation that opens something.
# Never after a letter, digit, ':' or '/', so "and/or", "3/4" and web addresses stay as they are.
_START = r"(?<![^\s'\"`(=\[{,])"
# The characters of a path outside quotes: anything up to a space, a quote or a bracket.
_BODY = r"[^\s'\"`<>|;,()\[\]{}]"
_QUOTED_PATH = re.compile(r"""(['"`])((?:[A-Za-z]:[\\/]|\\\\|/(?=[^\s/'"`])|~[\\/])[^'"`\n]*?)\1""")
_PATHS = re.compile(
    _START
    + r"(?:[A-Za-z]:[\\/]"  # C:\ or C:/
    + r"|\\\\[^\s\\/]+[\\/]"  # \\server\share
    + r"|~[\\/]"  # ~/ (the home folder, after it was written as ~)
    + r"|/(?=[^\s/'\"`<>|;,()\[\]{}]))"  # /absolute (not // or a lone slash)
    + _BODY
    + r"*"
)
_URL_CREDENTIALS = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^\s/@:]+(?::[^\s/@]*)?@")
_TOKEN_QUERY = re.compile(
    r"(?i)\b(token|access_token|api_key|apikey|password|passwd|secret)=[^\s&'\";,)]+"
)
_HF_TOKEN = re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{8,}=*")
_EMAIL = re.compile(r"(?<![\w.%+-])[\w.%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")


@dataclass
class ReportRow:
    label: str
    value: str


@dataclass
class ProblemReport:
    rows: list[ReportRow]
    # Whether the processor, GPUs, memory and the operating system's version are in it.
    hardware: bool
    # The rows as plain text, one "Label: value" a line, ready to paste into an issue.
    text: str
    new_issue_url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _names() -> list[tuple[str, str]]:
    """This machine's user and host names, each with its placeholder, longest first."""
    found: set[tuple[str, str]] = set()
    users: list[str] = []
    with contextlib.suppress(Exception):  # no account name to scrub
        users.append(getpass.getuser())
    with contextlib.suppress(RuntimeError, OSError):
        users.append(Path.home().name)
    hosts = [socket.gethostname(), platform.node()]
    for kind, names in (("<user>", users), ("<host>", hosts)):
        for name in names:
            for candidate in (name, name.split(".")[0]):
                if len(candidate) >= 3 and candidate.lower() not in _GENERIC_NAMES:
                    found.add((candidate, kind))
    return sorted(found, key=lambda pair: -len(pair[0]))


def _places(roots: Iterable[Path | str | None]) -> list[tuple[str, str]]:
    """This machine's folders that a message may name, each with its placeholder, longest
    first: the open project, the Hugging Face cache and the home folder."""
    places: list[tuple[str, str]] = []

    def add(path: Path | str, label: str) -> None:
        for form in {str(path), _resolved(path)}:
            if form not in ("", "/", "."):
                places.append((form, label))
                places.append((form.replace("\\", "/"), label))

    for root in roots:
        if root:
            add(root, "<project>")
    with contextlib.suppress(Exception):  # no cache to name
        from huggingface_hub import constants

        add(constants.HF_HUB_CACHE, "<huggingface cache>")
    with contextlib.suppress(RuntimeError, OSError):
        add(Path.home(), "~")
    return sorted(set(places), key=lambda pair: -len(pair[0]))


def _resolved(path: Path | str) -> str:
    try:
        return str(Path(path).resolve())
    except (OSError, RuntimeError):
        return str(path)


def scrub(
    text: str,
    roots: Iterable[Path | str | None] = (),
    secrets: Iterable[str] = (),
    *,
    names: bool = True,
) -> str:
    """``text`` without what identifies the person or the machine: secrets (``secrets``, Hugging
    Face tokens, credentials in addresses, e-mail addresses), paths (the open project's ``roots``
    become <project>, the Hugging Face cache <huggingface cache>, any other path <path>) and,
    with ``names``, this machine's user and host names."""
    for secret in secrets:
        if len(secret) >= 8:  # a real one is long; a letter or two would be scrubbed everywhere
            text = text.replace(secret, "<secret>")
    text = _HF_TOKEN.sub("<token>", text)
    text = _BEARER.sub("Bearer <token>", text)
    text = _URL_CREDENTIALS.sub(r"\1<credentials>@", text)
    text = _TOKEN_QUERY.sub(lambda m: f"{m[1]}=<secret>", text)
    text = _EMAIL.sub("<email>", text)
    for place, label in _places(roots):
        # Only the whole folder: /data/project must not turn /data/project-2 into <project>-2.
        pattern = re.escape(place) + r"(?=[\\/\s'\"`),;:]|$)"
        text = re.sub(pattern, lambda _m, label=label: label, text)
    text = _QUOTED_PATH.sub(lambda m: f"{m[1]}<path>{m[1]}", text)
    text = _PATHS.sub(_path_placeholder, text)
    if names:
        for name, label in _names():
            pattern = rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])"
            text = re.sub(pattern, lambda _m, label=label: label, text, flags=re.IGNORECASE)
    return text


def _path_placeholder(match: re.Match[str]) -> str:
    # A path at the end of a sentence keeps the sentence's full stop.
    path = match[0]
    tail = len(path) - len(path.rstrip(".:"))
    return "<path>" + path[len(path) - tail :] if tail else "<path>"


def _os_family() -> str:
    system = platform.system()
    family = {"Darwin": "macOS", "Windows": "Windows", "Linux": "Linux"}.get(
        system, system or "unknown"
    )
    if system == "Linux" and "microsoft" in platform.release().lower():
        family += " under WSL"
    return f"{family} ({platform.machine() or 'unknown processor type'})"


def _device(report: Any) -> str:
    if report.backend == "cuda":
        n = len(report.gpus)
        return f"CUDA, {n} GPU{'s' if n != 1 else ''}"
    if report.mps_available:
        return "CPU, with Apple's GPU (MPS) available when chosen"
    return "CPU"


def _model(status: dict[str, Any] | None, roots: list[Any], secrets: list[str]) -> str:
    """The loaded model as a report names it: the id, revision, dtype and device."""
    status = status or {}
    info = status.get("info") or {}
    model_id = info.get("id") or status.get("id")
    if not model_id:
        return "none loaded"
    # A model id is the person's own choice, so only paths and secrets are scrubbed from it.
    model_id = scrub(str(model_id), roots, secrets, names=False)
    state = status.get("state")
    if state == "error":
        return f"{model_id} (failed to load)"
    if state == "loading" or not info:
        return f"{model_id} (loading)"
    processing = "weights processed" if info.get("process_weights") else "weights not processed"
    revision = info.get("revision") or "unknown"
    return (
        f"{model_id}, revision {revision}, {info.get('dtype')} on {info.get('device')}, "
        f"{processing}"
    )


def build_report(
    *,
    model: dict[str, Any] | None = None,
    last_error: str | None = None,
    include_hardware: bool = False,
    roots: Iterable[Path | str | None] = (),
    secrets: Iterable[str] = (),
    system: Any = None,
) -> ProblemReport:
    """The problem report. ``model`` is the server's model status (``AppState.model_payload``;
    None, outside the app, leaves the model out), ``last_error`` the last error the app showed,
    ``roots`` the open project's folder and ``secrets`` text that must never appear, such as the
    session token. ``system`` is a :class:`logogram.system.SystemReport`, measured here if not
    given."""
    from logogram.system import format_bytes, system_report

    report = system if system is not None else system_report()
    roots = [r for r in roots if r]
    secrets = [s for s in secrets if s]
    torch = report.torch + (f" (CUDA {report.torch_cuda})" if report.torch_cuda else "")
    rows = [
        ReportRow("Logogram", __version__),
        ReportRow("Python", report.python),
        ReportRow("PyTorch", torch),
        ReportRow("TransformerLens", report.transformer_lens or "not installed"),
        ReportRow("Transformers", report.transformers or "not installed"),
        ReportRow("Operating system", _os_family()),
        ReportRow("Device", _device(report)),
        ReportRow(
            "System check",
            "; ".join(i.title for i in report.issues) if report.issues else "no problems found",
        ),
    ]
    if include_hardware:
        cores = f"{report.cpu_cores or '?'} cores, {report.cpu_threads or '?'} threads"
        rows += [
            ReportRow("Operating system version", report.os),
            ReportRow("Processor", f"{report.cpu} ({cores})"),
            ReportRow("Memory", format_bytes(report.memory_total)),
        ]
        for i, gpu in enumerate(report.gpus, start=1):
            label = "GPU" if len(report.gpus) == 1 else f"GPU {i}"
            rows.append(ReportRow(label, f"{gpu.name} ({format_bytes(gpu.memory_total)})"))
    if model is not None:  # the server's; `logogram doctor --report` has none to name
        rows.append(ReportRow("Model", _model(model, roots, secrets)))
    if last_error and last_error.strip():
        error = " ".join(scrub(last_error, roots, secrets).split())
        if len(error) > ERROR_LIMIT:
            error = error[: ERROR_LIMIT - 1].rstrip() + "…"
        rows.append(ReportRow("Last error", error))
    text = "\n".join(f"{row.label}: {row.value}" for row in rows) + "\n"
    return ProblemReport(rows, include_hardware, text, NEW_ISSUE_URL)
