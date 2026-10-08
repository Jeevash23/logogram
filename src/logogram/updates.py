"""Telling people a new version is out, without giving anything away.

Three layers, from no network to a little:

* Every copy knows when its version was released (``logogram.__released__``). After a few
  months the app says the version is getting old. No network.
* *Check now* asks PyPI for the newest version, only when the user asks.
* If the user allows it (the system check asks once), Logogram asks PyPI at most once a day.

A check is one HTTPS GET of the package's public JSON on pypi.org. It carries Logogram's version
in its User-Agent and nothing else: no identifiers, paths, machine details or anything about the
user's work. The answer is cached in the user's config folder.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from importlib import metadata
from pathlib import Path
from typing import Any

from logogram import __released__, __version__
from logogram.fileio import write_text_atomic

PACKAGE = "logogram"
PYPI_URL = f"https://pypi.org/pypi/{PACKAGE}/json"
CHECK_EVERY = timedelta(days=1)
OLD_AFTER = timedelta(days=120)
TIMEOUT_S = 6.0

Fetch = Callable[[], dict[str, Any]]

_VERSION = re.compile(
    r"(?P<release>\d+(?:\.\d+)*)(?:(?P<pre>a|b|rc)(?P<pre_n>\d+))?"
    r"(?:\.post(?P<post>\d+))?(?:\.dev(?P<dev>\d+))?"
)


def version_key(text: str) -> tuple[Any, ...] | None:
    """An orderable key for a PEP 440 version (the subset Logogram releases use), or None."""
    match = _VERSION.fullmatch(text.strip().lower().removeprefix("v"))
    if match is None:
        return None
    numbers = [int(x) for x in match["release"].split(".")]
    while len(numbers) > 1 and numbers[-1] == 0:
        numbers.pop()  # 1.2 and 1.2.0 are the same release
    if match["pre"]:
        stage = (1, ("a", "b", "rc").index(match["pre"]), int(match["pre_n"]))
    elif match["dev"] is not None and match["post"] is None:
        stage = (0, 0, 0)  # 1.2.dev3 comes before 1.2a1
    else:
        stage = (2, 0, 0)
    post = int(match["post"]) if match["post"] is not None else -1
    dev = int(match["dev"]) if match["dev"] is not None else 1 << 62  # a .devN precedes its release
    return (tuple(numbers), stage, post, dev)


def is_newer(candidate: str | None, current: str) -> bool:
    if not candidate:
        return False
    a, b = version_key(candidate), version_key(current)
    return a is not None and b is not None and a > b


def update_command() -> str:
    """How this copy is updated, judged locally from how it was installed."""
    try:
        direct = metadata.distribution(PACKAGE).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        direct = None
    if direct:
        try:
            info = json.loads(direct)
        except ValueError:
            info = {}
        if (info.get("dir_info") or {}).get("editable"):
            return "git pull, then uv sync"  # a development checkout runs the folder's code
        if str(info.get("url", "")).startswith("file:"):
            return "git pull, then uv tool install --reinstall ."
    parts = {p.lower() for p in Path(sys.prefix).parts}
    if "uv" in parts and "tools" in parts:
        return f"uv tool upgrade {PACKAGE}"
    return f"pip install --upgrade {PACKAGE}"


def fetch_pypi(timeout: float = TIMEOUT_S) -> dict[str, Any]:
    """The newest release on PyPI and where its notes are. Raises on network or HTTP errors."""
    request = urllib.request.Request(
        PYPI_URL,
        headers={"User-Agent": f"logogram/{__version__}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https URL
        data = json.load(response)
    info = data.get("info") or {}
    urls = info.get("project_urls") or {}
    notes = next(
        (urls[k] for k in ("Changelog", "Release notes", "Releases", "Changes") if k in urls),
        None,
    )
    latest = info.get("version")
    return {
        "latest": latest if isinstance(latest, str) and version_key(latest) else None,
        "notes_url": notes if isinstance(notes, str) and notes.startswith("https://") else None,
    }


def read_cache(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def check(path: Path, fetch: Fetch = fetch_pypi, now: datetime | None = None) -> dict[str, Any]:
    """Ask PyPI now and cache the answer. Errors are recorded, never raised."""
    stamp = (now or datetime.now(UTC)).replace(microsecond=0).isoformat()
    try:
        result = {**fetch(), "checked_at": stamp, "error": None}
    except urllib.error.HTTPError as exc:
        message = "Logogram isn't on PyPI yet." if exc.code == 404 else f"PyPI answered {exc.code}."
        result = {**read_cache(path), "checked_at": stamp, "error": message}
    except (OSError, ValueError) as exc:  # offline, DNS, timeout, bad JSON
        reason = getattr(exc, "reason", None) or exc
        result = {
            **read_cache(path),
            "checked_at": stamp,
            "error": f"Couldn't reach PyPI ({reason}).",
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps(result, indent=2) + "\n")
    return result


def due(cache: dict[str, Any], now: datetime | None = None) -> bool:
    checked = cache.get("checked_at")
    if not isinstance(checked, str):
        return True
    try:
        when = datetime.fromisoformat(checked)
    except ValueError:
        return True
    return (now or datetime.now(UTC)) - when >= CHECK_EVERY


@dataclass
class UpdateStatus:
    current: str
    released: str | None
    automatic: bool | None  # the user's choice; None until asked
    latest: str | None
    available: bool
    old: bool  # no network needed: this version is months old and nothing newer is known
    checked_at: str | None
    error: str | None
    notes_url: str | None
    command: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def status(
    automatic: bool | None, cache: dict[str, Any], today: date | None = None
) -> UpdateStatus:
    latest = cache.get("latest") if isinstance(cache.get("latest"), str) else None
    available = is_newer(latest, __version__)
    old = False
    try:
        released = date.fromisoformat(__released__)
        known_current = latest is not None and not available
        old = not known_current and (today or date.today()) - released >= OLD_AFTER
    except ValueError:
        pass
    notes = cache.get("notes_url")
    return UpdateStatus(
        current=__version__,
        released=__released__,
        automatic=automatic,
        latest=latest,
        available=available,
        old=old and not available,
        checked_at=cache.get("checked_at"),
        error=cache.get("error"),
        notes_url=notes if isinstance(notes, str) and notes.startswith("https://") else None,
        command=update_command(),
    )
