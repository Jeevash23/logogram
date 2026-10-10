"""The privacy check passes on this repository, and catches what it should."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NAME = "someone"


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_privacy", ROOT / "scripts" / "check_privacy.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_privacy"] = module
    spec.loader.exec_module(module)
    return module


def _home_paths() -> list[str]:
    """A home path in each form the check knows, built here so that this file holds none."""
    home, users, Users, root = "home", "users", "Users", "root"
    return [
        f"/{home}/{NAME}/data",
        f"/var/{home}/{NAME}/data",  # Fedora's atomic editions
        f"/usr/{home}/{NAME}",  # FreeBSD
        f"/{root}/.cache/huggingface",
        f"/{Users}/{NAME}/project",  # macOS
        f"/mnt/c/{Users}/{NAME}/Documents",  # WSL
        f"/mnt/d/{users}/{NAME}",
        f"/c/{Users}/{NAME}/project",  # Git Bash, MSYS2
        f"C:\\{Users}\\{NAME}\\project",
        f'"C:\\\\{Users}\\\\{NAME}"',  # escaped, as in JSON
        f"C:/{Users}/{NAME}/project",
        f"file:///c:/{users}/{NAME}",
    ]


# Paths that identify nobody, and text that only looks like a home path.
NOT_HOME_PATHS = [
    "/home/runner/work/logogram",
    "/Users/Shared/data",
    "/Users/runner/work",
    "C:/Users/Public/Documents",
    "C:\\Users\\runneradmin\\work",
    "/mnt/data/models",
    "src/root/file.py",
    "https://example.com/root/page",
    "/rooted/x",
    "/var/homework/x",
    'fetch("/users/me")',
    "~/projects/run",
]


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


@pytest.mark.parametrize("path", _home_paths())
def test_detects_home_paths_in_every_form(path):
    check = _load()
    found = list(check.scan_text("leak.txt", f"loaded {path} ok", []))
    assert [f.kind for f in found] == ["home directory path"], path
    assert NAME in found[0].text or "/.cache" in found[0].text


@pytest.mark.parametrize("text", NOT_HOME_PATHS)
def test_paths_that_identify_nobody_are_not_flagged(text):
    assert not list(_load().scan_text("code.py", text, []))


def test_binary_files_are_searched(tmp_path):
    """A stray coverage database or parquet table holds absolute paths in its bytes."""
    check = _load()
    home = ("/" + "home" + f"/{NAME}/project").encode()
    database = b"SQLite format 3\x00" + b"\x00\x01\xfe" * 100 + home + b"/a.py\x00\x02"
    (tmp_path / ".coverage").write_bytes(database)
    windows = ("C:\\" + "Users" + f"\\{NAME}\\run").encode("utf-16-le")
    (tmp_path / "shortcut.lnk").write_bytes(b"\x4c\x00\x00\x00\x01\x14" + windows + b"\x00\x00")
    found = {(f.path, f.kind, f.offset) for f in check.check(tmp_path)}
    assert found == {
        (".coverage", "home directory path", database.index(home)),
        ("shortcut.lnk", "home directory path", 6),
    }


def test_utf16_text_is_read_as_text(tmp_path):
    """Windows PowerShell writes redirected output as UTF-16 with a byte order mark."""
    check = _load()
    log = "\ufeffok\r\n" + "cd C:\\" + "Users" + f"\\{NAME}\\work\r\n"
    (tmp_path / "log.txt").write_bytes(log.encode("utf-16-le"))
    found = check.check(tmp_path)
    assert [(f.kind, f.line) for f in found] == [("home directory path", 2)]


def test_the_bundled_fonts_are_searched_as_binary_and_pass():
    check = _load()
    fonts = sorted((ROOT / "src" / "logogram" / "web_dist" / "assets").glob("*.woff2"))
    assert fonts
    for font in fonts:
        data = font.read_bytes()
        assert b"\0" in data[:8192]
        assert not list(check.scan_binary(font.name, data, check.identity_patterns()))


def test_generic_account_names_are_not_flagged(tmp_path, monkeypatch):
    check = _load()
    monkeypatch.setattr(check.getpass, "getuser", lambda: "runner")
    (tmp_path / "code.py").write_text("from logogram.runner import run_spec\n", encoding="utf-8")
    assert not [f for f in check.check(tmp_path) if f.kind == "username"]


def test_a_short_username_is_found_only_as_user_at_host(monkeypatch):
    """Two or three letters match ordinary words and code, so only ``name@host`` counts."""
    check = _load()
    user = "jd"
    monkeypatch.setattr(check.getpass, "getuser", lambda: user)
    identities = check.identity_patterns()
    flagged = [f"{user}@laptop:~/work$ ls", f"ssh {user}@buildbox", f"[{user}@fedora ~]$"]
    for line in flagged:
        assert [f.kind for f in check.scan_text("log.txt", line, identities)] == ["username"]
    fine = [
        f"{user} = max({user}, 1)",
        f"mail {user}@example.com or {user}@uni.edu",
        f"a{user}@laptop: and {user}.x@laptop:",
        f"y = {user}@w.T",  # matrix product
    ]
    for line in fine:
        assert not [f for f in check.scan_text("a.py", line, identities) if f.kind == "username"]


def test_names_too_short_to_tell_apart_are_skipped(monkeypatch):
    check = _load()
    monkeypatch.setattr(check.getpass, "getuser", lambda: "j")
    monkeypatch.setattr(check.socket, "gethostname", lambda: "pc")
    monkeypatch.setattr(check.platform, "node", lambda: "pc")
    identities = check.identity_patterns()
    assert not list(check.scan_text("log.txt", "j@pc:~$ ls; ssh j@pc; pc j", identities))


def test_long_names_are_found_as_words(monkeypatch):
    check = _load()
    monkeypatch.setattr(check.socket, "gethostname", lambda: "jds-workstation.local")
    identities = check.identity_patterns()
    found = check.scan_text("log.txt", "built on JDS-Workstation today", identities)
    assert [f.kind for f in found] == ["hostname"]
    assert not list(check.scan_text("log.txt", "jds-workstations", identities))
