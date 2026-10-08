"""Update notices: what is checked, when, and that nothing happens without consent."""

from __future__ import annotations

import json
import urllib.error
from datetime import UTC, date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from logogram import __version__, updates
from logogram.server.app import create_app
from logogram.server.security import SecurityConfig

HEADERS = {"authorization": "Bearer t", "origin": "http://127.0.0.1:8765"}


def test_versions_order_like_pep_440():
    ordered = [
        "0.1.0",
        "0.1.1",
        "0.2.0.dev1",
        "0.2.0a1",
        "0.2.0b2",
        "0.2.0rc1",
        "0.2.0",
        "0.2.0.post1",
        "0.10.0",
        "1.0",
    ]
    keys = [updates.version_key(v) for v in ordered]
    assert all(k is not None for k in keys)
    assert keys == sorted(keys)
    assert updates.version_key("1.0") == updates.version_key("1.0.0")
    assert updates.is_newer("0.2.0", "0.1.9")
    assert not updates.is_newer("0.1.0", "0.1.0")
    assert not updates.is_newer("not a version", "0.1.0")
    assert not updates.is_newer(None, "0.1.0")


def test_a_check_caches_the_answer_and_records_failures(tmp_path):
    path = tmp_path / "updates.json"
    result = updates.check(
        path, fetch=lambda: {"latest": "9.0.0", "notes_url": "https://example.com/notes"}
    )
    assert result["latest"] == "9.0.0" and result["error"] is None
    info = updates.status(None, updates.read_cache(path))
    assert (
        info.available and info.latest == "9.0.0" and info.notes_url == "https://example.com/notes"
    )

    def offline():
        raise urllib.error.URLError("no network")

    updates.check(path, fetch=offline)
    cache = updates.read_cache(path)
    assert "Couldn't reach PyPI" in cache["error"] and cache["latest"] == "9.0.0"

    def missing():
        raise urllib.error.HTTPError(updates.PYPI_URL, 404, "Not Found", {}, None)  # type: ignore[arg-type]

    assert updates.check(path, fetch=missing)["error"] == "Logogram isn't on PyPI yet."


def test_only_https_release_notes_are_offered(tmp_path):
    path = tmp_path / "updates.json"
    updates.check(path, fetch=lambda: {"latest": "9.0.0", "notes_url": "javascript:alert(1)"})
    assert updates.status(None, updates.read_cache(path)).notes_url is None


def test_a_day_passes_between_automatic_checks():
    now = datetime(2027, 1, 2, tzinfo=UTC)
    assert updates.due({}, now)
    assert not updates.due({"checked_at": (now - timedelta(hours=3)).isoformat()}, now)
    assert updates.due({"checked_at": (now - timedelta(days=2)).isoformat()}, now)


def test_an_old_version_says_so_without_any_network(monkeypatch):
    monkeypatch.setattr(updates, "__released__", "2026-01-01")
    assert updates.status(None, {}, today=date(2026, 9, 1)).old
    assert not updates.status(None, {}, today=date(2026, 2, 1)).old
    # Known to be the newest: not old, however long ago it was released.
    assert not updates.status(None, {"latest": __version__}, today=date(2030, 1, 1)).old


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    monkeypatch.setattr("logogram.server.state.config_dir", lambda: tmp_path / "config")
    application = create_app(SecurityConfig(token="t", port=8765), serve_web=False)
    calls = []

    def fetch():
        calls.append(1)
        return {"latest": "9.9.9", "notes_url": None}

    application.state.logogram.fetch_updates = fetch
    application.state.calls = calls
    return application


def test_nothing_is_checked_without_consent(app):
    state = app.state.logogram
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    update = client.get("/api/state", headers=HEADERS).json()["update"]
    assert update["automatic"] is None and update["latest"] is None
    assert state.maybe_check_updates() is False
    client.post("/api/settings", json={"update_check": False}, headers=HEADERS)
    assert state.maybe_check_updates() is False
    assert app.state.calls == []


def test_consent_and_check_now(app):
    state = app.state.logogram
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    # "Check now" is the user's own request: it works without consent to daily checks.
    now = client.post("/api/update/check", headers=HEADERS).json()
    assert now["available"] and now["latest"] == "9.9.9" and now["automatic"] is None
    client.post("/api/settings", json={"update_check": True}, headers=HEADERS)
    assert client.get("/api/update", headers=HEADERS).json()["automatic"] is True
    # Just checked: the daily check waits for a day to pass.
    calls = len(app.state.calls)
    assert state.maybe_check_updates() is False
    cache = json.loads(state.updates_path().read_text())
    cache["checked_at"] = "2000-01-01T00:00:00+00:00"
    state.updates_path().write_text(json.dumps(cache))
    assert state.maybe_check_updates() is True
    assert len(app.state.calls) > calls
