"""The server's address checks: the Host and Origin headers it accepts, with and without ports
that forward to it (an SSH tunnel's local end), and what the page that refuses a host says."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from typer.testing import CliRunner

from logogram.cli import app as cli
from logogram.server.app import create_app
from logogram.server.security import (
    FORWARDED_PORTS_ENV,
    SecurityConfig,
    SecurityMiddleware,
    parse_ports,
    unknown_host_help,
)

TOKEN = "test-token"
AUTH = {"authorization": f"Bearer {TOKEN}"}


async def _ok(scope: Any, receive: Any, send: Any) -> None:
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def call(
    config: SecurityConfig,
    headers: dict[str, str] | None = None,
    *,
    method: str = "GET",
    path: str = "/api/state",
    query: str = "",
) -> tuple[int, dict[bytes, bytes], str]:
    """One HTTP request through the security layer alone, exactly as the headers are given (a
    Host header too, or none)."""
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": query.encode("latin-1"),
        "headers": [
            (k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in (headers or {}).items()
        ],
    }
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asyncio.run(SecurityMiddleware(_ok, config)(scope, receive, send))
    body = b"".join(m.get("body", b"") for m in sent[1:]).decode("utf-8")
    return sent[0]["status"], dict(sent[0]["headers"]), body


def config(*forwarded: int, port: int = 8765) -> SecurityConfig:
    return SecurityConfig(token=TOKEN, port=port, forwarded_ports=list(forwarded))


# -- Host ----------------------------------------------------------------------------------------

LOOPBACK = ["127.0.0.1", "localhost", "[::1]"]
# Never this server, whatever ports forward to it: other names (a DNS rebinding attack names its
# own), other loopback spellings, and text that only starts like an allowed host.
FOREIGN = [
    "evil.example",
    "evil.example:8765",
    "evil.example:9000",
    "192.168.1.20:8765",
    "192.168.1.20:9000",
    "0.0.0.0:8765",
    "127.0.0.2:8765",
    "127.1:8765",
    "[::ffff:127.0.0.1]:8765",
    "[0:0:0:0:0:0:0:1]:8765",
    "localhost.:8765",
    "LOCALHOST:8765",
    "127.0.0.1:8765.evil.example",
    "127.0.0.1:9000.evil.example",
    "127.0.0.1:8765 ",
    "user@127.0.0.1:8765",
    "127.0.0.1:08765",
    "127.0.0.1:",
    "",
]


@pytest.mark.parametrize("name", LOOPBACK)
def test_loopback_names_on_the_servers_port_are_accepted(name):
    assert call(config(), {"host": f"{name}:8765", **AUTH})[0] == 200


@pytest.mark.parametrize("host", ["127.0.0.1:9000", "localhost:9000", "[::1]:9000", "127.0.0.1"])
def test_other_ports_are_refused_until_they_are_named(host):
    assert call(config(), {"host": host, **AUTH})[0] == 421
    port = int(host.rpartition(":")[2]) if host.endswith("9000") else 80
    assert call(config(port), {"host": host, **AUTH})[0] == 200


@pytest.mark.parametrize("host", FOREIGN)
@pytest.mark.parametrize("forwarded", [(), (9000,), (9000, 9001)])
def test_foreign_hosts_are_refused_with_or_without_forwarded_ports(host, forwarded):
    assert call(config(*forwarded), {"host": host, **AUTH})[0] == 421


def test_a_request_without_a_host_is_refused():
    status, _, body = call(config(9000), AUTH)
    assert status == 421 and "didn't name a host Logogram knows" in body


def test_forwarded_ports_add_loopback_names_on_those_ports_only():
    cfg = config(9000, 9001)
    for name in LOOPBACK:
        for port in (8765, 9000, 9001):
            assert call(cfg, {"host": f"{name}:{port}", **AUTH})[0] == 200
        for port in (9002, 80, 443, 8766):
            assert call(cfg, {"host": f"{name}:{port}", **AUTH})[0] == 421
    expected = {f"{n}:{p}" for n in LOOPBACK for p in (8765, 9000, 9001)}
    assert cfg.allowed_hosts() == expected


def test_port_80_is_named_without_a_port_as_browsers_write_it():
    cfg = config(80)
    for host in ("127.0.0.1", "localhost", "[::1]", "127.0.0.1:80"):
        assert call(cfg, {"host": host, **AUTH})[0] == 200
    assert "http://localhost" in cfg.allowed_origins()
    assert call(cfg, {"host": "127.0.0.1", "origin": "http://127.0.0.1", **AUTH})[0] == 200
    # The server's own port, when it is 80, is written the same way.
    assert call(config(port=80), {"host": "localhost", **AUTH})[0] == 200


# -- Origin --------------------------------------------------------------------------------------

ALLOWED_ORIGINS = [
    "http://127.0.0.1:9000",
    "http://localhost:9000",
    "http://[::1]:9000",
    "http://127.0.0.1:8765",
]
REFUSED_ORIGINS = [
    "https://127.0.0.1:9000",
    "http://127.0.0.1:9001",
    "http://evil.example:9000",
    "http://evil.example",
    "http://127.0.0.1",
    "http://localhost.:9000",
    "http://127.0.0.1:9000/",
    "http://127.0.0.1:9000.evil.example",
    "ws://127.0.0.1:9000",
    "null",
    "",
]


@pytest.mark.parametrize("origin", ALLOWED_ORIGINS)
def test_origins_on_forwarded_ports_are_accepted(origin):
    for method in ("GET", "POST"):
        headers = {"host": "127.0.0.1:9000", "origin": origin, **AUTH}
        assert call(config(9000), headers, method=method)[0] == 200


@pytest.mark.parametrize("origin", REFUSED_ORIGINS)
def test_other_origins_are_refused(origin):
    for method in ("GET", "POST"):
        headers = {"host": "127.0.0.1:9000", "origin": origin, **AUTH}
        assert call(config(9000), headers, method=method)[0] == 403


def test_without_forwarded_ports_their_origins_are_refused():
    headers = {"host": "127.0.0.1:8765", "origin": "http://127.0.0.1:9000", **AUTH}
    assert call(config(), headers)[0] == 403


# -- what a forwarded port doesn't change --------------------------------------------------------


def test_a_forwarded_port_still_needs_the_token_and_fetch_metadata():
    cfg = config(9000)
    host = {"host": "127.0.0.1:9000"}
    assert call(cfg, host)[0] == 401
    assert call(cfg, {**host, "authorization": "Bearer wrong"})[0] == 401
    assert call(cfg, {**host, **AUTH})[0] == 200
    cookie = {**host, "cookie": f"logogram_9000={TOKEN}"}
    assert call(cfg, cookie)[0] == 200
    assert call(cfg, {**host, "cookie": "logogram_9000=planted"})[0] == 401
    # A cookie that changes something must say where it comes from.
    assert call(cfg, cookie, method="POST")[0] == 403
    assert call(cfg, {**cookie, "origin": "http://127.0.0.1:9000"}, method="POST")[0] == 200
    assert call(cfg, {**cookie, "sec-fetch-site": "same-origin"}, method="POST")[0] == 200
    # Another local port is the same *site*: Fetch Metadata refuses it.
    for site in ("same-site", "cross-site"):
        assert call(cfg, {**host, **AUTH, "sec-fetch-site": site})[0] == 403
        assert call(cfg, {**cookie, "sec-fetch-site": site})[0] == 403


def test_the_launch_url_through_a_forwarded_port_sets_a_cookie_of_its_own():
    """The browser's computer may run a Logogram of its own on this server's port, and cookies
    aren't kept apart by port: through a forwarded port the cookie is named for that port."""
    cfg = config(9000)
    status, headers, _ = call(cfg, {"host": "127.0.0.1:9000"}, path="/", query=f"token={TOKEN}")
    assert status == 303
    cookie = headers[b"set-cookie"].decode()
    assert cookie.startswith(f"logogram_9000={TOKEN};")
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie
    # The single-use launch code works there too, once.
    code = cfg.new_launch_code()
    assert call(cfg, {"host": "localhost:9000"}, path="/", query=f"token={code}")[0] == 303
    assert call(cfg, {"host": "localhost:9000"}, path="/", query=f"token={code}")[0] == 401
    # On the server's own port, the cookie keeps its name.
    _, headers, _ = call(cfg, {"host": "127.0.0.1:8765"}, path="/", query=f"token={TOKEN}")
    assert headers[b"set-cookie"].decode().startswith(f"logogram_8765={TOKEN};")
    # Each address reads its own cookie: the other Logogram's is neither needed nor used.
    other = {"host": "127.0.0.1:9000", "cookie": f"logogram_8765={TOKEN}"}
    assert call(cfg, other)[0] == 401
    both = {"host": "127.0.0.1:9000", "cookie": f"logogram_8765=theirs; logogram_9000={TOKEN}"}
    assert call(cfg, both)[0] == 200


def test_through_the_real_app_a_forwarded_port_works_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    application = create_app(config(9000), serve_web=False)
    client = TestClient(application, base_url="http://127.0.0.1:9000")
    assert client.get("/api/state").status_code == 401
    login = client.get(f"/?token={TOKEN}", follow_redirects=False)
    assert login.status_code == 303 and "logogram_9000" in login.headers["set-cookie"]
    state = client.get("/api/state")
    assert state.status_code == 200
    # The page's content security policy lets it open its event stream on this address.
    assert "connect-src 'self' ws://127.0.0.1:9000" in state.headers["content-security-policy"]
    origin = {"origin": "http://127.0.0.1:9000"}
    assert client.post("/api/projects/close", headers=origin).status_code == 200
    # No CORS headers, ever.
    options = client.options("/api/state", headers={**AUTH, **origin})
    assert not any(k.startswith("access-control-") for k in options.headers)
    with client.websocket_connect("ws://127.0.0.1:9000/ws", headers=origin) as ws:
        assert ws.receive_json()["type"] == "hello"
    # An address that isn't forwarded, or a foreign origin, gets no event stream.
    refused = [
        ("ws://127.0.0.1:9001/ws", {"origin": "http://127.0.0.1:9001"}),
        ("ws://127.0.0.1:9000/ws", {"origin": "http://evil.example:9000"}),
        ("ws://127.0.0.1:9000/ws", {}),
    ]
    for url, headers in refused:
        with (
            pytest.raises(WebSocketDisconnect) as closed,
            client.websocket_connect(url, headers={**AUTH, **headers}) as ws,
        ):
            ws.receive_json()
        assert closed.value.code == 4403


# -- the page that refuses a host ----------------------------------------------------------------


def test_the_421_page_says_how_to_allow_a_forwarded_port():
    status, headers, body = call(config(), {"host": "127.0.0.1:9000", **AUTH})
    assert status == 421
    assert headers[b"content-type"] == b"text/plain; charset=utf-8"
    assert headers[b"x-content-type-options"] == b"nosniff"
    assert "sent to 127.0.0.1:9000, and this Logogram listens on port 8765" in body
    assert "logogram --no-browser --forwarded-port 9000" in body
    assert f"{FORWARDED_PORTS_ENV}=9000" in body
    assert "ssh -L 9000:127.0.0.1:8765 HOST" in body
    assert TOKEN not in body
    # Without a port in the address, the port is 80.
    body = call(config(), {"host": "localhost"})[2]
    assert "localhost (port 80)" in body and "--forwarded-port 80" in body


def test_the_421_page_for_another_name_explains_loopback_and_tunnels():
    status, _, body = call(config(9000), {"host": "gpu-box.example:8765"})
    assert status == 421
    assert "sent to gpu-box.example:8765" in body
    assert "127.0.0.1, localhost and [::1]" in body
    assert "http://127.0.0.1:8765/" in body and "ssh -L 8765:127.0.0.1:8765 HOST" in body
    assert "--forwarded-port 8765" not in body


@pytest.mark.parametrize(
    "host",
    [
        "<script>alert(1)</script>",
        "evil.example:8765/<img src=x onerror=alert(1)>",
        "evil.example\r\nSet-Cookie: logogram_8765=x",
        "evil.example:8765 <b>",
        '"evil"',
        "a" * 300,
        "évil.example",
        "evil.example:123456",
        # Ports no browser sends: the page mustn't suggest forwarding them.
        "127.0.0.1:99999",
        "127.0.0.1:0",
        "localhost:08765",
    ],
)
def test_the_421_page_never_repeats_arbitrary_text(host):
    status, _, body = call(config(), {"host": host})
    assert status == 421
    assert "didn't name a host Logogram knows" in body
    for piece in ("<", ">", '"', "évil", "alert", "Set-Cookie", "a" * 20):
        assert piece not in body
    assert body == unknown_host_help(host, config())


# -- configuration -------------------------------------------------------------------------------


def test_forwarded_ports_are_read_from_a_list():
    assert parse_ports(None) == [] and parse_ports("") == [] and parse_ports(" , ") == []
    assert parse_ports("9000") == [9000]
    assert parse_ports("9000,9001 9002, 1 65535") == [9000, 9001, 9002, 1, 65535]
    for bad in ("0", "65536", "-1", "90a0", "9000;9001", "http://127.0.0.1:9000", "1e3", "\u0669"):
        with pytest.raises(ValueError, match=FORWARDED_PORTS_ENV):
            parse_ports(bad)


@pytest.fixture
def started(monkeypatch, tmp_path):
    """Start the server from the command line without serving: what it was given, and the
    terminal's text."""
    import uvicorn

    given: dict[str, Any] = {}

    def fake_create_app(security: SecurityConfig, **kwargs: Any) -> Any:
        given["security"] = security
        given.update(kwargs)
        return _ok

    def fake_run(self: Any, sockets: Any = None) -> None:
        for sock in sockets or []:
            sock.close()

    monkeypatch.setattr("logogram.server.app.create_app", fake_create_app)
    monkeypatch.setattr(uvicorn.Server, "run", fake_run)
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    monkeypatch.delenv(FORWARDED_PORTS_ENV, raising=False)
    return given


@pytest.mark.parametrize("command", [[], ["open", "PROJECT"], ["serve"]])
def test_every_command_that_starts_the_server_takes_forwarded_ports(
    started, command, tmp_path, monkeypatch
):
    from logogram.project import Project

    root = Project.create(tmp_path, "Remote").root
    args = [str(root) if a == "PROJECT" else a for a in command]
    options = [
        "--port",
        "0",
        "--no-browser",
        "--forwarded-port",
        "9000",
        "--forwarded-port",
        "9001",
    ]
    monkeypatch.setenv(FORWARDED_PORTS_ENV, "9001,9002")
    result = CliRunner().invoke(cli, [*args, *options] if command else options)
    assert result.exit_code == 0, result.output
    security = started["security"]
    assert security.forwarded_ports == [9000, 9001, 9002]
    port = security.port
    assert f"http://127.0.0.1:{port}/?token={security.token}" in result.output
    assert "Through the forwarded ports, open:" in result.output
    for forwarded in (9000, 9001, 9002):
        assert f"  http://127.0.0.1:{forwarded}/?token={security.token}" in result.output


def test_one_forwarded_port_and_none(started, monkeypatch):
    result = CliRunner().invoke(cli, ["serve", "--port", "0", "--no-browser"])
    assert result.exit_code == 0, result.output
    assert started["security"].forwarded_ports == []
    assert "forwarded" not in result.output
    monkeypatch.setenv(FORWARDED_PORTS_ENV, "9000")
    result = CliRunner().invoke(cli, ["serve", "--port", "0", "--no-browser"])
    assert result.exit_code == 0, result.output
    assert started["security"].forwarded_ports == [9000]
    assert "Through the forwarded port, open:" in result.output


def test_a_bad_forwarded_port_stops_before_the_server_starts(started, monkeypatch):
    monkeypatch.setenv(FORWARDED_PORTS_ENV, "9000,tunnel")
    result = CliRunner().invoke(cli, ["serve", "--port", "0", "--no-browser"])
    assert result.exit_code == 1
    assert "'tunnel' isn't a port number" in result.output
    assert "security" not in started
    result = CliRunner().invoke(cli, ["serve", "--port", "0", "--forwarded-port", "0"])
    assert result.exit_code == 2 and "security" not in started
