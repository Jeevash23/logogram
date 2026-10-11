"""Local server security.

* The server binds to 127.0.0.1 only.
* Every HTTP and WebSocket request must carry the session token: as the HttpOnly cookie set when
  the launch URL (``/?token=...``) is opened, or as an ``Authorization: Bearer`` header.
* The browser is opened with a single-use launch code instead of the token, because a browser's
  command line can be read by other users of the machine. The URL printed in the terminal carries
  the token itself, for opening more tabs or another browser.
* The Host header must name this server (defeats DNS rebinding): a loopback name (127.0.0.1,
  localhost or [::1]) on its own port, or on a port the user named as forwarding to it, such as
  an SSH tunnel's local end (``--forwarded-port``). Requests that carry an Origin must come from
  one of those origins; WebSocket handshakes must carry one, and so must cookie-authenticated
  requests that change anything (or a Fetch Metadata header). Fetch Metadata headers must say
  same-origin. No CORS headers are sent.
"""

from __future__ import annotations

import hmac
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from starlette.types import ASGIApp, Receive, Scope, Send

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
# The launch URL. /api/session also works behind the Vite dev proxy.
LOGIN_PATHS = {"/", "/api/session"}
# The only names the server answers to. Any other name could be a DNS rebinding attack: a web
# page whose own name has been pointed at 127.0.0.1.
LOOPBACK_NAMES = ("127.0.0.1", "localhost", "[::1]")
# Where the forwarded ports can also be given, as a list such as "9000" or "9000,9001".
FORWARDED_PORTS_ENV = "LOGOGRAM_FORWARDED_PORTS"


def new_token() -> str:
    return secrets.token_urlsafe(32)


def parse_ports(text: str | None) -> list[int]:
    """Port numbers separated by commas or spaces, as in LOGOGRAM_FORWARDED_PORTS."""
    ports: list[int] = []
    for part in re.split(r"[\s,]+", text or ""):
        if not part:
            continue
        if not (part.isascii() and part.isdigit()) or not 1 <= int(part) <= 65535:
            raise ValueError(
                f"{part!r} isn't a port number. {FORWARDED_PORTS_ENV} lists ports from 1 to "
                "65535, separated by commas, such as 9000,9001."
            )
        ports.append(int(part))
    return ports


def _host_forms(name: str, port: int) -> set[str]:
    """How a browser writes a host and port in the Host header: port 80 is left out."""
    return {f"{name}:{port}", name} if port == 80 else {f"{name}:{port}"}


@dataclass
class SecurityConfig:
    token: str
    port: int
    dev_origins: list[str] = field(default_factory=list)
    # Local ports that forward to this server, as the end of an SSH tunnel does
    # (ssh -L 9000:127.0.0.1:8765). The browser addresses the server by that port, so loopback
    # names on it are this server too. Nothing else about a request is trusted more for it.
    forwarded_ports: list[int] = field(default_factory=list)
    _launch: tuple[str, float] | None = field(default=None, repr=False)
    _launch_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def cookie_name(self) -> str:
        return f"logogram_{self.port}"

    def cookie_name_for(self, host: str | None) -> str:
        """The session cookie for an allowed Host. Through a forwarded port it is named for that
        port: cookies aren't kept apart by port, and the browser's machine may run a Logogram of
        its own on this server's port, whose cookie this one mustn't replace."""
        for port in self.forwarded_ports:
            if port != self.port and any(host in _host_forms(n, port) for n in LOOPBACK_NAMES):
                return f"logogram_{port}"
        return self.cookie_name

    def allowed_hosts(self) -> set[str]:
        hosts = {
            form
            for port in (self.port, *self.forwarded_ports)
            for name in LOOPBACK_NAMES
            for form in _host_forms(name, port)
        }
        for origin in self.dev_origins:
            hosts.add(urlsplit(origin).netloc)
        return hosts

    def allowed_origins(self) -> set[str]:
        origins = {f"http://{h}" for h in self.allowed_hosts()}
        origins.update(self.dev_origins)
        return origins

    def token_ok(self, candidate: str | None) -> bool:
        return bool(candidate) and hmac.compare_digest(candidate.encode(), self.token.encode())

    def new_launch_code(self, ttl: float = 600.0) -> str:
        """A code that logs in once, within ``ttl`` seconds, for the URL the browser is opened
        with."""
        code = secrets.token_urlsafe(24)
        with self._launch_lock:
            self._launch = (code, time.monotonic() + ttl)
        return code

    def use_launch_code(self, candidate: str) -> bool:
        with self._launch_lock:
            if self._launch is None or not candidate:
                return False
            code, expires = self._launch
            if time.monotonic() > expires:
                self._launch = None
                return False
            if not hmac.compare_digest(candidate.encode(), code.encode()):
                return False
            self._launch = None  # once only
            return True


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key == name:
            return value.decode("latin-1")
    return None


def _cookies(scope: Scope, name: str) -> list[str]:
    """Every cookie called ``name``. Cookies aren't separated by port, so a page served from
    another local port can add one with the same name; it mustn't hide the real one."""
    values = []
    for key, value in scope.get("headers", []):
        if key != b"cookie":
            continue
        for part in value.decode("latin-1").split(";"):
            k, _, v = part.strip().partition("=")
            if k == name:
                values.append(v)
    return values


def _query_token(scope: Scope) -> str | None:
    query = scope.get("query_string", b"").decode("latin-1")
    for part in query.split("&"):
        key, _, value = part.partition("=")
        if key == "token":
            return value
    return None


class SecurityMiddleware:
    """Pure ASGI middleware so it covers HTTP and WebSocket alike."""

    def __init__(self, app: ASGIApp, config: SecurityConfig) -> None:
        self.app = app
        self.config = config

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        cfg = self.config
        host = _header(scope, b"host")
        if host not in cfg.allowed_hosts():
            await self._deny(scope, receive, send, 421, unknown_host_help(host, cfg))
            return
        origin = _header(scope, b"origin")
        method = scope.get("method", "GET")
        needs_origin = scope["type"] == "websocket" or method not in SAFE_METHODS
        if origin is not None and origin not in cfg.allowed_origins():
            await self._deny(scope, receive, send, 403, "Cross-origin requests are not allowed.")
            return
        if needs_origin and origin is None and scope["type"] == "websocket":
            await self._deny(scope, receive, send, 403, "Missing origin.")
            return
        # Fetch Metadata: refuse requests started by another site, including other local ports
        # (which count as the same *site*, so SameSite cookies alone don't stop them).
        fetch_site = _header(scope, b"sec-fetch-site")
        if fetch_site is not None and fetch_site not in ("same-origin", "none"):
            await self._deny(scope, receive, send, 403, "Cross-site requests are not allowed.")
            return

        # Opening the launch URL exchanges the token (or the single-use launch code) for a
        # cookie, then drops it from the URL.
        cookie_name = cfg.cookie_name_for(host)
        query_token = _query_token(scope)
        if scope["type"] == "http" and query_token is not None and scope["path"] in LOGIN_PATHS:
            if cfg.token_ok(query_token) or cfg.use_launch_code(query_token):
                await self._login(send, cookie_name)
            else:
                await self._deny(scope, receive, send, 401, _LOGIN_HELP)
            return

        bearer = _header(scope, b"authorization")
        bearer_token = bearer[7:] if bearer and bearer.lower().startswith("bearer ") else None
        if cfg.token_ok(bearer_token):
            await self.app(scope, receive, send)
            return
        if any(cfg.token_ok(c) for c in _cookies(scope, cookie_name)):
            # A browser sends the cookie by itself, so a request that changes something must
            # show where it came from (every current browser sends one of these).
            if needs_origin and origin is None and fetch_site is None:
                await self._deny(scope, receive, send, 403, "Missing origin.")
                return
            await self.app(scope, receive, send)
            return
        await self._deny(scope, receive, send, 401, _LOGIN_HELP)

    async def _login(self, send: Send, cookie_name: str) -> None:
        cookie = f"{cookie_name}={self.config.token}; HttpOnly; SameSite=Strict; Path=/"
        await send(
            {
                "type": "http.response.start",
                "status": 303,
                "headers": [
                    (b"location", b"/"),
                    (b"set-cookie", cookie.encode("latin-1")),
                    (b"cache-control", b"no-store"),
                    (b"referrer-policy", b"no-referrer"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": b""})

    async def _deny(
        self, scope: Scope, receive: Receive, send: Send, status: int, text: str
    ) -> None:
        if scope["type"] == "websocket":
            if status == 401:
                # Same host and origin, but no valid token: a tab whose session ended (the server
                # restarted). Browsers report a refused handshake only as "abnormal closure", so
                # accept and close at once with 4401, which the tab can read and explain. Nothing
                # is sent before the close.
                message = await receive()
                if message.get("type") == "websocket.connect":
                    await send({"type": "websocket.accept"})
                    await send({"type": "websocket.close", "code": 4401})
                return
            await send({"type": "websocket.close", "code": 4403})
            return
        body = text.encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"text/plain; charset=utf-8"),
                    (b"content-length", str(len(body)).encode()),
                    (b"cache-control", b"no-store"),
                    # Shown as text, whatever it contains.
                    (b"x-content-type-options", b"nosniff"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


_LOGIN_HELP = (
    "Logogram needs its session token.\n\n"
    "Open the link printed in the terminal where you started `logogram` "
    "(it ends with ?token=...). Each start creates a new token, and the link the browser was "
    "first opened with works only once.\n"
)

# A Host header as browsers write one: a name, or an IPv6 address in brackets, and maybe a port.
# The page that refuses a request repeats its Host header only when it has this shape.
_HOST_SHAPE = re.compile(
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9.-]{0,252}|\[[0-9A-Fa-f:.]{2,45}\])"
    r"(?::(?P<port>[1-9][0-9]{0,4}))?"
)


def unknown_host_help(host: str | None, config: SecurityConfig) -> str:
    """The page for a request whose Host header names no address of this server: what went wrong
    and how to fix it. The header is repeated only when it is plainly a host and port, so the
    page never echoes arbitrary text (and it is sent as plain text, never sniffed as HTML)."""
    port = config.port
    shape = _HOST_SHAPE.fullmatch(host or "")
    if shape is not None and int(shape["port"] or 80) > 65535:
        shape = None
    loopback = ", ".join(LOOPBACK_NAMES[:-1]) + f" and {LOOPBACK_NAMES[-1]}"
    why = (
        f"Logogram answers only to the loopback names {loopback}, on the ports it was started "
        "with, so that web pages from elsewhere can't reach it."
    )
    if shape is not None and shape["name"] in LOOPBACK_NAMES:
        asked = int(shape["port"] or 80)
        sent = host if shape["port"] else f"{host} (port 80)"
        return (
            f"Logogram didn't answer this request: it was sent to {sent}, and this Logogram "
            f"listens on port {port}.\n\n"
            f"If port {asked} forwards to Logogram, as the local end of an SSH tunnel does "
            f"(ssh -L {asked}:127.0.0.1:{port} HOST), start Logogram again with that port "
            "allowed:\n\n"
            f"    logogram --no-browser --forwarded-port {asked}\n\n"
            f"(`logogram open` and `logogram serve` take the same option, or set "
            f"{FORWARDED_PORTS_ENV}={asked}), then open the address it prints for port "
            f"{asked}.\n\n{why}\n"
        )
    sent = f"it was sent to {host}" if shape is not None else "it didn't name a host Logogram knows"
    return (
        f"Logogram didn't answer this request: {sent}. {why}\n\n"
        f"Open http://127.0.0.1:{port}/ on the machine where Logogram runs, with the token it "
        "printed. To use it from another computer, forward the port over SSH "
        f"(ssh -L {port}:127.0.0.1:{port} HOST) and open the same address there. If the port "
        "on that computer is another one, start Logogram with --forwarded-port and that port.\n"
    )
