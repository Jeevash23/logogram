"""Local server security.

* The server binds to 127.0.0.1 only.
* Every HTTP and WebSocket request must carry the session token: as the HttpOnly cookie set when
  the launch URL (``/?token=...``) is opened, or as an ``Authorization: Bearer`` header.
* The browser is opened with a single-use launch code instead of the token, because a browser's
  command line can be read by other users of the machine. The URL printed in the terminal carries
  the token itself, for opening more tabs or another browser.
* The Host header must name this server (defeats DNS rebinding). Requests that carry an Origin
  must come from this server's origin; WebSocket handshakes must carry one, and so must
  cookie-authenticated requests that change anything (or a Fetch Metadata header). Fetch Metadata
  headers must say same-origin. No CORS headers are sent.
"""

from __future__ import annotations

import hmac
import secrets
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from starlette.types import ASGIApp, Receive, Scope, Send

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
# The launch URL. /api/session also works behind the Vite dev proxy.
LOGIN_PATHS = {"/", "/api/session"}


def new_token() -> str:
    return secrets.token_urlsafe(32)


@dataclass
class SecurityConfig:
    token: str
    port: int
    dev_origins: list[str] = field(default_factory=list)
    _launch: tuple[str, float] | None = field(default=None, repr=False)
    _launch_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def cookie_name(self) -> str:
        return f"logogram_{self.port}"

    def allowed_hosts(self) -> set[str]:
        hosts = {f"127.0.0.1:{self.port}", f"localhost:{self.port}", f"[::1]:{self.port}"}
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
            await self._deny(scope, receive, send, 421, "Unknown host.")
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
        query_token = _query_token(scope)
        if scope["type"] == "http" and query_token is not None and scope["path"] in LOGIN_PATHS:
            if cfg.token_ok(query_token) or cfg.use_launch_code(query_token):
                await self._login(send)
            else:
                await self._deny(scope, receive, send, 401, _LOGIN_HELP)
            return

        bearer = _header(scope, b"authorization")
        bearer_token = bearer[7:] if bearer and bearer.lower().startswith("bearer ") else None
        if cfg.token_ok(bearer_token):
            await self.app(scope, receive, send)
            return
        if any(cfg.token_ok(c) for c in _cookies(scope, cfg.cookie_name)):
            # A browser sends the cookie by itself, so a request that changes something must
            # show where it came from (every current browser sends one of these).
            if needs_origin and origin is None and fetch_site is None:
                await self._deny(scope, receive, send, 403, "Missing origin.")
                return
            await self.app(scope, receive, send)
            return
        await self._deny(scope, receive, send, 401, _LOGIN_HELP)

    async def _login(self, send: Send) -> None:
        cookie = f"{self.config.cookie_name}={self.config.token}; HttpOnly; SameSite=Strict; Path=/"
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
            await send({"type": "websocket.close", "code": 4401 if status == 401 else 4403})
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
