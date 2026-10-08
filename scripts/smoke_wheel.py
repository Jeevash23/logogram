"""Verify the installed distribution, bundled assets and authenticated loopback server."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path


def main() -> None:
    import uvicorn

    import logogram
    from logogram.cli import _bind
    from logogram.project import example_source
    from logogram.server.app import create_app, web_dist
    from logogram.server.security import SecurityConfig
    from logogram.spec import Spec

    package = Path(logogram.__file__).resolve()
    assert "site-packages" in package.parts, "Smoke test must use an installed wheel."
    assert (
        subprocess.check_output([sys.executable, "-m", "logogram", "--version"], text=True)
        .strip()
        .startswith("logogram ")
    )
    assert web_dist().joinpath("index.html").is_file()
    assert list(web_dist().glob("assets/*.woff2")), "Local font missing from wheel"
    specs = list(example_source().glob("experiments/*/spec.json"))
    assert specs and Spec.from_path(specs[0]).dataset.path == "datasets/ioi.jsonl"
    assert not list(example_source().rglob("manifest.json"))
    assert not list(example_source().rglob("results.parquet"))

    sock = _bind(0)
    port = sock.getsockname()[1]
    security = SecurityConfig(token="wheel-smoke-token", port=port)
    app = create_app(security)
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        assert server.started, "Installed server did not start"
        headers = {"authorization": "Bearer wheel-smoke-token"}

        def get(path: str) -> bytes:
            request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=headers)
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.read()

        state = json.loads(get("/api/state"))
        assert state["version"] == logogram.__version__
        html = get("/").decode()
        assets = re.findall(r'(?:src|href)="(/assets/[^\"]+)"', html)
        assert assets
        for asset in assets:
            assert get(asset), f"Missing bundled asset: {asset}"
    finally:
        server.should_exit = True
        thread.join(5)
        sock.close()
    print("Installed wheel smoke check passed: CLI, example, local fonts, API and web assets.")


if __name__ == "__main__":
    main()
