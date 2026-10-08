"""Command line: ``logogram`` (start the app), ``open``, ``run``, ``doctor``, ``serve``."""

from __future__ import annotations

import contextlib
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Annotated, Any

import typer

from logogram import __version__


def _quiet_environment() -> None:
    """No telemetry from libraries we depend on, and deterministic cuBLAS."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    help="Logogram: a local workbench for causal experiments inside language models.",
    rich_markup_mode=None,
)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"logogram {__version__}")
        raise typer.Exit()


def _bind(preferred: int) -> socket.socket:
    """Bind 127.0.0.1 on ``preferred`` or the next free port (any free port for 0).

    The socket is handed to the server as it is, so no other process can take the port between
    choosing it and serving on it.
    """
    candidates = [preferred, *range(preferred + 1, min(preferred + 20, 65536))] if preferred else []
    for port in [*candidates, 0]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if sys.platform != "win32":
            # Like uvicorn: reuse a port a just-stopped server left in TIME_WAIT.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return sock
        except OSError:
            sock.close()
    raise RuntimeError("No free port on 127.0.0.1.")


def _open_when_started(server: Any, security: Any) -> None:
    """Open the browser once the server is serving, with a single-use launch code."""

    def wait_and_open() -> None:
        while not server.started:
            if server.should_exit:
                return
            time.sleep(0.1)
        code = security.new_launch_code()
        webbrowser.open(f"http://127.0.0.1:{security.port}/?token={code}")

    threading.Thread(target=wait_and_open, daemon=True).start()


def _serve(
    *,
    port: int,
    open_browser: bool,
    project: Path | None = None,
    dev: bool = False,
) -> None:
    from logogram.server.security import SecurityConfig, new_token

    sock = _bind(port)
    port = int(sock.getsockname()[1])
    token = os.environ.get("LOGOGRAM_TOKEN") if dev else None
    dev_list = (
        [o.strip() for o in os.environ.get("LOGOGRAM_DEV_ORIGINS", "").split(",") if o.strip()]
        or ["http://localhost:5173", "http://127.0.0.1:5173"]
        if dev
        else []
    )
    security = SecurityConfig(token=token or new_token(), port=port, dev_origins=dev_list)
    url = f"http://127.0.0.1:{port}/?token={security.token}"
    # Print the address first: loading PyTorch takes a few seconds.
    typer.echo(f"Logogram {__version__} is running at:\n\n  {url}\n")
    if dev:
        typer.echo(
            f"Development UI (npm run dev):\n\n  {dev_list[0]}/api/session?token={security.token}\n"
        )
    typer.echo("Everything runs on this machine. Press Ctrl+C to stop.")
    _update_notice()

    import uvicorn

    from logogram.runner import configure_determinism
    from logogram.server.app import create_app

    configure_determinism()
    application = create_app(
        security, initial_project=project, serve_web=not dev, check_updates=True
    )
    config = uvicorn.Config(
        application,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        ws="auto",
        # Open event streams (browser tabs) must not keep Ctrl+C from stopping the server.
        timeout_graceful_shutdown=2,
    )
    server = uvicorn.Server(config)
    if open_browser:
        _open_when_started(server, security)
    with contextlib.suppress(KeyboardInterrupt):
        server.run(sockets=[sock])


PortOption = typer.Option(
    min=0, max=65535, help="Port to listen on (127.0.0.1 only; 0 picks any free port)."
)


def _update_notice() -> None:
    """One line about newer versions, from what is already known (no network here)."""
    from logogram import updates
    from logogram.project import config_dir

    choice = _settings().get("update_check")
    info = updates.status(
        choice if isinstance(choice, bool) else None,
        updates.read_cache(config_dir() / "updates.json"),
    )
    if info.available:
        typer.echo(
            f"\nLogogram {info.latest} is out (you have {info.current}). Update: {info.command}"
        )
    elif info.old:
        typer.echo(
            f"\nThis version of Logogram is from {info.released}. "
            "Newer ones may be out: logogram check-updates"
        )


def _settings() -> dict:  # type: ignore[type-arg]
    import json

    from logogram.project import config_dir

    try:
        data = json.loads((config_dir() / "settings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


@app.callback()
def main_callback(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show the version."),
    ] = False,
    port: Annotated[int, PortOption] = 8765,
    no_browser: Annotated[bool, typer.Option("--no-browser", help="Don't open a browser.")] = False,
) -> None:
    """Start Logogram and open it in your browser."""
    _quiet_environment()
    if ctx.invoked_subcommand is None:
        _serve(port=port, open_browser=not no_browser)


@app.command("open")
def open_cmd(
    path: Annotated[Path, typer.Argument(help="A project folder.")],
    port: Annotated[int, PortOption] = 8765,
    no_browser: Annotated[bool, typer.Option("--no-browser")] = False,
) -> None:
    """Start Logogram with a project open."""
    from logogram.project import Project, ProjectError

    try:
        project = Project.open(path)
    except ProjectError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    _serve(port=port, open_browser=not no_browser, project=project.root)


@app.command()
def serve(
    port: Annotated[int, PortOption] = 8765,
    dev: Annotated[
        bool, typer.Option("--dev", help="API only, for the Vite dev server in web/.")
    ] = False,
    no_browser: Annotated[bool, typer.Option("--no-browser")] = False,
    project: Annotated[Path | None, typer.Option(help="Open this project.")] = None,
) -> None:
    """Run the server (use --dev with `npm run dev` when working on the web app)."""
    _serve(port=port, open_browser=not (no_browser or dev), project=project, dev=dev)


@app.command("check-updates")
def check_updates_cmd() -> None:
    """Ask PyPI whether a newer Logogram is out (sends nothing about you or your work)."""
    from logogram import updates
    from logogram.project import config_dir

    path = config_dir() / "updates.json"
    cache = updates.check(path)
    choice = _settings().get("update_check")
    info = updates.status(choice if isinstance(choice, bool) else None, cache)
    if info.error:
        typer.echo(info.error)
    if info.available:
        typer.echo(f"Logogram {info.latest} is out (you have {info.current}).")
        typer.echo(f"Update: {info.command}")
        if info.notes_url:
            typer.echo(f"What's new: {info.notes_url}")
    elif info.latest:
        typer.echo(f"Logogram {info.current} is the newest version.")


@app.command()
def doctor() -> None:
    """Report the environment and hardware, with fixes for common problems."""
    from logogram.system import format_bytes, system_report

    report = system_report()
    rows = [
        ("Logogram", __version__),
        ("Operating system", f"{report.os} ({report.machine})"),
        ("Python", report.python),
        (
            "Processor",
            f"{report.cpu} · {report.cpu_cores or '?'} cores, {report.cpu_threads} threads",
        ),
        (
            "Memory",
            f"{format_bytes(report.memory_total)} total, {format_bytes(report.memory_available)} available",
        ),
    ]
    for gpu in report.gpus:
        free = f", {format_bytes(gpu.memory_free)} free" if gpu.memory_free is not None else ""
        rows.append(("GPU", f"{gpu.name} · {format_bytes(gpu.memory_total)}{free}"))
    if not report.gpus:
        rows.append(("GPU", "none detected"))
    backend = {"cuda": "CUDA", "mps": "Apple Metal (MPS)", "cpu": "CPU"}[report.backend]
    rows += [
        ("Compute backend", backend),
        ("Recommended precision", report.recommended_dtype),
        ("PyTorch", report.torch + (f" (CUDA {report.torch_cuda})" if report.torch_cuda else "")),
        ("TransformerLens", report.transformer_lens or "not installed"),
        ("Transformers", report.transformers or "not installed"),
        ("Updates", _updates_row()),
    ]
    width = max(len(k) for k, _ in rows)
    for key, value in rows:
        typer.echo(f"{key:<{width}}  {value}")
    typer.echo("")
    typer.echo(report.precision_note)
    typer.echo("")
    if not report.issues:
        typer.echo("No problems found.")
        return
    for issue in report.issues:
        typer.echo(f"[{issue.severity}] {issue.title}")
        typer.echo(f"  {issue.detail}")
        if issue.fix:
            typer.echo(f"  Fix: {issue.fix}")
        typer.echo("")


def _updates_row() -> str:
    from logogram import updates
    from logogram.project import config_dir

    choice = _settings().get("update_check")
    info = updates.status(
        choice if isinstance(choice, bool) else None,
        updates.read_cache(config_dir() / "updates.json"),
    )
    automatic = {True: "checked daily", False: "automatic checks off", None: "not set up"}[
        info.automatic
    ]
    known = (
        f"{info.latest} available" if info.available else "up to date" if info.latest else "unknown"
    )
    return f"{known} · {automatic} · check now with logogram check-updates"


@app.command()
def run(
    spec_path: Annotated[Path, typer.Argument(metavar="SPEC.json", help="An experiment spec.")],
    project_dir: Annotated[
        Path | None,
        typer.Option("--project", help="Project folder (default: the one containing the spec)."),
    ] = None,
    top: Annotated[int, typer.Option(help="How many sites to list.")] = 10,
) -> None:
    """Run an experiment headlessly and print a summary."""
    from logogram.project import Project, ProjectError
    from logogram.runner import run_spec
    from logogram.spec import Spec, describe_experiment

    if not spec_path.exists():
        typer.echo(f"Error: {spec_path} doesn't exist.", err=True)
        raise typer.Exit(1)
    root = project_dir or Project.find_root(spec_path.parent)
    if root is None:
        typer.echo("Error: no project.json found above the spec. Pass --project PATH.", err=True)
        raise typer.Exit(1)
    try:
        project = Project.open(root)
        spec = Spec.from_path(project.readable(spec_path.absolute()))
    except (ProjectError, ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    original = spec_path.resolve().parent
    original_results = original / "results.parquet"
    compare_to = (
        project.readable(original_results)
        if project.inside(original_results) and original_results.is_file()
        else None
    )

    typer.echo(f"{spec.name}")
    typer.echo(f"  {describe_experiment(spec)}")
    typer.echo(f"  model {spec.model.id} · dataset {spec.dataset.path}")
    printer = _ProgressPrinter()
    outcome = run_spec(spec, project, on_event=printer)
    printer.close()
    if outcome.status != "finished":
        typer.echo(f"Run {outcome.status}: {outcome.manifest.get('error')}", err=True)
        raise typer.Exit(1)
    _print_summary(outcome.summary or {}, top)
    rel = outcome.folder.relative_to(project.root)
    typer.echo(f"\nSaved to {rel} ({outcome.manifest.get('wall_time_s')} s)")
    if compare_to is not None:
        _report_reproduction(compare_to, outcome.folder / "results.parquet", original.name)


class _ProgressPrinter:
    def __init__(self) -> None:
        self.last = 0.0
        self.active = False

    def __call__(self, kind: str, data: dict) -> None:  # type: ignore[type-arg]
        now = time.monotonic()
        if kind == "model":
            stage = data.get("stage")
            if stage == "downloading" and data.get("total"):
                if now - self.last > 0.25 or data.get("done") == data.get("total"):
                    self.last = now
                    pct = 100 * data["done"] / data["total"]
                    self._line(f"  downloading {pct:5.1f}% of {data['total'] / 1e6:,.0f} MB")
            elif stage in ("resolving", "loading", "processing"):
                self._line(f"  model: {stage}")
        elif kind == "progress":
            if now - self.last > 0.2 or data["done"] == data["total"]:
                self.last = now
                pct = 100 * data["done"] / data["total"]
                self._line(f"  running layer {data['layer']} · {pct:5.1f}%")
        elif kind == "status":
            self._line(f"  {data.get('message')}")

    def _line(self, text: str) -> None:
        if sys.stderr.isatty():
            sys.stderr.write("\r" + text.ljust(60))
            sys.stderr.flush()
            self.active = True
        else:
            sys.stderr.write(text + "\n")

    def close(self) -> None:
        if self.active:
            sys.stderr.write("\r" + " " * 60 + "\r")
            sys.stderr.flush()


def _fmt(x: float | None, signed: bool = True) -> str:
    if x is None:
        return "—"
    return f"{x:+.3f}" if signed else f"{x:.3f}"


def _print_summary(summary: dict, top: int) -> None:  # type: ignore[type-arg]
    base = summary["baseline"]
    clean = base["clean"]["logit_diff"]["mean"]
    corrupt = base["corrupt"]["logit_diff"]["mean"]
    typer.echo("")
    typer.echo(
        f"Baseline over {summary['n_prompts']} prompts: logit difference clean {_fmt(clean)}, "
        f"corrupt {_fmt(corrupt)}"
    )
    typer.echo(f"Normalized effect: {summary['metric']['normalized_effect']}")
    stats = summary["statistics"]
    ci = round(stats["ci"] * 100)
    typer.echo(
        f"Confidence intervals: {ci}% {stats['method']}, {stats['bootstrap']} resamples, seed {stats['seed']}"
    )
    # What the values are depends on the method: patched runs, estimates of them, or terms of a
    # split; only patched runs can flip a prompt.
    measure = summary.get("measure", "intervention")
    value, change = {
        "intervention": ("effect", "Δ logit diff"),
        "estimate": ("estimated", "estimated Δ"),
        "attribution": ("share", "direct"),
    }.get(measure, ("effect", "Δ logit diff"))
    flips = measure == "intervention"
    sites = sorted(summary["sites"], key=lambda s: -abs(s["effect"]["mean"] or 0.0))[:top]
    typer.echo("")
    header = f"{'site':<22} {value:>9}  {f'{ci}% CI':<19} {change:>12}"
    typer.echo(header + (f"  {'flipped':>8}" if flips else ""))
    for s in sites:
        e = s["effect"]
        interval = f"[{_fmt(e['lo'])}, {_fmt(e['hi'])}]"
        line = (
            f"{s['label']:<22} {_fmt(e['mean']):>9}  {interval:<19} {_fmt(s['delta']['mean']):>12}"
        )
        typer.echo(line + (f"  {s['sign_flips']:>3} / {s['n']:<3}" if flips else ""))
    if summary.get("direct"):
        d = summary["direct"]
        typer.echo(
            f"\nThe mean {d['prompts']} logit difference {_fmt(d['logit_diff'])} splits into attention "
            f"{_fmt(d['attention'])}, MLPs {_fmt(d['mlp'])}, embeddings {_fmt(d['embeddings'])} "
            f"and biases {_fmt(d['biases'])}."
        )
    if summary.get("steering"):
        st = summary["steering"]
        typer.echo(
            f"\nDirections from {len(st['train'])} training pairs; measured on the "
            f"{len(st['test'])} held-out pairs."
        )
    if summary.get("features"):
        f = summary["features"]
        fit = f.get("fit") or {}
        typer.echo(
            f"\nSAE {f['sae']['repo']} {f['sae']['path']}: explains "
            f"{fit.get('variance_explained', float('nan')):.1%} of the variance on these prompts."
        )
        if f.get("features_estimate") is not None:
            typer.echo(
                f"Estimated effect of the whole site {_fmt(f['site_estimate'])}, of which the "
                f"features carry {_fmt(f['features_estimate'])}."
            )
    for warning in summary.get("warnings") or []:
        typer.echo(f"\nNote: {warning}")


def _report_reproduction(original: Path, new: Path, original_id: str) -> None:
    import pyarrow.parquet as pq

    a = pq.read_table(original)
    b = pq.read_table(new)
    if a.equals(b):
        typer.echo(
            f"Identical to {original_id}: all {a.num_rows:,} per-prompt values match exactly."
        )
        return
    import numpy as np

    if a.num_rows != b.num_rows:
        typer.echo(f"Differs from {original_id}: {a.num_rows} rows before, {b.num_rows} now.")
        return
    diff = np.abs(
        np.asarray(a.column("patched_logit_diff")) - np.asarray(b.column("patched_logit_diff"))
    )
    typer.echo(
        f"Differs from {original_id}: largest change in a patched logit difference is "
        f"{diff.max():.3g}. Check the device, dtype and library versions in the two manifests."
    )


def main() -> None:
    _quiet_environment()
    app()


if __name__ == "__main__":
    main()
