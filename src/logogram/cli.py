"""Command line: ``logogram`` (start the app), ``open``, ``run``, ``doctor``, ``serve``, and
commands that read and check runs without the app: ``list``, ``show``, ``compare``, ``diff``,
``verify``, ``robustness``, ``export``, ``validate``, ``tasks`` and ``generate``."""

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


DELTA_LABELS = {
    "logit_diff": "Δ logit diff",
    "logprob_diff": "Δ log-prob diff",
    "logprob": "Δ log-prob",
    "prob": "Δ prob",
    "prob_diff": "Δ prob diff",
    "kl": "Δ KL",
}


def _interval(stat: dict[str, Any]) -> str:
    return f"{_fmt(stat['mean'])} [{_fmt(stat['lo'])}, {_fmt(stat['hi'])}]"


def _print_summary(summary: dict, top: int) -> None:  # type: ignore[type-arg]
    base = summary["baseline"]
    clean = base["clean"]["logit_diff"]["mean"]
    corrupt = base["corrupt"]["logit_diff"]["mean"]
    typer.echo("")
    typer.echo(
        f"Baseline over {summary['n_prompts']} prompts: logit difference clean {_fmt(clean)}, "
        f"corrupt {_fmt(corrupt)}"
    )
    metric = summary["metric"]
    if metric.get("kind", "logit_diff") != "logit_diff":
        values = {k: (base[k].get("metric") or {}).get("mean") for k in ("clean", "corrupt")}
        typer.echo(
            f"Metric: {metric.get('description')}; clean {_fmt(values['clean'])}, corrupt "
            f"{_fmt(values['corrupt'])}"
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
    delta = DELTA_LABELS.get(metric.get("kind", "logit_diff"), "Δ metric")
    value, change = {
        "intervention": ("effect", delta),
        "estimate": ("estimated", "estimated Δ"),
        "attribution": ("share", "direct"),
    }.get(measure, ("effect", delta))
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
        if f.get("chosen_on"):
            typer.echo(
                f"The strongest features were chosen on {len(f['chosen_on'])} prompts and are "
                f"reported on the other {len(f['reported_on'] or [])}."
            )
        if f.get("features_estimate") is not None:
            typer.echo(
                f"Estimated effect of the whole site {_fmt(f['site_estimate'])}, of which the "
                f"features carry {_fmt(f['features_estimate'])}."
            )
    if summary.get("circuit"):
        rows = [r for r in summary["circuit"]["rows"] if r.get("share")]
        if rows:
            typer.echo("\nAgainst replacing everything:")
            for r in rows:
                share = r["share"]
                line = f"  {r['label']:<24} share {_interval(share)}"
                if r.get("faithfulness"):
                    line += f"  faithfulness {_interval(r['faithfulness'])}"
                typer.echo(line)
                if r.get("without"):
                    w = r["without"]
                    typer.echo(
                        f"  {'':<24} {w['site']} adds {_interval(w['drop'])} faithfulness to "
                        f"{w['of']}"
                    )
        for r in summary["circuit"]["rows"]:
            if r.get("interaction"):
                i = r["interaction"]
                typer.echo(
                    f"\n{r['label']}: beyond {i['a']} and {i['b']} alone, an effect of "
                    f"{_interval(i['effect'])}"
                )
    if stats.get("band_level") is not None:
        sites_all = summary["sites"]
        alpha = 1 - stats["ci"]
        banded = sum(
            1 for x in sites_all if x.get("band") and (x["band"]["lo"] > 0 or x["band"]["hi"] < 0)
        )
        discoveries = sum(1 for x in sites_all if x.get("q") is not None and x["q"] < alpha)
        typer.echo(
            f"\nCorrected for {len(sites_all)} sites: {banded} exclude zero with simultaneous "
            f"bands, {discoveries} have q < {alpha:g} (Benjamini–Hochberg)."
        )
    for warning in summary.get("warnings") or []:
        typer.echo(f"\nNote: {warning}")


def _report_reproduction(original: Path, new: Path, original_id: str) -> None:
    import pyarrow.parquet as pq

    from logogram.results import largest_change

    a = pq.read_table(original)
    b = pq.read_table(new)
    largest = largest_change(a, b)
    if largest is None:
        typer.echo(f"Differs from {original_id}: {a.num_rows} rows before, {b.num_rows} now.")
    elif largest == 0.0:
        typer.echo(
            f"Identical to {original_id}: all {a.num_rows:,} per-prompt values match exactly."
        )
    else:
        typer.echo(
            f"Differs from {original_id}: largest change in a per-prompt value is {largest:.3g}. "
            "Check the device, dtype and library versions in the two manifests."
        )


# -- runs without the app ---------------------------------------------------------------------

ProjectOption = typer.Option(
    "--project", help="Project folder (default: the one containing the current folder)."
)


def _fail(message: str) -> typer.Exit:
    typer.echo(f"Error: {message}", err=True)
    return typer.Exit(1)


def _open_project(project_dir: Path | None, near: Path | None = None) -> Any:
    from logogram.project import Project, ProjectError

    root = project_dir or Project.find_root(near or Path.cwd())
    if root is None:
        raise _fail("no project.json found here or above. Pass --project PATH.")
    try:
        return Project.open(root)
    except ProjectError as exc:
        raise _fail(str(exc)) from exc


def _run_id(project: Any, run: str) -> str:
    """A run given by its id, or by its folder (absolute or relative)."""
    path = Path(run)
    if path.exists() and path.is_dir():
        run = path.resolve().name
    try:
        if not (project.run_dir(run) / "spec.json").is_file():
            raise _fail(f"there is no run {run} in {project.meta.name}.")
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    return run


@app.command("list")
def list_cmd(
    project_dir: Annotated[Path | None, ProjectOption] = None,
) -> None:
    """List the project's runs and drafts, newest first."""
    project = _open_project(project_dir)
    runs = project.list_runs()
    if not runs:
        typer.echo("No runs yet. Run a spec with `logogram run SPEC.json`.")
        return
    width = max(len(r.id) for r in runs)
    for r in runs:
        typer.echo(f"{r.id:<{width}}  {r.status:<9}  {r.name}")
        typer.echo(f"{'':<{width}}  {'':<9}  {r.description}")


@app.command()
def show(
    run: Annotated[str, typer.Argument(help="A run id, or its folder.")],
    project_dir: Annotated[Path | None, ProjectOption] = None,
    top: Annotated[int, typer.Option(help="How many sites to list.")] = 10,
) -> None:
    """Print a finished run's method, baseline and strongest sites."""
    project = _open_project(project_dir, Path(run) if Path(run).exists() else None)
    run_id = _run_id(project, run)
    summary = project.read_json(project.run_dir(run_id) / "summary.json", optional=True)
    if summary is None:
        listing = project.run_listing(run_id)
        raise _fail(f"{run_id} has no results ({listing.status if listing else 'unreadable'}).")
    typer.echo(f"{summary['name']}\n  {summary['description']}")
    _print_summary(summary, top)


@app.command()
def compare(
    run_a: Annotated[str, typer.Argument(help="The first run.")],
    run_b: Annotated[str, typer.Argument(help="The second run.")],
    project_dir: Annotated[Path | None, ProjectOption] = None,
) -> None:
    """Compare two finished runs: rank correlation, top sites, conclusions that changed, and
    paired differences where they measured the same prompts."""
    from logogram.compare import CompareError, compare_runs

    project = _open_project(project_dir, Path(run_a) if Path(run_a).exists() else None)
    a, b = _run_id(project, run_a), _run_id(project, run_b)
    try:
        result = compare_runs(project, a, b)
    except (CompareError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    _print_comparison(result)


def _print_comparison(result: dict[str, Any]) -> None:
    rho = result["spearman"]
    typer.echo(
        f"{result['n_common']} sites in common. Rank correlation of mean effects: "
        f"{'—' if rho is None else f'{rho:.3f}'}. Top {result['top_k']} overlap: "
        f"{result['top_overlap']} of {result['top_k']}."
    )
    for difference in result["spec_differences"]:
        typer.echo(f"  {difference['path']}: {difference['a']!r} → {difference['b']!r}")
    if result["paired"]:
        typer.echo(
            f"Paired prompt by prompt: {result['n_differs']} site(s) differ (interval of b − a "
            "excludes zero)."
        )
    for change in result["changes"]:
        flags = ", ".join(change["flags"]) or "top in both"
        a, b = change["effect_a"]["mean"], change["effect_b"]["mean"]
        line = f"  {change['label']:<20} {_fmt(a)} → {_fmt(b)}  ({flags})"
        if change.get("difference"):
            d = change["difference"]
            line += f"  b − a {_fmt(d['mean'])} [{_fmt(d['lo'])}, {_fmt(d['hi'])}]"
        typer.echo(line)


@app.command()
def diff(
    a: Annotated[str, typer.Argument(help="A run id, its folder, or a spec file.")],
    b: Annotated[str, typer.Argument(help="Another run id, folder or spec file.")],
    project_dir: Annotated[Path | None, ProjectOption] = None,
) -> None:
    """List every choice that differs between two specs (name and notes aside)."""
    from logogram.compare import spec_differences
    from logogram.spec import Spec

    def load(ref: str) -> Spec:
        path = Path(ref)
        if path.is_file():
            return Spec.from_path(path)
        project = _open_project(project_dir, path if path.exists() else None)
        return Spec.from_path(project.run_dir(_run_id(project, ref)) / "spec.json")

    try:
        differences = spec_differences(
            load(a).model_dump(mode="json"), load(b).model_dump(mode="json")
        )
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    if not differences:
        typer.echo("The two specs make the same choices.")
        return
    for d in differences:
        typer.echo(f"{d['path']}: {d['a']!r} → {d['b']!r}")


def _run_and_report(project: Any, spec: Any, derived: dict[str, Any], top: int) -> Any:
    from logogram.runner import run_spec

    printer = _ProgressPrinter()
    outcome = run_spec(spec, project, on_event=printer, derived_from=derived)
    printer.close()
    if outcome.status != "finished":
        raise _fail(f"the run {outcome.status}: {outcome.manifest.get('error')}")
    _print_summary(outcome.summary or {}, top)
    typer.echo(f"\nSaved to {outcome.folder.relative_to(project.root)}")
    return outcome


@app.command()
def verify(
    run: Annotated[str, typer.Argument(help="An attribution patching run.")],
    top: Annotated[int, typer.Option(help="How many of the strongest estimates to patch.")] = 10,
    project_dir: Annotated[Path | None, ProjectOption] = None,
) -> None:
    """Patch, for real, the sites an attribution patching run estimated to matter most, and
    compare the measurement with the estimate."""
    from logogram.compare import compare_runs
    from logogram.spec import Spec
    from logogram.verify import verification_spec

    project = _open_project(project_dir, Path(run) if Path(run).exists() else None)
    run_id = _run_id(project, run)
    folder = project.run_dir(run_id)
    summary = project.read_json(folder / "summary.json", optional=True)
    if summary is None:
        raise _fail("only a finished run can be verified.")
    try:
        spec = verification_spec(Spec.from_path(folder / "spec.json"), summary, top)
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    n = len(spec.scope.sites)  # type: ignore[union-attr]
    derived = {
        "run": run_id,
        "kind": "verification",
        "change": f"the {n} strongest estimated site{'s' if n != 1 else ''}, patched",
    }
    outcome = _run_and_report(project, spec, derived, top)
    typer.echo("")
    _print_comparison(compare_runs(project, run_id, outcome.run_id))


@app.command()
def robustness(
    run: Annotated[str, typer.Argument(help="A finished run.")],
    experiment: Annotated[
        str | None,
        typer.Option(help='The experiment to run instead, as JSON: {"kind": "ablation", ...}.'),
    ] = None,
    dtype: Annotated[
        str | None, typer.Option(help="The dtype to rerun in instead (float32, float16, bfloat16).")
    ] = None,
    top: Annotated[int, typer.Option(help="How many sites to list.")] = 10,
    project_dir: Annotated[Path | None, ProjectOption] = None,
) -> None:
    """Rerun with one methodological choice changed, then compare the two runs."""
    import json

    from logogram.compare import compare_runs
    from logogram.robustness import robustness_spec
    from logogram.spec import Spec

    project = _open_project(project_dir, Path(run) if Path(run).exists() else None)
    run_id = _run_id(project, run)
    try:
        change = json.loads(experiment) if experiment is not None else None
        spec, words = robustness_spec(
            Spec.from_path(project.run_dir(run_id) / "spec.json"), experiment=change, dtype=dtype
        )
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    outcome = _run_and_report(
        project, spec, {"run": run_id, "kind": "robustness", "change": words}, top
    )
    typer.echo("")
    _print_comparison(compare_runs(project, run_id, outcome.run_id))


@app.command()
def export(
    run: Annotated[str, typer.Argument(help="A finished run.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Where to write the CSV (default: stdout)."),
    ] = None,
    project_dir: Annotated[Path | None, ProjectOption] = None,
) -> None:
    """Write a finished run's per-prompt values as CSV: one row per site and prompt."""
    from logogram.exports import results_csv

    project = _open_project(project_dir, Path(run) if Path(run).exists() else None)
    run_id = _run_id(project, run)
    path = project.run_dir(run_id) / "results.parquet"
    if not path.is_file():
        raise _fail(f"{run_id} has no results to export.")
    chunks = results_csv(project.readable(path))
    if output is None:
        for chunk in chunks:
            sys.stdout.write(chunk)
        return
    with output.open("w", encoding="utf-8", newline="") as fh:
        for chunk in chunks:
            fh.write(chunk)
    typer.echo(f"Wrote {output}", err=True)


@app.command()
def validate(
    spec_path: Annotated[Path, typer.Argument(metavar="SPEC.json", help="An experiment spec.")],
) -> None:
    """Check a spec without running it, and say what it describes and what it left out."""
    from pydantic import ValidationError

    from logogram.spec import Spec, describe_experiment

    try:
        spec = Spec.from_path(spec_path)
    except ValidationError as exc:
        typer.echo(f"{spec_path} isn't a valid spec:", err=True)
        for error in exc.errors():
            where = ".".join(str(p) for p in error["loc"]) or "spec"
            typer.echo(f"  {where}: {error['msg']}", err=True)
        raise typer.Exit(1) from exc
    except (OSError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"{spec.name}\n  {describe_experiment(spec)}")
    if spec.upgraded_fields:
        typer.echo(
            "This is a version 1 spec. These choices weren't stated and take version 1's values:"
        )
        for field in spec.upgraded_fields:
            typer.echo(f"  {field}")
    else:
        typer.echo("Every choice that can change a number is stated.")


@app.command()
def tasks() -> None:
    """List the tasks Logogram generates prompts for, with their options and metric."""
    from logogram.tasks import TASKS

    for task in TASKS.values():
        typer.echo(f"{task.id}: {task.name}")
        typer.echo(f"  {task.description}")
        typer.echo(f"  Reads its answers with metric.kind = {task.metric!r}.")
        for option in task.options:
            default = option.default
            if isinstance(default, list | tuple):
                default = ",".join(default)
            allowed = f" ({', '.join(option.allowed)})" if option.allowed else ""
            typer.echo(f"  --option {option.name}={default}{allowed}")
            typer.echo(f"      {option.description}")


def _task_options(task_id: str, given: list[str]) -> dict[str, Any]:
    """Options typed as NAME=VALUE, read by the task's own option types."""
    from logogram.tasks import TaskError, get_task

    try:
        task = get_task(task_id)
    except TaskError as exc:
        raise _fail(str(exc)) from exc
    types = {o.name: o.type for o in task.options}
    options: dict[str, Any] = {}
    for item in given:
        name, sep, value = item.partition("=")
        if not sep or name not in types:
            known = ", ".join(types) or "none"
            raise _fail(f"give options as NAME=VALUE; {task_id}'s options are {known}.")
        if types[name] == "bool":
            if value.lower() not in ("true", "false"):
                raise _fail(f"{name} is true or false.")
            options[name] = value.lower() == "true"
        elif types[name] == "choices":
            options[name] = [v.strip() for v in value.split(",") if v.strip()]
        else:
            options[name] = value
    return options


@app.command()
def generate(
    task: Annotated[str, typer.Argument(help="A task, as `logogram tasks` lists them.")],
    n: Annotated[int, typer.Option("-n", help="How many prompt pairs.")],
    seed: Annotated[int, typer.Option(help="The generator's seed: the same seed, the same file.")],
    name: Annotated[
        str | None, typer.Option(help="The dataset's file name (default: the task's id).")
    ] = None,
    option: Annotated[
        list[str] | None,
        typer.Option("--option", "-O", help="A task option as NAME=VALUE (lists comma-separated)."),
    ] = None,
    tokenizer: Annotated[
        str | None,
        typer.Option(
            help="A model already in the Hugging Face cache whose tokenizer the prompts are "
            "made for (words it splits are left out). Nothing is downloaded."
        ),
    ] = None,
    overwrite: Annotated[bool, typer.Option(help="Replace a dataset of the same name.")] = False,
    project_dir: Annotated[Path | None, ProjectOption] = None,
) -> None:
    """Write a seeded dataset of prompt pairs for a task into the project's datasets folder."""
    from logogram.datasets import check_dataset_name, write_dataset
    from logogram.tasks import TaskError, generate_task, get_task

    project = _open_project(project_dir)
    options = _task_options(task, option or [])
    single = count = None
    if tokenizer is not None:
        single, count = _cached_tokenizer(tokenizer)
    try:
        records = generate_task(task, n, seed, options, single_token=single, token_count=count)
        path = project.dataset_file(check_dataset_name(name or task))
    except (TaskError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    if path.exists() and not overwrite:
        raise _fail(f"{path.name} already exists. Choose another --name, or pass --overwrite.")
    write_dataset(path, records)
    rel = path.relative_to(project.root).as_posix()
    metric = get_task(task).recommended_metric(options)
    typer.echo(f"Wrote {len(records)} prompt pairs to {rel}.")
    typer.echo(f'Read them with "metric": {{"kind": "{metric}", ...}} in the spec.')
    if tokenizer is None:
        typer.echo(
            "Words were assumed to be single tokens. Pass --tokenizer MODEL to check them "
            "against the model's own tokenizer."
        )


def _cached_tokenizer(model_id: str) -> tuple[Any, Any]:
    """single_token and token_count from a tokenizer in the Hugging Face cache (never fetched)."""
    _quiet_environment()
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            model_id, local_files_only=True, trust_remote_code=False
        )
    except Exception as exc:
        raise _fail(
            f"{model_id}'s tokenizer isn't in the Hugging Face cache. Load the model once (in the "
            "app or with `logogram run`), or leave out --tokenizer."
        ) from exc

    def ids(text: str) -> list[int]:
        return list(tok(text, add_special_tokens=False)["input_ids"])

    return (lambda text: len(ids(text)) == 1), (lambda text: len(ids(text)))


def main() -> None:
    _quiet_environment()
    app()


if __name__ == "__main__":
    main()
