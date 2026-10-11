"""Which process runs a run, Ctrl+C, the traceback a failed run keeps, and the environment a
run's numbers depend on."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import sysconfig
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import logogram as lg
from logogram import owner as owners
from logogram.cli import app as cli
from logogram.project import ProjectError
from logogram.runner import INTERRUPTED, run_spec, scrub_paths, write_json, write_manifest
from logogram.schema import RunOwner
from logogram.server.app import create_app
from logogram.server.security import SecurityConfig

BASE = "http://127.0.0.1:8765"
AUTH = {"authorization": "Bearer test", "origin": BASE}
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks and pipes")
runner = CliRunner()


@pytest.fixture
def server(tiny_backend, tmp_path, monkeypatch):
    """The app with a model loaded and no project open yet, so tests choose when it opens."""
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    app = create_app(SecurityConfig(token="test", port=8765), serve_web=False)
    state = app.state.logogram
    state.backend = tiny_backend
    state.model_status = {"state": "ready"}
    with TestClient(app, base_url=BASE, headers=AUTH) as client:
        yield state, client


@pytest.fixture
def sleeper():
    """Another process, standing in for one that runs a run."""
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    yield process
    process.kill()
    process.wait()


def _running_run(project: Any, spec: Any, process: psutil.Process | None) -> str:
    """A run whose manifest says it is running in ``process`` (no owner record with None)."""
    run_id, folder = project.new_run_dir(spec.name)
    write_json(folder / "spec.json", spec.model_dump(mode="json"))
    if process is not None:
        record = RunOwner(pid=process.pid, started=process.create_time(), token="0" * 32)
        write_json(folder / owners.OWNER_FILE, record.model_dump(mode="json"))
    manifest = {
        "run_id": run_id,
        "status": "running",
        "started_at": "2026-01-01T00:00:00Z",
        "versions": {},
    }
    write_manifest(folder / "manifest.json", manifest)
    return run_id


def _manifest(project: Any, run_id: str) -> dict[str, Any]:
    text = (project.run_dir(run_id) / "manifest.json").read_text(encoding="utf-8")
    return json.loads(text)


def _listed(client: TestClient, run_id: str) -> dict[str, Any]:
    response = client.get("/api/runs")
    assert response.status_code == 200, response.text
    return next(r for r in response.json() if r["id"] == run_id)


def _wait(
    condition: Callable[[], Any], process: subprocess.Popen[str], timeout: float = 300
) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = condition()
        if value:
            return value
        if process.poll() is not None:
            out, err = process.communicate()
            pytest.fail(f"The other process ended early:\n{out}\n{err}")
        time.sleep(0.1)
    pytest.fail("Timed out waiting for the other process.")


# -- the process that runs a run ----------------------------------------------------------------


def test_a_run_records_its_process_while_it_runs(project, tiny_backend, spec_factory):
    seen: dict[str, Any] = {}

    def on_event(kind: str, data: dict[str, Any]) -> None:
        if kind == "started":
            folder = project.run_dir(data["run_id"])
            seen["record"] = owners.read_owner(project, folder)
            seen["state"] = owners.owner_state(seen["record"])
            seen["listing"] = project.run_listing(data["run_id"])
            seen["file"] = json.loads((folder / owners.OWNER_FILE).read_text(encoding="utf-8"))

    outcome = run_spec(spec_factory(), project, backend=tiny_backend, on_event=on_event)
    assert outcome.status == "finished"
    record = seen["record"]
    assert record.pid == os.getpid()
    assert record.started == pytest.approx(psutil.Process().create_time(), abs=1)
    # The process id, its start time and a random token: nothing that names the machine or you.
    assert set(seen["file"]) == {"logogram_owner", "pid", "started", "token"}
    assert seen["state"] == "here"
    assert seen["listing"].status == "running" and not seen["listing"].elsewhere
    assert not (outcome.folder / owners.OWNER_FILE).exists()
    failed = run_spec(
        spec_factory(dataset={"path": "datasets/missing.jsonl"}), project, backend=tiny_backend
    )
    assert failed.status == "failed"
    assert not (failed.folder / owners.OWNER_FILE).exists()


def test_a_run_in_another_process_is_left_running_until_it_ends(
    server, project, tiny_model_dir, spec_factory, tmp_path
):
    """A run started by `logogram run` or the Python API, still going, isn't marked failed when
    the app opens the project, and the listing sees it finish."""
    _, client = server
    go = tmp_path / "go"
    spec_path = tmp_path / "spec.json"
    spec = spec_factory(name="Elsewhere").model_dump(mode="json")
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    script = tmp_path / "elsewhere.py"
    script.write_text(ELSEWHERE, encoding="utf-8")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1"}
    args = [str(project.root), str(tiny_model_dir), str(go), str(spec_path)]
    process = subprocess.Popen(
        [sys.executable, str(script), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        run_id = _wait(
            lambda: next((r.id for r in project.list_runs() if r.status == "running"), None),
            process,
        )
        response = client.post("/api/projects/open", json={"path": str(project.root)})
        assert response.status_code == 200, response.text
        listing = _listed(client, run_id)
        assert listing["status"] == "running" and listing["elsewhere"] is True
        assert _manifest(project, run_id)["status"] == "running"  # left alone
        detail = client.get(f"/api/runs/{run_id}").json()
        assert detail["listing"]["elsewhere"] is True and detail["traceback"] is None

        go.write_text("go", encoding="utf-8")
        out, err = process.communicate(timeout=300)
        assert out.strip().splitlines()[-1:] == ["finished"], err
        listing = _listed(client, run_id)
        assert listing["status"] == "finished" and listing["elsewhere"] is False
        assert not (project.run_dir(run_id) / owners.OWNER_FILE).exists()
        # The app opening the project again changes nothing about a finished run.
        assert (
            client.post("/api/projects/open", json={"path": str(project.root)}).status_code == 200
        )
        assert _manifest(project, run_id)["status"] == "finished"
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


# Runs a spec in its own process, as `logogram run` or a notebook would, holding it at the model
# load until the file named by its third argument exists.
ELSEWHERE = """
import json
import sys
import time
from pathlib import Path

from logogram.backends.transformer_lens import TransformerLensBackend, boot_local
from logogram.project import Project
from logogram.runner import configure_determinism, run_spec
from logogram.spec import Spec

root, model_dir, go, spec_path = sys.argv[1:5]
spec = Spec.model_validate(json.loads(Path(spec_path).read_text(encoding="utf-8")))


def provider(ref):
    while not Path(go).exists():
        time.sleep(0.05)
    configure_determinism()
    bridge = boot_local(Path(model_dir), device="cpu", dtype="float32")
    return TransformerLensBackend.from_bridge(
        bridge, model_id="tiny-gpt2", revision="test", dtype="float32", process_weights=True
    )


print(run_spec(spec, Project.open(root), provider=provider).status, flush=True)
"""


def test_a_run_whose_process_is_gone_reads_as_interrupted(server, project, spec_factory, sleeper):
    _, client = server
    run_id = _running_run(project, spec_factory(), psutil.Process(sleeper.pid))
    listing = project.run_listing(run_id)
    assert listing is not None and listing.status == "running" and listing.elsewhere
    listed = runner.invoke(cli, ["list", "--project", str(project.root)])
    assert "running in another process" in listed.output
    assert lg.load_run(project, run_id).status == "running"
    sleeper.kill()
    sleeper.wait()
    assert lg.load_run(project, run_id).status == "failed"
    listing = project.run_listing(run_id)
    assert listing is not None and listing.status == "failed" and not listing.elsewhere
    assert listing.error == owners.OWNER_STOPPED
    assert _manifest(project, run_id)["status"] == "running"  # listing reads, never writes
    # Opening the project records it, so every reader sees it the same way.
    assert client.post("/api/projects/open", json={"path": str(project.root)}).status_code == 200
    manifest = _manifest(project, run_id)
    assert manifest["status"] == "failed" and manifest["error"] == owners.OWNER_STOPPED
    assert not (project.run_dir(run_id) / owners.OWNER_FILE).exists()
    assert _listed(client, run_id)["error"] == owners.OWNER_STOPPED


def test_a_process_that_reuses_the_id_is_not_the_run(project, spec_factory):
    """The same process id with another start time is a later process: the run's is gone."""
    me = psutil.Process()
    folder = project.run_dir(_running_run(project, spec_factory(), None))
    record = RunOwner(pid=me.pid, started=me.create_time() - 3600, token="1" * 32)
    write_json(folder / owners.OWNER_FILE, record.model_dump(mode="json"))
    assert owners.owner_state(owners.read_owner(project, folder)) == "gone"
    assert owners.owner_state(None) == "unknown"
    gone = RunOwner(pid=2**31 + 5, started=1.0, token="2" * 32)
    assert owners.owner_state(gone) == "gone"


def test_without_an_owner_record_opening_marks_a_running_run_interrupted(
    server, project, spec_factory
):
    """Runs from before owner records: as before, opening the project marks them failed."""
    _, client = server
    run_id = _running_run(project, spec_factory(), None)
    assert client.post("/api/projects/open", json={"path": str(project.root)}).status_code == 200
    listing = _listed(client, run_id)
    assert listing["status"] == "failed" and listing["error"] == owners.SERVER_STOPPED
    assert _manifest(project, run_id)["status"] == "failed"


def test_an_owner_record_is_read_like_any_project_file(project, spec_factory, tmp_path):
    folder = project.run_dir(_running_run(project, spec_factory(), None))
    path = folder / owners.OWNER_FILE
    assert owners.read_owner(project, folder) is None
    valid = RunOwner(pid=os.getpid(), started=psutil.Process().create_time(), token="3" * 32)
    path.write_text(valid.model_dump_json(), encoding="utf-8")
    assert owners.read_owner(project, folder) == valid
    for broken in (
        "not json",
        "[]",
        json.dumps({**valid.model_dump(), "pid": "1"}),
        json.dumps({**valid.model_dump(), "token": "not hex"}),
        json.dumps({**valid.model_dump(), "started": -1}),
        json.dumps({**valid.model_dump(), "host": "a-laptop"}),
        valid.model_dump_json() + " " * owners.OWNER_LIMIT,  # valid, but larger than any record
    ):
        path.write_text(broken, encoding="utf-8")
        assert owners.read_owner(project, folder) is None, broken
    # Without a record it can trust, a running manifest is what it always was: not known to be
    # going elsewhere, and interrupted once the project opens.
    listing = project.run_listing(folder.name)
    assert listing is not None and listing.status == "running" and not listing.elsewhere


@posix_only
def test_an_owner_record_outside_the_project_or_not_a_file_is_ignored(
    project, spec_factory, tmp_path
):
    folder = project.run_dir(_running_run(project, spec_factory(), None))
    outside = tmp_path / "outside.json"
    me = psutil.Process()
    outside.write_text(
        RunOwner(pid=me.pid, started=me.create_time(), token="4" * 32).model_dump_json(),
        encoding="utf-8",
    )
    (folder / owners.OWNER_FILE).symlink_to(outside)
    assert owners.read_owner(project, folder) is None
    (folder / owners.OWNER_FILE).unlink()
    os.mkfifo(folder / owners.OWNER_FILE)
    assert owners.read_owner(project, folder) is None  # a pipe never blocks the listing
    owners.remove_owner(project, folder)
    assert not os.path.lexists(folder / owners.OWNER_FILE)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_an_owner_record_is_never_removed_outside_the_project(project, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / owners.OWNER_FILE).write_text("{}", encoding="utf-8")
    project.experiments_dir.mkdir(exist_ok=True)
    (project.experiments_dir / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ProjectError, match="outside the project"):
        owners.remove_owner(project, project.run_dir("linked"))
    assert (outside / owners.OWNER_FILE).exists()


def test_releasing_leaves_another_process_record(project, spec_factory):
    folder = project.run_dir(_running_run(project, spec_factory(), None))
    mine = owners.claim_run(project, folder)
    other = RunOwner(pid=mine.pid, started=mine.started, token="5" * 32)
    write_json(folder / owners.OWNER_FILE, other.model_dump(mode="json"))
    owners.release_run(project, folder, mine)
    assert owners.read_owner(project, folder) == other
    owners.release_run(project, folder, other)
    assert not (folder / owners.OWNER_FILE).exists()


# -- Ctrl+C -------------------------------------------------------------------------------------


def _interrupting(ref: Any) -> Any:
    raise KeyboardInterrupt


def test_ctrl_c_records_the_run_as_cancelled_then_stops(project, spec_factory):
    events: list[tuple[str, dict[str, Any]]] = []
    with pytest.raises(KeyboardInterrupt):
        run_spec(
            spec_factory(),
            project,
            provider=_interrupting,
            on_event=lambda kind, data: events.append((kind, data)),
        )
    (folder,) = [f for f in project.experiments_dir.iterdir() if f.is_dir()]
    manifest = _manifest(project, folder.name)
    assert manifest["status"] == "cancelled" and manifest["error"] == INTERRUPTED
    assert manifest["finished_at"] is not None
    assert not (folder / owners.OWNER_FILE).exists()
    listing = project.run_listing(folder.name)
    assert listing is not None and listing.status == "cancelled"
    assert ("cancelled", {"run_id": folder.name, "error": INTERRUPTED}) in events


def test_ctrl_c_in_logogram_run(project, spec_factory, monkeypatch):
    monkeypatch.setattr("logogram.runner.default_provider", lambda on_event=None: _interrupting)
    spec_path = project.root / "spec.json"
    spec_path.write_text(json.dumps(spec_factory().model_dump(mode="json")), encoding="utf-8")
    result = runner.invoke(cli, ["run", str(spec_path), "--project", str(project.root)])
    assert result.exit_code == 130, result.output
    assert "Interrupted. The run is recorded as cancelled" in result.output
    statuses = [r.status for r in project.list_runs()]
    assert "cancelled" in statuses and "running" not in statuses


# -- what a failed run keeps --------------------------------------------------------------------


def _load_from(folder: Path) -> None:
    raise RuntimeError(
        f"Unrecognized model in {Path.home()}/.cache/huggingface/hub/models--x/snapshots/abc "
        f"(read from {folder})"
    )


def test_a_failed_run_keeps_a_scrubbed_traceback(server, project, spec_factory, monkeypatch):
    state, client = server

    def broken(ref: Any) -> Any:
        _load_from(project.root / "models")

    outcome = run_spec(spec_factory(), project, provider=broken)
    assert outcome.status == "failed"
    saved = (outcome.folder / "traceback.txt").read_text(encoding="utf-8")
    assert saved == outcome.traceback
    assert saved.startswith("Traceback (most recent call last):")
    assert f"<logogram>{os.sep}runner.py" in saved and "in _load_from" in saved
    assert "RuntimeError: Unrecognized model in " in saved and f"<project>{os.sep}models" in saved
    for place in (str(Path.home()), str(project.root), str(Path(__file__).parent)):
        assert place not in saved
    assert outcome.manifest["error"] in saved  # the same words as the error, and more

    state.open_project(project)
    detail = client.get(f"/api/runs/{outcome.run_id}").json()
    assert detail["traceback"] == saved
    assert lg.load_run(project, outcome.run_id).traceback == saved
    monkeypatch.setattr("logogram.runner.default_provider", lambda on_event=None: broken)
    with pytest.raises(lg.api.LogogramError, match=r"traceback\.txt"):
        lg.run(spec_factory(), project)

    shown = runner.invoke(cli, ["show", outcome.run_id, "--project", str(project.root)])
    assert shown.exit_code == 1
    assert "Failed: Unrecognized model" in shown.output
    assert "Traceback (most recent call last):" in shown.output

    # Another attempt in a failed run's folder drops the traceback it kept.
    retry = run_spec(spec_factory(), project, run_id=outcome.run_id, backend=state.backend)
    assert retry.status == "finished" and retry.traceback is None
    assert not (outcome.folder / "traceback.txt").exists()
    assert client.get(f"/api/runs/{outcome.run_id}").json()["traceback"] is None


def test_logogram_run_prints_the_traceback_with_verbose(project, spec_factory, monkeypatch):
    def broken(ref: Any) -> Any:
        _load_from(project.root)

    monkeypatch.setattr("logogram.runner.default_provider", lambda on_event=None: broken)
    spec_path = project.root / "spec.json"
    spec_path.write_text(json.dumps(spec_factory().model_dump(mode="json")), encoding="utf-8")
    args = ["run", str(spec_path), "--project", str(project.root)]
    quiet = runner.invoke(cli, args)
    assert quiet.exit_code == 1
    assert "The run failed: Unrecognized model" in quiet.output
    assert "/traceback.txt; --verbose prints it." in quiet.output
    assert "Traceback (most recent call last)" not in quiet.output
    verbose = runner.invoke(cli, [*args, "--verbose"])
    assert verbose.exit_code == 1
    assert "Traceback (most recent call last)" in verbose.output
    assert f"<logogram>{os.sep}runner.py" in verbose.output
    assert str(Path.home()) not in verbose.output


def test_scrubbing_a_traceback_with_home_and_library_paths(tmp_path, monkeypatch):
    """A traceback through code in the home folder, the standard library and an installed
    library keeps its file names and lines, and loses this machine's folders and names."""
    from huggingface_hub.utils import validate_repo_id

    home = tmp_path / "home" / "ada-lovelace"
    module_path = home / "work" / "model_code.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(
        "import json\n\n\ndef parse(text):\n    return json.loads(text)\n\n\n"
        "def check(repo):\n    validate(repo)\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("model_code", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate = validate_repo_id

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr("getpass.getuser", lambda: "ada-lovelace")
    monkeypatch.setattr("socket.gethostname", lambda: "analytical-engine.local")
    monkeypatch.setattr("platform.node", lambda: "analytical-engine")
    tracebacks = []
    for call in (lambda: module.parse("{"), lambda: module.check("not/a/valid/repo")):
        try:
            call()
        except Exception as exc:  # noqa: BLE001 - the traceback is what this test reads
            import traceback

            tracebacks.append("".join(traceback.format_exception(exc)))
    site = sysconfig.get_paths()["purelib"]
    stdlib = sysconfig.get_paths()["stdlib"]
    raw = "\n".join(tracebacks)
    for there in (str(home), stdlib, site):  # what there is to scrub
        assert there.lower() in raw.lower()
    other = "/" + "home" + "/someone/venv/lib/python3.12/site-packages/torch/nn/module.py"
    windows = "C:\\" + "Users" + "\\Ada Lovelace\\AppData\\Local\\Temp\\run\\x.py"
    raw += (
        f'\n  File "{other}", line 1736, in _call_impl'
        f'\n  File "{windows}", line 3, in <module>'
        "\nOSError: ada-lovelace can't read /mnt/models/gpt2/config.json on analytical-engine "
        "(see https://huggingface.co/openai-community/gpt2, ~/notes and ./relative/path)"
    )
    text = scrub_paths(raw, tmp_path / "project")
    for leak in (str(home), stdlib, site, "ada-lovelace", "Ada Lovelace", "analytical-engine"):
        assert leak.lower() not in text.lower(), leak
    assert "someone" not in text and "/mnt/models" not in text
    s = os.sep
    assert f'File "~{s}work{s}model_code.py", line 5, in parse' in text
    assert f'File "<python>{s}json{s}decoder.py"' in text
    assert f'File "<site-packages>{s}huggingface_hub{s}utils{s}_validators.py"' in text
    assert 'File "~/venv/lib/python3.12/site-packages/torch/nn/module.py", line 1736' in text
    assert 'File "~\\AppData\\Local\\Temp\\run\\x.py", line 3' in text
    assert "OSError: <user> can't read <path>/config.json on <host>" in text
    assert "https://huggingface.co/openai-community/gpt2, ~/notes and ./relative/path" in text
    assert "JSONDecodeError" in text and "HFValidationError" in text


def test_scrubbing_keeps_names_that_are_not_paths():
    assert scrub_paths("ratio 3/4 and/or a/b, openai-community/gpt2", None) == (
        "ratio 3/4 and/or a/b, openai-community/gpt2"
    )
    # A path is matched as a whole name: /tmp in /tmp/x, not in /var/tmp/x or /tmpdata/x.
    assert scrub_paths("/proj/x /proj2/y /var/proj/z", "/proj") == "<project>/x <path>/y <path>/z"


# -- the environment ----------------------------------------------------------------------------


def test_logogram_env_names_versions_and_devices_but_not_the_machine():
    import getpass
    import platform
    import socket

    result = runner.invoke(cli, ["env"])
    assert result.exit_code == 0, result.output
    env = json.loads(result.output)
    assert env["logogram"] == lg.__version__ and env["python"] == platform.python_version()
    assert env["os"] in ("linux", "macos", "windows")
    assert "cpu" in env["devices"] and env["default_device"] in env["devices"]
    assert {"torch", "transformer_lens", "transformers", "numpy"} <= set(env["versions"])
    assert "hardware" not in env
    text = result.output.lower()
    names = {getpass.getuser(), socket.gethostname().split(".")[0], platform.node().split(".")[0]}
    for name in (n for n in names if len(n) >= 4):
        assert f'"{name.lower()}"' not in text
    for place in (str(Path.home()), sys.prefix, sys.executable):
        assert json.dumps(place).lower().strip('"') not in text
    from logogram.system import cpu_name

    if cpu_name() not in (platform.machine(), platform.processor()):  # a model name, not "arm64"
        assert cpu_name().lower() not in text

    hardware = json.loads(runner.invoke(cli, ["env", "--hardware"]).output)["hardware"]
    assert hardware["cpu"] == cpu_name() and hardware["memory"] > 0


def test_logogram_env_requirements_pin_the_versions():
    from importlib.metadata import version

    result = runner.invoke(cli, ["env", "--requirements"])
    assert result.exit_code == 0, result.output
    pins = {
        line.split("==")[0]: line.split("==")[1]
        for line in result.output.splitlines()
        if "==" in line and not line.startswith("#")
    }
    assert pins["logogram"] == lg.__version__
    assert pins["torch"] == version("torch").split("+")[0]
    assert pins["transformer-lens"] == version("transformer-lens")
    assert pins["numpy"] == version("numpy")
    assert result.output.startswith("# Logogram's environment: Python ")


def test_the_server_reports_its_environment_with_the_manifest_s_names(server):
    _, client = server
    env = client.get("/api/environment").json()
    from logogram.runner import environment_versions

    recorded = environment_versions()
    for key in ("torch", "transformer_lens", "transformers", "numpy"):
        assert env["versions"][key] == recorded[key]
    assert env["logogram"] == recorded["logogram"] and "hardware" not in env


def test_compare_says_which_versions_and_devices_differ(project, tiny_backend, spec_factory):
    a = run_spec(spec_factory(), project, backend=tiny_backend)
    b = run_spec(spec_factory(), project, backend=tiny_backend)
    assert (
        lg.compare_runs(lg.load_run(project, a.run_id), lg.load_run(project, b.run_id))[
            "environment_differences"
        ]
        == []
    )
    manifest = _manifest(project, b.run_id)
    manifest["versions"]["torch"] = "1.0.0"
    manifest["device"].update(type="cuda", name="Some GPU", cuda="12.4")
    write_manifest(project.run_dir(b.run_id) / "manifest.json", manifest)
    result = lg.compare_runs(lg.load_run(project, a.run_id), lg.load_run(project, b.run_id))
    differences = {d["key"]: (d["a"], d["b"]) for d in result["environment_differences"]}
    torch_version = _manifest(project, a.run_id)["versions"]["torch"]
    assert differences == {"torch": (torch_version, "1.0.0"), "device.type": ("cpu", "cuda")}
    compared = runner.invoke(cli, ["compare", a.run_id, b.run_id, "--project", str(project.root)])
    assert compared.exit_code == 0, compared.output
    assert "They ran with other versions or devices" in compared.output
    assert f"PyTorch: {torch_version} → 1.0.0" in compared.output
    assert "Device: cpu → cuda" in compared.output


def test_the_server_commands_take_verbose_and_a_log_file(project, monkeypatch, tmp_path):
    from logogram import cli as cli_module

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(cli_module, "_serve", lambda **kwargs: calls.append(kwargs))
    log_file = tmp_path / "server.log"
    for args in (
        ["--no-browser", "--verbose", "--log-file", str(log_file)],
        ["serve", "--no-browser", "-v", "--log-file", str(log_file)],
        ["open", str(project.root), "--no-browser", "--verbose"],
        ["serve", "--no-browser"],
    ):
        result = runner.invoke(cli, args)
        assert result.exit_code == 0, result.output
    assert [(c["verbose"], c["log_file"]) for c in calls] == [
        (True, log_file),
        (True, log_file),
        (True, None),
        (False, None),
    ]


def test_the_server_log_is_quiet_unless_verbose(tmp_path):
    import logging

    import typer

    from logogram import cli as cli_module

    log_file = tmp_path / "server.log"
    root = logging.getLogger()
    saved = (root.handlers[:], root.level, logging.getLogger("logogram").level)
    try:
        cli_module._configure_logging(False, log_file)
        logging.getLogger("logogram.server.state").info("what the server does")
        logging.getLogger("logogram.server.state").error("an error")
        cli_module._configure_logging(True, log_file)
        logging.getLogger("logogram.runner").info("a run failed, with its traceback")
        logging.getLogger("uvicorn.error").info("uvicorn says hello")
        for handler in root.handlers:
            handler.flush()
        written = log_file.read_text(encoding="utf-8")
        assert "an error" in written and "a run failed, with its traceback" in written
        assert "uvicorn says hello" in written
        assert "what the server does" not in written
        with pytest.raises(typer.Exit):
            cli_module._configure_logging(False, tmp_path / "missing" / "server.log")
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            handler.close()
        for handler in saved[0]:
            root.addHandler(handler)
        root.setLevel(saved[1])
        logging.getLogger("logogram").setLevel(saved[2])
