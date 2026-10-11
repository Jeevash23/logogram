"""The problem report: what it says, what it never says, and the places that build it (the app's
Report a problem and ``logogram doctor --report``)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from logogram import __version__, support
from logogram.cli import app as cli
from logogram.server.app import create_app
from logogram.server.security import SecurityConfig
from logogram.support import NEW_ISSUE_URL, REPOSITORY, build_report, scrub
from logogram.system import GPU, Issue, SystemReport

ROOT = Path(__file__).resolve().parents[1]
GB = 1024**3
# Made-up names of a person and a machine, as the report must never show them.
USER = "adalovelace"
HOST = "adas-workstation"


def fake_system(**changes: Any) -> SystemReport:
    values: dict[str, Any] = {
        "os": "Ubuntu 24.04 LTS",
        "machine": "x86_64",
        "python": "3.12.3",
        "cpu": "Contoso Quantum 9000X",
        "cpu_cores": 8,
        "cpu_threads": 16,
        "memory_total": 64 * GB,
        "memory_available": 32 * GB,
        "gpus": [GPU("NVIDIA GeForce RTX 9090", 24 * GB, None, True)],
        "backend": "cuda",
        "torch": "2.5.1",
        "torch_cuda": "12.4",
        "transformer_lens": "4.0.0",
        "transformers": "4.46.0",
        "recommended_dtype": "float32",
        "precision_note": "",
        "hf_cache_free": 100 * GB,
        "mps_available": False,
        "issues": [],
    }
    values.update(changes)
    return SystemReport(**values)


HARDWARE = ["Ubuntu", "Contoso", "Quantum", "RTX 9090", "64.0 GB", "24.0 GB", "8 cores"]


def rows(report: support.ProblemReport) -> dict[str, str]:
    return {row.label: row.value for row in report.rows}


@pytest.fixture
def machine(monkeypatch, tmp_path):
    """A person's machine: their account and host names, and a home folder with a space."""
    home = tmp_path / "Ada Lovelace"
    home.mkdir()
    monkeypatch.setattr(support.getpass, "getuser", lambda: USER)
    monkeypatch.setattr(support.socket, "gethostname", lambda: f"{HOST}.local")
    monkeypatch.setattr(support.platform, "node", lambda: HOST)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def test_the_report_names_versions_and_the_device_and_nothing_that_tells_machines_apart():
    report = build_report(model={"state": "none"}, system=fake_system())
    values = rows(report)
    assert list(values) == [
        "Logogram",
        "Python",
        "PyTorch",
        "TransformerLens",
        "Transformers",
        "Operating system",
        "Device",
        "System check",
        "Model",
    ]
    assert values["Logogram"] == __version__
    assert values["PyTorch"] == "2.5.1 (CUDA 12.4)"
    assert values["TransformerLens"] == "4.0.0" and values["Transformers"] == "4.46.0"
    assert values["Device"] == "CUDA, 1 GPU"
    assert values["System check"] == "no problems found"
    assert values["Model"] == "none loaded"
    assert not report.hardware
    for detail in HARDWARE:
        assert detail not in report.text
    assert report.text == "".join(f"{r.label}: {r.value}\n" for r in report.rows)
    # The issue form is named; the report itself never goes into an address.
    assert report.new_issue_url == f"{REPOSITORY}/issues/new?template=bug_report.yml"


def test_hardware_details_only_when_asked():
    two = [
        GPU("NVIDIA GeForce RTX 9090", 24 * GB, None, True),
        GPU("NVIDIA A100", 80 * GB, 0, True),
    ]
    report = build_report(include_hardware=True, system=fake_system(gpus=two))
    values = rows(report)
    assert report.hardware
    assert values["Operating system version"] == "Ubuntu 24.04 LTS"
    assert values["Processor"] == "Contoso Quantum 9000X (8 cores, 16 threads)"
    assert values["Memory"] == "64.0 GB"
    assert values["GPU 1"] == "NVIDIA GeForce RTX 9090 (24.0 GB)"
    assert values["GPU 2"] == "NVIDIA A100 (80.0 GB)"
    assert values["Device"] == "CUDA, 2 GPUs"
    one = rows(build_report(include_hardware=True, system=fake_system()))
    assert one["GPU"] == "NVIDIA GeForce RTX 9090 (24.0 GB)"
    # Outside the app (logogram doctor --report) there is no model to name.
    assert "Model" not in one


def test_the_device_and_what_the_system_check_found():
    mac = fake_system(backend="cpu", gpus=[], mps_available=True, torch_cuda=None)
    values = rows(build_report(system=mac))
    assert values["Device"] == "CPU, with Apple's GPU (MPS) available when chosen"
    assert values["PyTorch"] == "2.5.1"
    issues = [
        Issue("warning", "Your NVIDIA GPU isn't being used", "detail", "fix"),
        Issue("warning", "Low disk space for model downloads", "detail"),
    ]
    values = rows(build_report(system=fake_system(backend="cpu", gpus=[], issues=issues)))
    assert values["Device"] == "CPU"
    assert values["System check"] == (
        "Your NVIDIA GPU isn't being used; Low disk space for model downloads"
    )


@pytest.mark.parametrize(
    "system, machine, release, expected",
    [
        ("Linux", "x86_64", "6.8.0-45-generic", "Linux (x86_64)"),
        ("Linux", "x86_64", "5.15.153.1-microsoft-standard-WSL2", "Linux under WSL (x86_64)"),
        ("Darwin", "arm64", "24.0.0", "macOS (arm64)"),
        ("Windows", "AMD64", "11", "Windows (AMD64)"),
    ],
)
def test_the_operating_system_is_named_by_family(monkeypatch, system, machine, release, expected):
    monkeypatch.setattr(support.platform, "system", lambda: system)
    monkeypatch.setattr(support.platform, "machine", lambda: machine)
    monkeypatch.setattr(support.platform, "release", lambda: release)
    assert rows(build_report(system=fake_system()))["Operating system"] == expected


def _info(**changes: Any) -> dict[str, Any]:
    info = {
        "id": "openai-community/gpt2",
        "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "dtype": "float32",
        "device": "cuda",
        "device_name": "NVIDIA GeForce RTX 9090",
        "process_weights": True,
    }
    return {**info, **changes}


def test_the_model_row(machine):
    def model(status: dict[str, Any]) -> str:
        return rows(build_report(model=status, system=fake_system()))["Model"]

    assert model({"state": "ready", "info": _info()}) == (
        "openai-community/gpt2, revision 607a30d783dfa663caf39e06633721c8d4cfcd7e, float32 on "
        "cuda, weights processed"
    )
    assert model({"state": "ready", "info": _info(process_weights=False)}).endswith(
        "weights not processed"
    )
    assert model({"state": "loading", "id": "EleutherAI/pythia-70m"}) == (
        "EleutherAI/pythia-70m (loading)"
    )
    assert model({"state": "error", "id": "meta-llama/Llama-3.2-1B", "error": "gated"}) == (
        "meta-llama/Llama-3.2-1B (failed to load)"
    )
    # The GPU's name is a hardware detail, even when it comes with the model.
    assert "RTX" not in model({"state": "ready", "info": _info()})
    # A model loaded from a folder doesn't give the folder away; a Hub id is the person's own
    # choice, and stays as it is.
    assert model({"state": "error", "id": str(machine / "models" / "mine")}) == (
        "<path> (failed to load)"
    )
    assert model({"state": "loading", "id": f"{USER}/gpt2-finetune"}).startswith(
        f"{USER}/gpt2-finetune"
    )


def test_the_last_error_is_scrubbed_of_paths_names_and_secrets(machine, tmp_path):
    project = tmp_path / "projects" / "ioi"
    project.mkdir(parents=True)
    token = "session-token-0123456789"
    error = (
        f"Can't read '{machine}/Secret Plans/prompts.jsonl' or {project}/datasets/ioi.jsonl. "
        f"{USER} on {HOST}.local ({HOST}) used http://{USER}:hunter2@proxy.example.org:3128 "
        f"with hf_{'x' * 34} and http://127.0.0.1:8765/?token={token}; "
        f"also C:\\Users\\{USER}\\Documents\\x.jsonl and {Path('/srv') / USER / 'cache'}."
    )
    report = build_report(
        model={"state": "none"},
        last_error=error,
        roots=[project],
        secrets=[token],
        system=fake_system(),
    )
    last = rows(report)["Last error"]
    for secret in (USER, HOST, "Ada Lovelace", "Secret Plans", "hunter2", "x" * 34, token):
        assert secret.lower() not in last.lower(), secret
    assert str(tmp_path) not in last
    assert "'<path>'" in last
    assert "<project>/datasets/ioi.jsonl" in last
    assert "<user> on <host> (<host>)" in last
    assert "/?token=<secret>; also" in last
    assert "http://<credentials>@proxy.example.org:3128" in last
    assert "C:\\Users" not in last and last.endswith("<path> and <path>.")


@pytest.mark.parametrize(
    "text, expected",
    [
        # Paths become placeholders, whole, wherever they are.
        ("No such file: '/data/My Project/a.jsonl'", "No such file: '<path>'"),
        ('Not found: "/data/x y/z"', 'Not found: "<path>"'),
        ("Permission denied: /data/lab/x.txt.", "Permission denied: <path>."),
        ("open /data/x and /data/y", "open <path> and <path>"),
        ("(/opt/models/gpt2/config.json)", "(<path>)"),
        ("path=/opt/x", "path=<path>"),
        ("C:\\Projects\\ioi\\datasets\\x.jsonl is missing", "<path> is missing"),
        ("D:/Projects/ioi/x.jsonl is missing", "<path> is missing"),
        ("\\\\fileserver\\share\\lab\\x.jsonl is missing", "<path> is missing"),
        ("~/projects/ioi/x.jsonl is missing", "<path> is missing"),
        # Secrets.
        ("auth Bearer abcdefghijklmnop", "auth Bearer <token>"),
        ("see https://a:b@example.org/x", "see https://<credentials>@example.org/x"),
        (
            "visit http://127.0.0.1:8765/?token=abc&x=1",
            "visit http://127.0.0.1:8765/?token=<secret>&x=1",
        ),
        ("write to someone@" + "university.edu", "write to <email>"),
        # What isn't a path stays.
        (
            "https://huggingface.co/openai-community/gpt2/resolve/main/config.json",
            "https://huggingface.co/openai-community/gpt2/resolve/main/config.json",
        ),
        ("datasets/ioi.jsonl and/or 3/4 and 50%/s", "datasets/ioi.jsonl and/or 3/4 and 50%/s"),
        ("'/' isn't a dataset name; a / b", "'/' isn't a dataset name; a / b"),
        ("model openai-community/gpt2 at main", "model openai-community/gpt2 at main"),
        ("~3 GB free", "~3 GB free"),
    ],
)
def test_scrub(text, expected):
    assert scrub(text, names=False) == expected


def test_scrub_names_the_project_and_the_hugging_face_cache(tmp_path, monkeypatch):
    from huggingface_hub import constants

    cache = tmp_path / "hf" / "hub"
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    project = tmp_path / "work" / "ioi"
    text = (
        f"{project}/experiments/run-1/spec.json, {project}-2/x, '{project}', "
        f"{cache}/models--openai-community--gpt2/blobs/abc"
    )
    assert scrub(text, [project], names=False) == (
        "<project>/experiments/run-1/spec.json, <path>, '<project>', "
        "<huggingface cache>/models--openai-community--gpt2/blobs/abc"
    )


def test_generic_and_short_names_are_not_scrubbed_as_words(monkeypatch):
    monkeypatch.setattr(support.getpass, "getuser", lambda: "root")
    monkeypatch.setattr(support.socket, "gethostname", lambda: "server")
    monkeypatch.setattr(support.platform, "node", lambda: "pc")
    text = "The Logogram server, run as root on a pc, stopped."
    assert scrub(text) == text


def test_long_errors_are_kept_on_one_line_and_shortened():
    error = "first line\n\n  second   line\t" + "x" * 5000
    last = rows(build_report(last_error=error, system=fake_system()))["Last error"]
    assert last.startswith("first line second line xxx")
    assert len(last) == support.ERROR_LIMIT and last.endswith("…")
    assert "Last error" not in rows(build_report(last_error=" \n", system=fake_system()))


# -- where the report is built -------------------------------------------------------------------

TOKEN = "report-test-token"
BASE = "http://127.0.0.1:8765"
WRITE = {"authorization": f"Bearer {TOKEN}", "origin": BASE}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    monkeypatch.setattr("logogram.system.system_report", lambda: fake_system())
    application = create_app(SecurityConfig(token=TOKEN, port=8765), serve_web=False)
    return TestClient(application, base_url=BASE)


def test_the_app_builds_the_report_with_the_loaded_model(client, tiny_backend, project):
    state = client.app.state.logogram  # type: ignore[attr-defined]
    client.post("/api/projects/open", json={"path": str(project.root)}, headers=WRITE)
    state.backend = tiny_backend
    state.model_status = {"state": "ready"}
    error = f"Couldn't read {project.root}/datasets/ioi.jsonl (token {TOKEN}xyz)"
    body = {"include_hardware": False, "last_error": error}
    out = client.post("/api/report", json=body, headers=WRITE)
    assert out.status_code == 200, out.text
    report = out.json()
    values = {r["label"]: r["value"] for r in report["rows"]}
    assert values["Model"] == "tiny-gpt2, revision test, float32 on cpu, weights processed"
    assert values["Last error"] == "Couldn't read <project>/datasets/ioi.jsonl (token <secret>xyz)"
    assert report["text"] == "".join(f"{r['label']}: {r['value']}\n" for r in report["rows"])
    assert report["new_issue_url"] == NEW_ISSUE_URL and report["hardware"] is False
    for detail in HARDWARE:
        assert detail not in report["text"]
    hardware = client.post("/api/report", json={"include_hardware": True}, headers=WRITE).json()
    assert hardware["hardware"] is True and "Processor: Contoso Quantum 9000X" in hardware["text"]
    assert "Last error" not in hardware["text"]


def test_the_report_needs_the_token_like_everything_else(client):
    assert client.post("/api/report", json={}).status_code == 401
    assert client.post("/api/report", json={}, headers={"origin": BASE}).status_code == 401
    foreign = {**WRITE, "origin": "http://evil.example"}
    assert client.post("/api/report", json={}, headers=foreign).status_code == 403
    too_long = {"last_error": "x" * 20_001}
    assert client.post("/api/report", json=too_long, headers=WRITE).status_code == 422


def test_doctor_prints_the_same_report(monkeypatch):
    monkeypatch.setattr("logogram.system.system_report", lambda: fake_system())
    result = CliRunner().invoke(cli, ["doctor", "--report"])
    assert result.exit_code == 0, result.output
    assert result.stdout == build_report(system=fake_system()).text
    assert "Nothing was sent" in result.stderr and NEW_ISSUE_URL in result.stderr
    for detail in HARDWARE:
        assert detail not in result.stdout
    result = CliRunner().invoke(cli, ["doctor", "--report", "--hardware"])
    assert result.exit_code == 0, result.output
    assert result.stdout == build_report(include_hardware=True, system=fake_system()).text
    assert "Processor: Contoso Quantum 9000X (8 cores, 16 threads)" in result.stdout
    result = CliRunner().invoke(cli, ["doctor", "--hardware"])
    assert result.exit_code == 1 and "use it with --report" in result.output


def test_the_repository_is_the_one_pyproject_names():
    urls = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["urls"]
    assert urls["Source"] == REPOSITORY == urls["Homepage"]
    assert urls["Issues"] == f"{REPOSITORY}/issues"
    assert f"{urls['Issues']}/new?template=bug_report.yml" == NEW_ISSUE_URL
    assert (ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml").is_file()
