"""The bundled examples: valid, pinned specs with their datasets, listed and opened by the gallery.

Each example's dataset was generated with Logogram's own task generators for GPT-2's tokenizer.
Where that tokenizer is in the Hugging Face cache already, the test makes the dataset again and
checks that it is the same file; it never downloads anything.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from logogram.datasets import file_sha256, load_dataset, to_jsonl
from logogram.gallery import DEFAULT_EXAMPLE, EXAMPLES, example_spec, get_example, list_examples
from logogram.project import PROJECT_FILE, ProjectError, example_source, open_example
from logogram.server.app import create_app
from logogram.server.security import SecurityConfig
from logogram.spec import SPEC_VERSION, Spec, describe_experiment

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "src" / "logogram" / "examples"
IDS = [e.id for e in EXAMPLES]
NEW = [e.id for e in EXAMPLES if e.id != DEFAULT_EXAMPLE]
COMMIT = re.compile(r"^[0-9a-f]{40}$")


def test_the_gallery_lists_every_bundled_example():
    folders = sorted(p.name for p in EXAMPLES_DIR.iterdir() if p.is_dir())
    assert sorted(IDS) == folders
    assert len({e.folder for e in EXAMPLES}) == len(EXAMPLES)  # one folder per example
    assert DEFAULT_EXAMPLE in IDS and get_example("ioi-gpt2").folder == "ioi-example"
    with pytest.raises(ProjectError, match="no example called"):
        get_example("../ioi-gpt2")


@pytest.mark.parametrize("example_id", IDS)
def test_each_example_is_a_pinned_spec_of_this_version_with_its_dataset(example_id: str):
    source = example_source(example_id)
    experiment, spec = example_spec(example_id)
    path = source / "experiments" / experiment / "spec.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Version 2 states every choice: nothing is filled in from version 1's values.
    assert raw["logogram_spec"] == SPEC_VERSION and spec.upgraded_fields == []
    assert Spec.model_validate(raw) == spec
    assert spec.model.id == "openai-community/gpt2" and COMMIT.match(spec.model.revision or "")
    assert spec.model.dtype == "float32" and spec.dataset.limit is None
    # The dataset is bundled, pinned by its hash, and parses.
    assert spec.dataset.path.startswith("datasets/") and spec.dataset.sha256
    dataset = source / spec.dataset.path
    assert file_sha256(dataset) == spec.dataset.sha256
    records = load_dataset(dataset)
    assert len(records) >= 16
    assert len({r.clean for r in records}) == len(records)
    # Nothing in the example was made on a machine: no runs, no snapshots, no caches.
    files = sorted(p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file())
    assert files == sorted(
        [".gitignore", PROJECT_FILE, spec.dataset.path, f"experiments/{experiment}/spec.json"]
    )
    meta = json.loads((source / PROJECT_FILE).read_text(encoding="utf-8"))
    assert meta["logogram_project"] == 1 and meta["name"]


def test_every_example_checks_the_same_model():
    """The weekly check against real weights loads one model for every example (and caches it
    by the revision they pin)."""
    models = {example_spec(e)[1].model.model_dump_json() for e in IDS}
    assert len(models) == 1


@pytest.mark.parametrize("example_id", NEW)
def test_new_examples_state_the_question_the_finding_and_the_result(example_id: str):
    example = get_example(example_id)
    _, spec = example_spec(example_id)
    notes = spec.notes
    # The question, the published finding with its citation, and what Logogram measured.
    author = example.source.split()[0]
    assert author in notes and re.search(r"\b20\d\d\b", notes)
    assert "GPT-2 small" in notes and len(notes) > 400
    # What was measured names the weights it was measured on.
    assert f"revision {(spec.model.revision or '')[:7]}" in notes
    assert example.question.endswith("?") and example.expect
    assert spec.name.endswith("?")


@pytest.mark.parametrize("example_id", IDS)
def test_examples_were_generated_by_the_task_generators(example_id: str):
    """Made again for GPT-2's tokenizer, each dataset is byte for byte the bundled file. Skipped
    when the tokenizer isn't in the Hugging Face cache: the test never downloads it."""
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "openai-community/gpt2",
            revision=example_spec(example_id)[1].model.revision,
            local_files_only=True,
        )
    except OSError:
        pytest.skip("GPT-2's tokenizer isn't in the Hugging Face cache")
    from logogram.tasks import generate_task

    def ids(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    example = get_example(example_id)
    _, spec = example_spec(example_id)
    records = generate_task(
        example.task,
        example.n,
        example.seed,
        dict(example.options),
        single_token=lambda text: len(ids(text)) == 1,
        token_count=lambda text: len(ids(text)),
    )
    bundled = (example_source(example_id) / spec.dataset.path).read_text(encoding="utf-8")
    assert to_jsonl(records) == bundled


def test_opening_an_example_copies_it_once_into_its_own_folder(tmp_path: Path):
    for example in EXAMPLES:
        assert not any(e["copied"] for e in list_examples(tmp_path) if e["id"] == example.id)
        project = open_example(tmp_path, example=example.id)
        assert project.root == (tmp_path / example.folder).resolve()
        experiment, spec = example_spec(example.id)
        copied = Spec.from_path(project.root / "experiments" / experiment / "spec.json")
        assert copied == spec
        assert [d["path"] for d in project.list_datasets()] == [spec.dataset.path]
        # The copy is the user's: opening the example again opens it as it is.
        mine = project.root / "experiments" / experiment / "spec.json"
        mine.write_text(
            mine.read_text(encoding="utf-8").replace(spec.name, "Mine"), encoding="utf-8"
        )
        again = open_example(tmp_path, example=example.id)
        assert again.root == project.root
        assert Spec.from_path(mine).name == "Mine"
    listed = list_examples(tmp_path)
    assert [e["id"] for e in listed] == IDS
    assert all(e["copied"] for e in listed)
    # Where each copy is, as the project names itself once open.
    assert [e["path"] for e in listed] == [
        str(open_example(tmp_path, example=e.id).root) for e in EXAMPLES
    ]


def test_an_example_never_replaces_a_folder_that_isnt_its_copy(tmp_path: Path):
    example = get_example(NEW[0])
    (tmp_path / example.folder).mkdir()
    (tmp_path / example.folder / "mine.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ProjectError, match="isn't a Logogram project"):
        open_example(tmp_path, example=example.id)
    assert (tmp_path / example.folder / "mine.txt").read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / example.folder / PROJECT_FILE).exists()


def test_the_gallery_describes_each_example(tmp_path: Path):
    for entry in list_examples(tmp_path):
        example = get_example(entry["id"])
        _, spec = example_spec(example.id)
        assert entry["question"] == example.question and entry["expect"] == example.expect
        assert entry["source"] == example.source and entry["copied"] is False
        assert Spec.model_validate(entry["spec"]) == spec
        assert entry["n_prompts"] == len(
            load_dataset(example_source(example.id) / spec.dataset.path)
        )
        assert describe_experiment(spec)


TOKEN = "test-token"
BASE = "http://127.0.0.1:8765"
HEADERS = {"authorization": f"Bearer {TOKEN}", "origin": BASE}


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr("platformdirs.user_config_path", lambda *a, **k: tmp_path / "config")
    monkeypatch.setattr("logogram.project.config_dir", lambda: tmp_path / "config")
    monkeypatch.setattr("logogram.server.app.default_projects_parent", lambda: tmp_path / "work")
    app = create_app(SecurityConfig(token=TOKEN, port=8765), serve_web=False)
    return TestClient(app, base_url=BASE)


def test_the_server_lists_and_opens_each_example(client: TestClient, tmp_path: Path):
    listed = client.get("/api/examples", headers=HEADERS)
    assert listed.status_code == 200
    assert [e["id"] for e in listed.json()] == IDS
    for example in EXAMPLES:
        opened = client.post("/api/projects/example", json={"example": example.id}, headers=HEADERS)
        assert opened.status_code == 200, opened.text
        info = opened.json()
        assert Path(info["path"]) == (tmp_path / "work" / example.folder).resolve()
        name = json.loads((example_source(example.id) / PROJECT_FILE).read_text(encoding="utf-8"))[
            "name"
        ]
        assert info["name"] == name
        assert client.get("/api/project", headers=HEADERS).json()["path"] == info["path"]
    assert all(e["copied"] for e in client.get("/api/examples", headers=HEADERS).json())
    # Without a choice, the example that was there before the gallery.
    default = client.post("/api/projects/example", headers=HEADERS)
    assert default.json()["path"] == str((tmp_path / "work" / "ioi-example").resolve())
    unknown = client.post("/api/projects/example", json={"example": "../x"}, headers=HEADERS)
    assert unknown.status_code == 400 and "no example called" in unknown.json()["error"]
