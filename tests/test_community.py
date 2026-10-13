"""The repository's community files (issue forms, SECURITY.md, CONTRIBUTING.md, CITATION.cff)
and the README's guides: they are well formed, link where they should, and say only what the code
does."""

from __future__ import annotations

import re
import shlex
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from logogram.cli import app as cli
from logogram.support import NEW_ISSUE_URL, REPOSITORY

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / ".github" / "ISSUE_TEMPLATE"
README = (ROOT / "README.md").read_text(encoding="utf-8")


def load(path: Path) -> dict:  # type: ignore[type-arg]
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), path
    return data


def slug(heading: str) -> str:
    """The anchor GitHub gives a heading."""
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def readme_anchors() -> set[str]:
    return {slug(h) for h in re.findall(r"^#{1,6} (.+)$", README, re.M)}


def section(title: str) -> str:
    """A level-2 section of the README, its lines joined into one."""
    match = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## )", README, re.M | re.S)
    assert match, title
    return " ".join(match[1].split())


# -- CITATION.cff --------------------------------------------------------------------------------

# The keys CFF 1.2.0 allows at the top level.
CFF_KEYS = {
    "abstract",
    "authors",
    "cff-version",
    "commit",
    "contact",
    "date-released",
    "doi",
    "identifiers",
    "keywords",
    "license",
    "license-url",
    "message",
    "preferred-citation",
    "references",
    "repository",
    "repository-artifact",
    "repository-code",
    "title",
    "type",
    "url",
    "version",
}


def test_the_citation_file_names_the_project_and_no_person():
    cff = load(ROOT / "CITATION.cff")
    assert cff["cff-version"] == "1.2.0"
    assert {"cff-version", "message", "title", "authors"} <= set(cff) <= CFF_KEYS
    assert cff["title"] == "Logogram" and cff["type"] == "software"
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    # The authors are an entity, as in pyproject.toml: no person's name or address.
    assert cff["authors"] == project["authors"] == [{"name": "The Logogram contributors"}]
    assert cff["license"] == project["license"] == "MIT"
    assert cff["repository-code"] == cff["url"] == REPOSITORY
    assert all(isinstance(k, str) and k for k in cff["keywords"])


# -- issue forms ---------------------------------------------------------------------------------

FORM_KEYS = {"name", "description", "title", "labels", "assignees", "projects", "type", "body"}
ELEMENTS = {"markdown", "textarea", "input", "dropdown", "checkboxes"}


@pytest.mark.parametrize("name", ["bug_report.yml", "feature_request.yml"])
def test_issue_forms_are_well_formed(name):
    form = load(TEMPLATES / name)
    assert set(form) <= FORM_KEYS and form["name"] and form["description"]
    ids = []
    for element in form["body"]:
        assert element["type"] in ELEMENTS, element
        attributes = element["attributes"]
        if element["type"] == "markdown":
            assert attributes["value"].strip()
            continue
        ids.append(element["id"])
        assert re.fullmatch(r"[A-Za-z0-9_-]+", element["id"])
        assert attributes["label"].strip()
        if element["type"] == "dropdown":
            assert attributes["options"] and len(set(attributes["options"])) == len(
                attributes["options"]
            )
    assert ids and len(ids) == len(set(ids))


def test_the_bug_report_asks_for_the_problem_report():
    form = load(TEMPLATES / "bug_report.yml")
    assert NEW_ISSUE_URL.endswith("?template=bug_report.yml")
    report = next(e for e in form["body"] if e.get("id") == "report")
    assert report["type"] == "textarea" and report["validations"]["required"] is True
    # Pasted as it is: GitHub shows it as a block of text.
    assert report["attributes"]["render"] == "text"
    words = " ".join(report["attributes"]["description"].split())
    assert "Report a problem" in words and "`logogram doctor --report`" in words
    intro = form["body"][0]["attributes"]["value"]
    assert f"{REPOSITORY}#troubleshooting" in intro
    assert f"{REPOSITORY}/security/advisories/new" in intro


def test_the_issue_chooser_offers_private_security_reports_and_the_guides():
    config = load(TEMPLATES / "config.yml")
    assert config["blank_issues_enabled"] is False
    links = {link["url"]: link for link in config["contact_links"]}
    assert f"{REPOSITORY}/security/advisories/new" in links
    for url, link in links.items():
        assert url.startswith(f"{REPOSITORY}/") or url.startswith(f"{REPOSITORY}#")
        assert link["name"].strip() and link["about"].strip()
        if "#" in url:
            assert url.split("#", 1)[1] in readme_anchors(), url


# -- SECURITY.md and CONTRIBUTING.md -------------------------------------------------------------


def test_security_reports_go_through_private_vulnerability_reporting():
    text = " ".join((ROOT / "SECURITY.md").read_text(encoding="utf-8").split())
    assert f"<{REPOSITORY}/security/advisories/new>" in text
    for subject in ("session token", "Host, Origin", "symlink", "safetensors", "unpickled"):
        assert subject in text, subject


def test_contributing_lists_the_checks_ci_runs():
    text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    pyright = re.search(r"npx --yes (pyright@[\d.]+) --pythonpath \.venv/bin/python", ci)
    assert pyright and f"npx --yes {pyright[1]} --pythonpath .venv/bin/python" in text
    for check in (
        "uv run pytest",
        "ruff check src tests scripts",
        "ruff format --check src tests scripts",
        "python3 scripts/check_privacy.py",
        "npx tsc --noEmit -p .",
        "npm test",
        "npm run build",
        "npm run test:browser",
        "LOGOGRAM_COLOR_SCHEME=dark npm run test:browser",
    ):
        assert check in text, check
    assert "[AGENTS.md](AGENTS.md)" in text and "[SECURITY.md](SECURITY.md)" in text


# -- the README's commands and messages ---------------------------------------------------------


def shell_commands(text: str) -> list[list[str]]:
    """Every `logogram` command in a document's shell blocks, as its arguments."""
    commands = []
    for block in re.findall(r"```(?:bash|sh|shell)\n(.*?)```", text, re.S):
        for line in block.splitlines():
            words = shlex.split(line, comments=True)
            while words and re.fullmatch(r"[A-Z_][A-Z0-9_]*=\S*", words[0]):
                words.pop(0)  # VARIABLE=value before the command
            if words[:2] == ["uv", "run"]:
                words = words[2:]
            if words[:1] == ["logogram"]:
                commands.append(words[1:])
    return commands


@pytest.mark.parametrize("document", ["README.md", "CONTRIBUTING.md", "SECURITY.md"])
def test_every_logogram_command_in_the_docs_exists(document):
    """Each command and option the documents give is one Logogram has (checked with --help, so
    nothing runs)."""
    commands = shell_commands((ROOT / document).read_text(encoding="utf-8"))
    if document == "README.md":
        assert ["--no-browser", "--forwarded-port", "9000"] in commands
        assert ["--no-browser"] in commands and ["doctor"] in commands
    for args in commands:
        result = CliRunner().invoke(cli, [*args, "--help"])
        assert result.exit_code == 0, f"logogram {' '.join(args)}: {result.output}"


def test_the_remote_guide_keeps_the_token_off_command_lines():
    guide = section("Running on a remote GPU")
    assert "token=" not in " ".join(" ".join(c) for c in shell_commands(README))
    assert "LOGOGRAM_TOKEN" not in guide
    for step in ("--no-browser", "ssh -N -L 8765:127.0.0.1:8765", "--forwarded-port 9000"):
        assert step in guide, step
    # Where things live, as the code puts them.
    for place in ("~/Documents/Logogram", "~/.cache/huggingface/hub", "~/.config/logogram"):
        assert place in guide, place


def _source(path: Path) -> str:
    """Source text with string literals split across lines joined, and whitespace collapsed."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        text = re.sub(r"\"\s*\n\s*(?:rf|fr|f|r)?\"", "", text)
    return " ".join(text.split())


# Messages the troubleshooting section quotes, as the code writes them (variable parts left out).
MESSAGES = [
    "Logogram needs its session token.",
    "This page's session has ended. Open the link printed in the terminal where Logogram is "
    "running.",
    "Can't reach the Logogram server. Check that `logogram` is still running in your terminal.",
    "Logogram didn't answer this request: it was sent to",
    ", and this Logogram listens on port",
    "is gated. Accept its license on huggingface.co, then log in on this machine with "
    "`hf auth login` and try again.",
    "No model called",
    "was found on Hugging Face.",
    "Couldn't reach Hugging Face, and",
    "isn't in your local cache.",
    "This model only publishes pickle weights (.bin).",
    "models, so Logogram can't either.",
    "Processing the weights of",
    "changed its predictions",
    "'s own forward pass gives values that aren't finite numbers (it overflows), so its "
    "predictions can't be checked.",
    "Load it in bfloat16 or float32, which have a wider range than float16.",
    "CUDA was chosen, but PyTorch can't see a CUDA GPU here.",
    "while its weights are processed",
    "Choose a smaller model, a 16-bit dtype or another device.",
    "The run ran out of",
    "rows. Lower execution.batch_size in the spec",
    "is several tokens (",
    "The logit difference reads one token",
    ", so positions can't be aligned.",
    "doesn't cover any token.",
    "This model's tokenizer has no beginning-of-sequence token. Set tokenization.prepend_bos to "
    "false in the spec.",
    "Apple's MPS was chosen, but it isn't available here. Choose the device auto or cpu "
    "(model.device in a spec).",
    "Python is running under Rosetta",
    "Apple GPU (MPS) unavailable",
    "a path inside the project, written with forward slashes and without '..', such as "
    "datasets/ioi.jsonl",
    "Your NVIDIA GPU isn't being used",
    "The server stopped before this run finished. Rerun its saved spec.",
    "This run failed",
    "Edit and rerun",
]


def test_troubleshooting_quotes_the_messages_the_code_shows():
    guide = section("Troubleshooting")
    sources = [
        _source(p)
        for folder in (ROOT / "src" / "logogram", ROOT / "web" / "src")
        for p in sorted(folder.rglob("*"))
        if p.suffix in (".py", ".ts", ".tsx") and "web_dist" not in p.parts
    ]
    for message in MESSAGES:
        assert message in guide, f"not in the README: {message}"
        assert any(message in s for s in sources), f"not in the code: {message}"
