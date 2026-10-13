"""The documentation in docs/ and the README: every link inside the repository leads to a file
and, with an anchor, to one of its headings; links leave only for the places the docs need (the
repository, uv's documentation, Colab), never through a tracking service; and every spec the docs
show in JSON, whole or in part, is one Logogram accepts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted((ROOT / "docs").rglob("*.md"))
PAGES = [*DOCS, ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "notebooks" / "README.md"]
# Where the docs' own links may lead outside the repository.
ALLOWED = (
    "https://github.com/Jeevash23/logogram",
    "https://docs.astral.sh/uv/",
    "https://colab.research.google.com/github/Jeevash23/logogram/",
)
LINK = re.compile(r"\]\(([^)\s]+)\)")
EXAMPLE = ROOT / "src/logogram/examples/ioi-gpt2/experiments/ioi-head-patching/spec.json"


def _slug(heading: str) -> str:
    """The anchor GitHub gives a heading: lower case, punctuation dropped, spaces as hyphens."""
    text = re.sub(r"[`*_]", lambda m: "_" if m.group(0) == "_" else "", heading.strip())
    return re.sub(r"[^\w\- ]", "", text.lower()).replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
    seen: dict[str, int] = {}
    anchors = set()
    for heading in re.findall(r"^#{1,6} (.+)$", text, re.M):
        base = _slug(heading)
        n = seen.get(base, 0)
        seen[base] = n + 1
        anchors.add(base if n == 0 else f"{base}-{n}")
    return anchors


def _links(path: Path) -> list[str]:
    text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
    return LINK.findall(text)


def test_there_are_docs():
    names = {p.relative_to(ROOT / "docs").as_posix() for p in DOCS}
    for page in (
        "README.md",
        "getting-started.md",
        "concepts/activation-patching.md",
        "concepts/statistics.md",
        "guides/replicate-ioi.md",
        "reference/spec.md",
        "reference/cli.md",
        "reference/python-api.md",
    ):
        assert page in names


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_links_inside_the_repository_lead_somewhere(page: Path):
    for link in _links(page):
        if link.startswith(("http://", "https://", "mailto:")):
            continue
        target, _, anchor = link.partition("#")
        file = (page.parent / target).resolve() if target else page
        assert file.exists(), f"{page.relative_to(ROOT)} links to {link}, which doesn't exist"
        assert ROOT in file.parents or file == ROOT, f"{link} leaves the repository"
        if anchor and file.suffix == ".md":
            assert anchor in _anchors(file), (
                f"{page.relative_to(ROOT)} links to {link}, but {file.relative_to(ROOT)} has no "
                f"heading with that anchor"
            )


@pytest.mark.parametrize("page", DOCS, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_the_docs_link_out_only_where_they_must(page: Path):
    for link in _links(page):
        if link.startswith(("http://", "https://")):
            assert link.startswith(ALLOWED), f"{page.relative_to(ROOT)} links to {link}"


def _json_blocks(page: Path) -> list[str]:
    return re.findall(r"```json\n(.*?)```", page.read_text(encoding="utf-8"), flags=re.S)


@pytest.mark.parametrize(
    "page", [*DOCS, ROOT / "README.md"], ids=lambda p: p.relative_to(ROOT).as_posix()
)
def test_the_specs_the_docs_show_are_valid(page: Path):
    from logogram.spec import Spec

    example = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    for block in _json_blocks(page):
        if "…" in block or "..." in block:
            continue  # elided, to be read rather than run
        text = block.strip()
        data = json.loads(text if text.startswith("{") else "{" + text + "}")
        if "logogram_spec" in data:
            Spec.model_validate(data)
            continue
        if not set(data) <= set(Spec.model_fields):
            continue  # a prompt pair, or another kind of JSON
        merged = {**example, **data}
        if data.get("experiment", {}).get("kind") == "direct_logit_attribution":
            merged.setdefault("scope", {"kind": "heads", "position": {"kind": "last"}})
        uses_sae = (
            "sae_feature" in json.dumps(data) or data.get("scope", {}).get("kind") == "features"
        )
        if uses_sae and merged.get("sae") is None:
            merged["sae"] = {
                "repo": "an/sae",
                "path": "",
                "revision": None,
                "source": "hub",
                "sha256": None,
            }
        Spec.model_validate(merged)
