"""The reference pages in docs/reference/ are written from the code (scripts/build_reference.py):
these checks fail when the committed pages are out of date, and say how to bring them up to date.
They also check that the pages cover everything: every spec model and field, every command and
option, and every public name of the Python API."""

from __future__ import annotations

import importlib.util
import inspect
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMMAND = "uv run python scripts/build_reference.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "build_reference", ROOT / "scripts" / "build_reference.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_reference"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reference():
    return _load()


@pytest.fixture(scope="module")
def pages(reference):
    return reference.pages()


def test_the_committed_reference_pages_are_up_to_date(reference, pages):
    assert reference.COMMAND == COMMAND
    stale = reference.stale(pages)
    assert not stale, (
        f"{', '.join(stale)} {'is' if len(stale) == 1 else 'are'} out of date: the code changed "
        f"since {'it was' if len(stale) == 1 else 'they were'} written. Run `{COMMAND}` and commit "
        "the result."
    )


def test_every_spec_field_is_described(pages):
    from pydantic import BaseModel

    import logogram.spec as spec

    page = pages["docs/reference/spec.md"]
    models = [
        cls
        for cls in vars(spec).values()
        if inspect.isclass(cls) and issubclass(cls, BaseModel) and cls.__module__ == spec.__name__
    ]
    # Private helpers (the base class, a chat template read on its own) aren't fields of a spec.
    reachable = [cls for cls in models if not cls.__name__.startswith("_")]
    for cls in reachable:
        for name, info in cls.model_fields.items():
            assert f"| `{name}`" in page, f"{cls.__name__}.{name} is missing from the spec page"
            if name != "kind":
                assert info.description, f"{cls.__name__}.{name} has no description"
    # Each kind of experiment, scope, metric, position, ... has a section of its own. (A site's
    # or an edge sender's kind is a choice among several, not the name of one model.)
    for cls in reachable:
        kind = cls.model_fields.get("kind")
        if kind is not None and len(kind.annotation.__args__) == 1:
            (value,) = kind.annotation.__args__
            assert re.search(rf"^### .*: {value}$", page, re.M), f"no section for {value}"


def test_every_command_and_option_is_listed(pages):
    import typer.main

    from logogram.cli import app

    page = pages["docs/reference/cli.md"]

    def walk(group, path):
        for name, command in getattr(group, "commands", {}).items():
            yield f"{path} {name}", command
            yield from walk(command, f"{path} {name}")

    group = typer.main.get_command(app)
    for path, command in [("logogram", group), *walk(group, "logogram")]:
        assert f"## {path}\n" in page, f"{path} is missing from the command line page"
        for param in command.params:
            if type(param).__name__.endswith("Option"):
                for option in param.opts:
                    assert f"`{option}`" in page, f"{path} {option} is missing"


def test_every_public_name_of_the_python_api_is_documented(pages):
    import logogram
    from logogram import api

    page = pages["docs/reference/python-api.md"]
    assert sorted(logogram._API) == sorted(api.__all__), (
        "logogram._API and logogram.api.__all__ must name the same functions"
    )
    for name in api.__all__:
        assert f"### lg.{name}\n" in page, f"lg.{name} is missing from the Python API page"
    assert "Not described yet." not in page, "describe every public function in its docstring"


def test_anchors_are_the_ones_github_gives_headings(reference):
    anchors = reference.Anchors()
    assert anchors.add("lg.run") == "lgrun"
    assert anchors.add("lg.Run") == "lgrun-1"
    assert anchors.add("Experiment: activation_patching") == "experiment-activation_patching"
    assert reference.slug("Chat template: user_turn") == "chat-template-user_turn"


def test_links_within_each_page_lead_to_its_headings(reference, pages):
    for path, text in pages.items():
        anchors = reference.Anchors()
        headings = {anchors.add(m) for m in re.findall(r"^#{1,6} (.+)$", text, re.M)}
        for target in re.findall(r"\]\(#([^)]+)\)", text):
            assert target in headings, f"{path} links to #{target}, which no heading has"


def test_check_says_how_to_bring_pages_up_to_date(reference, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(reference, "ROOT", tmp_path)
    assert reference.main(["--check"]) == 1
    assert COMMAND in capsys.readouterr().out
    assert reference.main([]) == 0
    assert reference.main(["--check"]) == 0
    written = (tmp_path / "docs/reference/spec.md").read_bytes()
    assert b"\r\n" not in written
