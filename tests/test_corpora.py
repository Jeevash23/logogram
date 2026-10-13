"""Corpora: plain texts in a project, written exactly and read back checked."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from logogram.corpora import (
    CorpusError,
    CorpusSource,
    TextRecord,
    delete_corpus,
    list_corpora,
    parse_corpus,
    read_corpus,
    rename_corpus,
    write_corpus,
)
from logogram.project import Project

SOURCE = CorpusSource(kind="texts", detail="written in a test")


@pytest.fixture
def empty_project(tmp_path: Path) -> Project:
    return Project.create(tmp_path, "Corpora")


def test_texts_are_written_and_read_back_with_their_labels(empty_project: Project) -> None:
    records = [
        TextRecord(text="The sky is blue.", label=True),
        TextRecord(text="The sky is green.", label=False, meta={"topic": "sky"}),
        TextRecord(text="No label here."),
    ]
    written = write_corpus(empty_project, "facts", records, SOURCE)
    read = read_corpus(empty_project, "facts", written.sha256)
    assert [r.text for r in read.records] == [r.text for r in records]
    assert read.labels() == ["true", "false"]
    assert read.meta is not None and read.meta.n == 3 and read.meta.source.kind == "texts"
    assert read.path == "corpora/facts.jsonl"


def test_a_changed_corpus_is_refused_when_its_hash_is_pinned(empty_project: Project) -> None:
    written = write_corpus(empty_project, "c", [TextRecord(text="one")], SOURCE)
    write_corpus(empty_project, "c", [TextRecord(text="two")], SOURCE, replace=True)
    with pytest.raises(CorpusError, match="has changed"):
        read_corpus(empty_project, "c", written.sha256)


def test_lines_that_aren_t_texts_are_refused_with_their_line() -> None:
    with pytest.raises(CorpusError, match="line 2: missing 'text'"):
        parse_corpus('{"text": "fine"}\n{"label": "x"}\n')
    with pytest.raises(CorpusError, match="line 1: not valid JSON"):
        parse_corpus("not json\n")
    with pytest.raises(CorpusError, match="colour isn't a field of a text"):
        parse_corpus('{"text": "a", "colour": "red"}\n')
    with pytest.raises(CorpusError, match="has no texts"):
        parse_corpus("\n\n")


def test_names_are_checked_and_never_replaced_without_asking(empty_project: Project) -> None:
    for name in ("", "../x", ".hidden", "a/b"):
        with pytest.raises(CorpusError):
            write_corpus(empty_project, name, [TextRecord(text="x")], SOURCE)
    write_corpus(empty_project, "c", [TextRecord(text="x")], SOURCE)
    with pytest.raises(CorpusError, match="already exists"):
        write_corpus(empty_project, "c", [TextRecord(text="y")], SOURCE)


def test_list_rename_and_delete(empty_project: Project) -> None:
    write_corpus(empty_project, "b", [TextRecord(text="x", label="a")], SOURCE)
    write_corpus(empty_project, "a", [TextRecord(text="y")], SOURCE)
    (empty_project.root / "corpora" / "broken.jsonl").write_text("{oops\n", encoding="utf-8")
    listed = {c["name"]: c for c in list_corpora(empty_project)}
    assert listed["b"]["n"] == 1 and listed["b"]["labels"] == ["a"]
    assert "not valid JSON" in listed["broken"]["error"]
    rename_corpus(empty_project, "b", "renamed")
    delete_corpus(empty_project, "a")
    assert [c["name"] for c in list_corpora(empty_project)] == ["broken", "renamed"]


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_a_corpora_folder_that_links_outside_is_never_written(
    empty_project: Project, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (empty_project.root / "corpora").symlink_to(outside, target_is_directory=True)
    with pytest.raises(Exception, match="outside the project"):
        write_corpus(empty_project, "c", [TextRecord(text="x")], SOURCE)
    assert list(outside.iterdir()) == []


def test_the_description_names_no_folder(empty_project: Project) -> None:
    source = CorpusSource(kind="file", detail="reviews.txt, one text per line")
    write_corpus(empty_project, "c", [TextRecord(text="x")], source)
    data = json.loads((empty_project.root / "corpora" / "c.json").read_text(encoding="utf-8"))
    assert data["source"]["detail"] == "reviews.txt, one text per line"
    assert str(empty_project.root) not in json.dumps(data)
