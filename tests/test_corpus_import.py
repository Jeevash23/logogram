"""Corpora imported from files, from a Hugging Face dataset (its download mocked, or from the
cache offline) and from a project's prompts; and a corpus read a page at a time."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from logogram.corpora import CorpusError, corpus_page, read_corpus
from logogram.corpus_import import (
    format_of,
    import_file,
    import_hub,
    import_prompts,
    import_text,
    write_texts,
)
from logogram.project import Project


@pytest.fixture
def empty_project(tmp_path: Path) -> Project:
    return Project.create(tmp_path, "Imports")


def texts_of(project: Project, name: str) -> list[str]:
    return read_corpus(project, name).texts


def test_a_text_file_is_split_into_lines_or_paragraphs(empty_project: Project) -> None:
    content = "  The first text, spaces kept.\r\nThe second.\r\n\r\n\r\nA third\nover two lines.\n"
    lines = import_text(empty_project, "lines", content, filename="notes.txt", fmt="lines")
    assert texts_of(empty_project, "lines") == [
        "  The first text, spaces kept.",
        "The second.",
        "A third",
        "over two lines.",
    ]
    assert lines.skipped == 0 and lines.corpus.meta is not None
    source = lines.corpus.meta.source
    assert source.kind == "file" and source.path == "notes.txt" and source.sha256
    assert "one text per line" in source.detail
    import_text(empty_project, "paragraphs", content, filename="notes.txt", fmt="paragraphs")
    assert texts_of(empty_project, "paragraphs") == [
        "  The first text, spaces kept.\nThe second.",
        "A third\nover two lines.",
    ]


def test_json_lines_take_a_text_field_and_a_label_field(empty_project: Project) -> None:
    rows = [
        {"sentence": "Paris is in France.", "truth": True},
        {"sentence": "", "truth": False},  # no text: left out and counted
        {"sentence": "Paris is in Spain.", "truth": False},
        {"sentence": None, "truth": None},
    ]
    content = "\n".join(json.dumps(r) for r in rows) + "\n"
    done = import_text(
        empty_project,
        "facts",
        content,
        filename="facts.jsonl",
        fmt="jsonl",
        text_field="sentence",
        label_field="truth",
    )
    corpus = read_corpus(empty_project, "facts")
    assert corpus.texts == ["Paris is in France.", "Paris is in Spain."]
    assert [r.label for r in corpus.records] == [True, False] and corpus.labels() == [
        "true",
        "false",
    ]
    assert (
        done.skipped == 2 and "2 rows without text left out" in done.to_dict()["source"]["detail"]
    )

    with pytest.raises(
        CorpusError, match=r"no column 'text'\. Its columns are 'sentence', 'truth'"
    ):
        import_text(empty_project, "x", content, filename="facts.jsonl", fmt="jsonl")
    with pytest.raises(CorpusError, match="line 2: the text is int, not a string"):
        import_text(empty_project, "x", '{"text": "a"}\n{"text": 3}\n', filename="f", fmt="jsonl")
    with pytest.raises(CorpusError, match="line 1: not valid JSON"):
        import_text(empty_project, "x", "{oops\n", filename="f.jsonl", fmt="jsonl")
    with pytest.raises(CorpusError, match="the label is list"):
        import_text(
            empty_project,
            "x",
            '{"text": "a", "l": [1]}\n',
            filename="f",
            fmt="jsonl",
            label_field="l",
        )


def test_a_json_array_and_json_lines_named_json_both_read(empty_project: Project) -> None:
    array = json.dumps([{"text": "one"}, {"text": "two"}])
    import_text(empty_project, "array", array, filename="rows.json", fmt="json")
    assert texts_of(empty_project, "array") == ["one", "two"]
    import_text(
        empty_project, "lines", '{"text": "one"}\n{"text": "two"}\n', filename="r.json", fmt="json"
    )
    assert texts_of(empty_project, "lines") == ["one", "two"]
    with pytest.raises(CorpusError, match="its list is under 'rows'"):
        import_text(empty_project, "x", json.dumps({"rows": []}), filename="r.json", fmt="json")


def test_csv_and_tsv_read_columns_by_their_header(empty_project: Project) -> None:
    csv_text = 'review,stars\n"Good, really good",5\n"Two\nlines",1\n,3\n'
    done = import_text(
        empty_project,
        "reviews",
        csv_text,
        filename="reviews.csv",
        fmt="csv",
        text_field="review",
        label_field="stars",
    )
    corpus = read_corpus(empty_project, "reviews")
    assert corpus.texts == ["Good, really good", "Two\nlines"] and corpus.labels() == ["5", "1"]
    assert done.skipped == 1
    import_text(empty_project, "tabs", "text\tlabel\na b\tx\n", filename="t.tsv", fmt="tsv")
    # A cell as long as a corpus keeps a text, longer than the csv module reads by default.
    long = "word " * 60_000
    import_text(empty_project, "long", f'text\n"{long}"\n', filename="l.csv", fmt="csv")
    assert texts_of(empty_project, "long") == [long]
    assert texts_of(empty_project, "tabs") == ["a b"]
    with pytest.raises(CorpusError, match=r"no column 'body'\. Its columns are 'review', 'stars'"):
        import_text(empty_project, "x", csv_text, filename="r.csv", fmt="csv", text_field="body")


def test_parquet_reads_its_columns_and_whole_numbers_stay_whole(
    empty_project: Project, tmp_path: Path
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "data" / "train.parquet"
    path.parent.mkdir()
    table = pa.table({"text": ["alpha", "beta", "gamma", "delta"], "label": [1.0, 0.0, 1.0, 0.0]})
    pq.write_table(table, path, row_group_size=2)
    done = import_file(empty_project, "greek", path, label_field="label", limit=3)
    corpus = read_corpus(empty_project, "greek")
    assert corpus.texts == ["alpha", "beta", "gamma"] and [r.label for r in corpus.records] == [
        1,
        0,
        1,
    ]
    source = done.corpus.meta.source  # type: ignore[union-attr]
    # The file's name and hash, never its folder.
    assert source.path == "train.parquet" and str(tmp_path) not in source.model_dump_json()
    assert "the first 3 texts" in source.detail
    with pytest.raises(CorpusError, match="no column 'sentence'"):
        import_file(empty_project, "x", path, text_field="sentence")


def test_a_files_format_comes_from_its_name() -> None:
    assert format_of("a/b/train.parquet") == "parquet"
    assert format_of("x.JSONL") == "jsonl" and format_of("x.ndjson") == "jsonl"
    assert (
        format_of("notes.txt") == "lines" and format_of("notes.txt", "paragraphs") == "paragraphs"
    )
    with pytest.raises(CorpusError, match="compressed"):
        format_of("train.jsonl.gz")


def test_prompts_become_texts_labelled_by_side_or_by_meta(project: Project) -> None:
    both = import_prompts(
        project, "sides", dataset="datasets/ioi.jsonl", sides="both", label="side", limit=3
    )
    corpus = read_corpus(project, "sides")
    assert len(corpus.records) == 6 and corpus.labels() == ["clean", "corrupt"]
    assert [r.meta for r in corpus.records[:2]] == [
        {"prompt": 0, "side": "clean"},
        {"prompt": 0, "side": "corrupt"},
    ]
    source = both.corpus.meta.source  # type: ignore[union-attr]
    assert source.kind == "dataset" and source.path == "datasets/ioi.jsonl" and source.sha256
    import_prompts(
        project, "templates", dataset="datasets/ioi.jsonl", sides="clean", label="meta:template"
    )
    assert read_corpus(project, "templates").labels()  # the templates the prompts were made from
    with pytest.raises(CorpusError, match="has no meta field 'colour'"):
        import_prompts(
            project, "x", dataset="datasets/ioi.jsonl", sides="clean", label="meta:colour"
        )
    with pytest.raises(CorpusError, match="isn't a way to label prompts"):
        import_prompts(project, "x", dataset="datasets/ioi.jsonl", sides="clean", label="size")


def test_texts_written_from_python_take_one_label_each(empty_project: Project) -> None:
    import numpy as np

    write_texts(empty_project, "mine", ["a", "b"], [np.int64(1), "two"])
    assert [r.label for r in read_corpus(empty_project, "mine").records] == [1, "two"]
    with pytest.raises(CorpusError, match="2 texts and 1 labels"):
        write_texts(empty_project, "other", ["a", "b"], [1])


def test_a_hub_dataset_file_is_read_at_the_commit_it_was_downloaded_at(
    empty_project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from logogram.backends import hub

    local = tmp_path / "cache" / "train.jsonl"
    local.parent.mkdir()
    local.write_text('{"text": "one", "label": 0}\n{"text": "two", "label": 1}\n', encoding="utf-8")
    asked: list[tuple[str, str, str | None]] = []

    def download(repo, path, revision, progress=None, *, cancel=None, limit=None):  # type: ignore[no-untyped-def]
        asked.append((repo, path, revision))
        if progress is not None:
            progress(10, 10, path)
        return "f" * 40, local

    monkeypatch.setattr(hub, "download_dataset_file", download)
    seen: list[int] = []
    done = import_hub(
        empty_project,
        "small",
        repo="someone/small",
        path="data/train.jsonl",
        revision="main",
        label_field="label",
        progress=lambda d, t, f: seen.append(d),
    )
    assert asked == [("someone/small", "data/train.jsonl", "main")] and seen == [10]
    source = done.corpus.meta.source  # type: ignore[union-attr]
    assert (source.kind, source.repo, source.path, source.revision) == (
        "hub",
        "someone/small",
        "data/train.jsonl",
        "f" * 40,
    )
    assert read_corpus(empty_project, "small").texts == ["one", "two"]
    with pytest.raises(CorpusError, match="isn't a Hugging Face dataset id"):
        import_hub(empty_project, "x", repo="not a repo", path="a.txt", revision=None)
    with pytest.raises(CorpusError, match="isn't a file of a dataset"):
        import_hub(empty_project, "x", repo="a/b", path="../a.txt", revision=None)


def test_offline_a_dataset_file_comes_from_the_cache(
    empty_project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from huggingface_hub import constants

    from logogram.backends.base import BackendError

    # The tests run with HF_HUB_OFFLINE set, so Hugging Face is never asked.
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    commit = "e" * 40
    file = tmp_path / "datasets--org--texts" / "snapshots" / commit / "plain" / "all.txt"
    file.parent.mkdir(parents=True)
    file.write_text("first\nsecond\n", encoding="utf-8")
    done = import_hub(
        empty_project, "cached", repo="org/texts", path="plain/all.txt", revision=None
    )
    assert done.corpus.texts == ["first", "second"]
    assert done.corpus.meta.source.revision == commit  # type: ignore[union-attr]
    with pytest.raises(BackendError, match="isn't in your local cache"):
        import_hub(empty_project, "x", repo="org/texts", path="plain/other.txt", revision=None)


def test_a_page_of_texts_is_found_by_a_search(empty_project: Project) -> None:
    texts = [f"text number {i}" for i in range(250)]
    labels = ["even" if i % 2 == 0 else "odd" for i in range(250)]
    corpus = write_texts(empty_project, "many", texts, labels).corpus
    first = corpus_page(corpus)
    assert (first["matches"], first["pages"], len(first["texts"])) == (250, 3, 100)
    last = corpus_page(corpus, page=99)  # past the last page: the last
    assert last["page"] == 2 and [t["index"] for t in last["texts"]][:2] == [200, 201]
    found = corpus_page(corpus, query="NUMBER 12")
    assert [t["index"] for t in found["texts"]] == [12, *range(120, 130)]
    odd = corpus_page(corpus, query="odd")  # labels are searched too
    assert odd["matches"] == 125 and odd["texts"][0]["label"] == "odd"
