"""Seeded datasets for classic tasks: a seed gives the same prompts, every record is a usable pair
(same token length, answers that differ), named positions name the right words, words the model
splits are left out, and bad settings say what to change.

A tokenizer stand-in plays the model (a word with its leading space, a pair of digits, a run of
punctuation or of whitespace is one token), except for IOI, which also runs on the tiny model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import pytest

from logogram.backends.base import Tokenized
from logogram.datasets import parse_jsonl, to_jsonl
from logogram.ioi import generate_ioi
from logogram.prompts import prepare_prompts
from logogram.spec import METRIC_LABELS
from logogram.tasks import (
    ANIMATE_NOUNS,
    CAPITALS,
    FEMALE_NAMES,
    INANIMATE_NOUNS,
    MALE_NAMES,
    TASKS,
    TaskError,
    check_options,
    generate_task,
)

PIECE = re.compile(r" ?\d{1,2}| ?[A-Za-z]+| ?[^\w\s]+|\s+|.")

ALL_TASKS = list(TASKS)

LABELS = {
    "ioi": {"IO", "S1", "S2", "end"},
    "greater_than": {"YY", "XX2", "end"},
    "docstring": {"def_arg", "end"},
    "gendered_pronoun": {"name", "end"},
    "subject_verb_agreement": {"subject", "attractor", "end"},
    "factual_recall": {"country", "end"},
}


class StandIn:
    """A tokenizer stand-in. Words in ``split`` are two tokens instead of one."""

    def __init__(self, split: Iterable[str] = ()) -> None:
        self.split = frozenset(split)

    def spans(self, text: str) -> list[tuple[int, int]]:
        out = []
        for match in PIECE.finditer(text):
            start, end = match.span()
            word = match.group().strip()
            if word in self.split:
                middle = end - len(word) // 2
                out += [(start, middle), (middle, end)]
            else:
                out.append((start, end))
        return out

    def count(self, text: str) -> int:
        return len(self.spans(text))

    def single(self, text: str) -> bool:
        return self.count(text) == 1

    def callbacks(self) -> dict[str, Any]:
        return {"single_token": self.single, "token_count": self.count}


class StandInBackend:
    """As much of a model backend as preparing prompts uses, tokenizing like the stand-in."""

    def __init__(self, standin: StandIn) -> None:
        self.standin = standin
        self.vocab: dict[str, int] = {"<bos>": 0}
        self.info = SimpleNamespace(n_ctx=1024)

    def tokenize(self, text: str, prepend_bos: bool) -> Tokenized:
        offsets = self.standin.spans(text)
        tokens = [text[a:b] for a, b in offsets]
        ids = [self.vocab.setdefault(t, len(self.vocab)) for t in tokens]
        if prepend_bos:
            return Tokenized(ids=[0, *ids], tokens=["<bos>", *tokens], offsets=[(0, 0), *offsets])
        return Tokenized(ids=ids, tokens=tokens, offsets=offsets)

    def single_token_id(self, text: str) -> int | None:
        ids = self.tokenize(text, prepend_bos=False).ids
        return ids[0] if len(ids) == 1 else None

    def token_str(self, token_id: int) -> str:
        return next(token for token, i in self.vocab.items() if i == token_id)


def alternatives(answer: str | list[str]) -> set[str]:
    return set(answer) if isinstance(answer, list) else {answer}


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+", text)


def every_template(task_id: str) -> dict[str, Any]:
    return {"templates": [t.id for t in TASKS[task_id].templates]}


def dumps(records: list[Any]) -> list[dict[str, Any]]:
    return [r.model_dump() for r in records]


# -- the registry -------------------------------------------------------------------------------


def test_the_registry_describes_every_task():
    assert ALL_TASKS == [
        "ioi",
        "greater_than",
        "docstring",
        "gendered_pronoun",
        "subject_verb_agreement",
        "factual_recall",
    ]
    for task_id, task in TASKS.items():
        assert task.id == task_id
        assert task.name[0].isupper() and task.name[1:] == task.name[1:].lower()
        assert task.description.endswith(".")
        assert task.metric in METRIC_LABELS
        ids = [t.id for t in task.templates]
        assert len(set(ids)) == len(ids)
        options = check_options(task_id)
        assert list(options) == [o.name for o in task.options]
        assert options["templates"] == [t.id for t in task.templates if t.default]
        assert json.loads(json.dumps(task.to_dict()))["options"][0]["name"] == "templates"
    metrics = {task_id: task.recommended_metric() for task_id, task in TASKS.items()}
    assert metrics == {
        "ioi": "logit_diff",
        "greater_than": "prob_diff",
        "docstring": "logprob_diff",
        "gendered_pronoun": "logit_diff",
        "subject_verb_agreement": "logit_diff",
        "factual_recall": "logit_diff",
    }
    factual = TASKS["factual_recall"]
    assert factual.recommended_metric({"single_token_answers": False}) == "logprob_diff"


def test_the_word_lists_hold_what_they_say():
    assert len(CAPITALS) == 82
    assert len({c for c, _ in CAPITALS}) == len({cap for _, cap in CAPITALS}) == 82
    assert len(set(MALE_NAMES)) == len(set(FEMALE_NAMES)) == 40
    assert not set(MALE_NAMES) & set(FEMALE_NAMES)
    nouns = [form for pair in ANIMATE_NOUNS + INANIMATE_NOUNS for form in pair]
    assert len(ANIMATE_NOUNS + INANIMATE_NOUNS) >= 30 and len(set(nouns)) == len(nouns)


# -- every task ---------------------------------------------------------------------------------


@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_a_seed_gives_the_same_prompts(task_id):
    tok = StandIn()
    a = generate_task(task_id, 40, 3, {}, **tok.callbacks())
    assert dumps(a) == dumps(generate_task(task_id, 40, 3, {}, **tok.callbacks()))
    assert dumps(a) != dumps(generate_task(task_id, 40, 4, {}, **tok.callbacks()))
    assert len(a) == 40
    assert len({r.clean for r in a}) == 40 and len({r.id for r in a}) == 40


def test_seed_zero_gives_these_prompts():
    # Randomness comes from random.Random.random() alone, whose sequence Python keeps the same
    # across versions: a change here changes every dataset made from a seed.
    first = {task_id: generate_task(task_id, 1, 0)[0].clean for task_id in ALL_TASKS[1:]}
    assert first == {
        "greater_than": "The occupation continued from the year 1478 to the year 14",
        "docstring": (
            'def water(self, bread, token, state, root):\n    """chair price rate\n\n'
            "    :param bread: fruit fire\n    :param token: mode chair\n    :param"
        ),
        "gendered_pronoun": "So Jason was late for work again, wasn't",
        "subject_verb_agreement": "The picture beside the soldiers",
        "factual_recall": "The capital city of Zimbabwe is called",
    }


@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_records_are_usable_pairs(task_id):
    tok = StandIn()
    records = generate_task(task_id, 60, 0, every_template(task_id), **tok.callbacks())
    assert dumps(parse_jsonl(to_jsonl(records))) == dumps(records)
    template_ids = {t.id for t in TASKS[task_id].templates}
    for r in records:
        assert r.clean != r.corrupt
        assert tok.count(r.clean) == tok.count(r.corrupt)
        answers, distractors = alternatives(r.answer), alternatives(r.distractor)
        assert answers.isdisjoint(distractors)
        assert all(tok.single(t) for t in answers | distractors)
        assert r.meta["template"] in template_ids
        assert set(r.positions) == LABELS[task_id]
        start, end = r.positions["end"]
        assert end == len(r.clean) and r.clean[start - 1] in " \n"
        assert not any(c.isspace() for c in r.clean[start:end])


@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_tasks_work_without_a_model(task_id):
    records = generate_task(task_id, 30, 1)
    assert len(records) == 30
    for r in records:
        assert r.clean != r.corrupt
        assert alternatives(r.answer).isdisjoint(alternatives(r.distractor))


@pytest.mark.parametrize(
    "task_id, options",
    [(task_id, {}) for task_id in ALL_TASKS]
    + [("factual_recall", {"single_token_answers": False})],
)
def test_records_prepare_for_a_model(task_id, options):
    tok = StandIn(split={"Reykjavik", "Kathmandu"})
    records = generate_task(task_id, 50, 2, options, **tok.callbacks())
    continuations = TASKS[task_id].recommended_metric(options) != "logit_diff"
    prepared = prepare_prompts(StandInBackend(tok), records, True, continuations=continuations)
    assert len(prepared) == len(records)
    for p in prepared:
        assert set(p.labels) == LABELS[task_id]
        assert p.labels["end"] == p.length - 1
        if task_id == "greater_than":
            assert p.clean.tokens[p.labels["YY"]] == f"{p.record.meta['yy']:02d}"
        if task_id == "docstring":
            assert p.clean.tokens[p.labels["def_arg"]] == p.record.answer


def test_ioi_prompts_prepare_for_the_tiny_model(tiny_backend):
    backend = tiny_backend
    records = generate_task(
        "ioi",
        24,
        0,
        {},
        single_token=lambda text: backend.single_token_id(text) is not None,
        token_count=lambda text: len(backend.tokenize(text, prepend_bos=False).ids),
    )
    prepared = prepare_prompts(backend, records, prepend_bos=True)
    assert len(prepared) == 24
    for p in prepared:
        assert set(p.labels) == LABELS["ioi"]
        assert p.labels["end"] == p.length - 1
        assert p.clean.tokens[p.labels["IO"]] == p.record.answer
        assert p.clean.tokens[p.labels["S2"]] == p.record.distractor


@pytest.mark.parametrize(
    "task_id, word",
    [
        ("ioi", "Mary"),
        ("greater_than", "war"),
        ("docstring", "load"),
        ("gendered_pronoun", "Mary"),
        ("subject_verb_agreement", "keys"),
        ("factual_recall", "Paris"),
    ],
)
def test_words_the_model_splits_are_left_out(task_id, word):
    tok = StandIn(split={word})
    records = generate_task(task_id, 100, 0, every_template(task_id), **tok.callbacks())
    for r in records:
        texts = [r.clean, r.corrupt, *alternatives(r.answer), *alternatives(r.distractor)]
        assert all(word not in words(text) for text in texts)


# -- each task ----------------------------------------------------------------------------------


def test_ioi_is_made_by_the_ioi_generator():
    options = {"templates": ["went", "met"], "patterns": ["BABA"], "corruption": "abc"}
    made = generate_task("ioi", 20, 5, options)
    same = generate_ioi(20, seed=5, templates=["went", "met"], patterns=["BABA"], corruption="abc")
    assert dumps(made) == dumps(same)
    with pytest.raises(TaskError, match="splits too many"):
        generate_task("ioi", 5, 0, {}, single_token=lambda text: False)


def test_greater_than_answers_are_the_years_after_the_start():
    tok = StandIn(split={"07", "93"})
    kept = {f"{v:02d}" for v in range(100)} - {"07", "93"}
    records = generate_task(
        "greater_than", 80, 0, every_template("greater_than"), **tok.callbacks()
    )
    for r in records:
        yy, century = r.meta["yy"], r.meta["century"]
        assert 2 <= yy <= 98 and yy not in (7, 93)
        assert set(r.answer) | set(r.distractor) == kept
        assert all(int(a) > yy for a in r.answer) and all(int(d) <= yy for d in r.distractor)
        start, end = r.positions["YY"]
        assert r.clean[start:end] == f"{yy:02d}" and r.clean[start - 2 : start] == str(century)
        # The corrupt prompt starts in the year 01 of the same century.
        assert r.corrupt == r.clean[:start] + "01" + r.clean[end:]
        assert r.clean[slice(*r.positions["XX2"])] == str(century)
        assert r.positions["end"] == r.positions["XX2"] and r.clean.endswith(f" {century}")
        assert r.meta["noun"] in words(r.clean)


def test_greater_than_leaves_out_years_that_are_one_token():
    tok = StandIn()

    def count(text: str) -> int:  # like GPT-2, which has tokens such as " 1850"
        return 1 if re.fullmatch(r" 18[5-9]\d", text) else tok.count(text)

    records = generate_task("greater_than", 300, 0, {}, single_token=tok.single, token_count=count)
    eighteenth = [r.meta["yy"] for r in records if r.meta["century"] == 18]
    assert eighteenth and all(yy < 50 for yy in eighteenth)


def test_greater_than_needs_years_split_into_a_century_and_two_digits():
    digits = re.compile(r"\d| ?[A-Za-z]+| ?[^\w\s]+|\s+")  # one token per digit

    def count(text: str) -> int:
        return len(digits.findall(text))

    with pytest.raises(TaskError, match="century token"):
        generate_task(
            "greater_than", 10, 0, {}, single_token=lambda t: count(t) == 1, token_count=count
        )


def test_docstring_answer_is_the_next_argument_of_the_definition():
    tok = StandIn()
    records = generate_task("docstring", 60, 0, every_template("docstring"), **tok.callbacks())
    assert {r.meta["template"] for r in records} == {"four_args", "three_args", "five_args"}
    for r in records:
        definition = r.clean.split("\n")[0]
        args = re.findall(r", (\w+)", definition)
        documented = re.findall(r":param (\w+):", r.clean)
        assert args[: len(documented)] == documented
        assert r.answer == " " + args[len(documented)]
        assert r.distractor == [" " + a for a in args if " " + a != r.answer]
        start, end = r.positions["def_arg"]
        assert r.clean[start:end] == r.answer.strip() and end <= len(definition)
        assert r.clean[slice(*r.positions["end"])] == ":param"
        # A random def: every argument of the definition is renamed; the docstring stays.
        renamed = re.findall(r", (\w+)", r.corrupt.split("\n")[0])
        assert len(renamed) == len(args) and set(renamed).isdisjoint(args)
        assert r.corrupt.split("(")[0] == definition.split("(")[0]
        assert r.corrupt.split("\n")[1:] == r.clean.split("\n")[1:]


def test_pronoun_corruption_names_someone_of_the_other_gender():
    tok = StandIn()
    records = generate_task(
        "gendered_pronoun", 60, 0, every_template("gendered_pronoun"), **tok.callbacks()
    )
    assert [r.meta["gender"] for r in records] == ["male", "female"] * 30
    for r in records:
        start, end = r.positions["name"]
        name, tail = r.clean[start:end], r.clean[end:]
        assert r.corrupt.startswith(r.clean[:start]) and r.corrupt.endswith(tail)
        other = r.corrupt[start : len(r.corrupt) - len(tail)]
        if r.meta["gender"] == "male":
            assert name in MALE_NAMES and other in FEMALE_NAMES
            assert (r.answer, r.distractor) == (" he", " she")
        else:
            assert name in FEMALE_NAMES and other in MALE_NAMES
            assert (r.answer, r.distractor) == (" she", " he")


def test_pronoun_pairs_names_of_the_same_token_length():
    tok = StandIn()

    def count(text: str) -> int:  # Elizabeth and Richard are two tokens, other names one
        return tok.count(text) + text.count("Elizabeth") + text.count("Richard")

    records = generate_task(
        "gendered_pronoun",
        200,
        0,
        every_template("gendered_pronoun"),
        single_token=tok.single,
        token_count=count,
    )
    long_names = [r for r in records if {"Elizabeth", "Richard"} & set(words(r.clean))]
    assert long_names
    for r in records:
        assert count(r.clean) == count(r.corrupt)
    for r in long_names:
        assert {"Elizabeth", "Richard"} & set(words(r.corrupt))


def test_agreement_corruption_flips_the_subject_number():
    tok = StandIn()
    records = generate_task(
        "subject_verb_agreement", 60, 0, every_template("subject_verb_agreement"), **tok.callbacks()
    )
    assert [r.meta["number"] for r in records] == ["singular", "plural"] * 30
    pair_of = {form: pair for pair in ANIMATE_NOUNS + INANIMATE_NOUNS for form in pair}
    animate = {form for pair in ANIMATE_NOUNS for form in pair}
    for r in records:
        start, end = r.positions["subject"]
        subject = r.clean[start:end]
        attractor = r.clean[slice(*r.positions["attractor"])]
        singular, plural = pair_of[subject]
        flipped = plural if subject == singular else singular
        assert r.corrupt == r.clean[:start] + flipped + r.clean[end:]
        assert pair_of[attractor] != pair_of[subject]
        if r.meta["number"] == "singular":
            assert subject == singular and attractor == pair_of[attractor][1]
            assert (r.answer, r.distractor) == (" is", " are")
        else:
            assert subject == plural and attractor == pair_of[attractor][0]
            assert (r.answer, r.distractor) == (" are", " is")
        if r.meta["template"] == "relative":
            assert attractor in animate
            assert r.clean.endswith(" like" if r.meta["number"] == "singular" else " likes")


def test_factual_recall_keeps_capitals_that_are_single_tokens():
    tok = StandIn(split={"Reykjavik", "Kathmandu"})
    capital_of = dict(CAPITALS)
    country_of = {capital: country for country, capital in CAPITALS}
    # 82 countries, less five capitals of two words and the two split ones, in three templates.
    records = generate_task("factual_recall", 225, 0, {}, **tok.callbacks())
    for r in records:
        country = r.meta["country"]
        start, end = r.positions["country"]
        assert r.clean[start:end] == country
        assert r.answer == " " + capital_of[country]
        other = country_of[r.distractor[1:]]
        assert other != country
        assert r.corrupt == r.clean[:start] + other + r.clean[end:]
        assert tok.single(r.answer) and tok.single(r.distractor)
    used = {r.answer for r in records} | {r.distractor for r in records}
    assert len(used) == 75 and not {" Reykjavik", " New Delhi", " Phnom Penh"} & used
    with pytest.raises(TaskError, match="at most 225 distinct prompts"):
        generate_task("factual_recall", 226, 0, {}, **tok.callbacks())


def test_factual_recall_can_keep_capitals_of_several_tokens():
    tok = StandIn()
    options = {"single_token_answers": False}
    records = generate_task("factual_recall", 246, 0, options, **tok.callbacks())
    answers = {r.answer for r in records}
    assert {
        " New Delhi",
        " Buenos Aires",
        " Addis Ababa",
        " Kuala Lumpur",
        " Phnom Penh",
    } <= answers
    for r in records:
        assert tok.count(r.clean) == tok.count(r.corrupt)
    with pytest.raises(TaskError, match="at most 246 distinct prompts"):
        generate_task("factual_recall", 247, 0, options, **tok.callbacks())


# -- settings -----------------------------------------------------------------------------------


def test_every_distinct_prompt_can_be_made():
    # One template and 40 names of each gender: 80 prompts, each name once.
    records = generate_task("gendered_pronoun", 80, 0, {"templates": ["friend"]})
    assert sorted(r.clean.split()[1] for r in records) == sorted(MALE_NAMES + FEMALE_NAMES)
    message = r"at most 80 distinct prompts\. Choose at most 80 prompts, or choose more templates\."
    with pytest.raises(TaskError, match=message):
        generate_task("gendered_pronoun", 81, 0, {"templates": ["friend"]})


def test_generation_gives_up_when_pairs_never_line_up():
    tok = StandIn()

    def count(text: str) -> int:  # every corrupt year is one token longer
        return tok.count(text) + (re.search(r"\d\d01 ", text) is not None)

    with pytest.raises(TaskError, match="Could only make 0 distinct prompts"):
        generate_task("greater_than", 5, 0, {}, single_token=tok.single, token_count=count)


@pytest.mark.parametrize(
    "task_id, n, seed, options, message",
    [
        ("nope", 5, 0, {}, "Unknown task 'nope'. Choose one of: ioi, greater_than"),
        ("docstring", 0, 0, {}, "Choose at least one prompt"),
        ("docstring", 100_001, 0, {}, "Choose at most 100,000 prompts"),
        ("docstring", 2.5, 0, {}, "whole number of prompts"),
        ("docstring", 5, -1, {}, "Choose a seed from 0 to 4,294,967,295"),
        ("docstring", 5, 2**32, {}, "Choose a seed from 0"),
        ("docstring", 5, 0, ["templates"], "Options must map option names to values"),
        ("docstring", 5, 0, {"colour": 1}, "Docstring has no option 'colour'. Its options are"),
        ("docstring", 5, 0, {"templates": "four_args"}, "templates must be a list"),
        ("docstring", 5, 0, {"templates": []}, "Choose at least one template"),
        ("docstring", 5, 0, {"templates": ["four_args", "six"]}, r"Unknown template\(s\): six"),
        ("docstring", 5, 0, {"templates": ["four_args", "four_args"]}, "listed twice"),
        ("ioi", 5, 0, {"corruption": "swap"}, "Unknown corruption 'swap'. Use one of: flip"),
        ("ioi", 5, 0, {"patterns": ["ABBA", "ABAB"]}, r"Unknown pattern\(s\): ABAB"),
        ("factual_recall", 5, 0, {"single_token_answers": "yes"}, "must be true or false"),
        # Without a model, capitals of two words aren't single tokens: 77 countries, 3 templates.
        (
            "factual_recall",
            5000,
            0,
            {},
            r"make at most 231 distinct prompts\. Choose at most 231 prompts\.$",
        ),
    ],
)
def test_bad_settings_say_what_to_change(task_id, n, seed, options, message):
    with pytest.raises(TaskError, match=message):
        generate_task(task_id, n, seed, options)


def test_options_left_out_or_none_take_their_defaults():
    assert check_options("ioi", {"templates": None}) == {
        "templates": ["went", "arrived", "got"],
        "patterns": ["ABBA", "BABA"],
        "corruption": "flip",
    }
    assert check_options("factual_recall", {"single_token_answers": False}) == {
        "templates": ["capital_of", "capital_city", "called"],
        "single_token_answers": False,
    }
