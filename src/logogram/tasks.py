"""Seeded prompt datasets for classic interpretability tasks.

Every task fills templates with words from built-in lists, and pairs each clean prompt with a
corrupt prompt of the same token length that changes what the model should say next:

* ``ioi``: indirect object identification (Wang et al. 2022), made by :mod:`logogram.ioi`.
* ``greater_than``: "The war lasted from the year 1732 to the year 17" (Hanna et al. 2023). The
  answer is the set of two-digit years after 32, the distractor the set of the others.
* ``docstring``: the next argument of a Python function in its docstring (Heimersheim and Janiak
  2023).
* ``gendered_pronoun``: " he" or " she" after a first name (Mathwin et al. 2023).
* ``subject_verb_agreement``: " is" or " are" after a noun of the other number than the subject
  (Linzen et al. 2016; Finlayson et al. 2021).
* ``factual_recall``: a country's capital (as in Meng et al. 2022).

``single_token(text)`` says whether ``text`` is one token for the model the dataset is for, and
``token_count(text)`` how many tokens it is (without a beginning-of-sequence token). Words that
must be single tokens are filtered with the first; clean and corrupt prompts are paired with the
second. Without them (no model loaded), a word with its leading space, a pair of digits, a run of
punctuation or a run of whitespace counts as one token: the records are valid, but a model's own
tokenizer may split some words, so generate the dataset again once the model is loaded.

Named positions are character spans in the clean prompt, and ``meta["template"]`` names each
prompt's template, for a bootstrap clustered by template. Randomness uses only
``random.Random.random()``, whose sequence Python guarantees across versions, and lists are walked
in a fixed order, so a seed gives the same dataset on every machine.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from logogram.datasets import PromptRecord
from logogram.ioi import TEMPLATES as IOI_TEMPLATES
from logogram.ioi import IOIError, default_template_ids, generate_ioi
from logogram.spec import SEED_MAX

MAX_PROMPTS = 100_000

SingleToken = Callable[[str], bool]
TokenCount = Callable[[str], int]
# (n, seed, checked options, single_token, token_count) -> records
Generate = Callable[
    [int, int, dict[str, Any], SingleToken | None, TokenCount | None], list[PromptRecord]
]

T = TypeVar("T")


class TaskError(ValueError):
    """A dataset can't be generated with these settings. The message says how to fix it."""


@dataclass(frozen=True)
class TaskOption:
    """A setting of a task. ``type`` is ``"bool"`` (true or false), ``"choice"`` (one value of
    ``allowed``) or ``"choices"`` (a list of distinct values of ``allowed``, at least one)."""

    name: str
    type: Literal["bool", "choice", "choices"]
    default: Any
    description: str
    allowed: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskTemplate:
    """A prompt template: ``{SLOT}`` marks where words go. The ``default`` templates are used
    when the templates option isn't given."""

    id: str
    text: str
    default: bool = False


@dataclass(frozen=True)
class TaskInfo:
    """A task: what it asks of the model, the metric that reads its answers, its options and its
    templates. Make datasets with :func:`generate_task`, which checks the settings first;
    ``generate`` expects checked options with their defaults filled in."""

    id: str
    name: str
    description: str
    metric: str
    options: tuple[TaskOption, ...]
    templates: tuple[TaskTemplate, ...]
    generate: Generate
    # (option, value, metric): the metric to recommend instead of ``metric`` for that value.
    metric_when: tuple[tuple[str, Any, str], ...] = ()

    def recommended_metric(self, options: Mapping[str, Any] | None = None) -> str:
        """The metric kind that reads this task's answers, with these options."""
        chosen = _check_options(self, options)
        for name, value, metric in self.metric_when:
            if chosen[name] == value:
                return metric
        return self.metric

    def to_dict(self) -> dict[str, Any]:
        """The task as JSON-ready data, for listing the tasks."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "metric": self.metric,
            "metric_when": [
                {"option": name, "value": value, "metric": metric}
                for name, value, metric in self.metric_when
            ],
            "options": [
                {
                    "name": o.name,
                    "type": o.type,
                    "default": list(o.default) if isinstance(o.default, tuple) else o.default,
                    "description": o.description,
                    "allowed": list(o.allowed),
                }
                for o in self.options
            ],
            "templates": [
                {"id": t.id, "text": t.text, "default": t.default} for t in self.templates
            ],
        }


# ---------------------------------------------------------------------------------------------
# Shared pieces

# Without a model: a word with its leading space, a pair of digits, a run of punctuation or a run
# of whitespace is one token.
_PIECE = re.compile(r" ?\d{1,2}| ?[^\W\d]+| ?[^\w\s]+|\s+")
_SLOT = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")


class _Tokens:
    """The model's token checks, remembered per text, or word-level stand-ins without a model."""

    def __init__(self, single_token: SingleToken | None, token_count: TokenCount | None) -> None:
        self._single_token = single_token
        self._token_count = token_count
        self._single: dict[str, bool] = {}
        self._count: dict[str, int] = {}

    def count(self, text: str) -> int:
        if text not in self._count:
            if self._token_count is not None:
                self._count[text] = int(self._token_count(text))
            else:
                self._count[text] = len(_PIECE.findall(text))
        return self._count[text]

    def single(self, text: str) -> bool:
        if text not in self._single:
            if self._single_token is not None:
                self._single[text] = bool(self._single_token(text))
            else:
                self._single[text] = self.count(text) == 1
        return self._single[text]


def _pick(rng: random.Random, items: Sequence[T]) -> T:
    return items[int(rng.random() * len(items))]


def _sample(rng: random.Random, items: Sequence[T], k: int) -> list[T]:
    """``k`` distinct items in the order drawn (a partial Fisher–Yates shuffle)."""
    pool = list(items)
    for i in range(k):
        j = i + int(rng.random() * (len(pool) - i))
        pool[i], pool[j] = pool[j], pool[i]
    return pool[:k]


def _fill(template: str, values: Mapping[str, str]) -> tuple[str, dict[str, tuple[int, int]]]:
    """Fill a template's ``{SLOT}``s, returning the text and each slot's character span."""
    parts: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    length = 0
    cursor = 0
    for match in _SLOT.finditer(template):
        literal = template[cursor : match.start()]
        value = values[match.group(1)]
        start = length + len(literal)
        spans[match.group(1)] = (start, start + len(value))
        parts += [literal, value]
        length = start + len(value)
        cursor = match.end()
    parts.append(template[cursor:])
    return "".join(parts), spans


def _last_word(text: str) -> tuple[int, int]:
    """The span of the prompt's last word, whose last token predicts the answer."""
    return (max(text.rfind(" "), text.rfind("\n")) + 1, len(text))


def _chosen(templates: tuple[TaskTemplate, ...], ids: Sequence[str]) -> list[TaskTemplate]:
    by_id = {t.id: t for t in templates}
    return [by_id[i] for i in ids]


def _or_more_templates(more: bool) -> str:
    return ", or choose more templates" if more else ""


def _check_space(n: int, space: int, more_templates: bool) -> None:
    """Refuse at once to make more prompts than the settings allow."""
    if n > space:
        raise TaskError(
            f"These settings make at most {space:,} distinct prompts. Choose at most "
            f"{space:,} prompts{_or_more_templates(more_templates)}."
        )


def _collect(
    n: int, make: Callable[[int], PromptRecord | None], more_templates: bool
) -> list[PromptRecord]:
    """Call ``make(index)`` until it has given ``n`` records with distinct clean prompts. It
    returns None for a draw that can't be used, such as a pair of different token lengths."""
    records: list[PromptRecord] = []
    seen: set[str] = set()
    attempts = 0
    while len(records) < n:
        attempts += 1
        if attempts > 1000 + 50 * n:
            raise TaskError(
                f"Could only make {len(records):,} distinct prompts with these settings. Choose "
                f"fewer prompts{_or_more_templates(more_templates)}."
            )
        record = make(len(records))
        if record is None or record.clean in seen:
            continue
        seen.add(record.clean)
        records.append(record)
    return records


def _split_words(what: str) -> TaskError:
    return TaskError(
        f"This model's tokenizer splits too many of the built-in {what} into several tokens. "
        "Import a JSONL dataset written for this model instead."
    )


# ---------------------------------------------------------------------------------------------
# Indirect object identification


def _ioi(
    n: int,
    seed: int,
    options: dict[str, Any],
    single_token: SingleToken | None,
    token_count: TokenCount | None,
) -> list[PromptRecord]:
    try:
        return generate_ioi(
            n,
            seed=seed,
            templates=options["templates"],
            patterns=options["patterns"],
            corruption=options["corruption"],
            single_token=single_token,
        )
    except IOIError as exc:
        raise TaskError(str(exc)) from exc


# ---------------------------------------------------------------------------------------------
# Greater-than (Hanna et al. 2023)

# The default templates share one token structure when the noun is one token.
GREATER_THAN_TEMPLATES: tuple[TaskTemplate, ...] = (
    TaskTemplate("lasted", "The {NOUN} lasted from the year {XX}{YY} to the year {XX2}", True),
    TaskTemplate("ran", "The {NOUN} ran from the year {XX}{YY} to the year {XX2}", True),
    TaskTemplate(
        "continued", "The {NOUN} continued from the year {XX}{YY} to the year {XX2}", True
    ),
    TaskTemplate("started", "The {NOUN} started in the year {XX}{YY} and ended in the year {XX2}"),
    TaskTemplate("began", "The {NOUN} began in the year {XX}{YY} and ended in the year {XX2}"),
)

GREATER_THAN_NOUNS: tuple[str, ...] = (
    "war", "expedition", "siege", "journey", "reign", "strike", "drought", "project", "voyage",
    "contract", "campaign", "dispute", "occupation", "famine", "rebellion", "alliance",
)  # fmt: skip

CENTURIES: tuple[int, ...] = tuple(range(11, 20))


def _greater_than(
    n: int,
    seed: int,
    options: dict[str, Any],
    single_token: SingleToken | None,
    token_count: TokenCount | None,
) -> list[PromptRecord]:
    tok = _Tokens(single_token, token_count)
    templates = _chosen(GREATER_THAN_TEMPLATES, options["templates"])
    nouns = [w for w in GREATER_THAN_NOUNS if tok.single(" " + w)]
    if not nouns:
        raise _split_words("nouns")
    # Two-digit tokens, written without a space: " 17" is followed directly by "32".
    digit = [tok.single(f"{v:02d}") for v in range(100)]
    sets: dict[int, tuple[list[str], list[str]]] = {}
    for yy in range(2, 99):
        later = [f"{v:02d}" for v in range(yy + 1, 100) if digit[v]]
        rest = [f"{v:02d}" for v in range(yy + 1) if digit[v]]
        if digit[yy] and later and rest:
            sets[yy] = (later, rest)
    # A start year must be the century's token followed by the two digits' token, and so must the
    # corrupt prompt's year (century and 01).
    years = [
        (xx, yy)
        for xx in CENTURIES
        if tok.single(f" {xx}") and tok.count(f" {xx}01") == 2
        for yy in sets
        if tok.count(f" {xx}{yy:02d}") == 2
    ]
    if not years:
        raise TaskError(
            "This model's tokenizer doesn't split a year such as 1732 into a century token and a "
            "two-digit token, as GPT-2's does, so it can't read these prompts. Import a JSONL "
            "dataset written for this model instead."
        )
    more = len(templates) < len(GREATER_THAN_TEMPLATES)
    _check_space(n, len(templates) * len(nouns) * len(years), more)
    rng = random.Random(seed)

    def make(index: int) -> PromptRecord | None:
        template = _pick(rng, templates)
        noun = _pick(rng, nouns)
        xx, yy = _pick(rng, years)
        values = {"NOUN": noun, "XX": str(xx), "YY": f"{yy:02d}", "XX2": str(xx)}
        clean, spans = _fill(template.text, values)
        corrupt, _ = _fill(template.text, {**values, "YY": "01"})
        if tok.count(clean) != tok.count(corrupt):
            return None
        later, rest = sets[yy]
        return PromptRecord(
            clean=clean,
            corrupt=corrupt,
            answer=list(later),
            distractor=list(rest),
            positions={"YY": spans["YY"], "XX2": spans["XX2"], "end": _last_word(clean)},
            id=f"greater-than-{index:05d}",
            meta={"template": template.id, "noun": noun, "century": xx, "yy": yy},
        )

    return _collect(n, make, more)


# ---------------------------------------------------------------------------------------------
# Docstring (Heimersheim and Janiak 2023)

# Template id -> (arguments after self, arguments documented before the answer).
_DOCSTRING_SHAPES: dict[str, tuple[int, int]] = {
    "four_args": (4, 2),
    "three_args": (3, 1),
    "five_args": (5, 3),
}
_ARG_SLOTS = "ABCDE"


def _docstring_text(n_args: int, n_documented: int) -> str:
    slots = _ARG_SLOTS[:n_args]
    params = ", ".join(f"{{DEF_{x}}}" for x in slots)
    lines = [f"def {{FUNCTION}}(self, {params}):", '    """{SUMMARY}', ""]
    lines += [f"    :param {{DOC_{x}}}: {{DESC_{x}}}" for x in slots[:n_documented]]
    lines.append("    :param")
    return "\n".join(lines)


DOCSTRING_TEMPLATES: tuple[TaskTemplate, ...] = tuple(
    TaskTemplate(tid, _docstring_text(*shape), tid == "four_args")
    for tid, shape in _DOCSTRING_SHAPES.items()
)

# Short common words for function names, arguments and descriptions: no Python keywords or
# built-in names.
DOCSTRING_WORDS: tuple[str, ...] = (
    "load", "size", "files", "last", "port", "oil", "column", "piece", "crime", "population",
    "unit", "dark", "name", "value", "page", "line", "user", "level", "mode", "count", "step",
    "rate", "state", "color", "image", "node", "edge", "link", "model", "field", "source",
    "target", "result", "error", "token", "label", "title", "group", "order", "price", "money",
    "tree", "root", "block", "frame", "layer", "shape", "scale", "speed", "power", "water",
    "fire", "stone", "paper", "glass", "metal", "wood", "salt", "sugar", "milk", "bread",
    "fruit", "seed", "leaf", "branch", "river", "road", "bridge", "tower", "wall", "door",
    "window", "table", "chair", "box", "bag", "ship", "train", "plane", "car",
)  # fmt: skip


def _docstring(
    n: int,
    seed: int,
    options: dict[str, Any],
    single_token: SingleToken | None,
    token_count: TokenCount | None,
) -> list[PromptRecord]:
    tok = _Tokens(single_token, token_count)
    templates = _chosen(DOCSTRING_TEMPLATES, options["templates"])
    words = [w for w in DOCSTRING_WORDS if tok.single(" " + w)]
    # The summary's first word follows the quotes without a space.
    starts = [w for w in words if tok.single(w)]
    most = max(_DOCSTRING_SHAPES[t.id][0] for t in templates)
    # Arguments, their replacements, the function's name and three words for descriptions.
    if len(words) < 2 * most + 4 or not starts:
        raise _split_words("words")
    more = len(templates) < len(DOCSTRING_TEMPLATES)
    rng = random.Random(seed)

    def make(index: int) -> PromptRecord | None:
        template = _pick(rng, templates)
        n_args, n_documented = _DOCSTRING_SHAPES[template.id]
        slots = _ARG_SLOTS[:n_args]
        chosen = _sample(rng, words, 2 * n_args + 1)
        args, others, function = chosen[:n_args], chosen[n_args : 2 * n_args], chosen[-1]
        rest = [w for w in words if w not in chosen]
        first = [w for w in starts if w not in chosen]
        if not first:
            return None
        head = _pick(rng, first)
        summary = [head, *_sample(rng, [w for w in rest if w != head], 2)]
        values = {"FUNCTION": function, "SUMMARY": " ".join(summary)}
        for x, arg in zip(slots, args, strict=True):
            values[f"DEF_{x}"] = arg
        for x, arg in zip(slots[:n_documented], args[:n_documented], strict=True):
            values[f"DOC_{x}"] = arg
            values[f"DESC_{x}"] = " ".join(_sample(rng, rest, 2))
        clean, spans = _fill(template.text, values)
        # A random def: the definition's arguments get other names; the docstring keeps them.
        renamed = {f"DEF_{x}": other for x, other in zip(slots, others, strict=True)}
        corrupt, _ = _fill(template.text, {**values, **renamed})
        if tok.count(clean) != tok.count(corrupt):
            return None
        return PromptRecord(
            clean=clean,
            corrupt=corrupt,
            answer=" " + args[n_documented],
            distractor=[" " + a for i, a in enumerate(args) if i != n_documented],
            positions={"def_arg": spans[f"DEF_{slots[n_documented]}"], "end": _last_word(clean)},
            id=f"docstring-{index:05d}",
            meta={"template": template.id, "corruption": "random_def"},
        )

    return _collect(n, make, more)


# ---------------------------------------------------------------------------------------------
# Gendered pronouns (Mathwin et al. 2023)

# Each template ends where the pronoun comes next, and every name follows a space. The default
# templates share one token structure when the name is one token.
PRONOUN_TEMPLATES: tuple[TaskTemplate, ...] = (
    TaskTemplate("friend", "So {NAME} is a really great friend, isn't", True),
    TaskTemplate("musician", "So {NAME} is such a talented musician, isn't", True),
    TaskTemplate("late", "So {NAME} was late for work again, wasn't", True),
    TaskTemplate("meeting", "After the long meeting, {NAME} said that"),
    TaskTemplate("phone", "On the phone yesterday, {NAME} told me that"),
)

# Common first names that are clearly used for men or for women (no Alex, Sam or Taylor).
MALE_NAMES: tuple[str, ...] = (
    "John", "James", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas",
    "Charles", "Daniel", "Matthew", "Mark", "Paul", "Steven", "Andrew", "Kevin", "Brian",
    "George", "Edward", "Peter", "Henry", "Frank", "Jack", "Eric", "Jacob", "Ryan", "Gary",
    "Scott", "Adam", "Jason", "Larry", "Justin", "Tom", "Carl", "Fred", "Harry", "Luke", "Bruce",
    "Walter",
)  # fmt: skip

FEMALE_NAMES: tuple[str, ...] = (
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah",
    "Karen", "Nancy", "Lisa", "Margaret", "Sandra", "Emily", "Donna", "Michelle", "Carol",
    "Amanda", "Melissa", "Deborah", "Stephanie", "Rebecca", "Laura", "Helen", "Anna", "Emma",
    "Rachel", "Julia", "Alice", "Kate", "Lucy", "Ruth", "Diana", "Grace", "Sophie", "Claire",
    "Megan", "Amy", "Hannah",
)  # fmt: skip

_GENDERS = ("male", "female")
_PRONOUNS = {"male": " he", "female": " she"}


def _gendered_pronoun(
    n: int,
    seed: int,
    options: dict[str, Any],
    single_token: SingleToken | None,
    token_count: TokenCount | None,
) -> list[PromptRecord]:
    tok = _Tokens(single_token, token_count)
    if not (tok.single(" he") and tok.single(" she")):
        raise TaskError(
            "' he' and ' she' aren't both single tokens for this model. Import a JSONL dataset "
            "written for this model instead."
        )
    templates = _chosen(PRONOUN_TEMPLATES, options["templates"])
    names = {
        "male": [w for w in MALE_NAMES if tok.single(" " + w)],
        "female": [w for w in FEMALE_NAMES if tok.single(" " + w)],
    }
    if not names["male"] or not names["female"]:
        raise _split_words("names")
    # Per template and gender: each name whose prompt has as many tokens as the prompt with some
    # name of the other gender, and those names, which can replace it in the corrupt prompt.
    entries: list[tuple[TaskTemplate, dict[str, list[tuple[str, list[str]]]]]] = []
    for template in templates:
        counts = {
            g: [(name, tok.count(_fill(template.text, {"NAME": name})[0])) for name in names[g]]
            for g in _GENDERS
        }
        usable: dict[str, list[tuple[str, list[str]]]] = {}
        for g, other in zip(_GENDERS, reversed(_GENDERS), strict=True):
            usable[g] = []
            for name, count in counts[g]:
                partners = [m for m, k in counts[other] if k == count]
                if partners:
                    usable[g].append((name, partners))
        entries.append((template, usable))
    # Prompts alternate between the genders, starting with a name that takes " he".
    males = sum(len(usable["male"]) for _, usable in entries)
    females = sum(len(usable["female"]) for _, usable in entries)
    more = len(templates) < len(PRONOUN_TEMPLATES)
    _check_space(n, min(2 * males, 2 * females + 1), more)
    rng = random.Random(seed)

    def make(index: int) -> PromptRecord | None:
        gender = _GENDERS[index % 2]
        template, usable = _pick(rng, entries)
        if not usable[gender]:
            return None
        name, partners = _pick(rng, usable[gender])
        partner = _pick(rng, partners)
        clean, spans = _fill(template.text, {"NAME": name})
        corrupt, _ = _fill(template.text, {"NAME": partner})
        other = _GENDERS[1 - index % 2]
        return PromptRecord(
            clean=clean,
            corrupt=corrupt,
            answer=_PRONOUNS[gender],
            distractor=_PRONOUNS[other],
            positions={"name": spans["NAME"], "end": _last_word(clean)},
            id=f"gendered-pronoun-{index:05d}",
            meta={"template": template.id, "gender": gender},
        )

    return _collect(n, make, more)


# ---------------------------------------------------------------------------------------------
# Subject-verb agreement (Linzen et al. 2016; Finlayson et al. 2021)

# Each template ends where the main verb comes next, after an attractor: a noun of the other
# number than the subject. The default templates share one token structure when the nouns are
# one token. In "relative" the attractor likes the subject, so it is a person or an animal, and
# {VERB} agrees with it ("likes" or "like").
AGREEMENT_TEMPLATES: tuple[TaskTemplate, ...] = (
    TaskTemplate("near", "The {SUBJECT} near the {ATTRACTOR}", True),
    TaskTemplate("behind", "The {SUBJECT} behind the {ATTRACTOR}", True),
    TaskTemplate("beside", "The {SUBJECT} beside the {ATTRACTOR}", True),
    TaskTemplate("next_to", "The {SUBJECT} next to the {ATTRACTOR}"),
    TaskTemplate("relative", "The {SUBJECT} that the {ATTRACTOR} {VERB}"),
)
_ANIMATE_ATTRACTOR = frozenset({"relative"})

# (singular, plural): people and animals, then things.
ANIMATE_NOUNS: tuple[tuple[str, str], ...] = (
    ("author", "authors"), ("pilot", "pilots"), ("doctor", "doctors"), ("farmer", "farmers"),
    ("teacher", "teachers"), ("senator", "senators"), ("lawyer", "lawyers"),
    ("officer", "officers"), ("student", "students"), ("painter", "painters"),
    ("writer", "writers"), ("manager", "managers"), ("driver", "drivers"), ("guard", "guards"),
    ("singer", "singers"), ("dancer", "dancers"), ("surgeon", "surgeons"),
    ("banker", "bankers"), ("editor", "editors"), ("soldier", "soldiers"),
    ("minister", "ministers"), ("judge", "judges"), ("athlete", "athletes"),
    ("customer", "customers"), ("worker", "workers"), ("player", "players"), ("dog", "dogs"),
    ("cat", "cats"), ("horse", "horses"), ("bird", "birds"),
)  # fmt: skip

INANIMATE_NOUNS: tuple[tuple[str, str], ...] = (
    ("key", "keys"), ("cabinet", "cabinets"), ("book", "books"), ("table", "tables"),
    ("picture", "pictures"), ("window", "windows"), ("box", "boxes"), ("bottle", "bottles"),
    ("letter", "letters"), ("car", "cars"), ("house", "houses"), ("door", "doors"),
    ("chair", "chairs"), ("lamp", "lamps"), ("bag", "bags"), ("tree", "trees"),
)  # fmt: skip

_NUMBERS = ("singular", "plural")
_VERBS = {"singular": (" is", " are"), "plural": (" are", " is")}


def _subject_verb_agreement(
    n: int,
    seed: int,
    options: dict[str, Any],
    single_token: SingleToken | None,
    token_count: TokenCount | None,
) -> list[PromptRecord]:
    tok = _Tokens(single_token, token_count)
    if not (tok.single(" is") and tok.single(" are")):
        raise TaskError(
            "' is' and ' are' aren't both single tokens for this model. Import a JSONL dataset "
            "written for this model instead."
        )
    templates = _chosen(AGREEMENT_TEMPLATES, options["templates"])

    def usable(pairs: tuple[tuple[str, str], ...]) -> list[tuple[str, str]]:
        return [p for p in pairs if tok.single(" " + p[0]) and tok.single(" " + p[1])]

    animate = usable(ANIMATE_NOUNS)
    nouns = animate + usable(INANIMATE_NOUNS)
    entries = [(t, animate if t.id in _ANIMATE_ATTRACTOR else nouns) for t in templates]
    # Per number: every attractor with every other noun as the subject. Numbers alternate.
    per_number = sum(len(attractors) * (len(nouns) - 1) for _, attractors in entries)
    if per_number == 0:
        raise _split_words("nouns")
    more = len(templates) < len(AGREEMENT_TEMPLATES)
    _check_space(n, 2 * per_number, more)
    rng = random.Random(seed)

    def make(index: int) -> PromptRecord | None:
        number = _NUMBERS[index % 2]
        template, attractors = _pick(rng, entries)
        if not attractors:
            return None
        subject = _pick(rng, nouns)
        attractor = _pick(rng, attractors)
        if attractor == subject:
            return None
        if number == "singular":
            said, flipped, other, verb = subject[0], subject[1], attractor[1], "like"
        else:
            said, flipped, other, verb = subject[1], subject[0], attractor[0], "likes"
        values = {"SUBJECT": said, "ATTRACTOR": other, "VERB": verb}
        clean, spans = _fill(template.text, values)
        corrupt, _ = _fill(template.text, {**values, "SUBJECT": flipped})
        if tok.count(clean) != tok.count(corrupt):
            return None
        answer, distractor = _VERBS[number]
        return PromptRecord(
            clean=clean,
            corrupt=corrupt,
            answer=answer,
            distractor=distractor,
            positions={
                "subject": spans["SUBJECT"],
                "attractor": spans["ATTRACTOR"],
                "end": _last_word(clean),
            },
            id=f"subject-verb-agreement-{index:05d}",
            meta={"template": template.id, "number": number},
        )

    return _collect(n, make, more)


# ---------------------------------------------------------------------------------------------
# Factual recall: capital cities (as in Meng et al. 2022)

FACTUAL_TEMPLATES: tuple[TaskTemplate, ...] = (
    TaskTemplate("capital_of", "The capital of {COUNTRY} is", True),
    TaskTemplate("capital_city", "{COUNTRY}'s capital city is", True),
    TaskTemplate("called", "The capital city of {COUNTRY} is called", True),
)

# Well-established countries and capitals only.
CAPITALS: tuple[tuple[str, str], ...] = (
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"),
    ("Portugal", "Lisbon"), ("Japan", "Tokyo"), ("China", "Beijing"), ("Russia", "Moscow"),
    ("Canada", "Ottawa"), ("Australia", "Canberra"), ("Egypt", "Cairo"), ("Kenya", "Nairobi"),
    ("India", "New Delhi"), ("Argentina", "Buenos Aires"), ("Peru", "Lima"),
    ("Chile", "Santiago"), ("Greece", "Athens"), ("Turkey", "Ankara"), ("Iran", "Tehran"),
    ("Iraq", "Baghdad"), ("Poland", "Warsaw"), ("Austria", "Vienna"), ("Hungary", "Budapest"),
    ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Finland", "Helsinki"),
    ("Denmark", "Copenhagen"), ("Belgium", "Brussels"), ("Switzerland", "Bern"),
    ("Ireland", "Dublin"), ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"),
    ("South Korea", "Seoul"), ("North Korea", "Pyongyang"), ("Cuba", "Havana"),
    ("Nigeria", "Abuja"), ("Ghana", "Accra"), ("Ethiopia", "Addis Ababa"), ("Morocco", "Rabat"),
    ("Algeria", "Algiers"), ("Tunisia", "Tunis"), ("Saudi Arabia", "Riyadh"),
    ("Pakistan", "Islamabad"), ("Bangladesh", "Dhaka"), ("Nepal", "Kathmandu"),
    ("Afghanistan", "Kabul"), ("Romania", "Bucharest"), ("Bulgaria", "Sofia"),
    ("Serbia", "Belgrade"), ("Croatia", "Zagreb"), ("Iceland", "Reykjavik"),
    ("New Zealand", "Wellington"), ("Venezuela", "Caracas"), ("Uruguay", "Montevideo"),
    ("Jamaica", "Kingston"), ("Malaysia", "Kuala Lumpur"), ("Lebanon", "Beirut"),
    ("Syria", "Damascus"), ("Jordan", "Amman"), ("Qatar", "Doha"), ("Uganda", "Kampala"),
    ("Senegal", "Dakar"), ("Zimbabwe", "Harare"), ("Zambia", "Lusaka"), ("Angola", "Luanda"),
    ("Estonia", "Tallinn"), ("Latvia", "Riga"), ("Lithuania", "Vilnius"),
    ("Slovakia", "Bratislava"), ("Slovenia", "Ljubljana"), ("Ecuador", "Quito"),
    ("Cambodia", "Phnom Penh"), ("Laos", "Vientiane"), ("Libya", "Tripoli"),
    ("Somalia", "Mogadishu"), ("Rwanda", "Kigali"), ("Madagascar", "Antananarivo"),
    ("Cyprus", "Nicosia"), ("Malta", "Valletta"), ("Albania", "Tirana"), ("Armenia", "Yerevan"),
    ("Azerbaijan", "Baku"),
)  # fmt: skip


def _factual_recall(
    n: int,
    seed: int,
    options: dict[str, Any],
    single_token: SingleToken | None,
    token_count: TokenCount | None,
) -> list[PromptRecord]:
    tok = _Tokens(single_token, token_count)
    templates = _chosen(FACTUAL_TEMPLATES, options["templates"])
    single = options["single_token_answers"]
    countries = [(c, cap) for c, cap in CAPITALS if not single or tok.single(" " + cap)]
    if len(countries) < 2:
        raise TaskError(
            "Fewer than two of the built-in capitals are single tokens for this model. Turn off "
            "single-token answers and read the capitals with a log-probability metric, or import "
            "a JSONL dataset written for this model."
        )
    # Per template: each country whose prompt has as many tokens as another country's, and those
    # countries, which can replace it in the corrupt prompt.
    entries: list[tuple[TaskTemplate, list[tuple[str, str, list[tuple[str, str]]]]]] = []
    for template in templates:
        counted = [
            (c, cap, tok.count(_fill(template.text, {"COUNTRY": c})[0])) for c, cap in countries
        ]
        usable = []
        for country, capital, count in counted:
            partners = [(c, cap) for c, cap, k in counted if k == count and c != country]
            if partners:
                usable.append((country, capital, partners))
        entries.append((template, usable))
    space = sum(len(usable) for _, usable in entries)
    if space == 0:
        raise TaskError(
            "No two countries make prompts of the same token length for this model. Import a "
            "JSONL dataset written for this model instead."
        )
    more = len(templates) < len(FACTUAL_TEMPLATES)
    _check_space(n, space, more)
    rng = random.Random(seed)

    def make(index: int) -> PromptRecord | None:
        template, usable = _pick(rng, entries)
        if not usable:
            return None
        country, capital, partners = _pick(rng, usable)
        other, other_capital = _pick(rng, partners)
        clean, spans = _fill(template.text, {"COUNTRY": country})
        corrupt, _ = _fill(template.text, {"COUNTRY": other})
        return PromptRecord(
            clean=clean,
            corrupt=corrupt,
            answer=" " + capital,
            distractor=" " + other_capital,
            positions={"country": spans["COUNTRY"], "end": _last_word(clean)},
            id=f"factual-recall-{index:05d}",
            meta={"template": template.id, "country": country},
        )

    return _collect(n, make, more)


# ---------------------------------------------------------------------------------------------
# The registry

_SHARED_STRUCTURE = (
    "Sentence templates to fill. The default ones share one token structure, so their prompts "
    "line up token by token."
)


def _templates_option(templates: tuple[TaskTemplate, ...], description: str) -> TaskOption:
    return TaskOption(
        name="templates",
        type="choices",
        default=tuple(t.id for t in templates if t.default),
        description=description,
        allowed=tuple(t.id for t in templates),
    )


TASKS: dict[str, TaskInfo] = {
    task.id: task
    for task in (
        TaskInfo(
            id="ioi",
            name="Indirect object identification",
            description=(
                'The model must complete "When Mary and John went to the store, John gave a '
                'drink to" with " Mary", the name that isn\'t repeated, as in Wang et al. 2022, '
                '"Interpretability in the wild". The corrupt prompt repeats the other name '
                "instead (flip), or uses three new names (abc)."
            ),
            metric="logit_diff",
            options=(
                TaskOption(
                    name="templates",
                    type="choices",
                    default=tuple(default_template_ids()),
                    description=_SHARED_STRUCTURE,
                    allowed=tuple(t.id for t in IOI_TEMPLATES),
                ),
                TaskOption(
                    name="patterns",
                    type="choices",
                    default=("ABBA", "BABA"),
                    description=(
                        "Whether the indirect object is named first (ABBA) or second (BABA). "
                        "Prompts alternate between the chosen patterns."
                    ),
                    allowed=("ABBA", "BABA"),
                ),
                TaskOption(
                    name="corruption",
                    type="choice",
                    default="flip",
                    description=(
                        "flip: the subject's second mention becomes the indirect object's name. "
                        "abc: three unrelated names replace all three."
                    ),
                    allowed=("flip", "abc"),
                ),
            ),
            templates=tuple(TaskTemplate(t.id, t.text, t.default) for t in IOI_TEMPLATES),
            generate=_ioi,
        ),
        TaskInfo(
            id="greater_than",
            name="Greater-than",
            description=(
                'The model must continue "The war lasted from the year 1732 to the year 17" with '
                'a later year, as in Hanna et al. 2023, "How does GPT-2 compute greater-than?". '
                "The answer is the set of two-digit years after the start and the distractor the "
                "rest; the corrupt prompt starts in the year 01, so almost every year comes after "
                "it."
            ),
            metric="prob_diff",
            options=(_templates_option(GREATER_THAN_TEMPLATES, _SHARED_STRUCTURE),),
            templates=GREATER_THAN_TEMPLATES,
            generate=_greater_than,
        ),
        TaskInfo(
            id="docstring",
            name="Docstring",
            description=(
                "The model must name the next argument of a Python function in its docstring, by "
                'copying it from the definition, as in Heimersheim and Janiak 2023, "A circuit '
                'for Python docstrings in a 4-layer attention-only transformer". The corrupt '
                "prompt gives the definition other argument names, so the next one can't be "
                "copied from it."
            ),
            metric="logprob_diff",
            options=(
                _templates_option(
                    DOCSTRING_TEMPLATES,
                    "Function shapes: four_args documents two of four arguments, as in the "
                    "paper; three_args documents one of three, and five_args three of five. The "
                    "answer is always the next argument.",
                ),
            ),
            templates=DOCSTRING_TEMPLATES,
            generate=_docstring,
        ),
        TaskInfo(
            id="gendered_pronoun",
            name="Gendered pronoun",
            description=(
                'The model must choose the pronoun for a first name ("So Mary is a really great '
                'friend, isn\'t" → " she"), as in Mathwin et al. 2023 on gendered pronouns in '
                "GPT-2 small. The corrupt prompt uses a name that takes the other pronoun."
            ),
            metric="logit_diff",
            options=(_templates_option(PRONOUN_TEMPLATES, _SHARED_STRUCTURE),),
            templates=PRONOUN_TEMPLATES,
            generate=_gendered_pronoun,
        ),
        TaskInfo(
            id="subject_verb_agreement",
            name="Subject–verb agreement",
            description=(
                'The model must choose " is" or " are" to agree with the subject across a noun '
                'of the other number ("The author near the pilots" → " is"), as in Linzen et al. '
                "2016 and Finlayson et al. 2021. The corrupt prompt changes the subject's "
                "number, so the other verb is right."
            ),
            metric="logit_diff",
            options=(
                _templates_option(
                    AGREEMENT_TEMPLATES,
                    f"{_SHARED_STRUCTURE} In relative, the attractor is a person or an animal "
                    "that likes the subject.",
                ),
            ),
            templates=AGREEMENT_TEMPLATES,
            generate=_subject_verb_agreement,
        ),
        TaskInfo(
            id="factual_recall",
            name="Factual recall",
            description=(
                'The model must recall a country\'s capital ("The capital of France is" → '
                '" Paris"), as in factual-recall studies such as Meng et al. 2022, "Locating and '
                'editing factual associations in GPT". The corrupt prompt names another country, '
                "whose capital is the distractor; capitals of several tokens need a "
                "log-probability metric."
            ),
            metric="logit_diff",
            options=(
                _templates_option(FACTUAL_TEMPLATES, "Sentence templates to fill."),
                TaskOption(
                    name="single_token_answers",
                    type="bool",
                    default=True,
                    description=(
                        "Keep only countries whose capitals are single tokens, which the logit "
                        "difference can read. Turn it off to keep capitals of several tokens "
                        "too, read with the log-probability difference."
                    ),
                ),
            ),
            templates=FACTUAL_TEMPLATES,
            generate=_factual_recall,
            metric_when=(("single_token_answers", False, "logprob_diff"),),
        ),
    )
}


# ---------------------------------------------------------------------------------------------
# Checking settings and generating


def get_task(task_id: str) -> TaskInfo:
    """The task with this id, or a TaskError naming the tasks there are."""
    task = TASKS.get(task_id) if isinstance(task_id, str) else None
    if task is None:
        raise TaskError(f"Unknown task {task_id!r}. Choose one of: {', '.join(TASKS)}.")
    return task


def check_options(task_id: str, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """A task's options checked, with the defaults filled in for those not given (or None)."""
    return _check_options(get_task(task_id), options)


def _check_options(task: TaskInfo, options: Mapping[str, Any] | None) -> dict[str, Any]:
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise TaskError("Options must map option names to values.")
    known = [o.name for o in task.options]
    unknown = [str(name) for name in options if name not in known]
    if unknown:
        hint = f"Its options are {', '.join(known)}." if known else "It takes no options."
        raise TaskError(f"{task.name} has no option {', '.join(repr(u) for u in unknown)}. {hint}")
    checked: dict[str, Any] = {}
    for option in task.options:
        value = options.get(option.name)
        checked[option.name] = _check_value(option, option.default if value is None else value)
    return checked


def _check_value(option: TaskOption, value: Any) -> Any:
    if option.type == "bool":
        if not isinstance(value, bool):
            raise TaskError(f"{option.name} must be true or false.")
        return value
    if option.type == "choice":
        if not isinstance(value, str) or value not in option.allowed:
            raise TaskError(
                f"Unknown {option.name} {value!r}. Use one of: {', '.join(option.allowed)}."
            )
        return value
    item = option.name.removesuffix("s")
    if isinstance(value, str) or not isinstance(value, list | tuple):
        raise TaskError(f"{option.name} must be a list, such as [{option.allowed[0]!r}].")
    if not value:
        raise TaskError(f"Choose at least one {item}.")
    bad = [str(v) for v in value if not isinstance(v, str) or v not in option.allowed]
    if bad:
        raise TaskError(
            f"Unknown {item}(s): {', '.join(bad)}. Use any of: {', '.join(option.allowed)}."
        )
    repeated = [v for i, v in enumerate(value) if v in value[:i]]
    if repeated:
        raise TaskError(f"{repeated[0]!r} is listed twice. List each {item} once.")
    return list(value)


def generate_task(
    task_id: str,
    n: int,
    seed: int,
    options: Mapping[str, Any] | None = None,
    *,
    single_token: SingleToken | None = None,
    token_count: TokenCount | None = None,
) -> list[PromptRecord]:
    """Generate ``n`` prompt pairs for a task, the same for the same seed and settings.

    ``single_token(text)`` says whether ``text`` is one token for the model the dataset is for,
    and ``token_count(text)`` how many tokens it has (without a beginning-of-sequence token).
    Without them, every word counts as one token (see the module's description).
    """
    task = get_task(task_id)
    if isinstance(n, bool) or not isinstance(n, int):
        raise TaskError("Choose a whole number of prompts.")
    if n < 1:
        raise TaskError("Choose at least one prompt.")
    if n > MAX_PROMPTS:
        raise TaskError(f"Choose at most {MAX_PROMPTS:,} prompts.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < SEED_MAX:
        raise TaskError(f"Choose a seed from 0 to {SEED_MAX - 1:,}.")
    checked = _check_options(task, options)
    return task.generate(n, seed, checked, single_token, token_count)
