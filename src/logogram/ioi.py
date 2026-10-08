"""Indirect object identification (IOI) prompt generator.

Each clean prompt names two people, repeats one of them (the subject, S), and ends where the
other (the indirect object, IO) is the natural next word::

    ABBA: When Mary and John went to the store, John gave a drink to   -> " Mary"
    BABA: When John and Mary went to the store, John gave a drink to   -> " Mary"

The answer is the IO name and the distractor is the S name. Two corruptions are available:

* ``flip``: the second mention of the subject (S2) becomes the IO name, so the model should now
  prefer the other name. Corrupt prompts have a negative logit difference.
* ``abc``: all three names are replaced by unrelated names, so neither answer is supported.

Named positions (IO, S1, S2, end) are recorded as character spans in the clean prompt.
Randomness uses only ``random.Random.random()``, whose sequence Python guarantees across versions.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from logogram.datasets import PromptRecord


@dataclass(frozen=True)
class IOITemplate:
    id: str
    text: str
    default: bool = False


# The default templates share one token structure, so named positions sit at the same index.
TEMPLATES: tuple[IOITemplate, ...] = (
    IOITemplate("went", "When {A} and {B} went to the {PLACE}, {C} gave a {OBJECT} to", True),
    IOITemplate(
        "arrived", "After {A} and {B} arrived at the {PLACE}, {C} handed a {OBJECT} to", True
    ),
    IOITemplate("got", "When {A} and {B} got to the {PLACE}, {C} brought a {OBJECT} to", True),
    IOITemplate(
        "working", "While {A} and {B} were working at the {PLACE}, {C} passed a {OBJECT} to"
    ),
    IOITemplate(
        "walked", "Then {A} and {B} walked into the {PLACE}, and {C} offered a {OBJECT} to"
    ),
    IOITemplate("reached", "Once {A} and {B} reached the {PLACE}, {C} threw a {OBJECT} to"),
    IOITemplate("left", "As {A} and {B} left the {PLACE}, {C} gave the {OBJECT} to"),
    IOITemplate("met", "Later, {A} and {B} met at the {PLACE}, where {C} gave a {OBJECT} to"),
)

NAMES: tuple[str, ...] = (
    "Mary", "John", "Alice", "Bob", "Tom", "James", "Sarah", "Emma", "David", "Michael",
    "Daniel", "Laura", "Anna", "Kate", "Paul", "Lisa", "Peter", "Jack", "Rachel", "Eric",
    "Susan", "Sam", "Ben", "Amy", "Henry", "Lucy", "Adam", "Emily", "Kevin", "Linda", "Ryan",
    "Megan", "Jason", "Helen", "Chris", "Nancy", "Brian", "Karen", "George", "Ruth", "Frank",
    "Steve", "Jane", "Matt", "Claire", "Scott", "Diana", "Tim", "Julia", "Andrew", "Victoria",
    "Robert", "Joseph", "Jennifer", "Thomas", "Charles", "Jessica", "Anthony", "Nicole",
    "Joshua", "Amanda", "Justin", "Kelly", "Sean", "Hannah", "Carl", "Fiona", "Leo", "Nora",
    "Ivan", "Olivia", "Jake", "Luke", "Zoe", "Noah", "Sophie", "Oscar", "Isaac", "Vera", "Hugo",
    "Owen", "Alex",
)  # fmt: skip

PLACES: tuple[str, ...] = (
    "store", "park", "school", "hospital", "station", "beach", "office", "restaurant", "market",
    "library", "garden", "museum", "airport", "kitchen", "church", "bank", "house", "cafe",
    "theater", "zoo", "mall", "hotel", "lake", "river", "forest", "farm", "gym", "club",
)  # fmt: skip

OBJECTS: tuple[str, ...] = (
    "drink", "book", "ring", "bag", "snack", "letter", "basket", "bottle", "gift", "card", "key",
    "ball", "cake", "flower", "pen", "hat", "box", "ticket", "necklace", "computer", "phone",
    "coat", "bowl", "shirt", "toy", "cup", "map", "plate",
)  # fmt: skip

Pattern = Literal["ABBA", "BABA"]
Corruption = Literal["flip", "abc"]


class IOIError(ValueError):
    pass


def default_template_ids() -> list[str]:
    return [t.id for t in TEMPLATES if t.default]


def _pick(rng: random.Random, items: Sequence[str]) -> str:
    return items[int(rng.random() * len(items))]


def _pick_distinct(rng: random.Random, items: Sequence[str], k: int, avoid: set[str]) -> list[str]:
    pool = [x for x in items if x not in avoid]
    if len(pool) < k:
        raise IOIError(f"Need at least {k} usable names but only {len(pool)} are available.")
    chosen: list[str] = []
    while len(chosen) < k:
        candidate = _pick(rng, pool)
        if candidate not in chosen:
            chosen.append(candidate)
    return chosen


def _fill(template: str, values: dict[str, str]) -> tuple[str, dict[str, tuple[int, int]]]:
    """Fill ``{A}``-style slots left to right, returning the text and each slot's span."""
    out: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    cursor = 0
    length = 0
    while cursor < len(template):
        start = template.find("{", cursor)
        if start == -1:
            out.append(template[cursor:])
            length += len(template) - cursor
            break
        out.append(template[cursor:start])
        length += start - cursor
        end = template.index("}", start)
        slot = template[start + 1 : end]
        value = values[slot]
        spans[slot] = (length, length + len(value))
        out.append(value)
        length += len(value)
        cursor = end + 1
    return "".join(out), spans


def generate_ioi(
    n: int,
    seed: int = 0,
    templates: Sequence[str] | None = None,
    patterns: Sequence[Pattern] = ("ABBA", "BABA"),
    corruption: Corruption = "flip",
    single_token: Callable[[str], bool] | None = None,
) -> list[PromptRecord]:
    """Generate ``n`` IOI prompt pairs.

    ``single_token``, if given, filters names, places and objects to words that are a single
    token (with a leading space) for the model that will be used.
    """
    if n < 1:
        raise IOIError("Choose at least one prompt.")
    if n > 100_000:
        raise IOIError("Choose at most 100,000 prompts.")
    chosen_ids = list(templates) if templates else default_template_ids()
    by_id = {t.id: t for t in TEMPLATES}
    unknown = [t for t in chosen_ids if t not in by_id]
    if unknown:
        raise IOIError(f"Unknown template(s): {', '.join(unknown)}.")
    if not patterns:
        raise IOIError("Choose at least one of ABBA and BABA.")
    for p in patterns:
        if p not in ("ABBA", "BABA"):
            raise IOIError(f"Unknown pattern {p!r}; use ABBA or BABA.")
    if corruption not in ("flip", "abc"):
        raise IOIError(f"Unknown corruption {corruption!r}; use flip or abc.")

    def usable(words: Sequence[str]) -> list[str]:
        if single_token is None:
            return list(words)
        return [w for w in words if single_token(" " + w)]

    names, places, objects = usable(NAMES), usable(PLACES), usable(OBJECTS)
    if len(names) < 5 or not places or not objects:
        raise IOIError(
            "This model's tokenizer splits too many of the built-in names, places or objects "
            "into several tokens. Import a JSONL dataset written for this model instead."
        )

    rng = random.Random(seed)
    records: list[PromptRecord] = []
    seen: set[str] = set()
    attempts = 0
    while len(records) < n:
        attempts += 1
        if attempts > n * 50:
            raise IOIError(
                f"Could only make {len(records)} distinct prompts with these settings. "
                "Choose more templates or fewer prompts."
            )
        index = len(records)
        template = by_id[chosen_ids[int(rng.random() * len(chosen_ids))]]
        pattern = patterns[index % len(patterns)]
        io, s = _pick_distinct(rng, names, 2, set())
        place, obj = _pick(rng, places), _pick(rng, objects)
        a, b = (io, s) if pattern == "ABBA" else (s, io)
        clean, spans = _fill(template.text, {"A": a, "B": b, "C": s, "PLACE": place, "OBJECT": obj})
        if clean in seen:
            continue
        if corruption == "flip":
            corrupt, _ = _fill(
                template.text, {"A": a, "B": b, "C": io, "PLACE": place, "OBJECT": obj}
            )
        else:
            x, y, z = _pick_distinct(rng, names, 3, {io, s})
            corrupt, _ = _fill(
                template.text, {"A": x, "B": y, "C": z, "PLACE": place, "OBJECT": obj}
            )
        seen.add(clean)
        io_slot, s1_slot = ("A", "B") if pattern == "ABBA" else ("B", "A")
        end_start = clean.rfind(" ") + 1
        records.append(
            PromptRecord(
                clean=clean,
                corrupt=corrupt,
                answer=" " + io,
                distractor=" " + s,
                positions={
                    "IO": spans[io_slot],
                    "S1": spans[s1_slot],
                    "S2": spans["C"],
                    "end": (end_start, len(clean)),
                },
                id=f"ioi-{index:05d}",
                meta={"template": template.id, "pattern": pattern, "corruption": corruption},
            )
        )
    return records
