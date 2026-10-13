#!/usr/bin/env python3
"""Write the reference pages in docs/reference/ from the code itself:

* ``spec.md``: every field of the experiment spec, with its type, allowed values and description,
  from the pydantic models in ``src/logogram/spec.py`` (a string below a field describes it);
* ``training.md``: the same for the training specs of SAEs and tuned lenses trained in a project,
  from ``src/logogram/sae_training.py`` and ``src/logogram/lenses.py``;
* ``cli.md``: every command of the command line and its arguments and options, from typer
  (``src/logogram/cli.py``);
* ``python-api.md``: every public function and class of the Python API, from the docstrings in
  ``src/logogram/api.py``.

Run it after changing any of them, and commit what it writes::

    uv run python scripts/build_reference.py

``--check`` writes nothing and exits with status 1 when a committed page is out of date, as
``tests/test_reference.py`` does. The pages come out the same on every machine and Python version:
nothing in them depends on where or when they were made.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import re
import sys
import types
import typing
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "docs/reference"
COMMAND = "uv run python scripts/build_reference.py"
BLOB = "https://github.com/Jeevash23/logogram/blob/main"


def generated_note(source: str) -> str:
    return (
        f"<!-- Generated from {source} by scripts/build_reference.py. Don't edit it by hand: run "
        f"`{COMMAND}`. -->"
    )


# ---------------------------------------------------------------------------------------------
# Text


def slug(heading: str) -> str:
    """The anchor GitHub gives a heading: lower case, punctuation dropped, spaces as hyphens."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


class Anchors:
    """Headings of one page, each with the anchor GitHub gives it (repeats get -1, -2, ...)."""

    def __init__(self) -> None:
        self.seen: dict[str, int] = {}

    def add(self, heading: str) -> str:
        base = slug(heading)
        n = self.seen.get(base, 0)
        self.seen[base] = n + 1
        return base if n == 0 else f"{base}-{n}"


def rst_to_markdown(text: str, links: dict[str, str] | None = None) -> str:
    """A docstring's reStructuredText as Markdown: ``code`` as `code`, :func:`x` as a link to x
    when this page has it, and a paragraph ending in :: followed by an indented block as a fenced
    code block."""
    links = links or {}

    def role(match: re.Match[str]) -> str:
        kind, target = match.group(1), match.group(2).lstrip("~")
        name = target.rsplit(".", 1)[-1] if target.startswith("logogram.api.") else target
        shown = f"{name}()" if kind in ("func", "meth") else name
        anchor = links.get(name)
        return f"[`{shown}`](#{anchor})" if anchor else f"`{shown}`"

    def inline(prose: str) -> str:
        prose = re.sub(r":(func|meth|class|mod|attr|data|exc):`([^`]+)`", role, prose)
        return re.sub(
            r"``(.+?)``", lambda m: "`" + " ".join(m.group(1).split()) + "`", prose, flags=re.S
        )

    lines = inspect.cleandoc(text).split("\n")
    out: list[str] = []
    prose: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.rstrip().endswith("::") and i + 1 < len(lines) and not lines[i + 1].strip():
            head = line.rstrip()[:-2].rstrip()
            prose.append(head + ":" if head and not head.endswith(":") else head)
            out.append(inline("\n".join(prose)))
            prose = []
            block: list[str] = []
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or lines[j].startswith((" ", "\t"))):
                block.append(lines[j])
                j += 1
            while block and not block[-1].strip():
                block.pop()
            body = inspect.cleandoc("\n".join(block))
            out.extend(["", "```python", body, "```", ""])
            i = j
            continue
        prose.append(line)
        i += 1
    out.append(inline("\n".join(prose)))
    return "\n".join(out).strip()


def one_line(text: str) -> str:
    """A description for a table cell: one line, with its pipes escaped."""
    return " ".join(text.split()).replace("|", "\\|")


def first_sentence(text: str) -> str:
    text = " ".join(text.split())
    match = re.match(r"(.+?[.!?])(\s|$)", text)
    return match.group(1) if match else text


def number(value: float) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:g}"


def json_value(value: Any) -> str:
    return f"`{json.dumps(value)}`"


def either(items: list[str], word: str = "or") -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" {word} " + items[-1]


def jsonish(text: str, values: Iterable[str] = ()) -> str:
    """A description as the spec's JSON says things: Python's None, True and False as `null`,
    `true` and `false`, and these values of the field, where they are written as identifiers
    (dataset_gap, clean_to_corrupt), as code. Code spans stay as they are."""
    words = {"None": "null", "True": "true", "False": "false", "null": "null"}
    words |= {v: v for v in values if "_" in v}
    pattern = re.compile(r"(?<![\w`.])(" + "|".join(map(re.escape, words)) + r")(?![\w`])")
    parts = re.split(r"(`[^`]*`)", text)
    return "".join(
        part if part.startswith("`") else pattern.sub(lambda m: f"`{words[m.group(1)]}`", part)
        for part in parts
    )


# ---------------------------------------------------------------------------------------------
# The spec


def _unions() -> tuple[type, ...]:
    return (typing.Union, types.UnionType)


def _is_model(tp: Any) -> bool:
    from pydantic import BaseModel

    return inspect.isclass(tp) and issubclass(tp, BaseModel)


def _unwrap(tp: Any) -> tuple[Any, list[Any]]:
    """A type without its Annotated layers, and the constraints those layers held."""
    constraints: list[Any] = []
    while typing.get_origin(tp) is typing.Annotated:
        args = typing.get_args(tp)
        tp = args[0]
        for extra in args[1:]:
            constraints.extend(getattr(extra, "metadata", None) or [extra])
    return tp, constraints


def _members(tp: Any) -> tuple[list[Any], bool]:
    """A union's members and whether it allows null, unions in it flattened; a single type
    otherwise. Members that are models or unions are unwrapped from their Annotated layers."""
    tp, _ = _unwrap(tp)
    if typing.get_origin(tp) not in _unions():
        return [tp], False
    members: list[Any] = []
    nullable = False
    for arg in typing.get_args(tp):
        if arg is type(None):
            nullable = True
            continue
        inner, _ = _unwrap(arg)
        if typing.get_origin(inner) in _unions() or _is_model(inner):
            more, null = _members(inner)
            members.extend(more)
            nullable = nullable or null
        else:
            members.append(arg)
    return members, nullable


def _limits(constraints: Iterable[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in constraints:
        for key in ("ge", "gt", "le", "lt", "min_length", "max_length", "pattern"):
            value = getattr(c, key, None)
            if value is not None:
                out[key] = value
    return out


def _range(limits: dict[str, Any], integer: bool) -> str:
    low = limits.get("ge")
    high = limits.get("le")
    if integer:
        if low is None and "gt" in limits:
            low = limits["gt"] + 1
        if high is None and "lt" in limits:
            high = limits["lt"] - 1
        if low is not None and high is not None:
            return f"{number(low)} to {number(high)}"
        if low is not None:
            return f"{number(low)} or more"
        if high is not None:
            return f"{number(high)} or less"
        return ""
    parts = []
    if "ge" in limits:
        parts.append(f"at least {number(limits['ge'])}")
    if "gt" in limits:
        parts.append(f"more than {number(limits['gt'])}")
    if "le" in limits:
        parts.append(f"at most {number(limits['le'])}")
    if "lt" in limits:
        parts.append(f"less than {number(limits['lt'])}")
    return either(parts, "and")


def _length(limits: dict[str, Any], unit: str) -> str:
    low, high = limits.get("min_length"), limits.get("max_length")
    if low is not None and high is not None:
        return f"{number(low)} to {number(high)} {unit}s"
    if low is not None:
        return "not empty" if low == 1 else f"at least {number(low)} {unit}s"
    if high is not None:
        return f"at most {number(high)} {unit}s"
    return ""


class SpecPage:
    """The spec's models, each documented once, in the order a reader meets them from the top."""

    def __init__(self, roots: list[Any] | None = None, modules: list[Any] | None = None) -> None:
        import logogram.spec as spec

        self.module = spec
        # The models documented from the top (the experiment spec, or the training specs).
        self.roots = roots or [spec.Spec]
        self.root = self.roots[0]
        # Union aliases defined in spec.py (Position, Experiment, ...), by their members.
        self.alias: dict[frozenset[type], str] = {}
        self.unions: list[tuple[type, ...]] = []
        for module in modules or [spec]:
            for name, value in vars(module).items():
                if typing.get_origin(value) is not typing.Annotated:
                    continue
                members, _ = _members(value)
                models = len(members) > 1 and all(_is_model(m) for m in members)
                if models and frozenset(members) not in self.alias:
                    self.alias[frozenset(members)] = name
                    self.unions.append(tuple(members))
        self.order: list[Any] = []  # a model class, or a tuple of union members
        # Where each one appears: its paths in a spec, each with the kinds of the objects on the
        # way that have it (the experiment is an ablation, say).
        self.used_at: dict[Any, dict[str, list[tuple[str, str]]]] = {}
        self.anchors = Anchors()
        self.anchor: dict[Any, str] = {}
        self._walk()

    # -- the walk --------------------------------------------------------------------------------

    def _key(self, members: list[Any]) -> Any:
        """A model, or a union: the smallest named union that holds these members (a field that
        allows only some kinds of positions links to the positions' section)."""
        if len(members) == 1:
            return members[0]
        holding = [u for u in self.unions if set(members) <= set(u)]
        return min(holding, key=len) if holding else tuple(members)

    def _walk(self) -> None:
        queue: list[tuple[Any, str, list[tuple[str, str]]]] = [(r, "", []) for r in self.roots]
        while queue:
            key, path, when = queue.pop(0)
            if key not in self.used_at:
                self.order.append(key)
                self.used_at[key] = {}
            if path:
                kinds = self.used_at[key].setdefault(path, [])
                kinds.extend(w for w in when if w not in kinds)
            if path.count(".") > 12:  # the spec's objects don't nest that deep, or in a loop
                continue
            classes = list(key) if isinstance(key, tuple) else [key]
            for cls in classes:
                for name, info in cls.model_fields.items():
                    for nested, suffix in self._nested(info.annotation):
                        where = f"{path}.{name}{suffix}" if path else f"{name}{suffix}"
                        via = when
                        if isinstance(key, tuple):
                            via = [*when, (self.title(key).lower(), self.kind_of(cls))]
                        queue.append((nested, where, via))

    def _nested(self, tp: Any) -> list[tuple[Any, str]]:
        members, _ = _members(tp)
        if len(members) == 1 and typing.get_origin(members[0]) is list:
            inner, _ = _members(typing.get_args(members[0])[0])
            if all(_is_model(m) for m in inner):
                return [(self._key(inner), "[]")]
            return []
        if all(_is_model(m) for m in members):
            return [(self._key(members), "")]
        return []

    def where(self, key: Any) -> str:
        """Where it appears, such as `experiment.baseline` (when the experiment is ablation)."""
        out = []
        for path, kinds in self.used_at[key].items():
            text = f"`{path}`"
            by_union: dict[str, list[str]] = {}
            for union, kind in kinds:
                by_union.setdefault(union, []).append(f"`{kind}`")
            if by_union:
                text += " (" + "; ".join(f"{u} {either(k)}" for u, k in by_union.items()) + ")"
            out.append(text)
        return either(out, "and")

    # -- names -----------------------------------------------------------------------------------

    @staticmethod
    def kind_of(cls: type) -> str:
        field = cls.model_fields.get("kind")  # type: ignore[attr-defined]
        if field is not None:
            values = typing.get_args(_unwrap(field.annotation)[0])
            if len(values) == 1:
                return str(values[0])
        return cls.__name__

    def title(self, key: Any) -> str:
        if isinstance(key, tuple):
            name = self.alias.get(frozenset(key), "")
            if not name:
                return next(iter(self.used_at[key]), "One of several")
        else:
            name = key.__name__
            if name.endswith("Ref"):
                name = name[: -len("Ref")]
        words = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|\d+", name)
        return " ".join([words[0]] + [w if w.isupper() else w.lower() for w in words[1:]])

    def variant_title(self, union: tuple[type, ...], cls: type) -> str:
        return f"{self.title(union)}: {self.kind_of(cls)}"

    # -- types -----------------------------------------------------------------------------------

    def link(self, key: Any, text: str) -> str:
        return f"[{text}](#{self.anchor[key]})"

    def describe(self, info: Any, cls: type, union: bool) -> tuple[str, str]:
        """A field's type and its allowed values, as table cells."""
        constraints = list(info.metadata)
        members, nullable = _members(info.annotation)
        if len(members) == 1:
            constraints.extend(_unwrap(info.annotation)[1])
            inner, more = _unwrap(members[0])
            constraints.extend(more)
            members = [inner]
        kind, values = self._describe(members, constraints)
        if nullable:
            kind += " or null"
            values = f"{values}, or `null`" if values else ""
        return kind, values

    def _describe(self, members: list[Any], constraints: list[Any]) -> tuple[str, str]:
        limits = _limits(constraints)
        if all(_is_model(m) for m in members):
            key = self._key(members)
            if len(members) == 1:
                return "object", self.link(key, self.title(key))
            kinds = [self.link((key, m), f"`{self.kind_of(m)}`") for m in members]
            return "object", f"one of {either(kinds)} ({self.link(key, self.title(key))})"
        if len(members) != 1:
            kinds = [self._describe([m], [])[0] for m in members]
            return either(kinds), ""
        tp = members[0]
        if typing.get_origin(tp) is typing.Literal:
            values = typing.get_args(tp)
            kind = {str: "string", bool: "boolean", int: "integer"}.get(type(values[0]), "value")
            return kind, either([json_value(v) for v in values])
        if typing.get_origin(tp) is tuple:
            items = [self._describe([_unwrap(i)[0]], _unwrap(i)[1]) for i in typing.get_args(tp)]
            if len(set(items)) == 1:
                item_kind, item_values = items[0]
                plural = "objects" if item_kind == "object" else f"{item_kind}s"
                return (
                    f"list of {len(items)} {plural}",
                    f"each {item_values}" if item_values else "",
                )
            return "list", "; ".join(f"{k} {v}".strip() for k, v in items)
        if typing.get_origin(tp) is list:
            item = typing.get_args(tp)[0]
            item_members, _ = _members(item)
            item_kind, item_values = self._describe(
                [_unwrap(m)[0] for m in item_members], _unwrap(item)[1]
            )
            plural = "objects" if item_kind == "object" else f"{item_kind}s"
            count = _length(limits, "item")
            parts = [p for p in (count, f"each {item_values}" if item_values else "") if p]
            return f"list of {plural}", "; ".join(parts)
        if tp is bool:
            return "boolean", "`true` or `false`"
        if tp is int:
            return "integer", _range(limits, integer=True)
        if tp is float:
            return "number", _range(limits, integer=False)
        if tp is str:
            return "string", _length(limits, "character")
        return getattr(tp, "__name__", str(tp)), ""

    # -- the page --------------------------------------------------------------------------------

    def fields(self, cls: type, union: tuple[type, ...] | None) -> list[str]:
        rows = ["| Field | Type | Allowed values | Description |", "|---|---|---|---|"]
        for name, info in cls.model_fields.items():  # type: ignore[attr-defined]
            kind, values = self.describe(info, cls, union is not None)
            literals = [str(v) for v in self.literals(info.annotation)]
            description = jsonish(rst_to_markdown(info.description or ""), literals)
            if name == "kind" and union is not None:
                description = description or (
                    f"Which {self.title(union).lower()} this is: always `{self.kind_of(cls)}` here."
                )
            required = info.is_required() or (name == "kind" and union is not None)
            shown = (
                f"`{name}`"
                if required
                else f"`{name}` (optional, default {json_value(info.default)})"
            )
            rows.append(
                f"| {shown} | {kind} | {one_line(values) or 'any'} | {one_line(description)} |"
            )
        return rows

    @staticmethod
    def literals(tp: Any) -> list[Any]:
        """The values a field's type allows by name, in lists too."""
        members, _ = _members(tp)
        found: list[Any] = []
        for member in members:
            inner, _ = _unwrap(member)
            if typing.get_origin(inner) is list:
                found += SpecPage.literals(typing.get_args(inner)[0])
            elif typing.get_origin(inner) is typing.Literal:
                found += typing.get_args(inner)
        return found

    def docstring(self, cls: type) -> list[str]:
        doc = cls.__doc__ or ""
        if not doc or doc == inspect.getdoc(cls.__mro__[1]):
            return []
        return [jsonish(rst_to_markdown(doc)), ""]

    def render(self) -> str:
        # Anchors first, in page order, so links can point anywhere on the page.
        self.headings("The spec")

        package = Path(self.module.__file__ or "").parent
        example = package / "examples/ioi-gpt2/experiments/ioi-head-patching/spec.json"
        out = [
            generated_note("src/logogram/spec.py"),
            "",
            "# The spec",
            "",
            "Every experiment is a `spec.json`: the single description of an experiment, which the "
            "app, `logogram run` and the Python API all run the same way. Nothing that can change a "
            "number is left implicit. A spec states every such field, and a spec that leaves one out "
            "is refused with the field's name. This page lists every field, from the models in "
            f"[`src/logogram/spec.py`]({BLOB}/src/logogram/spec.py).",
            "",
            f"This is version {self.module.SPEC_VERSION}. Version 1 specs (Logogram 0.1) still run: "
            "each choice one leaves out takes the value version 1 gave it, the run's first warning "
            "lists those values, and `logogram validate SPEC.json` shows them without running "
            "anything. Version 2 specs (Logogram 0.2 to 0.4) read with what version 3 added turned "
            "off, which is what they did: no chat template, a model from Hugging Face, and steering "
            "along the mean difference without continuations. Version 3 specs (Logogram 0.5 and "
            "0.6) read as version 4 with what version 4 adds turned off: the SAE a spec names is "
            "a published one, and the layer predictions read the logit lens. Version 4 also adds "
            "methods and sources that a version 3 spec can't name.",
            "",
            "Objects that come in several kinds (an experiment, a scope, a position, ...) say which "
            "with their `kind`, and each kind has fields of its own.",
            "",
            "The bundled IOI example's spec:",
            "",
            "```json",
            json.dumps(json.loads(example.read_text(encoding="utf-8")), indent=2),
            "```",
            "",
        ]
        out += self.sections("a spec")
        return "\n".join(out).rstrip() + "\n"

    def headings(self, page: str) -> None:
        """Every heading's anchor, in page order, so links can point anywhere on the page."""
        self.anchors.add(page)
        headings: list[tuple[Any, str]] = []
        for key in self.order:
            headings.append((key, self.title(key)))
            if isinstance(key, tuple):
                headings.extend(((key, m), self.variant_title(key, m)) for m in key)
        for key, title in headings:
            self.anchor[key] = self.anchors.add(title)

    def sections(self, where: str) -> list[str]:
        """Each model's section: where it appears in ``where`` (a spec), and its fields."""
        out: list[str] = []
        for key in self.order:
            out += [f"## {self.title(key)}", ""]
            if self.used_at[key]:
                out += [f"In {where} at {self.where(key)}.", ""]
            if isinstance(key, tuple):
                kinds = [self.link((key, m), f"`{self.kind_of(m)}`") for m in key]
                out += [f"One of these, chosen by `kind`: {either(kinds)}.", ""]
                for member in key:
                    out += [f"### {self.variant_title(key, member)}", ""]
                    out += self.docstring(member)
                    out += [*self.fields(member, key), ""]
            else:
                out += self.docstring(key)
                out += [*self.fields(key, None), ""]
        return out


def spec_page() -> str:
    return SpecPage().render()


def training_page() -> str:
    """The training specs of SAEs and tuned lenses trained in a project, documented as the
    experiment spec is."""
    import logogram.lenses as lenses
    import logogram.sae_training as sae_training

    page = SpecPage(
        roots=[sae_training.SAETrainingSpec, lenses.LensTrainingSpec],
        modules=[sae_training, lenses],
    )
    page.headings("Training specs")
    out = [
        generated_note("src/logogram/sae_training.py and src/logogram/lenses.py"),
        "",
        "# Training specs",
        "",
        "An SAE or a tuned lens trained in a project is trained from a training spec, which the "
        "app, `logogram sae train` and `logogram lens train`, and the Python API "
        "(`lg.train_sae`, `lg.train_lens`) all train the same way. As in an experiment spec, every "
        "choice that can change the weights is stated, with no default, and a spec that leaves one "
        "out is refused with the field's name. The spec as it ran (with the corpus's hash and the "
        "model's revision pinned) is saved beside what it trained, as training.json. This page "
        f"lists every field, from the models in [`src/logogram/sae_training.py`]({BLOB}/src/"
        f"logogram/sae_training.py) and [`src/logogram/lenses.py`]({BLOB}/src/logogram/lenses.py). "
        "What is computed is described in [Training an SAE](../concepts/sae-training.md) and [The "
        "tuned lens](../concepts/tuned-lens.md).",
        "",
    ]
    out += page.sections("a training spec")
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------------------------
# The command line


def _value_kind(param: Any) -> str:
    """What an option or argument takes, in words: never a library's own name for it."""
    t = param.type
    name = type(t).__name__
    if "Choice" in name:
        return either([f"`{c}`" for c in getattr(t, "choices", [])])
    if "Bool" in name:
        return ""
    low, high = getattr(t, "min", None), getattr(t, "max", None)
    if "Int" in name:
        kind = "integer"
    elif "Float" in name:
        kind = "number"
    elif "Path" in name or "File" in name:
        if getattr(t, "dir_okay", True) is False:
            kind = "file"
        elif getattr(t, "file_okay", True) is False:
            kind = "folder"
        else:
            kind = "path"
    else:
        kind = "text"
    if low is not None and high is not None:
        kind += f", {number(low)} to {number(high)}"
    elif low is not None:
        kind += f", {number(low)} or more"
    return kind


def _default(param: Any) -> str:
    if param.required:
        return "required"
    default = param.default
    if getattr(param, "is_flag", False) and isinstance(default, bool):
        return "on" if default else "off"
    if isinstance(default, bool | int | float) or (isinstance(default, str) and default):
        return f"`{default}`"
    return "—"


def _usage(path: str, command: Any) -> str:
    arguments = []
    for param in command.params:
        if type(param).__name__.endswith("Argument"):
            name = param.metavar or param.name.upper()
            name += "..." if param.multiple or getattr(param, "nargs", 1) == -1 else ""
            arguments.append(name if param.required else f"[{name}]")
    options = any(type(p).__name__.endswith("Option") for p in command.params)
    return " ".join([path, *(["[OPTIONS]"] if options else []), *arguments])


def _help(command: Any) -> str:
    text = inspect.cleandoc(command.help or "")
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    return "\n\n".join(paragraphs)


def _declared_help(command: Any) -> dict[str, str]:
    """The help each parameter of a command declares in its typer.Argument or typer.Option, by
    parameter name. Read from the code rather than from the command line library, which loses an
    argument's help at some versions of typer and click."""
    func = getattr(command.callback, "__wrapped__", command.callback)
    try:
        hints = typing.get_type_hints(func, include_extras=True)
    except (NameError, TypeError, AttributeError):
        return {}
    found: dict[str, str] = {}
    for name, hint in hints.items():
        if typing.get_origin(hint) is typing.Annotated:
            for extra in typing.get_args(hint)[1:]:
                text = getattr(extra, "help", None)
                if isinstance(text, str) and text:
                    found[name] = text
    return found


def _param_help(param: Any, declared: dict[str, str]) -> str:
    return one_line(declared.get(param.name or "") or getattr(param, "help", None) or "")


def _commands(group: Any, path: str) -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    for name in sorted(getattr(group, "commands", {})):
        command = group.commands[name]
        if getattr(command, "hidden", False):
            continue
        full = f"{path} {name}"
        found.append((full, command))
        found.extend(_commands(command, full))
    return found


def cli_page() -> str:
    import typer.main

    from logogram.cli import app

    group = typer.main.get_command(app)
    commands = [("logogram", group), *_commands(group, "logogram")]
    anchors = Anchors()
    anchor = {path: anchors.add(path) for path, _ in commands}

    callback = getattr(getattr(app, "registered_callback", None), "callback", None)
    start = inspect.getdoc(callback) if callback else None
    out = [
        generated_note("src/logogram/cli.py"),
        "",
        "# The command line",
        "",
        f"{_help(group)} Every command and option below is read from "
        f"[`src/logogram/cli.py`]({BLOB}/src/logogram/cli.py).",
        "",
        "Commands that read runs use the project in the current folder or above it, or "
        "`--project PATH`; a run is its id (its folder's name in `experiments/`) or its folder. "
        "`logogram COMMAND --help` prints the same as this page, for one command.",
        "",
        "| Command | What it does |",
        "|---|---|",
    ]
    for path, command in commands:
        summary = start if command is group and start else _help(command)
        if getattr(command, "commands", None) and command is not group:
            summary = _help(command)
        out.append(f"| [`{path}`](#{anchor[path]}) | {one_line(first_sentence(summary))} |")
    out.append("")

    for path, command in commands:
        out += [f"## {path}", ""]
        text = start if command is group and start else _help(command)
        if text:
            out += [text, ""]
        if getattr(command, "commands", None) and command is not group:
            subs = [f"[`{p}`](#{anchor[p]})" for p, _ in commands if p.startswith(path + " ")]
            out += [f"Its commands: {either(subs, 'and')}.", ""]
            # A group that does something of its own without a command (lists, say) has options.
            if not any(type(p).__name__.endswith("Option") for p in command.params):
                continue
        out += ["```text", _usage(path, command), "```", ""]
        declared = _declared_help(command)
        arguments = [p for p in command.params if type(p).__name__.endswith("Argument")]
        options = [
            p
            for p in command.params
            if type(p).__name__.endswith("Option") and not getattr(p, "hidden", False)
        ]
        if arguments:
            out += ["| Argument | Takes | Description |", "|---|---|---|"]
            for p in arguments:
                name = p.metavar or p.name.upper()
                out.append(f"| `{name}` | {_value_kind(p)} | {_param_help(p, declared)} |")
            out.append("")
        if options:
            out += ["| Option | Takes | Default | Description |", "|---|---|---|---|"]
            for p in options:
                names = " / ".join(f"`{o}`" for o in [*p.opts, *p.secondary_opts])
                takes = _value_kind(p)
                if p.multiple:
                    takes = f"{takes}; repeat it for more" if takes else "repeat it for more"
                description = _param_help(p, declared)
                out.append(f"| {names} | {takes or 'nothing'} | {_default(p)} | {description} |")
            out.append("")
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------------------------
# The Python API


def _annotation(value: Any) -> str:
    if value is inspect.Parameter.empty:
        return ""
    if isinstance(value, str):
        return value
    return inspect.formatannotation(value)


def _signature(name: str, func: Callable[..., Any], skip_self: bool = False) -> str:
    """A signature as it is written in the code, one parameter per line when it is long."""
    signature = inspect.signature(func)
    parts: list[str] = []
    star = False
    for param in signature.parameters.values():
        if skip_self and param.name in ("self", "cls"):
            skip_self = False
            continue
        if param.kind is param.KEYWORD_ONLY and not star:
            parts.append("*")
            star = True
        if param.kind is param.VAR_POSITIONAL:
            star = True
            text = f"*{param.name}"
        elif param.kind is param.VAR_KEYWORD:
            text = f"**{param.name}"
        else:
            text = param.name
        annotation = _annotation(param.annotation)
        if annotation:
            text += f": {annotation}"
        if param.default is not param.empty:
            text += f" = {param.default!r}" if annotation else f"={param.default!r}"
        parts.append(text)
    returns = _annotation(signature.return_annotation)
    tail = f" -> {returns}" if returns else ""
    flat = f"{name}({', '.join(parts)}){tail}"
    if len(flat) <= 92:
        return flat
    body = "".join(f"    {p},\n" for p in parts)
    return f"{name}(\n{body}){tail}"


def _doc(obj: Any, links: dict[str, str]) -> str:
    doc = inspect.getdoc(obj)
    return rst_to_markdown(doc, links) if doc else "Not described yet."


def _members_of(cls: type) -> list[tuple[str, Any]]:
    """A class's own public properties and methods, in the order the code defines them."""
    found = []
    for name, value in vars(cls).items():
        if name.startswith("_"):
            continue
        if isinstance(value, property) or inspect.isfunction(value):
            found.append((name, value))
    return found


def api_page() -> str:
    import logogram
    from logogram import api

    names = list(api.__all__)
    functions = [n for n in names if inspect.isfunction(getattr(api, n))]
    classes = [n for n in names if inspect.isclass(getattr(api, n))]
    anchors = Anchors()
    links: dict[str, str] = {}
    headings: list[tuple[str, str]] = []
    for n in functions:
        headings.append((n, f"lg.{n}"))
    for n in classes:
        headings.append((n, f"lg.{n}"))
        headings.extend(
            (f"{n}.{member}", f"{n}.{member}") for member, _ in _members_of(getattr(api, n))
        )
    headings.append(("LogogramError", "lg.api.LogogramError"))
    for key, title in [("Functions", "Functions"), ("Classes", "Classes"), ("Errors", "Errors")]:
        links[key] = anchors.add(title)
    for key, title in headings:
        links[key] = anchors.add(title)

    module = rst_to_markdown(inspect.getdoc(api) or "", links)
    missing = sorted(set(logogram._API) ^ set(names))
    out = [
        generated_note("src/logogram/api.py"),
        "",
        "# The Python API",
        "",
        module,
        "",
        f"Every public name of `import logogram as lg`, from the docstrings in "
        f"[`src/logogram/api.py`]({BLOB}/src/logogram/api.py). Importing `logogram` loads no "
        "PyTorch; each function loads what it needs on first use.",
        "",
    ]
    if missing:  # The two lists must agree; the test says so if they don't.
        out += [
            f"Not exported the same way by `logogram` and `logogram.api`: {', '.join(missing)}",
            "",
        ]
    out += ["## Functions", "", "| Function | What it does |", "|---|---|"]
    for n in functions:
        summary = first_sentence(rst_to_markdown(inspect.getdoc(getattr(api, n)) or "", links))
        out.append(f"| [`lg.{n}`](#{links[n]}) | {one_line(summary)} |")
    out.append("")
    for n in functions:
        func = getattr(api, n)
        out += [f"### lg.{n}", "", "```python", _signature(f"lg.{n}", func), "```", ""]
        out += [_doc(func, links), ""]
    out += ["## Classes", ""]
    for n in classes:
        cls = getattr(api, n)
        out += [f"### lg.{n}", "", _doc(cls, links), ""]
        if dataclasses.is_dataclass(cls):
            attributes = [f"`{f.name}` ({_annotation(f.type)})" for f in dataclasses.fields(cls)]
            out += [f"Its attributes: {either(attributes, 'and')}.", ""]
        for member, value in _members_of(cls):
            out += [f"#### {n}.{member}", ""]
            if isinstance(value, property):
                returns = _annotation(inspect.signature(value.fget).return_annotation)  # type: ignore[arg-type]
                out += [
                    "```python",
                    f"{n.lower()}.{member}" + (f"  # {returns}" if returns else ""),
                    "```",
                    "",
                ]
                out += [_doc(value.fget, links), ""]
            else:
                out += ["```python", _signature(f"{n.lower()}.{member}", value, True), "```", ""]
                out += [_doc(value, links), ""]
    out += [
        "## Errors",
        "",
        "### lg.api.LogogramError",
        "",
        _doc(api.LogogramError, links),
        "",
    ]
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------------------------


def pages() -> dict[str, str]:
    """Each reference page's path in the repository, and what it should hold."""
    return {
        f"{REFERENCE}/spec.md": spec_page(),
        f"{REFERENCE}/training.md": training_page(),
        f"{REFERENCE}/cli.md": cli_page(),
        f"{REFERENCE}/python-api.md": api_page(),
    }


def stale(written: dict[str, str] | None = None) -> list[str]:
    """The pages whose committed text differs from what the code says now."""
    written = pages() if written is None else written
    out = []
    for path, text in written.items():
        file = ROOT.joinpath(*path.split("/"))
        try:
            current = file.read_text(encoding="utf-8")
        except OSError:
            current = None
        if current != text:
            out.append(path)
    return out


def main(argv: list[str]) -> int:
    written = pages()
    if "--check" in argv:
        out_of_date = stale(written)
        if out_of_date:
            print(f"Out of date: {', '.join(out_of_date)}. Run `{COMMAND}` and commit the result.")
            return 1
        print("The reference pages are up to date.")
        return 0
    for path, text in written.items():
        file = ROOT.joinpath(*path.split("/"))
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
