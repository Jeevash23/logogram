"""The examples bundled with Logogram: experiments on GPT-2 small whose results can be read against
the literature.

Each example is a project folder in ``logogram/examples`` holding one version 2 spec, which states
every choice, and the dataset it runs on, made with Logogram's own seeded task generators. The
spec's notes say what the experiment asks, what was published about it, and what Logogram measured
on GPT-2 small, and how the two compare. The gallery adds the question and the result to expect,
in a line each.

Opening an example copies it into the projects folder, one folder per example. A copy made before
is opened as it is: Logogram never overwrites it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from logogram.datasets import load_dataset
from logogram.fileio import JSON_LIMIT, read_limited
from logogram.project import (
    PROJECT_FILE,
    ProjectError,
    ProjectMeta,
    default_projects_parent,
    example_source,
)
from logogram.spec import Spec

# The example "Open the example project" opened before there were several.
DEFAULT_EXAMPLE = "ioi-gpt2"


@dataclass(frozen=True)
class Example:
    """A bundled example. ``id`` names its folder in logogram/examples and ``folder`` the folder
    its copy gets in the projects folder. ``question``, ``expect`` and ``source`` are what the
    gallery shows beside the spec's model and method: what the experiment asks, the result to
    expect on GPT-2 small, and the publication it is read against.

    ``task``, ``n``, ``seed`` and ``options`` made its dataset with logogram.tasks, for GPT-2's
    tokenizer, as the app makes one once the model is loaded."""

    id: str
    folder: str
    question: str
    expect: str
    source: str
    task: str
    n: int
    seed: int
    options: Mapping[str, Any] = field(default_factory=dict)


EXAMPLES: tuple[Example, ...] = (
    Example(
        id="ioi-gpt2",
        folder="ioi-example",
        question="In “When Mary and John went to the store, John gave a drink to”, which attention "
        "heads carry the answer, Mary?",
        expect="The induction head 5.5 and the S-inhibition heads 8.6, 8.10 and 7.9 restore the most, "
        "a fifth to a third of the gap each, then the name movers 9.9 and 10.0; the negative name "
        "movers 10.7 and 11.10 work against the answer.",
        source="Wang et al. 2022, “Interpretability in the wild”",
        task="ioi",
        n=32,
        seed=0,
    ),
    Example(
        id="greater-than-gpt2",
        folder="greater-than-example",
        question="In “The war lasted from the year 1732 to the year 17”, which layers make GPT-2 "
        "small predict a year after 32?",
        expect="MLPs 9 and 10 carry the largest effects, about a third of the gap each, then MLP 8, "
        "attention in layer 9 and MLP 11; attention in layers 5 to 8 adds a little, and layers 0 "
        "to 4 nothing.",
        source="Hanna et al. 2023, “How does GPT-2 compute greater-than?”",
        task="greater_than",
        n=96,
        seed=0,
    ),
    Example(
        id="factual-recall-gpt2",
        folder="factual-recall-example",
        question="In “The capital of France is”, where does GPT-2 small turn the country into its "
        "capital?",
        expect="The country's last token restores the capital from the first layer through layer "
        "9; the last token takes over only in layers 10 and 11, the early and the late site of "
        "causal tracing.",
        source="Meng et al. 2022, “Locating and editing factual associations in GPT”",
        task="factual_recall",
        n=96,
        seed=0,
    ),
    Example(
        id="gendered-pronoun-gpt2",
        folder="gendered-pronoun-example",
        question="In “So John is a really great friend, isn't”, which attention heads write “he”?",
        expect="Heads 10.9 and 9.7 write the most toward the right pronoun, about a third and a "
        "fifth of the logit difference; every other head a tenth or less.",
        source="Mathwin et al. 2023, as described in Conmy et al. 2023",
        task="gendered_pronoun",
        n=64,
        seed=0,
    ),
)


def get_example(example_id: str) -> Example:
    """The bundled example with this id, or a ProjectError naming the ones there are."""
    for example in EXAMPLES:
        if example.id == example_id:
            return example
    names = ", ".join(e.id for e in EXAMPLES)
    raise ProjectError(f"There is no example called {example_id!r}. Choose one of: {names}.")


def example_spec(example_id: str) -> tuple[str, Spec]:
    """The example's experiment: its folder in experiments/ and its spec."""
    folder = example_source(get_example(example_id).id) / "experiments"
    specs = sorted(folder.glob("*/spec.json"))
    if len(specs) != 1:
        raise ProjectError(f"The example {example_id!r} must hold exactly one experiment.")
    return specs[0].parent.name, Spec.from_path(specs[0])


def list_examples(parent: Path | None = None) -> list[dict[str, Any]]:
    """Every example as the gallery shows it, and where its copy goes in ``parent`` (the projects
    folder by default), saying whether a copy is there already: opening it opens that copy."""
    where = (parent or default_projects_parent()).expanduser()
    out = []
    for example in EXAMPLES:
        source = example_source(example.id)
        meta = ProjectMeta.model_validate_json(
            read_limited(source / PROJECT_FILE, JSON_LIMIT, "Reinstall Logogram.")
        )
        experiment, spec = example_spec(example.id)
        try:
            # As the project will name itself once open (its folder, with links followed).
            target = (where / example.folder).resolve()
            copied = (target / PROJECT_FILE).is_file()
        except OSError:  # a folder that can't be looked into: opening it says why
            target, copied = where / example.folder, False
        out.append(
            {
                "id": example.id,
                "name": meta.name,
                "question": example.question,
                "expect": example.expect,
                "source": example.source,
                "path": str(target),
                "copied": copied,
                "experiment": experiment,
                "n_prompts": len(load_dataset(source / spec.dataset.path)),
                "spec": spec.model_dump(mode="json"),
            }
        )
    return out
