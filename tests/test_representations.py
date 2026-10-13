"""Top-activating examples over a corpus, principal components, mean differences and exported
activations, on the tiny model: each checked against a direct computation, and run twice."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from logogram.corpora import read_corpus
from logogram.corpus_import import write_texts
from logogram.ioi import generate_ioi
from logogram.project import Project
from logogram.representations import (
    CorpusRows,
    DatasetRows,
    LoadedRows,
    MeanPosition,
    ReadSite,
    TextRow,
    collect,
    export_activations,
    load_rows,
    mean_difference,
    parse_position,
    parse_site,
    pca_report,
    principal_components,
    save_component,
    save_mean_difference,
)
from logogram.spec import IndexPosition, LabelPosition, LastPosition
from logogram.top_activations import (
    DirectionUnit,
    FeatureUnit,
    HeadUnit,
    NeuronUnit,
    _ranked,
    top_examples,
)


@pytest.fixture
def stories(project: Project) -> Any:
    """IOI sentences of several lengths as a corpus, labelled by their pattern."""
    records = generate_ioi(10, seed=3, templates=["went", "working", "reached"])
    texts = [r.clean for r in records]
    labels = [r.meta["pattern"] for r in records]  # type: ignore[index]
    return write_texts(project, "stories", texts, labels).corpus


def per_text(backend: Any, texts: list[str], site: tuple[str, int]) -> list[torch.Tensor]:
    """Each text's activations at ``site``, run on its own, after the BOS token."""
    out = []
    for text in texts:
        ids = torch.tensor([backend.tokenize(text, True).ids])
        out.append(backend.capture(ids, [site])[site][0].double())
    return out


def ranked(values: list[torch.Tensor], k: int, sign: float = 1.0) -> list[tuple[int, int]]:
    rows = [
        (-sign * float(v[pos]), t, pos) for t, v in enumerate(values) for pos in range(1, len(v))
    ]
    return [(t, pos) for _, t, pos in sorted(rows)[:k]]


def test_a_neurons_top_examples_are_the_largest_activations(tiny_backend, stories) -> None:
    settings = dict(texts=None, max_tokens=64, prepend_bos=True, top_k=5, context=3)
    unit = NeuronUnit(layer=1, neuron=7)
    # Each text in a batch of its own, as the direct computation runs it: the same numbers.
    one = top_examples(tiny_backend, stories, unit, batch_size=1, **settings)
    acts = [a[:, 7] for a in per_text(tiny_backend, stories.texts, ("neuron", 1))]
    assert [(e["text"], e["position"]) for e in one["top"]] == ranked(acts, 5)
    best = one["top"][0]
    assert best["value"] == float(acts[best["text"]][best["position"]])
    window = acts[best["text"]][best["start"] : best["start"] + len(best["tokens"])]
    shown = [v for v in best["values"] if v is not None]
    assert shown == [float(v) for v in window[1 if best["start"] == 0 else 0 :]]
    if best["start"] == 0:
        assert best["values"][0] is None  # the BOS token is shown, never counted
    assert one["bottom"] is None and not one["signed"]
    counted = torch.cat([a[1:] for a in acts])
    assert one["tokens"] == len(counted) and one["active"] == int((counted > 0).sum())
    assert sum(one["histogram"]["counts"]) == len(counted)
    assert one["settings"] == {**settings, "batch_size": 1}
    assert one["corpus"]["sha256"] == stories.sha256

    # Batched by length, the same examples; and the same numbers every time.
    batched = top_examples(tiny_backend, stories, unit, batch_size=4, **settings)
    assert [(e["text"], e["position"]) for e in batched["top"]] == ranked(acts, 5)
    assert batched == top_examples(tiny_backend, stories, unit, batch_size=4, **settings)


def test_a_direction_is_signed_and_its_most_negative_tokens_are_listed(
    tiny_backend, project, stories
) -> None:
    from logogram.directions import model_of, save_direction

    torch.manual_seed(0)
    vector = torch.randn(tiny_backend.info.d_model)
    saved = save_direction(
        project,
        "random",
        vector,
        model=model_of(tiny_backend.info),
        site={"kind": "resid_post", "layer": 0},
        method="import",
        run=None,
        detail="a test",
    )
    unit = DirectionUnit(name="random", sha256=saved.sha256)
    out = top_examples(
        tiny_backend,
        stories,
        unit,
        direction=saved,
        texts=6,
        max_tokens=64,
        prepend_bos=True,
        batch_size=1,
        top_k=4,
        context=2,
    )
    unit_vector = (vector / vector.norm()).double()
    acts = [a @ unit_vector for a in per_text(tiny_backend, stories.texts[:6], ("resid_post", 0))]
    assert out["signed"] and out["unit"]["site"] == {"kind": "resid_post", "layer": 0}
    assert [(e["text"], e["position"]) for e in out["top"]] == ranked(acts, 4)
    assert [(e["text"], e["position"]) for e in out["bottom"]] == ranked(acts, 4, -1.0)
    assert out["bottom"][0]["value"] < 0 and not out["notes"]
    assert out["texts_read"] == 6


def test_a_heads_value_is_the_length_of_what_it_writes(tiny_backend, stories) -> None:
    out = top_examples(
        tiny_backend,
        stories,
        HeadUnit(layer=1, head=2),
        texts=4,
        max_tokens=64,
        prepend_bos=True,
        batch_size=1,
        top_k=3,
        context=1,
    )
    w_o = tiny_backend.head_output_weights(1)[2].double().cpu()
    z = per_text(tiny_backend, stories.texts[:4], ("head", 1))
    norms = [(zi[:, 2] @ w_o).norm(dim=-1) for zi in z]
    assert [(e["text"], e["position"]) for e in out["top"]] == ranked(norms, 3)
    assert all(e["value"] >= 0 for e in out["top"])


def test_an_sae_features_activation_is_read_from_the_encoder(
    tiny_backend, stories, tmp_path
) -> None:
    from test_sae import exact_sae

    d = tiny_backend.info.d_model
    sae = exact_sae(tmp_path / "exact", d)  # feature i is relu(x_i) at layer 1's input
    out = top_examples(
        tiny_backend,
        stories,
        FeatureUnit(feature=3),
        sae=sae,
        texts=5,
        max_tokens=64,
        prepend_bos=True,
        batch_size=1,
        top_k=3,
        context=1,
    )
    acts = [
        a[:, 3].clamp(min=0) for a in per_text(tiny_backend, stories.texts[:5], ("resid_pre", 1))
    ]
    assert [(e["text"], e["position"]) for e in out["top"]] == ranked(acts, 3)
    assert out["unit"]["sae"]["repo"] == "local/exact"
    with pytest.raises(ValueError, match="no feature"):
        top_examples(
            tiny_backend,
            stories,
            FeatureUnit(feature=2 * d),
            sae=sae,
            texts=1,
            max_tokens=8,
            prepend_bos=True,
            batch_size=1,
            top_k=1,
            context=0,
        )


def test_ties_are_broken_by_text_then_token() -> None:
    values = np.array([1.0, 2.0, 2.0, 2.0, 0.5, 2.0])
    texts = np.array([0, 3, 1, 1, 0, 0])
    tokens = np.array([1, 1, 4, 2, 2, 7])
    assert list(_ranked(values, texts, tokens, 3)) == [5, 3, 2]
    assert list(_ranked(values, texts, tokens, 6)) == [5, 3, 2, 1, 0, 4]


def test_texts_are_cut_to_their_first_tokens_and_settings_are_checked(
    tiny_backend, stories
) -> None:
    out = top_examples(
        tiny_backend,
        stories,
        NeuronUnit(layer=0, neuron=0),
        texts=None,
        max_tokens=5,
        prepend_bos=True,
        batch_size=8,
        top_k=50,
        context=0,
    )
    assert out["truncated"] == len(stories.records) and out["tokens"] == 4 * len(stories.records)
    assert len(out["top"]) == 40 and all(e["position"] < 5 for e in out["top"])
    base = dict(texts=None, prepend_bos=True, batch_size=8, top_k=5, context=0)
    with pytest.raises(ValueError, match="reads at most 64 tokens"):
        top_examples(tiny_backend, stories, NeuronUnit(layer=0, neuron=0), max_tokens=65, **base)
    with pytest.raises(ValueError, match="no neuron 999"):
        top_examples(tiny_backend, stories, NeuronUnit(layer=0, neuron=999), max_tokens=8, **base)
    with pytest.raises(ValueError, match="Layer 5 isn't in this model"):
        top_examples(tiny_backend, stories, HeadUnit(layer=5, head=0), max_tokens=8, **base)


# -- principal components and mean differences ---------------------------------------------------


def test_principal_components_of_a_known_set() -> None:
    rng = np.random.default_rng(0)
    axis = np.array([0.6, -0.8, 0.0])  # its largest loading is negative: the sign flips it
    spread = rng.normal(size=(200, 1)) * 5.0
    noise = rng.normal(size=(200, 3)) * 0.1
    values = spread * axis + noise + np.array([10.0, 2.0, -3.0])
    found = principal_components(values)
    assert np.allclose(found.components[0], -axis, atol=0.01)
    assert found.components[0][np.argmax(np.abs(found.components[0]))] > 0
    assert found.explained[0] > 0.99 and np.isclose(found.explained.sum(), 1.0)
    assert np.allclose(found.coords.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(found.components @ found.components.T, np.eye(3), atol=1e-9)
    again = principal_components(values)
    assert np.array_equal(again.components, found.components)
    assert np.array_equal(again.coords, found.coords)
    with pytest.raises(ValueError, match="at least two"):
        principal_components(values[:1])
    with pytest.raises(ValueError, match="no variance"):
        principal_components(np.ones((4, 3)))


def _loaded(marks: list[str]) -> LoadedRows:
    from logogram.backends.base import Tokenized

    empty = Tokenized(ids=[0], tokens=[""], offsets=[(0, 0)])
    rows = [TextRow(i, None, m, "", empty) for i, m in enumerate(marks)]
    return LoadedRows(rows, {"kind": "corpus"})


def test_the_mean_difference_between_two_classes() -> None:
    values = np.array([[1.0, 0.0], [3.0, 2.0], [0.0, 1.0], [0.0, 3.0], [9.0, 9.0]])
    loaded = _loaded(["a", "a", "b", "b", "c"])
    vector, n_a, n_b = mean_difference(values, loaded, "a", "b")
    assert np.allclose(vector, [2.0, -1.0]) and (n_a, n_b) == (2, 2)
    with pytest.raises(ValueError, match="No prompt or text is marked 'd'"):
        mean_difference(values, loaded, "a", "d")


def test_principal_components_over_prompts_marked_by_side(tiny_backend, project) -> None:
    source = DatasetRows(path="datasets/ioi.jsonl", sides="both", mark="side", limit=6)
    loaded = load_rows(tiny_backend, project, source, True)
    assert [r.mark for r in loaded.rows[:2]] == ["clean", "corrupt"] and loaded.marks == [
        "clean",
        "corrupt",
    ]
    site = ReadSite(kind="resid_post", layer=1)
    report = pca_report(
        tiny_backend, loaded, site, LabelPosition(label="S2"), prepend_bos=True, batch_size=4
    )
    assert report["n"] == 12 and report["components"] == 10
    assert report["explained"] == sorted(report["explained"], reverse=True)
    assert report["source"]["sha256"] and report["site"] == {"kind": "resid_post", "layer": 1}
    assert len(report["points"][0]["coords"]) == 10 and report["points"][1]["side"] == "corrupt"
    # The same rows read one at a time, at the S2 token: the same components.
    rows = []
    for row in loaded.rows:
        ids = torch.tensor([row.tokens.ids])
        rows.append(
            tiny_backend.capture(ids, [("resid_post", 1)])[("resid_post", 1)][0, row.labels["S2"]]
        )
    direct = principal_components(torch.stack(rows).double().numpy())
    assert np.allclose(report["explained"], direct.explained, atol=1e-6)
    again = pca_report(
        tiny_backend, loaded, site, LabelPosition(label="S2"), prepend_bos=True, batch_size=4
    )
    assert again == report


def test_positions_are_read_or_refused_with_the_row(tiny_backend, project, stories) -> None:
    corpus = load_rows(tiny_backend, project, CorpusRows(name="stories", max_tokens=64), True)
    site = [ReadSite(kind="resid_pre", layer=0)]
    mean = collect(tiny_backend, corpus, site, MeanPosition(), prepend_bos=True, batch_size=3)
    text = stories.texts[0]
    acts = per_text(tiny_backend, [text], ("resid_pre", 0))[0]
    assert np.allclose(mean["resid_pre:0"][0], acts[1:].mean(dim=0).numpy())
    last = collect(
        tiny_backend, corpus, site, IndexPosition(index=-1), prepend_bos=True, batch_size=3
    )
    assert np.allclose(last["resid_pre:0"][0], acts[-1].numpy())
    with pytest.raises(ValueError, match="no labelled positions"):
        collect(
            tiny_backend, corpus, site, LabelPosition(label="S2"), prepend_bos=True, batch_size=3
        )
    with pytest.raises(ValueError, match=r"Text 0 has \d+ tokens, so token 99 isn't in it"):
        collect(tiny_backend, corpus, site, IndexPosition(index=99), prepend_bos=True, batch_size=3)
    with pytest.raises(ValueError, match="has no meta field 'colour'"):
        load_rows(
            tiny_backend,
            project,
            DatasetRows(path="datasets/ioi.jsonl", sides="clean", mark="meta:colour"),
            True,
        )


def test_sites_and_positions_are_read_as_the_command_line_writes_them() -> None:
    assert parse_site("resid_post:6") == ReadSite(kind="resid_post", layer=6)
    assert isinstance(parse_position("last"), LastPosition)
    assert isinstance(parse_position("mean"), MeanPosition)
    assert parse_position("-2") == IndexPosition(index=-2)
    assert parse_position("label:S2") == LabelPosition(label="S2")
    with pytest.raises(ValueError, match="isn't a site"):
        parse_site("resid:6")
    with pytest.raises(ValueError, match="isn't a position"):
        parse_position("first")


def test_a_component_and_a_mean_difference_are_saved_as_directions(
    tiny_backend, project, stories
) -> None:
    from logogram.directions import read_direction

    loaded = load_rows(tiny_backend, project, CorpusRows(name="stories", max_tokens=64), True)
    site = ReadSite(kind="resid_post", layer=0)
    settings = dict(prepend_bos=True, batch_size=4)
    saved = save_component(
        project, tiny_backend, loaded, site, LastPosition(), 1, "pc1", **settings
    )
    values = collect(tiny_backend, loaded, [site], LastPosition(), **settings)["resid_post:0"]
    first = principal_components(values).components[0]
    direction = read_direction(project, "pc1")
    assert np.allclose(direction.vectors[0].numpy(), first, atol=1e-6)
    assert direction.meta.made_by.method == "pca" and direction.meta.site is not None
    assert direction.meta.model.id == tiny_backend.info.id
    assert "principal component 1" in direction.meta.made_by.detail and saved.sha256

    save_mean_difference(
        project, tiny_backend, loaded, site, LastPosition(), "ABBA", "BABA", "patterns", **settings
    )
    vector, _, _ = mean_difference(values, loaded, "ABBA", "BABA")
    found = read_direction(project, "patterns")
    assert found.meta.made_by.method == "mean_difference"
    assert np.allclose(found.vectors[0].numpy(), vector / np.linalg.norm(vector), atol=1e-6)
    assert np.isclose(found.meta.norms[0], np.linalg.norm(vector), rtol=1e-6)


# -- exported activations ------------------------------------------------------------------------


def test_exported_activations_read_back_as_they_were_computed(tiny_backend, project) -> None:
    from safetensors.numpy import load_file

    source = DatasetRows(path="datasets/ioi.jsonl", sides="both", mark="meta:pattern", limit=4)
    loaded = load_rows(tiny_backend, project, source, True)
    sites = [ReadSite(kind="resid_post", layer=0), ReadSite(kind="mlp_out", layer=1)]
    written = export_activations(
        project,
        "ioi-acts",
        tiny_backend,
        loaded,
        sites,
        LastPosition(),
        prepend_bos=True,
        batch_size=3,
    )
    assert written["path"] == "activations/ioi-acts.safetensors" and written["n"] == 8
    tensors = load_file(project.root / written["path"])
    direct = collect(tiny_backend, loaded, sites, LastPosition(), prepend_bos=True, batch_size=3)
    assert set(tensors) == {"resid_post:0", "mlp_out:1"}
    for key, values in direct.items():
        assert tensors[key].dtype == np.float32 and tensors[key].shape == values.shape
        assert np.array_equal(tensors[key], values.astype(np.float32))
    meta = json.loads((project.root / "activations" / "ioi-acts.json").read_text(encoding="utf-8"))
    assert meta["model"]["id"] == tiny_backend.info.id and meta["dtype"] == "float32"
    assert meta["source"]["path"] == "datasets/ioi.jsonl" and meta["source"]["sha256"]
    assert meta["rows"][1] == {"row": 1, "index": 0, "side": "corrupt", "label": "ABBA"}
    assert meta["position"] == {"kind": "last"} and meta["tokenization"]["prepend_bos"] is True
    assert str(project.root) not in json.dumps(meta)
    with pytest.raises(ValueError, match="already exists"):
        export_activations(
            project,
            "ioi-acts",
            tiny_backend,
            loaded,
            sites,
            LastPosition(),
            prepend_bos=True,
            batch_size=3,
        )


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_an_activations_folder_that_links_outside_is_never_written(
    tiny_backend, project, tmp_path: Path
) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (project.root / "activations").symlink_to(outside, target_is_directory=True)
    loaded = load_rows(
        tiny_backend, project, DatasetRows(path="datasets/ioi.jsonl", sides="clean", limit=2), True
    )
    with pytest.raises(Exception, match="outside the project"):
        export_activations(
            project,
            "x",
            tiny_backend,
            loaded,
            [ReadSite(kind="resid_pre", layer=0)],
            LastPosition(),
            prepend_bos=True,
            batch_size=2,
        )
    assert list(outside.iterdir()) == []


def test_a_corpus_that_changed_is_refused(tiny_backend, project, stories) -> None:
    write_texts(project, "stories", ["Mary went home."], overwrite=True)
    with pytest.raises(ValueError, match="has changed"):
        load_rows(
            tiny_backend,
            project,
            CorpusRows(name="stories", sha256=stories.sha256, max_tokens=8),
            True,
        )
    assert read_corpus(project, "stories").texts == ["Mary went home."]
