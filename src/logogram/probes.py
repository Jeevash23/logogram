"""Linear probes: what can be read from a site's activation (``spec.Probe``).

At each site of the scope, a logistic regression is trained to read a label from the activation at
one position of each prompt: which prompt of the pair it is (clean or corrupt), or a field of the
pair's ``meta``. The pairs are split with a seed into training pairs and held-out pairs, so the two
prompts of a pair are never on both sides, and each held-out pair's value is the share of its
examples the probe labels correctly. The statistics are those of any run, over the held-out pairs:
a site's mean is the probe's held-out accuracy, with its bootstrap interval.

A probe that reads a label shows that the information is there, in a form a linear map can read,
not that the model uses it: patching the site, or patching along the probe's direction (save it as
a direction, then use subspace patching), tests that. With ``control``, a second probe at each site
is trained on the training labels shuffled with the seed and measured on the true held-out labels:
what a probe reaches without the information. The difference between the two (selectivity) is
measured pair by pair, with its own interval, in the summary.

Each probe is fitted in float64 on the CPU by L-BFGS, from zero, to the mean cross-entropy plus
``regularization`` / 2 times the squared length of its weights, on activations standardized with
the training examples' mean and spread. Its weights are kept in the run's vectors file in the
activation's own units (undoing the standardization), so the direction toward a class can be saved
and used elsewhere.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from logogram.backends.base import STREAM_KINDS, Cancelled, ModelBackend
from logogram.engine import (
    EngineError,
    EngineResult,
    LayerFn,
    ProgressFn,
    _LayerSources,
    compute_baselines,
    empty_values,
)
from logogram.metrics import Scorer
from logogram.prompts import PreparedPrompt, group_by_length, seeded_split
from logogram.sites import ResolvedSite, ScopeError, expand_scope, site_label, site_unit
from logogram.spec import (
    AllPositions,
    LogProbDiffMetric,
    MetaLabels,
    Probe,
    SideLabels,
    Spec,
    labels_text,
)

# Sites a probe reads: whole vectors of the residual stream and of attention and MLP outputs, and
# one head's output, query, key or value.
PROBE_KINDS = (*STREAM_KINDS, "head", "q", "k", "v")


def label_of(value: Any) -> str:
    """A meta value as the label a probe reads: text as it is, true and false as words, numbers
    as written, anything else as its JSON."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str | int | float):
        return str(value)
    return json.dumps(value, sort_keys=True)


def examples_of(
    labels: SideLabels | MetaLabels, prompts: list[PreparedPrompt]
) -> list[tuple[int, str, str]]:
    """Each example a probe reads: (the pair's position in ``prompts``, which prompt of the pair,
    its label), pairs in order and each pair's clean prompt first."""
    if isinstance(labels, SideLabels):
        return [(i, side, side) for i in range(len(prompts)) for side in ("clean", "corrupt")]
    sides = {"clean": ("clean",), "corrupt": ("corrupt",), "both": ("clean", "corrupt")}
    missing = [p.index for p in prompts if labels.field not in (p.record.meta or {})]
    if missing:
        shown = ", ".join(str(i) for i in missing[:5])
        more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise EngineError(
            f"Prompt(s) {shown}{more} have no {labels.field!r} in their meta, so they have no "
            "label. Label every prompt, or choose a field every prompt has."
        )
    return [
        (i, side, label_of((p.record.meta or {})[labels.field]))
        for i, p in enumerate(prompts)
        for side in sides[labels.prompts]
    ]


def fit_probe(
    x: torch.Tensor, y: torch.Tensor, classes: int, l2: float, iterations: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Logistic regression on ``x`` ``[m, d]`` (float64, on the CPU) with labels ``y`` ``[m]``:
    the weights ``[classes, d]`` and biases ``[classes]`` in the activation's own units, and the
    standardization (mean, spread) the fit used. Fitted by L-BFGS from zero, on standardized
    activations, to the mean cross-entropy plus ``l2`` / 2 times the squared length of the
    (standardized) weights."""
    mean = x.mean(dim=0)
    spread = x.std(dim=0, unbiased=False)
    spread = torch.where(spread > 0, spread, torch.ones_like(spread))
    z = (x - mean) / spread
    weights = torch.zeros(classes, x.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(classes, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weights, bias],
        lr=1.0,
        max_iter=iterations,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(z @ weights.T + bias, y)
        loss = loss + 0.5 * l2 * (weights * weights).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        optimizer.step(closure)  # type: ignore[no-untyped-call]
    w = weights.detach()
    # In the activation's own units: w · (x - mean) / spread + b = (w / spread) · x + b'.
    raw = w / spread
    raw_bias = bias.detach() - raw @ mean
    return raw, raw_bias, mean, spread


def predict(x: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """The class each row of ``x`` reads as (the first, where classes tie)."""
    return torch.argmax(x @ weights.T + bias, dim=1)


def run_probe(
    spec: Spec,
    backend: ModelBackend,
    prompts: list[PreparedPrompt],
    *,
    on_progress: ProgressFn | None = None,
    on_layer: LayerFn | None = None,
    cancel: threading.Event | None = None,
    on_start: Callable[[list[ResolvedSite], dict[str, Any]], None] | None = None,
) -> EngineResult:
    exp = spec.experiment
    assert isinstance(exp, Probe)
    info = backend.info
    batch_size = spec.execution.batch_size
    base_sites, grid = expand_scope(spec, info, prompts)
    base = [rs.site for rs in base_sites]
    for rs in base_sites:
        if isinstance(rs.site.position, AllPositions):
            raise ScopeError(
                "A probe reads the activation at one position of each prompt: choose the last "
                "token, a token index or a labelled position, not every position."
            )
        if rs.kind not in PROBE_KINDS:
            raise ScopeError(
                f"A probe reads whole activations: the residual stream, attention and MLP "
                f"outputs, or one head's output, query, key or value; {rs.label} isn't one."
            )
    examples = examples_of(exp.labels, prompts)
    train_at, test_at = seeded_split(len(prompts), exp.train_fraction, exp.seed)
    if len(train_at) < 1 or len(test_at) < 2:
        raise EngineError(
            f"A probe needs at least one prompt pair to train on and two to measure it on, but "
            f"{len(prompts)} pairs with a training share of {exp.train_fraction:.0%} leaves "
            f"{len(train_at)} and {len(test_at)}. Use more prompts or another training share."
        )
    train_set, test_set = set(train_at), set(test_at)
    train_examples = [e for e in examples if e[0] in train_set]
    test_examples = [e for e in examples if e[0] in test_set]
    classes = sorted({label for _, _, label in train_examples})
    if len(classes) < 2:
        raise EngineError(
            f"Every training example is labelled {classes[0]!r}, so there is nothing to tell "
            f"apart. Use labels with at least two values among the training pairs "
            f"({labels_text(exp.labels)})."
        )
    unseen = sorted({label for _, _, label in test_examples} - set(classes))
    if unseen:
        raise EngineError(
            f"Held-out examples are labelled {', '.join(repr(u) for u in unseen)}, which no "
            "training example is, so no probe could read them. Use more prompts, another "
            "training share or another seed, so every label is among the training pairs."
        )
    class_of = {label: c for c, label in enumerate(classes)}
    y_train = torch.tensor([class_of[label] for _, _, label in train_examples], dtype=torch.long)
    y_test = torch.tensor([class_of[label] for _, _, label in test_examples], dtype=torch.long)
    # The control: the training labels shuffled with the seed (a raw PCG64 stream, the same on
    # every machine), measured on the true held-out labels.
    shuffled = y_train.clone()
    raw = np.random.PCG64(exp.seed).random_raw(len(shuffled))
    for t in range(len(shuffled) - 1, 0, -1):
        j = int(raw[t] % np.uint64(t + 1))
        shuffled[[t, j]] = shuffled[[j, t]]

    variants = [("probe", False)] + ([("control", True)] if exp.control else [])
    one_per_layer = len({s.layer for s in base}) == len(base)
    uniform = one_per_layer and len({(s.kind, s.position.model_dump_json()) for s in base}) == 1
    sites: list[ResolvedSite] = []
    for b, site in enumerate(base):
        for v, (key, control) in enumerate(variants):
            sites.append(
                ResolvedSite(
                    index=len(sites),
                    site=site,
                    row=b,
                    col=v,
                    label=f"{site_label(site)} control" if control else site_label(site),
                    variant={"control": control},
                    variant_key=key,
                )
            )
    layout = {
        "kind": "probe",
        "site": base[0].kind if uniform else None,
        "row_title": "Layer" if uniform else "Site",
        "col_title": "Probe",
        "rows": [
            {"key": str(b), "label": str(s.layer) if uniform else site_label(s)}
            for b, s in enumerate(base)
        ],
        "cols": [
            {"key": key, "label": "Control (shuffled labels)" if control else "Probe"}
            for key, control in variants
        ],
    }
    if on_start is not None:
        on_start(sites, layout)

    test = [prompts[i] for i in test_at]
    test_groups = group_by_length(test, "clean")
    # The held-out prompts' own preferences, for context beside the probes: read as
    # log-probabilities, which read any answer.
    scorer = Scorer(LogProbDiffMetric(kind="logprob_diff", normalization="dataset_gap"), test)
    baselines = compute_baselines(backend, test, test_groups, batch_size, cancel, scorer)
    n = len(test)
    position_of = {p: k for k, p in enumerate(test_at)}
    # Each held-out pair's examples, by their rows among the held-out examples.
    rows_of: dict[int, list[int]] = {}
    for row, (p, _side, _label) in enumerate(test_examples):
        rows_of.setdefault(position_of[p], []).append(row)
    nan = np.full((len(sites), n), np.nan)
    delta, _, _ = empty_values(len(sites), n)
    result = EngineResult(
        sites=sites,
        layout=layout,
        prompts=test,
        baselines=baselines,
        receiver="clean",
        reference="corrupt",
        patched=nan.copy(),
        patched_prob=nan.copy(),
        patched_pref=nan.copy(),
        measure="probe",
        delta=delta,
        gap=np.ones(n),
    )
    chance = float((y_test == int(torch.bincount(y_train).argmax())).double().mean())
    found: list[dict[str, Any]] = []
    vectors: dict[str, torch.Tensor] = {}
    sides_needed = sorted({side for _, side, _ in examples})
    groups = group_by_length(prompts, "clean")
    total = len(sites)
    done = 0
    for layer in sorted({s.layer for s in base}):
        at_layer = [b for b, s in enumerate(base) if s.layer == layer]
        kinds = list(dict.fromkeys(base[b].kind for b in at_layer))
        captured = {
            side: _LayerSources(backend, prompts, groups, layer, kinds, side, batch_size, cancel)
            for side in sides_needed
        }
        finished: list[int] = []
        for b in at_layer:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            site = base_sites[b]
            x_train = _features(captured, site, train_examples)
            x_test = _features(captured, site, test_examples)
            for v, (_key, control) in enumerate(variants):
                index = b * len(variants) + v
                labels = shuffled if control else y_train
                weights, bias, _mean, _spread = fit_probe(
                    x_train, labels, len(classes), exp.regularization, exp.iterations
                )
                correct = (predict(x_test, weights, bias) == y_test).double()
                for k, rows in rows_of.items():
                    delta[index, k] = float(correct[rows].mean())
                train_accuracy = float((predict(x_train, weights, bias) == labels).double().mean())
                found.append(
                    {
                        "index": index,
                        "control": control,
                        "train_accuracy": train_accuracy,
                        "test_accuracy": float(correct.mean()),
                    }
                )
                if not control:
                    vectors[f"site{index}"] = weights
                    vectors[f"site{index}.bias"] = bias
                finished.append(index)
                done += 1
                if on_progress is not None:
                    on_progress(done, total, layer)
        if on_layer is not None:
            on_layer(layer, finished, result)
    result.extra["vectors"] = vectors
    result.extra["probe"] = {
        "labels": labels_text(exp.labels),
        "classes": classes,
        "chance": chance,
        "train": [prompts[i].index for i in train_at],
        "test": [prompts[i].index for i in test_at],
        "examples": {"train": len(train_examples), "test": len(test_examples)},
        "grid": {**grid, "sites": [{"row": rs.row, "col": rs.col} for rs in base_sites]},
        "sites": found,
    }
    return result


def _features(
    captured: dict[str, _LayerSources], site: ResolvedSite, chosen: list[tuple[int, str, str]]
) -> torch.Tensor:
    """The examples' activations at ``site`` ``[examples, width]``, in float64 on the CPU: each
    prompt at its own position (``captured`` holds each side's)."""
    tables = {side: captured[side].table(site.kind, site) for side in {s for _, s, _ in chosen}}
    x = torch.stack([tables[side][p] for p, side, _ in chosen])
    unit = site_unit(site.site)
    if unit is not None:  # one head's vector, of [examples, heads, d_head]
        x = x[:, unit]
    return x.detach().to(device="cpu", dtype=torch.float64).reshape(len(chosen), -1)


def selectivity(result: EngineResult, stats: Any, ci: float) -> list[dict[str, Any]]:
    """Each site's probe against its control, pair by pair: the same held-out pairs and the same
    resamples measure both, so the difference of their accuracies gets its own interval. A probe
    is selective where that interval lies above zero."""
    pairs = {}
    for rs in result.sites:
        if rs.variant is None:
            continue
        pairs.setdefault(rs.row, {})[bool(rs.variant.get("control"))] = rs.index
    boot = stats.effect_boot
    alpha = (1.0 - ci) / 2.0
    out: list[dict[str, Any]] = []
    for row, both in sorted(pairs.items()):
        i, j = both.get(False), both.get(True)
        if i is None or j is None:
            continue
        mean = float(stats.effect_mean[i]) - float(stats.effect_mean[j])
        lo = hi = None
        if boot is not None and stats.n >= 2:
            difference = boot[i] - boot[j]
            if np.isfinite(difference).all():
                lo, hi = (float(x) for x in np.quantile(difference, [alpha, 1.0 - alpha]))
        out.append(
            {
                "row": row,
                "index": i,
                "control_index": j,
                "difference": mean if np.isfinite(mean) else None,
                "lo": lo,
                "hi": hi,
                "selective": lo is not None and lo > 0,
            }
        )
    return out
