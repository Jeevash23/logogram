"""Run an experiment spec against a loaded model.

The sweep works layer by layer. For each layer it captures the source activations it needs (only
that layer's sites, stopping the forward pass early), then runs the receiver prompts with one
site replaced per row. Rows for many sites share a batch; batches never mix token lengths, so no
padding or attention masks are involved. After each layer the finished sites are reported, so
results can be painted while the sweep continues.

Terminology: the *receiver* is the prompt the model runs on; the *source* supplies the patched
activation (the other prompt of the pair, a mean, a donor, or zeros). The *reference* run defines
the gap used for normalization: the source prompt for patching, the corrupt prompt for ablation.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, fields
from typing import Any, cast

import numpy as np
import torch

from logogram.backends.base import STREAM_KINDS, UNIT_KINDS, Cancelled, ModelBackend, Patch, Patches
from logogram.metrics import Reading, Scorer
from logogram.prompts import LengthGroup, PreparedPrompt, group_by_length, side_groups
from logogram.sites import (
    ResolvedSite,
    ScopeError,
    check_lined_up,
    expand_scope,
    resolve_position,
    site_unit,
)
from logogram.spec import (
    ACDC,
    METRIC_LABELS,
    Ablation,
    ActivationPatching,
    AllPositions,
    AttributionPatching,
    DirectionReadout,
    DirectLogitAttribution,
    DistributedAlignmentSearch,
    EdgeAttribution,
    FeatureReadout,
    FeaturesScope,
    MeanBaseline,
    PathPatching,
    Probe,
    ProjectionBaseline,
    ReadoutMetric,
    ResampleBaseline,
    ScaleBaseline,
    SiteSetsScope,
    SitesScope,
    Spec,
    Steering,
    SubspacePatching,
    ZeroBaseline,
)


class EngineError(ValueError):
    pass


@dataclass
class Baselines:
    """The unpatched clean and corrupt runs: the metric, P(answer) and the preference
    log P(answer) − log P(distractor) (the logit difference for single tokens), per prompt."""

    clean: np.ndarray
    corrupt: np.ndarray
    clean_prob: np.ndarray
    corrupt_prob: np.ndarray
    clean_pref: np.ndarray
    corrupt_pref: np.ndarray

    def metric(self, which: str) -> np.ndarray:
        return self.clean if which == "clean" else self.corrupt

    def prob(self, which: str) -> np.ndarray:
        return self.clean_prob if which == "clean" else self.corrupt_prob

    def pref(self, which: str) -> np.ndarray:
        return self.clean_pref if which == "clean" else self.corrupt_pref

    def subset(self, at: list[int]) -> Baselines:
        """The baselines of the prompts at these positions, in that order."""
        return Baselines(*(getattr(self, f.name)[at] for f in fields(self)))


@dataclass
class EngineResult:
    sites: list[ResolvedSite]
    layout: dict[str, Any]
    prompts: list[PreparedPrompt]
    baselines: Baselines
    receiver: str
    reference: str
    patched: np.ndarray  # [S, n] the metric; NaN where nothing was run patched
    patched_prob: np.ndarray  # [S, n]; NaN where not measured
    patched_pref: np.ndarray  # [S, n]; NaN where not measured
    donors: list[list[int]] | None = None
    warnings: list[str] = field(default_factory=list)
    # What the per-prompt values are: "intervention" (a patched forward pass), "estimate" (a
    # linear estimate of one) or "attribution" (a term of a decomposition of one forward pass).
    measure: str = "intervention"
    # Per-prompt values [S, n] when they aren't patched - receiver (an estimate or a term), and
    # what normalizes them [n] when it isn't reference - receiver.
    delta: np.ndarray | None = None
    gap: np.ndarray | None = None
    # Method-specific numbers for the summary.
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def receiver_metric(self) -> np.ndarray:
        return self.baselines.metric(self.receiver)

    @property
    def reference_metric(self) -> np.ndarray:
        return self.baselines.metric(self.reference)

    @property
    def receiver_prob(self) -> np.ndarray:
        return self.baselines.prob(self.receiver)

    @property
    def receiver_pref(self) -> np.ndarray:
        return self.baselines.pref(self.receiver)

    @property
    def reference_pref(self) -> np.ndarray:
        return self.baselines.pref(self.reference)


def empty_values(n_sites: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zeroed arrays for the metric, P(answer) and the preference of S sites over n prompts."""
    return np.zeros((n_sites, n)), np.zeros((n_sites, n)), np.zeros((n_sites, n))


ProgressFn = Callable[[int, int, int], None]  # (rows done, rows total, current layer)
LayerFn = Callable[[int, list[int], "EngineResult"], None]  # (layer, sites done, partial)
# Methods that measure every site in one pass per batch of prompts report the prompts they have
# finished: their positions in the prompt list, and a function that builds the result over just
# those prompts, as a run of only them would give it (for a run that stops early).
PromptsFn = Callable[[list[int], Callable[[], "EngineResult"]], None]


def _chunks(n: int, size: int) -> Iterator[slice]:
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def answer_tensors(
    prompts: list[PreparedPrompt], idx: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """The answer and distractor tokens of single-token prompts, for methods that read one."""
    return (
        torch.tensor([prompts[i].answer_id for i in idx], dtype=torch.long),
        torch.tensor([prompts[i].distractor_id for i in idx], dtype=torch.long),
    )


def make_scorer(
    spec: Spec,
    prompts: list[PreparedPrompt],
    *,
    sae: Any = None,
    directions: dict[str, Any] | None = None,
) -> Scorer:
    """The spec's metric as a scorer; a readout also gets what it reads: the SAE, or the saved
    direction (pinned by the runner)."""
    metric = spec.scored
    reading = None
    if isinstance(metric, ReadoutMetric):
        unit = metric.unit
        vector = None
        if isinstance(unit, DirectionReadout):
            vector = _direction_vector(directions, unit.direction.name, "A direction readout")
        reading = Reading(
            metric, prompts, sae=sae if isinstance(unit, FeatureReadout) else None, direction=vector
        )
    return Scorer(metric, prompts, reading)


def _direction_vector(directions: dict[str, Any] | None, name: str, what: str) -> torch.Tensor:
    """The unit vector of a saved direction of one dimension."""
    direction = (directions or {}).get(name)
    if direction is None:
        raise EngineError(f"{what} needs the direction {name}, which this run didn't load.")
    if direction.dimensions != 1:
        raise EngineError(
            f"{what} reads one dimension, and the direction {name} has {direction.dimensions}. "
            "Save one of its vectors as a direction of its own."
        )
    return direction.vectors[0]


class _Forward:
    """A forward pass with ``patch`` applied, as the scorer calls it. With a readout (``reading``),
    it also keeps the activations the readout reads from its last pass (``reads``)."""

    def __init__(self, backend: ModelBackend, patch: Patches, reading: Reading | None) -> None:
        self.backend = backend
        self.patch = patch
        self.reading = reading
        self.reads: dict[tuple[str, int], torch.Tensor] | None = None

    def run(self, tokens: torch.Tensor, patch: Patches, keep: int) -> torch.Tensor:
        if self.reading is None:
            return self.backend.logits(tokens, patch, keep)
        logits, self.reads = self.backend.logits_and_reads(tokens, patch, keep, self.reading.sites)
        return logits

    def __call__(self, tokens: torch.Tensor, keep: int) -> torch.Tensor:
        return self.run(tokens, self.patch, keep)


def patched_forward(
    backend: ModelBackend, patch: Patches, reading: Reading | None = None
) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """A forward pass with ``patch`` applied, as the scorer calls it."""
    return _Forward(backend, patch, reading)


# Makes the values of positions appended to read a continuation from the source prompt's values
# there and the receiver's own (None when they aren't needed): a scale, a projection, a patch
# within a subspace.
Combine = Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]


class _ContinuedForward(_Forward):
    def __init__(
        self,
        backend: ModelBackend,
        patch: Patch,
        source: torch.Tensor | None,
        reading: Reading | None,
        combine: Combine | None,
        own: torch.Tensor | None,
    ) -> None:
        super().__init__(backend, patch, reading)
        self.whole = patch
        self.source = source
        self.combine = combine
        self.own = own

    def _appended(self, prompts: torch.Tensor, tail: torch.Tensor, length: int) -> torch.Tensor:
        patch = self.whole
        key = (patch.kind, patch.layer)
        full = self.backend.capture(torch.cat([prompts, tail.to(prompts.device)], dim=1), [key])[
            key
        ]
        units = patch.units()
        if units is not None:  # a head's (or key/value head's) vectors, or a neuron's values
            rows = torch.arange(full.shape[0], device=full.device)
            full = full[rows, :, units.to(full.device)]
        return full[:, length:].to(dtype=patch.values.dtype, device=patch.values.device)

    def __call__(self, tokens: torch.Tensor, keep: int) -> torch.Tensor:
        patch = self.whole
        length = patch.values.shape[1]
        if patch.positions is not None or tokens.shape[1] == length:
            return self.run(tokens, patch, keep)
        tail = tokens[:, length:]
        if self.source is None:
            shape = (patch.values.shape[0], tail.shape[1], *patch.values.shape[2:])
            extra = torch.zeros(shape, dtype=patch.values.dtype, device=patch.values.device)
        else:
            extra = self._appended(self.source, tail, length)
        if self.combine is not None:
            own = None if self.own is None else self._appended(self.own, tail, length)
            extra = self.combine(extra, own)
        values = torch.cat([patch.values, extra], dim=1)
        whole = Patch(
            kind=patch.kind,
            layer=patch.layer,
            values=values,
            heads=patch.heads,
            neurons=patch.neurons,
        )
        return self.run(tokens, whole, keep)


def continued_forward(
    backend: ModelBackend,
    patch: Patch,
    source: torch.Tensor | None,
    reading: Reading | None = None,
    *,
    combine: Combine | None = None,
    own: torch.Tensor | None = None,
) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """A forward pass with ``patch``, where a patch of every position also covers tokens appended
    to read a continuation: their positions take the source prompt's own values with the same
    continuation appended (zeros for zero ablation), so patching every position of a layer still
    reproduces the source run. ``source``: each row's source prompt ``[B, L]``, or None for zeros.
    With ``combine``, the appended positions' values are made from the source's values there and,
    with ``own`` (each row's receiver prompt), the receiver's own, as the patch's values were.
    """
    return _ContinuedForward(backend, patch, source, reading, combine, own)


def check_mean_continuations(scorer: Scorer, every_position: bool) -> None:
    if every_position and not scorer.single_position():
        raise EngineError(
            "Mean ablation at every position replaces each position with its mean over prompts "
            "of the same length, and the tokens appended to read answers of several tokens have "
            "no such mean. Ablate at one position, use zero or resample ablation, or use "
            "single-token answers."
        )


def compute_baselines(
    backend: ModelBackend,
    prompts: list[PreparedPrompt],
    groups: list[LengthGroup],
    batch_size: int,
    cancel: threading.Event | None = None,
    scorer: Scorer | None = None,
) -> Baselines:
    """Run every clean and corrupt prompt unpatched. For the KL divergence, the target prompts'
    own predictions are recorded first, in the same batches, so a target's divergence from itself
    is exactly zero. Each side runs in batches of its own prompts' length (``groups`` themselves,
    for pairs of equal length)."""
    from logogram.spec import LogitDiffMetric

    scorer = scorer or Scorer(
        LogitDiffMetric(kind="logit_diff", normalization="dataset_gap"), prompts
    )
    n = len(prompts)
    if scorer.kind == "kl" and len(scorer.target_logits) < n:
        target = scorer.target or "clean"
        for group in side_groups(groups, prompts, target):
            tokens = group.tokens(target)
            for sl in _chunks(len(group.members), batch_size):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                logits = backend.logits(tokens[sl], None, 1)[:, 0].float().cpu()
                for row, p in enumerate(group.members[sl]):
                    scorer.target_logits[p] = logits[row]
    out = {
        k: np.zeros(n)
        for k in ("clean", "corrupt", "clean_prob", "corrupt_prob", "clean_pref", "corrupt_pref")
    }
    plain = patched_forward(backend, None, scorer.reading)
    for which in ("clean", "corrupt"):
        for group in side_groups(groups, prompts, which):
            tokens = group.tokens(which)
            for sl in _chunks(len(group.members), batch_size):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                idx = group.members[sl]
                scores = scorer.score(plain, tokens[sl], idx, which)
                out[which][idx] = scores.metric
                out[f"{which}_prob"][idx] = scores.prob
                out[f"{which}_pref"][idx] = scores.pref
    return Baselines(**out)


def behavior_warnings(baselines: Baselines) -> list[str]:
    """Warn when the model doesn't do the task: the answer is defined by the clean prompt, so a
    model that prefers the distractor there doesn't show the behavior the prompts test, and every
    effect describes something else."""
    clean = baselines.clean_pref
    if not len(clean) or not np.isfinite(clean).all() or float(clean.mean()) >= 0:
        return []
    return [
        f"On the clean prompts the model prefers the distractor (mean log P(answer) − "
        f"log P(distractor) {float(clean.mean()):.3f}; {int((clean > 0).sum())} of {len(clean)} "
        "prefer the answer), so it doesn't show the behavior these prompts test. Check the "
        "baseline, or use prompts this model solves."
    ]


def check_finite(baselines: Baselines, dtype: str) -> None:
    """Refuse a run whose unpatched prompts already give values that aren't finite numbers: every
    effect would be undefined. This happens when a 16-bit dtype overflows on these prompts."""
    for which in ("clean", "corrupt"):
        values = np.concatenate([baselines.metric(which), baselines.pref(which)])
        bad = int((~np.isfinite(values)).sum())
        if bad:
            fix = (
                "Run it in float32, which has a far wider range."
                if dtype != "float32"
                else "Check the model's weights: delete it from the Hugging Face cache and load it again."
            )
            raise EngineError(
                f"The unpatched {which} prompts give values that aren't finite numbers in "
                f"{dtype} ({bad} of them), so no effect can be measured. {fix}"
            )


def check_values(values: np.ndarray, what: str, dtype: str) -> list[str]:
    """Warn when patched or estimated values aren't finite numbers (their statistics are left
    out)."""
    bad = int((~np.isfinite(values)).sum())
    if not bad:
        return []
    fix = "Run it in float32." if dtype != "float32" else "Check the model and the prompts."
    return [
        f"{bad} {what} aren't finite numbers (an overflow in {dtype}); the sites they belong to "
        f"have no statistics. {fix}"
    ]


def check_gap(
    spec: Spec,
    baselines: Baselines,
    prompts: list[PreparedPrompt],
    receiver: str,
    reference: str,
) -> list[str]:
    """Refuse a normalization the gap can't support, and warn when it is unreliable."""
    warnings: list[str] = behavior_warnings(baselines)
    n = len(prompts)
    what = METRIC_LABELS[spec.scored.kind]
    gap = baselines.metric(reference) - baselines.metric(receiver)
    mean_gap = float(gap.mean())
    if not np.isfinite(mean_gap):
        raise EngineError(
            f"The clean–corrupt gap in the {what} isn't a finite number, so effects can't be "
            "normalized. Run the baseline check, and use float32 if the model overflows."
        )
    if spec.normalization == "dataset_gap":
        if abs(mean_gap) < 1e-3:
            raise EngineError(
                f"The clean and corrupt prompts give almost the same {what} "
                f"(mean gap {mean_gap:.4f}), so a normalized effect is undefined. Run the "
                "baseline check: the model may not show the behavior on these prompts."
            )
        if n > 1:
            se = float(gap.std(ddof=1)) / np.sqrt(n)
            if abs(mean_gap) < 3 * se:
                warnings.append(
                    f"The mean clean–corrupt gap ({mean_gap:.3f}) is small next to its standard "
                    f"error ({se:.3f}), so normalized effects and their intervals are unreliable. "
                    "Use more prompts, or check the baseline."
                )
    if spec.normalization == "prompt_gap":
        zero = [p.index for p, g in zip(prompts, gap, strict=True) if abs(g) < 1e-6]
        if zero:
            raise EngineError(
                f"Prompt(s) {', '.join(map(str, zero[:5]))} have no clean–corrupt gap, so their "
                "own gap can't normalize an effect. Normalize by the dataset gap, or fix those "
                "prompts."
            )
        small = int((np.abs(gap) < 0.1).sum())
        if small:
            warnings.append(
                f"{small} prompt(s) have a clean–corrupt gap below 0.1, so their per-prompt "
                "normalized effects are unstable."
            )
    return warnings


def draw_donors(
    prompts: list[PreparedPrompt],
    groups: list[LengthGroup],
    k: int,
    seed: int,
    same_length: bool,
) -> list[list[int]]:
    """For each prompt, ``k`` distinct donors drawn without replacement, never the prompt itself.

    Uses the raw PCG64 stream (stable across NumPy versions) and a partial Fisher-Yates shuffle.
    """
    bitgen = np.random.PCG64(seed)
    group_of = {i: g for g in groups for i in g.members}
    donors: list[list[int]] = []
    for i in range(len(prompts)):
        pool = group_of[i].members if same_length else list(range(len(prompts)))
        candidates = [j for j in pool if j != i]
        if len(candidates) < k:
            where = "of the same length " if same_length else ""
            raise EngineError(
                f"Resampling needs {k} donor prompts {where}for prompt {i}, but only "
                f"{len(candidates)} are available. Lower the donor count, use more prompts, or "
                "use prompts of equal length."
            )
        raw = bitgen.random_raw(k)
        for t in range(k):
            j = t + int(raw[t] % np.uint64(len(candidates) - t))
            candidates[t], candidates[j] = candidates[j], candidates[t]
        donors.append(candidates[:k])
    return donors


@dataclass
class _Row:
    site: ResolvedSite
    prompt: int  # index into prompts
    local: int  # index within the prompt's length group
    donor: int | None = None  # index into prompts (resample)


class _LayerSources:
    """Activations captured for one layer: per length group, and at resolved positions.

    ``which`` side of each pair is captured, in batches of that side's length (``groups``
    themselves, for pairs of equal length), and positions are resolved in that side's prompts.

    A layer's activations for every prompt can be far larger than a batch. On a CUDA device they
    are kept in CPU memory when they would take more than a quarter of its free memory, and each
    batch's rows are moved back as they are used. Copies are exact, so where they are kept never
    changes a number.
    """

    def __init__(
        self,
        backend: ModelBackend,
        prompts: list[PreparedPrompt],
        groups: list[LengthGroup],
        layer: int,
        kinds: list[str],
        which: str,
        batch_size: int,
        cancel: threading.Event | None = None,
    ) -> None:
        groups = side_groups(groups, prompts, which)
        self.prompts = prompts
        self.groups = groups
        self.which = which
        self.device = backend.device
        self.offloaded: bool | None = None  # decided from the first batch's size
        self.by_group: dict[str, list[torch.Tensor]] = {k: [] for k in kinds}
        tokens_total = sum(len(g.members) * g.length for g in groups)
        for group in groups:
            tokens = group.tokens(which)
            # Capture in batches, so a forward pass never holds more than batch_size prompts.
            parts: dict[str, list[torch.Tensor]] = {k: [] for k in kinds}
            for sl in _chunks(len(group.members), batch_size):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                acts = backend.capture(tokens[sl], [(k, layer) for k in kinds])
                if self.offloaded is None:
                    per_token = sum(
                        acts[(k, layer)][0, 0].numel() * acts[(k, layer)].element_size()
                        for k in kinds
                    )
                    self.offloaded = _keep_on_cpu(self.device, per_token * tokens_total)
                for k in kinds:
                    part = acts[(k, layer)]
                    parts[k].append(part.to("cpu") if self.offloaded else part)
            for k in kinds:
                self.by_group[k].append(torch.cat(parts[k], dim=0))
        self._tables: dict[tuple[str, str], torch.Tensor] = {}
        self._means: dict[tuple[str, int], torch.Tensor] = {}

    def group_rows(
        self, kind: str, group_index: int, local: list[int], units: torch.Tensor | None
    ) -> torch.Tensor:
        """``[B, L, ...]``: these prompts' activations at every position (of these heads, or
        neurons: ``[B, L]``)."""
        acts = self.by_group[kind][group_index]
        idx = torch.tensor(local, dtype=torch.long, device=acts.device)
        rows = acts[idx, :, units.to(acts.device)] if units is not None else acts[idx]
        return rows.to(self.device)

    def group_mean(self, kind: str, group_index: int) -> torch.Tensor:
        """``[L, ...]``: the mean over a length group's prompts, at each position, in float32."""
        key = (kind, group_index)
        if key not in self._means:
            acts = self.by_group[kind][group_index].to(self.device)
            self._means[key] = acts.float().mean(dim=0)
        return self._means[key]

    def table(self, kind: str, site: ResolvedSite) -> torch.Tensor:
        """``[n, ...]``: each prompt's activation at its own resolved position for this site."""
        key = (kind, site.site.position.model_dump_json())
        if key not in self._tables:
            rows = []
            for gi, group in enumerate(self.groups):
                acts = self.by_group[kind][gi]
                for local, p in enumerate(group.members):
                    pos = resolve_position(site.site.position, self.prompts[p], self.which)
                    rows.append((p, acts[local, pos]))
            rows.sort(key=lambda r: r[0])
            self._tables[key] = torch.stack([r[1] for r in rows]).to(self.device)
        return self._tables[key]


def _keep_on_cpu(device: torch.device, size: int) -> bool:
    """Whether captured activations of ``size`` bytes are kept in CPU memory: on a CUDA device,
    when they would take more than a quarter of its free memory. (On Apple's unified memory,
    moving them would free nothing.)"""
    if device.type != "cuda":
        return False
    free, _total = torch.cuda.mem_get_info(device)
    return size > free // 4


def run_experiment(
    spec: Spec,
    backend: ModelBackend,
    prompts: list[PreparedPrompt],
    **kwargs: Any,
) -> EngineResult:
    """Run whichever experiment the spec describes. The app and the CLI both come through here.

    Sweeps report the sites they finish through ``on_layer``; the methods that measure every site
    in one pass per batch of prompts report the prompts they finish through ``on_prompts``."""
    sae = kwargs.pop("sae", None)
    on_prompts: PromptsFn | None = kwargs.pop("on_prompts", None)
    # The saved directions the spec names, pinned by the runner (logogram.directions).
    directions: dict[str, Any] = kwargs.pop("directions", None) or {}
    if isinstance(spec.experiment, Probe):
        from logogram.probes import run_probe

        return run_probe(spec, backend, prompts, **kwargs)
    if isinstance(spec.experiment, DistributedAlignmentSearch):
        from logogram.das import run_das

        return run_das(spec, backend, prompts, **kwargs)
    if isinstance(spec.scope, FeaturesScope) or (
        isinstance(spec.scope, SitesScope)
        and any(s.kind == "sae_feature" for s in spec.scope.sites)
    ):
        if sae is None:
            raise EngineError("This spec measures SAE features, but no SAE is loaded.")
        from logogram.features import run_features

        return run_features(spec, backend, prompts, sae, **kwargs)
    if isinstance(spec.experiment, DirectLogitAttribution):
        from logogram.direct import run_direct_effects

        return run_direct_effects(spec, backend, prompts, on_prompts=on_prompts, **kwargs)
    if isinstance(spec.experiment, EdgeAttribution):
        from logogram.edges import run_edge_attribution

        return run_edge_attribution(spec, backend, prompts, on_prompts=on_prompts, **kwargs)
    if isinstance(spec.experiment, ACDC):
        from logogram.acdc import run_acdc

        return run_acdc(spec, backend, prompts, **kwargs)
    if isinstance(spec.experiment, AttributionPatching):
        from logogram.atp import run_attribution_patching

        return run_attribution_patching(spec, backend, prompts, on_prompts=on_prompts, **kwargs)
    if isinstance(spec.experiment, Steering):
        from logogram.steering import run_steering

        # Steering along an SAE feature adds its decoder direction where the SAE writes.
        return run_steering(spec, backend, prompts, sae=sae, directions=directions, **kwargs)
    if isinstance(spec.experiment, PathPatching):
        from logogram.paths import run_path_patching

        return run_path_patching(spec, backend, prompts, **kwargs)
    if isinstance(spec.scope, SiteSetsScope):
        from logogram.circuits import run_site_sets

        return run_site_sets(spec, backend, prompts, **kwargs)
    return run_engine(spec, backend, prompts, sae=sae, directions=directions, **kwargs)


def _subspace(
    directions: dict[str, Any], name: str, info: Any, device: torch.device
) -> torch.Tensor:
    """The rows ``[k, d_model]`` (orthonormal, float32, on ``device``) of a saved direction that a
    run projects onto, checked against the model's width."""
    from logogram.directions import check_fits

    direction = directions.get(name)
    if direction is None:
        raise EngineError(f"This run needs the direction {name}, which it didn't load.")
    check_fits(direction, info, "the residual stream")
    return direction.vectors.to(device=device, dtype=torch.float32)


def _project(values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """The component of ``values`` (``[..., d]``) within the span of ``basis``'s orthonormal rows
    ``[k, d]``, in float32."""
    v = values.float()
    return (v @ basis.T) @ basis


def run_engine(
    spec: Spec,
    backend: ModelBackend,
    prompts: list[PreparedPrompt],
    *,
    on_progress: ProgressFn | None = None,
    on_layer: LayerFn | None = None,
    cancel: threading.Event | None = None,
    on_start: Callable[[list[ResolvedSite], dict[str, Any]], None] | None = None,
    receiver_override: str | None = None,
    source_override: str | None = None,
    sae: Any = None,
    directions: dict[str, Any] | None = None,
) -> EngineResult:
    """Run the sweep described by ``spec`` on already-prepared prompts.

    ``receiver_override``/``source_override`` exist for sanity tests (for example patching
    corrupt activations into the corrupt run, which must change nothing).
    """
    info = backend.info
    batch_size = spec.execution.batch_size
    directions = directions or {}
    sites, layout = expand_scope(spec, info, prompts)
    exp = spec.experiment
    baseline_kind: str
    # Directional ablation and subspace patching replace an activation's component within the
    # span of a saved direction's rows; scaling multiplies it.
    basis: torch.Tensor | None = None
    factor = 1.0
    mean_side: str | None = None
    if isinstance(exp, ActivationPatching | SubspacePatching):
        receiver, source = (
            ("corrupt", "clean") if exp.direction == "clean_to_corrupt" else ("clean", "corrupt")
        )
        reference = source
        baseline_kind = "patch"
        if isinstance(exp, SubspacePatching):
            baseline_kind = "subspace"
            basis = _subspace(directions, exp.subspace.name, info, backend.device)
    elif isinstance(exp, Ablation):
        receiver, reference = "clean", "corrupt"
        b = exp.baseline
        if isinstance(b, ZeroBaseline):
            source, baseline_kind = "", "zero"
        elif isinstance(b, MeanBaseline):
            source, baseline_kind = b.reference, "mean"
        elif isinstance(b, ResampleBaseline):
            source, baseline_kind = b.pool, "resample"
        elif isinstance(b, ScaleBaseline):
            source, baseline_kind, factor = receiver, "scale", float(b.factor)
        elif isinstance(b, ProjectionBaseline):
            source, baseline_kind = receiver, "projection"
            basis = _subspace(directions, b.direction.name, info, backend.device)
            mean_side = b.reference
        else:  # pragma: no cover
            raise EngineError(f"Unknown baseline {b!r}")
    else:  # pragma: no cover
        raise EngineError(f"Unknown experiment {exp!r}")
    receiver = receiver_override or receiver
    source = source_override or source
    if basis is not None:
        wide = [s for s in sites if s.kind not in STREAM_KINDS]
        if wide:
            what = "Subspace patching" if baseline_kind == "subspace" else "Directional ablation"
            raise ScopeError(
                f"{what} works on activations as wide as the residual stream (stream sites, "
                f"attention and MLP outputs); {wide[0].label} isn't one. Choose stream sites."
            )
    # Between prompts of different lengths, a value is carried at a label or the last token only.
    # (Zero ablation, scaling and removing a direction carry nothing between prompts, and a prompt
    # patched into itself lines up with itself.)
    carries = {
        "zero": False,
        "scale": False,
        "projection": mean_side is not None,
        "patch": source != receiver,
    }.get(baseline_kind, True)
    if carries:
        check_lined_up(
            [s.site.position for s in sites],
            prompts,
            baseline_kind,
        )
    scorer = make_scorer(spec, prompts, sae=sae, directions=directions)
    warnings: list[str] = []
    if scorer.reading is not None:
        warnings += _after_the_readout(sites, scorer.reading.site)
    if basis is not None:
        name = (
            exp.subspace.name
            if isinstance(exp, SubspacePatching)
            else cast(ProjectionBaseline, cast(Ablation, exp).baseline).direction.name
        )
        from logogram.directions import check_fits

        warnings += check_fits(directions[name], info, "the residual stream")
    if on_start is not None:
        on_start(sites, layout)
    # The receivers run in batches of their own length.
    groups = group_by_length(prompts, receiver)
    group_of = {i: gi for gi, g in enumerate(groups) for i in g.members}
    local_of = {i: li for g in groups for li, i in enumerate(g.members)}
    n = len(prompts)

    baselines = compute_baselines(backend, prompts, groups, batch_size, cancel, scorer)
    check_finite(baselines, info.dtype)
    warnings.extend(check_gap(spec, baselines, prompts, receiver, reference))

    donors: list[list[int]] | None = None
    k = 1
    if baseline_kind == "resample":
        assert isinstance(exp, Ablation) and isinstance(exp.baseline, ResampleBaseline)
        k = exp.baseline.donors
        same_length = any(isinstance(s.site.position, AllPositions) for s in sites)
        donors = draw_donors(prompts, groups, k, exp.baseline.seed, same_length)
    if baseline_kind == "mean" or mean_side is not None:
        check_mean_continuations(
            scorer, any(isinstance(s.site.position, AllPositions) for s in sites)
        )
    if (baseline_kind == "mean" or mean_side is not None) and any(
        isinstance(s.site.position, AllPositions) for s in sites
    ):
        alone = sum(1 for g in groups if len(g.members) == 1)
        if alone:
            warnings.append(
                f"{alone} prompt(s) are the only prompt of their length, so their per-position "
                "mean is their own activation."
            )

    patched, patched_prob, patched_pref = empty_values(len(sites), n)
    # Shares the arrays being filled, so per-layer callbacks can summarize finished sites.
    partial = EngineResult(
        sites=sites,
        layout=layout,
        prompts=prompts,
        baselines=baselines,
        receiver=receiver,
        reference=reference,
        patched=patched,
        patched_prob=patched_prob,
        patched_pref=patched_pref,
        donors=donors,
        warnings=warnings,
    )
    layers = sorted({s.layer for s in sites})
    total_rows = len(sites) * n * k
    done_rows = 0
    model_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        info.dtype
    ]

    for layer in layers:
        layer_sites = [s for s in sites if s.layer == layer]
        kinds = list(dict.fromkeys(s.kind for s in layer_sites))
        sources = (
            _LayerSources(backend, prompts, groups, layer, kinds, source, batch_size, cancel)
            if baseline_kind != "zero"
            else None
        )
        # Patching within a subspace also needs the receivers' own activations; ablating a
        # direction to a mean needs the reference prompts' mean.
        own = (
            _LayerSources(backend, prompts, groups, layer, kinds, receiver, batch_size, cancel)
            if baseline_kind == "subspace"
            else sources
        )
        means = (
            _LayerSources(backend, prompts, groups, layer, kinds, mean_side, batch_size, cancel)
            if mean_side is not None
            else None
        )
        for gi, group in enumerate(groups):
            receiver_tokens = group.tokens(receiver)
            # Rows are grouped by (kind, all-positions?) so one hook serves the whole batch.
            buckets: dict[tuple[str, bool], list[_Row]] = {}
            for site in layer_sites:
                all_pos = isinstance(site.site.position, AllPositions)
                bucket = buckets.setdefault((site.kind, all_pos), [])
                for p in group.members:
                    if donors is not None:
                        for d in donors[p]:
                            bucket.append(_Row(site, p, local_of[p], d))
                    else:
                        bucket.append(_Row(site, p, local_of[p]))
            for (kind, all_pos), rows in buckets.items():
                for sl in _chunks(len(rows), batch_size):
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
                    chunk = rows[sl]

                    common: dict[str, Any] = {
                        "group_index": gi,
                        "length": group.length,
                        "group_of": group_of,
                        "local_of": local_of,
                        "prompts": prompts,
                        "d_model": info.d_model,
                        "d_head": info.d_head,
                        "dtype": model_dtype,
                        "device": backend.device,
                        "receiver": receiver,
                    }
                    pick = (chunk, kind, layer, all_pos)

                    combine: Combine | None = None
                    if baseline_kind in ("scale", "projection", "subspace"):
                        # Built from the receivers' own activations ("patch" from themselves).
                        patch = _build_patch(*pick, "patch", own, **common)
                        combine = _combiner(baseline_kind, factor, basis)
                        mean = (
                            None
                            if means is None
                            else _build_patch(*pick, "mean", means, **common).values
                        )
                        source_values = (
                            _build_patch(*pick, "patch", sources, **common).values
                            if baseline_kind == "subspace"
                            else None
                        )
                        patch.values = _combined(
                            baseline_kind, patch.values, source_values, mean, factor, basis
                        )
                    else:
                        patch = _build_patch(*pick, baseline_kind, sources, **common)
                    local_idx = torch.tensor([r.local for r in chunk], dtype=torch.long)
                    reading = scorer.reading
                    if baseline_kind in ("patch", "resample", "subspace") and all_pos:
                        # Every position lines up: pairs of equal length, or a prompt with itself.
                        donor = [r.prompt if r.donor is None else r.donor for r in chunk]
                        donors_tokens = group.tokens(source)[
                            torch.tensor([local_of[p] for p in donor])
                        ]
                        forward = continued_forward(
                            backend,
                            patch,
                            donors_tokens,
                            reading,
                            combine=combine,
                            own=receiver_tokens[local_idx] if combine is not None else None,
                        )
                    elif baseline_kind in ("scale", "projection") and all_pos:
                        forward = continued_forward(
                            backend, patch, receiver_tokens[local_idx], reading, combine=combine
                        )
                    elif baseline_kind == "zero" and all_pos:
                        forward = continued_forward(backend, patch, None, reading)
                    else:
                        forward = patched_forward(backend, patch, reading)
                    scores = scorer.score(
                        forward, receiver_tokens[local_idx], [r.prompt for r in chunk], receiver
                    )
                    for r, value, prob, pref in zip(
                        chunk, scores.metric, scores.prob, scores.pref, strict=True
                    ):
                        patched[r.site.index, r.prompt] += value / k
                        patched_prob[r.site.index, r.prompt] += prob / k
                        patched_pref[r.site.index, r.prompt] += pref / k
                    done_rows += len(chunk)
                    if on_progress is not None:
                        on_progress(done_rows, total_rows, layer)
        if on_layer is not None:
            on_layer(layer, [s.index for s in layer_sites], partial)

    warnings.extend(check_values(patched, "patched values", info.dtype))
    return partial


def _build_patch(
    rows: list[_Row],
    kind: str,
    layer: int,
    all_pos: bool,
    baseline_kind: str,
    sources: _LayerSources | None,
    group_index: int,
    length: int,
    group_of: dict[int, int],
    local_of: dict[int, int],
    prompts: list[PreparedPrompt],
    d_model: int,
    d_head: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    receiver: str = "clean",
) -> Patch:
    """The patch for a batch of rows of one length group of ``receiver`` prompts: each row's
    value goes into its own prompt's token at the site's position, from the source's token at
    that position (``sources`` resolves it in the source prompts)."""
    B = len(rows)
    # A head's output, query, key or value is one head's vector of its layer's activation; a
    # neuron is one number of the MLP's hidden activation. Other sites are whole vectors.
    is_unit = kind in UNIT_KINDS
    units = (
        torch.tensor([site_unit(r.site.site) for r in rows], dtype=torch.long) if is_unit else None
    )
    heads, neurons = (None, units) if kind == "neuron" else (units, None)
    positions = None
    if not all_pos:
        positions = torch.tensor(
            [resolve_position(r.site.site.position, prompts[r.prompt], receiver) for r in rows],
            dtype=torch.long,
        )
    width: tuple[int, ...] = () if kind == "neuron" else (d_head,) if is_unit else (d_model,)

    def patch(values: torch.Tensor) -> Patch:
        return Patch(
            kind=kind,
            layer=layer,
            values=values,
            heads=heads,
            positions=positions,
            neurons=neurons,
        )

    if baseline_kind == "zero":
        shape = (B, length, *width) if all_pos else (B, *width)
        return patch(torch.zeros(shape, dtype=dtype, device=device))

    assert sources is not None
    dev_units = units.to(device) if units is not None else None

    if baseline_kind in ("patch", "resample"):
        if baseline_kind == "patch":
            src_prompts = [r.prompt for r in rows]
        else:
            src_prompts = cast(list[int], [r.donor for r in rows])
        if all_pos:
            # Patch pairs share a length; resample donors are drawn from the same length group.
            assert all(group_of[p] == group_index for p in src_prompts)
            values = sources.group_rows(
                kind, group_index, [local_of[p] for p in src_prompts], dev_units
            )
        else:
            table = sources.table(kind, rows[0].site)
            if any(r.site.site.position != rows[0].site.site.position for r in rows):
                # Rows with different positions (layer x position sweeps) gather from their own table.
                values = torch.stack(
                    [
                        _table_row(sources.table(kind, r.site), p, site_unit(r.site.site))
                        for r, p in zip(rows, src_prompts, strict=True)
                    ]
                )
            else:
                idx = torch.tensor(src_prompts, dtype=torch.long, device=device)
                values = table[idx, dev_units] if is_unit else table[idx]
        return patch(values)

    # Mean ablation.
    if all_pos:
        mean = sources.group_mean(kind, group_index)  # [L, ...]
        if is_unit:
            values = mean[:, dev_units].movedim(1, 0)  # [B, L, d_head], or [B, L] for neurons
        else:
            values = mean.unsqueeze(0).expand(B, *mean.shape)
    else:
        means = {}
        out = []
        for r in rows:
            key = r.site.site.position.model_dump_json()
            if key not in means:
                means[key] = sources.table(kind, r.site).float().mean(dim=0)
            m = means[key]
            out.append(m[site_unit(r.site.site)] if is_unit else m)
        values = torch.stack(out)
    return patch(values)


def _table_row(table: torch.Tensor, prompt: int, unit: int | None) -> torch.Tensor:
    row = table[prompt]
    return row[unit] if unit is not None else row


def _combined(
    kind: str,
    own: torch.Tensor,
    source: torch.Tensor | None,
    mean: torch.Tensor | None,
    factor: float,
    basis: torch.Tensor | None,
) -> torch.Tensor:
    """A patch's values made from the receivers' own activations: scaled, with a direction's
    component removed (or set to the reference prompts' ``mean`` component), or with the
    component within a subspace taken from the ``source`` prompts."""
    if kind == "scale":
        return (own.float() * factor).to(own.dtype)
    assert basis is not None
    if kind == "projection":
        out = own.float() - _project(own, basis)
        if mean is not None:
            out = out + _project(mean, basis)
        return out.to(own.dtype)
    assert source is not None
    return (own.float() + _project(source.float() - own.float(), basis)).to(own.dtype)


def _combiner(kind: str, factor: float, basis: torch.Tensor | None) -> Combine:
    """How appended positions' values are made, as :func:`_combined` makes the prompt's."""

    def combine(source: torch.Tensor, own: torch.Tensor | None) -> torch.Tensor:
        if kind == "subspace":
            assert own is not None
            return _combined(kind, own, source, None, factor, basis)
        return _combined(kind, source, None, None, factor, basis)

    return combine


# Where each kind of site is computed within a layer: a site can only change what is computed
# after it, in its own layer or a later one.
_ORDER = {
    "resid_pre": 0,
    "q": 1,
    "k": 1,
    "v": 1,
    "head": 2,
    "attn_out": 3,
    "resid_mid": 4,
    "mlp_in": 5,
    "neuron": 6,
    "mlp_out": 7,
    "resid_post": 8,
}


def _after_the_readout(sites: list[ResolvedSite], read: tuple[str, int]) -> list[str]:
    """Warn that sites computed after what a readout reads can't change it: their effects are 0.
    (A layer whose attention and MLP run in parallel computes its MLP from resid_pre; the order
    here is the sequential one, so a parallel layer's MLP may read as after its attention.)"""
    kind, layer = read
    at = (layer, _ORDER.get(kind, 0))
    after = [s for s in sites if (s.layer, _ORDER.get(s.kind, 0)) > at]
    if not after:
        return []
    return [
        f"{len(after)} of the {len(sites)} sites are computed after what the readout reads "
        f"({kind} in layer {layer}), so they can't change it: their effects are 0."
    ]
