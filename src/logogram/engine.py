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
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from logogram.backends.base import Cancelled, ModelBackend, Patch
from logogram.prompts import LengthGroup, PreparedPrompt, group_by_length
from logogram.sites import ResolvedSite, expand_scope, resolve_position
from logogram.spec import (
    Ablation,
    ActivationPatching,
    AllPositions,
    AttributionPatching,
    DirectLogitAttribution,
    MeanBaseline,
    ResampleBaseline,
    Spec,
    ZeroBaseline,
)


class EngineError(ValueError):
    pass


@dataclass
class Baselines:
    clean_ld: np.ndarray
    corrupt_ld: np.ndarray
    clean_prob: np.ndarray
    corrupt_prob: np.ndarray

    def ld(self, which: str) -> np.ndarray:
        return self.clean_ld if which == "clean" else self.corrupt_ld

    def prob(self, which: str) -> np.ndarray:
        return self.clean_prob if which == "clean" else self.corrupt_prob


@dataclass
class EngineResult:
    sites: list[ResolvedSite]
    layout: dict[str, Any]
    prompts: list[PreparedPrompt]
    baselines: Baselines
    receiver: str
    reference: str
    patched_ld: np.ndarray  # [S, n]; NaN where nothing was run patched
    patched_prob: np.ndarray  # [S, n]; NaN where not measured
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
    def receiver_ld(self) -> np.ndarray:
        return self.baselines.ld(self.receiver)

    @property
    def reference_ld(self) -> np.ndarray:
        return self.baselines.ld(self.reference)

    @property
    def receiver_prob(self) -> np.ndarray:
        return self.baselines.prob(self.receiver)


ProgressFn = Callable[[int, int, int], None]  # (rows done, rows total, current layer)
LayerFn = Callable[[int, list[int], "EngineResult"], None]  # (layer, sites done, partial)


def _chunks(n: int, size: int) -> Iterator[slice]:
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def _metric(
    logits: torch.Tensor, answers: torch.Tensor, distractors: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    rows = torch.arange(logits.shape[0], device=logits.device)
    answers = answers.to(logits.device)
    distractors = distractors.to(logits.device)
    ld = logits[rows, answers] - logits[rows, distractors]
    prob = torch.log_softmax(logits, dim=-1)[rows, answers].exp()
    return ld.double().cpu().numpy(), prob.double().cpu().numpy()


def _answer_tensors(prompts: list[PreparedPrompt], idx: list[int]) -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor([prompts[i].answer_id for i in idx], dtype=torch.long),
        torch.tensor([prompts[i].distractor_id for i in idx], dtype=torch.long),
    )


def compute_baselines(
    backend: ModelBackend,
    prompts: list[PreparedPrompt],
    groups: list[LengthGroup],
    batch_size: int,
    cancel: threading.Event | None = None,
) -> Baselines:
    n = len(prompts)
    out = {k: np.zeros(n) for k in ("clean_ld", "corrupt_ld", "clean_prob", "corrupt_prob")}
    for which in ("clean", "corrupt"):
        for group in groups:
            tokens = group.clean if which == "clean" else group.corrupt
            for sl in _chunks(len(group.members), batch_size):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                idx = group.members[sl]
                logits = backend.final_logits(tokens[sl])
                ld, prob = _metric(logits, *_answer_tensors(prompts, idx))
                out[f"{which}_ld"][idx] = ld
                out[f"{which}_prob"][idx] = prob
    return Baselines(**out)


def check_gap(
    spec: Spec,
    baselines: Baselines,
    prompts: list[PreparedPrompt],
    receiver: str,
    reference: str,
) -> list[str]:
    """Refuse a normalization the gap can't support, and warn when it is unreliable."""
    warnings: list[str] = []
    n = len(prompts)
    gap = baselines.ld(reference) - baselines.ld(receiver)
    mean_gap = float(gap.mean())
    if spec.metric.normalization == "dataset_gap":
        if abs(mean_gap) < 1e-3:
            raise EngineError(
                "The clean and corrupt prompts give almost the same logit difference "
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
    if spec.metric.normalization == "prompt_gap":
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
    """Activations captured for one layer: per length group, and at resolved positions."""

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
        self.prompts = prompts
        self.groups = groups
        self.by_group: dict[str, list[torch.Tensor]] = {k: [] for k in kinds}
        for group in groups:
            tokens = group.clean if which == "clean" else group.corrupt
            # Capture in batches, so a forward pass never holds more than batch_size prompts.
            parts: dict[str, list[torch.Tensor]] = {k: [] for k in kinds}
            for sl in _chunks(len(group.members), batch_size):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                acts = backend.capture(tokens[sl], [(k, layer) for k in kinds])
                for k in kinds:
                    parts[k].append(acts[(k, layer)])
            for k in kinds:
                self.by_group[k].append(torch.cat(parts[k], dim=0))
        self._tables: dict[tuple[str, str], torch.Tensor] = {}

    def group_tensor(self, kind: str, group_index: int) -> torch.Tensor:
        return self.by_group[kind][group_index]

    def table(self, kind: str, site: ResolvedSite) -> torch.Tensor:
        """``[n, ...]``: each prompt's activation at its own resolved position for this site."""
        key = (kind, site.site.position.model_dump_json())
        if key not in self._tables:
            rows = []
            for gi, group in enumerate(self.groups):
                acts = self.by_group[kind][gi]
                for local, p in enumerate(group.members):
                    pos = resolve_position(site.site.position, self.prompts[p])
                    rows.append((p, acts[local, pos]))
            rows.sort(key=lambda r: r[0])
            self._tables[key] = torch.stack([r[1] for r in rows])
        return self._tables[key]


def run_experiment(
    spec: Spec,
    backend: ModelBackend,
    prompts: list[PreparedPrompt],
    **kwargs: Any,
) -> EngineResult:
    """Run whichever experiment the spec describes. The app and the CLI both come through here."""
    if isinstance(spec.experiment, DirectLogitAttribution):
        from logogram.direct import run_direct_effects

        return run_direct_effects(spec, backend, prompts, **kwargs)
    if isinstance(spec.experiment, AttributionPatching):
        from logogram.atp import run_attribution_patching

        return run_attribution_patching(spec, backend, prompts, **kwargs)
    return run_engine(spec, backend, prompts, **kwargs)


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
) -> EngineResult:
    """Run the sweep described by ``spec`` on already-prepared prompts.

    ``receiver_override``/``source_override`` exist for sanity tests (for example patching
    corrupt activations into the corrupt run, which must change nothing).
    """
    info = backend.info
    batch_size = spec.execution.batch_size
    sites, layout = expand_scope(spec, info, prompts)
    if on_start is not None:
        on_start(sites, layout)
    groups = group_by_length(prompts)
    group_of = {i: gi for gi, g in enumerate(groups) for i in g.members}
    local_of = {i: li for g in groups for li, i in enumerate(g.members)}
    n = len(prompts)
    warnings: list[str] = []

    baselines = compute_baselines(backend, prompts, groups, batch_size, cancel)

    exp = spec.experiment
    baseline_kind: str
    if isinstance(exp, ActivationPatching):
        receiver, source = (
            ("corrupt", "clean") if exp.direction == "clean_to_corrupt" else ("clean", "corrupt")
        )
        reference = source
        baseline_kind = "patch"
    elif isinstance(exp, Ablation):
        receiver, reference = "clean", "corrupt"
        b = exp.baseline
        if isinstance(b, ZeroBaseline):
            source, baseline_kind = "", "zero"
        elif isinstance(b, MeanBaseline):
            source, baseline_kind = b.reference, "mean"
        elif isinstance(b, ResampleBaseline):
            source, baseline_kind = b.pool, "resample"
        else:  # pragma: no cover
            raise EngineError(f"Unknown baseline {b!r}")
    else:  # pragma: no cover
        raise EngineError(f"Unknown experiment {exp!r}")
    receiver = receiver_override or receiver
    source = source_override or source
    warnings.extend(check_gap(spec, baselines, prompts, receiver, reference))

    donors: list[list[int]] | None = None
    k = 1
    if baseline_kind == "resample":
        assert isinstance(exp, Ablation) and isinstance(exp.baseline, ResampleBaseline)
        k = exp.baseline.donors
        same_length = any(isinstance(s.site.position, AllPositions) for s in sites)
        donors = draw_donors(prompts, groups, k, exp.baseline.seed, same_length)
    if baseline_kind == "mean" and any(isinstance(s.site.position, AllPositions) for s in sites):
        alone = sum(1 for g in groups if len(g.members) == 1)
        if alone:
            warnings.append(
                f"{alone} prompt(s) are the only prompt of their length, so their per-position "
                "mean is their own activation."
            )

    patched_ld = np.zeros((len(sites), n))
    patched_prob = np.zeros((len(sites), n))
    # Shares the arrays being filled, so per-layer callbacks can summarize finished sites.
    partial = EngineResult(
        sites=sites,
        layout=layout,
        prompts=prompts,
        baselines=baselines,
        receiver=receiver,
        reference=reference,
        patched_ld=patched_ld,
        patched_prob=patched_prob,
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
        for gi, group in enumerate(groups):
            receiver_tokens = group.clean if receiver == "clean" else group.corrupt
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
                    patch = _build_patch(
                        chunk,
                        kind,
                        layer,
                        all_pos,
                        baseline_kind,
                        sources,
                        gi,
                        group.length,
                        group_of,
                        local_of,
                        prompts,
                        info.d_model,
                        info.d_head,
                        model_dtype,
                        backend.device,
                    )
                    local_idx = torch.tensor([r.local for r in chunk], dtype=torch.long)
                    logits = backend.final_logits(receiver_tokens[local_idx], patch)
                    ld, prob = _metric(logits, *_answer_tensors(prompts, [r.prompt for r in chunk]))
                    for r, v_ld, v_prob in zip(chunk, ld, prob, strict=True):
                        patched_ld[r.site.index, r.prompt] += v_ld / k
                        patched_prob[r.site.index, r.prompt] += v_prob / k
                    done_rows += len(chunk)
                    if on_progress is not None:
                        on_progress(done_rows, total_rows, layer)
        if on_layer is not None:
            on_layer(layer, [s.index for s in layer_sites], partial)

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
) -> Patch:
    B = len(rows)
    is_head = kind == "head"
    heads = torch.tensor([r.site.head for r in rows], dtype=torch.long) if is_head else None
    positions = None
    if not all_pos:
        positions = torch.tensor(
            [resolve_position(r.site.site.position, prompts[r.prompt]) for r in rows],
            dtype=torch.long,
        )
    d = d_head if is_head else d_model

    if baseline_kind == "zero":
        shape = (B, length, d) if all_pos else (B, d)
        values = torch.zeros(shape, dtype=dtype, device=device)
        return Patch(kind=kind, layer=layer, values=values, heads=heads, positions=positions)

    assert sources is not None
    dev_heads = heads.to(device) if heads is not None else None

    if baseline_kind in ("patch", "resample"):
        if baseline_kind == "patch":
            src_prompts = [r.prompt for r in rows]
        else:
            src_prompts = [r.donor for r in rows]  # type: ignore[misc]
        if all_pos:
            # Patch pairs share a length; resample donors are drawn from the same length group.
            assert all(group_of[p] == group_index for p in src_prompts)
            acts = sources.group_tensor(kind, group_index)
            idx = torch.tensor([local_of[p] for p in src_prompts], dtype=torch.long, device=device)
            values = acts[idx, :, dev_heads] if is_head else acts[idx]
        else:
            table = sources.table(kind, rows[0].site)
            if any(r.site.site.position != rows[0].site.site.position for r in rows):
                # Rows with different positions (layer x position sweeps) gather from their own table.
                values = torch.stack(
                    [
                        _table_row(sources.table(kind, r.site), p, r.site.head if is_head else None)
                        for r, p in zip(rows, src_prompts, strict=True)
                    ]
                )
            else:
                idx = torch.tensor(src_prompts, dtype=torch.long, device=device)
                values = table[idx, dev_heads] if is_head else table[idx]
        return Patch(kind=kind, layer=layer, values=values, heads=heads, positions=positions)

    # Mean ablation.
    if all_pos:
        mean = sources.group_tensor(kind, group_index).float().mean(dim=0)  # [L, ...]
        if is_head:
            values = mean[:, dev_heads].permute(1, 0, 2)  # [B, L, d_head]
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
            out.append(m[r.site.head] if is_head else m)
        values = torch.stack(out)
    return Patch(kind=kind, layer=layer, values=values, heads=heads, positions=positions)


def _table_row(table: torch.Tensor, prompt: int, head: int | None) -> torch.Tensor:
    row = table[prompt]
    return row[head] if head is not None else row
