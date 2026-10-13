"""What a run that stops early keeps.

A sweep streams its results as it goes. One that goes layer by layer (patching, ablation, path
patching), steered site by steered site, or set by set reports each layer's (site's, set's) sites
once every prompt has run at them. Attribution patching and direct logit attribution measure every
site in one pass per batch of prompts, and report the prompts they have finished.

When a run is cancelled (in the app, or with Ctrl+C) or fails mid-sweep, the runner keeps those
rows: results.parquet holds them, and summary.json holds their statistics with ``partial``, which
says how far the run got and what it doesn't keep. Its status stays cancelled or failed.

Statistics come from the kept rows exactly as for a finished run, so a site kept by a run that
stopped has the numbers the whole run gives it: the same per-prompt values, the same resamples
and the same normalization. What depends on every site is left out or kept only where every site
it needs finished:

* simultaneous bands and q-values correct for every site of a sweep together, so a run that kept
  some of its sites leaves them out (a run that kept every site, over fewer prompts, has them);
* a set of sites measured against the set that replaces everything (its share, faithfulness,
  minimality, interactions) needs the sets it compares, so it is kept only where they finished;
* a steered site is kept with its random control, which runs with it.

A run that kept every site over the prompts that finished is a run of those prompts: its numbers
are what a finished run of just them gives. A sweep stopped after its last site (a steering run
records what the steered model says last) keeps every site, with everything the whole run gives
it. Methods that report nothing before they end (SAE features, and ACDC, whose circuit is
ACDC's only once every edge has been tested) keep nothing.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from logogram.engine import EngineResult, check_values
from logogram.results import build_summary, compute_stats, results_table, write_results
from logogram.sites import ResolvedSite
from logogram.spec import EdgeAttribution, Spec

CountsFn = Callable[[EngineResult], np.ndarray]


@dataclasses.dataclass
class Progress:
    """Follows a sweep as it streams, so a run that stops early can keep what finished."""

    spec: Spec | None = None
    model: dict[str, Any] | None = None
    counts: CountsFn | None = None
    # How many prompts the run measures.
    n_prompts: int = 0
    # Notes the run adds to its result's warnings (a version 1 spec's filled-in choices).
    notes: list[str] = dataclasses.field(default_factory=list)
    # How the prompts were wrapped in the model's chat template, for the summary to say.
    chat_template: dict[str, Any] | None = None
    sites: list[ResolvedSite] = dataclasses.field(default_factory=list)
    layout: dict[str, Any] | None = None
    # A sweep's result so far (it shares the arrays being filled), and the sites it finished.
    result: EngineResult | None = None
    done: list[int] = dataclasses.field(default_factory=list)
    # A method that measures every site at once: the prompts it finished, and the result over them.
    prompts: list[int] = dataclasses.field(default_factory=list)
    over: Callable[[], EngineResult] | None = None
    # The sweep returned: a run that fails after that (writing its files) keeps nothing.
    complete: bool = False

    def prepare(
        self,
        spec: Spec,
        model: dict[str, Any],
        counts: CountsFn,
        notes: list[str],
        n_prompts: int,
        chat_template: dict[str, Any] | None = None,
    ) -> None:
        self.spec, self.model, self.counts, self.notes = spec, model, counts, notes
        self.n_prompts, self.chat_template = n_prompts, chat_template

    def started(self, sites: list[Any], layout: dict[str, Any]) -> None:
        self.sites, self.layout = list(sites), layout
        self.result, self.done = None, []

    def finished_sites(self, indices: list[int], result: EngineResult) -> None:
        self.result = result
        seen = set(self.done)
        self.done += [i for i in indices if i not in seen]

    def finished_prompts(self, positions: list[int], build: Callable[[], EngineResult]) -> None:
        self.prompts, self.over = list(positions), build


def keep_finished(
    progress: Progress, folder: Path, run_id: str, status: str
) -> dict[str, Any] | None:
    """Write results.parquet and summary.json for what a run that stopped early finished, and
    return the summary; None when nothing finished. Raises when what finished can't be kept (its
    statistics can't be computed): the caller then keeps nothing."""
    spec, counts = progress.spec, progress.counts
    if progress.complete or spec is None or counts is None:
        return None
    if progress.done and progress.result is not None:
        sweep = progress.result
        kept = sorted(progress.done)
        values = sweep.delta if sweep.delta is not None else sweep.patched
        what = {"estimate": "estimates", "attribution": "terms"}.get(sweep.measure) or (
            "steered values" if "steering" in sweep.extra else "patched values"
        )
        notes = [
            *progress.notes,
            *sweep.warnings,
            *check_values(values[kept], what, spec.model.dtype),
        ]
        result = dataclasses.replace(sweep, warnings=notes)
        if len(kept) == len(sweep.sites):
            # Every site finished, but the sweep hadn't returned: what stopped the run came after
            # its last site (a steering run's continuations, say). Every site is kept, with
            # everything the whole run gives it, corrections for the number of sites included.
            kept = None
            stats = compute_stats(spec, result, counts(result))
            partial = _all_sites(spec, result, status)
        else:
            stats = compute_stats(spec, result, counts(result), kept)
            partial = _by_sites(spec, result, kept, status)
    elif progress.over is not None:
        # Every site, over the prompts that finished (this checks their gap as a run does).
        result = progress.over()
        result = dataclasses.replace(result, warnings=[*progress.notes, *result.warnings])
        kept = None
        stats = compute_stats(spec, result, counts(result))
        curve = isinstance(spec.experiment, EdgeAttribution) and spec.experiment.curve is not None
        partial = _by_prompts(result, progress.n_prompts, status, curve)
    else:
        return None
    summary = build_summary(
        spec,
        result,
        stats,
        run_id,
        progress.model,
        progress.chat_template,
        kept=kept,
        partial=partial,
    )
    write_results(folder / "results.parquet", results_table(result, stats, kept))
    if "edges" in result.extra:
        from logogram.edges import write_edge_table
        from logogram.schema import EDGES_FILE

        write_edge_table(folder / EDGES_FILE, result, stats)
    from logogram.runner import write_json

    write_json(folder / "summary.json", summary)
    return summary


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}" if n != 1 else f"1 {word[:-1]}"


def _units(result: EngineResult, kept: list[int]) -> tuple[str, int, int]:
    """What a sweep goes through (layers, sites or sets), how many of them the ``kept`` sites
    finish, and how many there are."""
    sites = result.sites
    layout = result.layout
    chosen = [sites[i] for i in kept]
    if layout.get("kind") == "steering":
        # One row per steered site, its strengths and their random controls run together.
        unit = "layers" if layout.get("row_title") == "Layer" else "sites"
        return unit, len({s.row for s in chosen}), len(layout.get("rows") or [])
    if layout.get("kind") == "site_sets":
        return "sets", len(chosen), len(sites)
    if layout.get("kind") == "sites":
        return "sites", len(chosen), len(sites)
    return "layers", len({s.layer for s in chosen}), len({s.layer for s in sites})


def _all_sites(spec: Spec, result: EngineResult, status: str) -> dict[str, Any]:
    """What a sweep kept that finished every site but stopped before it returned."""
    unit, done, total = _units(result, list(range(len(result.sites))))
    left_out: list[str] = []
    exp = spec.experiment
    if getattr(exp, "continuations", None) is not None:
        left_out.append("What the steered model says wasn't recorded: it comes last.")
    return {
        "status": status,
        "unit": unit,
        "done": done,
        "total": total,
        "sites": len(result.sites),
        "sites_total": len(result.sites),
        "prompts": len(result.prompts),
        "prompts_total": len(result.prompts),
        "note": f"Stopped before the run ended, once its {_plural(total, unit)} had finished.",
        "left_out": left_out,
    }


def _by_sites(spec: Spec, result: EngineResult, kept: list[int], status: str) -> dict[str, Any]:
    """What a sweep that finished some of its sites kept, and what it leaves out."""
    sites = result.sites
    layout = result.layout
    unit, done, total = _units(result, kept)
    left_out: list[str] = []
    if len(result.prompts) >= 2 and len(sites) > 1:
        left_out.append(
            f"Simultaneous bands and q-values are left out: they correct for all "
            f"{len(sites):,} sites of the sweep together, and {len(sites) - len(kept):,} of "
            "them didn't finish."
        )
    if layout.get("kind") == "site_sets":
        everything = next(
            (
                rs.index
                for rs in sites
                if rs.site_set is not None and rs.site_set.complement and not rs.site_set.sites
            ),
            None,
        )
        if everything is not None and everything not in kept:
            left_out.append(
                "Shares and faithfulness are left out: they compare each set with the set that "
                "replaces everything, which didn't finish."
            )
        left_out.append(
            "Minimality and interactions are kept only where every set they compare finished."
        )
    if layout.get("kind") == "steering":
        left_out.append("The comparison with the random control covers the sites that finished.")
    return {
        "status": status,
        "unit": unit,
        "done": done,
        "total": total,
        "sites": len(kept),
        "sites_total": len(sites),
        "prompts": len(result.prompts),
        "prompts_total": len(result.prompts),
        "note": f"Stopped after {done:,} of {_plural(total, unit)}: these are the {unit} that finished.",
        "left_out": left_out,
    }


def _by_prompts(
    result: EngineResult, total: int, status: str, curve: bool = False
) -> dict[str, Any]:
    """What a method that measures every site at once kept: every site, over the prompts that
    finished, of ``total``. ``curve``: the run would have measured a faithfulness curve after
    estimating every edge."""
    done = len(result.prompts)
    rest = total - done
    others = "The other prompt wasn't" if rest == 1 else f"The other {rest:,} prompts weren't"
    left_out = (
        [
            f"{others} measured. Prompts run in groups of the same token length, so the ones "
            "that finished aren't a random sample of the dataset.",
        ]
        if rest
        else []
    )
    if "direct" in result.extra:
        left_out.append("The split of the logit difference is over the prompts that finished.")
    if curve and "curve" not in result.extra.get("edges", {}):
        left_out.append(
            "The faithfulness curve is left out: it runs once every edge is estimated on every "
            "prompt, and it didn't finish."
        )
    return {
        "status": status,
        "unit": "prompts",
        "done": done,
        "total": total,
        "sites": len(result.sites),
        "sites_total": len(result.sites),
        "prompts": done,
        "prompts_total": total,
        "note": (
            f"Stopped after {done:,} of {_plural(total, 'prompts')}: every site is measured on "
            "the prompts that finished."
        ),
        "left_out": left_out,
    }
