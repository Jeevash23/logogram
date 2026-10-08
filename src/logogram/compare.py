"""Compare two runs: the robustness check and side-by-side comparisons share this.

Sites are matched by (kind, layer, head, position). The comparison reports Spearman's rank
correlation of mean effects, the overlap of the top-k components (ranked by |effect|), and the
components whose conclusion changed: their sign reversed (both confidence intervals exclude zero,
on opposite sides) or they entered or left the top k.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from logogram.spec import Spec
from logogram.stats import spearman


class CompareError(ValueError):
    pass


def site_key(site: dict[str, Any]) -> tuple[Any, ...]:
    return (
        site["kind"],
        site["layer"],
        site.get("head"),
        site["position_key"],
        site.get("variant_key"),
    )


def _excludes_zero(effect: dict[str, Any]) -> bool:
    lo, hi = effect.get("lo"), effect.get("hi")
    if lo is None or hi is None or lo >= hi:
        return False  # no interval (n < 2), or a degenerate one, is never confident
    return lo > 0 or hi < 0


def _ranks_by_magnitude(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.abs(values), kind="mergesort")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def default_top_k(n_sites: int) -> int:
    """About the top fifth of the sites, at most 10, so membership can actually change."""
    return max(1, min(10, math.ceil(n_sites / 5)))


def spec_differences(
    a: dict[str, Any], b: dict[str, Any], prefix: str = ""
) -> list[dict[str, Any]]:
    """Paths where two spec dicts differ, ignoring the name and notes."""
    out: list[dict[str, Any]] = []
    for key in sorted(set(a) | set(b)):
        if not prefix and key in ("name", "notes"):
            continue
        path = f"{prefix}.{key}" if prefix else key
        va, vb = a.get(key), b.get(key)
        if isinstance(va, dict) and isinstance(vb, dict):
            out.extend(spec_differences(va, vb, path))
        elif va != vb:
            out.append({"path": path, "a": va, "b": vb})
    return out


def compare_summaries(
    summary_a: dict[str, Any],
    summary_b: dict[str, Any],
    spec_a: Spec | None = None,
    spec_b: Spec | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    sites_a = {site_key(s): s for s in summary_a["sites"]}
    sites_b = {site_key(s): s for s in summary_b["sites"]}
    # Sites without a value in either run (an undefined statistic) can't be ranked.
    common = [
        k
        for k in sites_a
        if k in sites_b
        and sites_a[k]["effect"]["mean"] is not None
        and sites_b[k]["effect"]["mean"] is not None
    ]
    if not common:
        raise CompareError(
            "These runs share no sites (for example a head sweep and a layer × position sweep), "
            "so they can't be compared."
        )
    ea = np.array([sites_a[k]["effect"]["mean"] for k in common], dtype=np.float64)
    eb = np.array([sites_b[k]["effect"]["mean"] for k in common], dtype=np.float64)
    k = top_k or default_top_k(len(common))
    rank_a, rank_b = _ranks_by_magnitude(ea), _ranks_by_magnitude(eb)
    top_a = {i for i in range(len(common)) if rank_a[i] <= k}
    top_b = {i for i in range(len(common)) if rank_b[i] <= k}

    changes = []
    flagged = []
    for i, key in enumerate(common):
        sa, sb = sites_a[key], sites_b[key]
        flags = []
        # A reversal only counts when both runs are confident about the sign.
        if (
            np.sign(ea[i]) != np.sign(eb[i])
            and _excludes_zero(sa["effect"])
            and _excludes_zero(sb["effect"])
        ):
            flags.append("sign")
        if i in top_a and i not in top_b:
            flags.append("left_top")
        if i in top_b and i not in top_a:
            flags.append("entered_top")
        entry = {
            "label": sa["label"],
            "kind": sa["kind"],
            "layer": sa["layer"],
            "head": sa.get("head"),
            "position_key": sa["position_key"],
            "variant_key": sa.get("variant_key"),
            "index_a": sa["index"],
            "index_b": sb["index"],
            "row": sa["row"],
            "col": sa["col"],
            "effect_a": sa["effect"],
            "effect_b": sb["effect"],
            "rank_a": int(rank_a[i]),
            "rank_b": int(rank_b[i]),
            "flags": flags,
        }
        if flags or i in top_a or i in top_b:
            changes.append(entry)
        if flags:
            flagged.append(sa["index"])
    changes.sort(key=lambda c: (min(c["rank_a"], c["rank_b"]), c["rank_a"]))

    same_layout = (
        summary_a["layout"]["kind"] == summary_b["layout"]["kind"]
        and len(summary_a["layout"]["rows"]) == len(summary_b["layout"]["rows"])
        and len(summary_a["layout"]["cols"]) == len(summary_b["layout"]["cols"])
    )
    diff = [
        {
            "index_a": sites_a[key]["index"],
            "row": sites_a[key]["row"],
            "col": sites_a[key]["col"],
            "value": float(eb[i] - ea[i]),
        }
        for i, key in enumerate(common)
    ]
    differences = []
    if spec_a is not None and spec_b is not None:
        differences = spec_differences(
            spec_a.model_dump(mode="json"), spec_b.model_dump(mode="json")
        )
    rho = spearman(ea, eb)
    return {
        "run_a": summary_a["run_id"],
        "run_b": summary_b["run_id"],
        "n_common": len(common),
        "n_a": len(sites_a),
        "n_b": len(sites_b),
        "spearman": rho if np.isfinite(rho) else None,
        "top_k": k,
        "top_overlap": len(top_a & top_b),
        "top_a": [sites_a[common[i]]["label"] for i in sorted(top_a, key=lambda i: rank_a[i])],
        "top_b": [sites_b[common[i]]["label"] for i in sorted(top_b, key=lambda i: rank_b[i])],
        "changes": changes,
        "flagged": flagged,
        "n_sign_changes": sum(1 for c in changes if "sign" in c["flags"]),
        "diff": diff,
        "same_layout": same_layout,
        "spec_differences": differences,
    }
