"""Verify an attribution patching run: patch its strongest estimated sites for real."""

from __future__ import annotations

from typing import Any

from logogram.spec import NAME_MAX, ActivationPatching, AttributionPatching, Site, SitesScope, Spec


def verification_spec(spec: Spec, summary: dict[str, Any], top: int) -> Spec:
    """The same spec with activation patching in the same direction, at the ``top`` sites with the
    largest estimated effects (by magnitude), so estimate and measurement can be compared."""
    exp = spec.experiment
    if not isinstance(exp, AttributionPatching):
        raise ValueError("Only attribution patching runs are verified by patching.")
    ranked = sorted(
        (s for s in summary["sites"] if s["effect"]["mean"] is not None),
        key=lambda s: (-abs(s["effect"]["mean"]), s["index"]),
    )[:top]
    if not ranked:
        raise ValueError("This run has no estimated effects to verify.")
    sites = [
        Site(kind=s["kind"], layer=s["layer"], head=s["head"], position=s["position"])
        for s in ranked
    ]
    suffix = " · verified"
    return spec.model_copy(
        update={
            "name": spec.name[: NAME_MAX - len(suffix)].rstrip() + suffix,
            "experiment": ActivationPatching(direction=exp.direction),
            "scope": SitesScope(sites=sites),
        }
    )
