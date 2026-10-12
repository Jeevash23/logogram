"""Verify an estimate: patch its strongest estimated sites (or edges) for real."""

from __future__ import annotations

from typing import Any

from logogram.spec import (
    NAME_MAX,
    ActivationPatching,
    AttributionPatching,
    EdgeAttribution,
    EdgesScope,
    Site,
    SitesScope,
    Spec,
)


def verification_spec(spec: Spec, summary: dict[str, Any], top: int) -> Spec:
    """The same spec with activation patching in the same direction, at the ``top`` sites with the
    largest estimated effects (by magnitude), so estimate and measurement can be compared. An edge
    attribution run's strongest edges are measured one at a time by path patching instead (see
    logogram.edges.edge_verification_spec)."""
    exp = spec.experiment
    if isinstance(exp, EdgeAttribution):
        from logogram.edges import edge_verification_spec

        return edge_verification_spec(spec, summary, top)
    if not isinstance(exp, AttributionPatching):
        raise ValueError("Only attribution patching runs are verified by patching.")
    ranked = sorted(
        (s for s in summary["sites"] if s["effect"]["mean"] is not None),
        key=lambda s: (-abs(s["effect"]["mean"]), s["index"]),
    )[:top]
    if not ranked:
        raise ValueError("This run has no estimated effects to verify.")
    sites = [
        Site(
            kind=s["kind"],
            layer=s["layer"],
            head=s["head"],
            feature=s.get("feature"),
            position=s["position"],
        )
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


def verification_change(spec: Spec) -> str:
    """What a verification spec measures, in words, for the run's provenance."""
    scope = spec.scope
    if isinstance(scope, EdgesScope):
        n = len(scope.edges)
        return (
            f"the {n} strongest estimated edge{'s' if n != 1 else ''}, path-patched one at a time"
        )
    assert isinstance(scope, SitesScope)
    n = len(scope.sites)
    return f"the {n} strongest estimated site{'s' if n != 1 else ''}, patched"
