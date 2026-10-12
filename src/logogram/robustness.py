"""Robustness checks: the same spec with one methodological choice changed.

The app's Check robustness, ``logogram robustness`` and the Python API all derive the variant
here, so the three change exactly the same thing.
"""

from __future__ import annotations

from typing import Any

from logogram.spec import NAME_MAX, Spec, describe_intervention

SUFFIX = " · robustness"


class RobustnessError(ValueError):
    pass


# How a run that hasn't finished stands, in words.
_UNFINISHED = {
    "running": "is still running",
    "draft": "hasn't run",
    "failed": "failed",
    "cancelled": "was cancelled",
}


def require_finished(project: Any, run_id: str, action: str) -> None:
    """Refuse ``action`` ("Checking robustness", "Verifying by patching") on a run that hasn't
    finished. One that stopped early kept only part of its sweep (see logogram.partial), so what
    would be checked isn't all there."""
    from logogram.schema import stopped_text

    listing = project.run_listing(run_id)
    if listing is None:
        raise RobustnessError(f"There is no run {run_id} in this project.")
    if listing.status == "finished":
        return
    if listing.partial is not None:
        raise RobustnessError(
            f"{action} needs a finished run, and this one {stopped_text(listing.partial)}. Rerun "
            "it to the end first."
        )
    state = _UNFINISHED.get(listing.status, listing.status)
    raise RobustnessError(f"{action} needs a finished run, and this one {state}.")


def robustness_spec(
    original: Spec, *, experiment: dict[str, Any] | None = None, dtype: str | None = None
) -> tuple[Spec, str]:
    """``original`` with either its experiment or its model's dtype changed, and the change in
    words for the run's provenance."""
    if (experiment is None) == (dtype is None):
        raise RobustnessError("Change either the experiment or the dtype.")
    data = original.model_dump(mode="json")
    if dtype is not None:
        if dtype == original.model.dtype:
            raise RobustnessError(f"This run already ran in {dtype}.")
        # The same run in another precision: what rounding changed.
        data["model"]["dtype"] = dtype
    else:
        data["experiment"] = experiment
    data["name"] = original.name[: NAME_MAX - len(SUFFIX)].rstrip() + SUFFIX
    variant = Spec.model_validate(data)
    if variant.model_dump(mode="json") == original.model_dump(mode="json") | {"name": variant.name}:
        raise RobustnessError("This changes nothing: choose another experiment.")
    change = (
        f"In {dtype} instead of {original.model.dtype}"
        if dtype is not None
        else describe_intervention(variant.experiment)
    )
    return variant, change
