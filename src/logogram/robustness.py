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
