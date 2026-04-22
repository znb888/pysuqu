"""Explicit opt-in boundary for unfinished qubit APIs.

The stable public surface is exported from :mod:`pysuqu.qubit`. This module is
reserved for placeholder-heavy or high-churn qubit surfaces that are kept for
exploration and old scripts, but are not yet part of the stable API contract.
"""

from importlib import import_module
from typing import Dict, Optional, Tuple


class QubitFeatureBoundaryError(NotImplementedError):
    """Raised when a compatibility or experimental qubit placeholder is used."""

    def __init__(
        self,
        owner: str,
        member: str,
        *,
        boundary: str,
        replacement: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        self.owner = owner
        self.member = member
        self.boundary = boundary
        self.replacement = replacement
        self.details = details

        article = "an" if boundary[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
        message = (
            f"{owner}.{member}() is {article} {boundary}-only qubit surface and is not "
            "implemented on the stable API."
        )
        if replacement:
            message += f" Prefer {replacement} for supported workflows."
        message += (
            " Import from pysuqu.qubit.experimental only when you intentionally "
            "opt into unfinished surfaces."
        )
        if details:
            message += f" {details}"
        super().__init__(message)


def raise_qubit_feature_boundary(
    owner: str,
    member: str,
    *,
    boundary: str,
    replacement: Optional[str] = None,
    details: Optional[str] = None,
) -> None:
    """Raise the shared boundary error with consistent user guidance."""

    raise QubitFeatureBoundaryError(
        owner,
        member,
        boundary=boundary,
        replacement=replacement,
        details=details,
    )


_EXPERIMENTAL_EXPORTS: Dict[str, Tuple[str, str]] = {
    "FGFGG1V1V3Coupling": ("pysuqu.qubit.multi", "FGFGG1V1V3Coupling"),
    "GroundedTransmonList": ("pysuqu.qubit.multi", "GroundedTransmonList"),
}


def __getattr__(name: str):
    if name not in _EXPERIMENTAL_EXPORTS:
        raise AttributeError(name)

    module_name, attr_name = _EXPERIMENTAL_EXPORTS[name]
    attr = getattr(import_module(module_name), attr_name)
    globals()[name] = attr
    return attr


def __dir__():
    return sorted(set(globals()) | set(_EXPERIMENTAL_EXPORTS))


__all__ = [
    "FGFGG1V1V3Coupling",
    "GroundedTransmonList",
    "QubitFeatureBoundaryError",
    "raise_qubit_feature_boundary",
]
