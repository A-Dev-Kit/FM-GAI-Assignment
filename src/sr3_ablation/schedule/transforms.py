# ADDED: new file, not part of the upstream SR3 codebase.
"""Strategies that turn the original schedule theta_t into the schedule a run uses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

EXPONENT_LOWER = 1.5
EXPONENT_UPPER = 3.0


class ScheduleTransform(Protocol):
    """Maps the original per-step schedule theta_t to the schedule actually used."""

    @property
    def exponent(self) -> float: ...

    def apply(self, theta: np.ndarray) -> np.ndarray: ...

    def describe(self) -> str: ...


class IdentityTransform:
    """Leaves the schedule unchanged (Run 0 and Run A)."""

    @property
    def exponent(self) -> float:
        return 1.0

    def apply(self, theta: np.ndarray) -> np.ndarray:
        return np.array(theta, dtype=np.float64, copy=True)

    def describe(self) -> str:
        return "theta_t (original schedule)"


@dataclass(frozen=True)
class PowerTransform:
    """theta_t -> theta_t ** x with x in the open interval (1.5, 3.0) (Run B)."""

    exponent: float

    def __post_init__(self) -> None:
        if not EXPONENT_LOWER < self.exponent < EXPONENT_UPPER:
            raise ValueError(
                f"Exponent x must lie in ({EXPONENT_LOWER}, {EXPONENT_UPPER}); got {self.exponent}."
            )

    def apply(self, theta: np.ndarray) -> np.ndarray:
        values = np.asarray(theta, dtype=np.float64)
        assert_open_unit_interval(values)
        return values**self.exponent

    def describe(self) -> str:
        return f"theta_t ** {self.exponent:.2f}"


def transform_for(exponent: float) -> ScheduleTransform:
    """Select the strategy for ``exponent`` (1.0 means the original schedule)."""
    return IdentityTransform() if exponent == 1.0 else PowerTransform(exponent)


def assert_open_unit_interval(values: np.ndarray, name: str = "theta_t") -> None:
    """Raise ``ValueError`` unless every value lies strictly inside (0, 1)."""
    if not np.all((values > 0.0) & (values < 1.0)):
        raise ValueError(
            f"{name} must lie in the open interval (0, 1); "
            f"observed range [{values.min():.3e}, {values.max():.3e}]."
        )
