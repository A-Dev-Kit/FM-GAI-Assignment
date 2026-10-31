# ADDED: new file, not part of the upstream SR3 codebase.
"""Strategies that turn the original schedule theta_t into the schedule a run uses."""  # ADDED

from __future__ import annotations  # ADDED

from dataclasses import dataclass  # ADDED
from typing import Protocol  # ADDED

import numpy as np  # ADDED

EXPONENT_LOWER = 1.5  # ADDED
EXPONENT_UPPER = 3.0  # ADDED


class ScheduleTransform(Protocol):  # ADDED
    """Maps the original per-step schedule theta_t to the schedule actually used."""  # ADDED

    @property  # ADDED
    def exponent(self) -> float: ...  # ADDED

    def apply(self, theta: np.ndarray) -> np.ndarray: ...  # ADDED

    def describe(self) -> str: ...  # ADDED


class IdentityTransform:  # ADDED
    """Leaves the schedule unchanged (Run 0 and Run A)."""  # ADDED

    @property  # ADDED
    def exponent(self) -> float:  # ADDED
        return 1.0  # ADDED

    def apply(self, theta: np.ndarray) -> np.ndarray:  # ADDED
        return np.array(theta, dtype=np.float64, copy=True)  # ADDED

    def describe(self) -> str:  # ADDED
        return "theta_t (original schedule)"  # ADDED


@dataclass(frozen=True)  # ADDED
class PowerTransform:  # ADDED
    """theta_t -> theta_t ** x with x in the open interval (1.5, 3.0) (Run B)."""  # ADDED

    exponent: float  # ADDED

    def __post_init__(self) -> None:  # ADDED
        if not EXPONENT_LOWER < self.exponent < EXPONENT_UPPER:  # ADDED
            raise ValueError(  # ADDED
                f"Exponent x must lie in ({EXPONENT_LOWER}, {EXPONENT_UPPER}); got {self.exponent}."  # ADDED
            )  # ADDED

    def apply(self, theta: np.ndarray) -> np.ndarray:  # ADDED
        values = np.asarray(theta, dtype=np.float64)  # ADDED
        assert_open_unit_interval(values)  # ADDED
        return values**self.exponent  # ADDED

    def describe(self) -> str:  # ADDED
        return f"theta_t ** {self.exponent:.2f}"  # ADDED


@dataclass(frozen=True)  # ADDED
class AlphaBarPowerTransform:  # ADDED
    """Raise cumulative alpha_bar_t to the power x, then re-derive beta_t (supplementary Run S)."""  # ADDED

    exponent: float  # ADDED

    def __post_init__(self) -> None:  # ADDED
        if not EXPONENT_LOWER < self.exponent < EXPONENT_UPPER:  # ADDED
            raise ValueError(  # ADDED
                f"Exponent x must lie in ({EXPONENT_LOWER}, {EXPONENT_UPPER}); got {self.exponent}."  # ADDED
            )  # ADDED

    def apply(self, theta: np.ndarray) -> np.ndarray:  # ADDED
        betas = np.asarray(theta, dtype=np.float64)  # ADDED
        assert_open_unit_interval(betas, name="beta_t")  # ADDED
        alphas_cumprod = np.cumprod(1.0 - betas)  # ADDED
        powered = alphas_cumprod**self.exponent  # ADDED
        previous = np.append(1.0, powered[:-1])  # ADDED
        modified = 1.0 - powered / previous  # ADDED
        assert_open_unit_interval(modified, name="beta_t")  # ADDED
        return modified  # ADDED

    def describe(self) -> str:  # ADDED
        return f"alpha_bar_t ** {self.exponent:.2f}"  # ADDED


def transform_for(exponent: float, *, power_on: str = "beta") -> ScheduleTransform:  # ADDED
    """Select the strategy for ``exponent`` (1.0 means the original schedule)."""  # ADDED
    if exponent == 1.0:  # ADDED
        return IdentityTransform()  # ADDED
    if power_on == "alpha_bar":  # ADDED
        return AlphaBarPowerTransform(exponent)  # ADDED
    return PowerTransform(exponent)  # ADDED


def assert_open_unit_interval(values: np.ndarray, name: str = "theta_t") -> None:  # ADDED
    """Raise ``ValueError`` unless every value lies strictly inside (0, 1)."""  # ADDED
    if not np.all((values > 0.0) & (values < 1.0)):  # ADDED
        raise ValueError(  # ADDED
            f"{name} must lie in the open interval (0, 1); "  # ADDED
            f"observed range [{values.min():.3e}, {values.max():.3e}]."  # ADDED
        )  # ADDED
