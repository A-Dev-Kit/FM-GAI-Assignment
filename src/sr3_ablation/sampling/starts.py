# ADDED: new file, not part of the upstream SR3 codebase.
"""Initial state x_T of the reverse process.

SR3 (and every result in the report's main table) starts from pure Gaussian noise. The
forward-noised start is a diagnostic only: it draws x_T from q(x_T | x_0) with the bicubic
image standing in for the unknown x_0, i.e. from the distribution the model saw at t = T while
training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch


class StartState(Protocol):
    """Chooses the initial state x_T of the reverse process."""

    @property
    def name(self) -> str: ...

    def initial(self, condition: torch.Tensor) -> torch.Tensor:
        """x_T for a batch of conditioning images, drawn from the current global RNG."""
        ...


@dataclass(frozen=True)
class PureNoiseStart:
    """x_T ~ N(0, I), as in upstream ``p_sample_loop``."""

    name: str = "pure_noise"

    def initial(self, condition: torch.Tensor) -> torch.Tensor:
        return torch.randn(condition.shape, device=condition.device)


PURE_NOISE = PureNoiseStart()


@dataclass(frozen=True)
class ForwardNoisedStart:
    """x_T = sqrt(alpha_bar_T) * condition + sqrt(1 - alpha_bar_T) * eps, eps ~ N(0, I)."""

    alpha_bar_t: float
    name: str = "forward_noised"

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha_bar_t < 1.0:
            raise ValueError(f"alpha_bar_T must lie in (0, 1), got {self.alpha_bar_t}.")

    @property
    def signal_coefficient(self) -> float:
        return math.sqrt(self.alpha_bar_t)

    @property
    def noise_coefficient(self) -> float:
        return math.sqrt(1.0 - self.alpha_bar_t)

    def initial(self, condition: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(condition.shape, device=condition.device)
        return self.signal_coefficient * condition + self.noise_coefficient * noise
