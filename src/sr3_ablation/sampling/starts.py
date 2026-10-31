# ADDED: new file, not part of the upstream SR3 codebase.
"""Initial state x_T of the reverse process.  # ADDED

SR3 (and every result in the report's main table) starts from pure Gaussian noise. The  # ADDED
forward-noised start is a diagnostic only: it draws x_T from q(x_T | x_0) with the bicubic  # ADDED
image standing in for the unknown x_0, i.e. from the distribution the model saw at t = T while  # ADDED
training.  # ADDED
"""  # ADDED

from __future__ import annotations  # ADDED

import math  # ADDED
from dataclasses import dataclass  # ADDED
from typing import Protocol  # ADDED

import torch  # ADDED


class StartState(Protocol):  # ADDED
    """Chooses the initial state x_T of the reverse process."""  # ADDED

    @property  # ADDED
    def name(self) -> str: ...  # ADDED

    def initial(self, condition: torch.Tensor) -> torch.Tensor:  # ADDED
        """x_T for a batch of conditioning images, drawn from the current global RNG."""  # ADDED
        ...  # ADDED


@dataclass(frozen=True)  # ADDED
class PureNoiseStart:  # ADDED
    """x_T ~ N(0, I), as in upstream ``p_sample_loop``."""  # ADDED

    name: str = "pure_noise"  # ADDED

    def initial(self, condition: torch.Tensor) -> torch.Tensor:  # ADDED
        return torch.randn(condition.shape, device=condition.device)  # ADDED


PURE_NOISE = PureNoiseStart()  # ADDED


@dataclass(frozen=True)  # ADDED
class ForwardNoisedStart:  # ADDED
    """x_T = sqrt(alpha_bar_T) * condition + sqrt(1 - alpha_bar_T) * eps, eps ~ N(0, I)."""  # ADDED

    alpha_bar_t: float  # ADDED
    name: str = "forward_noised"  # ADDED

    def __post_init__(self) -> None:  # ADDED
        if not 0.0 < self.alpha_bar_t < 1.0:  # ADDED
            raise ValueError(f"alpha_bar_T must lie in (0, 1), got {self.alpha_bar_t}.")  # ADDED

    @property  # ADDED
    def signal_coefficient(self) -> float:  # ADDED
        return math.sqrt(self.alpha_bar_t)  # ADDED

    @property  # ADDED
    def noise_coefficient(self) -> float:  # ADDED
        return math.sqrt(1.0 - self.alpha_bar_t)  # ADDED

    def initial(self, condition: torch.Tensor) -> torch.Tensor:  # ADDED
        noise = torch.randn(condition.shape, device=condition.device)  # ADDED
        return self.signal_coefficient * condition + self.noise_coefficient * noise  # ADDED
