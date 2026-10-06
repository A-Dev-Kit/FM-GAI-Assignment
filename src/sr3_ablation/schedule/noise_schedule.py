# ADDED: new file, not part of the upstream SR3 codebase.
"""Independent (NumPy-only) computation of a noise schedule and its derived quantities."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from sr3_ablation.config.schema import ScheduleConfig
from sr3_ablation.schedule.transforms import ScheduleTransform, assert_open_unit_interval


def base_betas(config: ScheduleConfig) -> np.ndarray:
    """Original schedule theta_t = beta_t for t = 1..T, matching upstream ``make_beta_schedule``."""
    if config.schedule == "linear":
        return np.linspace(
            config.linear_start, config.linear_end, config.n_timestep, dtype=np.float64
        )
    if config.schedule == "quad":
        root = np.linspace(
            config.linear_start**0.5, config.linear_end**0.5, config.n_timestep, dtype=np.float64
        )
        return root**2
    raise ValueError(f"Unsupported schedule {config.schedule!r}; use 'linear' or 'quad'.")


@dataclass(frozen=True)
class ScheduleSummary:
    """Headline numbers of a schedule, written to ``schedule.json`` for every run."""

    description: str
    exponent: float
    n_timestep: int
    beta_first: float
    beta_last: float
    alpha_bar_last: float
    snr_last: float
    first_t_snr_below_one: int | None

    def to_dict(self) -> dict[str, float | int | str | None]:
        return asdict(self)


class NoiseSchedule:
    """A per-step schedule beta_1..beta_T and the quantities derived from it.

    Timesteps are 1-indexed (t = 1..T) in every public method, as in the DDPM paper.
    """

    def __init__(self, betas: np.ndarray, exponent: float = 1.0, description: str = "") -> None:
        values = np.asarray(betas, dtype=np.float64)
        assert_open_unit_interval(values, name="beta_t")
        self._betas = values
        self._alphas_cumprod = np.cumprod(1.0 - values)
        self.exponent = exponent
        self.description = description

    @classmethod
    def from_config(cls, config: ScheduleConfig, transform: ScheduleTransform) -> NoiseSchedule:
        return cls(transform.apply(base_betas(config)), transform.exponent, transform.describe())

    @property
    def n_timestep(self) -> int:
        return int(self._betas.size)

    @property
    def betas(self) -> np.ndarray:
        return self._betas.copy()

    @property
    def alphas_cumprod(self) -> np.ndarray:
        return self._alphas_cumprod.copy()

    @property
    def sqrt_alphas_cumprod_prev(self) -> np.ndarray:
        """sqrt(alpha_bar) for t = 0..T (length T + 1), as used by SR3 for noise levels."""
        return np.sqrt(np.append(1.0, self._alphas_cumprod))

    @property
    def snr(self) -> np.ndarray:
        """Signal-to-noise ratio alpha_bar_t / (1 - alpha_bar_t) for t = 1..T."""
        return self._alphas_cumprod / (1.0 - self._alphas_cumprod)

    def alpha_bar_at(self, t: int) -> float:
        return float(self._alphas_cumprod[self._index(t)])

    def snr_at(self, t: int) -> float:
        return float(self.snr[self._index(t)])

    def first_t_snr_below_one(self) -> int | None:
        below = np.flatnonzero(self.snr < 1.0)
        return int(below[0]) + 1 if below.size else None

    def summary(self) -> ScheduleSummary:
        return ScheduleSummary(
            description=self.description,
            exponent=self.exponent,
            n_timestep=self.n_timestep,
            beta_first=float(self._betas[0]),
            beta_last=float(self._betas[-1]),
            alpha_bar_last=self.alpha_bar_at(self.n_timestep),
            snr_last=self.snr_at(self.n_timestep),
            first_t_snr_below_one=self.first_t_snr_below_one(),
        )

    def _index(self, t: int) -> int:
        if not 1 <= t <= self.n_timestep:
            raise IndexError(f"Timestep t={t} outside 1..{self.n_timestep}.")
        return t - 1
