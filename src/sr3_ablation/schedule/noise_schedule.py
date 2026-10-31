# ADDED: new file, not part of the upstream SR3 codebase.
"""Independent (NumPy-only) computation of a noise schedule and its derived quantities."""  # ADDED

from __future__ import annotations  # ADDED

from dataclasses import asdict, dataclass  # ADDED

import numpy as np  # ADDED

from sr3_ablation.config.schema import ScheduleConfig  # ADDED
from sr3_ablation.schedule.transforms import ScheduleTransform, assert_open_unit_interval  # ADDED


def base_betas(config: ScheduleConfig) -> np.ndarray:  # ADDED
    """Original schedule theta_t = beta_t for t = 1..T, matching upstream ``make_beta_schedule``."""  # ADDED
    if config.schedule == "linear":  # ADDED
        return np.linspace(  # ADDED
            config.linear_start,  # ADDED
            config.linear_end,  # ADDED
            config.n_timestep,  # ADDED
            dtype=np.float64,  # ADDED
        )  # ADDED
    if config.schedule == "quad":  # ADDED
        root = np.linspace(  # ADDED
            config.linear_start**0.5,  # ADDED
            config.linear_end**0.5,  # ADDED
            config.n_timestep,  # ADDED
            dtype=np.float64,  # ADDED
        )  # ADDED
        return root**2  # ADDED
    raise ValueError(f"Unsupported schedule {config.schedule!r}; use 'linear' or 'quad'.")  # ADDED


@dataclass(frozen=True)  # ADDED
class ScheduleSummary:  # ADDED
    """Headline numbers of a schedule, written to ``schedule.json`` for every run."""  # ADDED

    description: str  # ADDED
    exponent: float  # ADDED
    n_timestep: int  # ADDED
    beta_first: float  # ADDED
    beta_last: float  # ADDED
    alpha_bar_last: float  # ADDED
    snr_last: float  # ADDED
    first_t_snr_below_one: int | None  # ADDED

    def to_dict(self) -> dict[str, float | int | str | None]:  # ADDED
        return asdict(self)  # ADDED


class NoiseSchedule:  # ADDED
    """A per-step schedule beta_1..beta_T and the quantities derived from it.  # ADDED

    Timesteps are 1-indexed (t = 1..T) in every public method, as in the DDPM paper.  # ADDED
    """  # ADDED

    def __init__(self, betas: np.ndarray, exponent: float = 1.0, description: str = "") -> None:  # ADDED
        values = np.asarray(betas, dtype=np.float64)  # ADDED
        assert_open_unit_interval(values, name="beta_t")  # ADDED
        self._betas = values  # ADDED
        self._alphas_cumprod = np.cumprod(1.0 - values)  # ADDED
        self.exponent = exponent  # ADDED
        self.description = description  # ADDED

    @classmethod  # ADDED
    def from_config(cls, config: ScheduleConfig, transform: ScheduleTransform) -> NoiseSchedule:  # ADDED
        return cls(transform.apply(base_betas(config)), transform.exponent, transform.describe())  # ADDED

    @property  # ADDED
    def n_timestep(self) -> int:  # ADDED
        return int(self._betas.size)  # ADDED

    @property  # ADDED
    def betas(self) -> np.ndarray:  # ADDED
        return self._betas.copy()  # ADDED

    @property  # ADDED
    def alphas_cumprod(self) -> np.ndarray:  # ADDED
        return self._alphas_cumprod.copy()  # ADDED

    @property  # ADDED
    def sqrt_alphas_cumprod_prev(self) -> np.ndarray:  # ADDED
        """sqrt(alpha_bar) for t = 0..T (length T + 1), as used by SR3 for noise levels."""  # ADDED
        return np.sqrt(np.append(1.0, self._alphas_cumprod))  # ADDED

    @property  # ADDED
    def snr(self) -> np.ndarray:  # ADDED
        """Signal-to-noise ratio alpha_bar_t / (1 - alpha_bar_t) for t = 1..T."""  # ADDED
        return self._alphas_cumprod / (1.0 - self._alphas_cumprod)  # ADDED

    def alpha_bar_at(self, t: int) -> float:  # ADDED
        return float(self._alphas_cumprod[self._index(t)])  # ADDED

    def snr_at(self, t: int) -> float:  # ADDED
        return float(self.snr[self._index(t)])  # ADDED

    def first_t_snr_below_one(self) -> int | None:  # ADDED
        below = np.flatnonzero(self.snr < 1.0)  # ADDED
        return int(below[0]) + 1 if below.size else None  # ADDED

    def summary(self) -> ScheduleSummary:  # ADDED
        return ScheduleSummary(  # ADDED
            description=self.description,  # ADDED
            exponent=self.exponent,  # ADDED
            n_timestep=self.n_timestep,  # ADDED
            beta_first=float(self._betas[0]),  # ADDED
            beta_last=float(self._betas[-1]),  # ADDED
            alpha_bar_last=self.alpha_bar_at(self.n_timestep),  # ADDED
            snr_last=self.snr_at(self.n_timestep),  # ADDED
            first_t_snr_below_one=self.first_t_snr_below_one(),  # ADDED
        )  # ADDED

    def _index(self, t: int) -> int:  # ADDED
        if not 1 <= t <= self.n_timestep:  # ADDED
            raise IndexError(f"Timestep t={t} outside 1..{self.n_timestep}.")  # ADDED
        return t - 1  # ADDED
