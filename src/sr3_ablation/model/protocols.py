# ADDED: new file, not part of the upstream SR3 codebase.
"""Narrow interfaces the rest of the package depends on instead of the SR3 classes.

Training needs :class:`TrainableModel`, sampling needs :class:`Denoiser`, schedule checks need
:class:`ScheduleHolder` and checkpointing needs :class:`WeightStore`. :class:`DiffusionBackend`
combines them for code that builds a model.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Protocol

import torch
from torch import nn

from sr3_ablation.config.schema import ScheduleConfig
from sr3_ablation.schedule.verification import ScheduleBuffers

StateDict = dict[str, torch.Tensor]


class ModeSwitchable(Protocol):
    @property
    def device(self) -> torch.device: ...

    @property
    def is_training(self) -> bool: ...

    def train_mode(self) -> None: ...

    def eval_mode(self) -> None: ...


class TrainableModel(ModeSwitchable, Protocol):
    def parameters(self) -> Iterator[nn.Parameter]: ...

    def training_loss(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Mean per-element loss of one batch with keys ``HR`` and ``SR``."""
        ...


class Denoiser(ModeSwitchable, Protocol):
    @property
    def num_timesteps(self) -> int: ...

    def reverse_step(
        self, x_t: torch.Tensor, step_index: int, condition: torch.Tensor
    ) -> torch.Tensor:
        """One reverse step: x at timestep ``step_index + 1`` -> x at timestep ``step_index``."""
        ...


class ScheduleHolder(Protocol):
    def set_schedule(self, config: ScheduleConfig, exponent: float) -> None: ...

    def schedule_buffers(self) -> ScheduleBuffers: ...


class WeightStore(Protocol):
    def denoiser_state(self) -> StateDict: ...

    def load_denoiser_state(self, state: StateDict) -> None: ...


class DiffusionBackend(TrainableModel, Denoiser, ScheduleHolder, WeightStore, Protocol):
    """Everything a conditional diffusion super-resolution model offers to this package."""
