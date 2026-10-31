# ADDED: new file, not part of the upstream SR3 codebase.
"""Narrow interfaces the rest of the package depends on instead of the SR3 classes.  # ADDED

Training needs :class:`TrainableModel`, sampling needs :class:`Denoiser`, schedule checks need  # ADDED
:class:`ScheduleHolder` and checkpointing needs :class:`WeightStore`. :class:`DiffusionBackend`  # ADDED
combines them for code that builds a model.  # ADDED
"""  # ADDED

from __future__ import annotations  # ADDED

from collections.abc import Iterator, Mapping  # ADDED
from typing import Protocol  # ADDED

import torch  # ADDED
from torch import nn  # ADDED

from sr3_ablation.config.schema import ScheduleConfig  # ADDED
from sr3_ablation.runs import RunSpec  # ADDED
from sr3_ablation.schedule.verification import ScheduleBuffers  # ADDED

StateDict = dict[str, torch.Tensor]  # ADDED


class ModeSwitchable(Protocol):  # ADDED
    """A model on a device that switches between train and eval mode."""  # ADDED

    @property  # ADDED
    def device(self) -> torch.device: ...  # ADDED

    @property  # ADDED
    def is_training(self) -> bool: ...  # ADDED

    def train_mode(self) -> None: ...  # ADDED

    def eval_mode(self) -> None: ...  # ADDED


class TrainableModel(ModeSwitchable, Protocol):  # ADDED
    """What the fine-tuning loop needs: parameters and a training loss."""  # ADDED

    def parameters(self) -> Iterator[nn.Parameter]: ...  # ADDED

    def training_loss(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:  # ADDED
        """Mean per-element loss of one batch with keys ``HR`` and ``SR``."""  # ADDED
        ...  # ADDED


class Denoiser(ModeSwitchable, Protocol):  # ADDED
    """What the reverse sampler needs: the horizon T and one reverse step."""  # ADDED

    @property  # ADDED
    def num_timesteps(self) -> int: ...  # ADDED

    def reverse_step(  # ADDED
        self,  # ADDED
        x_t: torch.Tensor,  # ADDED
        step_index: int,  # ADDED
        condition: torch.Tensor,  # ADDED
    ) -> torch.Tensor:  # ADDED
        """One reverse step: x at timestep ``step_index + 1`` -> x at timestep ``step_index``."""  # ADDED
        ...  # ADDED


class ScheduleHolder(Protocol):  # ADDED
    """A model whose noise schedule can be installed and read back."""  # ADDED

    def set_schedule(self, config: ScheduleConfig, run: RunSpec) -> None: ...  # ADDED

    def schedule_buffers(self) -> ScheduleBuffers: ...  # ADDED


class WeightStore(Protocol):  # ADDED
    """A model whose denoiser weights can be saved and restored."""  # ADDED

    def denoiser_state(self) -> StateDict: ...  # ADDED

    def load_denoiser_state(self, state: StateDict) -> None: ...  # ADDED


class DiffusionBackend(TrainableModel, Denoiser, ScheduleHolder, WeightStore, Protocol):  # ADDED
    """Everything a conditional diffusion super-resolution model offers to this package."""  # ADDED
