# ADDED: new file, not part of the upstream SR3 codebase.
"""Adapter around the vendored upstream SR3 ``GaussianDiffusion``.

This is the only module that imports from ``third_party/sr3``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from types import ModuleType
from typing import Any

import numpy as np
import torch
from torch import nn

from sr3_ablation.config.schema import ModelConfig, ScheduleConfig
from sr3_ablation.model.protocols import StateDict
from sr3_ablation.paths import UPSTREAM_SR3_DIR
from sr3_ablation.schedule.verification import ScheduleBuffers


def upstream_networks() -> ModuleType:
    """Import upstream ``model.networks`` (the vendored folder is put on ``sys.path``)."""
    upstream = str(UPSTREAM_SR3_DIR)
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    from model import networks  # type: ignore[import-not-found]

    return networks


def upstream_options(config: ModelConfig, schedule: ScheduleConfig) -> dict[str, Any]:
    """The subset of SR3's JSON options that ``networks.define_G`` reads."""
    unet = config.unet
    schedule_options = schedule.to_upstream()
    return {
        # "val" skips define_G's orthogonal re-initialisation; weights come from a checkpoint.
        "phase": "val",
        "gpu_ids": None,
        "distributed": False,
        "model": {
            "which_model_G": "sr3",
            "unet": {
                "in_channel": unet.in_channel,
                "out_channel": unet.out_channel,
                "inner_channel": unet.inner_channel,
                "norm_groups": unet.norm_groups,
                "channel_multiplier": list(unet.channel_multiplier),
                "attn_res": list(unet.attn_res),
                "res_blocks": unet.res_blocks,
                "dropout": unet.dropout,
            },
            "beta_schedule": {"train": schedule_options, "val": schedule_options},
            "diffusion": {
                "image_size": config.image_size,
                "channels": config.channels,
                "conditional": True,
            },
        },
    }


class Sr3Backend:
    """:class:`~sr3_ablation.model.protocols.DiffusionBackend` backed by upstream SR3."""

    def __init__(self, network: nn.Module, device: torch.device) -> None:
        self._network = network.to(device)
        self._network.set_loss(device)
        self._device = device

    @classmethod
    def create(
        cls, config: ModelConfig, schedule: ScheduleConfig, device: torch.device
    ) -> Sr3Backend:
        network = upstream_networks().define_G(upstream_options(config, schedule))
        return cls(network, device)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def network(self) -> nn.Module:
        return self._network

    @property
    def is_training(self) -> bool:
        return self._network.training

    def train_mode(self) -> None:
        self._network.train()

    def eval_mode(self) -> None:
        self._network.eval()

    def parameters(self) -> Iterator[nn.Parameter]:
        return self._network.parameters()

    def training_loss(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        hr = batch["HR"].to(self._device, non_blocking=True)
        sr = batch["SR"].to(self._device, non_blocking=True)
        loss_sum = self._network({"HR": hr, "SR": sr})
        # Same normalisation as upstream DDPM.optimize_parameters: summed L1 / (b * c * h * w).
        return loss_sum / hr.numel()

    @property
    def num_timesteps(self) -> int:
        if not hasattr(self._network, "num_timesteps"):
            raise RuntimeError("No noise schedule set; call set_schedule() first.")
        return int(self._network.num_timesteps)

    def reverse_step(
        self, x_t: torch.Tensor, step_index: int, condition: torch.Tensor
    ) -> torch.Tensor:
        return self._network.p_sample(x_t, step_index, condition_x=condition)

    def set_schedule(self, config: ScheduleConfig, exponent: float) -> None:
        self._network.set_new_noise_schedule(config.to_upstream(exponent), self._device)

    def schedule_buffers(self) -> ScheduleBuffers:
        return ScheduleBuffers(
            betas=self._network.betas.detach().cpu().double().numpy(),
            alphas_cumprod=self._network.alphas_cumprod.detach().cpu().double().numpy(),
            sqrt_alphas_cumprod_prev=np.asarray(
                self._network.sqrt_alphas_cumprod_prev, dtype=np.float64
            ),
        )

    def denoiser_state(self) -> StateDict:
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self._network.denoise_fn.state_dict().items()
        }

    def load_denoiser_state(self, state: StateDict) -> None:
        self._network.denoise_fn.load_state_dict(state, strict=True)
