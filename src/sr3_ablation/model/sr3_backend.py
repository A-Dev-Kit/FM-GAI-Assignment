# ADDED: new file, not part of the upstream SR3 codebase.
"""Adapter around the vendored upstream SR3 ``GaussianDiffusion``.  # ADDED

This is the only module that imports from ``third_party/sr3``.  # ADDED
"""  # ADDED

from __future__ import annotations  # ADDED

import sys  # ADDED
from collections.abc import Iterator, Mapping  # ADDED
from types import ModuleType  # ADDED
from typing import Any  # ADDED

import numpy as np  # ADDED
import torch  # ADDED
from torch import nn  # ADDED

from sr3_ablation.config.schema import ModelConfig, ScheduleConfig  # ADDED
from sr3_ablation.model.protocols import StateDict  # ADDED
from sr3_ablation.paths import UPSTREAM_SR3_DIR  # ADDED
from sr3_ablation.runs import RunSpec  # ADDED
from sr3_ablation.schedule.verification import ScheduleBuffers  # ADDED


def upstream_networks() -> ModuleType:  # ADDED
    """Import upstream ``model.networks`` (the vendored folder is put on ``sys.path``)."""  # ADDED
    upstream = str(UPSTREAM_SR3_DIR)  # ADDED
    if upstream not in sys.path:  # ADDED
        sys.path.insert(0, upstream)  # ADDED
    from model import networks  # type: ignore[import-not-found]  # ADDED

    return networks  # ADDED


def upstream_options(config: ModelConfig, schedule: ScheduleConfig) -> dict[str, Any]:  # ADDED
    """The subset of SR3's JSON options that ``networks.define_G`` reads."""  # ADDED
    unet = config.unet  # ADDED
    schedule_options = schedule.to_upstream()  # ADDED
    return {  # ADDED
        # "val" skips define_G's orthogonal re-initialisation; weights come from a checkpoint.  # ADDED
        "phase": "val",  # ADDED
        "gpu_ids": None,  # ADDED
        "distributed": False,  # ADDED
        "model": {  # ADDED
            "which_model_G": "sr3",  # ADDED
            "unet": {  # ADDED
                "in_channel": unet.in_channel,  # ADDED
                "out_channel": unet.out_channel,  # ADDED
                "inner_channel": unet.inner_channel,  # ADDED
                "norm_groups": unet.norm_groups,  # ADDED
                "channel_multiplier": list(unet.channel_multiplier),  # ADDED
                "attn_res": list(unet.attn_res),  # ADDED
                "res_blocks": unet.res_blocks,  # ADDED
                "dropout": unet.dropout,  # ADDED
            },  # ADDED
            "beta_schedule": {"train": schedule_options, "val": schedule_options},  # ADDED
            "diffusion": {  # ADDED
                "image_size": config.image_size,  # ADDED
                "channels": config.channels,  # ADDED
                "conditional": True,  # ADDED
            },  # ADDED
        },  # ADDED
    }  # ADDED


class Sr3Backend:  # ADDED
    """:class:`~sr3_ablation.model.protocols.DiffusionBackend` backed by upstream SR3."""  # ADDED

    def __init__(self, network: nn.Module, device: torch.device) -> None:  # ADDED
        self._network = network.to(device)  # ADDED
        self._network.set_loss(device)  # ADDED
        self._device = device  # ADDED

    @classmethod  # ADDED
    def create(  # ADDED
        cls,  # ADDED
        config: ModelConfig,  # ADDED
        schedule: ScheduleConfig,  # ADDED
        device: torch.device,  # ADDED
    ) -> Sr3Backend:  # ADDED
        network = upstream_networks().define_G(upstream_options(config, schedule))  # ADDED
        return cls(network, device)  # ADDED

    @property  # ADDED
    def device(self) -> torch.device:  # ADDED
        return self._device  # ADDED

    @property  # ADDED
    def network(self) -> nn.Module:  # ADDED
        return self._network  # ADDED

    @property  # ADDED
    def is_training(self) -> bool:  # ADDED
        return self._network.training  # ADDED

    def train_mode(self) -> None:  # ADDED
        self._network.train()  # ADDED

    def eval_mode(self) -> None:  # ADDED
        self._network.eval()  # ADDED

    def parameters(self) -> Iterator[nn.Parameter]:  # ADDED
        return self._network.parameters()  # ADDED

    def training_loss(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:  # ADDED
        hr = batch["HR"].to(self._device, non_blocking=True)  # ADDED
        sr = batch["SR"].to(self._device, non_blocking=True)  # ADDED
        loss_sum = self._network({"HR": hr, "SR": sr})  # ADDED
        # Same normalisation as upstream DDPM.optimize_parameters: summed L1 / (b * c * h * w).  # ADDED
        return loss_sum / hr.numel()  # ADDED

    @property  # ADDED
    def num_timesteps(self) -> int:  # ADDED
        if not hasattr(self._network, "num_timesteps"):  # ADDED
            raise RuntimeError("No noise schedule set; call set_schedule() first.")  # ADDED
        return int(self._network.num_timesteps)  # ADDED

    def reverse_step(  # ADDED
        self,  # ADDED
        x_t: torch.Tensor,  # ADDED
        step_index: int,  # ADDED
        condition: torch.Tensor,  # ADDED
    ) -> torch.Tensor:  # ADDED
        return self._network.p_sample(x_t, step_index, condition_x=condition)  # ADDED

    def set_schedule(self, config: ScheduleConfig, run: RunSpec) -> None:  # ADDED
        if run.power_target == "alpha_bar":  # ADDED
            options = config.to_upstream(exponent=1.0, alpha_bar_exponent=run.exponent)  # ADDED
        else:  # ADDED
            options = config.to_upstream(exponent=run.exponent, alpha_bar_exponent=1.0)  # ADDED
        self._network.set_new_noise_schedule(options, self._device)  # ADDED

    def schedule_buffers(self) -> ScheduleBuffers:  # ADDED
        return ScheduleBuffers(  # ADDED
            betas=self._network.betas.detach().cpu().double().numpy(),  # ADDED
            alphas_cumprod=self._network.alphas_cumprod.detach().cpu().double().numpy(),  # ADDED
            sqrt_alphas_cumprod_prev=np.asarray(  # ADDED
                self._network.sqrt_alphas_cumprod_prev,  # ADDED
                dtype=np.float64,  # ADDED
            ),  # ADDED
        )  # ADDED

    def denoiser_state(self) -> StateDict:  # ADDED
        return {  # ADDED
            name: tensor.detach().cpu().clone()  # ADDED
            for name, tensor in self._network.denoise_fn.state_dict().items()  # ADDED
        }  # ADDED

    def load_denoiser_state(self, state: StateDict) -> None:  # ADDED
        self._network.denoise_fn.load_state_dict(state, strict=True)  # ADDED
