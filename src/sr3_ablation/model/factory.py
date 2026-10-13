# ADDED: new file, not part of the upstream SR3 codebase.
"""Assemble a ready-to-use backend: architecture, weights, then the run's schedule."""

from __future__ import annotations

from pathlib import Path

import torch

from sr3_ablation.config.schema import ModelConfig, ScheduleConfig
from sr3_ablation.model.checkpoints import read_denoiser_state
from sr3_ablation.model.sr3_backend import Sr3Backend


def build_backend(
    model: ModelConfig,
    schedule: ScheduleConfig,
    exponent: float,
    device: torch.device,
    weights: Path | None,
) -> Sr3Backend:
    """Create the SR3 backend, load ``weights`` (UNet only) and install the run's schedule.

    The schedule is installed after the weights so that nothing stored in a checkpoint can
    override it.
    """
    backend = Sr3Backend.create(model, schedule, device)
    if weights is not None:
        backend.load_denoiser_state(read_denoiser_state(weights))
    backend.set_schedule(schedule, exponent)
    return backend
