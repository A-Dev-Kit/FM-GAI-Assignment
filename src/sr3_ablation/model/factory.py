# ADDED: new file, not part of the upstream SR3 codebase.
"""Assemble a ready-to-use backend: architecture, weights, then the run's schedule."""  # ADDED

from __future__ import annotations  # ADDED

from pathlib import Path  # ADDED

import torch  # ADDED

from sr3_ablation.config.schema import ModelConfig, ScheduleConfig  # ADDED
from sr3_ablation.model.checkpoints import read_denoiser_state  # ADDED
from sr3_ablation.model.sr3_backend import Sr3Backend  # ADDED
from sr3_ablation.runs import RunSpec  # ADDED


def build_backend(  # ADDED
    model: ModelConfig,  # ADDED
    schedule: ScheduleConfig,  # ADDED
    run: RunSpec,  # ADDED
    device: torch.device,  # ADDED
    weights: Path | None,  # ADDED
) -> Sr3Backend:  # ADDED
    """Create the SR3 backend, load ``weights`` (UNet only) and install the run's schedule.  # ADDED

    The schedule is installed after the weights so that nothing stored in a checkpoint can  # ADDED
    override it.  # ADDED
    """  # ADDED
    backend = Sr3Backend.create(model, schedule, device)  # ADDED
    if weights is not None:  # ADDED
        backend.load_denoiser_state(read_denoiser_state(weights))  # ADDED
    backend.set_schedule(schedule, run)  # ADDED
    return backend  # ADDED
