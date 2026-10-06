# ADDED: new file, not part of the upstream SR3 codebase.
"""File names required by the brief, e.g. ``runB_epoch04_sample1.png``."""

from __future__ import annotations


def epoch_sample_name(run_name: str, epoch: int, sample_index: int) -> str:
    """Name of a training-snapshot image; ``sample_index`` starts at 1."""
    return f"{run_name}_epoch{epoch:02d}_sample{sample_index}.png"


def trajectory_name(run_name: str, timestep: int, sample_index: int) -> str:
    """Name of a reverse-process state x_t; ``sample_index`` starts at 1."""
    return f"{run_name}_t{timestep:04d}_sample{sample_index}.png"


def super_resolved_name(run_name: str, image_id: str) -> str:
    """Name of a super-resolved test image."""
    return f"{run_name}_{image_id}.png"


def checkpoint_name(epoch: int) -> str:
    return f"epoch{epoch:02d}_gen.pth"
