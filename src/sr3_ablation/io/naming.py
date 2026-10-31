# ADDED: new file, not part of the upstream SR3 codebase.
"""File names required by the brief, e.g. ``runB_epoch04_sample1.png``."""  # ADDED

from __future__ import annotations  # ADDED


def epoch_sample_name(run_name: str, epoch: int, sample_index: int) -> str:  # ADDED
    """Name of a training-snapshot image; ``sample_index`` starts at 1."""  # ADDED
    return f"{run_name}_epoch{epoch:02d}_sample{sample_index}.png"  # ADDED


def trajectory_name(run_name: str, timestep: int, sample_index: int) -> str:  # ADDED
    """Name of a reverse-process state x_t; ``sample_index`` starts at 1."""  # ADDED
    return f"{run_name}_t{timestep:04d}_sample{sample_index}.png"  # ADDED


def diagnostic_prefix(run_name: str, start: str) -> str:  # ADDED
    """Prefix of a diagnostic state's file name, e.g. ``runB_forward_noised``."""  # ADDED
    return f"{run_name}_{start}"  # ADDED


def diagnostic_name(run_name: str, start: str, timestep: int, sample_index: int) -> str:  # ADDED
    """Name of x_t from a diagnostic run, e.g. ``runB_forward_noised_t2000_sample1.png``."""  # ADDED
    return trajectory_name(diagnostic_prefix(run_name, start), timestep, sample_index)  # ADDED


def super_resolved_name(run_name: str, image_id: str) -> str:  # ADDED
    """Name of a super-resolved test image."""  # ADDED
    return f"{run_name}_{image_id}.png"  # ADDED


def checkpoint_name(epoch: int) -> str:  # ADDED
    """File name of the checkpoint saved at ``epoch``."""  # ADDED
    return f"epoch{epoch:02d}_gen.pth"  # ADDED
