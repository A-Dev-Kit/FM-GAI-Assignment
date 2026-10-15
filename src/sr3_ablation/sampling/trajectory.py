# ADDED: new file, not part of the upstream SR3 codebase.
"""Capture and save x_t at chosen reverse-process timesteps."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import torch

from sr3_ablation.io.images import save_image, tensor_to_uint8
from sr3_ablation.io.naming import trajectory_name


class TrajectoryRecorder:
    """:class:`~sr3_ablation.sampling.sampler.StepObserver` keeping x_t for selected t."""

    def __init__(self, timesteps: Iterable[int]) -> None:
        self._wanted = frozenset(timesteps)
        self._states: dict[int, torch.Tensor] = {}

    def on_state(self, timestep: int, x_t: torch.Tensor) -> None:
        if timestep in self._wanted:
            self._states[timestep] = x_t.detach().cpu().clone()

    @property
    def states(self) -> dict[int, torch.Tensor]:
        """Recorded states ordered from the noisiest (largest t) to the cleanest."""
        return dict(sorted(self._states.items(), reverse=True))

    @property
    def missing(self) -> list[int]:
        return sorted(self._wanted - self._states.keys(), reverse=True)


def save_trajectory(
    states: Mapping[int, torch.Tensor], run_name: str, output_dir: Path
) -> list[Path]:
    """Save every sample of every recorded state, e.g. ``runA_t1600_sample1.png``.

    States are clipped to [-1, 1] for display, like upstream ``Metrics.tensor2img``.
    """
    written = []
    for timestep, batch in states.items():
        for index, image in enumerate(tensor_to_uint8(batch), start=1):
            path = output_dir / trajectory_name(run_name, timestep, index)
            save_image(image, path)
            written.append(path)
    return written
