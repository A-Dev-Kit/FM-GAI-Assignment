# ADDED: new file, not part of the upstream SR3 codebase.
"""Capture and save x_t at chosen reverse-process timesteps."""  # ADDED

from __future__ import annotations  # ADDED

from collections.abc import Iterable, Mapping  # ADDED
from pathlib import Path  # ADDED

import torch  # ADDED

from sr3_ablation.io.images import save_image, tensor_to_uint8  # ADDED
from sr3_ablation.io.naming import trajectory_name  # ADDED


class TrajectoryRecorder:  # ADDED
    """:class:`~sr3_ablation.sampling.sampler.StepObserver` keeping x_t for selected t."""  # ADDED

    def __init__(self, timesteps: Iterable[int]) -> None:  # ADDED
        self._wanted = frozenset(timesteps)  # ADDED
        self._states: dict[int, torch.Tensor] = {}  # ADDED

    def on_state(self, timestep: int, x_t: torch.Tensor) -> None:  # ADDED
        if timestep in self._wanted:  # ADDED
            self._states[timestep] = x_t.detach().cpu().clone()  # ADDED

    @property  # ADDED
    def states(self) -> dict[int, torch.Tensor]:  # ADDED
        """Recorded states ordered from the noisiest (largest t) to the cleanest."""  # ADDED
        return dict(sorted(self._states.items(), reverse=True))  # ADDED

    @property  # ADDED
    def missing(self) -> list[int]:  # ADDED
        return sorted(self._wanted - self._states.keys(), reverse=True)  # ADDED


def save_trajectory(  # ADDED
    states: Mapping[int, torch.Tensor],  # ADDED
    run_name: str,  # ADDED
    output_dir: Path,  # ADDED
) -> list[Path]:  # ADDED
    """Save every sample of every recorded state, e.g. ``runA_t1600_sample1.png``.  # ADDED

    States are clipped to [-1, 1] for display, like upstream ``Metrics.tensor2img``.  # ADDED
    """  # ADDED
    written = []  # ADDED
    for timestep, batch in states.items():  # ADDED
        for index, image in enumerate(tensor_to_uint8(batch), start=1):  # ADDED
            path = output_dir / trajectory_name(run_name, timestep, index)  # ADDED
            save_image(image, path)  # ADDED
            written.append(path)  # ADDED
    return written  # ADDED
