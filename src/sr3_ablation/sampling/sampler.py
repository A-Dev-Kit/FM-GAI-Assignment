# ADDED: new file, not part of the upstream SR3 codebase.
"""Batched conditional reverse diffusion with observers for intermediate states.

Equivalent to upstream ``GaussianDiffusion.p_sample_loop`` (conditional branch) but works on a
whole batch, uses a local seed, and reports every state x_t to :class:`StepObserver` objects
instead of keeping a fixed ``T // 10`` stride.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Protocol

import torch
from tqdm import tqdm

from sr3_ablation.model.protocols import Denoiser, ModeSwitchable


class StepObserver(Protocol):
    def on_state(self, timestep: int, x_t: torch.Tensor) -> None:
        """Called with x_T before the first step and with x_t after each reverse step."""
        ...


@contextmanager
def evaluation_mode(model: ModeSwitchable) -> Iterator[None]:
    """Switch ``model`` to eval mode for the block and restore the previous mode afterwards."""
    was_training = model.is_training
    model.eval_mode()
    try:
        yield
    finally:
        if was_training:
            model.train_mode()


class ReverseSampler:
    def __init__(self, denoiser: Denoiser, seed: int, show_progress: bool = True) -> None:
        self._denoiser = denoiser
        self._seed = seed
        self._show_progress = show_progress

    def sample(
        self,
        condition: torch.Tensor,
        observers: Sequence[StepObserver] = (),
        seed: int | None = None,
    ) -> torch.Tensor:
        """Generate HR images for a batch of bicubic-upsampled conditioning images.

        Timesteps are 1-indexed: the loop starts from pure noise x_T and the state after the
        reverse step with upstream index ``i`` is x_i, so the final state is x_0.
        The global RNG state is restored afterwards, so sampling during training does not
        change the training random stream.
        """
        device = self._denoiser.device
        condition = condition.to(device)
        horizon = self._denoiser.num_timesteps
        fork_devices = [device] if device.type == "cuda" else []
        with (
            torch.random.fork_rng(devices=fork_devices),
            torch.no_grad(),
            evaluation_mode(self._denoiser),
        ):
            torch.manual_seed(self._seed if seed is None else seed)
            state = torch.randn(condition.shape, device=device)
            self._notify(observers, horizon, state)
            steps = tqdm(
                reversed(range(horizon)),
                total=horizon,
                desc="reverse diffusion",
                leave=False,
                disable=not self._show_progress,
            )
            for step_index in steps:
                state = self._denoiser.reverse_step(state, step_index, condition)
                self._notify(observers, step_index, state)
        return state

    @staticmethod
    def _notify(observers: Sequence[StepObserver], timestep: int, state: torch.Tensor) -> None:
        for observer in observers:
            observer.on_state(timestep, state)
