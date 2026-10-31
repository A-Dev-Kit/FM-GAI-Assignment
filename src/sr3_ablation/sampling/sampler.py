# ADDED: new file, not part of the upstream SR3 codebase.
"""Batched conditional reverse diffusion with observers for intermediate states.  # ADDED

Equivalent to upstream ``GaussianDiffusion.p_sample_loop`` (conditional branch) but works on a  # ADDED
whole batch, uses a local seed, and reports every state x_t to :class:`StepObserver` objects  # ADDED
instead of keeping a fixed ``T // 10`` stride.  # ADDED
"""  # ADDED

from __future__ import annotations  # ADDED

from collections.abc import Iterator, Sequence  # ADDED
from contextlib import contextmanager  # ADDED
from typing import Protocol  # ADDED

import torch  # ADDED
from tqdm import tqdm  # ADDED

from sr3_ablation.model.protocols import Denoiser, ModeSwitchable  # ADDED
from sr3_ablation.sampling.starts import PURE_NOISE, StartState  # ADDED


class StepObserver(Protocol):  # ADDED
    """Receives every intermediate state of the reverse process."""  # ADDED

    def on_state(self, timestep: int, x_t: torch.Tensor) -> None:  # ADDED
        """Called with x_T before the first step and with x_t after each reverse step."""  # ADDED
        ...  # ADDED


@contextmanager  # ADDED
def evaluation_mode(model: ModeSwitchable) -> Iterator[None]:  # ADDED
    """Switch ``model`` to eval mode for the block and restore the previous mode afterwards."""  # ADDED
    was_training = model.is_training  # ADDED
    model.eval_mode()  # ADDED
    try:  # ADDED
        yield  # ADDED
    finally:  # ADDED
        if was_training:  # ADDED
            model.train_mode()  # ADDED


class ReverseSampler:  # ADDED
    """Batched conditional reverse diffusion from a :class:`StartState` down to x_0."""  # ADDED

    def __init__(  # ADDED
        self,  # ADDED
        denoiser: Denoiser,  # ADDED
        seed: int,  # ADDED
        show_progress: bool = True,  # ADDED
        start: StartState = PURE_NOISE,  # ADDED
    ) -> None:  # ADDED
        self._denoiser = denoiser  # ADDED
        self._seed = seed  # ADDED
        self._show_progress = show_progress  # ADDED
        self._start = start  # ADDED

    def sample(  # ADDED
        self,  # ADDED
        condition: torch.Tensor,  # ADDED
        observers: Sequence[StepObserver] = (),  # ADDED
        seed: int | None = None,  # ADDED
    ) -> torch.Tensor:  # ADDED
        """Generate HR images for a batch of bicubic-upsampled conditioning images.  # ADDED

        Timesteps are 1-indexed: the loop starts from pure noise x_T and the state after the  # ADDED
        reverse step with upstream index ``i`` is x_i, so the final state is x_0.  # ADDED
        The global RNG state is restored afterwards, so sampling during training does not  # ADDED
        change the training random stream.  # ADDED
        """  # ADDED
        device = self._denoiser.device  # ADDED
        condition = condition.to(device)  # ADDED
        horizon = self._denoiser.num_timesteps  # ADDED
        fork_devices = [device] if device.type == "cuda" else []  # ADDED
        with (  # ADDED
            torch.random.fork_rng(devices=fork_devices),  # ADDED
            torch.no_grad(),  # ADDED
            evaluation_mode(self._denoiser),  # ADDED
        ):  # ADDED
            torch.manual_seed(self._seed if seed is None else seed)  # ADDED
            state = self._start.initial(condition)  # ADDED
            self._notify(observers, horizon, state)  # ADDED
            steps = tqdm(  # ADDED
                reversed(range(horizon)),  # ADDED
                total=horizon,  # ADDED
                desc="reverse diffusion",  # ADDED
                leave=False,  # ADDED
                disable=not self._show_progress,  # ADDED
            )  # ADDED
            for step_index in steps:  # ADDED
                state = self._denoiser.reverse_step(state, step_index, condition)  # ADDED
                self._notify(observers, step_index, state)  # ADDED
        return state  # ADDED

    @staticmethod  # ADDED
    def _notify(observers: Sequence[StepObserver], timestep: int, state: torch.Tensor) -> None:  # ADDED
        for observer in observers:  # ADDED
            observer.on_state(timestep, state)  # ADDED
