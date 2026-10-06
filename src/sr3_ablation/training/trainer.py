# ADDED: new file, not part of the upstream SR3 codebase.
"""Epoch-based fine-tuning loop with pluggable callbacks."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import torch

from sr3_ablation.model.protocols import TrainableModel
from sr3_ablation.utils.log import get_logger

logger = get_logger("training")


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    train_loss: float
    iterations: int
    seconds: float


class TrainingCallback:
    """Hook points of :class:`FineTuner`; subclasses override only what they need."""

    def on_train_start(self) -> None:
        """Called once before the first epoch."""

    def on_epoch_end(self, result: EpochResult) -> None:
        """Called after every epoch with that epoch's mean training loss."""

    def on_train_end(self, results: Sequence[EpochResult]) -> None:
        """Called once after the last epoch."""


class NonFiniteLossError(RuntimeError):
    pass


class FineTuner:
    """Minimise the model's training loss for a number of epochs.

    Args:
        model: Model exposing ``training_loss``.
        optimizer: Optimiser over ``model.parameters()``.
        batches: Re-iterable training batches (a ``DataLoader``).
        epochs: Number of passes over ``batches``.
        grad_accum_steps: Micro-batches per optimiser step.
        max_iterations: Stop after this many micro-batches in total (0 = no limit).
        log_every: Log the running loss every this many micro-batches.
        callbacks: Observers notified at the start, after each epoch and at the end.
    """

    def __init__(
        self,
        model: TrainableModel,
        optimizer: torch.optim.Optimizer,
        batches: Iterable[Mapping[str, torch.Tensor]],
        *,
        epochs: int,
        grad_accum_steps: int = 1,
        max_iterations: int = 0,
        log_every: int = 100,
        callbacks: Sequence[TrainingCallback] = (),
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._batches = batches
        self._epochs = epochs
        self._grad_accum_steps = grad_accum_steps
        self._max_iterations = max_iterations
        self._log_every = log_every
        self._callbacks = list(callbacks)
        self._iteration = 0

    @property
    def iteration(self) -> int:
        return self._iteration

    def fit(self) -> list[EpochResult]:
        for callback in self._callbacks:
            callback.on_train_start()
        results: list[EpochResult] = []
        for epoch in range(1, self._epochs + 1):
            result = self._run_epoch(epoch)
            results.append(result)
            logger.info(
                "epoch %d: train_loss=%.6f (%d iterations, %.1f s)",
                epoch,
                result.train_loss,
                result.iterations,
                result.seconds,
            )
            for callback in self._callbacks:
                callback.on_epoch_end(result)
            if self._limit_reached():
                logger.info("Reached max_iterations=%d; stopping.", self._max_iterations)
                break
        for callback in self._callbacks:
            callback.on_train_end(results)
        return results

    def _run_epoch(self, epoch: int) -> EpochResult:
        self._model.train_mode()
        self._optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        total_loss, steps = 0.0, 0
        for steps, batch in enumerate(self._batches, start=1):
            loss = self._model.training_loss(batch)
            value = float(loss.detach())
            if not math.isfinite(value):
                raise NonFiniteLossError(f"Loss became {value} at epoch {epoch}, step {steps}.")
            (loss / self._grad_accum_steps).backward()
            if steps % self._grad_accum_steps == 0:
                self._step()
            total_loss += value
            self._iteration += 1
            if self._iteration % self._log_every == 0:
                logger.info("iter %d: loss=%.6f", self._iteration, total_loss / steps)
            if self._limit_reached():
                break
        if steps == 0:
            raise RuntimeError("The training loader produced no batches.")
        if steps % self._grad_accum_steps:
            self._step()
        return EpochResult(epoch, total_loss / steps, steps, time.perf_counter() - start)

    def _step(self) -> None:
        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)

    def _limit_reached(self) -> bool:
        return 0 < self._max_iterations <= self._iteration
