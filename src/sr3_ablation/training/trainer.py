# ADDED: new file, not part of the upstream SR3 codebase.
"""Epoch-based fine-tuning loop with pluggable callbacks."""  # ADDED

from __future__ import annotations  # ADDED

import math  # ADDED
import time  # ADDED
from collections.abc import Iterable, Mapping, Sequence  # ADDED
from dataclasses import dataclass  # ADDED

import torch  # ADDED

from sr3_ablation.model.protocols import TrainableModel  # ADDED
from sr3_ablation.utils.log import get_logger  # ADDED

logger = get_logger("training")  # ADDED


@dataclass(frozen=True)  # ADDED
class EpochResult:  # ADDED
    """Mean training loss and timing of one epoch."""  # ADDED

    epoch: int  # ADDED
    train_loss: float  # ADDED
    iterations: int  # ADDED
    seconds: float  # ADDED


class TrainingCallback:  # ADDED
    """Hook points of :class:`FineTuner`; subclasses override only what they need."""  # ADDED

    def on_train_start(self) -> None:  # ADDED
        """Called once before the first epoch."""  # ADDED

    def on_epoch_end(self, result: EpochResult) -> None:  # ADDED
        """Called after every epoch with that epoch's mean training loss."""  # ADDED

    def on_train_end(self, results: Sequence[EpochResult]) -> None:  # ADDED
        """Called once after the last epoch."""  # ADDED


class NonFiniteLossError(RuntimeError):  # ADDED
    """Raised when the training loss becomes NaN or infinite."""  # ADDED


class FineTuner:  # ADDED
    """Minimise the model's training loss for a number of epochs.  # ADDED

    Args:  # ADDED
        model: Model exposing ``training_loss``.  # ADDED
        optimizer: Optimiser over ``model.parameters()``.  # ADDED
        batches: Re-iterable training batches (a ``DataLoader``).  # ADDED
        epochs: Number of passes over ``batches``.  # ADDED
        grad_accum_steps: Micro-batches per optimiser step.  # ADDED
        max_iterations: Stop after this many micro-batches in total (0 = no limit).  # ADDED
        log_every: Log the running loss every this many micro-batches.  # ADDED
        callbacks: Observers notified at the start, after each epoch and at the end.  # ADDED
    """  # ADDED

    def __init__(  # ADDED
        self,  # ADDED
        model: TrainableModel,  # ADDED
        optimizer: torch.optim.Optimizer,  # ADDED
        batches: Iterable[Mapping[str, torch.Tensor]],  # ADDED
        *,  # ADDED
        epochs: int,  # ADDED
        grad_accum_steps: int = 1,  # ADDED
        max_iterations: int = 0,  # ADDED
        log_every: int = 100,  # ADDED
        callbacks: Sequence[TrainingCallback] = (),  # ADDED
        start_epoch: int = 1,  # ADDED
    ) -> None:  # ADDED
        self._model = model  # ADDED
        self._optimizer = optimizer  # ADDED
        self._batches = batches  # ADDED
        self._epochs = epochs  # ADDED
        self._start_epoch = start_epoch  # ADDED
        self._grad_accum_steps = grad_accum_steps  # ADDED
        self._max_iterations = max_iterations  # ADDED
        self._log_every = log_every  # ADDED
        self._callbacks = list(callbacks)  # ADDED
        self._iteration = 0  # ADDED

    @property  # ADDED
    def iteration(self) -> int:  # ADDED
        return self._iteration  # ADDED

    def fit(self) -> list[EpochResult]:  # ADDED
        for callback in self._callbacks:  # ADDED
            callback.on_train_start()  # ADDED
        results: list[EpochResult] = []  # ADDED
        for epoch in range(self._start_epoch, self._epochs + 1):  # ADDED
            result = self._run_epoch(epoch)  # ADDED
            results.append(result)  # ADDED
            logger.info(  # ADDED
                "epoch %d: train_loss=%.6f (%d iterations, %.1f s)",  # ADDED
                epoch,  # ADDED
                result.train_loss,  # ADDED
                result.iterations,  # ADDED
                result.seconds,  # ADDED
            )  # ADDED
            for callback in self._callbacks:  # ADDED
                callback.on_epoch_end(result)  # ADDED
            if self._limit_reached():  # ADDED
                logger.info("Reached max_iterations=%d; stopping.", self._max_iterations)  # ADDED
                break  # ADDED
        for callback in self._callbacks:  # ADDED
            callback.on_train_end(results)  # ADDED
        return results  # ADDED

    def _run_epoch(self, epoch: int) -> EpochResult:  # ADDED
        self._model.train_mode()  # ADDED
        self._optimizer.zero_grad(set_to_none=True)  # ADDED
        start = time.perf_counter()  # ADDED
        total_loss, steps = 0.0, 0  # ADDED
        for steps, batch in enumerate(self._batches, start=1):  # ADDED
            loss = self._model.training_loss(batch)  # ADDED
            value = float(loss.detach())  # ADDED
            if not math.isfinite(value):  # ADDED
                raise NonFiniteLossError(f"Loss became {value} at epoch {epoch}, step {steps}.")  # ADDED
            (loss / self._grad_accum_steps).backward()  # ADDED
            if steps % self._grad_accum_steps == 0:  # ADDED
                self._step()  # ADDED
            total_loss += value  # ADDED
            self._iteration += 1  # ADDED
            if self._iteration % self._log_every == 0:  # ADDED
                logger.info("iter %d: loss=%.6f", self._iteration, total_loss / steps)  # ADDED
            if self._limit_reached():  # ADDED
                break  # ADDED
        if steps == 0:  # ADDED
            raise RuntimeError("The training loader produced no batches.")  # ADDED
        if steps % self._grad_accum_steps:  # ADDED
            self._step()  # ADDED
        return EpochResult(epoch, total_loss / steps, steps, time.perf_counter() - start)  # ADDED

    def _step(self) -> None:  # ADDED
        self._optimizer.step()  # ADDED
        self._optimizer.zero_grad(set_to_none=True)  # ADDED

    def _limit_reached(self) -> bool:  # ADDED
        return 0 < self._max_iterations <= self._iteration  # ADDED
