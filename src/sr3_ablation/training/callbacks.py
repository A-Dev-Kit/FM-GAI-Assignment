# ADDED: new file, not part of the upstream SR3 codebase.
"""Training callbacks producing the artefacts the brief asks for."""

from __future__ import annotations

import csv
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from sr3_ablation.evaluation.metrics import ImageMetric
from sr3_ablation.io.images import save_image, tensor_to_uint8
from sr3_ablation.io.jsonio import write_json
from sr3_ablation.io.naming import epoch_sample_name
from sr3_ablation.io.run_paths import RunPaths
from sr3_ablation.model.checkpoints import save_denoiser_state
from sr3_ablation.model.protocols import ScheduleHolder, WeightStore
from sr3_ablation.sampling.sampler import ReverseSampler
from sr3_ablation.schedule.noise_schedule import NoiseSchedule
from sr3_ablation.schedule.verification import verify_schedule
from sr3_ablation.training.trainer import EpochResult, TrainingCallback
from sr3_ablation.utils.log import get_logger
from sr3_ablation.utils.selection import evenly_spaced_indices

logger = get_logger("training.callbacks")

LOSS_COLUMNS = ("epoch", "train_loss")


class LossCsvLogger(TrainingCallback):
    """Writes ``loss.csv`` with exactly the columns ``epoch,train_loss``."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def on_train_start(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(LOSS_COLUMNS)

    def on_epoch_end(self, result: EpochResult) -> None:
        with self._path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([result.epoch, f"{result.train_loss:.6f}"])


class CheckpointSaver(TrainingCallback):
    """Saves UNet weights at the chosen epochs and a ``final_gen.pth`` at the end."""

    def __init__(
        self,
        store: WeightStore,
        paths: RunPaths,
        epochs: Collection[int],
        metadata: Mapping[str, Any],
    ) -> None:
        self._store = store
        self._paths = paths
        self._epochs = frozenset(epochs)
        self._metadata = dict(metadata)

    def on_epoch_end(self, result: EpochResult) -> None:
        if result.epoch in self._epochs:
            self._save(self._paths.checkpoint(result.epoch), result)

    def on_train_end(self, results: Sequence[EpochResult]) -> None:
        if results:
            self._save(self._paths.final_checkpoint(), results[-1])

    def _save(self, path: Path, result: EpochResult) -> None:
        metadata = {**self._metadata, "epoch": result.epoch, "train_loss": result.train_loss}
        save_denoiser_state(self._store.denoiser_state(), path, metadata)
        logger.info("Saved %s", path)


class ScheduleEvidenceWriter(TrainingCallback):
    """Verifies the model's schedule before and after training and writes ``schedule.json``."""

    def __init__(self, holder: ScheduleHolder, expected: NoiseSchedule, path: Path) -> None:
        self._holder = holder
        self._expected = expected
        self._path = path

    def on_train_start(self) -> None:
        self._write("before_training")

    def on_train_end(self, results: Sequence[EpochResult]) -> None:
        self._write("after_training")

    def _write(self, stage: str) -> None:
        evidence = verify_schedule(self._holder.schedule_buffers(), self._expected)
        write_json({**evidence, "stage": stage}, self._path)
        logger.info("Schedule verified (%s): %s", stage, self._expected.description)


@dataclass(frozen=True)
class ValidationOutputs:
    epoch: int
    image_ids: list[str]
    predictions: list[np.ndarray]
    targets: list[np.ndarray]


class ValidationSink(Protocol):
    def consume(self, outputs: ValidationOutputs) -> None: ...


class ValidationCallback(TrainingCallback):
    """At the chosen epochs, super-resolves a fixed validation set once and hands the
    results to every sink (snapshot images, validation metrics)."""

    def __init__(
        self,
        sampler: ReverseSampler,
        batches: Sequence[Mapping[str, Any]],
        epochs: Collection[int],
        sinks: Sequence[ValidationSink],
        seed: int,
    ) -> None:
        self._sampler = sampler
        self._batches = list(batches)
        self._epochs = frozenset(epochs)
        self._sinks = list(sinks)
        self._seed = seed

    def on_epoch_end(self, result: EpochResult) -> None:
        if result.epoch not in self._epochs:
            return
        outputs = ValidationOutputs(result.epoch, [], [], [])
        for index, batch in enumerate(self._batches):
            generated = self._sampler.sample(batch["SR"], seed=self._seed + index)
            outputs.image_ids.extend(batch["image_id"])
            outputs.predictions.extend(tensor_to_uint8(generated))
            outputs.targets.extend(tensor_to_uint8(torch.as_tensor(batch["HR"])))
        for sink in self._sinks:
            sink.consume(outputs)


class SnapshotWriter:
    """Saves ``count`` evenly spaced validation predictions as ``<run>_epochNN_sampleK.png``.

    The same images are chosen at every epoch, so snapshots are directly comparable.
    """

    def __init__(self, run_name: str, directory: Path, count: int) -> None:
        self._run_name = run_name
        self._directory = directory
        self._count = count

    def consume(self, outputs: ValidationOutputs) -> None:
        chosen = evenly_spaced_indices(len(outputs.predictions), self._count)
        for sample, index in enumerate(chosen, start=1):
            name = epoch_sample_name(self._run_name, outputs.epoch, sample)
            save_image(outputs.predictions[index], self._directory / name)


class ValidationMetricLogger:
    """Appends ``epoch,<metric>...`` rows (means over the validation images) to a CSV."""

    def __init__(self, path: Path, metrics: Sequence[ImageMetric]) -> None:
        self._path = path
        self._metrics = list(metrics)
        self.history: list[dict[str, float]] = []

    def consume(self, outputs: ValidationOutputs) -> None:
        row: dict[str, float] = {"epoch": outputs.epoch}
        for metric in self._metrics:
            scores = [
                metric(pred, target)
                for pred, target in zip(outputs.predictions, outputs.targets, strict=True)
            ]
            row[metric.name] = float(np.mean(scores))
        new_file = not self.history
        with self._path.open("w" if new_file else "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        self.history.append(row)
        logger.info("validation epoch %d: %s", outputs.epoch, row)


class RunRecordWriter(TrainingCallback):
    """Writes ``run.json`` (run identity, exponent, per-epoch results) when training ends."""

    def __init__(self, path: Path, record: Mapping[str, Any]) -> None:
        self._path = path
        self._record = dict(record)

    def on_train_end(self, results: Sequence[EpochResult]) -> None:
        epochs = [
            {
                "epoch": r.epoch,
                "train_loss": r.train_loss,
                "iterations": r.iterations,
                "seconds": r.seconds,
            }
            for r in results
        ]
        write_json({**self._record, "epochs": epochs}, self._path)
