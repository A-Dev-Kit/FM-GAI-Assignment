# ADDED: new file, not part of the upstream SR3 codebase.
"""Training callbacks producing the artefacts the brief asks for."""  # ADDED

from __future__ import annotations  # ADDED

import csv  # ADDED
from collections.abc import Collection, Mapping, Sequence  # ADDED
from dataclasses import dataclass  # ADDED
from pathlib import Path  # ADDED
from typing import Any, Protocol  # ADDED

import numpy as np  # ADDED
import torch  # ADDED

from sr3_ablation.evaluation.metrics import ImageMetric  # ADDED
from sr3_ablation.io.images import save_image, tensor_to_uint8  # ADDED
from sr3_ablation.io.jsonio import write_json  # ADDED
from sr3_ablation.io.naming import epoch_sample_name  # ADDED
from sr3_ablation.io.run_paths import RunPaths  # ADDED
from sr3_ablation.model.checkpoints import save_denoiser_state  # ADDED
from sr3_ablation.model.protocols import ScheduleHolder, WeightStore  # ADDED
from sr3_ablation.sampling.sampler import ReverseSampler  # ADDED
from sr3_ablation.schedule.noise_schedule import NoiseSchedule  # ADDED
from sr3_ablation.schedule.verification import verify_schedule  # ADDED
from sr3_ablation.training.trainer import EpochResult, TrainingCallback  # ADDED
from sr3_ablation.utils.heartbeat import touch  # ADDED
from sr3_ablation.utils.log import get_logger  # ADDED
from sr3_ablation.utils.selection import evenly_spaced_indices  # ADDED

logger = get_logger("training.callbacks")  # ADDED

LOSS_COLUMNS = ("epoch", "train_loss")  # ADDED


class LossCsvLogger(TrainingCallback):  # ADDED
    """Writes ``loss.csv`` with exactly the columns ``epoch,train_loss``."""  # ADDED

    def __init__(self, path: Path) -> None:  # ADDED
        self._path = path  # ADDED

    def on_train_start(self) -> None:  # ADDED
        self._path.parent.mkdir(parents=True, exist_ok=True)  # ADDED
        with self._path.open("w", newline="", encoding="utf-8") as handle:  # ADDED
            csv.writer(handle).writerow(LOSS_COLUMNS)  # ADDED

    def on_epoch_end(self, result: EpochResult) -> None:  # ADDED
        with self._path.open("a", newline="", encoding="utf-8") as handle:  # ADDED
            csv.writer(handle).writerow([result.epoch, f"{result.train_loss:.6f}"])  # ADDED


class HeartbeatWriter(TrainingCallback):  # ADDED
    """Touch a file each epoch so an external supervisor can detect stalls."""  # ADDED

    def __init__(self, path: Path) -> None:  # ADDED
        self._path = path  # ADDED

    def on_train_start(self) -> None:  # ADDED
        touch(self._path)  # ADDED

    def on_epoch_end(self, result: EpochResult) -> None:  # ADDED
        touch(self._path)  # ADDED


class CheckpointSaver(TrainingCallback):  # ADDED
    """Saves UNet weights at the chosen epochs and a ``final_gen.pth`` at the end."""  # ADDED

    def __init__(  # ADDED
        self,  # ADDED
        store: WeightStore,  # ADDED
        paths: RunPaths,  # ADDED
        epochs: Collection[int],  # ADDED
        metadata: Mapping[str, Any],  # ADDED
        optimizer: torch.optim.Optimizer | None = None,  # ADDED
    ) -> None:  # ADDED
        self._store = store  # ADDED
        self._paths = paths  # ADDED
        self._epochs = frozenset(epochs)  # ADDED
        self._metadata = dict(metadata)  # ADDED
        self._optimizer = optimizer  # ADDED

    def on_epoch_end(self, result: EpochResult) -> None:  # ADDED
        if result.epoch in self._epochs:  # ADDED
            self._save(self._paths.checkpoint(result.epoch), result)  # ADDED
        self._save_resume(result)  # ADDED

    def on_train_end(self, results: Sequence[EpochResult]) -> None:  # ADDED
        if results:  # ADDED
            self._save(self._paths.final_checkpoint(), results[-1])  # ADDED

    def _save(self, path: Path, result: EpochResult) -> None:  # ADDED
        metadata = {**self._metadata, "epoch": result.epoch, "train_loss": result.train_loss}  # ADDED
        save_denoiser_state(self._store.denoiser_state(), path, metadata)  # ADDED
        logger.info("Saved %s", path)  # ADDED

    def _save_resume(self, result: EpochResult) -> None:  # ADDED
        if self._optimizer is None:  # ADDED
            return  # ADDED
        path = self._paths.checkpoints / "training_last.pth"  # ADDED
        path.parent.mkdir(parents=True, exist_ok=True)  # ADDED
        torch.save(  # ADDED
            {  # ADDED
                "denoiser": self._store.denoiser_state(),  # ADDED
                "optimizer": self._optimizer.state_dict(),  # ADDED
                "epoch": result.epoch,  # ADDED
            },  # ADDED
            path,  # ADDED
        )  # ADDED


class ScheduleEvidenceWriter(TrainingCallback):  # ADDED
    """Verifies the model's schedule before and after training and writes ``schedule.json``."""  # ADDED

    def __init__(self, holder: ScheduleHolder, expected: NoiseSchedule, path: Path) -> None:  # ADDED
        self._holder = holder  # ADDED
        self._expected = expected  # ADDED
        self._path = path  # ADDED

    def on_train_start(self) -> None:  # ADDED
        self._write("before_training")  # ADDED

    def on_train_end(self, results: Sequence[EpochResult]) -> None:  # ADDED
        self._write("after_training")  # ADDED

    def _write(self, stage: str) -> None:  # ADDED
        evidence = verify_schedule(self._holder.schedule_buffers(), self._expected)  # ADDED
        write_json({**evidence, "stage": stage}, self._path)  # ADDED
        logger.info("Schedule verified (%s): %s", stage, self._expected.description)  # ADDED


@dataclass(frozen=True)  # ADDED
class ValidationOutputs:  # ADDED
    """Super-resolved validation images of one checkpoint epoch and their targets."""  # ADDED

    epoch: int  # ADDED
    image_ids: list[str]  # ADDED
    predictions: list[np.ndarray]  # ADDED
    targets: list[np.ndarray]  # ADDED


class ValidationSink(Protocol):  # ADDED
    """Consumes the validation outputs of a checkpoint epoch."""  # ADDED

    def consume(self, outputs: ValidationOutputs) -> None: ...  # ADDED


class ValidationCallback(TrainingCallback):  # ADDED
    """At the chosen epochs, super-resolves a fixed validation set once and hands the  # ADDED
    results to every sink (snapshot images, validation metrics)."""  # ADDED

    def __init__(  # ADDED
        self,  # ADDED
        sampler: ReverseSampler,  # ADDED
        batches: Sequence[Mapping[str, Any]],  # ADDED
        epochs: Collection[int],  # ADDED
        sinks: Sequence[ValidationSink],  # ADDED
        seed: int,  # ADDED
    ) -> None:  # ADDED
        self._sampler = sampler  # ADDED
        self._batches = list(batches)  # ADDED
        self._epochs = frozenset(epochs)  # ADDED
        self._sinks = list(sinks)  # ADDED
        self._seed = seed  # ADDED

    def on_epoch_end(self, result: EpochResult) -> None:  # ADDED
        if result.epoch not in self._epochs:  # ADDED
            return  # ADDED
        outputs = ValidationOutputs(result.epoch, [], [], [])  # ADDED
        for index, batch in enumerate(self._batches):  # ADDED
            generated = self._sampler.sample(batch["SR"], seed=self._seed + index)  # ADDED
            outputs.image_ids.extend(batch["image_id"])  # ADDED
            outputs.predictions.extend(tensor_to_uint8(generated))  # ADDED
            outputs.targets.extend(tensor_to_uint8(torch.as_tensor(batch["HR"])))  # ADDED
        for sink in self._sinks:  # ADDED
            sink.consume(outputs)  # ADDED


class SnapshotWriter:  # ADDED
    """Saves ``count`` evenly spaced validation predictions as ``<run>_epochNN_sampleK.png``.  # ADDED

    The same images are chosen at every epoch, so snapshots are directly comparable.  # ADDED
    """  # ADDED

    def __init__(self, run_name: str, directory: Path, count: int) -> None:  # ADDED
        self._run_name = run_name  # ADDED
        self._directory = directory  # ADDED
        self._count = count  # ADDED

    def consume(self, outputs: ValidationOutputs) -> None:  # ADDED
        chosen = evenly_spaced_indices(len(outputs.predictions), self._count)  # ADDED
        for sample, index in enumerate(chosen, start=1):  # ADDED
            name = epoch_sample_name(self._run_name, outputs.epoch, sample)  # ADDED
            save_image(outputs.predictions[index], self._directory / name)  # ADDED


class ValidationMetricLogger:  # ADDED
    """Appends ``epoch,<metric>...`` rows (means over the validation images) to a CSV."""  # ADDED

    def __init__(self, path: Path, metrics: Sequence[ImageMetric]) -> None:  # ADDED
        self._path = path  # ADDED
        self._metrics = list(metrics)  # ADDED
        self.history: list[dict[str, float]] = []  # ADDED

    def consume(self, outputs: ValidationOutputs) -> None:  # ADDED
        row: dict[str, float] = {"epoch": outputs.epoch}  # ADDED
        for metric in self._metrics:  # ADDED
            scores = [  # ADDED
                metric(pred, target)  # ADDED
                for pred, target in zip(outputs.predictions, outputs.targets, strict=True)  # ADDED
            ]  # ADDED
            row[metric.name] = float(np.mean(scores))  # ADDED
        new_file = not self.history  # ADDED
        with self._path.open("w" if new_file else "a", newline="", encoding="utf-8") as handle:  # ADDED
            writer = csv.DictWriter(handle, fieldnames=list(row))  # ADDED
            if new_file:  # ADDED
                writer.writeheader()  # ADDED
            writer.writerow(row)  # ADDED
        self.history.append(row)  # ADDED
        logger.info("validation epoch %d: %s", outputs.epoch, row)  # ADDED


class RunRecordWriter(TrainingCallback):  # ADDED
    """Writes ``run.json`` (run identity, exponent, per-epoch results) when training ends."""  # ADDED

    def __init__(self, path: Path, record: Mapping[str, Any]) -> None:  # ADDED
        self._path = path  # ADDED
        self._record = dict(record)  # ADDED

    def on_train_end(self, results: Sequence[EpochResult]) -> None:  # ADDED
        epochs = [  # ADDED
            {  # ADDED
                "epoch": r.epoch,  # ADDED
                "train_loss": r.train_loss,  # ADDED
                "iterations": r.iterations,  # ADDED
                "seconds": r.seconds,  # ADDED
            }  # ADDED
            for r in results  # ADDED
        ]  # ADDED
        write_json({**self._record, "epochs": epochs}, self._path)  # ADDED
