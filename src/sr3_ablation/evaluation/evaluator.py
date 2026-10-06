# ADDED: new file, not part of the upstream SR3 codebase.
"""Test-set evaluation: super-resolve, save outputs, score against the HR ground truth."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sr3_ablation.evaluation.metrics import ImageMetric
from sr3_ablation.io.images import save_image, tensor_to_uint8
from sr3_ablation.io.jsonio import write_json
from sr3_ablation.io.naming import super_resolved_name
from sr3_ablation.sampling.sampler import ReverseSampler
from sr3_ablation.utils.log import get_logger

logger = get_logger("evaluation")

Batch = Mapping[str, Any]


@dataclass(frozen=True)
class EvaluationSummary:
    run_name: str
    label: str
    images: int
    mean: dict[str, float]
    std: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_images(
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    image_ids: Sequence[str],
    metrics: Sequence[ImageMetric],
) -> list[dict[str, Any]]:
    """One row per image: ``image_id`` plus one column per metric."""
    return [
        {"image_id": image_id, **{metric.name: metric(pred, target) for metric in metrics}}
        for image_id, pred, target in zip(image_ids, predictions, targets, strict=True)
    ]


def summarise(
    rows: Sequence[Mapping[str, Any]], metrics: Sequence[ImageMetric], run_name: str, label: str
) -> EvaluationSummary:
    mean, std = {}, {}
    for metric in metrics:
        values = np.array([row[metric.name] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        mean[metric.name] = float(finite.mean()) if finite.size else math.nan
        std[metric.name] = float(finite.std()) if finite.size else math.nan
    return EvaluationSummary(run_name, label, len(rows), mean, std)


def write_scores(rows: Sequence[Mapping[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class SuperResolutionEvaluator:
    """Runs a sampler over the test set; batch ``k`` uses seed ``seed + k`` so that every run
    starts from the same initial noise for the same image."""

    def __init__(self, sampler: ReverseSampler, metrics: Sequence[ImageMetric], seed: int) -> None:
        self._sampler = sampler
        self._metrics = list(metrics)
        self._seed = seed

    def evaluate(
        self,
        batches: Iterable[Batch],
        run_name: str,
        label: str,
        image_dir: Path,
        csv_path: Path,
        summary_path: Path,
    ) -> EvaluationSummary:
        rows: list[dict[str, Any]] = []
        for index, batch in enumerate(batches):
            generated = self._sampler.sample(batch["SR"], seed=self._seed + index)
            predictions = tensor_to_uint8(generated)
            targets = tensor_to_uint8(torch.as_tensor(batch["HR"]))
            for image_id, image in zip(batch["image_id"], predictions, strict=True):
                save_image(image, image_dir / super_resolved_name(run_name, image_id))
            rows.extend(score_images(predictions, targets, batch["image_id"], self._metrics))
            logger.info("%s: %d images evaluated", label, len(rows))
        return _finish(rows, self._metrics, run_name, label, csv_path, summary_path)


def evaluate_bicubic(
    batches: Iterable[Batch],
    metrics: Sequence[ImageMetric],
    csv_path: Path,
    summary_path: Path,
) -> EvaluationSummary:
    """Reference row: the bicubic-upsampled conditioning image scored against HR."""
    rows: list[dict[str, Any]] = []
    for batch in batches:
        predictions = tensor_to_uint8(torch.as_tensor(batch["SR"]))
        targets = tensor_to_uint8(torch.as_tensor(batch["HR"]))
        rows.extend(score_images(predictions, targets, batch["image_id"], metrics))
    return _finish(rows, metrics, "bicubic", "Bicubic", csv_path, summary_path)


def _finish(
    rows: list[dict[str, Any]],
    metrics: Sequence[ImageMetric],
    run_name: str,
    label: str,
    csv_path: Path,
    summary_path: Path,
) -> EvaluationSummary:
    if not rows:
        raise RuntimeError("No test images were evaluated.")
    write_scores(rows, csv_path)
    summary = summarise(rows, metrics, run_name, label)
    write_json(summary.to_dict(), summary_path)
    logger.info("%s summary: %s", label, summary.mean)
    return summary
