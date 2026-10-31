# ADDED: new file, not part of the upstream SR3 codebase.
"""Test-set evaluation: super-resolve, save outputs, score against the HR ground truth."""  # ADDED

from __future__ import annotations  # ADDED

import csv  # ADDED
import math  # ADDED
from collections.abc import Iterable, Mapping, Sequence  # ADDED
from dataclasses import asdict, dataclass  # ADDED
from pathlib import Path  # ADDED
from typing import Any  # ADDED

import numpy as np  # ADDED
import torch  # ADDED
from PIL import Image  # ADDED

from sr3_ablation.evaluation.metrics import ImageMetric  # ADDED
from sr3_ablation.io.images import save_image, tensor_to_uint8  # ADDED
from sr3_ablation.io.jsonio import write_json  # ADDED
from sr3_ablation.io.naming import super_resolved_name  # ADDED
from sr3_ablation.runs import BICUBIC_LABEL, BICUBIC_RUN  # ADDED
from sr3_ablation.sampling.sampler import ReverseSampler  # ADDED
from sr3_ablation.utils.log import get_logger  # ADDED

logger = get_logger("evaluation")  # ADDED

Batch = Mapping[str, Any]  # ADDED


@dataclass(frozen=True)  # ADDED
class EvaluationSummary:  # ADDED
    """Mean and standard deviation of each metric over a run's test images."""  # ADDED

    run_name: str  # ADDED
    label: str  # ADDED
    images: int  # ADDED
    mean: dict[str, float]  # ADDED
    std: dict[str, float]  # ADDED

    def to_dict(self) -> dict[str, Any]:  # ADDED
        return asdict(self)  # ADDED


def score_images(  # ADDED
    predictions: Sequence[np.ndarray],  # ADDED
    targets: Sequence[np.ndarray],  # ADDED
    image_ids: Sequence[str],  # ADDED
    metrics: Sequence[ImageMetric],  # ADDED
) -> list[dict[str, Any]]:  # ADDED
    """One row per image: ``image_id`` plus one column per metric."""  # ADDED
    return [  # ADDED
        {"image_id": image_id, **{metric.name: metric(pred, target) for metric in metrics}}  # ADDED
        for image_id, pred, target in zip(image_ids, predictions, targets, strict=True)  # ADDED
    ]  # ADDED


def summarise(  # ADDED
    rows: Sequence[Mapping[str, Any]],  # ADDED
    metrics: Sequence[ImageMetric],  # ADDED
    run_name: str,  # ADDED
    label: str,  # ADDED
) -> EvaluationSummary:  # ADDED
    """Mean and std per metric, ignoring non-finite values such as an infinite PSNR."""  # ADDED
    mean, std = {}, {}  # ADDED
    for metric in metrics:  # ADDED
        values = np.array([row[metric.name] for row in rows], dtype=np.float64)  # ADDED
        finite = values[np.isfinite(values)]  # ADDED
        mean[metric.name] = float(finite.mean()) if finite.size else math.nan  # ADDED
        std[metric.name] = float(finite.std()) if finite.size else math.nan  # ADDED
    return EvaluationSummary(run_name, label, len(rows), mean, std)  # ADDED


def write_scores(rows: Sequence[Mapping[str, Any]], csv_path: Path) -> None:  # ADDED
    """Write the per-image score rows to CSV."""  # ADDED
    csv_path.parent.mkdir(parents=True, exist_ok=True)  # ADDED
    with csv_path.open("w", newline="", encoding="utf-8") as handle:  # ADDED
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))  # ADDED
        writer.writeheader()  # ADDED
        writer.writerows(rows)  # ADDED


class SuperResolutionEvaluator:  # ADDED
    """Runs a sampler over the test set; batch ``k`` uses seed ``seed + k`` so that every run  # ADDED
    starts from the same initial noise for the same image."""  # ADDED

    def __init__(self, sampler: ReverseSampler, metrics: Sequence[ImageMetric], seed: int) -> None:  # ADDED
        self._sampler = sampler  # ADDED
        self._metrics = list(metrics)  # ADDED
        self._seed = seed  # ADDED

    def evaluate(  # ADDED
        self,  # ADDED
        batches: Iterable[Batch],  # ADDED
        run_name: str,  # ADDED
        label: str,  # ADDED
        image_dir: Path,  # ADDED
        csv_path: Path,  # ADDED
        summary_path: Path,  # ADDED
    ) -> EvaluationSummary:  # ADDED
        rows: list[dict[str, Any]] = []  # ADDED
        for index, batch in enumerate(batches):  # ADDED
            image_ids = list(batch["image_id"])  # ADDED
            paths = [image_dir / super_resolved_name(run_name, i) for i in image_ids]  # ADDED
            if all(path.is_file() for path in paths):  # ADDED
                predictions = [np.asarray(Image.open(path).convert("RGB")) for path in paths]  # ADDED
            else:  # ADDED
                generated = self._sampler.sample(batch["SR"], seed=self._seed + index)  # ADDED
                predictions = tensor_to_uint8(generated)  # ADDED
                for image_id, image in zip(image_ids, predictions, strict=True):  # ADDED
                    save_image(image, image_dir / super_resolved_name(run_name, image_id))  # ADDED
            targets = tensor_to_uint8(torch.as_tensor(batch["HR"]))  # ADDED
            rows.extend(score_images(predictions, targets, image_ids, self._metrics))  # ADDED
            logger.info("%s: %d images evaluated", label, len(rows))  # ADDED
        return _finish(rows, self._metrics, run_name, label, csv_path, summary_path)  # ADDED


def evaluate_bicubic(  # ADDED
    batches: Iterable[Batch],  # ADDED
    metrics: Sequence[ImageMetric],  # ADDED
    csv_path: Path,  # ADDED
    summary_path: Path,  # ADDED
) -> EvaluationSummary:  # ADDED
    """Reference row: the bicubic-upsampled conditioning image scored against HR."""  # ADDED
    rows: list[dict[str, Any]] = []  # ADDED
    for batch in batches:  # ADDED
        predictions = tensor_to_uint8(torch.as_tensor(batch["SR"]))  # ADDED
        targets = tensor_to_uint8(torch.as_tensor(batch["HR"]))  # ADDED
        rows.extend(score_images(predictions, targets, batch["image_id"], metrics))  # ADDED
    return _finish(rows, metrics, BICUBIC_RUN, BICUBIC_LABEL, csv_path, summary_path)  # ADDED


def _finish(  # ADDED
    rows: list[dict[str, Any]],  # ADDED
    metrics: Sequence[ImageMetric],  # ADDED
    run_name: str,  # ADDED
    label: str,  # ADDED
    csv_path: Path,  # ADDED
    summary_path: Path,  # ADDED
) -> EvaluationSummary:  # ADDED
    if not rows:  # ADDED
        raise RuntimeError("No test images were evaluated.")  # ADDED
    write_scores(rows, csv_path)  # ADDED
    summary = summarise(rows, metrics, run_name, label)  # ADDED
    write_json(summary.to_dict(), summary_path)  # ADDED
    logger.info("%s summary: %s", label, summary.mean)  # ADDED
    return summary  # ADDED
