# ADDED: new file, not part of the upstream SR3 codebase.
"""Sections of the fine-tuning results: loss curves, validation PSNR and checkpoint samples."""

from __future__ import annotations

from pathlib import Path

from sr3_ablation.io.naming import epoch_sample_name
from sr3_ablation.reporting.grids import image_grid
from sr3_ablation.reporting.plots import plot_loss_curves, plot_metric_curves
from sr3_ablation.reporting.sections.base import (
    FINETUNE_BOTH_COMMAND,
    ReportContext,
    pending,
)
from sr3_ablation.reporting.tables import markdown_table
from sr3_ablation.runs import FINE_TUNED_RUNS

LOSS_FIGURE_FILE = "loss_curves.png"
VALIDATION_FIGURE_FILE = "validation_psnr.png"
CHECKPOINT_COMPARISON_FILE = "checkpoint_comparison.png"
VALIDATION_METRIC = "psnr"


def loss_series(context: ReportContext) -> dict[str, Path]:
    """``loss.csv`` of every fine-tuned run that has one, keyed by run label."""
    series = {}
    for name, label in FINE_TUNED_RUNS.items():
        path = context.run_paths(name).loss_csv
        if path.is_file():
            series[label] = path
    return series


def read_loss_rows(path: Path) -> dict[int, str]:
    """``epoch -> train_loss`` (kept as written) from a ``loss.csv`` file."""
    lines = path.read_text(encoding="utf-8").strip().splitlines()[1:]
    return {int(epoch): loss for epoch, loss in (line.split(",") for line in lines)}


class LossFigureSection:
    """Training loss of Run A and Run B on one plot."""

    placeholder = "LOSS_FIGURE"

    def render(self, context: ReportContext) -> str:
        series = loss_series(context)
        if not series:
            return pending(FINETUNE_BOTH_COMMAND)
        path = plot_loss_curves(series, context.assets / LOSS_FIGURE_FILE)
        return context.image("Training loss per epoch", path)


class LossTableSection:
    """Per-epoch training loss of both runs side by side."""

    placeholder = "LOSS_TABLE"

    def render(self, context: ReportContext) -> str:
        series = loss_series(context)
        if not series:
            return ""
        columns = {label: read_loss_rows(path) for label, path in series.items()}
        epochs = sorted({epoch for rows in columns.values() for epoch in rows})
        body = [[epoch, *[columns[label].get(epoch, "–") for label in columns]] for epoch in epochs]
        return markdown_table(["Epoch", *[f"{label} train_loss" for label in columns]], body)


class ValidationFigureSection:
    """Validation PSNR of both runs at the checkpoint epochs."""

    placeholder = "VALIDATION_FIGURE"

    def render(self, context: ReportContext) -> str:
        series = {}
        for name, label in FINE_TUNED_RUNS.items():
            path = context.run_paths(name).validation_csv
            if path.is_file():
                series[label] = path
        if not series:
            return pending(FINETUNE_BOTH_COMMAND)
        path = plot_metric_curves(
            series,
            VALIDATION_METRIC,
            "PSNR (dB)",
            "Validation PSNR at checkpoint epochs",
            context.assets / VALIDATION_FIGURE_FILE,
        )
        return context.image("Validation PSNR at checkpoint epochs", path)


class SnapshotFiguresSection:
    """One grid per run: fixed validation samples (rows) at each checkpoint epoch (columns)."""

    placeholder = "SNAPSHOT_FIGURES"

    def render(self, context: ReportContext) -> str:
        epochs = context.config.train.checkpoint_epochs
        samples = range(1, context.config.sampling.snapshot_samples + 1)
        parts = []
        for name, label in FINE_TUNED_RUNS.items():
            folder = context.run_paths(name).samples
            rows = [[folder / epoch_sample_name(name, e, s) for e in epochs] for s in samples]
            if not any(path.is_file() for row in rows for path in row):
                continue
            path = image_grid(
                rows,
                [f"sample {s}" for s in samples],
                [f"epoch {e}" for e in epochs],
                context.assets / f"{name}_snapshots.png",
                title=f"{label}: validation samples",
            )
            parts.append(context.image(f"{label} samples at checkpoint epochs", path))
        return "\n\n".join(parts) or pending(FINETUNE_BOTH_COMMAND)


class CheckpointComparisonSection:
    """Run A and Run B interleaved per sample, one column per checkpoint epoch."""

    placeholder = "CHECKPOINT_COMPARISON_FIGURE"

    def render(self, context: ReportContext) -> str:
        epochs = context.config.train.checkpoint_epochs
        rows, labels = [], []
        for sample in range(1, context.config.sampling.snapshot_samples + 1):
            for name, label in FINE_TUNED_RUNS.items():
                folder = context.run_paths(name).samples
                rows.append([folder / epoch_sample_name(name, e, sample) for e in epochs])
                labels.append(f"sample {sample}, {label}")
        if not all(any(path.is_file() for path in row) for row in rows):
            return pending(FINETUNE_BOTH_COMMAND)
        path = image_grid(
            rows,
            labels,
            [f"epoch {e}" for e in epochs],
            context.assets / CHECKPOINT_COMPARISON_FILE,
            title="Run A vs Run B at each checkpoint epoch",
        )
        return context.image("Run A and Run B side by side at each checkpoint epoch", path)
