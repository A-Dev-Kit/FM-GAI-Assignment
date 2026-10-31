# ADDED: new file, not part of the upstream SR3 codebase.
"""Sections of the fine-tuning results: loss curves, validation PSNR and checkpoint samples."""  # ADDED

from __future__ import annotations  # ADDED

from pathlib import Path  # ADDED

from sr3_ablation.io.naming import epoch_sample_name  # ADDED
from sr3_ablation.reporting.grids import image_grid  # ADDED
from sr3_ablation.reporting.plots import plot_loss_curves, plot_metric_curves  # ADDED
from sr3_ablation.reporting.sections.base import (  # ADDED
    FINETUNE_BOTH_COMMAND,  # ADDED
    ReportContext,  # ADDED
    pending,  # ADDED
)  # ADDED
from sr3_ablation.reporting.tables import markdown_table  # ADDED
from sr3_ablation.runs import FINE_TUNED_RUNS  # ADDED

LOSS_FIGURE_FILE = "loss_curves.png"  # ADDED
VALIDATION_FIGURE_FILE = "validation_psnr.png"  # ADDED
CHECKPOINT_COMPARISON_FILE = "checkpoint_comparison.png"  # ADDED
VALIDATION_METRIC = "psnr"  # ADDED


def loss_series(context: ReportContext) -> dict[str, Path]:  # ADDED
    """``loss.csv`` of every fine-tuned run that has one, keyed by run label."""  # ADDED
    series = {}  # ADDED
    for name, label in FINE_TUNED_RUNS.items():  # ADDED
        path = context.run_paths(name).loss_csv  # ADDED
        if path.is_file():  # ADDED
            series[label] = path  # ADDED
    return series  # ADDED


def read_loss_rows(path: Path) -> dict[int, str]:  # ADDED
    """``epoch -> train_loss`` (kept as written) from a ``loss.csv`` file."""  # ADDED
    lines = path.read_text(encoding="utf-8").strip().splitlines()[1:]  # ADDED
    return {int(epoch): loss for epoch, loss in (line.split(",") for line in lines)}  # ADDED


class LossFigureSection:  # ADDED
    """Training loss of Run A and Run B on one plot."""  # ADDED

    placeholder = "LOSS_FIGURE"  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        series = loss_series(context)  # ADDED
        if not series:  # ADDED
            return pending(FINETUNE_BOTH_COMMAND)  # ADDED
        path = plot_loss_curves(series, context.assets / LOSS_FIGURE_FILE)  # ADDED
        return context.image("Training loss per epoch", path)  # ADDED


class LossTableSection:  # ADDED
    """Per-epoch training loss of both runs side by side."""  # ADDED

    placeholder = "LOSS_TABLE"  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        series = loss_series(context)  # ADDED
        if not series:  # ADDED
            return ""  # ADDED
        columns = {label: read_loss_rows(path) for label, path in series.items()}  # ADDED
        epochs = sorted({epoch for rows in columns.values() for epoch in rows})  # ADDED
        body = [[epoch, *[columns[label].get(epoch, "–") for label in columns]] for epoch in epochs]  # ADDED
        return markdown_table(["Epoch", *[f"{label} train_loss" for label in columns]], body)  # ADDED


class ValidationFigureSection:  # ADDED
    """Validation PSNR of both runs at the checkpoint epochs."""  # ADDED

    placeholder = "VALIDATION_FIGURE"  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        series = {}  # ADDED
        for name, label in FINE_TUNED_RUNS.items():  # ADDED
            path = context.run_paths(name).validation_csv  # ADDED
            if path.is_file():  # ADDED
                series[label] = path  # ADDED
        if not series:  # ADDED
            return pending(FINETUNE_BOTH_COMMAND)  # ADDED
        path = plot_metric_curves(  # ADDED
            series,  # ADDED
            VALIDATION_METRIC,  # ADDED
            "PSNR (dB)",  # ADDED
            "Validation PSNR at checkpoint epochs",  # ADDED
            context.assets / VALIDATION_FIGURE_FILE,  # ADDED
        )  # ADDED
        return context.image("Validation PSNR at checkpoint epochs", path)  # ADDED


class SnapshotFiguresSection:  # ADDED
    """One grid per run: fixed validation samples (rows) at each checkpoint epoch (columns)."""  # ADDED

    placeholder = "SNAPSHOT_FIGURES"  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        epochs = context.config.train.checkpoint_epochs  # ADDED
        samples = range(1, context.config.sampling.snapshot_samples + 1)  # ADDED
        parts = []  # ADDED
        for name, label in FINE_TUNED_RUNS.items():  # ADDED
            folder = context.run_paths(name).samples  # ADDED
            rows = [[folder / epoch_sample_name(name, e, s) for e in epochs] for s in samples]  # ADDED
            if not any(path.is_file() for row in rows for path in row):  # ADDED
                continue  # ADDED
            path = image_grid(  # ADDED
                rows,  # ADDED
                [f"sample {s}" for s in samples],  # ADDED
                [f"epoch {e}" for e in epochs],  # ADDED
                context.assets / f"{name}_snapshots.png",  # ADDED
                title=f"{label}: validation samples",  # ADDED
            )  # ADDED
            parts.append(context.image(f"{label} samples at checkpoint epochs", path))  # ADDED
        return "\n\n".join(parts) or pending(FINETUNE_BOTH_COMMAND)  # ADDED


class CheckpointComparisonSection:  # ADDED
    """Run A and Run B interleaved per sample, one column per checkpoint epoch."""  # ADDED

    placeholder = "CHECKPOINT_COMPARISON_FIGURE"  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        epochs = context.config.train.checkpoint_epochs  # ADDED
        rows, labels = [], []  # ADDED
        for sample in range(1, context.config.sampling.snapshot_samples + 1):  # ADDED
            for name, label in FINE_TUNED_RUNS.items():  # ADDED
                folder = context.run_paths(name).samples  # ADDED
                rows.append([folder / epoch_sample_name(name, e, sample) for e in epochs])  # ADDED
                labels.append(f"sample {sample}, {label}")  # ADDED
        if not all(any(path.is_file() for path in row) for row in rows):  # ADDED
            return pending(FINETUNE_BOTH_COMMAND)  # ADDED
        path = image_grid(  # ADDED
            rows,  # ADDED
            labels,  # ADDED
            [f"epoch {e}" for e in epochs],  # ADDED
            context.assets / CHECKPOINT_COMPARISON_FILE,  # ADDED
            title="Run A vs Run B at each checkpoint epoch",  # ADDED
        )  # ADDED
        return context.image("Run A and Run B side by side at each checkpoint epoch", path)  # ADDED
