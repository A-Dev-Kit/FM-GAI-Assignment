# ADDED: new file, not part of the upstream SR3 codebase.
"""Line plots: training loss, validation metrics and the two noise schedules."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sr3_ablation.config.schema import ScheduleConfig
from sr3_ablation.schedule.noise_schedule import NoiseSchedule
from sr3_ablation.schedule.transforms import IdentityTransform, PowerTransform

DPI = 160


def read_columns(path: Path) -> dict[str, list[float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {key: [float(row[key]) for row in rows] for key in (rows[0] if rows else {})}


def plot_metric_curves(
    series: Mapping[str, Path], column: str, ylabel: str, title: str, output: Path
) -> Path:
    """One line per run of ``column`` against ``epoch`` read from each CSV."""
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    for label, path in series.items():
        columns = read_columns(path)
        axis.plot(columns["epoch"], columns[column], marker="o", label=label)
    axis.set_xlabel("epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.3)
    axis.legend()
    return _save(figure, output)


def plot_loss_curves(series: Mapping[str, Path], output: Path) -> Path:
    return plot_metric_curves(
        series, "train_loss", "mean L1 noise-prediction loss", "Training loss per epoch", output
    )


def plot_schedule_comparison(
    config: ScheduleConfig, exponent: float, marks: Sequence[int], output: Path
) -> Path:
    """beta_t, alpha_bar_t and SNR(t) for the original and the power-transformed schedule."""
    original = NoiseSchedule.from_config(config, IdentityTransform())
    modified = NoiseSchedule.from_config(config, PowerTransform(exponent))
    steps = np.arange(1, config.n_timestep + 1)
    panels = (
        ("$\\theta_t$ (log scale)", lambda s: s.betas, True),
        ("$\\bar\\alpha_t$", lambda s: s.alphas_cumprod, False),
        ("SNR$(t)$ (log scale)", lambda s: s.snr, True),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.6))
    for axis, (title, values, log_scale) in zip(axes, panels, strict=True):
        axis.plot(steps, values(original), label="original $\\theta_t$")
        axis.plot(steps, values(modified), label=f"$\\theta_t^{{{exponent:.2f}}}$")
        for mark in marks:
            if 1 <= mark <= config.n_timestep:
                axis.axvline(mark, color="grey", lw=0.6, ls=":")
        if log_scale:
            axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("timestep $t$")
        axis.grid(alpha=0.3)
    axes[2].axhline(1.0, color="black", lw=0.8, ls="--")
    axes[0].legend()
    figure.suptitle("Noise schedules (dotted lines: saved trajectory timesteps)")
    return _save(figure, output)


def _save(figure: plt.Figure, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=DPI)
    plt.close(figure)
    return output
