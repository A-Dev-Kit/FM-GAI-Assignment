# ADDED: new file, not part of the upstream SR3 codebase.
"""Line plots: training loss, validation metrics and the two noise schedules."""  # ADDED

from __future__ import annotations  # ADDED

import csv  # ADDED
from collections.abc import Mapping, Sequence  # ADDED
from pathlib import Path  # ADDED

import matplotlib  # ADDED

matplotlib.use("Agg")  # ADDED
import matplotlib.pyplot as plt  # ADDED
import numpy as np  # ADDED

from sr3_ablation.config.schema import ScheduleConfig  # ADDED
from sr3_ablation.schedule.noise_schedule import NoiseSchedule  # ADDED
from sr3_ablation.schedule.transforms import IdentityTransform, PowerTransform  # ADDED

DPI = 160  # ADDED


def read_columns(path: Path) -> dict[str, list[float]]:  # ADDED
    """CSV columns as lists of floats, keyed by header."""  # ADDED
    with path.open(newline="", encoding="utf-8") as handle:  # ADDED
        rows = list(csv.DictReader(handle))  # ADDED
    return {key: [float(row[key]) for row in rows] for key in (rows[0] if rows else {})}  # ADDED


def plot_metric_curves(  # ADDED
    series: Mapping[str, Path],  # ADDED
    column: str,  # ADDED
    ylabel: str,  # ADDED
    title: str,  # ADDED
    output: Path,  # ADDED
) -> Path:  # ADDED
    """One line per run of ``column`` against ``epoch`` read from each CSV."""  # ADDED
    figure, axis = plt.subplots(figsize=(6.4, 3.6))  # ADDED
    for label, path in series.items():  # ADDED
        columns = read_columns(path)  # ADDED
        axis.plot(columns["epoch"], columns[column], marker="o", label=label)  # ADDED
    axis.set_xlabel("epoch")  # ADDED
    axis.set_ylabel(ylabel)  # ADDED
    axis.set_title(title)  # ADDED
    axis.grid(alpha=0.3)  # ADDED
    axis.legend()  # ADDED
    return _save(figure, output)  # ADDED


def plot_loss_curves(series: Mapping[str, Path], output: Path) -> Path:  # ADDED
    """Training-loss curves of all runs on one set of axes."""  # ADDED
    return plot_metric_curves(  # ADDED
        series,  # ADDED
        "train_loss",  # ADDED
        "mean L1 noise-prediction loss",  # ADDED
        "Training loss per epoch",  # ADDED
        output,  # ADDED
    )  # ADDED


def plot_schedule_comparison(  # ADDED
    config: ScheduleConfig,  # ADDED
    exponent: float,  # ADDED
    marks: Sequence[int],  # ADDED
    output: Path,  # ADDED
) -> Path:  # ADDED
    """beta_t, alpha_bar_t and SNR(t) for the original and the power-transformed schedule."""  # ADDED
    original = NoiseSchedule.from_config(config, IdentityTransform())  # ADDED
    modified = NoiseSchedule.from_config(config, PowerTransform(exponent))  # ADDED
    steps = np.arange(1, config.n_timestep + 1)  # ADDED
    panels = (  # ADDED
        ("$\\theta_t$ (log scale)", lambda s: s.betas, True),  # ADDED
        ("$\\bar\\alpha_t$", lambda s: s.alphas_cumprod, False),  # ADDED
        ("SNR$(t)$ (log scale)", lambda s: s.snr, True),  # ADDED
    )  # ADDED
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.6))  # ADDED
    for axis, (title, values, log_scale) in zip(axes, panels, strict=True):  # ADDED
        axis.plot(steps, values(original), label="original $\\theta_t$")  # ADDED
        axis.plot(steps, values(modified), label=f"$\\theta_t^{{{exponent:.2f}}}$")  # ADDED
        for mark in marks:  # ADDED
            if 1 <= mark <= config.n_timestep:  # ADDED
                axis.axvline(mark, color="grey", lw=0.6, ls=":")  # ADDED
        if log_scale:  # ADDED
            axis.set_yscale("log")  # ADDED
        axis.set_title(title)  # ADDED
        axis.set_xlabel("timestep $t$")  # ADDED
        axis.grid(alpha=0.3)  # ADDED
    axes[2].axhline(1.0, color="black", lw=0.8, ls="--")  # ADDED
    axes[0].legend()  # ADDED
    figure.suptitle("Noise schedules (dotted lines: saved trajectory timesteps)")  # ADDED
    return _save(figure, output)  # ADDED


def _save(figure: plt.Figure, output: Path) -> Path:  # ADDED
    output.parent.mkdir(parents=True, exist_ok=True)  # ADDED
    figure.tight_layout()  # ADDED
    figure.savefig(output, dpi=DPI)  # ADDED
    plt.close(figure)  # ADDED
    return output  # ADDED
