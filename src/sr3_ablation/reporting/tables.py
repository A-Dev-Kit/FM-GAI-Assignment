# ADDED: new file, not part of the upstream SR3 codebase.
"""Markdown and CSV tables for the report."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sr3_ablation.evaluation.metrics import HIGHER_IS_BETTER
from sr3_ablation.io.jsonio import read_json
from sr3_ablation.io.run_paths import RunPaths
from sr3_ablation.runs import ALL_RUNS, BICUBIC_RUN
from sr3_ablation.schedule.noise_schedule import NoiseSchedule

RESULT_ROWS = (BICUBIC_RUN, *ALL_RUNS)
_ARROWS = {True: "↑", False: "↓"}


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """GitHub-style Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def collect_summaries(output_root: Path) -> list[dict[str, Any]]:
    """Test summaries of the runs evaluated so far, in table order."""
    summaries = []
    for name in RESULT_ROWS:
        path = RunPaths.for_run(output_root, name).metrics_summary
        if path.is_file():
            summaries.append(read_json(path))
    return summaries


def metric_header(metric: str) -> str:
    """Column title such as ``PSNR ↑``."""
    return f"{metric.upper()} {_ARROWS[HIGHER_IS_BETTER.get(metric, True)]}"


def metric_value(metric: str, value: float) -> str:
    """PSNR with 2 decimals, the bounded metrics with 4."""
    precision = 2 if metric == "psnr" else 4
    return f"{value:.{precision}f}"


def results_table(summaries: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> str:
    """Mean ± std of each metric for every evaluated run."""
    header = ["Method", "Test images"] + [metric_header(name) for name in metrics]
    rows = [
        [summary["label"], summary["images"]] + [_mean_std(summary, name) for name in metrics]
        for summary in summaries
    ]
    return markdown_table(header, rows)


def write_results_csv(
    summaries: Sequence[Mapping[str, Any]], metrics: Sequence[str], path: Path
) -> None:
    """The results table as CSV, with means and stds in separate columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["method", "images", *[f"{m}_mean" for m in metrics], *[f"{m}_std" for m in metrics]]
        )
        for summary in summaries:
            writer.writerow(
                [summary["label"], summary["images"]]
                + [summary["mean"].get(m, "") for m in metrics]
                + [summary["std"].get(m, "") for m in metrics]
            )


def schedule_table(original: NoiseSchedule, modified: NoiseSchedule) -> str:
    """Headline numbers of the original and the modified schedule."""
    header = [
        "Schedule",
        "$\\theta_1$",
        "$\\theta_T$",
        "$\\bar\\alpha_T$",
        "SNR$(T)$",
        "first $t$ with SNR $< 1$",
    ]
    rows = []
    for schedule in (original, modified):
        summary = schedule.summary()
        rows.append(
            [
                summary.description,
                scientific(summary.beta_first),
                scientific(summary.beta_last),
                scientific(summary.alpha_bar_last),
                scientific(summary.snr_last),
                summary.first_t_snr_below_one or "never",
            ]
        )
    return markdown_table(header, rows)


def snr_at_timesteps_table(
    original: NoiseSchedule, modified: NoiseSchedule, timesteps: Sequence[int]
) -> str:
    """SNR of both schedules at the given timesteps, and their ratio."""
    noisy = [t for t in timesteps if t >= 1]
    header = ["Quantity", *[f"$t={t}$" for t in noisy]]
    rows = [
        ["SNR, original", *[scientific(original.snr_at(t)) for t in noisy]],
        [f"SNR, {modified.description}", *[scientific(modified.snr_at(t)) for t in noisy]],
        ["ratio", *[f"{modified.snr_at(t) / original.snr_at(t):,.0f}×" for t in noisy]],
    ]
    return markdown_table(header, rows)


def _mean_std(summary: Mapping[str, Any], metric: str) -> str:
    mean = summary["mean"].get(metric)
    std = summary["std"].get(metric)
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "–"
    return f"{metric_value(metric, mean)} ± {metric_value(metric, std)}"


def scientific(value: float) -> str:
    """Plain number for moderate magnitudes, otherwise ``$m\\times10^{e}$`` in LaTeX."""
    if value == 0 or 1e-2 <= abs(value) < 1e4:
        return f"{value:.4g}"
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"
