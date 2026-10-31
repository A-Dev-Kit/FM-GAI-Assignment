# ADDED: new file, not part of the upstream SR3 codebase.
"""Markdown and CSV tables for the report."""  # ADDED

from __future__ import annotations  # ADDED

import csv  # ADDED
import math  # ADDED
from collections.abc import Mapping, Sequence  # ADDED
from pathlib import Path  # ADDED
from typing import Any  # ADDED

from sr3_ablation.evaluation.metrics import HIGHER_IS_BETTER  # ADDED
from sr3_ablation.io.jsonio import read_json  # ADDED
from sr3_ablation.io.run_paths import RunPaths  # ADDED
from sr3_ablation.runs import ALL_RUNS, BICUBIC_RUN  # ADDED
from sr3_ablation.schedule.noise_schedule import NoiseSchedule  # ADDED

RESULT_ROWS = (BICUBIC_RUN, *ALL_RUNS)  # ADDED
_ARROWS = {True: "↑", False: "↓"}  # ADDED


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:  # ADDED
    """GitHub-style Markdown table."""  # ADDED
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]  # ADDED
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]  # ADDED
    return "\n".join(lines)  # ADDED


def collect_summaries(output_root: Path) -> list[dict[str, Any]]:  # ADDED
    """Test summaries of the runs evaluated so far, in table order."""  # ADDED
    summaries = []  # ADDED
    for name in RESULT_ROWS:  # ADDED
        path = RunPaths.for_run(output_root, name).metrics_summary  # ADDED
        if path.is_file():  # ADDED
            summaries.append(read_json(path))  # ADDED
    return summaries  # ADDED


def metric_header(metric: str) -> str:  # ADDED
    """Column title such as ``PSNR ↑``."""  # ADDED
    return f"{metric.upper()} {_ARROWS[HIGHER_IS_BETTER.get(metric, True)]}"  # ADDED


def metric_value(metric: str, value: float) -> str:  # ADDED
    """PSNR with 2 decimals, the bounded metrics with 4."""  # ADDED
    precision = 2 if metric == "psnr" else 4  # ADDED
    return f"{value:.{precision}f}"  # ADDED


def results_table(summaries: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> str:  # ADDED
    """Mean ± std of each metric for every evaluated run."""  # ADDED
    header = ["Method", "Test images"] + [metric_header(name) for name in metrics]  # ADDED
    rows = [  # ADDED
        [summary["label"], summary["images"]] + [_mean_std(summary, name) for name in metrics]  # ADDED
        for summary in summaries  # ADDED
    ]  # ADDED
    return markdown_table(header, rows)  # ADDED


def write_results_csv(  # ADDED
    summaries: Sequence[Mapping[str, Any]],  # ADDED
    metrics: Sequence[str],  # ADDED
    path: Path,  # ADDED
) -> None:  # ADDED
    """The results table as CSV, with means and stds in separate columns."""  # ADDED
    path.parent.mkdir(parents=True, exist_ok=True)  # ADDED
    with path.open("w", newline="", encoding="utf-8") as handle:  # ADDED
        writer = csv.writer(handle)  # ADDED
        writer.writerow(  # ADDED
            ["method", "images", *[f"{m}_mean" for m in metrics], *[f"{m}_std" for m in metrics]]  # ADDED
        )  # ADDED
        for summary in summaries:  # ADDED
            writer.writerow(  # ADDED
                [summary["label"], summary["images"]]  # ADDED
                + [summary["mean"].get(m, "") for m in metrics]  # ADDED
                + [summary["std"].get(m, "") for m in metrics]  # ADDED
            )  # ADDED


def schedule_table(original: NoiseSchedule, modified: NoiseSchedule) -> str:  # ADDED
    """Headline numbers of the original and the modified schedule."""  # ADDED
    header = [  # ADDED
        "Schedule",  # ADDED
        "$\\theta_1$",  # ADDED
        "$\\theta_T$",  # ADDED
        "$\\bar\\alpha_T$",  # ADDED
        "SNR$(T)$",  # ADDED
        "first $t$ with SNR $< 1$",  # ADDED
    ]  # ADDED
    rows = []  # ADDED
    for schedule in (original, modified):  # ADDED
        summary = schedule.summary()  # ADDED
        rows.append(  # ADDED
            [  # ADDED
                summary.description,  # ADDED
                scientific(summary.beta_first),  # ADDED
                scientific(summary.beta_last),  # ADDED
                scientific(summary.alpha_bar_last),  # ADDED
                scientific(summary.snr_last),  # ADDED
                summary.first_t_snr_below_one or "never",  # ADDED
            ]  # ADDED
        )  # ADDED
    return markdown_table(header, rows)  # ADDED


def snr_at_timesteps_table(  # ADDED
    original: NoiseSchedule,  # ADDED
    modified: NoiseSchedule,  # ADDED
    timesteps: Sequence[int],  # ADDED
) -> str:  # ADDED
    """SNR of both schedules at the given timesteps, and their ratio."""  # ADDED
    noisy = [t for t in timesteps if t >= 1]  # ADDED
    header = ["Quantity", *[f"$t={t}$" for t in noisy]]  # ADDED
    rows = [  # ADDED
        ["SNR, original", *[scientific(original.snr_at(t)) for t in noisy]],  # ADDED
        [f"SNR, {modified.description}", *[scientific(modified.snr_at(t)) for t in noisy]],  # ADDED
        ["ratio", *[f"{modified.snr_at(t) / original.snr_at(t):,.0f}×" for t in noisy]],  # ADDED
    ]  # ADDED
    return markdown_table(header, rows)  # ADDED


def _mean_std(summary: Mapping[str, Any], metric: str) -> str:  # ADDED
    mean = summary["mean"].get(metric)  # ADDED
    std = summary["std"].get(metric)  # ADDED
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):  # ADDED
        return "–"  # ADDED
    return f"{metric_value(metric, mean)} ± {metric_value(metric, std)}"  # ADDED


def scientific(value: float) -> str:  # ADDED
    """Plain number for moderate magnitudes, otherwise ``$m\\times10^{e}$`` in LaTeX."""  # ADDED
    if value == 0 or 1e-2 <= abs(value) < 1e4:  # ADDED
        return f"{value:.4g}"  # ADDED
    mantissa, exponent = f"{value:.2e}".split("e")  # ADDED
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"  # ADDED
