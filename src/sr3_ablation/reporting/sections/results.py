# ADDED: new file, not part of the upstream SR3 codebase.
"""Sections of the test-set results: Run 0 samples, score table and the comparison grid."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sr3_ablation.data.preprocess import layout_for
from sr3_ablation.io.naming import super_resolved_name
from sr3_ablation.reporting.grids import image_grid
from sr3_ablation.reporting.sections.base import (
    EVALUATE_ALL_COMMAND,
    PREPARE_DATA_COMMAND,
    RUN0_COMMAND,
    ReportContext,
    pending,
)
from sr3_ablation.reporting.tables import collect_summaries, results_table, write_results_csv
from sr3_ablation.runs import ALL_RUNS, BICUBIC_LABEL, run_name
from sr3_ablation.utils.selection import evenly_spaced_indices

COMPARISON_IMAGES = 3
RUN0_IMAGES = 4
COMPARED_RUNS = tuple(ALL_RUNS)
RESULTS_CSV_FILE = "results_table.csv"
COMPARISON_FIGURE_FILE = "test_comparison.png"
RUN0_FIGURE_FILE = "run0_samples.png"
INPUT_LABELS = ("LR 16×16", BICUBIC_LABEL)
TARGET_LABEL = "HR 128×128"


@dataclass(frozen=True)
class SampleGrid:
    """Evenly spaced test images, each row: LR input, bicubic, one column per run, ground truth."""

    image_ids: list[str]
    rows: list[list[Path]]
    run_columns: slice

    def any_run_output(self) -> bool:
        return any(path.is_file() for row in self.rows for path in row[self.run_columns])


def sample_grid(context: ReportContext, runs: Sequence[str], count: int) -> SampleGrid | None:
    """Rows for ``count`` evenly spaced test images, or ``None`` before ``prepare-data``."""
    cfg = context.config
    layout = layout_for(
        cfg.paths.data_root, "test", cfg.data.low_resolution, cfg.data.high_resolution
    )
    if not layout.hr_dir.is_dir():
        return None
    image_ids = layout.image_ids(cfg.evaluation.test_limit)
    chosen = [image_ids[i] for i in evenly_spaced_indices(len(image_ids), count)]
    rows = [
        [layout.lr_path(image_id), layout.sr_path(image_id)]
        + [context.run_paths(run).test_outputs / super_resolved_name(run, image_id) for run in runs]
        + [layout.hr_path(image_id)]
        for image_id in chosen
    ]
    first_run = len(INPUT_LABELS)
    return SampleGrid(chosen, rows, slice(first_run, first_run + len(runs)))


def grid_columns(runs: Sequence[str]) -> list[str]:
    """Column titles matching :func:`sample_grid`."""
    return [*INPUT_LABELS, *(ALL_RUNS[run] for run in runs), TARGET_LABEL]


class Run0SamplesSection:
    """The pretrained model applied to our test inputs, before any fine-tuning."""

    placeholder = "PRETRAINED_FIGURE"

    def render(self, context: ReportContext) -> str:
        runs = (run_name("0"),)
        grid = sample_grid(context, runs, RUN0_IMAGES)
        if grid is None:
            return pending(PREPARE_DATA_COMMAND)
        if not grid.any_run_output():
            return pending(RUN0_COMMAND)
        path = image_grid(
            grid.rows, grid.image_ids, grid_columns(runs), context.assets / RUN0_FIGURE_FILE
        )
        return context.image("Run 0: the pretrained FFHQ model on Kvasir-SEG test inputs", path)


class ResultsTableSection:
    """Mean ± std of each test metric for bicubic, Run 0, Run A and Run B."""

    placeholder = "RESULTS_TABLE"

    def render(self, context: ReportContext) -> str:
        summaries = collect_summaries(context.outputs)
        if not summaries:
            return pending(EVALUATE_ALL_COMMAND)
        metrics = list(context.config.evaluation.metrics)
        write_results_csv(summaries, metrics, context.assets / RESULTS_CSV_FILE)
        return results_table(summaries, metrics)


class ComparisonFigureSection:
    """Input, bicubic, Run 0, Run A, Run B and ground truth for evenly spaced test images."""

    placeholder = "COMPARISON_FIGURE"

    def render(self, context: ReportContext) -> str:
        grid = sample_grid(context, COMPARED_RUNS, COMPARISON_IMAGES)
        if grid is None:
            return pending(PREPARE_DATA_COMMAND)
        if not grid.any_run_output():
            return pending(EVALUATE_ALL_COMMAND)
        path = image_grid(
            grid.rows,
            grid.image_ids,
            grid_columns(COMPARED_RUNS),
            context.assets / COMPARISON_FIGURE_FILE,
        )
        return context.image("Test images: input, baselines and both fine-tuned runs", path)
