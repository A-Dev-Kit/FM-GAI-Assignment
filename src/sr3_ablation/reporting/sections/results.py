# ADDED: new file, not part of the upstream SR3 codebase.
"""Sections of the test-set results: Run 0 samples, score table and the comparison grid."""  # ADDED

from __future__ import annotations  # ADDED

from collections.abc import Sequence  # ADDED
from dataclasses import dataclass  # ADDED
from pathlib import Path  # ADDED

from sr3_ablation.data.preprocess import layout_for  # ADDED
from sr3_ablation.io.naming import super_resolved_name  # ADDED
from sr3_ablation.reporting.grids import image_grid  # ADDED
from sr3_ablation.reporting.sections.base import (  # ADDED
    EVALUATE_ALL_COMMAND,  # ADDED
    PREPARE_DATA_COMMAND,  # ADDED
    RUN0_COMMAND,  # ADDED
    ReportContext,  # ADDED
    pending,  # ADDED
)  # ADDED
from sr3_ablation.reporting.tables import collect_summaries, results_table, write_results_csv  # ADDED
from sr3_ablation.runs import ALL_RUNS, BICUBIC_LABEL, run_name  # ADDED
from sr3_ablation.utils.selection import evenly_spaced_indices  # ADDED

COMPARISON_IMAGES = 3  # ADDED
RUN0_IMAGES = 4  # ADDED
COMPARED_RUNS = tuple(ALL_RUNS)  # ADDED
RESULTS_CSV_FILE = "results_table.csv"  # ADDED
COMPARISON_FIGURE_FILE = "test_comparison.png"  # ADDED
RUN0_FIGURE_FILE = "run0_samples.png"  # ADDED
INPUT_LABELS = ("LR 16×16", BICUBIC_LABEL)  # ADDED
TARGET_LABEL = "HR 128×128"  # ADDED


@dataclass(frozen=True)  # ADDED
class SampleGrid:  # ADDED
    """Evenly spaced test images, each row: LR input, bicubic, one column per run, ground truth."""  # ADDED

    image_ids: list[str]  # ADDED
    rows: list[list[Path]]  # ADDED
    run_columns: slice  # ADDED

    def any_run_output(self) -> bool:  # ADDED
        return any(path.is_file() for row in self.rows for path in row[self.run_columns])  # ADDED


def sample_grid(context: ReportContext, runs: Sequence[str], count: int) -> SampleGrid | None:  # ADDED
    """Rows for ``count`` evenly spaced test images, or ``None`` before ``prepare-data``."""  # ADDED
    cfg = context.config  # ADDED
    layout = layout_for(  # ADDED
        cfg.paths.data_root,  # ADDED
        "test",  # ADDED
        cfg.data.low_resolution,  # ADDED
        cfg.data.high_resolution,  # ADDED
    )  # ADDED
    if not layout.hr_dir.is_dir():  # ADDED
        return None  # ADDED
    image_ids = layout.image_ids(cfg.evaluation.test_limit)  # ADDED
    chosen = [image_ids[i] for i in evenly_spaced_indices(len(image_ids), count)]  # ADDED
    rows = [  # ADDED
        [layout.lr_path(image_id), layout.sr_path(image_id)]  # ADDED
        + [context.run_paths(run).test_outputs / super_resolved_name(run, image_id) for run in runs]  # ADDED
        + [layout.hr_path(image_id)]  # ADDED
        for image_id in chosen  # ADDED
    ]  # ADDED
    first_run = len(INPUT_LABELS)  # ADDED
    return SampleGrid(chosen, rows, slice(first_run, first_run + len(runs)))  # ADDED


def grid_columns(runs: Sequence[str]) -> list[str]:  # ADDED
    """Column titles matching :func:`sample_grid`."""  # ADDED
    return [*INPUT_LABELS, *(ALL_RUNS[run] for run in runs), TARGET_LABEL]  # ADDED


class Run0SamplesSection:  # ADDED
    """The pretrained model applied to our test inputs, before any fine-tuning."""  # ADDED

    placeholder = "PRETRAINED_FIGURE"  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        runs = (run_name("0"),)  # ADDED
        grid = sample_grid(context, runs, RUN0_IMAGES)  # ADDED
        if grid is None:  # ADDED
            return pending(PREPARE_DATA_COMMAND)  # ADDED
        if not grid.any_run_output():  # ADDED
            return pending(RUN0_COMMAND)  # ADDED
        path = image_grid(  # ADDED
            grid.rows,  # ADDED
            grid.image_ids,  # ADDED
            grid_columns(runs),  # ADDED
            context.assets / RUN0_FIGURE_FILE,  # ADDED
        )  # ADDED
        return context.image("Run 0: the pretrained FFHQ model on Kvasir-SEG test inputs", path)  # ADDED


class ResultsTableSection:  # ADDED
    """Mean ± std of each test metric for bicubic, Run 0, Run A and Run B."""  # ADDED

    placeholder = "RESULTS_TABLE"  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        summaries = collect_summaries(context.outputs)  # ADDED
        if not summaries:  # ADDED
            return pending(EVALUATE_ALL_COMMAND)  # ADDED
        metrics = list(context.config.evaluation.metrics)  # ADDED
        write_results_csv(summaries, metrics, context.assets / RESULTS_CSV_FILE)  # ADDED
        return results_table(summaries, metrics)  # ADDED


class ComparisonFigureSection:  # ADDED
    """Input, bicubic, Run 0, Run A, Run B and ground truth for evenly spaced test images."""  # ADDED

    placeholder = "COMPARISON_FIGURE"  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        grid = sample_grid(context, COMPARED_RUNS, COMPARISON_IMAGES)  # ADDED
        if grid is None:  # ADDED
            return pending(PREPARE_DATA_COMMAND)  # ADDED
        if not grid.any_run_output():  # ADDED
            return pending(EVALUATE_ALL_COMMAND)  # ADDED
        path = image_grid(  # ADDED
            grid.rows,  # ADDED
            grid.image_ids,  # ADDED
            grid_columns(COMPARED_RUNS),  # ADDED
            context.assets / COMPARISON_FIGURE_FILE,  # ADDED
        )  # ADDED
        return context.image("Test images: input, baselines and both fine-tuned runs", path)  # ADDED
