# ADDED: new file, not part of the upstream SR3 codebase.
"""What every report section shares: the read-only context and the section interface."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sr3_ablation.config.schema import ExperimentConfig
from sr3_ablation.io.run_paths import RunPaths
from sr3_ablation.schedule.noise_schedule import NoiseSchedule
from sr3_ablation.schedule.transforms import IdentityTransform, PowerTransform

ASSETS_FOLDER = "report_assets"

PREPARE_DATA_COMMAND = "python -m sr3_ablation prepare-data"
RUN0_COMMAND = "python -m sr3_ablation run0"
FINETUNE_B_COMMAND = "python scripts/finetune.py --run B"
FINETUNE_BOTH_COMMAND = "python scripts/finetune.py --run A  (and --run B)"
EVALUATE_ALL_COMMAND = "python -m sr3_ablation run0 / evaluate --run A / evaluate --run B"
TRAJECTORIES_BOTH_COMMAND = "python -m sr3_ablation trajectories --run A  (and --run B)"
DIAGNOSE_START_COMMAND = "python -m sr3_ablation diagnose-start --run B"


def pending(command: str) -> str:
    """Markdown note shown in place of an item whose inputs do not exist yet."""
    return f"> *Pending: run `{command}` to generate this item.*"


@dataclass(frozen=True)
class ReportContext:
    """Everything a section may read; sections never write outside ``assets``.

    Attributes:
        config: Resolved experiment configuration.
        exponent: Declared exponent x of Run B.
        original: The native schedule theta_t.
        modified: The power-transformed schedule theta_t ** x.
        build_dir: Folder the report files are written to; image links are relative to it.
    """

    config: ExperimentConfig
    exponent: float
    original: NoiseSchedule
    modified: NoiseSchedule
    build_dir: Path

    @classmethod
    def create(cls, config: ExperimentConfig, exponent: float, build_dir: Path) -> ReportContext:
        """Build the context, deriving both schedules from ``config.schedule``."""
        return cls(
            config=config,
            exponent=exponent,
            original=NoiseSchedule.from_config(config.schedule, IdentityTransform()),
            modified=NoiseSchedule.from_config(config.schedule, PowerTransform(exponent)),
            build_dir=build_dir,
        )

    @property
    def outputs(self) -> Path:
        """Root of all run outputs."""
        return self.config.paths.output_root

    @property
    def assets(self) -> Path:
        """Folder for generated figures and CSV tables."""
        return self.outputs / ASSETS_FOLDER

    def run_paths(self, run_name: str) -> RunPaths:
        """Output layout of one run, e.g. ``runB``."""
        return RunPaths.for_run(self.outputs, run_name)

    def image(self, caption: str, path: Path) -> str:
        """Markdown image link to ``path``, relative to the build folder."""
        relative = Path(os.path.relpath(path, self.build_dir)).as_posix()
        return f"![{caption}]({relative})"


class ReportSection(Protocol):
    """Renders the Markdown that replaces ``{{placeholder}}`` in the report template.

    Implementations must not raise when their inputs are missing; they return
    :func:`pending` text instead, so a partial report can always be built.
    """

    @property
    def placeholder(self) -> str:
        """Template name without braces, e.g. ``EXPONENT``."""
        ...

    def render(self, context: ReportContext) -> str:
        """Return the Markdown for this placeholder."""
        ...
