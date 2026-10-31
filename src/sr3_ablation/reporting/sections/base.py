# ADDED: new file, not part of the upstream SR3 codebase.
"""What every report section shares: the read-only context and the section interface."""  # ADDED

from __future__ import annotations  # ADDED

import os  # ADDED
from dataclasses import dataclass  # ADDED
from pathlib import Path  # ADDED
from typing import Protocol  # ADDED

from sr3_ablation.config.schema import ExperimentConfig  # ADDED
from sr3_ablation.io.run_paths import RunPaths  # ADDED
from sr3_ablation.schedule.noise_schedule import NoiseSchedule  # ADDED
from sr3_ablation.schedule.transforms import IdentityTransform, PowerTransform  # ADDED

ASSETS_FOLDER = "report_assets"  # ADDED

PREPARE_DATA_COMMAND = "python -m sr3_ablation prepare-data"  # ADDED
RUN0_COMMAND = "python -m sr3_ablation run0"  # ADDED
FINETUNE_B_COMMAND = "python scripts/finetune.py --run B"  # ADDED
FINETUNE_BOTH_COMMAND = "python scripts/finetune.py --run A  (and --run B)"  # ADDED
EVALUATE_ALL_COMMAND = "python -m sr3_ablation run0 / evaluate --run A / evaluate --run B"  # ADDED
TRAJECTORIES_BOTH_COMMAND = "python -m sr3_ablation trajectories --run A  (and --run B)"  # ADDED
DIAGNOSE_START_COMMAND = "python -m sr3_ablation diagnose-start --run B"  # ADDED


def pending(command: str) -> str:  # ADDED
    """Markdown note shown in place of an item whose inputs do not exist yet."""  # ADDED
    return f"> *Pending: run `{command}` to generate this item.*"  # ADDED


@dataclass(frozen=True)  # ADDED
class ReportContext:  # ADDED
    """Everything a section may read; sections never write outside ``assets``.  # ADDED

    Attributes:  # ADDED
        config: Resolved experiment configuration.  # ADDED
        exponent: Declared exponent x of Run B.  # ADDED
        original: The native schedule theta_t.  # ADDED
        modified: The power-transformed schedule theta_t ** x.  # ADDED
        build_dir: Folder the report files are written to; image links are relative to it.  # ADDED
    """  # ADDED

    config: ExperimentConfig  # ADDED
    exponent: float  # ADDED
    original: NoiseSchedule  # ADDED
    modified: NoiseSchedule  # ADDED
    build_dir: Path  # ADDED

    @classmethod  # ADDED
    def create(cls, config: ExperimentConfig, exponent: float, build_dir: Path) -> ReportContext:  # ADDED
        """Build the context, deriving both schedules from ``config.schedule``."""  # ADDED
        return cls(  # ADDED
            config=config,  # ADDED
            exponent=exponent,  # ADDED
            original=NoiseSchedule.from_config(config.schedule, IdentityTransform()),  # ADDED
            modified=NoiseSchedule.from_config(config.schedule, PowerTransform(exponent)),  # ADDED
            build_dir=build_dir,  # ADDED
        )  # ADDED

    @property  # ADDED
    def outputs(self) -> Path:  # ADDED
        """Root of all run outputs."""  # ADDED
        return self.config.paths.output_root  # ADDED

    @property  # ADDED
    def assets(self) -> Path:  # ADDED
        """Folder for generated figures and CSV tables."""  # ADDED
        return self.outputs / ASSETS_FOLDER  # ADDED

    def run_paths(self, run_name: str) -> RunPaths:  # ADDED
        """Output layout of one run, e.g. ``runB``."""  # ADDED
        return RunPaths.for_run(self.outputs, run_name)  # ADDED

    def image(self, caption: str, path: Path) -> str:  # ADDED
        """Markdown image link to ``path``, relative to the build folder."""  # ADDED
        relative = Path(os.path.relpath(path, self.build_dir)).as_posix()  # ADDED
        return f"![{caption}]({relative})"  # ADDED


class ReportSection(Protocol):  # ADDED
    """Renders the Markdown that replaces ``{{placeholder}}`` in the report template.  # ADDED

    Implementations must not raise when their inputs are missing; they return  # ADDED
    :func:`pending` text instead, so a partial report can always be built.  # ADDED
    """  # ADDED

    @property  # ADDED
    def placeholder(self) -> str:  # ADDED
        """Template name without braces, e.g. ``EXPONENT``."""  # ADDED
        ...  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        """Return the Markdown for this placeholder."""  # ADDED
        ...  # ADDED
