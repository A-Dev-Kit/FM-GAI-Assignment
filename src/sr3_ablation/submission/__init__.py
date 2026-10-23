# ADDED: new file, not part of the upstream SR3 codebase.
"""Build the Phase 2 submission zip: code, logs, samples, trajectories, metrics and report.

Checkpoints and Kvasir-SEG images are never included (size, and the dataset licence).
"""

from __future__ import annotations

from pathlib import Path

from sr3_ablation.config.schema import ExperimentConfig
from sr3_ablation.paths import PROJECT_ROOT, REPORT_DIR
from sr3_ablation.runs import ALL_RUNS, BICUBIC_RUN, FINE_TUNED_RUNS, run_name
from sr3_ablation.submission.artifacts import (
    DEFAULT_EXCLUSIONS,
    Artifact,
    ArtifactSelector,
    ExclusionRule,
)
from sr3_ablation.submission.packager import (
    IncompleteSubmissionError,
    PackageResult,
    Requirement,
    SubmissionPackager,
)
from sr3_ablation.submission.selectors import (
    CodeSelector,
    DatasetManifestSelector,
    ReportSelector,
    RunArtifactSelector,
)

PACKAGE_NAME = "Group_3_Phase2"
DEFAULT_ZIP = PROJECT_ROOT / "dist" / f"{PACKAGE_NAME}.zip"
PACKAGED_RUNS = (*ALL_RUNS, BICUBIC_RUN)


def brief_requirements(config: ExperimentConfig) -> list[Requirement]:
    """Items the Phase 2 brief asks for, with counts taken from the configuration."""
    snapshots = config.sampling.snapshot_samples * len(config.train.checkpoint_epochs)
    trajectory = config.sampling.trajectory_samples * len(config.sampling.trajectory_timesteps)
    requirements = [
        Requirement("README with reproduction commands", "README.md"),
        Requirement("fine-tuning script declaring NOISE_EXPONENT_X", "scripts/finetune.py"),
        Requirement("modified upstream SR3 file", "third_party/sr3/model/sr3_modules/diffusion.py"),
        Requirement("report PDF", "report/build/report.pdf"),
        Requirement("Run 0 test metrics", f"outputs/{run_name('0')}/test_metrics.json"),
    ]
    for run in FINE_TUNED_RUNS:
        requirements += [
            Requirement(f"{run} loss log", f"outputs/{run}/loss.csv"),
            Requirement(f"{run} checkpoint samples", f"outputs/{run}/samples/*.png", snapshots),
            Requirement(f"{run} trajectories", f"outputs/{run}/trajectories/*.png", trajectory),
            Requirement(f"{run} test metrics", f"outputs/{run}/test_metrics.json"),
            Requirement(f"{run} schedule evidence", f"outputs/{run}/schedule.json"),
        ]
    return requirements


def default_selectors(config: ExperimentConfig, build_dir: Path) -> list[ArtifactSelector]:
    """Code, every run's artifacts, the dataset manifest and the built report."""
    output_root = config.paths.output_root
    return [
        CodeSelector(PROJECT_ROOT),
        *(RunArtifactSelector(output_root, run) for run in PACKAGED_RUNS),
        DatasetManifestSelector(output_root),
        ReportSelector(PROJECT_ROOT, build_dir, output_root),
    ]


def default_packager(
    config: ExperimentConfig, build_dir: Path = REPORT_DIR / "build"
) -> SubmissionPackager:
    """The packager used by ``python -m sr3_ablation package``."""
    return SubmissionPackager(
        default_selectors(config, build_dir), brief_requirements(config), PACKAGE_NAME
    )


__all__ = [
    "DEFAULT_EXCLUSIONS",
    "DEFAULT_ZIP",
    "Artifact",
    "ArtifactSelector",
    "ExclusionRule",
    "IncompleteSubmissionError",
    "PackageResult",
    "Requirement",
    "SubmissionPackager",
    "brief_requirements",
    "default_packager",
    "default_selectors",
]
