# ADDED: new file, not part of the upstream SR3 codebase.
"""Build the Phase 2 submission zip: code, logs, samples, trajectories, metrics and report.  # ADDED

Checkpoints and Kvasir-SEG images are never included (size, and the dataset licence).  # ADDED
"""  # ADDED

from __future__ import annotations  # ADDED

from pathlib import Path  # ADDED

from sr3_ablation.config.schema import ExperimentConfig  # ADDED
from sr3_ablation.paths import PROJECT_ROOT, REPORT_DIR  # ADDED
from sr3_ablation.runs import ALL_RUNS, BICUBIC_RUN, FINE_TUNED_RUNS, run_name  # ADDED
from sr3_ablation.submission.artifacts import (  # ADDED
    DEFAULT_EXCLUSIONS,  # ADDED
    Artifact,  # ADDED
    ArtifactSelector,  # ADDED
    ExclusionRule,  # ADDED
)  # ADDED
from sr3_ablation.submission.packager import (  # ADDED
    IncompleteSubmissionError,  # ADDED
    PackageResult,  # ADDED
    Requirement,  # ADDED
    SubmissionPackager,  # ADDED
)  # ADDED
from sr3_ablation.submission.selectors import (  # ADDED
    CodeSelector,  # ADDED
    DatasetManifestSelector,  # ADDED
    ReportSelector,  # ADDED
    RunArtifactSelector,  # ADDED
)  # ADDED

PACKAGE_NAME = "Group_3_Phase2"  # ADDED
DEFAULT_ZIP = PROJECT_ROOT / "dist" / f"{PACKAGE_NAME}.zip"  # ADDED
PACKAGED_RUNS = (*ALL_RUNS, BICUBIC_RUN)  # ADDED


def brief_requirements(config: ExperimentConfig) -> list[Requirement]:  # ADDED
    """Items the Phase 2 brief asks for, with counts taken from the configuration."""  # ADDED
    snapshots = config.sampling.snapshot_samples * len(config.train.checkpoint_epochs)  # ADDED
    trajectory = config.sampling.trajectory_samples * len(config.sampling.trajectory_timesteps)  # ADDED
    requirements = [  # ADDED
        Requirement("README with reproduction commands", "README.md"),  # ADDED
        Requirement("fine-tuning script declaring NOISE_EXPONENT_X", "scripts/finetune.py"),  # ADDED
        Requirement("modified upstream SR3 file", "third_party/sr3/model/sr3_modules/diffusion.py"),  # ADDED
        Requirement("report PDF", "report/build/report.pdf"),  # ADDED
        Requirement("Run 0 test metrics", f"outputs/{run_name('0')}/test_metrics.json"),  # ADDED
    ]  # ADDED
    for run in FINE_TUNED_RUNS:  # ADDED
        requirements += [  # ADDED
            Requirement(f"{run} loss log", f"outputs/{run}/loss.csv"),  # ADDED
            Requirement(f"{run} checkpoint samples", f"outputs/{run}/samples/*.png", snapshots),  # ADDED
            Requirement(f"{run} trajectories", f"outputs/{run}/trajectories/*.png", trajectory),  # ADDED
            Requirement(f"{run} test metrics", f"outputs/{run}/test_metrics.json"),  # ADDED
            Requirement(f"{run} schedule evidence", f"outputs/{run}/schedule.json"),  # ADDED
        ]  # ADDED
    return requirements  # ADDED


def default_selectors(config: ExperimentConfig, build_dir: Path) -> list[ArtifactSelector]:  # ADDED
    """Code, every run's artifacts, the dataset manifest and the built report."""  # ADDED
    output_root = config.paths.output_root  # ADDED
    return [  # ADDED
        CodeSelector(PROJECT_ROOT),  # ADDED
        *(RunArtifactSelector(output_root, run) for run in PACKAGED_RUNS),  # ADDED
        DatasetManifestSelector(output_root),  # ADDED
        ReportSelector(PROJECT_ROOT, build_dir, output_root),  # ADDED
    ]  # ADDED


def default_packager(  # ADDED
    config: ExperimentConfig,  # ADDED
    build_dir: Path = REPORT_DIR / "build",  # ADDED
) -> SubmissionPackager:  # ADDED
    """The packager used by ``python -m sr3_ablation package``."""  # ADDED
    return SubmissionPackager(  # ADDED
        default_selectors(config, build_dir),  # ADDED
        brief_requirements(config),  # ADDED
        PACKAGE_NAME,  # ADDED
    )  # ADDED


__all__ = [  # ADDED
    "DEFAULT_EXCLUSIONS",  # ADDED
    "DEFAULT_ZIP",  # ADDED
    "Artifact",  # ADDED
    "ArtifactSelector",  # ADDED
    "ExclusionRule",  # ADDED
    "IncompleteSubmissionError",  # ADDED
    "PackageResult",  # ADDED
    "Requirement",  # ADDED
    "SubmissionPackager",  # ADDED
    "brief_requirements",  # ADDED
    "default_packager",  # ADDED
    "default_selectors",  # ADDED
]  # ADDED
