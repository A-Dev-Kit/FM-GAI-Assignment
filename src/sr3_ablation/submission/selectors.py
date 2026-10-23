# ADDED: new file, not part of the upstream SR3 codebase.
"""Selectors for the code, each run's artifacts, the dataset manifest and the report."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

from sr3_ablation.io.run_paths import RunPaths
from sr3_ablation.submission.artifacts import Artifact

OUTPUTS_FOLDER = "outputs"
IMAGE_PATTERN = "*.png"
STATE_PATTERNS = ("*.png", "*.json")
DATASET_FILES = ("manifest.csv", "split_summary.json")
REPORT_FILES = ("report.pdf", "report.html", "report.md")
ASSETS_FOLDER = "report_assets"

FileLister = Callable[[Path], Sequence[str]]


def git_tracked_files(root: Path) -> list[str]:
    """POSIX paths of the files committed under ``root`` (so local and generated files stay out)."""
    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            capture_output=True,
            check=True,
            timeout=60,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"{root} is not a git checkout; the submission zip is built from committed files."
        ) from error
    return [name for name in completed.stdout.decode("utf-8").split("\0") if name]


class CodeSelector:
    """Every committed file of the repository: code, configs, tests, README, report sources."""

    def __init__(self, project_root: Path, lister: FileLister = git_tracked_files) -> None:
        self._root = project_root
        self._lister = lister

    def select(self) -> Iterator[Artifact]:
        for name in self._lister(self._root):
            source = self._root / name
            if source.is_file():
                yield Artifact(source, name)


class RunArtifactSelector:
    """One run's logs, records, checkpoint-epoch samples, trajectories and diagnostics.

    Weights and the super-resolved test images stay out; ``test_metrics.csv`` scores them.
    """

    RECORD_FILES = (
        "loss.csv",
        "validation.csv",
        "test_metrics.csv",
        "test_metrics.json",
        "schedule.json",
        "schedule_sampling.json",
        "run.json",
        "environment.json",
        "resolved_config.json",
        "log.txt",
    )

    def __init__(self, output_root: Path, run_name: str) -> None:
        self._output_root = output_root
        self._paths = RunPaths.for_run(output_root, run_name)

    def select(self) -> Iterator[Artifact]:
        records = (self._paths.root / name for name in self.RECORD_FILES)
        samples = sorted(self._paths.samples.glob(IMAGE_PATTERN))
        states = sorted(
            path
            for folder in (self._paths.trajectories, self._paths.diagnostics)
            for pattern in STATE_PATTERNS
            for path in folder.glob(pattern)
        )
        yield from _under_outputs(self._output_root, [*records, *samples, *states])


class DatasetManifestSelector:
    """File lists of the seeded split; image names only, never the images themselves."""

    def __init__(self, output_root: Path) -> None:
        self._output_root = output_root

    def select(self) -> Iterator[Artifact]:
        folder = self._output_root / "dataset"
        yield from _under_outputs(self._output_root, [folder / name for name in DATASET_FILES])


class ReportSelector:
    """The built report and the figures and tables it links to."""

    def __init__(self, project_root: Path, build_dir: Path, output_root: Path) -> None:
        self._project_root = project_root
        self._build_dir = build_dir
        self._output_root = output_root

    def select(self) -> Iterator[Artifact]:
        for name in REPORT_FILES:
            source = self._build_dir / name
            if source.is_file():
                yield Artifact(source, source.relative_to(self._project_root).as_posix())
        assets = sorted(p for p in (self._output_root / ASSETS_FOLDER).glob("*") if p.is_file())
        yield from _under_outputs(self._output_root, assets)


def _under_outputs(output_root: Path, sources: Iterable[Path]) -> Iterator[Artifact]:
    """Existing files, archived as ``outputs/<path relative to output_root>``."""
    for source in sources:
        if source.is_file():
            relative = source.relative_to(output_root).as_posix()
            yield Artifact(source, f"{OUTPUTS_FOLDER}/{relative}")
