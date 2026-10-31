# ADDED: new file, not part of the upstream SR3 codebase.
"""Selectors for the code, each run's artifacts, the dataset manifest and the report."""  # ADDED

from __future__ import annotations  # ADDED

import subprocess  # ADDED
from collections.abc import Callable, Iterable, Iterator, Sequence  # ADDED
from pathlib import Path  # ADDED

from sr3_ablation.io.run_paths import RunPaths  # ADDED
from sr3_ablation.submission.artifacts import Artifact  # ADDED

OUTPUTS_FOLDER = "outputs"  # ADDED
IMAGE_PATTERN = "*.png"  # ADDED
STATE_PATTERNS = ("*.png", "*.json")  # ADDED
DATASET_FILES = ("manifest.csv", "split_summary.json")  # ADDED
REPORT_FILES = ("report.pdf", "report.html", "report.md")  # ADDED
ASSETS_FOLDER = "report_assets"  # ADDED

FileLister = Callable[[Path], Sequence[str]]  # ADDED


def git_tracked_files(root: Path) -> list[str]:  # ADDED
    """POSIX paths of the files committed under ``root`` (so local and generated files stay out)."""  # ADDED
    try:  # ADDED
        completed = subprocess.run(  # ADDED
            ["git", "ls-files", "-z"],  # ADDED
            cwd=root,  # ADDED
            capture_output=True,  # ADDED
            check=True,  # ADDED
            timeout=60,  # ADDED
        )  # ADDED
    except subprocess.CalledProcessError as error:  # ADDED
        raise RuntimeError(  # ADDED
            f"{root} is not a git checkout; the submission zip is built from committed files."  # ADDED
        ) from error  # ADDED
    return [name for name in completed.stdout.decode("utf-8").split("\0") if name]  # ADDED


class CodeSelector:  # ADDED
    """Every committed file of the repository: code, configs, tests, README, report sources."""  # ADDED

    def __init__(self, project_root: Path, lister: FileLister = git_tracked_files) -> None:  # ADDED
        self._root = project_root  # ADDED
        self._lister = lister  # ADDED

    def select(self) -> Iterator[Artifact]:  # ADDED
        for name in self._lister(self._root):  # ADDED
            source = self._root / name  # ADDED
            if source.is_file():  # ADDED
                yield Artifact(source, name)  # ADDED


class RunArtifactSelector:  # ADDED
    """One run's logs, records, checkpoint-epoch samples, trajectories and diagnostics.  # ADDED

    Weights and the super-resolved test images stay out; ``test_metrics.csv`` scores them.  # ADDED
    """  # ADDED

    RECORD_FILES = (  # ADDED
        "loss.csv",  # ADDED
        "validation.csv",  # ADDED
        "test_metrics.csv",  # ADDED
        "test_metrics.json",  # ADDED
        "schedule.json",  # ADDED
        "schedule_sampling.json",  # ADDED
        "run.json",  # ADDED
        "environment.json",  # ADDED
        "resolved_config.json",  # ADDED
        "log.txt",  # ADDED
    )  # ADDED

    def __init__(self, output_root: Path, run_name: str) -> None:  # ADDED
        self._output_root = output_root  # ADDED
        self._paths = RunPaths.for_run(output_root, run_name)  # ADDED

    def select(self) -> Iterator[Artifact]:  # ADDED
        records = (self._paths.root / name for name in self.RECORD_FILES)  # ADDED
        samples = sorted(self._paths.samples.glob(IMAGE_PATTERN))  # ADDED
        states = sorted(  # ADDED
            path  # ADDED
            for folder in (self._paths.trajectories, self._paths.diagnostics)  # ADDED
            for pattern in STATE_PATTERNS  # ADDED
            for path in folder.glob(pattern)  # ADDED
        )  # ADDED
        yield from _under_outputs(self._output_root, [*records, *samples, *states])  # ADDED


class DatasetManifestSelector:  # ADDED
    """File lists of the seeded split; image names only, never the images themselves."""  # ADDED

    def __init__(self, output_root: Path) -> None:  # ADDED
        self._output_root = output_root  # ADDED

    def select(self) -> Iterator[Artifact]:  # ADDED
        folder = self._output_root / "dataset"  # ADDED
        yield from _under_outputs(self._output_root, [folder / name for name in DATASET_FILES])  # ADDED


class ReportSelector:  # ADDED
    """The built report and the figures and tables it links to."""  # ADDED

    def __init__(self, project_root: Path, build_dir: Path, output_root: Path) -> None:  # ADDED
        self._project_root = project_root  # ADDED
        self._build_dir = build_dir  # ADDED
        self._output_root = output_root  # ADDED

    def select(self) -> Iterator[Artifact]:  # ADDED
        for name in REPORT_FILES:  # ADDED
            source = self._build_dir / name  # ADDED
            if source.is_file():  # ADDED
                yield Artifact(source, source.relative_to(self._project_root).as_posix())  # ADDED
        assets = sorted(p for p in (self._output_root / ASSETS_FOLDER).glob("*") if p.is_file())  # ADDED
        yield from _under_outputs(self._output_root, assets)  # ADDED


def _under_outputs(output_root: Path, sources: Iterable[Path]) -> Iterator[Artifact]:  # ADDED
    """Existing files, archived as ``outputs/<path relative to output_root>``."""  # ADDED
    for source in sources:  # ADDED
        if source.is_file():  # ADDED
            relative = source.relative_to(output_root).as_posix()  # ADDED
            yield Artifact(source, f"{OUTPUTS_FOLDER}/{relative}")  # ADDED
