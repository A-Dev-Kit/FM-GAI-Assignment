# ADDED: new file, not part of the upstream SR3 codebase.
"""Write the submission zip from a set of selectors and check it covers the brief."""

from __future__ import annotations

import fnmatch
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from sr3_ablation.submission.artifacts import (
    DEFAULT_EXCLUSIONS,
    Artifact,
    ArtifactSelector,
    ExclusionRule,
)

MANIFEST_NAME = "MANIFEST.txt"


@dataclass(frozen=True)
class Requirement:
    """At least ``minimum`` archived files must match the glob ``pattern``."""

    description: str
    pattern: str
    minimum: int = 1

    def count(self, names: Sequence[str]) -> int:
        return sum(1 for name in names if fnmatch.fnmatch(name, self.pattern))

    def is_met(self, names: Sequence[str]) -> bool:
        return self.count(names) >= self.minimum


@dataclass(frozen=True)
class PackageResult:
    """The zip written and what it holds."""

    path: Path
    names: tuple[str, ...]
    missing: tuple[Requirement, ...]

    @property
    def complete(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict[str, object]:
        return {
            "zip": str(self.path),
            "files": len(self.names),
            "megabytes": round(self.path.stat().st_size / 2**20, 2),
            "missing": [f"{r.description} ({r.pattern}, need {r.minimum})" for r in self.missing],
        }


class IncompleteSubmissionError(RuntimeError):
    """Raised when brief items are missing and an incomplete zip was not allowed."""


class SubmissionPackager:
    """Collects artifacts from every selector, filters them and writes one zip.

    Every archive name is placed under ``root_folder`` so unzipping creates one folder.
    """

    def __init__(
        self,
        selectors: Sequence[ArtifactSelector],
        requirements: Sequence[Requirement],
        root_folder: str,
        exclusions: ExclusionRule = DEFAULT_EXCLUSIONS,
    ) -> None:
        self._selectors = tuple(selectors)
        self._requirements = tuple(requirements)
        self._root_folder = root_folder
        self._exclusions = exclusions

    def build(self, zip_path: Path, allow_incomplete: bool = False) -> PackageResult:
        """Write ``zip_path``; refuse (before writing) if requirements fail, unless allowed."""
        artifacts: dict[str, Artifact] = {}
        for selector in self._selectors:
            for artifact in selector.select():
                if self._exclusions.excludes(artifact.archive_name):
                    continue
                previous = artifacts.setdefault(artifact.archive_name, artifact)
                if previous.source != artifact.source:
                    raise ValueError(f"Two files claim {artifact.archive_name}")
        names = tuple(sorted(artifacts))
        missing = tuple(r for r in self._requirements if not r.is_met(names))
        if missing and not allow_incomplete:
            details = "; ".join(f"{r.description} ({r.pattern})" for r in missing)
            raise IncompleteSubmissionError(f"Missing brief items: {details}")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                archive.write(artifacts[name].source, f"{self._root_folder}/{name}")
            archive.writestr(f"{self._root_folder}/{MANIFEST_NAME}", _manifest(names, artifacts))
        return PackageResult(zip_path, names, missing)


def _manifest(names: Sequence[str], artifacts: Mapping[str, Artifact]) -> str:
    lines = [f"{artifacts[name].source.stat().st_size:>12,}  {name}" for name in names]
    return "\n".join(["       bytes  file", *lines, ""])
