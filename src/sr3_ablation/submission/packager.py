# ADDED: new file, not part of the upstream SR3 codebase.
"""Write the submission zip from a set of selectors and check it covers the brief."""  # ADDED

from __future__ import annotations  # ADDED

import fnmatch  # ADDED
import zipfile  # ADDED
from collections.abc import Mapping, Sequence  # ADDED
from dataclasses import dataclass  # ADDED
from pathlib import Path  # ADDED

from sr3_ablation.submission.artifacts import (  # ADDED
    DEFAULT_EXCLUSIONS,  # ADDED
    Artifact,  # ADDED
    ArtifactSelector,  # ADDED
    ExclusionRule,  # ADDED
)  # ADDED

MANIFEST_NAME = "MANIFEST.txt"  # ADDED


@dataclass(frozen=True)  # ADDED
class Requirement:  # ADDED
    """At least ``minimum`` archived files must match the glob ``pattern``."""  # ADDED

    description: str  # ADDED
    pattern: str  # ADDED
    minimum: int = 1  # ADDED

    def count(self, names: Sequence[str]) -> int:  # ADDED
        return sum(1 for name in names if fnmatch.fnmatch(name, self.pattern))  # ADDED

    def is_met(self, names: Sequence[str]) -> bool:  # ADDED
        return self.count(names) >= self.minimum  # ADDED


@dataclass(frozen=True)  # ADDED
class PackageResult:  # ADDED
    """The zip written and what it holds."""  # ADDED

    path: Path  # ADDED
    names: tuple[str, ...]  # ADDED
    missing: tuple[Requirement, ...]  # ADDED

    @property  # ADDED
    def complete(self) -> bool:  # ADDED
        return not self.missing  # ADDED

    def to_dict(self) -> dict[str, object]:  # ADDED
        return {  # ADDED
            "zip": str(self.path),  # ADDED
            "files": len(self.names),  # ADDED
            "megabytes": round(self.path.stat().st_size / 2**20, 2),  # ADDED
            "missing": [f"{r.description} ({r.pattern}, need {r.minimum})" for r in self.missing],  # ADDED
        }  # ADDED


class IncompleteSubmissionError(RuntimeError):  # ADDED
    """Raised when brief items are missing and an incomplete zip was not allowed."""  # ADDED


class SubmissionPackager:  # ADDED
    """Collects artifacts from every selector, filters them and writes one zip.  # ADDED

    Every archive name is placed under ``root_folder`` so unzipping creates one folder.  # ADDED
    """  # ADDED

    def __init__(  # ADDED
        self,  # ADDED
        selectors: Sequence[ArtifactSelector],  # ADDED
        requirements: Sequence[Requirement],  # ADDED
        root_folder: str,  # ADDED
        exclusions: ExclusionRule = DEFAULT_EXCLUSIONS,  # ADDED
    ) -> None:  # ADDED
        self._selectors = tuple(selectors)  # ADDED
        self._requirements = tuple(requirements)  # ADDED
        self._root_folder = root_folder  # ADDED
        self._exclusions = exclusions  # ADDED

    def build(self, zip_path: Path, allow_incomplete: bool = False) -> PackageResult:  # ADDED
        """Write ``zip_path``; refuse (before writing) if requirements fail, unless allowed."""  # ADDED
        artifacts: dict[str, Artifact] = {}  # ADDED
        for selector in self._selectors:  # ADDED
            for artifact in selector.select():  # ADDED
                if self._exclusions.excludes(artifact.archive_name):  # ADDED
                    continue  # ADDED
                previous = artifacts.setdefault(artifact.archive_name, artifact)  # ADDED
                if previous.source != artifact.source:  # ADDED
                    raise ValueError(f"Two files claim {artifact.archive_name}")  # ADDED
        names = tuple(sorted(artifacts))  # ADDED
        missing = tuple(r for r in self._requirements if not r.is_met(names))  # ADDED
        if missing and not allow_incomplete:  # ADDED
            details = "; ".join(f"{r.description} ({r.pattern})" for r in missing)  # ADDED
            raise IncompleteSubmissionError(f"Missing brief items: {details}")  # ADDED
        zip_path.parent.mkdir(parents=True, exist_ok=True)  # ADDED
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:  # ADDED
            for name in names:  # ADDED
                archive.write(artifacts[name].source, f"{self._root_folder}/{name}")  # ADDED
            archive.writestr(f"{self._root_folder}/{MANIFEST_NAME}", _manifest(names, artifacts))  # ADDED
        return PackageResult(zip_path, names, missing)  # ADDED


def _manifest(names: Sequence[str], artifacts: Mapping[str, Artifact]) -> str:  # ADDED
    lines = [f"{artifacts[name].source.stat().st_size:>12,}  {name}" for name in names]  # ADDED
    return "\n".join(["       bytes  file", *lines, ""])  # ADDED
