# ADDED: new file, not part of the upstream SR3 codebase.
"""What goes into the submission zip: artifacts, the selector interface and exclusion rules."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol


@dataclass(frozen=True)
class Artifact:
    """A file to archive.

    Attributes:
        source: Where the file is on disk.
        archive_name: POSIX path inside the zip, relative to the package root folder.
    """

    source: Path
    archive_name: str


class ArtifactSelector(Protocol):
    """Chooses one group of files for the zip, e.g. the code or one run's outputs."""

    def select(self) -> Iterable[Artifact]:
        """Yield the artifacts of this group; missing optional files are simply skipped."""
        ...


@dataclass(frozen=True)
class ExclusionRule:
    """Files that must never be archived, whichever selector proposes them.

    Attributes:
        suffixes: Lower-case file suffixes, e.g. ``.pth`` checkpoints or ``.jpg`` dataset frames.
        folder_names: Folder names excluded wherever they appear, e.g. ``__pycache__``.
        top_level_folders: First path components excluded, e.g. ``data`` (but not ``src/.../data``).
        file_names: Exact file names, e.g. the machine-specific ``local.toml``.
    """

    suffixes: frozenset[str]
    folder_names: frozenset[str]
    top_level_folders: frozenset[str]
    file_names: frozenset[str]

    def excludes(self, archive_name: str) -> bool:
        """Whether ``archive_name`` (POSIX, relative) matches any rule."""
        path = PurePosixPath(archive_name)
        folders = path.parts[:-1]
        return (
            path.suffix.lower() in self.suffixes
            or path.name in self.file_names
            or bool(folders and folders[0] in self.top_level_folders)
            or any(folder in self.folder_names for folder in folders)
        )


DEFAULT_EXCLUSIONS = ExclusionRule(
    suffixes=frozenset({".pth", ".pt", ".ckpt", ".jpg", ".jpeg", ".zip", ".pyc"}),
    folder_names=frozenset({"__pycache__", "checkpoints", ".pytest_cache", ".ruff_cache"}),
    top_level_folders=frozenset({"data", "dist", ".venv", ".git"}),
    file_names=frozenset({"local.toml"}),
)
