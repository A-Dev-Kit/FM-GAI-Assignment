# ADDED: new file, not part of the upstream SR3 codebase.
"""What goes into the submission zip: artifacts, the selector interface and exclusion rules."""  # ADDED

from __future__ import annotations  # ADDED

from collections.abc import Iterable  # ADDED
from dataclasses import dataclass  # ADDED
from pathlib import Path, PurePosixPath  # ADDED
from typing import Protocol  # ADDED


@dataclass(frozen=True)  # ADDED
class Artifact:  # ADDED
    """A file to archive.  # ADDED

    Attributes:  # ADDED
        source: Where the file is on disk.  # ADDED
        archive_name: POSIX path inside the zip, relative to the package root folder.  # ADDED
    """  # ADDED

    source: Path  # ADDED
    archive_name: str  # ADDED


class ArtifactSelector(Protocol):  # ADDED
    """Chooses one group of files for the zip, e.g. the code or one run's outputs."""  # ADDED

    def select(self) -> Iterable[Artifact]:  # ADDED
        """Yield the artifacts of this group; missing optional files are simply skipped."""  # ADDED
        ...  # ADDED


@dataclass(frozen=True)  # ADDED
class ExclusionRule:  # ADDED
    """Files that must never be archived, whichever selector proposes them.  # ADDED

    Attributes:  # ADDED
        suffixes: Lower-case file suffixes, e.g. ``.pth`` checkpoints or ``.jpg`` dataset frames.  # ADDED
        folder_names: Folder names excluded wherever they appear, e.g. ``__pycache__``.  # ADDED
        top_level_folders: First path components excluded, e.g. ``data`` (but not ``src/.../data``).  # ADDED
        file_names: Exact file names, e.g. the machine-specific ``local.toml``.  # ADDED
    """  # ADDED

    suffixes: frozenset[str]  # ADDED
    folder_names: frozenset[str]  # ADDED
    top_level_folders: frozenset[str]  # ADDED
    file_names: frozenset[str]  # ADDED

    def excludes(self, archive_name: str) -> bool:  # ADDED
        """Whether ``archive_name`` (POSIX, relative) matches any rule."""  # ADDED
        path = PurePosixPath(archive_name)  # ADDED
        folders = path.parts[:-1]  # ADDED
        return (  # ADDED
            path.suffix.lower() in self.suffixes  # ADDED
            or path.name in self.file_names  # ADDED
            or bool(folders and folders[0] in self.top_level_folders)  # ADDED
            or any(folder in self.folder_names for folder in folders)  # ADDED
        )  # ADDED


DEFAULT_EXCLUSIONS = ExclusionRule(  # ADDED
    suffixes=frozenset({".pth", ".pt", ".ckpt", ".jpg", ".jpeg", ".zip", ".pyc"}),  # ADDED
    folder_names=frozenset({"__pycache__", "checkpoints", ".pytest_cache", ".ruff_cache"}),  # ADDED
    top_level_folders=frozenset({"data", "dist", ".venv", ".git"}),  # ADDED
    file_names=frozenset({"local.toml"}),  # ADDED
)  # ADDED
