# ADDED: new file, not part of the upstream SR3 codebase.
"""Data cleaning: drop byte-identical copies so no image can sit in two splits."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple


class Deduplicated(NamedTuple):
    unique: list[Path]
    duplicates: list[Path]


def drop_exact_duplicates(paths: Sequence[Path]) -> Deduplicated:
    """Keep the first file of every SHA-1 digest, in the given order."""
    seen: set[str] = set()
    unique: list[Path] = []
    duplicates: list[Path] = []
    for path in paths:
        digest = hashlib.sha1(path.read_bytes()).hexdigest()
        (duplicates if digest in seen else unique).append(path)
        seen.add(digest)
    return Deduplicated(unique, duplicates)
