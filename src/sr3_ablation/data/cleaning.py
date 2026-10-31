# ADDED: new file, not part of the upstream SR3 codebase.
"""Data cleaning: drop byte-identical copies so no image can sit in two splits."""  # ADDED

from __future__ import annotations  # ADDED

import hashlib  # ADDED
from collections.abc import Sequence  # ADDED
from pathlib import Path  # ADDED
from typing import NamedTuple  # ADDED


class Deduplicated(NamedTuple):  # ADDED
    """Result of :func:`drop_exact_duplicates`."""  # ADDED

    unique: list[Path]  # ADDED
    duplicates: list[Path]  # ADDED


def drop_exact_duplicates(paths: Sequence[Path]) -> Deduplicated:  # ADDED
    """Keep the first file of every SHA-1 digest, in the given order."""  # ADDED
    seen: set[str] = set()  # ADDED
    unique: list[Path] = []  # ADDED
    duplicates: list[Path] = []  # ADDED
    for path in paths:  # ADDED
        digest = hashlib.sha1(path.read_bytes()).hexdigest()  # ADDED
        (duplicates if digest in seen else unique).append(path)  # ADDED
        seen.add(digest)  # ADDED
    return Deduplicated(unique, duplicates)  # ADDED
