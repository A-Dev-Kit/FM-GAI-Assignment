# ADDED: new file, not part of the upstream SR3 codebase.
"""Seeded train / validation / test split recorded as a CSV manifest."""

from __future__ import annotations

import csv
import random
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import astuple, dataclass, fields
from pathlib import Path

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ManifestEntry:
    """One image of the experiment.

    Attributes:
        image_id: Stable identifier used for processed file names, e.g. ``test_00042``.
        split: ``train``, ``val`` or ``test``.
        source: Path of the original image relative to the image folder (POSIX style).
    """

    image_id: str
    split: str
    source: str


def build_manifest(
    sources: Sequence[str], val_size: int, test_size: int, seed: int
) -> list[ManifestEntry]:
    """Shuffle ``sources`` with ``seed``; the first ``test_size`` go to test, the next
    ``val_size`` to validation and the rest to training. Each split is listed in name order."""
    if val_size + test_size >= len(sources):
        raise ValueError(
            f"val_size + test_size = {val_size + test_size} leaves no training images "
            f"out of {len(sources)}."
        )
    order = sorted(sources)
    random.Random(seed).shuffle(order)
    test = order[:test_size]
    val = order[test_size : test_size + val_size]
    train = order[test_size + val_size :]
    return [*_entries("train", train), *_entries("val", val), *_entries("test", test)]


def split_counts(entries: Iterable[ManifestEntry]) -> dict[str, int]:
    counts = Counter(entry.split for entry in entries)
    return {split: counts[split] for split in SPLITS}


def write_manifest(entries: Sequence[ManifestEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([field.name for field in fields(ManifestEntry)])
        writer.writerows(astuple(entry) for entry in entries)


def read_manifest(path: Path) -> list[ManifestEntry]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [ManifestEntry(**row) for row in csv.DictReader(handle)]


def _entries(split: str, sources: Iterable[str]) -> list[ManifestEntry]:
    return [
        ManifestEntry(image_id=f"{split}_{index:05d}", split=split, source=source)
        for index, source in enumerate(sorted(sources))
    ]
