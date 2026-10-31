# ADDED: new file, not part of the upstream SR3 codebase.
"""Seeded train / validation / test split recorded as a CSV manifest."""  # ADDED

from __future__ import annotations  # ADDED

import csv  # ADDED
import random  # ADDED
from collections import Counter  # ADDED
from collections.abc import Iterable, Sequence  # ADDED
from dataclasses import astuple, dataclass, fields  # ADDED
from pathlib import Path  # ADDED

SPLITS = ("train", "val", "test")  # ADDED


@dataclass(frozen=True)  # ADDED
class ManifestEntry:  # ADDED
    """One image of the experiment.  # ADDED

    Attributes:  # ADDED
        image_id: Stable identifier used for processed file names, e.g. ``test_00042``.  # ADDED
        split: ``train``, ``val`` or ``test``.  # ADDED
        source: Path of the original image relative to the image folder (POSIX style).  # ADDED
    """  # ADDED

    image_id: str  # ADDED
    split: str  # ADDED
    source: str  # ADDED


def build_manifest(  # ADDED
    sources: Sequence[str],  # ADDED
    val_size: int,  # ADDED
    test_size: int,  # ADDED
    seed: int,  # ADDED
) -> list[ManifestEntry]:  # ADDED
    """Shuffle ``sources`` with ``seed``, then assign contiguous blocks: training first,  # ADDED
    validation next and test last (same cut as Rikin / Sonalika: 800 / 100 / 100 on Kvasir)."""  # ADDED
    if val_size + test_size >= len(sources):  # ADDED
        raise ValueError(  # ADDED
            f"val_size + test_size = {val_size + test_size} leaves no training images "  # ADDED
            f"out of {len(sources)}."  # ADDED
        )  # ADDED
    order = sorted(sources)  # ADDED
    random.Random(seed).shuffle(order)  # ADDED
    train_count = len(order) - val_size - test_size  # ADDED
    train = order[:train_count]  # ADDED
    val = order[train_count : train_count + val_size]  # ADDED
    test = order[train_count + val_size :]  # ADDED
    return [*_entries("train", train), *_entries("val", val), *_entries("test", test)]  # ADDED


def split_counts(entries: Iterable[ManifestEntry]) -> dict[str, int]:  # ADDED
    """Number of entries per split, in ``SPLITS`` order."""  # ADDED
    counts = Counter(entry.split for entry in entries)  # ADDED
    return {split: counts[split] for split in SPLITS}  # ADDED


def write_manifest(entries: Sequence[ManifestEntry], path: Path) -> None:  # ADDED
    """Write one CSV row per entry, with the field names as header."""  # ADDED
    path.parent.mkdir(parents=True, exist_ok=True)  # ADDED
    with path.open("w", newline="", encoding="utf-8") as handle:  # ADDED
        writer = csv.writer(handle)  # ADDED
        writer.writerow([field.name for field in fields(ManifestEntry)])  # ADDED
        writer.writerows(astuple(entry) for entry in entries)  # ADDED


def read_manifest(path: Path) -> list[ManifestEntry]:  # ADDED
    """Read a manifest written by :func:`write_manifest`."""  # ADDED
    with path.open(newline="", encoding="utf-8") as handle:  # ADDED
        return [ManifestEntry(**row) for row in csv.DictReader(handle)]  # ADDED


def _entries(split: str, sources: Iterable[str]) -> list[ManifestEntry]:  # ADDED
    return [  # ADDED
        ManifestEntry(image_id=f"{split}_{index:05d}", split=split, source=source)  # ADDED
        for index, source in enumerate(sorted(sources))  # ADDED
    ]  # ADDED
