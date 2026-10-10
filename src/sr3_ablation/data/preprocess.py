# ADDED: new file, not part of the upstream SR3 codebase.
"""Create the SR3 ``img`` layout (``lr_16``, ``hr_128``, ``sr_16_128``) for each split.

The resizing protocol mirrors upstream ``data/prepare_data.py`` (``resize_multiple``): the LR
and HR images are both bicubic resizes (with centre crop) of the original image, and the
conditioning image ``sr`` is the LR image bicubically upsampled back to HR size.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from sr3_ablation.data.kvasir_source import DATASET_SLUG
from sr3_ablation.data.manifest import SPLITS, ManifestEntry
from sr3_ablation.utils.selection import evenly_spaced_indices

_BICUBIC = Image.Resampling.BICUBIC


@dataclass(frozen=True)
class SrLayout:
    """Folder layout of one processed split, named exactly as upstream SR3 expects."""

    root: Path
    low: int
    high: int

    @property
    def hr_dir(self) -> Path:
        return self.root / f"hr_{self.high}"

    @property
    def lr_dir(self) -> Path:
        return self.root / f"lr_{self.low}"

    @property
    def sr_dir(self) -> Path:
        return self.root / f"sr_{self.low}_{self.high}"

    def hr_path(self, image_id: str) -> Path:
        return self.hr_dir / f"{image_id}.png"

    def lr_path(self, image_id: str) -> Path:
        return self.lr_dir / f"{image_id}.png"

    def sr_path(self, image_id: str) -> Path:
        return self.sr_dir / f"{image_id}.png"

    def image_ids(self, limit: int = 0) -> list[str]:
        """Sorted image ids; a positive ``limit`` keeps an evenly spaced subset."""
        image_ids = sorted(path.stem for path in self.hr_dir.glob("*.png"))
        return [image_ids[i] for i in evenly_spaced_indices(len(image_ids), limit)]


def processed_root(data_root: Path, low: int, high: int) -> Path:
    return data_root / "processed" / f"{DATASET_SLUG}_{low}_{high}"


def layout_for(data_root: Path, split: str, low: int, high: int) -> SrLayout:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}.")
    return SrLayout(processed_root(data_root, low, high) / split, low, high)


def resize_and_crop(image: Image.Image, size: int) -> Image.Image:
    """Resize the shorter side to ``size`` (bicubic) and centre-crop to a square."""
    width, height = image.size
    scale = size / min(width, height)
    resized = image.resize(
        (max(size, round(width * scale)), max(size, round(height * scale))), _BICUBIC
    )
    left = (resized.width - size) // 2
    top = (resized.height - size) // 2
    return resized.crop((left, top, left + size, top + size))


def make_triplet(image: Image.Image, low: int, high: int) -> tuple[Image.Image, ...]:
    """Return ``(lr, hr, sr)`` for one RGB image."""
    lr = resize_and_crop(image, low)
    hr = resize_and_crop(image, high)
    sr = resize_and_crop(lr, high)
    return lr, hr, sr


def prepare_split(
    entries: Sequence[ManifestEntry], image_root: Path, layout: SrLayout, workers: int = 8
) -> int:
    """Write the processed triplets of ``entries``; existing files are kept. Returns count."""
    for folder in (layout.hr_dir, layout.lr_dir, layout.sr_dir):
        folder.mkdir(parents=True, exist_ok=True)

    def process(entry: ManifestEntry) -> None:
        targets = (
            layout.lr_path(entry.image_id),
            layout.hr_path(entry.image_id),
            layout.sr_path(entry.image_id),
        )
        if all(target.is_file() for target in targets):
            return
        with Image.open(image_root / entry.source) as handle:
            triplet = make_triplet(handle.convert("RGB"), layout.low, layout.high)
        for image, target in zip(triplet, targets, strict=True):
            image.save(target)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(tqdm(pool.map(process, entries), total=len(entries), desc=layout.root.name))
    return len(entries)
