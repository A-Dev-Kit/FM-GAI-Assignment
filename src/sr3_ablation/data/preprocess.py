# ADDED: new file, not part of the upstream SR3 codebase.
"""Create the SR3 ``img`` layout (``lr_16``, ``hr_128``, ``sr_16_128``) for each split.  # ADDED

The resizing protocol follows the Phase 1 declaration: centre-crop a square, resize to  # ADDED
128×128 for the target, shrink to 16×16 for the input, then upsample the input back to  # ADDED
128×128 for the conditioning image SR3 expects.  # ADDED
"""  # ADDED

from __future__ import annotations  # ADDED

from collections.abc import Sequence  # ADDED
from concurrent.futures import ThreadPoolExecutor  # ADDED
from dataclasses import dataclass  # ADDED
from pathlib import Path  # ADDED

from PIL import Image  # ADDED
from tqdm import tqdm  # ADDED

from sr3_ablation.data.kvasir_source import DATASET_SLUG  # ADDED
from sr3_ablation.data.manifest import SPLITS, ManifestEntry  # ADDED
from sr3_ablation.utils.selection import evenly_spaced_indices  # ADDED

_BICUBIC = Image.Resampling.BICUBIC  # ADDED


@dataclass(frozen=True)  # ADDED
class SrLayout:  # ADDED
    """Folder layout of one processed split, named exactly as upstream SR3 expects."""  # ADDED

    root: Path  # ADDED
    low: int  # ADDED
    high: int  # ADDED

    @property  # ADDED
    def hr_dir(self) -> Path:  # ADDED
        return self.root / f"hr_{self.high}"  # ADDED

    @property  # ADDED
    def lr_dir(self) -> Path:  # ADDED
        return self.root / f"lr_{self.low}"  # ADDED

    @property  # ADDED
    def sr_dir(self) -> Path:  # ADDED
        return self.root / f"sr_{self.low}_{self.high}"  # ADDED

    def hr_path(self, image_id: str) -> Path:  # ADDED
        return self.hr_dir / f"{image_id}.png"  # ADDED

    def lr_path(self, image_id: str) -> Path:  # ADDED
        return self.lr_dir / f"{image_id}.png"  # ADDED

    def sr_path(self, image_id: str) -> Path:  # ADDED
        return self.sr_dir / f"{image_id}.png"  # ADDED

    def image_ids(self, limit: int = 0) -> list[str]:  # ADDED
        """Sorted image ids; a positive ``limit`` keeps an evenly spaced subset."""  # ADDED
        image_ids = sorted(path.stem for path in self.hr_dir.glob("*.png"))  # ADDED
        return [image_ids[i] for i in evenly_spaced_indices(len(image_ids), limit)]  # ADDED


def processed_root(data_root: Path, low: int, high: int) -> Path:  # ADDED
    """Folder holding every split of the processed LR/HR dataset."""  # ADDED
    return data_root / "processed" / f"{DATASET_SLUG}_{low}_{high}"  # ADDED


def layout_for(data_root: Path, split: str, low: int, high: int) -> SrLayout:  # ADDED
    """SR3 folder layout of one split; an unknown split is a ``ValueError``."""  # ADDED
    if split not in SPLITS:  # ADDED
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}.")  # ADDED
    return SrLayout(processed_root(data_root, low, high) / split, low, high)  # ADDED


def resize_and_crop(image: Image.Image, size: int) -> Image.Image:  # ADDED
    """Resize the shorter side to ``size`` (bicubic) and centre-crop to a square."""  # ADDED
    width, height = image.size  # ADDED
    scale = size / min(width, height)  # ADDED
    resized = image.resize(  # ADDED
        (max(size, round(width * scale)), max(size, round(height * scale))),  # ADDED
        _BICUBIC,  # ADDED
    )  # ADDED
    left = (resized.width - size) // 2  # ADDED
    top = (resized.height - size) // 2  # ADDED
    return resized.crop((left, top, left + size, top + size))  # ADDED


def centre_square(image: Image.Image) -> Image.Image:  # ADDED
    """Centre-crop to the largest square, as in the Phase 1 declaration preprocessing."""  # ADDED
    width, height = image.size  # ADDED
    side = min(width, height)  # ADDED
    left = (width - side) // 2  # ADDED
    top = (height - side) // 2  # ADDED
    return image.crop((left, top, left + side, top + side))  # ADDED


def make_triplet(image: Image.Image, low: int, high: int) -> tuple[Image.Image, ...]:  # ADDED
    """Return ``(lr, hr, sr)`` for one RGB image.  # ADDED

    Ground truth is a centre square resized to ``high``; the input is that image shrunk to  # ADDED
    ``low`` and the conditioning image is the input upsampled back to ``high`` (declaration).  # ADDED
    """  # ADDED
    square = centre_square(image.convert("RGB"))  # ADDED
    hr = square.resize((high, high), _BICUBIC)  # ADDED
    lr = hr.resize((low, low), _BICUBIC)  # ADDED
    sr = lr.resize((high, high), _BICUBIC)  # ADDED
    return lr, hr, sr  # ADDED


def prepare_split(  # ADDED
    entries: Sequence[ManifestEntry],  # ADDED
    image_root: Path,  # ADDED
    layout: SrLayout,  # ADDED
    workers: int = 8,  # ADDED
) -> int:  # ADDED
    """Write the processed triplets of ``entries``; existing files are kept. Returns count."""  # ADDED
    for folder in (layout.hr_dir, layout.lr_dir, layout.sr_dir):  # ADDED
        folder.mkdir(parents=True, exist_ok=True)  # ADDED

    def process(entry: ManifestEntry) -> None:  # ADDED
        targets = (  # ADDED
            layout.lr_path(entry.image_id),  # ADDED
            layout.hr_path(entry.image_id),  # ADDED
            layout.sr_path(entry.image_id),  # ADDED
        )  # ADDED
        if all(target.is_file() for target in targets):  # ADDED
            return  # ADDED
        with Image.open(image_root / entry.source) as handle:  # ADDED
            triplet = make_triplet(handle.convert("RGB"), layout.low, layout.high)  # ADDED
        for image, target in zip(triplet, targets, strict=True):  # ADDED
            image.save(target)  # ADDED

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:  # ADDED
        list(tqdm(pool.map(process, entries), total=len(entries), desc=layout.root.name))  # ADDED
    return len(entries)  # ADDED
