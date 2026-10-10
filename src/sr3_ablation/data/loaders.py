# ADDED: new file, not part of the upstream SR3 codebase.
"""PyTorch dataset and loader factory over the processed SR3 layout.

Each item is the dictionary SR3's ``GaussianDiffusion.p_losses`` expects: ``HR`` (target) and
``SR`` (bicubic-upsampled conditioning image), both ``(3, H, W)`` in [-1, 1], plus ``Index``
and ``image_id``. Training items are flipped horizontally with probability 0.5, applied
jointly to ``HR`` and ``SR`` as in upstream ``LRHRDataset``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from sr3_ablation.data.preprocess import SrLayout
from sr3_ablation.io.images import load_image, uint8_to_tensor


class PairedSrDataset(Dataset):
    """Processed HR / bicubic pairs of one split; ``limit`` keeps an evenly spaced subset."""

    def __init__(self, layout: SrLayout, augment: bool = False, limit: int = 0) -> None:
        self._image_ids = layout.image_ids(limit)
        if not self._image_ids:
            raise FileNotFoundError(f"No processed images in {layout.hr_dir}; run prepare-data.")
        self._layout = layout
        self._augment = augment

    def __len__(self) -> int:
        return len(self._image_ids)

    @property
    def image_ids(self) -> list[str]:
        return list(self._image_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id = self._image_ids[index]
        hr = uint8_to_tensor(load_image(self._layout.hr_path(image_id)))
        sr = uint8_to_tensor(load_image(self._layout.sr_path(image_id)))
        if self._augment and torch.rand(()) < 0.5:
            hr, sr = hr.flip(-1), sr.flip(-1)
        return {"HR": hr, "SR": sr, "Index": index, "image_id": image_id}


def make_loader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    """Build a loader whose shuffling order is fixed by ``seed``."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )
