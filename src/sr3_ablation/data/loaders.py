# ADDED: new file, not part of the upstream SR3 codebase.
"""PyTorch dataset and loader factory over the processed SR3 layout.  # ADDED

Each item is the dictionary SR3's ``GaussianDiffusion.p_losses`` expects: ``HR`` (target) and  # ADDED
``SR`` (bicubic-upsampled conditioning image), both ``(3, H, W)`` in [-1, 1], plus ``Index``  # ADDED
and ``image_id``. Training items are flipped horizontally with probability 0.5, applied  # ADDED
jointly to ``HR`` and ``SR`` as in upstream ``LRHRDataset``.  # ADDED
"""  # ADDED

from __future__ import annotations  # ADDED

from typing import Any  # ADDED

import torch  # ADDED
from torch.utils.data import DataLoader, Dataset  # ADDED

from sr3_ablation.data.preprocess import SrLayout  # ADDED
from sr3_ablation.io.images import load_image, uint8_to_tensor  # ADDED


class PairedSrDataset(Dataset):  # ADDED
    """Processed HR / bicubic pairs of one split; ``limit`` keeps an evenly spaced subset."""  # ADDED

    def __init__(self, layout: SrLayout, augment: bool = False, limit: int = 0) -> None:  # ADDED
        self._image_ids = layout.image_ids(limit)  # ADDED
        if not self._image_ids:  # ADDED
            raise FileNotFoundError(f"No processed images in {layout.hr_dir}; run prepare-data.")  # ADDED
        self._layout = layout  # ADDED
        self._augment = augment  # ADDED

    def __len__(self) -> int:  # ADDED
        return len(self._image_ids)  # ADDED

    @property  # ADDED
    def image_ids(self) -> list[str]:  # ADDED
        return list(self._image_ids)  # ADDED

    def __getitem__(self, index: int) -> dict[str, Any]:  # ADDED
        image_id = self._image_ids[index]  # ADDED
        hr = uint8_to_tensor(load_image(self._layout.hr_path(image_id)))  # ADDED
        sr = uint8_to_tensor(load_image(self._layout.sr_path(image_id)))  # ADDED
        if self._augment and torch.rand(()) < 0.5:  # ADDED
            hr, sr = hr.flip(-1), sr.flip(-1)  # ADDED
        return {"HR": hr, "SR": sr, "Index": index, "image_id": image_id}  # ADDED


def make_loader(  # ADDED
    dataset: Dataset,  # ADDED
    batch_size: int,  # ADDED
    *,  # ADDED
    shuffle: bool,  # ADDED
    num_workers: int,  # ADDED
    seed: int,  # ADDED
) -> DataLoader:  # ADDED
    """Build a loader whose shuffling order is fixed by ``seed``."""  # ADDED
    generator = torch.Generator()  # ADDED
    generator.manual_seed(seed)  # ADDED
    return DataLoader(  # ADDED
        dataset,  # ADDED
        batch_size=batch_size,  # ADDED
        shuffle=shuffle,  # ADDED
        num_workers=num_workers,  # ADDED
        pin_memory=torch.cuda.is_available(),  # ADDED
        persistent_workers=num_workers > 0,  # ADDED
        generator=generator,  # ADDED
    )  # ADDED
