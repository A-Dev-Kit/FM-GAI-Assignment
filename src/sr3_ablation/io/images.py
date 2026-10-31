# ADDED: new file, not part of the upstream SR3 codebase.
"""Conversions between model tensors in [-1, 1] and 8-bit RGB images."""  # ADDED

from __future__ import annotations  # ADDED

from pathlib import Path  # ADDED

import numpy as np  # ADDED
import torch  # ADDED
from PIL import Image  # ADDED


def tensor_to_uint8(batch: torch.Tensor) -> list[np.ndarray]:  # ADDED
    """Convert a ``(B, C, H, W)`` tensor in [-1, 1] to a list of ``(H, W, C)`` uint8 arrays."""  # ADDED
    if batch.ndim != 4:  # ADDED
        raise ValueError(f"Expected a (B, C, H, W) tensor, got shape {tuple(batch.shape)}.")  # ADDED
    scaled = (batch.detach().float().clamp(-1.0, 1.0) + 1.0) * 127.5  # ADDED
    arrays = scaled.round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()  # ADDED
    return list(arrays)  # ADDED


def uint8_to_tensor(image: np.ndarray) -> torch.Tensor:  # ADDED
    """Convert an ``(H, W, C)`` uint8 array to a ``(C, H, W)`` float tensor in [-1, 1]."""  # ADDED
    tensor = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).permute(2, 0, 1).float()  # ADDED
    return tensor / 127.5 - 1.0  # ADDED


def save_image(image: np.ndarray, path: Path) -> None:  # ADDED
    """Save a ``(H, W, 3)`` uint8 array, creating parent folders."""  # ADDED
    path.parent.mkdir(parents=True, exist_ok=True)  # ADDED
    Image.fromarray(image).save(path)  # ADDED


def load_image(path: Path) -> np.ndarray:  # ADDED
    """Load an image file as a ``(H, W, 3)`` uint8 RGB array."""  # ADDED
    with Image.open(path) as handle:  # ADDED
        return np.asarray(handle.convert("RGB"))  # ADDED
