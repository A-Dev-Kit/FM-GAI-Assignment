# ADDED: new file, not part of the upstream SR3 codebase.
"""Conversions between model tensors in [-1, 1] and 8-bit RGB images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def tensor_to_uint8(batch: torch.Tensor) -> list[np.ndarray]:
    """Convert a ``(B, C, H, W)`` tensor in [-1, 1] to a list of ``(H, W, C)`` uint8 arrays."""
    if batch.ndim != 4:
        raise ValueError(f"Expected a (B, C, H, W) tensor, got shape {tuple(batch.shape)}.")
    scaled = (batch.detach().float().clamp(-1.0, 1.0) + 1.0) * 127.5
    arrays = scaled.round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return list(arrays)


def uint8_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert an ``(H, W, C)`` uint8 array to a ``(C, H, W)`` float tensor in [-1, 1]."""
    tensor = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).permute(2, 0, 1).float()
    return tensor / 127.5 - 1.0


def save_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"))
