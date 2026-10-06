# ADDED: new file, not part of the upstream SR3 codebase.
"""Download the pretrained SR3 checkpoint and read / write denoiser-only weights.

Only the UNet weights (``denoise_fn``) are ever loaded. Upstream ``DDPM.load_network`` would
also restore the optimiser, the step counter and the noise-schedule buffers stored in the
checkpoint; none of those may leak into the ablation runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from sr3_ablation.model.protocols import StateDict
from sr3_ablation.utils.log import get_logger

DENOISER_PREFIX = "denoise_fn."
PARALLEL_PREFIX = "module."
CHECKPOINT_FORMAT = "sr3_ablation/denoiser-v1"

logger = get_logger("model.checkpoints")


def download_pretrained(folder_url: str, filename: str, destination_dir: Path) -> Path:
    """Fetch ``filename`` from the public Google Drive folder (skipped if already present)."""
    target = destination_dir / filename
    if target.is_file():
        logger.info("Pretrained checkpoint already at %s", target)
        return target
    import gdown

    destination_dir.mkdir(parents=True, exist_ok=True)
    listing = gdown.download_folder(url=folder_url, skip_download=True, quiet=True)
    match = next((item for item in listing or [] if Path(item.path).name == filename), None)
    if match is None:
        raise FileNotFoundError(f"{filename} not found in Google Drive folder {folder_url}.")
    logger.info("Downloading %s (Drive id %s)", filename, match.id)
    gdown.download(id=match.id, output=str(target), quiet=False)
    return target


def read_denoiser_state(path: Path) -> StateDict:
    """Load UNet weights from an upstream ``*_gen.pth`` file or one written by this package."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, Mapping) and payload.get("format") == CHECKPOINT_FORMAT:
        return dict(payload["denoiser"])
    return extract_denoiser_state(payload)


def extract_denoiser_state(full_state: Mapping[str, torch.Tensor]) -> StateDict:
    """Keep only ``denoise_fn.*`` entries of a full ``GaussianDiffusion`` state dict."""
    denoiser = {}
    for name, tensor in full_state.items():
        key = name.removeprefix(PARALLEL_PREFIX)
        if key.startswith(DENOISER_PREFIX):
            denoiser[key.removeprefix(DENOISER_PREFIX)] = tensor
    if not denoiser:
        raise KeyError("Checkpoint contains no 'denoise_fn.*' weights.")
    return denoiser


def save_denoiser_state(state: StateDict, path: Path, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": CHECKPOINT_FORMAT, "denoiser": state, "metadata": dict(metadata)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
