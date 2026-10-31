# ADDED: new file, not part of the upstream SR3 codebase.
"""Download the pretrained SR3 checkpoint and read / write denoiser-only weights.  # ADDED

Only the UNet weights (``denoise_fn``) are ever loaded. Upstream ``DDPM.load_network`` would  # ADDED
also restore the optimiser, the step counter and the noise-schedule buffers stored in the  # ADDED
checkpoint; none of those may leak into the ablation runs.  # ADDED
"""  # ADDED

from __future__ import annotations  # ADDED

from collections.abc import Mapping  # ADDED
from pathlib import Path  # ADDED
from typing import Any  # ADDED

import torch  # ADDED

from sr3_ablation.model.protocols import StateDict  # ADDED
from sr3_ablation.utils.log import get_logger  # ADDED

DENOISER_PREFIX = "denoise_fn."  # ADDED
PARALLEL_PREFIX = "module."  # ADDED
CHECKPOINT_FORMAT = "sr3_ablation/denoiser-v1"  # ADDED

logger = get_logger("model.checkpoints")  # ADDED


def download_pretrained(folder_url: str, filename: str, destination_dir: Path) -> Path:  # ADDED
    """Fetch ``filename`` from the public Google Drive folder (skipped if already present)."""  # ADDED
    target = destination_dir / filename  # ADDED
    if target.is_file():  # ADDED
        logger.info("Pretrained checkpoint already at %s", target)  # ADDED
        return target  # ADDED
    import gdown  # ADDED

    destination_dir.mkdir(parents=True, exist_ok=True)  # ADDED
    listing = gdown.download_folder(url=folder_url, skip_download=True, quiet=True)  # ADDED
    match = next((item for item in listing or [] if Path(item.path).name == filename), None)  # ADDED
    if match is None:  # ADDED
        raise FileNotFoundError(f"{filename} not found in Google Drive folder {folder_url}.")  # ADDED
    logger.info("Downloading %s (Drive id %s)", filename, match.id)  # ADDED
    gdown.download(id=match.id, output=str(target), quiet=False)  # ADDED
    return target  # ADDED


def read_denoiser_state(path: Path) -> StateDict:  # ADDED
    """Load UNet weights from an upstream ``*_gen.pth`` file or one written by this package."""  # ADDED
    if not path.is_file():  # ADDED
        raise FileNotFoundError(f"Checkpoint not found: {path}")  # ADDED
    payload = torch.load(path, map_location="cpu", weights_only=True)  # ADDED
    if isinstance(payload, Mapping) and payload.get("format") == CHECKPOINT_FORMAT:  # ADDED
        return dict(payload["denoiser"])  # ADDED
    return extract_denoiser_state(payload)  # ADDED


def extract_denoiser_state(full_state: Mapping[str, torch.Tensor]) -> StateDict:  # ADDED
    """Keep only ``denoise_fn.*`` entries of a full ``GaussianDiffusion`` state dict."""  # ADDED
    denoiser = {}  # ADDED
    for name, tensor in full_state.items():  # ADDED
        key = name.removeprefix(PARALLEL_PREFIX)  # ADDED
        if key.startswith(DENOISER_PREFIX):  # ADDED
            denoiser[key.removeprefix(DENOISER_PREFIX)] = tensor  # ADDED
    if not denoiser:  # ADDED
        raise KeyError("Checkpoint contains no 'denoise_fn.*' weights.")  # ADDED
    return denoiser  # ADDED


def save_denoiser_state(state: StateDict, path: Path, metadata: Mapping[str, Any]) -> None:  # ADDED
    """Save UNet weights and metadata atomically (write ``.tmp``, then rename)."""  # ADDED
    path.parent.mkdir(parents=True, exist_ok=True)  # ADDED
    payload = {"format": CHECKPOINT_FORMAT, "denoiser": state, "metadata": dict(metadata)}  # ADDED
    temporary = path.with_suffix(path.suffix + ".tmp")  # ADDED
    torch.save(payload, temporary)  # ADDED
    temporary.replace(path)  # ADDED
