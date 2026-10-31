# ADDED: new file, not part of the upstream SR3 codebase.
"""Builders shared by the tests: a tiny CPU configuration and a synthetic Kvasir-SEG tree."""  # ADDED

from __future__ import annotations  # ADDED

import shutil  # ADDED
from pathlib import Path  # ADDED
from typing import Any  # ADDED

import numpy as np  # ADDED
from PIL import Image  # ADDED

from sr3_ablation.data.archive import EXTRACTED_MARKER  # ADDED

DECLARED_EXPONENT = 2.50  # ADDED
"""Exponent x from the Phase 1 declaration; must equal ``NOISE_EXPONENT_X``."""  # ADDED


def tiny_raw_config(root: Path) -> dict[str, Any]:  # ADDED
    """Configuration small enough to train and sample on a CPU in seconds."""  # ADDED
    return {  # ADDED
        "experiment": {"name": "tiny", "device": "cpu"},  # ADDED
        "paths": {  # ADDED
            "data_root": str(root / "data"),  # ADDED
            "output_root": str(root / "outputs"),  # ADDED
            "checkpoint_dir": str(root / "checkpoints"),  # ADDED
        },  # ADDED
        "data": {  # ADDED
            "source_url": "file://unused",  # ADDED
            "split_seed": 42,  # ADDED
            "val_size": 3,  # ADDED
            "test_size": 3,  # ADDED
            "low_resolution": 4,  # ADDED
            "high_resolution": 16,  # ADDED
            "train_limit": 0,  # ADDED
            "num_workers": 0,  # ADDED
        },  # ADDED
        "schedule": {  # ADDED
            "schedule": "linear",  # ADDED
            "n_timestep": 10,  # ADDED
            "linear_start": 1e-6,  # ADDED
            "linear_end": 1e-2,  # ADDED
        },  # ADDED
        "model": {  # ADDED
            "pretrained_folder_url": "https://example.invalid",  # ADDED
            "pretrained_file": "pretrained_gen.pth",  # ADDED
            "image_size": 16,  # ADDED
            "channels": 3,  # ADDED
            "unet": {  # ADDED
                "in_channel": 6,  # ADDED
                "out_channel": 3,  # ADDED
                "inner_channel": 8,  # ADDED
                "norm_groups": 4,  # ADDED
                "channel_multiplier": [1, 2],  # ADDED
                "attn_res": [8],  # ADDED
                "res_blocks": 1,  # ADDED
                "dropout": 0.0,  # ADDED
            },  # ADDED
        },  # ADDED
        "train": {  # ADDED
            "seed": 42,  # ADDED
            "batch_size": 2,  # ADDED
            "grad_accum_steps": 1,  # ADDED
            "learning_rate": 1e-4,  # ADDED
            "epochs": 2,  # ADDED
            "checkpoint_epochs": [1, 2],  # ADDED
            "max_iterations": 0,  # ADDED
            "log_every": 1,  # ADDED
        },  # ADDED
        "sampling": {  # ADDED
            "seed": 7,  # ADDED
            "batch_size": 2,  # ADDED
            "snapshot_samples": 1,  # ADDED
            "validation_images": 2,  # ADDED
            "trajectory_samples": 1,  # ADDED
            "trajectory_timesteps": [8, 4, 0],  # ADDED
        },  # ADDED
        "evaluation": {"metrics": ["psnr", "ssim"], "test_limit": 0},  # ADDED
    }  # ADDED


FAKE_DUPLICATE = "zz_copy_of_frame_000.jpg"  # ADDED


def make_fake_kvasir(raw_dir: Path, frames: int = 15) -> Path:  # ADDED
    """Write ``raw_dir/Kvasir-SEG/{images,masks}/*.jpg`` like the real archive, plus the marker.  # ADDED

    ``images/`` holds ``frames`` distinct non-square frames and one byte-identical copy of the  # ADDED
    first (``FAKE_DUPLICATE``), so cleaning has something to remove. Returns ``images/``.  # ADDED
    """  # ADDED
    rng = np.random.default_rng(0)  # ADDED
    images = raw_dir / "Kvasir-SEG" / "images"  # ADDED
    masks = raw_dir / "Kvasir-SEG" / "masks"  # ADDED
    images.mkdir(parents=True)  # ADDED
    masks.mkdir(parents=True)  # ADDED
    for index in range(frames):  # ADDED
        pixels = rng.integers(0, 256, size=(32, 40, 3), dtype=np.uint8)  # ADDED
        Image.fromarray(pixels).save(images / f"frame_{index:03d}.jpg")  # ADDED
        Image.new("L", (40, 32), 255).save(masks / f"frame_{index:03d}.jpg")  # ADDED
    shutil.copyfile(images / "frame_000.jpg", images / FAKE_DUPLICATE)  # ADDED
    (raw_dir / EXTRACTED_MARKER).write_text("test", encoding="utf-8")  # ADDED
    return images  # ADDED
