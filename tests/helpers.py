# ADDED: new file, not part of the upstream SR3 codebase.
"""Builders shared by the tests: a tiny CPU configuration and a synthetic AFHQ tree."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sr3_ablation.data.afhq_source import AFHQ_CLASSES, EXTRACTED_MARKER


def tiny_raw_config(root: Path) -> dict[str, Any]:
    """Configuration small enough to train and sample on a CPU in seconds."""
    return {
        "experiment": {"name": "tiny", "device": "cpu"},
        "paths": {
            "data_root": str(root / "data"),
            "output_root": str(root / "outputs"),
            "checkpoint_dir": str(root / "checkpoints"),
        },
        "data": {
            "afhq_url": "file://unused",
            "split_seed": 42,
            "val_size": 3,
            "test_per_class": 1,
            "low_resolution": 4,
            "high_resolution": 16,
            "train_limit": 0,
            "num_workers": 0,
        },
        "schedule": {
            "schedule": "linear",
            "n_timestep": 10,
            "linear_start": 1e-6,
            "linear_end": 1e-2,
        },
        "model": {
            "pretrained_folder_url": "https://example.invalid",
            "pretrained_file": "pretrained_gen.pth",
            "image_size": 16,
            "channels": 3,
            "unet": {
                "in_channel": 6,
                "out_channel": 3,
                "inner_channel": 8,
                "norm_groups": 4,
                "channel_multiplier": [1, 2],
                "attn_res": [8],
                "res_blocks": 1,
                "dropout": 0.0,
            },
        },
        "train": {
            "seed": 42,
            "batch_size": 2,
            "grad_accum_steps": 1,
            "learning_rate": 1e-4,
            "epochs": 2,
            "checkpoint_epochs": [1, 2],
            "max_iterations": 0,
            "log_every": 1,
        },
        "sampling": {
            "seed": 7,
            "batch_size": 2,
            "snapshot_samples": 1,
            "validation_images": 2,
            "trajectory_samples": 1,
            "trajectory_timesteps": [8, 4, 0],
        },
        "evaluation": {"metrics": ["psnr", "ssim"], "test_limit": 0},
    }


def make_fake_afhq(raw_dir: Path, train_per_class: int = 4, val_per_class: int = 2) -> Path:
    """Write ``raw_dir/afhq/{train,val}/{cat,dog,wild}/*.jpg`` and the extraction marker."""
    rng = np.random.default_rng(0)
    root = raw_dir / "afhq"
    for split, count in (("train", train_per_class), ("val", val_per_class)):
        for animal in AFHQ_CLASSES:
            folder = root / split / animal
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                pixels = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{animal}_{split}_{index:03d}.jpg")
    (raw_dir / EXTRACTED_MARKER).write_text("test", encoding="utf-8")
    return root
