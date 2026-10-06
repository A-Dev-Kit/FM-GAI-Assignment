# ADDED: new file, not part of the upstream SR3 codebase.
"""Record the software and hardware environment of a run (for the README and report)."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from typing import Any

import torch

from sr3_ablation.paths import PROJECT_ROOT

TRACKED_PACKAGES = ("torch", "torchvision", "numpy", "pillow", "matplotlib", "gdown", "lpips")


def collect_environment(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: _version(name) for name in TRACKED_PACKAGES},
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "git_commit": _git_commit(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info["gpu"] = {
            "name": properties.name,
            "total_memory_mib": round(properties.total_memory / 2**20),
            "capability": f"{properties.major}.{properties.minor}",
        }
    return info


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None
